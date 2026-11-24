"""Learned source state and differentiable three-corner spectral coefficients.

Rendering/training source preparation; the mesh and eigensystem stay fixed.
Main API: LiveHeatField, SourceState."""
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class SourceState:
    """Differentiable source grid, decay, normalization and embedding values for one forward."""
    grid_ids: torch.Tensor
    grid_weights: torch.Tensor
    decay: torch.Tensor
    phi: torch.Tensor
    normalizer: torch.Tensor
    embedding: torch.Tensor


class LiveHeatField:
    """Refresh source state and triangle-corner coefficients from the original HKTex model."""
    def __init__(self, trainer, pair_chunk=1024, checkpoint_chunks=True,
                 coefficient_backend='eager', profile=False, support_selector='legacy'):
        self.model, self.mesh, self.eigen = trainer.model, trainer.mesh, trainer.eigalbo_interp
        m, e = self.model, self.eigen
        if (m.cfg.out_net or m.cfg.normalize_colours or m.cfg.power_diffused_diracs != 1 or not e.use_weighting or
            e.heat_weighting.config.kernel != 'gaussian' or e.heat_weighting.config.normalize):
            raise ValueError('Requires the audited RGB, unnormalized spectral-distance weighting recipe')
        if pair_chunk < 1:
            raise ValueError('pair_chunk must be positive')
        self.pair_chunk, self.checkpoint_chunks = pair_chunk, checkpoint_chunks
        if coefficient_backend not in ('eager', 'triton'):
            raise ValueError('Unknown coefficient backend')
        self.coefficient_backend, self.profile = coefficient_backend, profile
        if support_selector not in ('legacy', 'fused'):
            raise ValueError('Unknown support selector')
        self.support_selector = support_selector
        self.last_support_stats={}
        from hktex_splatting.knn_heat.triangle_kernel_support import KernelSurfaceSupport
        options=getattr(trainer,'surface_options',{})
        hops=options.get('surface_support_hops',8)
        domain=options.get('support_domain','graph')
        support=getattr(trainer,'_kernel_surface_support',None)
        if (support is None or support.domain!=domain or support.hops!=hops or support.faces is not self.mesh.faces or
                support.embedding is not e._iso_embeddings or not support.valid()):
            support=KernelSurfaceSupport(self.mesh.faces,e._iso_embeddings,hops,
                options.get('max_buffer_mib',512)*1024**2,domain=domain)
            trainer._kernel_surface_support=support
        self.kernel_surface_support=support


    def prepare_support(self, face_ids, face_chunk=128):
        from hktex_splatting.knn_heat.triangle_kernel_support import (
            ResidentSupportGeometry, FusedResidentSupportGeometry,
        )
        selector = (FusedResidentSupportGeometry if self.support_selector == 'fused'
                    else ResidentSupportGeometry)
        with torch.profiler.record_function('support::geometry_preparation') if self.profile else nullcontext():
            return selector(self.mesh.faces, self.eigen._iso_embeddings, face_ids,
                            self.model.cfg.knn_outer_k, face_chunk, self.profile)

    def sources(self, diffusion_time=None, *, backend=None):
        """Refresh differentiable source state; eager remains the independent oracle."""
        m, e, mesh = self.model, self.eigen, self.mesh
        prepared = getattr(self, 'prepared_source_geometry', None)
        self.prepared_source_geometry = None
        if (prepared is not None and prepared[2] is m.kernel_locations and
                prepared[3] == m.kernel_locations._version and prepared[4] is m.kernel_face_ids and
                prepared[5] == m.kernel_face_ids._version and prepared[6] == torch.is_grad_enabled()):
            vertices, bary = prepared[:2]
            vertices = vertices.long()
        else:
            vertices = mesh.get_face_vertices(m.kernel_face_ids).long()
            bary = mesh.cartesian_to_barycentric(m.kernel_locations, vertices)
        # Native GeodesicOpt optionally reads detached barycentric metadata
        # from the location Parameter. Refresh it just as prepare_kernels does
        # the live bary tensor below remains connected to the learned location.
        if hasattr(m, 'save_barycentric_locations'):
            m.save_barycentric_locations(bary)
        if self.kernel_surface_support.domain == 'spectral':
            # Canonical native interpolation for discrete selection only.
            # Coefficient/source adjoints keep the existing fused/eager graph.
            with torch.no_grad():
                self.selection_embedding = (e._iso_embeddings[vertices]*bary[:, :, None]).sum(1)
        cfg = e.cfg
        backend = self.coefficient_backend if backend is None else backend
        # Preserve extended time overrides accepted by the original formula.
        # The fused source operator handles source-wise CUDA float32 times.
        if (backend == 'triton' and diffusion_time is not None and
                (not isinstance(diffusion_time, torch.Tensor) or
                 diffusion_time.dtype != m.kernel_locations.dtype or
                 diffusion_time.device != m.kernel_locations.device or
                 diffusion_time.numel() not in (1, len(vertices)) or
                 diffusion_time.ndim > 2 or
                 (diffusion_time.ndim == 2 and diffusion_time.shape[1] != 1))):
            backend = 'eager'
        if (backend == 'triton' and e.knn_gather.config.select_impl == 'bilinear4' and
                e.knn_gather.config.compile_select):
            from hktex_splatting.modules.spectral_heat_coefficients import fused_source_grid
            gi, gw = fused_source_grid(e._smp_coords_angle, e._smp_coords_scale,
                m.angles, m.anisotropies, scale_map=cfg.grid_scale_map,
                grid_layout=e.GRID_CONFIGURATION, eps=cfg.grid_eps, clamp_unit=cfg.grid_clamp_unit)
        else:
            gi, gw = e.knn_gather.select_grid(
                e._smp_coords_angle, e._smp_coords_scale, m.angles, m.anisotropies,
                abs_sin=cfg.grid_abs_sin, scale_map=cfg.grid_scale_map,
                grid_layout=e.GRID_CONFIGURATION, eps=cfg.grid_eps, clamp_unit=cfg.grid_clamp_unit)
        if backend == 'triton':
            from hktex_splatting.modules.spectral_heat_coefficients import fused_source_state
            t = m.diff_times if diffusion_time is None else diffusion_time
            t = t.reshape(-1).expand(len(vertices))
            decay, phi, normalizer, embedding = fused_source_state(vertices, bary, gi, gw, t, e)
            return SourceState(gi, gw, decay, phi, normalizer, embedding)
        if backend != 'eager':
            raise ValueError('Unknown source preparation backend')
        vals, phi = e.knn_gather.gather_src(gi, gw, vertices, bary,
                                            e._eigen_val_knn, e._eigen_vec_knn)
        t = m.diff_times if diffusion_time is None else diffusion_time
        if t.ndim == 1:
            t = t[:, None]
        decay = torch.exp(-vals*t)
        normalizer = (decay*phi*phi).sum(-1)+e.knn_heat.config.eps
        embedding = (e._iso_embeddings[vertices]*bary[:, :, None]).sum(1)
        return SourceState(gi, gw, decay, phi, normalizer, embedding)

    def _chunk(self, vertices, kernels, gi, gw, decay, phi, normalizer, source):
        # Contract the four-grid spectral mixture at each triangle corner.
        # Streaming the three corners bounds the large [pairs,4,modes] gather.
        e = self.eigen
        heat = []
        for i in range(3):
            raw = e._eigen_vec_knn[gi[kernels], vertices[:, i, None]]
            mixed = (raw*gw[kernels, :, None]).sum(1)
            heat.append((decay[kernels]*mixed*phi[kernels]).sum(-1)/normalizer[kernels])
        delta = e._iso_embeddings[vertices].double()-source[kernels, None, :].double()
        metric = torch.stack([(delta[:, i]*delta[:, j]).sum(-1)
            for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], 1)
        return torch.stack(heat, 1), metric.to(phi.dtype)


    def coefficients(self, mapping, state=None, *, topology=None):
        """Fresh tensors with gradients to location, angle, scale and time.

        Diffusion time is a configuration constant in original HKTex. Passing
        a live diffusion_time to sources() enables its mathematical derivative
        without silently adding a new learned parameter to the original model.
        topology optionally shares immutable scene indices; state and outputs
        remain fresh. A stale snapshot is rejected rather than silently reused.
        """
        state = self.sources() if state is None else state
        device = state.phi.device
        with torch.profiler.record_function('spectral::coefficient_topology_inputs') if self.profile else nullcontext():
            if topology is None:
                faces = np.repeat(np.arange(len(mapping.offsets)-1), np.diff(mapping.offsets))
                vertices = self.mesh.faces[torch.as_tensor(faces, device=device)].long()
                kernels = torch.as_tensor(mapping.kernels.astype(np.int64), device=device)
            else:
                if not topology.matches(mapping, self.mesh, self.eigen._iso_embeddings, device):
                    raise ValueError('Coefficient topology changed; rebuild the snapshot')
                vertices, kernels = topology.vertices, topology.coefficient_kernels
        if not len(kernels):
            return state.phi.new_empty((0, 3)), state.phi.new_empty((0, 6))
        if self.coefficient_backend == 'triton':
            from hktex_splatting.modules.spectral_heat_coefficients import fused_coefficients
            with torch.profiler.record_function('spectral::fused_coefficients'):
                return fused_coefficients(vertices, kernels, state, self.eigen,layout=getattr(self,'coefficient_layout','serial'))
        hs, ms = [], []
        for begin in range(0, len(kernels), self.pair_chunk):
            sl = slice(begin, begin+self.pair_chunk)
            args = (vertices[sl], kernels[sl], state.grid_ids, state.grid_weights,
                    state.decay, state.phi, state.normalizer, state.embedding)
            if self.checkpoint_chunks and torch.is_grad_enabled():
                kwargs = {}
                if self.profile:
                    kwargs['context_fn'] = lambda: (
                        torch.profiler.record_function('spectral::chunk_forward'),
                        torch.profiler.record_function('spectral::chunk_recompute'))
                h, g = checkpoint(self._chunk, *args, use_reentrant=False, **kwargs)
            else:
                h, g = self._chunk(*args)
            hs.append(h)
            ms.append(g)
        if not hs:
            return state.phi.new_empty((0, 3)), state.phi.new_empty((0, 6))
        return torch.cat(hs), torch.cat(ms)

    @torch.no_grad()
    def support(self, face_ids, state=None, pool_cap=128, face_chunk=128, plan=None,
                *, dispatch_geometry=None, max_records=1048576):
        """Refresh ordered device support for benchmark correctness controls."""
        if not 0 <= pool_cap <= 512 or face_chunk < 1:
            raise ValueError('Invalid support limits')
        state = self.sources() if state is None else state
        plan = self.prepare_support(face_ids, face_chunk) if plan is None else plan
        if not plan.matches(self.mesh.faces, self.eigen._iso_embeddings, face_ids,
                            self.model.cfg.knn_outer_k, face_chunk):
            raise ValueError('Prepared support geometry changed; rebuild the plan')
        if dispatch_geometry is None:
            raise ValueError('Resident support requires fixed device dispatch geometry')
        mapping, self.last_support_stats = plan.select(state.embedding, pool_cap,
            dispatch_geometry=dispatch_geometry, max_records=max_records)
        return mapping
