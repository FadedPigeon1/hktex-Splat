"""Triangle-supported, tile-based Heat-Kernel Splatting (HKS) and bounded benchmark correctness oracles.

Rendering/training uses triangle support, corner coefficients and a kernel/tile queue.
Main API: DirectTileRenderer, TileSurface, triangle_tile_work, _HybridSplat."""
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
import copy

import numpy as np
import torch
from hktex_splatting.rendering.splat_autograd import _Splat


def scope(enabled, name):
    prefix = 'spectral::' if name in ('source_state', 'spectral_shading', 'triangle_coefficients') else 'support::'
    return torch.profiler.record_function(prefix+'direct_tiles_'+name) if enabled else nullcontext()


class TileSurface:
    """Fixed mesh visibility snapshot; includes every foreground pixel once.

    Camera rasterization happens outside this object. Rebuild after changing
    camera, mesh, spectral embedding or visibility. No geometry gradients.
    """
    def __init__(self, faces, embedding, face_ids, bary, tile_size=8, *, positions=None, depth=None, camera=None):
        ids, b = np.asarray(face_ids), np.asarray(bary)
        if (ids.ndim != 2 or ids.dtype.kind not in 'iu' or
                b.shape != (*ids.shape, 3) or tile_size not in (1, 2, 4, 8, 16) or
                np.any(ids < -1) or np.any(ids >= len(faces))):
            raise ValueError('Invalid visibility snapshot')
        foreground = ids >= 0
        if (not np.isfinite(b[foreground]).all() or
                not np.allclose(b[foreground].sum(-1), 1, atol=2e-6) or
                np.any(b[foreground] < -2e-6)):
            raise ValueError('Invalid visible barycentrics')
        if faces.device != embedding.device or embedding.requires_grad or faces.requires_grad:
            raise ValueError('Expected fixed mesh and embedding on one device')
        self.shape, self.tile_size = ids.shape, tile_size
        if camera is not None and (camera.get('height') != ids.shape[0] or camera.get('width') != ids.shape[1]):
            raise ValueError('Camera dimensions differ from the visibility snapshot')
        self._ids_cpu, self._bary_cpu = ids.copy(), b.copy()
        self._ids_cpu.flags.writeable = self._bary_cpu.flags.writeable = False
        self.camera = copy.deepcopy(camera)
        self.faces, self.embedding = faces, embedding
        self._versions = (faces._version, embedding._version)
        self.pixels = torch.as_tensor(np.flatnonzero(foreground), device=embedding.device)
        self.active_ordinals = torch.arange(len(self.pixels), device=embedding.device, dtype=torch.int32)
        fi = torch.as_tensor(ids[foreground].astype(np.int64), device=embedding.device)
        self.vertices = faces[fi].long()
        self.bary = torch.tensor(b[foreground], device=embedding.device, dtype=embedding.dtype)
        self.queries = (embedding[self.vertices]*self.bary[:, :, None]).sum(1).detach()
        h, w = ids.shape
        self.tiles_x = (w+tile_size-1)//tile_size
        self.tile_count = self.tiles_x*((h+tile_size-1)//tile_size)
        self.tiles = (self.pixels//w//tile_size)*self.tiles_x+self.pixels%w//tile_size
        # Kernel/tile work reads a fixed, padded list of visible pixel ordinals.
        # Construct it once from visibility, with no source/triangle support.
        flat_pixels = np.flatnonzero(foreground)
        tile_ids = (flat_pixels//w//tile_size)*self.tiles_x+flat_pixels%w//tile_size
        tile_pixels = np.full((self.tile_count, tile_size**2), -1, np.int32)
        lengths = np.zeros(self.tile_count, np.int32)
        for ordinal, tile in enumerate(tile_ids):
            tile_pixels[tile, lengths[tile]] = ordinal
            lengths[tile] += 1
        self.tile_pixels = torch.as_tensor(tile_pixels, device=embedding.device)
        self.tile_lengths = torch.as_tensor(lengths, device=embedding.device)
        self.active_tile_count = int((lengths > 0).sum())
        self.face_ids = fi
        self.positions = (None if positions is None else
                          (positions[self.vertices]*self.bary[:, :, None]).sum(1).detach())
        self.depth = (None if depth is None else
                      torch.as_tensor(np.asarray(depth)[foreground].copy(), device=embedding.device))
        self.active = torch.bincount(self.tiles, minlength=self.tile_count) > 0
        self._buffer_names = ('vertices', 'bary', 'queries', 'pixels', 'tiles', 'active',
                              'tile_pixels', 'tile_lengths', 'face_ids', 'active_ordinals')
        self._buffers = tuple(getattr(self, name) for name in self._buffer_names)
        self._buffer_versions = tuple(t._version for t in self._buffers)

    def valid(self):
        return (self._versions == (self.faces._version, self.embedding._version) and
                all(getattr(self, name) is t for name,t in zip(self._buffer_names, self._buffers)) and
                self._buffer_versions == tuple(t._version for t in self._buffers))


@dataclass(frozen=True)
class KernelTileWork:
    membership: torch.Tensor   # [source,tile]; disconnected screen-space regions
    queue: torch.Tensor        # compact live prefix, padded to source*tile capacity
    count: torch.Tensor        # device scalar; no download for dispatch

    @property
    def counts(self):
        return self.membership.sum(0)

    @property
    def device_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.membership, self.queue, self.count))


@torch.no_grad()
def full_spectrum_work(surface, source_count):
    """Enumerate all active tiles for the benchmark's bounded spectral oracle."""
    membership = surface.active[None, :].expand(source_count, -1).clone()
    flat = membership.flatten()
    queue = torch.full((flat.numel(),), -1, device=flat.device, dtype=torch.int64)
    live = torch.arange(flat.numel(), device=flat.device)[flat]
    queue[:len(live)] = live
    count = flat.sum()
    return KernelTileWork(membership, queue, count)


def _pixel_coefficients(field, state, surface, pixels, kernels, fused):
    """Reuse shared spectral coefficient construction, with pixel work inputs.

    These are selected kernel/pixel records, not a face support table. No
    unique-face enumeration, face CSR, support selector or triangle dispatch.
    """
    vertices = surface.vertices[pixels]
    if fused:
        from hktex_splatting.modules.spectral_heat_coefficients import fused_coefficients
        return fused_coefficients(vertices, kernels, state, field.eigen)
    return field._chunk(vertices, kernels, state.grid_ids, state.grid_weights, state.decay,
                        state.phi, state.normalizer, state.embedding)


def _responses_from_coefficients(heat, metric, bary, eigen):
    h = (heat*bary).sum(-1)
    if not eigen.use_weighting:
        return h
    b0, b1, b2 = bary.unbind(1)
    terms = torch.stack((b0*b0, b1*b1, b2*b2, 2*b0*b1, 2*b0*b2, 2*b1*b2), 1)
    distance = (metric*terms).sum(1).clamp_min(0)
    return h*eigen.heat_weighting.compute(distance[:, None]).flatten()


@torch.no_grad()
def _torch_splat_responses(field, state, surface, work, batch):
    """Bounded eager kernel-major oracle; never the default CUDA execution path."""
    n, t = work.membership.shape
    response = state.phi.new_zeros((len(surface.pixels), n))
    # CPU/eager diagnostics may compact on the host; device primary never does.
    tasks = work.queue[work.queue >= 0]
    for start in range(0, len(tasks), max(1, batch//surface.tile_size**2)):
        task = tasks[start:start+max(1, batch//surface.tile_size**2)]
        pixels = surface.tile_pixels[task % t].long()
        kernels = (task//t)[:, None].expand_as(pixels)
        valid = pixels >= 0
        p, k = pixels[valid], kernels[valid]
        h, metric = _pixel_coefficients(field, state, surface, p, k, False)
        response[p, k] = _responses_from_coefficients(h, metric, surface.bary[p], field.eigen)
    return response


@dataclass
class TriangleTileWork:
    """One current-forward queue; each live key identifies a kernel and tile.

    Each key's sorted entry range holds its supported visible triangles. Pixel
    record offsets use face support only for addressing, never for scheduling.
    """
    support: object
    surface: object
    keys: torch.Tensor
    ranges: torch.Tensor
    count: torch.Tensor
    entries: torch.Tensor
    pixel_offsets: torch.Tensor

    @property
    def kernels(self): return self.support.kernels
    @property
    def bary(self): return self.surface.bary
    @property
    def face_ids(self): return self.surface.face_ids
    @property
    def counts(self):
        result = torch.zeros(self.surface.tile_count, device=self.keys.device, dtype=torch.int64)
        result.scatter_add_(0, self.keys.clamp_min(0) % self.surface.tile_count, (self.keys >= 0).long())
        return result
    @property
    def device_bytes(self):
        return sum(t.numel()*t.element_size() for t in
                   (self.keys, self.ranges, self.count, self.entries, self.pixel_offsets))


@torch.no_grad()
def triangle_tile_work(support, geometry, surface, backend='torch', patch_tiles=None):
    """Group kernel-triangle support incidences by (kernel,tile), not face patches.

    Only the fixed geometry layout is reused. No benchmark reference renderer
    is invoked. Sort stability leaves each
    group's coefficient entries in face order for the emitter's lookup.
    """
    if not support.valid() or not geometry.valid():
        raise ValueError('Stale hybrid support/visibility')
    k, f = support.kernels, support.entry_faces
    torch._assert_async(((k >= 0) & (k < support.source_count)).all(), 'Invalid support kernel ID')
    torch._assert_async(((f[1:] > f[:-1]) | ((f[1:] == f[:-1]) & (k[1:] > k[:-1]))).all(),
                        'Duplicate or unordered kernel/triangle support')
    # CSR already has ascending faces. One stable kernel/tile key sort is
    # sufficient; sorting kernels first would add redundant per-step work.
    order = torch.arange(len(k), device=k.device)
    faces = f.long()
    repetitions = geometry.face_counts[faces]
    prefix = repetitions.cumsum(0)-repetitions
    entries = torch.repeat_interleave(order, repetitions, output_size=support.tasks)
    patch = (torch.repeat_interleave(geometry.face_starts[faces]-prefix, repetitions,
                                    output_size=support.tasks)+torch.arange(support.tasks, device=k.device))
    if patch_tiles is None:
        patch_tiles = torch.as_tensor(np.array(geometry.geometry._layout.patch_tiles, copy=True), device=k.device)
    tiles = patch_tiles[patch]
    keys = k[entries].long()*surface.tile_count+tiles
    keys, permutation = keys.sort(stable=True)
    entries = entries[permutation].int()
    valid = keys != torch.iinfo(torch.int64).max
    first = valid & torch.cat((valid.new_ones(min(1, len(keys))), keys[1:] != keys[:-1]))
    last = valid & torch.cat((keys[1:] != keys[:-1], valid.new_ones(min(1, len(keys)))))
    groups = first.long().cumsum(0)-1
    capacity = min(len(keys), support.source_count*surface.active_tile_count)
    queue = torch.full((capacity,), -1, device=k.device, dtype=torch.int64)
    ranges = torch.zeros((capacity, 2), device=k.device, dtype=torch.int32)
    if backend == 'triton':
        from hktex_splatting.triton.heat_kernel_splatting import group_triangle_work
        group_triangle_work(keys, first, last, groups, queue, ranges)
    else:
        index = torch.arange(len(keys), device=k.device)
        queue[groups[first]] = keys[first]
        ranges[groups[first], 0] = index[first].int()
        ranges[groups[last], 1] = index[last].int()+1
    counts = support.counts[surface.face_ids]
    pixel_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return TriangleTileWork(support, surface, queue, ranges, first.sum(), entries,
                            pixel_offsets)


def _hybrid_eager(field, heat, metric, work, inner):
    """Differentiable compact-record oracle with the triangle-reference entry-order ties."""
    g, support, model = work.surface, work.support, field.model
    p, fragments = len(g.pixels), support.fragments
    counts = support.counts[g.face_ids]
    pixels = torch.repeat_interleave(torch.arange(p, device=heat.device), counts, output_size=fragments)
    local = torch.arange(fragments, device=heat.device)-work.pixel_offsets[pixels]
    entries = support.offsets[g.face_ids[pixels]]+local
    response = _responses_from_coefficients(heat[entries], metric[entries], g.bary[pixels], field.eigen)
    keep = torch.ones_like(entries, dtype=torch.bool)
    kernels = support.kernels[entries].long()
    weight = model.kernel_filter_func(response, epsilon=model.thresholds[kernels], sharpness=model.sharpnesses[kernels])
    weight = torch.where(keep, weight, 0.)
    width = max(1, support.maximum)
    lane = torch.arange(width, device=heat.device)[None, :]
    valid = lane < counts[:, None]
    record = work.pixel_offsets[:-1, None]+lane
    re = torch.full((p, width), -1, device=heat.device, dtype=torch.long)
    rw = heat.new_zeros((p, width))
    if fragments:
        rw[valid] = weight[record[valid]]
        re[valid] = torch.where(keep[record[valid]], entries[record[valid]], -1)
    picked = rw.detach().argsort(dim=1, descending=True, stable=True)[:, :min(inner, width)]
    se, sw = re.gather(1, picked), rw.gather(1, picked)
    colors = heat.new_zeros((*se.shape, 3))
    if len(support.kernels):
        source = support.kernels[se.clamp_min(0)].long()
        colors = model.kernel_colours[source]*((se >= 0) & (sw > 0))[:, :, None]
    rgb = (model._mean_colour+(sw[:, :, None]*colors).sum(1)/sw.sum(1, keepdim=True).clamp_min(1)).clamp(0, 1)
    work.selected_entries, work.selected_weights = se, sw
    work.record_entries = torch.where(keep, entries, -1)
    work.responses = torch.where(keep, response, 0.)
    return rgb


class _HybridSplat(_Splat):
    """Own kernel-driven forward, the unchanged shared spectral backward operator."""
    @staticmethod
    def forward(ctx, h, gram, c, b, ep, sh, scene, inner_k, sig, eps, power, runtime):
        from hktex_splatting.triton.heat_kernel_splatting import hybrid_forward
        rgb, se, sw = hybrid_forward(h, gram, c, b, ep, sh, scene, inner_k, sig, eps)
        ctx.save_for_backward(h, gram, c, b, ep, sh, se, sw, scene.kernels, scene.bary, scene.face_ids)
        ctx.recipe, ctx.runtime_counts = (sig, eps, power), True
        return rgb


def _hybrid_device(heat, metric, colors, base, epsilon, sharpness, work, inner, sigma, distance_eps):
    tensors = (heat, metric, colors, base, epsilon, sharpness)
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError('Hybrid Triton forward/backward requires float32')
    return _HybridSplat.apply(*(t.contiguous() for t in tensors), work, inner, sigma, distance_eps, 1, True)


class DirectTileRenderer:
    """Main triangle-supported, tile-based Heat-Kernel Splatting API.

    Triangle support and coefficients are refreshed every call, then grouped
    by kernel/tile for emission. The shared backward is inherited unchanged.
    oracle/full_reference are bounded correctness diagnostics, not backends.
    The compatibility value selection='hybrid' selects this HKS implementation.
    Gaussian attenuation here weights intrinsic distances; the primitives are
    spectral heat kernels.
    """
    def __init__(self, field, surface, *, pixel_batch=256,
                 max_bin_bytes=512*1024**2, profile=False, selection='hybrid',
                 splat_backend='auto', max_contributions=16777216,
                 support_cap=128, support_debug_dir=None):
        if field.mesh.faces is not surface.faces or field.eigen._iso_embeddings is not surface.embedding:
            raise ValueError('Surface belongs to another field')
        cfg = field.eigen.faiss_index.config
        if cfg.metric != 'l2' or cfg.index_type != 'flat' or cfg.use_float16:
            raise ValueError('Only native float32/64 flat L2 KNN is audited')
        if pixel_batch < 1:
            raise ValueError('pixel_batch must be positive')
        if (selection != 'hybrid' or splat_backend not in ('auto', 'triton', 'torch') or
                max_contributions < 1 or not 0 <= support_cap <= 512):
            raise ValueError('Only hybrid rendering is supported; use the native backend for original HKTex')
        cfg = field.eigen.heat_weighting.config
        if (field.model.cfg.power_diffused_diracs != 1 or
                (field.eigen.use_weighting and (cfg.kernel != 'gaussian' or cfg.normalize))):
            raise ValueError('Kernel splats require audited power=1, unnormalized Gaussian attenuation')
        self.field, self.surface = field, surface
        self._mesh_vertices, self._mesh_version = field.mesh.verts, field.mesh.verts._version
        self.pixel_batch, self.max_bin_bytes, self.profile = pixel_batch, max_bin_bytes, profile
        self.selection = selection
        self.max_contributions, self.support_cap = max_contributions, support_cap
        self.support_debug_dir = support_debug_dir
        self.splat_backend = ('triton' if surface.embedding.is_cuda else 'torch') if splat_backend == 'auto' else splat_backend
        self._archive = field.eigen._eigen_vec_knn
        self._archive_version = self._archive._version
        from hktex_splatting.rendering.prepared_geometry import PreparedGeometry
        from hktex_splatting.rendering.resident import ResidentDispatchGeometry
        from hktex_splatting.knn_heat.triangle_kernel_support import ResidentSupportGeometry
        if getattr(field, 'support_selector', 'legacy') == 'fused':
            from hktex_splatting.knn_heat.triangle_kernel_support import FusedResidentSupportGeometry
            ResidentSupportGeometry = FusedResidentSupportGeometry
        fixed = PreparedGeometry(len(field.mesh.faces), surface._ids_cpu, surface._bary_cpu,
                                 surface.embedding.device, surface.tile_size, 'stable')
        self.hybrid_geometry = ResidentDispatchGeometry(fixed)
        self.hybrid_patch_tiles = torch.tensor(fixed._layout.patch_tiles.copy(), device=surface.embedding.device)
        self._hybrid_patch_version = self.hybrid_patch_tiles._version
        self.hybrid_support = ResidentSupportGeometry(field.mesh.faces, surface.embedding,
            surface._ids_cpu, field.model.cfg.knn_outer_k, profile=profile)

    def __call__(self, *, diffusion_time=None, trace=False, oracle=False, full_reference=False):
        f, g = self.field, self.surface
        m, e = f.model, f.eigen
        if (not g.valid() or f.mesh.faces is not g.faces or e._iso_embeddings is not g.embedding or
                f.mesh.verts is not self._mesh_vertices or f.mesh.verts._version != self._mesh_version or
                e._eigen_vec_knn is not self._archive or self._archive._version != self._archive_version):
            raise ValueError('Mesh/embedding changed; rebuild TileSurface')
        if full_reference:
            return self._splat(diffusion_time, trace, True)
        if m.cfg.knn_outer_k != self.hybrid_support.neighbors:
            raise ValueError('Triangle support policy changed; rebuild the renderer')
        if self.hybrid_patch_tiles._version != self._hybrid_patch_version:
            raise ValueError('Hybrid tile geometry changed; rebuild the snapshot')
        return self._hybrid(diffusion_time, trace, oracle)


    def _splat(self, diffusion_time, trace, oracle):
        f, g = self.field, self.surface
        m, e = f.model, f.eigen
        n, p = m.N_sources, len(g.pixels)
        # Capacity is known from static dimensions before any record allocation.
        # Reject, never truncate. Includes dense response/filter scratch and queue.
        inner = m.cfg.knn_inner_k or n
        estimate = n*p*24+n*g.tile_count*32+p*min(inner, n)*64
        if n*p > self.max_contributions or estimate > self.max_bin_bytes:
            raise MemoryError('Kernel/pixel contributions exceed capacity; reduce workload or explicitly increase budgets')
        backend = 'torch'
        with scope(self.profile, 'source_state'):
            state = f.sources(diffusion_time)
        torch._assert_async(torch.isfinite(state.embedding).all() & torch.isfinite(state.normalizer).all() &
                            (state.normalizer > 0).all(), 'Invalid live kernel state')
        work = full_spectrum_work(g, len(state.embedding))
        with scope(self.profile, 'kernel_tile_emission'):
            response = _torch_splat_responses(f, state, g, work, max(self.pixel_batch, 4096))
        # Candidate selection is discrete, like native KNN/topk. Recompute only
        # selected records through validated shared splat autograd to bound the graph.
        with torch.no_grad():
            filtered = m.kernel_filter_func(response, epsilon=m.thresholds[None, :], sharpness=m.sharpnesses[None, :])
            picked = filtered.topk(min(inner, n), dim=1).indices
        k = picked.shape[1]
        pixels = torch.arange(p, device=g.queries.device)[:, None].expand(-1, k).reshape(-1)
        with scope(self.profile, 'spectral_shading'):
            fused = backend == 'triton' and state.phi.dtype == torch.float32
            h, metric = _pixel_coefficients(f, state, g, pixels, picked.reshape(-1), fused)
            selected_response = _responses_from_coefficients(h, metric, g.bary[pixels], e).reshape(p, k)
            # A rejected tile is a discrete selection, including zero-weight
            # filler slots: no output or gradient may leak back through it.
            admitted = work.membership[picked, g.tiles[:, None]]
            selected_response = torch.where(admitted, selected_response, 0.)
            proxy = SimpleNamespace(diffuse_heat=lambda *args, **kwargs: (selected_response, None))
            stub = selected_response[:, :, None]
            info = dict(albo_evals=stub, albo_evecs=stub, indices=picked,
                        distances=None, weights=None)
            value, _, top, weights = m.diffuse_heat_kernels(proxy, info)
            colors = m(value)
        rgb = m._mean_colour.sum()*0+g.queries.new_zeros((g.shape[0]*g.shape[1], 3))
        rgb = rgb.index_copy(0, g.pixels, colors).reshape(*g.shape, 3)
        if not trace:
            return rgb
        return rgb, dict(bins=work, outer=torch.arange(n, device=g.queries.device).expand(p, -1),
            response=response, top=top[:, :, 0].T, weights=weights[:, :, 0].T,
            selected_response=selected_response, filtered_response=filtered, query_embedding=g.queries,
            source_embedding=state.embedding.detach(),
            spectral_pairs=(work.membership*g.tile_lengths[None, :]).sum(),
            selection_backend='kernel_splat', tile_candidates_used_for_selection=True,
            faiss_index_rebuilt=False, kernel_triangle_support_constructed=False,
            dispatch='kernel_tile_queue', footprint_cutoff=0.,
            queue_count=work.count, queue_capacity=work.queue.numel(),
            contribution_capacity=n*p, backward_records=p*k,
            semantics='all-positive spectral heat then top-k; no native outer-50 preselection')

    def _hybrid(self, diffusion_time, trace, oracle):
        from hktex_splatting.rendering.resident import ResidentTopology
        f, g = self.field, self.surface
        m, e = f.model, f.eigen
        if self.max_bin_bytes <= 0:
            raise MemoryError('Hybrid buffer budget must be positive')
        inner = m.cfg.knn_inner_k or 30
        if not 1 <= inner <= 32:
            raise ValueError('Hybrid reconstruction uses the resident 1..32 reduction contract')
        backend = 'torch' if oracle else self.splat_backend
        with scope(self.profile, 'source_state'):
            state = f.sources(diffusion_time)
        torch._assert_async(torch.isfinite(state.embedding).all() & torch.isfinite(state.normalizer).all() &
                            (state.normalizer > 0).all(), 'Invalid live hybrid kernel state')
        debugger = None
        if self.support_debug_dir is not None:
            from hktex_splatting.utils.support_debug import SupportDebugger
            debugger = SupportDebugger(self.support_debug_dir, f, state, g.camera)
        with scope(self.profile, 'triangle_support'):
            support, stats = self.hybrid_support.select(state.embedding, self.support_cap,
                dispatch_geometry=self.hybrid_geometry, max_records=self.max_contributions,
                max_dispatch_bytes=self.max_bin_bytes,
                **(dict(debug_callback=debugger.check) if debugger is not None else {}))
        if (support.fragments*24+support.tasks*112+len(support.kernels)*128+
                len(g.pixels)*32*16 > self.max_bin_bytes):
            raise MemoryError('Hybrid coefficients/queue/pixel scratch exceed configured budget')
        with scope(self.profile, 'triangle_coefficients'):
            topology = ResidentTopology(support, f.mesh, e._iso_embeddings)
            heat, metric = f.coefficients(support, state, topology=topology)
        if debugger is not None:
            debugger.check(None, support, stage='after_coefficients')
        projection_stats = dict(cutoff=0., extra_dynamic_d2h_bytes=0)
        with scope(self.profile, 'kernel_triangle_tile_queue'):
            work = triangle_tile_work(support, self.hybrid_geometry, g, backend, self.hybrid_patch_tiles)
            work.profile = self.profile
            work.capture_response = trace
        if not len(g.pixels):
            # Match the shared backward operator's zero adjoints for appearance/filter inputs even
            # on an empty view, so Adam still decays their existing moments.
            zero = (m._mean_colour.sum()+m.kernel_colours.sum()+m.thresholds.sum()+m.sharpnesses.sum())*0
            rgb = zero+heat.new_zeros((0, 3))
            work.selected_entries = torch.empty((0, 32), device=heat.device, dtype=torch.long)
            work.selected_weights = heat.new_zeros((0, 32))
            work.record_entries = torch.empty(0, device=heat.device, dtype=torch.long)
            work.responses = heat.new_empty(0)
        elif backend == 'triton':
            rgb = _hybrid_device(heat, metric, m.kernel_colours, m._mean_colour, m.thresholds, m.sharpnesses,
                                  work, inner, e.heat_weighting.config.gaussian_std, e.heat_weighting.config.eps)
        else:
            with scope(self.profile, 'hybrid_eager_reduction'):
                rgb = _hybrid_eager(f, heat, metric, work, inner)
        colors = m(rgb)
        image = m._mean_colour.sum()*0+g.queries.new_zeros((g.shape[0]*g.shape[1], 3))
        image = image.index_copy(0, g.pixels, colors).reshape(*g.shape, 3)
        if not trace:
            return image
        width = max(1, support.maximum)
        lane = torch.arange(width, device=heat.device)[None, :]
        valid = lane < support.counts[g.face_ids, None]
        records = work.pixel_offsets[:-1, None]+lane
        entry = torch.full_like(records, -1)
        response = heat.new_zeros(records.shape)
        if support.fragments:
            entry[valid] = work.record_entries[records[valid]].long()
            response[valid] = work.responses[records[valid]]
        outer = torch.full_like(entry, -1)
        top = torch.full_like(work.selected_entries, -1)
        if len(support.kernels):
            outer = torch.where(entry >= 0, support.kernels[entry.clamp_min(0)].long(), -1)
            top = torch.where((work.selected_entries >= 0) & (work.selected_entries < len(support.kernels)),
                support.kernels[work.selected_entries.clamp(0, len(support.kernels)-1).long()].long(), -1)
        return image, dict(bins=work, outer=outer, response=response, top=top, weights=work.selected_weights,
            corner_heat=heat, gram=metric, mapping_kernels=support.kernels, entry_faces=support.entry_faces,
            selected_entries=work.selected_entries, support=stats, projection=projection_stats,
            query_embedding=g.queries, source_embedding=state.embedding.detach(),
            spectral_pairs=len(support.kernels)*3, coefficient_pairs=len(support.kernels),
            spectral_corner_evaluations=len(support.kernels)*3, spectral_modes=state.phi.shape[1],
            pixel_contributions=support.fragments, emitted_pixel_contributions=int((work.record_entries >= 0).sum()),
            selection_backend='resident_triangle_support',
            tile_candidates_used_for_selection=True, faiss_index_rebuilt=False,
            kernel_triangle_support_constructed=True, dispatch='kernel_triangle_tile_queue',
            footprint_cutoff=0., queue_count=work.count,
            queue_capacity=len(work.keys), contribution_capacity=support.fragments,
            backward_records=len(g.pixels)*32,
            semantics='Triangle-union candidate pool/cap and stable entry-order ties; distinct from pointwise outer-50 and full spectral coverage')
