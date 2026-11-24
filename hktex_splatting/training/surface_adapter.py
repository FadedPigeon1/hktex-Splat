"""Pack original mesh samples for the existing HKS renderer and density callbacks.

These tiles are sample work groups, not a camera image or projected footprint.
Packing preserves sample order, face IDs and barycentrics. Density adapters omit
padding slots while calling the original controllers' accumulation methods.
"""
import numpy as np
from copy import copy
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace


def prepare_sample_grid(face_ids, barycentrics, *, grouped=False):
    """Optionally pack triangles together; inverse restores the original sample order."""
    ids, bary = np.asarray(face_ids), np.asarray(barycentrics)
    if not grouped: return sample_grid(ids, bary), None
    order = np.argsort(ids, kind='stable')
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return sample_grid(ids[order], bary[order]), inverse


def sample_grid(face_ids, barycentrics, tile_size=8):
    """Return a padded face/barycentric grid with each input sample exactly once."""
    ids, bary = np.asarray(face_ids), np.asarray(barycentrics)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or
            bary.shape != (len(ids), 3) or np.any(ids < 0) or
            not np.isfinite(bary).all() or np.any(bary < -2e-6) or
            not np.allclose(bary.sum(1), 1., atol=2e-6) or tile_size != 8):
        raise ValueError('Expected nonempty surface face IDs and valid barycentrics; tile size is 8')
    width = max(tile_size, int(np.ceil(np.sqrt(len(ids))/tile_size))*tile_size)
    height = (len(ids)+width-1)//width
    packed_ids = np.full((height, width), -1, dtype=np.int64)
    packed_bary = np.zeros((height, width, 3), dtype=bary.dtype)
    packed_ids.flat[:len(ids)] = ids
    packed_bary.reshape(-1, 3)[:len(ids)] = bary
    return packed_ids, packed_bary


def contribution_validity(indices, weights, source_count):
    """Keep shape and invalid-slot safety checks even for inactive controllers."""
    if (indices is None or weights is None or indices.ndim != 3 or
            indices.shape[-1] != 1 or weights.shape != indices.shape):
        raise ValueError('Expected equally shaped [k,P,1] contribution IDs and weights')
    valid = (indices[..., 0] >= 0) & (indices[..., 0] < source_count)
    invalid_weight = (~valid & (weights[..., 0] != 0)).any()
    if indices.is_cuda:
        import torch
        torch._assert_async(~invalid_weight, 'Invalid contribution IDs carry nonzero weights')
    elif bool(invalid_weight):
        raise ValueError('Invalid contribution IDs carry nonzero weights')
    return valid


def valid_contribution_groups(indices, weights, source_count, *, compact_widths=False):
    """Yield (point mask, compact [k,P,1] IDs/weights) for each valid top-k width.

    Original density callbacks require rectangular, nonnegative index arrays.
    Grouping by actual width avoids inventing hits for HKS's sentinel padding.
    Valid zero-weight contributions remain valid hits, as in original HKTex.
    """
    valid = contribution_validity(indices, weights, source_count)
    widths = valid.sum(0)
    present = (widths.bincount(minlength=indices.shape[0]+1).cpu().tolist()
               if compact_widths else None)
    for width in range(1, indices.shape[0]+1):
        if present is not None and not present[width]: continue
        points = widths == width
        if present is None and not bool(points.any()):
            continue
        # Boolean indexing is point-major, preserving per-point selected order.
        keep = valid[:, points].T
        ids = indices[:, points, 0].T[keep].reshape(-1, width).T[..., None]
        values = weights[:, points, 0].T[keep].reshape(-1, width).T[..., None]
        yield points, ids, values


class ValidDensityContributions:
    """Delegate density operations to an original controller, excluding HKS padding."""
    def __init__(self, controller, model):
        self.controller, self.model = controller, model
        self.group_cache = None
        self.skip_inactive, self.accumulation_option = False, None
        self.accumulator_names = ()

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def needs_contributions(self, step):
        """Mirror the original callback's start/stop predicate, without GPU reads."""
        if not self.skip_inactive or self.accumulation_option is None:
            return True
        cfg = self.controller.cfg
        return (step >= cfg.start_iter-getattr(cfg, self.accumulation_option) and
                (cfg.stop_iter is None or step < cfg.stop_iter))

    def _accumulate_padded(self, valid, indices, weights, **kwargs):
        """Call an audited native callback once; discard its padding scratch row.

        Original model buffers and controller identity never change. Only exact
        native controller types opt in, because their pre callbacks read these
        buffers and no other model attributes. Zero-weight valid hits still count.
        """
        import torch
        count = self.model.N_sources
        originals = {name:getattr(self.model, name) for name in self.accumulator_names}
        if any(value.ndim != 2 or value.shape != (count, 1) for value in originals.values()):
            raise ValueError('Density accumulator shape differs from current source count')
        with torch.no_grad():
            scratch = {name:torch.cat((value, value.new_zeros((1, 1))))
                       for name, value in originals.items()}
            shadow = copy(self.controller)
            shadow._model = SimpleNamespace(**scratch)
            routed = torch.where(valid[..., None], indices, count)
            try:
                shadow.pre_backward_step(topk_kernel_idxs=routed,
                    topk_kernel_contribs=weights, kernel_contributions=None, **kwargs)
            finally:
                for name, value in originals.items():
                    value.copy_(scratch[name][:-1])

    def pre_backward_step(self, *, rendered_colours, gt_colours, topk_kernel_idxs,
                          topk_kernel_contribs, step, **kwargs):
        kwargs.pop('kernel_contributions', None)
        if not self.needs_contributions(step):
            if topk_kernel_idxs is not None or topk_kernel_contribs is not None:
                contribution_validity(topk_kernel_idxs, topk_kernel_contribs, self.model.N_sources)
            return
        if (self.accumulator_names and self.skip_inactive and
                'pre_backward_step' not in vars(self.controller)):
            valid = contribution_validity(topk_kernel_idxs, topk_kernel_contribs, self.model.N_sources)
            return self._accumulate_padded(valid, topk_kernel_idxs, topk_kernel_contribs,
                step=step, rendered_colours=rendered_colours, gt_colours=gt_colours, **kwargs)
        groups = (self.group_cache.groups(topk_kernel_idxs, topk_kernel_contribs, self.model.N_sources)
                  if self.group_cache is not None else valid_contribution_groups(
                      topk_kernel_idxs, topk_kernel_contribs, self.model.N_sources))
        for mask, indices, weights in groups:
            self.controller.pre_backward_step(step=step,
                rendered_colours=rendered_colours[mask], gt_colours=gt_colours[mask],
                kernel_contributions=None, topk_kernel_idxs=indices,
                topk_kernel_contribs=weights, **kwargs)


class ContributionGroups:
    """Per-batch cache shared by controllers; cleared before the next forward.

    Store tensor identities/versions as well as source count. Never reuse groups
    after mutation, a new batch, or a density change.
    """
    def clear(self):
        self.key, self.value = None, None
        self.inputs = None

    def __init__(self): self.clear()

    def groups(self, indices, weights, source_count):
        key = (id(indices), indices._version, id(weights), weights._version, source_count)
        if key != self.key:
            self.value = list(valid_contribution_groups(indices, weights, source_count, compact_widths=True))
            self.key = key
            self.inputs = (indices, weights)
        return self.value


def surface_forward(trainer, data, *, trace=False, coefficient_backend='triton', splat_backend='triton', reference_adapter=False,
                    diagnostic_full_trace=False, validation_candidates=None):
    """Run live source/corner/tile evaluation on camera pixels or original surface samples."""
    import torch
    from hktex_splatting.rendering.reference_mesh import shade_batch
    from hktex_splatting.modules.spectral_heat_field import LiveHeatField
    from hktex_splatting.rendering.heat_kernel_splatting import DirectTileRenderer, TileSurface
    options = trainer.surface_options
    if validation_candidates is not None and getattr(trainer,'_initial_recorded',False):
        raise ValueError('Native candidate replay is restricted to initial validation')
    camera = '_camera_ids' in data and not reference_adapter
    direct = options.get('surface_adapter') == 'direct' and not reference_adapter and not camera
    optimized = options.get('surface_adapter') in ('optimized', 'aggressive', 'direct') and not reference_adapter
    profiler = getattr(trainer, 'surface_profiler', None)
    cached_renderer = None
    renderer_key = None
    def stage(name): return profiler.stage(name) if profiler is not None else nullcontext()
    with stage('sample_geometry'):
        grid = None if reference_adapter else data.get('_hks_grid')
        inverse = None if reference_adapter else data.get('_hks_inverse')
        if camera:
            from hktex_splatting.training.camera_supervision import cpu_array
            device_camera=data.get('_camera_device',False)
            # Saved verification/evaluation receipts intentionally live on CPU;
            # restore their tensors once. Live raster tensors already use this device.
            ids,bary=((data['_camera_ids'].to(trainer.eigalbo_interp._iso_embeddings.device),
                       data['_camera_bary'].to(trainer.eigalbo_interp._iso_embeddings.device)) if device_camera else
                      (cpu_array(data['_camera_ids']),cpu_array(data['_camera_bary'])))
            inverse = None
            if '_camera_key' in data and not diagnostic_full_trace and not device_camera:
                renderer_key = (data['_camera_key'],id(trainer.model),id(trainer.mesh),id(trainer.eigalbo_interp),
                    coefficient_backend,splat_backend,options.get('selector_merge','reference'),
                    options['support_cap'],options['max_contributions'],options['max_buffer_mib'],
                    options.get('density_policy','original'),options.get('corner_evaluation','serial'),options.get('association_cache','none'),options.get('support_domain','graph'),id(profiler))
                cache = getattr(trainer,'_camera_renderers',None)
                if cache is None: trainer._camera_renderers = cache = OrderedDict()
                cached_renderer = cache.get(renderer_key)
                if cached_renderer is not None:
                    prior = cached_renderer.surface
                    if not np.array_equal(prior._ids_cpu,ids) or not np.array_equal(prior._bary_cpu,bary,equal_nan=True):
                        raise ValueError('Camera key reused for different visibility')
                    cache.move_to_end(renderer_key)
                    trainer.camera_geometry_hits = getattr(trainer,'camera_geometry_hits',0)+1
        elif direct:
            from hktex_splatting.training.sample_tile_geometry import SampleTileSurface
            surface = SampleTileSurface(trainer.mesh.faces, trainer.eigalbo_interp._iso_embeddings,
                data['face_id'], data['bary'], host_ids=data.get('_hks_host_ids'),
                max_bytes=options['max_buffer_mib']*1024**2)
            inverse = surface.inverse
            ids = surface._ids_cpu
        elif grid is None:
            grid, inverse = prepare_sample_grid(data['face_id'].detach().cpu().numpy(),
                data['bary'].detach().cpu().numpy(), grouped=optimized)
        if not direct and not camera: ids, bary = grid
        field = LiveHeatField(trainer, coefficient_backend=coefficient_backend, support_selector='fused')
        if not reference_adapter and validation_candidates is None:
            field.prepared_source_geometry = trainer.__dict__.pop('_hks_prepared_source_geometry', None)
        field.validation_candidates=validation_candidates
        field.coefficient_layout=options.get('corner_evaluation','serial') if not reference_adapter else 'serial'
        trainer.surface_optimization=dict(association=('native_spectral_pixel_union' if options.get('support_domain','graph')=='spectral' else 'kernel_seeded_edge_graph'),
            support_hops=field.kernel_surface_support.hops,approximate=options.get('support_domain','graph')!='spectral',
            support_rebuild='every_forward',fixed_connectivity_reused=True)
        if cached_renderer is not None:
            surface = cached_renderer.surface
        elif not direct:
            if camera and device_camera:
                from hktex_splatting.rendering.device_camera import DeviceCameraSurface
                surface=DeviceCameraSurface(trainer.mesh.faces,trainer.eigalbo_interp._iso_embeddings,ids,bary,
                    max_bytes=options['max_buffer_mib']*1024**2,
                    pixels=None if '_camera_pixels' not in data else data['_camera_pixels'].to(ids.device))
            else:surface = TileSurface(trainer.mesh.faces, trainer.eigalbo_interp._iso_embeddings, ids, bary)
        if profiler is not None: profiler.samples(ids)
    with stage('renderer_preparation'):
        renderer = cached_renderer
        if renderer is None:
            renderer = DirectTileRenderer(field, surface, selection='hybrid', splat_backend=splat_backend,
                support_cap=options['support_cap'], max_contributions=options['max_contributions'],
                max_bin_bytes=options['max_buffer_mib']*1024**2, profile=profiler or False)
            if camera:
                visible=len(renderer.hybrid_support.visible)
                trainer.camera_workload=dict(visible_pixels=len(surface.pixels),visible_triangles=visible,
                    pixels_per_triangle=len(surface.pixels)/max(1,visible),screen_tiles=surface.tile_count,
                    scope='Most recent HKS camera frame; not a whole-run average')
            if renderer_key is not None:
                cache = trainer._camera_renderers
                while len(cache) >= 4: cache.popitem(last=False)
                cache[renderer_key] = renderer
                trainer.camera_geometry_builds = getattr(trainer,'camera_geometry_builds',0)+1
    result = renderer(trace=trace and (not optimized or diagnostic_full_trace),
                      contributions=trace and optimized and not diagnostic_full_trace)
    if camera:trainer.camera_selector_stats=renderer.last_support_stats
    inverse_device = None if inverse is None else torch.as_tensor(inverse, device=surface.embedding.device)
    def restore(value):
        return value if inverse_device is None else value[inverse_device]
    if not trace:
        flat = result.reshape(-1, 3)
        rgb = flat[surface.pixels] if camera else restore(flat[:len(data['face_id'])])
        with stage('material_lighting'):
            return shade_batch(rgb, data, options.get('lighting'),
                               backend='torch' if coefficient_backend == 'eager' else None)
    image, tape = result
    if inverse is not None:
        tape = dict(tape, top=restore(tape['top']), weights=restore(tape['weights']))
        if diagnostic_full_trace:
            for key in ('outer','response','filtered_response','selected_entries','query_embedding'):
                if key in tape:tape[key]=restore(tape[key])
    flat = image.reshape(-1, 3)
    rgb = flat[surface.pixels] if camera else restore(flat[:len(data['face_id'])])
    with stage('material_lighting'):
        return shade_batch(rgb, data, options.get('lighting'),
                           backend='torch' if coefficient_backend == 'eager' else None), tape
