"""Observe existing renderer, optimizer and GPU calls without replacing their math.

Diagnostic benchmarking/profiling with an observer-on/off RGB/VJP gate.
Main API: PipelineProfiler, validate_observer, tensor_storages."""
from contextlib import contextmanager, ExitStack
from functools import wraps
import time
import weakref
import threading
import math


def storage_record(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def tensor_storages(values):
    """Measured unique underlying storages, not logical view sizes."""
    found = {}
    def visit(value):
        if hasattr(value, 'untyped_storage'):
            key = storage_record(value)
            found[key] = key[2]
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value: visit(item)
    visit(values)
    return found


class _Saved:
    def __init__(self, tensor, owner):
        self.tensor = tensor.detach()  # same storage/data, no copy or reference cycle
        self.owner, self.key = weakref.ref(owner), storage_record(tensor)
        owner.saved[self.key] = owner.saved.get(self.key, 0)+1

    def __del__(self):
        owner = self.owner()
        if owner is not None:
            count = owner.saved.get(self.key, 1)-1
            if count: owner.saved[self.key] = count
            else: owner.saved.pop(self.key, None)


class _Kernel:
    def __init__(self, kernel, observer, name):
        self.kernel, self.observer, self.name = kernel, observer, name

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        def call(*args, **kwargs):
            with self.observer.scope(self.name):
                return launch(*args, **kwargs)
        return call


class PipelineProfiler:
    def __init__(self, torch, trainer, backend, device=None):
        self.torch, self.trainer, self.backend = torch, trainer, backend
        self.device = device or next(trainer.model.parameters()).device
        self.cuda = self.device.type == 'cuda'
        self.rows, self.current, self._local = [], None, threading.local()
        self.saved, self.work, self.selector_events, self.optimizer_failures = {}, [], [], []
        self.components, self.extra_components, self.setup_rows, self.validation_rows = {}, {}, [], []

    @property
    def labels(self):
        if not hasattr(self._local,'labels'): self._local.labels=[]
        return self._local.labels

    def allocator(self):
        if not self.cuda: return None
        t = self.torch
        stats = t.cuda.memory_stats(self.device)
        return dict(allocated_bytes=t.cuda.memory_allocated(self.device),
                    reserved_bytes=t.cuda.memory_reserved(self.device),
                    global_peak_allocated_bytes=t.cuda.max_memory_allocated(self.device),
                    allocation_requests=stats.get('allocation.all.allocated'),
                    device_allocations=stats.get('num_device_alloc'))

    @contextmanager
    def scope(self, name):
        if self.current is None:
            yield
            return
        t = self.torch
        with t.profiler.record_function('bench::diagnostic.observer'):
            a = t.cuda.Event(enable_timing=True) if self.cuda else None
            b = t.cuda.Event(enable_timing=True) if self.cuda else None
            memory_before = self.allocator()
            if a is not None: a.record()
            start, cpu = time.perf_counter(), time.process_time()
        self.labels.append(name)
        try:
            with t.profiler.record_function('bench::'+name):
                yield
        finally:
            end, cpu_end = time.perf_counter(), time.process_time()
            with t.profiler.record_function('bench::diagnostic.observer'):
                if b is not None: b.record()
                self.labels.pop()
                self.current['stages'].append(dict(name=name, host_wall_ms=(end-start)*1000,
                    process_cpu_ms=(cpu_end-cpu)*1000, before=memory_before, after=self.allocator(), events=(a,b)))

    def begin(self, iteration, camera, phase='training'):
        self.current = dict(iteration=iteration, camera=camera, phase=phase, stages=[])

    @contextmanager
    def saved_tensors(self):
        def pack(tensor):
            with self.torch.profiler.record_function('bench::diagnostic.saved_tensor_hooks'):
                return _Saved(tensor,self)
        with self.torch.autograd.graph.saved_tensors_hooks(pack, lambda token:token.tensor):
            yield

    def retained(self):
        if self.current is None: return
        known = set().union(*(set(v) for v in self.components.values())) if self.components else set()
        self.current['saved_for_backward'] = dict(unique_storage_bytes=sum(k[2] for k in self.saved),
            storage_count=len(self.saved), known_resident_storage_bytes=sum(k[2] for k in self.saved if k in known),
            other_storage_bytes=sum(k[2] for k in self.saved if k not in known),
            scope='Live saved-tensor-hook storage at forward/loss boundary; includes shared assets/parameters, not additional allocator bytes')

    def finish(self, synchronized_update_ms, final_sync_ms):
        """Called only after the existing synchronized update boundary."""
        if self.current is None: return
        for stage in self.current['stages']:
            a,b = stage.pop('events')
            stage['cuda_event_interval_ms'] = a.elapsed_time(b) if a is not None else None
        self.current.update(synchronized_update_ms=synchronized_update_ms, final_sync_wait_ms=final_sync_ms)
        # Scalar diagnostic reductions were already enqueued, with their own range.
        for row in self.work:
            for name,value in list(row.items()):
                if hasattr(value, 'item'): row[name] = value.item()
            if row['kind']=='support':
                cap=row['support_cap']
                row['mean_support_cap_utilization']=(row['mean_kernels_per_triangle']/cap
                    if cap and row['mean_kernels_per_triangle'] is not None else None)
                row['maximum_support_cap_utilization']=row['maximum_kernels_per_triangle']/cap if cap else None
            elif row['kind']=='splat':
                count=row['emitted_pixel_contributions']
                row['contributions_per_visible_pixel']=count/row['visible_pixels'] if row['visible_pixels'] else None
                row['contribution_capacity_utilization']=count/row['contribution_capacity'] if row['contribution_capacity'] else None
                row['kernel_tile_capacity_utilization']=row['kernel_tile_items']/row['kernel_tile_capacity'] if row['kernel_tile_capacity'] else None
                row['discarded_by_top30_percent']=100*row['discarded_contributions']/count if count else None
        self.current['work'] = self.work
        destination={'setup':self.setup_rows,'validation':self.validation_rows}.get(self.current['phase'],self.rows)
        destination.append(self.current)
        self.current, self.work = None, []

    def residency(self):
        tr = self.trainer
        eigen = tr.eigalbo_interp
        self.components = dict(model_parameters=tensor_storages(list(tr.model.parameters())),
            parameter_gradients=tensor_storages([p.grad for p in tr.model.parameters()]),
            source_face_ids=tensor_storages(getattr(tr.model,'_kernel_face_ids',None)),
            mesh=tensor_storages(list(tr.mesh.buffers()) if hasattr(tr.mesh,'buffers') else [tr.mesh.verts,tr.mesh.faces]),
            spectral_assets=tensor_storages([v for k,v in vars(eigen).items() if k.startswith('_') and hasattr(v,'untyped_storage')]),
            optimizer_state=tensor_storages([opt.state for opt in tr.optimizers]))
        if hasattr(eigen,'faiss_index'):
            self.components['native_source_index_tensors']=tensor_storages(list(vars(eigen.faiss_index).values()))
        self.components.update({name:tensor_storages(callback()) for name,callback in self.extra_components.items()})
        return {name:dict(storage_bytes=sum(values.values()), storages=len(values),
                         cuda_storage_bytes=sum(size for key,size in values.items() if key[0].startswith('cuda')),
                         scope='Measured underlying tensor storages; categories can share storage, do not sum blindly')
                for name,values in self.components.items()}

    def wrap(self, stack, obj, attribute, name, observe=None):
        from unittest.mock import patch
        if not hasattr(obj, attribute): return
        original = getattr(obj, attribute)
        @wraps(original)
        def call(*args, **kwargs):
            with self.scope(name):
                result = original(*args, **kwargs)
            if self.current is not None and observe is not None:
                observe(args, kwargs, result)
            return result
        stack.enter_context(patch.object(obj, attribute, call))

    def support_result(self, args, kwargs, result):
        if 'support.csr' in self.labels: return  # super() fallback, observed by outer call
        support, stats = result
        self.selector_events.append(dict(iteration=self.current['iteration'], **stats))
        with self.scope('diagnostic.work_counts'):
            counts = support.counts[support.geometry.visible_device]
            self.work.append(dict(kind='support', visible_triangles=len(support.geometry.visible),
                kernel_triangle_pairs=support.kernels.numel(), maximum_kernels_per_triangle=support.maximum,
                mean_kernels_per_triangle=(support.kernels.numel()/len(counts) if len(counts) else None),
                support_cap=kwargs.get('pool_cap', args[2] if len(args)>2 else 128),
                support_csr_bytes=sum(v.numel()*v.element_size() for v in (support.counts,support.offsets,support.kernels,support.entry_faces)),
                expanded_triangle_tile_incidences=support.tasks, pixel_contribution_capacity=support.fragments,
                selector=stats))

    def hybrid_result(self, args, kwargs, result):
        rgb, se, sw = result
        work = args[6]
        support, surface = work.support, work.surface
        with self.scope('diagnostic.work_counts'):
            valid = (se >= 0) & (se < support.kernels.numel())
            emitted = (work.record_entries >= 0).sum()
            kept = valid.sum()
            self.work.append(dict(kind='splat', visible_pixels=len(surface.pixels), kernel_tile_items=work.count.detach(),
                kernel_tile_capacity=work.keys.numel(), emitted_pixel_contributions=emitted,
                selected_contributions=kept, selected_positive_contributions=(valid & (sw>0)).sum(),
                discarded_contributions=emitted-kept,
                contribution_capacity=support.fragments,
                contribution_records_bytes=support.fragments*(work.record_entries.element_size()+args[0].element_size()),
                selected_tape_bytes=se.numel()*se.element_size()+sw.numel()*sw.element_size(),
                coefficient_heat_bytes=args[0].numel()*args[0].element_size(),
                coefficient_metric_bytes=args[1].numel()*args[1].element_size(), queue_bytes=work.device_bytes,
                scope='Counts measured from actual forward buffers; record bytes derive from actual tensor shapes/dtypes'))

    @contextmanager
    def install(self):
        """Patch observers in this diagnostic process, restoring every callable."""
        t, tr = self.torch, self.trainer
        from hktex_splatting.modules.spectral_heat_field import LiveHeatField
        from hktex_splatting.knn_heat.packed_support import PreparedSupport
        from hktex_splatting.knn_heat.triangle_kernel_support import (
            ResidentSupportGeometry,
            FusedResidentSupportGeometry,
        )
        from hktex_splatting.rendering.mesh_visibility_raster import BVHRaster
        from hktex_splatting.rendering.resident import ResidentTopology
        from hktex_splatting.rendering import heat_kernel_splatting as direct_tiles
        from hktex_splatting.triton import (
            spectral_heat_coefficients as coefficient_gpu,
            heat_kernel_splatting as direct_gpu,
            splat_contributions as gpu,
            resident_gpu,
            splat_reconstruction as runtime_counts,
        )
        from unittest.mock import patch
        with ExitStack() as stack:
            e, m = tr.eigalbo_interp, tr.model
            for obj,attr,name in (
                (m,'prepare_kernels','native.source_state'), (e,'build_kernel_graph','native.source_graph'),
                (e.faiss_index,'build','native.faiss_build'), (e.faiss_index,'search','native.knn_search'),
                (e.faiss_index,'knn_distances','native.distances'),
                (m,'prepare_points','native.query_preparation'),
                (e.knn_gather,'select_grid','source.grid_selection'), (e.knn_gather,'gather_src','source.spectral_gather'),
                (e.knn_gather,'gather_queries','native.spectral_gather'), (e.knn_heat,'build','native.source_heat'),
                (e.knn_heat,'query','native.spectral_heat'), (m,'kernel_filter_func','native.filter'),
                (m,'diffuse_heat_kernels','native.shading'), (LiveHeatField,'sources','hybrid.source_state'),
                (m,'forward','native.output'),
                (LiveHeatField,'coefficients','hybrid.corner_coefficients'),
                (ResidentTopology,'__init__','hybrid.coefficient_topology'),
                (PreparedSupport,'candidate_rows','support.legacy_candidates'),
                (PreparedSupport,'__init__','support.triangle_embedding_setup'),
                (direct_tiles.TileSurface,'__init__','visibility.tile_geometry'),
                (FusedResidentSupportGeometry,'candidate_rows','support.selector_dispatch'),
                (direct_tiles,'triangle_tile_work','hybrid.queue'), (BVHRaster,'all_buffers','visibility.rasterization')):
                self.wrap(stack,obj,attr,name)
            if getattr(e,'heat_weighting',None) is not None:
                self.wrap(stack,e.heat_weighting,'compute','native.distance_weighting')
            if hasattr(tr,'tracer'):
                self.wrap(stack,tr.tracer,'trace','optimizer.geodesic_trace')
            # Observe only the outer select; super() fallback remains a nested scope.
            for cls in (ResidentSupportGeometry,FusedResidentSupportGeometry):
                self.wrap(stack,cls,'select','support.csr',self.support_result)
            self.wrap(stack,direct_gpu,'hybrid_forward','hybrid.splat',self.hybrid_result)
            for obj,attrs in ((coefficient_gpu,{'coefficients_forward':'spectral.coefficient_forward','coefficients_backward':'spectral.coefficient_backward'}),
                             (direct_gpu,{'_hybrid_emit':'hybrid.emission','_triangle_group_scan':'hybrid.queue_scan','_triangle_group_pack':'hybrid.queue_pack'}),
                             (gpu,{'merge':'hybrid.top30','emit':'resident.emission','reconstruct':'hybrid.rgb'}),
                             (runtime_counts,{'reconstruct':'hybrid.rgb','reconstruct_backward':'backward.rgb','emit_backward':'backward.weights'}),
                             (resident_gpu,{'support_distances':'support.distances','select_support_rows':'support.fused_selection',
                                 'merge_support_tile':'support.tiled_selection','finish_support_tiles':'support.tiled_threshold_output',
                                 'support_scan_blocks':'support.scans','support_scan_totals':'support.scans','support_scan_offsets':'support.scans','pack_support':'support.packing'})):
                for attr,name in attrs.items(): stack.enter_context(patch.object(obj,attr,_Kernel(getattr(obj,attr),self,name)))
            # Fallback Tensor operators are classified from existing profiler
            # CPU-op/launch correlations. Never patch Torch operators or its
            # compiled functions: that could invalidate warmed Dynamo graphs.
            self.residency()
            yield self

    def export(self):
        return dict(format='hktex_pipeline_observer_v1', backend=self.backend, updates=self.rows,
            setup=self.setup_rows,
            validation=self.validation_rows,
            selector_events=self.selector_events, component_memory=self.residency(),
            cuda_available=self.cuda,
            scope='Diagnostic only. Nested stages overlap. CUDA event intervals include host gaps; CUPTI kernels are separate. '
                  'Saved-storage bytes include aliased shared GPU assets. Allocator before/after values are live snapshots, not exclusive stage peaks.')


class ObserverValidationError(AssertionError):
    """Keep failed RGB/VJP diagnostics available to the benchmark caller."""
    def __init__(self, report):
        self.report = report
        failed = [row['parameter'] for row in report['parameter_comparisons']
                  if not row['passed'] or not row['unprofiled_repeat']['passed']]
        super().__init__('Profiler correctness gate failed: '+report['failure_reason']+
                         '; parameters='+str(failed)+'; inspect profile_validation.json')


def validate_observer(observer, render, parameters, *, loss_fn=None, objective=None):
    """Same-backend numerical gate with the actual profiler enabled.

    The training caller supplies its actual foreground pixel-mean loss. The
    historical RGB-squared-sum probe remains the default for API compatibility.
    Tolerances are unchanged. An unprofiled repeat checks numerical stability;
    neither unstable repeats nor observer mismatches are accepted. No updates.
    """
    t=observer.torch
    from splatting_script.benchmarks.training_research import preserve_rng
    parameters=tuple(parameters)
    names={id(p):name for name,p in observer.trainer.model.named_parameters()}
    objective=objective or ('custom_loss' if loss_fn is not None else 'rgb_squared_sum')
    loss_fn=loss_fn if loss_fn is not None else (lambda rgb:rgb.square().sum())
    atol,rtol=(3e-5,3e-4) if observer.cuda else (0.,0.)
    def unprofiled():
        with preserve_rng(t,observer.device):
            rgb=render()
            loss=loss_fn(rgb)
            gradients=t.autograd.grad(loss,parameters,allow_unused=True)
        return rgb.detach().clone(),loss.detach(),tuple(g.detach().clone() if g is not None else None for g in gradients)
    reference,reference_loss,expected=unprofiled()
    repeat,repeat_loss,repeated=unprofiled()
    activities=[t.profiler.ProfilerActivity.CPU]
    if observer.cuda: activities.append(t.profiler.ProfilerActivity.CUDA)
    with preserve_rng(t,observer.device), t.profiler.profile(activities=activities,profile_memory=True,record_shapes=True):
        observer.begin(-2,'correctness',phase='validation')
        with observer.scope('update'):
            with observer.saved_tensors():
                with observer.scope('forward'): actual=render()
                loss=loss_fn(actual);observer.retained()
            with observer.scope('backward'): gradients=t.autograd.grad(loss,parameters,allow_unused=True)
        if observer.cuda: t.cuda.synchronize(observer.device)
        observer.finish(0.,0.)
    def comparison(value,baseline):
        if value is None or baseline is None:
            return dict(passed=value is baseline,gradient_present=value is not None,
                        reference_gradient_present=baseline is not None,max_abs=None)
        if value.shape!=baseline.shape:
            return dict(passed=False,shape=list(value.shape),reference_shape=list(baseline.shape),max_abs=None)
        finite=bool(t.isfinite(value).all() & t.isfinite(baseline).all())
        delta=(value-baseline).detach().abs()
        maximum=float(delta.max()) if delta.numel() else 0.
        return dict(passed=finite and bool(t.allclose(value,baseline,atol=atol,rtol=rtol)),finite=finite,
                    max_abs=maximum if math.isfinite(maximum) else None,
                    mismatched_elements=int((~t.isclose(value,baseline,atol=atol,rtol=rtol)).sum()),elements=value.numel())
    comparisons=[]
    for i,(actual_grad,expected_grad,repeat_grad) in enumerate(zip(gradients,expected,repeated)):
        comparisons.append(dict(parameter=names.get(id(parameters[i]),f'parameter_{i}'),shape=list(parameters[i].shape),
            **comparison(actual_grad,expected_grad),unprofiled_repeat=comparison(repeat_grad,expected_grad)))
    rgb_check,loss_check=comparison(actual,reference),comparison(loss,reference_loss)
    repeat_rgb,repeat_loss_check=comparison(repeat,reference),comparison(repeat_loss,reference_loss)
    repeat_ok=repeat_rgb['passed'] and repeat_loss_check['passed'] and all(row['unprofiled_repeat']['passed'] for row in comparisons)
    observer_ok=rgb_check['passed'] and loss_check['passed'] and all(row['passed'] for row in comparisons)
    report=dict(passed=repeat_ok and observer_ok,device=str(observer.device),objective=objective,
        rgb_max_abs=rgb_check['max_abs'],rgb=rgb_check,loss=loss_check,
        parameter_vjp_max_abs=[row['max_abs'] for row in comparisons if row.get('gradient_present',True)],
        parameter_comparisons=comparisons,unprofiled_repeat=dict(passed=repeat_ok,rgb=repeat_rgb,loss=repeat_loss_check),
        absolute_tolerance=atol,relative_tolerance=rtol,
        failure_reason=('unprofiled_repeatability_failed' if not repeat_ok else 'observer_mismatch' if not observer_ok else None),
        scope='Same backend, same parameters and RNG, all first-order parameter VJPs; actual profiler and saved-tensor observers. '
              'Independent unprofiled repeat must also satisfy the same tolerances; no optimizer updates.')
    if not report['passed']: raise ObserverValidationError(report)
    return report
