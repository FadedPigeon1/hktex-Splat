"""Compare HKTex (Original) and HKTex Splat (Ours), with correctness gates.

Controlled training, benchmarking and numerical diagnostics; dry unless --execute.
Profile/validation runs are separate diagnostics, not steady-state speed evidence.
Main API: parser, main, _execute, JITCacheDiagnostics."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--mode', choices=('native', 'direct'), default='direct')
    p.add_argument('--config', type=Path, default=Path('splatting_script/configs/converse_training.yaml'))
    p.add_argument('--initial-state', type=Path)
    p.add_argument('--kernels', type=int, default=1000)
    p.add_argument('--resolution', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--iterations', type=int, default=40)
    p.add_argument('--support-cap', type=int, default=128)
    p.add_argument('--support-selector', choices=('legacy', 'fused'), default='legacy',
                   help='GPU triangle selection; bounded tiled merging above 4096 sources with deterministic distance/source-ID ties')
    p.add_argument('--debug-support', action='store_true',
                   help='HKS only: synchronize/download support checks and save failure snapshots; diagnostic timing')
    p.add_argument('--coefficient-backend', choices=('eager', 'triton'), default='triton')
    p.add_argument('--runtime-counts', action=argparse.BooleanOptionalAction, default=True,
                   help='Use the validated shared HKS GPU configuration; avoid pair-count specialization')
    p.add_argument('--splat-backend', choices=('triton', 'torch'), default='triton',
                   help='Kernel/tile GPU emission, or the bounded eager spectral oracle')
    p.add_argument('--max-contributions', type=int, default=16777216)
    p.add_argument('--max-buffer-mib', type=int, default=512)
    p.add_argument('--visibility', choices=('reuse', 'fresh'), default='reuse')
    p.add_argument('--validate', action='store_true', help='Compare initial images, contributions, VJPs, optimizer state')
    p.add_argument('--validate-only', action='store_true',
                   help='Run the correctness comparison and stop before warmup/timing; implies --validate')
    p.add_argument('--preflight-only', action='store_true',
                   help='Untimed RGB/loss, finite VJP and optimizer replay gate on the actual workload; requires --initial-state')
    p.add_argument('--profile', action='store_true', help='Separate instrumented diagnostic process, not a speed run')
    p.add_argument('--pipeline-profile', action='store_true',
                   help='Comprehensive diagnostic stage/events/memory profile; implies --profile; never speed evidence')
    p.add_argument('--profile-validate', action='store_true',
                   help='With --pipeline-profile, gate observer-enabled RGB/VJPs before warmup/reset')
    p.add_argument('--profile-steps', type=int,
                   help='Number of active profile updates; preceding trajectory still executes')
    p.add_argument('--profile-start', type=int, default=0,
                   help='Start CUDA/CPU profiling at this measured iteration; earlier updates still execute')
    p.add_argument('--host-profile', action='store_true',
                   help='Separate cProfile diagnostic of Python/native host calls; do not use for speed evidence')
    p.add_argument('--research', action='store_true', help='Compare original HKTex and Heat-Kernel Splatting convergence with periodic quality and archival snapshots')
    p.add_argument('--initialization-only', action='store_true', help='With --research, save a seed-labelled matched initialization and stop')
    p.add_argument('--progress-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--checkpoint-every', type=int, default=1000)
    p.add_argument('--check-every', type=int, default=100, help='Research-only parameter/gradient checks outside step timers; use 1 for pilot')
    p.add_argument('--psnr-thresholds', type=float, nargs='+', default=[20.,25.,30.])
    p.add_argument('--matmul-precision', choices=('highest','high'), help='Explicit common policy; research defaults to highest, ordinary runs keep their existing policy unless set')
    p.add_argument('--allow-selector-fallback', action='store_true', help='Allow explicit selector fallbacks for unsupported inputs; large source counts alone do not require this')
    p.add_argument('--allow-native-variation', action='store_true',
                   help='Diagnostic only: allow bounded native loss/Adam replay variability; keep finite/gradient/input checks')
    # Preserve fixed report/control fields for reading historical benchmark records.
    p.set_defaults(selection='hybrid', footprint_cutoff=0.)
    return p


class JITCacheDiagnostics:
    """Observe Triton's in-memory misses without touching GPU data or launches.

    A miss interval includes compiler work OR disk-cache loading. These hooks
    cannot distinguish the two, so durations are never labeled pure compiler
    time. Existing hooks are chained/restored, including on exceptions.
    """
    def __init__(self, runtime=None):
        import threading
        self.runtime = runtime
        self.lock = threading.Lock()
        self.events, self.pending = [], {}
        self.phase, self.iteration = 'initialization', None
        self.completed = 0
        self.wall = self.cpu = 0.
        self.output = None
        self.status = 'not_installed'

    def set_phase(self, phase, iteration=None):
        self.phase, self.iteration = phase, iteration

    def snapshot(self):
        with self.lock:
            return self.completed, self.wall, self.cpu

    @staticmethod
    def _identity(kwargs):
        info = kwargs['fn']
        module = getattr(info, 'module', 'unknown')
        if not isinstance(module, str): module = getattr(module, '__name__', 'unknown')
        return module+'.'+info.name, kwargs['key']

    @staticmethod
    def _constant(value):
        if value is None or isinstance(value, (str, int, float, bool)): return value
        if isinstance(value, (tuple, list)) and len(value) <= 16:
            return [JITCacheDiagnostics._constant(x) for x in value]
        # In particular, never repr()/item() a Tensor from inside a JIT hook.
        return '<'+type(value).__name__+'>'

    def _before(self, **kwargs):
        answer = self.previous_before(**kwargs) if self.previous_before else None
        if answer: return answer
        name, key = self._identity(kwargs)
        constants = {}
        params = getattr(getattr(kwargs['fn'], 'jit_function', None), 'params', ())
        for path, value in (kwargs.get('compile', {}).get('constants') or {}).items():
            index = path[0] if isinstance(path, tuple) and len(path) == 1 else path
            label = params[index].name if isinstance(index, int) and 0 <= index < len(params) else str(path)
            constants[label] = self._constant(value)
        event = dict(kernel=name, phase=self.phase, iteration=self.iteration, constants=constants,
                     key_sha256=hashlib.sha256(str(key).encode()).hexdigest(),
                     status='pending', wall_seconds=None, process_cpu_seconds=None,
                     _start=time.perf_counter(), _cpu=time.process_time())
        with self.lock:
            self.events.append(event)
            self.pending.setdefault((name, key), []).append(event)
        return answer

    def _after(self, **kwargs):
        identity = self._identity(kwargs)
        wall, cpu = time.perf_counter(), time.process_time()
        with self.lock:
            pending = self.pending.get(identity, [])
            if pending:
                event = pending.pop(0)
                event.update(status='completed', wall_seconds=wall-event['_start'],
                             process_cpu_seconds=cpu-event['_cpu'])
                self.completed += 1
                self.wall += event['wall_seconds']; self.cpu += event['process_cpu_seconds']
        if self.previous_after: return self.previous_after(**kwargs)

    def __enter__(self):
        if self.runtime is None:
            try:
                from triton import knobs
                self.runtime = knobs.runtime
            except (ImportError, AttributeError):
                self.status = 'unavailable'; return self
        if not all(hasattr(self.runtime, n) for n in ('jit_cache_hook', 'jit_post_compile_hook')):
            self.status = 'unavailable'; return self
        self.previous_before, self.previous_after = self.runtime.jit_cache_hook, self.runtime.jit_post_compile_hook
        self.runtime.jit_cache_hook, self.runtime.jit_post_compile_hook = self._before, self._after
        self.status = 'installed'
        return self

    def __exit__(self, *exc):
        if self.status == 'installed':
            self.runtime.jit_cache_hook, self.runtime.jit_post_compile_hook = self.previous_before, self.previous_after

    def report(self):
        with self.lock:
            events = [{k:v for k,v in e.items() if not k.startswith('_')} for e in self.events]
        phases = {}
        for event in events:
            row = phases.setdefault(event['phase'], dict(misses=0, completed=0, wall_seconds=0., process_cpu_seconds=0.))
            row['misses'] += 1
            if event['status'] == 'completed':
                row['completed'] += 1
                row['wall_seconds'] += event['wall_seconds']
                row['process_cpu_seconds'] += event['process_cpu_seconds']
        return dict(status=self.status, phases=phases, events=events,
            scope='Triton in-memory JIT cache misses: compiler OR disk-cache loading, not pure compilation or GPU execution. '
                  'No hit callbacks, Tensor data reads, CUDA launches or synchronizations are added. '
                  'Events are assigned to the phase/iteration where the miss started; unfinished events remain explicit.')


def latency_summary(rows):
    """Retain the short window and subsequent trajectory instead of hiding it."""
    fields = ('seconds', 'host_enqueue_seconds', 'final_sync_wait_seconds', 'process_cpu_seconds')
    def block(values):
        if not values: return None
        return dict(iterations=len(values), median={k:statistics.median(r[k] for r in values) for k in fields},
            jit_cache_misses=sum(r.get('jit_cache_misses', 0) for r in values),
            jit_cache_miss_wall_seconds=sum(r.get('jit_cache_miss_wall_seconds', 0.) for r in values))
    return dict(first_eight=block(rows[:8]), after_first_eight=block(rows[8:]), last_eight=block(rows[-8:]),
        by_camera={name:block([r for r in rows if r.get('camera') == name])
                   for name in sorted({r['camera'] for r in rows if 'camera' in r})},
        scope='Host enqueue includes CPU work, launch overhead and mid-step waits; final drain is not all synchronization. '
              'CPU and wall/compilation/GPU intervals overlap and must not be added.')


def host_profile_summary(profiler):
    import pstats
    stats = pstats.Stats(profiler)
    rows = []
    for (file, line, function), (primitive, calls, self_time, cumulative, _) in stats.stats.items():
        rows.append(dict(file=file, line=line, function=function, calls=calls, primitive_calls=primitive,
                         self_wall_seconds=self_time, cumulative_wall_seconds=cumulative))
    return dict(total_calls=stats.total_calls, total_wall_seconds=stats.total_tt,
                top_calls=sorted(rows, key=lambda r:-r['cumulative_wall_seconds'])[:80],
                scope='cProfile host-call wall time, including blocking native/CUDA API calls; not GPU execution or pure CPU time. '
                      'Nested cumulative times overlap. Instrumented diagnostic, not speed evidence.')


def candidate_profile_summary(events, iterations):
    """Offline CUPTI accounting by launch thread/correlation, not enqueue walls."""
    from hktex_splatting.utils.training_profile import _union_us
    roots = {'support::resident_'+name for name in (
        'candidate_selection_float64', 'count_and_scan', 'allocation_metadata_read', 'compact_csr')}
    complete = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    threads, correlations, external, syncs = {}, {}, {}, []
    stages = ('distance', 'threshold_selection', 'topk', 'masking_output', 'threshold_topk_mask_output',
              'tile_merge', 'threshold_mask_output',
              'scoring_filtering', 'ranking', 'ranking_and_row_output', 'compaction_output', 'allocation_metadata_read')
    def classify(stack):
        selected = [e['name'] for e in stack if e.get('name') in roots]
        if not selected: return None
        root = selected[-1]
        if root.endswith('allocation_metadata_read'): return 'allocation_metadata_read'
        if root.endswith(('count_and_scan', 'compact_csr')): return 'compaction_output'
        for e in reversed(stack):
            name = e.get('name', '')
            for stage in stages:
                if name == 'support::resident_fused_'+stage: return stage
        # Use the outer native operation: cdist's internal copy remains scoring.
        root_index = max(i for i,e in enumerate(stack) if e.get('name') in roots)
        operations = [e['name'] for e in stack[root_index+1:] if e.get('cat') == 'cpu_op']
        name = operations[0] if operations else ''
        if name in ('aten::cdist', 'aten::_cdist_forward'): return 'distance'
        if name == 'aten::kthvalue': return 'threshold_selection'
        if name == 'aten::topk': return 'topk'
        if name in ('aten::masked_fill', 'aten::masked_fill_'): return 'masking_output'
        if name == 'aten::sort': return 'ranking_and_row_output'
        if name in ('aten::gather', 'aten::where', 'aten::sum', 'aten::copy_', 'aten::arange'): return 'compaction_output'
        return 'scoring_filtering'
    for e in complete:
        if e.get('cat') in ('cpu_op', 'user_annotation', 'cuda_runtime', 'cuda_driver'):
            threads.setdefault((e.get('pid'), e.get('tid')), []).append(e)
    allocations = []
    for events_on_thread in threads.values():
        stack = []
        for e in sorted(events_on_thread, key=lambda e:(e['ts'], -e['dur'])):
            while stack and stack[-1]['ts']+stack[-1]['dur'] <= e['ts']: stack.pop()
            if e['cat'] in ('cpu_op', 'user_annotation'): stack.append(e)
            stage = classify(stack)
            if stage and e.get('name') in ('aten::empty', 'aten::empty_strided', 'aten::empty_like'):
                allocations.append(e)
            args = e.get('args', {})
            if stage and 'External id' in args: external[args['External id']] = stage
            if e['cat'] in ('cuda_runtime', 'cuda_driver'):
                if 'correlation' in args: correlations[args['correlation']] = stage
                if stage and 'synchronize' in e.get('name', '').lower(): syncs.append(e)
    grouped = {stage:[] for stage in stages}
    all_gpu, unattributed = [], 0
    for e in complete:
        if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'): continue
        args = e.get('args', {})
        # An explicitly observed out-of-scope launch is not reassigned by GPU timestamps.
        stage = correlations.get(args.get('correlation'), external.get(args.get('External id')))
        if stage:
            grouped[stage].append(e); all_gpu.append(e)
        elif args.get('correlation') not in correlations and args.get('External id') not in external:
            unattributed += 1
    def metrics(items):
        kernels = [e for e in items if e['cat'] == 'kernel']
        copies = [e for e in items if e['cat'] == 'gpu_memcpy']
        names = {}
        for e in kernels:
            row = names.setdefault(e['name'], dict(calls=0, duration_sum_ms=0.))
            row['calls'] += 1; row['duration_sum_ms'] += e['dur']/1000
        return dict(kernel_launches=len(kernels), kernel_launches_per_iteration=len(kernels)/max(iterations, 1),
            kernel_duration_sum_ms=sum(e['dur'] for e in kernels)/1000,
            kernel_duration_sum_ms_per_iteration=sum(e['dur'] for e in kernels)/1000/max(iterations, 1),
            gpu_active_union_ms=_union_us(items)/1000,
            gpu_active_union_ms_per_iteration=_union_us(items)/1000/max(iterations, 1),
            gpu_timeline_span_ms=(max(e['ts']+e['dur'] for e in items)-min(e['ts'] for e in items))/1000 if items else 0.,
            memcpy_calls=len(copies), memcpy_bytes=sum(int(e.get('args', {}).get('Bytes', e.get('args', {}).get('bytes', 0))) for e in copies),
            device_to_host_copies=sum('dtoh' in e.get('name', '').lower() for e in copies),
            memset_calls=sum(e['cat'] == 'gpu_memset' for e in items), kernel_types=names)
    return dict(profiled_iterations=iterations, total=metrics(all_gpu), stages={k:metrics(v) for k,v in grouped.items()},
        unattributed_gpu_events_in_complete_trace=unattributed,
        gpu_to_host_syncs=dict(calls=len(syncs), host_wait_sum_ms=sum(e['dur'] for e in syncs)/1000),
        allocation_operators=dict(calls=len(allocations), host_wall_union_ms=_union_us(allocations)/1000,
            scope='Scoped host allocation operators, including allocator/API waits; not GPU execution.'),
        selection_host_wall_sum_ms=sum(e['dur'] for e in complete
            if e.get('cat') == 'user_annotation' and e.get('name') in roots)/1000,
        fused_stage_breakdown=dict(combined=bool(grouped['threshold_topk_mask_output']),
            threshold_selection_gpu_ms=None if grouped['threshold_topk_mask_output'] else metrics(grouped['threshold_selection'])['kernel_duration_sum_ms'],
            topk_gpu_ms=None if grouped['threshold_topk_mask_output'] else metrics(grouped['topk'])['kernel_duration_sum_ms'],
            masking_output_gpu_ms=None if grouped['threshold_topk_mask_output'] else metrics(grouped['masking_output'])['kernel_duration_sum_ms'],
            scope='Threshold, cap selection, masking and row output share one kernel. Their separate GPU times are unmeasurable within that kernel; use the joint stage.'),
        tiled_stage_breakdown=dict(used=bool(grouped['tile_merge'] or grouped['threshold_mask_output']),
            merge_gpu_ms=metrics(grouped['tile_merge'])['kernel_duration_sum_ms'],
            threshold_output_gpu_ms=metrics(grouped['threshold_mask_output'])['kernel_duration_sum_ms'],
            scope='Bounded source-tile merging and final bound/row output; multiple launches, no native selector fallback.'),
        scope='CUPTI activities attributed by CPU launch thread and correlation/external id. '
              'Active union excludes idle gaps; timeline span includes them. Kernel-duration sums may overlap. '
              'Fused threshold/topk/masking/output stays a joint stage. Native fallback stages are separately attributed. '
              'Synchronization API wall overlaps GPU work. No timers or extra CUDA synchronizations in normal training.')


def selection_diagnostics(plan, sources, cap, mapping):
    """Extra reads/reductions only AFTER timing; no learned support is cached."""
    import torch
    with torch.no_grad():
        candidates = []
        for _, center, radius in plan.chunks:
            distance = torch.cdist(center, sources.double(), compute_mode='donot_use_mm_for_euclid_dist')
            kth = distance.kthvalue(min(plan.neighbors, len(sources)), dim=1).values
            admitted = distance <= kth[:, None]+2*radius[:, None]+1e-5*(1+kth[:, None]+radius[:, None])
            candidates.extend(admitted.sum(1).cpu().tolist())
        per_source = torch.bincount(mapping.kernels.long(), minlength=len(sources)).cpu().tolist()
        return dict(visible_triangles=len(plan.visible), visible_triangle_ids=plan.visible.tolist(),
            eligible_kernels_per_visible_triangle=candidates,
            selected_kernels_per_visible_triangle=mapping.counts[plan.visible_device].cpu().tolist(),
            selected_triangles_per_kernel=per_source, selected_pairs=len(mapping.kernels), support_cap=cap,
            scope='Post-timing fresh diagnostic; cap limits kernels per triangle, not triangles per kernel.')


def reset_model_for_comparison(model, initial):
    """Restore parameters AND clear gradients before a backend/optimizer reset.

    Neither load_state_dict nor constructing a fresh optimizer clears .grad.
    Leaving it populated accumulates previous backends' VJPs and corrupts the
    comparison and first Adam/GeodesicOpt update.
    """
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)


def differences(a, b):
    delta = (a.detach()-b.detach()).abs()
    return dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()))


def common_response_differences(current, reference):
    """Compare heat for the same query/source pair, even when KNN sets differ.

    Sorting two different sets and subtracting slots mixes source identities.
    Missing pairs remain a strict selection failure, reported separately.
    """
    import torch
    # Linear-sized alignment also handles all-source splats. A broadcasted
    # [pixels,sources,sources] identity comparison would exhaust GPU memory.
    ordered, permutation = reference['outer'].sort(1)
    position = torch.searchsorted(ordered.contiguous(), current['outer'].contiguous())
    position = position.clamp_max(ordered.shape[1]-1)
    current_valid = current['outer'] >= 0
    common = (ordered.gather(1, position) == current['outer']) & current_valid
    paired = reference['response'].gather(1, permutation.gather(1, position))
    delta = (current['response'].detach()-paired.detach()).abs()[common]
    count = int(common.sum())
    pairs = int(current_valid.sum())
    return dict(common_pairs=count, current_pairs=pairs,
                max_abs=float(delta.max()) if count else None,
                mean_abs=float(delta.mean()) if count else None,
                complete_pair_coverage=count == pairs,
                passed=bool(torch.allclose(current['response'].detach()[common],
                    paired.detach()[common], atol=3e-5, rtol=3e-4)))


def full_coverage_omissions(current, reference):
    """Diagnostic only: actual emitted support versus full positive responses.

    Padding uses a separate sentinel column so it cannot overwrite source zero.
    This dense mask is built only in validation, never the HKS training path.
    """
    import torch
    positive = reference['filtered_response'] > 0
    p, n = positive.shape
    mask = torch.zeros((p, n+1), device=positive.device, dtype=torch.int32)
    ids = current['outer']
    valid = ids >= 0
    mask.scatter_add_(1, torch.where(valid, ids, n).long(), valid.int())
    admitted = mask[:, :n] > 0
    top = reference['top'].long()
    selected = reference['weights'] > 0
    missing_top = selected & ~admitted.gather(1, top.clamp_min(0))
    return dict(full_positive_contributions=int(positive.sum()),
                omitted_positive_contributions=int((positive & ~admitted).sum()),
                full_positive_top30=int(selected.sum()), omitted_positive_top30=int(missing_top.sum()),
                pixels_missing_positive_top30=int(missing_top.any(1).sum()),
                complete_for_observed_positive_contributions=not bool((positive & ~admitted).any()),
                scope='Initial validation camera only; candidate coverage and tie policy are separate from full-coverage equivalence')


def main(argv=None):
    process_start = time.perf_counter()
    args = parser().parse_args(argv)
    if args.pipeline_profile:
        args.profile = True
    if args.profile_validate and not args.pipeline_profile:
        raise ValueError('--profile-validate requires --pipeline-profile')
    if args.profile_steps is not None and (not args.profile or args.profile_steps < 1 or
            args.profile_start+args.profile_steps > args.iterations):
        raise ValueError('Invalid active profile window')
    if args.pipeline_profile and (args.mode not in ('native', 'direct') or args.debug_support):
        raise ValueError('Pipeline profiles require native/hybrid without --debug-support; diagnose support separately')
    if args.validate_only:
        args.validate = True
    if args.preflight_only and (args.initial_state is None or args.research or args.profile or args.host_profile or args.validate or args.initialization_only):
        raise ValueError('--preflight-only requires --initial-state and a separate uninstrumented process')
    if args.debug_support and args.mode != 'direct':
        raise ValueError('--debug-support requires the hybrid renderer')
    if args.initialization_only and not args.research:
        raise ValueError('--initialization-only requires --research')
    if args.research:
        if not args.initialization_only and args.initial_state is None:
            raise ValueError('Research training requires --initial-state; create one with --initialization-only')
        if (args.mode not in ('native', 'direct') or args.support_cap != 128 or args.splat_backend != 'triton' or (args.support_selector != 'fused') or (args.coefficient_backend != 'triton') or (not args.runtime_counts) or args.profile or args.host_profile or args.validate):
            raise ValueError('Research requires native/hybrid, fused selector, Triton coefficients/splats, cap128, cutoff0, runtime counts; profile/validation separately')
        if min(args.progress_every, args.eval_every, args.checkpoint_every, args.check_every) < 1:
            raise ValueError('Research logging/checkpoint/check intervals must be positive')
        import math
        if not args.psnr_thresholds or not all(math.isfinite(x) for x in args.psnr_thresholds):
            raise ValueError('PSNR thresholds must be finite')
    if (args.kernels < 1 or args.resolution < 8 or args.warmup < 4 or args.iterations < 1 or
            not 0 <= args.support_cap <= 512 or
            args.max_contributions < 1 or args.max_buffer_mib < 1 or
            not 0 <= args.profile_start < args.iterations or (args.profile_start and not args.profile)):
        raise ValueError('Invalid budget; warmup must cover all four training cameras')
    if not args.execute:
        from splatting_script.benchmarks.training_research import backend_label
        print(json.dumps(dict(gpu_started=False, implementation_label=backend_label(args.mode), arguments={k:str(v) if isinstance(v, Path) else v
            for k,v in vars(args).items()}, support_free=False,
            dispatch='kernel_triangle_tile_queue' if args.mode == 'direct' else 'pointwise_outer_knn',
            scope=__doc__), indent=2))
        return
    if args.initial_state is not None:
        args.initial_state = args.initial_state.expanduser().resolve()
        if not args.initial_state.is_file():
            raise FileNotFoundError('--initial-state file does not exist: '+str(args.initial_state)+
                '. Create an initialization with --research --initialization-only and use the printed RESEARCH_INITIAL_STATE path.')
    import os
    from hktex_splatting.utils.experiment_io import gpu_guard, write_json
    gpu_guard()
    if os.environ.get('TRITON_INTERPRET'):
        raise RuntimeError('Unset TRITON_INTERPRET for real GPU validation')
    jit = JITCacheDiagnostics()
    try:
        from contextlib import ExitStack
        with ExitStack() as instrumentation:
            return _execute(args, process_start, jit, instrumentation)
    finally:
        jit.__exit__()
        if jit.output is not None: write_json(jit.output, jit.report())


def _execute(args, process_start, jit, instrumentation=None):
    from types import SimpleNamespace as NS
    from hktex_splatting.utils.experiment_io import (
        configure_cache, unique_output, install_import_paths, write_json, MESH,
    )
    out = unique_output('direct_tiles_'+args.mode)
    from splatting_script.benchmarks.training_research import backend_label
    jit.output = out/'jit_cache.json'
    print('DIRECT_TILES_RUN='+str(out), flush=True)
    print('IMPLEMENTATION='+backend_label(args.mode), flush=True)
    if args.debug_support:
        print('SUPPORT_DEBUG=enabled; support downloads/synchronization are diagnostic, not speed evidence', flush=True)
    if args.mode == 'direct' or args.validate:
        print('DIRECT_TILES_SELECTION=hybrid; triangle support, kernel/tile queue, shared spectral coefficients', flush=True)
    write_json(out/'arguments.json', {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    cache = configure_cache(out.name)
    # Configure the fresh per-run cache BEFORE importing Triton or Torch.
    install_import_paths()
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Execution requires CUDA-enabled PyTorch and a usable NVIDIA GPU; '
                           'CPU command previews remain available without --execute')
    if args.research or args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision or 'highest')
    from interactive_density_knn import load_datamodule_and_trainer_only
    from hktex.trainers.utils import parse_optimizers_and_schedulers
    from hktex.utils import seed_everything
    from hktex_splatting.modules.spectral_heat_field import LiveHeatField
    from hktex_splatting.rendering.heat_kernel_splatting import DirectTileRenderer, TileSurface
    from hktex_splatting.rendering.prepared_geometry import PreparedGeometry
    from hktex_splatting.rendering.resident import ResidentDispatchGeometry, build_scene
    from hktex_splatting.rendering.mesh_visibility_raster import BVHRaster
    from hktex_splatting.rendering.reference_mesh import texture_sample
    from hktex_splatting.rendering.cameras import cameras
    from hktex_splatting.utils.reference_assets import load_cpu_mesh
    # Preserve hooks installed by imports, then observe all setup/training work.
    jit.__enter__()
    seed_everything(args.seed)
    _, _, trainer = load_datamodule_and_trainer_only(NS(config=str(args.config.resolve())), [
        'trainer.eigen_albo.use_precomputed=true', 'trainer.eigen_albo.error_if_not_precomputed=true',
        'optim.iters=0', 'optim.save_model=false', 'optim.save_logs=false',
        'data.use_importance_sampling=false', f'seed={args.seed}', f'trainer.model.n_sources={args.kernels}',
        'use_timestamp=false', 'tag=direct_tiles_control', f'exp_root_dir={cache / "loader"}'])
    m, e, mesh = trainer.model, trainer.eigalbo_interp, trainer.mesh
    if args.initial_state:
        initial = torch.load(args.initial_state, map_location=mesh.verts.device, weights_only=False)
        if args.research or args.preflight_only or args.pipeline_profile:
            from splatting_script.benchmarks.training_research import check_initialization
            check_initialization(initial, args)
        elif initial['kernel_count'] != args.kernels:
            raise ValueError('Initial state kernel count mismatch')
        m.load_state_dict(initial['model'])
    else:
        trainer.data_dependent_initialisation(random_ratio=trainer.cfg.data_initialisation_random_ratio)
    initial = {k:v.detach().clone() for k,v in m.state_dict().items()}
    digest = hashlib.sha256()
    for name, value in sorted(initial.items()):
        digest.update(name.encode()); digest.update(value.cpu().contiguous().numpy().tobytes())
    with (out/'initial.pt').open('xb') as stream:
        metadata = dict(initialization_seed=args.seed, initial_model_sha256=digest.hexdigest()) if args.research else {}
        torch.save(dict(model=initial, kernel_count=m.N_sources, **metadata), stream)
    if args.initialization_only:
        from splatting_script.benchmarks.training_research import provenance
        from hktex_splatting.utils.experiment_io import ROOT
        write_json(out/'research_manifest.json', provenance(args, ROOT, digest.hexdigest(), {}, trainer, torch))
        write_json(out/'initialization.json', dict(seed=args.seed, kernel_count=m.N_sources,
            initial_model_sha256=digest.hexdigest(), path=str(out/'initial.pt'),
            scope='Initialization only, no training/warmup; use this same file for both backends'))
        print('RESEARCH_INITIAL_STATE='+str(out/'initial.pt'), flush=True)
        return
    field = LiveHeatField(trainer, profile=args.profile,
                          coefficient_backend=args.coefficient_backend, support_selector=args.support_selector)
    observer = None
    if args.pipeline_profile:
        from splatting_script.benchmarks.pipeline_profile import PipelineProfiler
        observer = PipelineProfiler(torch, trainer, 'native' if args.mode == 'native' else 'hybrid')
        instrumentation.enter_context(observer.install())
        observer.begin(-1,'setup',phase='setup')
        profile_setup_start=time.perf_counter()
    vertices, faces = mesh.verts.detach().cpu().numpy(), mesh.faces.cpu().numpy()
    raster = BVHRaster(vertices, faces)
    target_mesh = load_cpu_mesh(MESH)
    if not np.array_equal(faces, target_mesh.faces):
        raise ValueError('Target topology mismatch')
    if vertices.shape != target_mesh.vertices.shape or not np.allclose(vertices,target_mesh.vertices,atol=1e-6,rtol=1e-5):
        raise ValueError('Target mesh vertex alignment mismatch')
    names = ('top', 'side', 'three_quarter', 'logo_closeup')
    geometry_names = names+('heldout_left','heldout_right') if args.research or args.preflight_only else names
    cams = {n:dict(c, width=args.resolution, height=args.resolution) for n,c in cameras().items()}
    geometry, direct_renderers, resident_geometry = {}, {}, {}
    if observer is not None:
        observer.extra_components['camera_geometry_and_targets'] = lambda:[list(geometry.values()),[
            [r.surface._buffers, r.hybrid_geometry.geometry._buffers,
             r.hybrid_support.visible_device, [(c,radius) for _,c,radius in r.hybrid_support.chunks]]
            for r in direct_renderers.values()]]
    def make_geometry(name):
        _, depth, _, ids, bary = raster.all_buffers(cams[name])
        target, _ = texture_sample(target_mesh, ids, bary, 'nearest')
        geometry[name] = (ids, bary, torch.tensor(target, device=mesh.verts.device),
                          torch.tensor(ids >= 0, device=mesh.verts.device))
        if args.mode == 'direct' or args.validate:
            surface = TileSurface(mesh.faces, e._iso_embeddings, ids, bary, positions=mesh.verts, depth=depth, camera=cams[name])
            direct_renderers[name] = DirectTileRenderer(
                field,
                surface,
                profile=args.profile,
                splat_backend=args.splat_backend,
                max_contributions=args.max_contributions,
                max_bin_bytes=args.max_buffer_mib * 1024 ** 2,
                support_cap=args.support_cap,
                support_debug_dir=out / 'support_debug' if args.debug_support else None,
            )
        if args.validate:
            fixed = PreparedGeometry(len(faces), ids, bary, mesh.verts.device, dispatch_backend='stable')
            geo = ResidentDispatchGeometry(fixed)
            resident_geometry[name] = (geo, field.prepare_support(ids))
        return ids, bary, depth
    visibility = {name: make_geometry(name) for name in geometry_names}
    if observer is not None:
        torch.cuda.synchronize()
        observer.finish((time.perf_counter()-profile_setup_start)*1000,0.)
    def reset():
        reset_model_for_comparison(m, initial)
        trainer.optimizers, trainer.schedulers = parse_optimizers_and_schedulers(trainer.cfg.optimizers, trainer)
        # load_state_dict also writes embedded mesh buffers/version counters.
        # Rebuild fixed snapshots outside measured iterations after each reset.
        for name in geometry_names: make_geometry(name)
    def annotation(name):
        return torch.profiler.record_function('training::'+name) if args.profile else nullcontext()
    def resident_rgb(scene, mapping, state):
        # Build the internal triangle reference with the shared spectral
        # coefficient and splat operators used by HKS.
        from hktex_splatting.rendering.splat_autograd import splat
        h, gram = field.coefficients(mapping, state, topology=scene.coefficient_topology)
        return splat(scene, h, gram, m.kernel_colours, m._mean_colour, m.thresholds, m.sharpnesses,
            inner_k=m.cfg.knn_inner_k or 30, sigma=e.heat_weighting.config.gaussian_std,
            distance_eps=e.heat_weighting.config.eps, power=m.cfg.power_diffused_diracs,
            runtime_counts=args.runtime_counts)
    def render(mode, name, trace=False):
        ids, bary, _, mask = geometry[name]
        if mode in ('direct', 'kernel_oracle'):
            with annotation('direct_forward'):
                result = direct_renderers[name](trace=trace, oracle=mode == 'kernel_oracle',
                    full_reference=mode == 'kernel_oracle' and args.selection == 'hybrid')
                if trace and mode == 'direct':
                    mapping = result[1]['bins'].support
                    result[1].update(mapping_counts=mapping.counts, mapping_offsets=mapping.offsets,
                                     allocation_metadata=(mapping.maximum, mapping.tasks, mapping.fragments))
                return result
        if mode in ('resident', 'resident_selector_reference'):
            with annotation('support_and_dispatch'):
                state = field.sources()
                geo, plan = resident_geometry[name]
                if mode == 'resident_selector_reference':
                    from hktex_splatting.knn_heat.triangle_kernel_support import ResidentSupportGeometry, StableSupportReference
                    tiled = getattr(direct_renderers[name].hybrid_support, 'selection_algorithm', None) == 'tiled_merge'
                    reference_type = StableSupportReference if tiled else ResidentSupportGeometry
                    plan = reference_type(mesh.faces, e._iso_embeddings, ids, m.cfg.knn_outer_k)
                mapping = field.support(ids, state, pool_cap=args.support_cap, plan=plan,
                                        dispatch_geometry=geo, max_records=args.max_contributions)
                scene = build_scene(mapping, geo, mesh, e._iso_embeddings, profile=args.profile,
                    max_records=args.max_contributions, max_dispatch_bytes=args.max_buffer_mib*1024**2)
            with annotation('live_coefficients_and_forward'):
                if trace:
                    # Read actual selected tape without changing baseline source.
                    from unittest.mock import patch
                    if args.runtime_counts:
                        from hktex_splatting.triton import splat_reconstruction as gpu
                    else:
                        from hktex_splatting.triton import splat_contributions as gpu
                    original = gpu.reconstruct
                    tape = {}
                    original_coefficients = field.coefficients
                    def capture_coefficients(*values, **kwargs):
                        h, gram = original_coefficients(*values, **kwargs)
                        tape['corner_heat'], tape['gram'] = h.detach(), gram.detach()
                        return h, gram
                    class Capture:
                        def __getitem__(self, launch_grid):
                            launch = original[launch_grid]
                            def call(*values, **kwargs):
                                tape['entries'], tape['weights'] = values[0].detach().clone(), values[1].detach().clone()
                                return launch(*values, **kwargs)
                            return call
                    with patch.object(gpu, 'reconstruct', Capture()), \
                         patch.object(field, 'coefficients', capture_coefficients):
                        rgb = resident_rgb(scene, mapping, state)
                    entry = tape.pop('entries')[torch.as_tensor(ids >= 0, device=mesh.verts.device).flatten()]
                    tape['weights'] = tape['weights'][mask.flatten()]
                    valid = (entry < len(mapping.kernels)) & (tape['weights'] > 0)
                    tape['top'] = (torch.where(valid, scene.kernels[entry.clamp(0, len(mapping.kernels)-1).long()], -1)
                                   if len(mapping.kernels) else torch.full_like(entry, -1))
                    tape['support'] = field.last_support_stats
                    tape['mapping_kernels'] = mapping.kernels
                    tape['entry_faces'] = mapping.entry_faces
                    tape['selected_entries'] = entry
                    tape.update(mapping_counts=mapping.counts, mapping_offsets=mapping.offsets,
                                allocation_metadata=(mapping.maximum, mapping.tasks, mapping.fragments))
                    return rgb, tape
                return resident_rgb(scene, mapping, state)
        with annotation('native_source_and_forward'):
            m.prepare_kernels(mesh, e, save_barycentric=True)
            fi = torch.as_tensor(ids[ids >= 0], device=mesh.verts.device)
            bc = torch.as_tensor(bary[ids >= 0], dtype=mesh.verts.dtype, device=mesh.verts.device)
            parts, tops, ws, responses, outers, queries, search_distances = [], [], [], [], [], [], []
            for begin in range(0, len(fi), 256):
                if trace:
                    from unittest.mock import patch
                    original_search = e.faiss_index.search
                    def capture_search(q, k):
                        distance, picked = original_search(q, k)
                        queries.append(q.detach().clone())
                        search_distances.append(distance.detach().clone())
                        return distance, picked
                    with patch.object(e.faiss_index, 'search', capture_search):
                        info = m.prepare_points(mesh, e, fi[begin:begin+256], barys=bc[begin:begin+256])
                else:
                    info = m.prepare_points(mesh, e, fi[begin:begin+256], barys=bc[begin:begin+256])
                value, _, top, w = m.diffuse_heat_kernels(e, info)
                parts.append(m(value))
                if trace:
                    tops.append(top[:, :, 0].T); ws.append(w[:, :, 0].T); outers.append(info['indices'])
                    responses.append(e.diffuse_heat(info['albo_evals'], info['albo_evecs'], info['indices'], info['weights'])[1])
            rgb = m._mean_colour.sum()*0+mesh.verts.new_zeros((*ids.shape, 3))
            rgb[mask] = torch.cat(parts)
            if trace:
                return rgb, dict(top=torch.cat(tops), weights=torch.cat(ws), outer=torch.cat(outers),
                                 response=torch.cat(responses), query_embedding=torch.cat(queries),
                                 source_embedding=e.faiss_index._db.detach(), search_distances=torch.cat(search_distances))
            return rgb
    def uninstrumented_step(mode, name):
        for opt in trainer.optimizers: opt.zero_grad(set_to_none=True)
        if args.visibility == 'fresh':
            with annotation('visibility'): make_geometry(name)
        rgb = render(mode, name)
        _, _, target, mask = geometry[name]
        loss = (rgb[mask]-target[mask]).square().sum(-1).mean()
        with annotation('backward'): loss.backward()
        with annotation('optimizer'):
            for opt in trainer.optimizers: opt.step()
            for scheduler in trainer.schedulers: scheduler.step()
            m.post_optimizer_step()
        return loss

    def profiled_step(mode, name):
        stage = observer.scope
        with stage('update'):
            with stage('zero_grad'):
                for opt in trainer.optimizers: opt.zero_grad(set_to_none=True)
            if args.visibility == 'fresh':
                with annotation('visibility'), stage('visibility.rasterization_and_preparation'): make_geometry(name)
            elif observer is not None:
                with stage('visibility.reuse'): pass
            with observer.saved_tensors():
                with stage('forward'):
                    rgb = render(mode, name)
                _, _, target, mask = geometry[name]
                with stage('loss'):
                    loss = (rgb[mask]-target[mask]).square().sum(-1).mean()
                with stage('diagnostic.storage_inventory'):
                    observer.residency()
                    observer.retained()
            with annotation('backward'), stage('backward'): loss.backward()
            with annotation('optimizer'), stage('optimizer'):
                for opt in trainer.optimizers:
                    with stage('optimizer.geodesic' if type(opt).__name__ == 'GeodesicOpt' else 'optimizer.other'):
                        opt.step()
                with stage('optimizer.scheduler_and_clamp'):
                    for scheduler in trainer.schedulers: scheduler.step()
                    m.post_optimizer_step()
            return loss
    step = profiled_step if observer is not None else uninstrumented_step
    validation = {}
    if args.validate:
        # Each mode starts at exactly the same parameters and optimizer state.
        records = {}
        validation_modes = ('native', 'resident', 'direct') + (('kernel_oracle',))
        if args.support_selector == 'fused': validation_modes += ('resident_selector_reference',)
        for mode in validation_modes:
            jit.set_phase('validation_'+mode)
            reset()
            if any(p.grad is not None for p in m.parameters()):
                raise RuntimeError('Comparison reset left stale gradients')
            rgb, tape = render(mode, names[0], True)
            _, _, target, mask = geometry[names[0]]
            loss = (rgb[mask]-target[mask]).square().sum(-1).mean()
            loss.backward()
            grads = {n:p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                     for n,p in m.named_parameters() if p.requires_grad}
            for opt in trainer.optimizers: opt.step()
            m.post_optimizer_step()
            updates = {n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
            moments = {}
            for opt in trainer.optimizers:
                for n,p in m.named_parameters():
                    for key in ('exp_avg', 'exp_avg_sq'):
                        if p in opt.state and key in opt.state[p]: moments[n+'/'+key] = opt.state[p][key].detach().clone()
            records[mode] = dict(rgb=rgb.detach(), gradients=grads, parameters=updates, moments=moments, tape=tape)
        if args.support_selector == 'fused':
            reference = records['resident_selector_reference']
            selector_checks = {}
            for mode in ('resident', 'direct'):
                current = records[mode]
                checks = {name:torch.equal(current['tape'][name], reference['tape'][name])
                          for name in ('mapping_kernels', 'entry_faces', 'mapping_counts', 'mapping_offsets')}
                checks['allocation_metadata'] = current['tape']['allocation_metadata'] == reference['tape']['allocation_metadata']
                checks['image'] = bool(torch.allclose(current['rgb'], reference['rgb'], atol=3e-5, rtol=3e-4))
                for group in ('gradients', 'parameters', 'moments'):
                    checks[group] = all(torch.allclose(t, reference[group][n], atol=3e-5, rtol=3e-4)
                                        for n,t in current[group].items())
                selector_checks[mode] = dict(reference='resident_selector_reference',
                    tie_policy='distance_then_source_id' if getattr(direct_renderers[names[0]].hybrid_support,
                        'selection_algorithm', None) == 'tiled_merge' else 'native_topk',
                    checks=checks, passed=all(checks.values()))
            validation['selectors'] = selector_checks
            if not all(row['passed'] for row in selector_checks.values()):
                write_json(out/'validation.json', validation)
                raise RuntimeError('Fused/independent selector validation differs; inspect '+str(out/'validation.json'))
        ref = records['native']
        for mode in ('resident', 'direct'):
            row, current = {}, records[mode]
            native_ref = records['native']
            ref_name = 'resident' if mode == 'direct' else 'native'
            ref = records[ref_name]
            row['reference'] = ref_name
            if ref is not native_ref:
                row['vs_native'] = dict(image=differences(current['rgb'], native_ref['rgb']),
                    semantics='Triangle support/all-positive splats differ from native outer-50; these differences are not native parity evidence')
            for key in ('gradients', 'parameters', 'moments'):
                row[key] = {n:differences(t, ref[key][n]) for n,t in current[key].items()}
            row['image'] = differences(current['rgb'], ref['rgb'])
            # Positive selected IDs and corresponding filtered weights;
            # zero-weight native ties are deliberately excluded.
            for label, tape in ((row['reference'], ref['tape']), (mode, current['tape'])):
                np.savez(out/(label+'_tape.npz'), **{key:value.detach().cpu().numpy()
                    for key,value in tape.items() if isinstance(value, torch.Tensor)})
            row['top30'] = 'Actual IDs/weights saved in *_tape.npz; inspect ties and union differences'
            if mode == 'direct':
                row['selection_backend'] = current['tape']['selection_backend']
                row['tile_candidates_used_for_selection'] = current['tape']['tile_candidates_used_for_selection']
                row['faiss_index_rebuilt'] = current['tape']['faiss_index_rebuilt']
                ca, cb = current['tape'], ref['tape']
                def positive_ids(t, width):
                    result = torch.where(t['weights'] > 0, t['top'], -1)
                    return torch.nn.functional.pad(result, (0, width-result.shape[1]), value=-1).sort(1).values
                selected_width = max(ca['top'].shape[1], cb['top'].shape[1])
                row['positive_top30_same_set_fraction'] = float((positive_ids(ca, selected_width) == positive_ids(cb, selected_width)).all(1).float().mean())
                row['tolerance'] = dict(atol=3e-5, rtol=3e-4)
                checks = dict(image=bool(torch.allclose(current['rgb'], ref['rgb'], atol=3e-5, rtol=3e-4)))
                for key in ('gradients', 'parameters', 'moments'):
                    checks[key] = all(torch.allclose(t, ref[key][n], atol=3e-5, rtol=3e-4) for n,t in current[key].items())
                checks['support_sets'] = all(torch.equal(ca[k], cb[k]) for k in ('mapping_kernels','entry_faces'))
                row['triangle_coefficients'] = {k:differences(ca[k], cb[k]) for k in ('corner_heat','gram')}
                checks['triangle_coefficients'] = all(torch.allclose(ca[k], cb[k], atol=3e-5, rtol=3e-4) for k in ('corner_heat','gram'))
                checks['kernel_tile_architecture'] = (ca['dispatch'] == 'kernel_triangle_tile_queue' and
                    ca['kernel_triangle_support_constructed'] and not ca['faiss_index_rebuilt'])
                full = records['kernel_oracle']
                row['vs_full_coverage'] = dict(image=differences(current['rgb'], full['rgb']),
                    omissions=full_coverage_omissions(ca, full['tape']))
                row['response_on_admitted_source_ids'] = common_response_differences(ca, full['tape'])
                checks['spectral_responses'] = (row['response_on_admitted_source_ids']['passed'] and
                    row['response_on_admitted_source_ids']['complete_pair_coverage'])
                row['support'] = ca['support']; row['projection'] = ca['projection']
                row['native_parity_claimed'] = row['full_coverage_equivalence_claimed'] = False
                row['footprint_cutoff'] = args.footprint_cutoff
                np.savez(out/'kernel_oracle_tape.npz', **{k:v.detach().cpu().numpy()
                    for k,v in full['tape'].items() if isinstance(v, torch.Tensor)})
                checks['positive_top30_sets'] = row['positive_top30_same_set_fraction'] == 1
                row['checks'] = checks; row['gradients_cleared_between_backends'] = True
                row['passed'] = all(checks.values())
                validation[mode] = row
                continue
            validation[mode] = row
        write_json(out/'validation.json', validation)
        if not validation['direct']['passed']:
            raise RuntimeError('Direct/'+validation['direct']['reference']+' validation differs; inspect '+str(out/'validation.json'))
        if args.validate_only:
            print('DIRECT_TILES_VALIDATION=PASSED; reference='+validation['direct']['reference']+
                  '; '+str(out/'validation.json'), flush=True)
            return
    if args.profile_validate:
        from splatting_script.benchmarks.pipeline_profile import validate_observer, ObserverValidationError
        _, _, profile_target, profile_mask = geometry[names[0]]
        def profile_loss(rgb):
            return (rgb[profile_mask]-profile_target[profile_mask]).square().sum(-1).mean()
        try:
            profile_validation = validate_observer(observer,lambda:render(args.mode,names[0]),tuple(m.parameters()),
                loss_fn=profile_loss,objective='foreground_rgb_squared_error_pixel_mean')
        except ObserverValidationError as error:
            write_json(out/'profile_validation.json',error.report)
            raise
        write_json(out/'profile_validation.json',profile_validation)
    def support_stats(name):
        plan = direct_renderers[name].hybrid_support
        return dict(fused_chunks=plan.fused_chunks, selector_fallback=plan.fallback_reason,
                    native_distance_fallback=plan.native_distance_fallback,
                    native_topk_fallback=plan.native_topk_fallback,
                    selection_algorithm=plan.selection_algorithm, selection_tie_policy=plan.selection_tie_policy,
                    dispatch='kernel_triangle_tile_queue')
    if args.research or args.pipeline_profile or args.preflight_only:
        from splatting_script.benchmarks.training_research import capture_rng, restore_rng, provenance, model_hash, run_training, geometry_identity
        from hktex_splatting.utils.experiment_io import ROOT
        manifest = provenance(args, ROOT, digest.hexdigest(), cams, trainer, torch)
        manifest['geometry_identity'] = geometry_identity(geometry)
        write_json(out/'research_manifest.json', manifest)
    if args.preflight_only:
        from splatting_script.benchmarks.training_research import run_preflight
        jit.set_phase('preflight')
        return run_preflight(args,out,torch,trainer,initial,render,geometry,reset,step,manifest,support_stats)
    reset()
    if args.research or args.pipeline_profile:
        rng_before_warmup = capture_rng(torch, mesh.verts.device)
    torch.cuda.synchronize()
    warm_start = time.perf_counter()
    setup_before_warmup_seconds = warm_start-process_start
    warmup_rows = []
    for i in range(args.warmup):
        jit.set_phase('warmup', i)
        misses_before = jit.snapshot()
        start = time.perf_counter()
        name = names[i%len(names)]
        step(args.mode, name)
        torch.cuda.synchronize()
        misses_after = jit.snapshot()
        warmup_rows.append(dict(camera=name, seconds=time.perf_counter()-start,
            jit_cache_misses=misses_after[0]-misses_before[0],
            jit_cache_miss_wall_seconds=misses_after[1]-misses_before[1]))
    warm_seconds = time.perf_counter()-warm_start
    reset_start = time.perf_counter()
    jit.set_phase('post_warmup_reset')
    reset()  # Warm compilation/allocator caches, reset learned state and moments.
    torch.cuda.synchronize()
    post_warmup_reset_seconds = time.perf_counter()-reset_start
    if args.research or args.pipeline_profile:
        restore_rng(torch, rng_before_warmup)
        if model_hash(m) != digest.hexdigest() or any(o.state for o in trainer.optimizers):
            raise RuntimeError('Warmup reset did not restore initial model and empty optimizer state')
    if args.research:
        manifest['warmup_reset'] = dict(model_restored=True, optimizers_recreated=True,
            schedulers_recreated=True, rng_restored=True, next_camera=names[0])
        for name, (ids, bary, depth) in visibility.items():
            np.savez(out/(name+'_visibility.npz'), face_ids=ids, bary=bary, depth=depth)
        result = run_training(args, out, torch, trainer, initial, render, geometry, support_stats, jit,
            manifest, process_start, dict(step_function=step,
                setup_and_warmup_seconds=time.perf_counter()-process_start,
                setup_before_warmup_seconds=setup_before_warmup_seconds,
                warmup_seconds=warm_seconds, post_warmup_reset_seconds=post_warmup_reset_seconds,
                first_camera_visits=warmup_rows[:4], compilation_seconds=None))
        print('DIRECT_TILES_RUN='+str(out), flush=True)
        return result
    setup_seconds = time.perf_counter()-process_start
    torch.cuda.reset_peak_memory_stats()
    from contextlib import ExitStack
    host_profiler = None
    if args.host_profile:
        import cProfile
        host_profiler = cProfile.Profile()
    rows = []
    prof = None
    with ExitStack() as stack:
        for i in range(args.iterations):
            # Run the same preceding optimizer trajectory before a late trace.
            # Profiling setup/flush and Python profiling overhead are diagnostic.
            if args.profile and i == args.profile_start:
                prof = stack.enter_context(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=args.pipeline_profile))
            active_profile = args.profile and args.profile_start <= i < args.profile_start+(args.profile_steps or args.iterations)
            if args.profile and args.profile_steps is not None and i == args.profile_start+args.profile_steps:
                stack.close()
            if observer is not None and active_profile: observer.begin(i, names[i%len(names)])
            torch.cuda.synchronize()
            allocator = torch.cuda.memory_stats()
            jit.set_phase('measurement', i)
            misses_before = jit.snapshot()
            start, cpu_start = time.perf_counter(), time.process_time()
            if host_profiler: host_profiler.enable()
            try:
                loss = step(args.mode, names[i%len(names)])
            finally:
                if host_profiler: host_profiler.disable()
            enqueue_seconds = time.perf_counter()-start
            sync_start = time.perf_counter(); torch.cuda.synchronize()
            sync_seconds = time.perf_counter()-sync_start
            seconds = time.perf_counter()-start
            update_cpu_seconds = time.process_time()-cpu_start
            if observer is not None: observer.finish(seconds*1000, sync_seconds*1000)
            after = torch.cuda.memory_stats()
            misses_after = jit.snapshot()
            rows.append(dict(iteration=i, camera=names[i%len(names)], seconds=seconds, host_enqueue_seconds=enqueue_seconds,
                final_sync_wait_seconds=sync_seconds,
                process_cpu_seconds=update_cpu_seconds, loss=float(loss.detach()),
                profiled=bool(active_profile),
                jit_cache_misses=misses_after[0]-misses_before[0],
                jit_cache_miss_wall_seconds=misses_after[1]-misses_before[1],
                jit_cache_miss_process_cpu_seconds=misses_after[2]-misses_before[2],
                allocator={k:after.get(k, 0)-allocator.get(k, 0) for k in
                    ('allocation.all.allocated', 'num_device_alloc', 'num_device_free', 'num_sync_all_streams')}))
    training_peak = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    if args.profile:
        prof.export_chrome_trace(str(out/'trace.json'))
        from hktex_splatting.utils.training_profile import summarize_trace
        write_json(out/'cuda_activity.json', summarize_trace(out/'trace.json'))
        write_json(out/'candidate_activity.json', candidate_profile_summary(
            json.loads((out/'trace.json').read_text())['traceEvents'], sum(r['profiled'] for r in rows)))
        from splatting_script.benchmarks.training_research import stage_activity
        write_json(out/'stage_activity.json', stage_activity(
            json.loads((out/'trace.json').read_text())['traceEvents'], sum(r['profiled'] for r in rows)))
        if observer is not None:
            from splatting_script.benchmarks.pipeline_trace import analyze_trace
            observed = observer.export()
            from splatting_script.benchmarks.training_research import parameter_health
            observed.update(initial_model_sha256=digest.hexdigest(), precision=manifest['precision'],
                controls=manifest, health=parameter_health(torch,m,initial,mesh=mesh))
            write_json(out/'pipeline_observations.json', observed)
            write_json(out/'pipeline_profile.json', analyze_trace(
                json.loads((out/'trace.json').read_text())['traceEvents'], observed))
    if host_profiler:
        write_json(out/'host_profile.json', host_profile_summary(host_profiler))
    jit.set_phase('quality_and_diagnostics')
    quality = {}
    with torch.no_grad():
        quality_names = names+('heldout_left', 'heldout_right')
        for name in quality_names:
            if name not in geometry:
                visibility[name] = make_geometry(name)
            rgb = render(args.mode, name)
            reference = render('native', name)
            ids, _, target, mask = geometry[name]
            mse = float((rgb[mask]-target[mask]).square().mean())
            quality[name] = dict(foreground_mse=mse, psnr_db=-10*np.log10(max(mse, 1e-30)),
                                 vs_native=differences(rgb, reference),
                                 split='training_view' if name in names else 'heldout_view')
            if args.pipeline_profile:
                native_mse=float((reference[mask]-target[mask]).square().mean())
                quality[name]['native_field']=dict(foreground_mse=native_mse,psnr_db=-10*np.log10(max(native_mse,1e-30)))
            np.save(out/(name+'_rgb.npy'), rgb.cpu().numpy())
        if names[0] not in direct_renderers:
            ids, bary, _, _ = geometry[names[0]]
            direct_renderers[names[0]] = DirectTileRenderer(
                field,
                TileSurface(mesh.faces, e._iso_embeddings, ids, bary, camera=cams[names[0]]),
                profile=args.profile,
                splat_backend=args.splat_backend,
                max_contributions=args.max_contributions,
                max_bin_bytes=args.max_buffer_mib * 1024 ** 2,
                support_cap=args.support_cap,
                support_debug_dir=out / 'support_debug' if args.debug_support else None,
            )
        _, tape = direct_renderers[names[0]](trace=True)
        bins = tape['bins']
        counts = bins.counts.cpu().numpy()
        candidate_stats = dict(per_tile=counts.tolist(), mean=float(counts.mean()), maximum=int(counts.max()),
            bin_device_bytes=bins.device_bytes, spectral_pairs=int(tape['spectral_pairs']),
            selection_backend=tape['selection_backend'],
            tile_candidates_used_for_selection=tape['tile_candidates_used_for_selection'],
            faiss_index_rebuilt=tape['faiss_index_rebuilt'],
            scope='Post-timing direct-path diagnostic on final state, not included in training iteration time')
        candidate_stats.update(queue_count=int(tape['queue_count']), queue_capacity=tape['queue_capacity'],
            contribution_capacity=tape['contribution_capacity'], backward_records=tape['backward_records'],
            dispatch=tape['dispatch'], footprint_cutoff=tape['footprint_cutoff'], semantics=tape['semantics'])
        candidate_stats.update(support=tape['support'], projection=tape['projection'],
            coefficient_pairs=tape['coefficient_pairs'], pixel_contributions=tape['pixel_contributions'],
            emitted_pixel_contributions=tape['emitted_pixel_contributions'],
            spectral_corner_evaluations=tape['spectral_corner_evaluations'], spectral_modes=tape['spectral_modes'])
        if args.mode == 'direct':
            support_plan = direct_renderers[names[0]].hybrid_support
            geo = direct_renderers[names[0]].hybrid_geometry
            state = field.sources()
            mapping, _ = support_plan.select(state.embedding, args.support_cap, dispatch_geometry=geo,
                max_records=args.max_contributions, max_dispatch_bytes=args.max_buffer_mib*1024**2)
            candidate_stats['selection'] = selection_diagnostics(support_plan, state.embedding, args.support_cap, mapping)
            candidate_stats['selection']['camera'] = names[0]
    for name, (ids, bary, depth) in visibility.items():
        np.savez(out/(name+'_visibility.npz'), face_ids=ids, bary=bary, depth=depth)
    write_json(out/'iterations.json', rows)
    write_json(out/'report.json', dict(mode=args.mode, implementation_label=backend_label(args.mode),
        initial_sha256=digest.hexdigest(), final_training_loss=rows[-1]['loss'],
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        setup_and_warmup_seconds=setup_seconds, warmup_seconds=warm_seconds,
        setup_before_warmup_seconds=setup_before_warmup_seconds,
        first_camera_visits=warmup_rows[:len(names)],
        remaining_warmup=warmup_rows[len(names):],
        post_warmup_reset_seconds=post_warmup_reset_seconds,
        compilation_seconds=None,
        compilation_scope='First camera visits include JIT and training work; later cache misses, if any, '
            'remain included in measured iterations. Inspect jit_cache.json and per-iteration miss counters. '
            'Miss wall/CPU intervals include compiler OR disk-cache loading; isolated compiler duration is not measured.',
        steady_iteration_median_seconds=statistics.median(r['seconds'] for r in rows),
        steady_iteration_p95_seconds=sorted(r['seconds'] for r in rows)[max(0, (95*len(rows)+99)//100-1)],
        measured_training_seconds=sum(r['seconds'] for r in rows),
        process_elapsed_before_final_report_seconds=time.perf_counter()-process_start,
        peak_allocated_bytes=training_peak[0], peak_reserved_bytes=training_peak[1],
        latency_windows=latency_summary(rows), jit_cache=jit.report(),
        measurement_diagnostic=bool(args.profile or args.validate or args.host_profile or args.debug_support),
        quality=quality, heldout_quality={n:quality[n] for n in ('heldout_left','heldout_right')},
        candidates=candidate_stats, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        torch_git_version=torch.version.git_version,
        timing_scope='Synchronized whole iteration including support/bounds, forward, loss, backward and native optimizers. '
            'Host enqueue walls and CUDA event spans are not isolated kernel time. '
            'Use trace.json CUPTI activities for kernel time; use /usr/bin/time for full process including imports/output. '
            'Profile/validation runs are separate diagnostics. Fixed geometry is reused unless --visibility fresh.',
        correctness_scope='HKS uses shared triangle support/cap, coefficients and stable entry-order ties, with its kernel/tile queue. '
            'Its gate compares against the internal triangle reference and reports omissions versus the full spectral oracle. '
            'The internal full-spectrum oracle is diagnostic; the pointwise baseline retains native outer-50 selection. '
            'HKS triangle support can omit full-positive contributions. '
            'Reference triangle-union/cap/tie behavior remains unchanged. No speedup claimed by this harness.'))
    print('DIRECT_TILES_RUN='+str(out), flush=True)


if __name__ == '__main__':
    main()
