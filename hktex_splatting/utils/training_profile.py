"""Summarize correlated CUDA trace activity for retained benchmark diagnostics.

Timing/report calculations are preserved; the historical trainer timer was retired.
Main API: _union_us, summarize_trace."""
import json
from pathlib import Path


def _union_us(events):
    intervals = sorted((float(e['ts']), float(e['ts'])+float(e.get('dur', 0))) for e in events)
    total, end = 0., float('-inf')
    for a, b in intervals:
        total += max(0., b-max(a, end))
        end = max(end, b)
    return total


def summarize_trace(path):
    events = json.loads(Path(path).read_text())['traceEvents']
    complete = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    phases = sorted([e for e in complete if e.get('cat') == 'user_annotation'
                     and e.get('name', '').startswith('training::')], key=lambda e:e['ts'])
    spectra = sorted([e for e in complete if e.get('cat') == 'user_annotation'
                      and e.get('name', '').startswith(('spectral::','support::'))], key=lambda e:e['ts'])
    drains = [e for e in complete if e.get('cat') == 'user_annotation'
              and e.get('name','').startswith('training_sync::') and e['name'].endswith('_after')]
    extended_phases = []
    for phase in phases:
        end = phase['ts']+phase['dur']
        expected = 'training_sync::'+phase['name'].split('::',1)[1]+'_after'
        candidates = [d for d in drains if d['name'] == expected and d['ts'] >= end]
        if candidates:
            drain = min(candidates,key=lambda d:d['ts'])
            extended_phases.append(dict(phase, dur=drain['ts']+drain['dur']-phase['ts']))
    # Profiled phases are sequential and synchronized. Attribution follows the
    # CPU launch correlation, not the later kernel timestamp or CPU duration.
    def enclosing(ts, ranges):
        found = None
        for e in ranges:
            if e['ts'] > ts:
                break
            if ts <= e['ts']+e['dur']:
                found = e['name']
        return found
    launches = {}
    external = {}
    sync_api = []
    for e in complete:
        cat, args = e.get('cat', ''), e.get('args', {})
        if cat in ('cpu_op', 'cuda_runtime', 'cuda_driver'):
            pair = (enclosing(e['ts'], phases), enclosing(e['ts'], spectra))
            if 'External id' in args:
                external[args['External id']] = pair
            if cat in ('cuda_runtime', 'cuda_driver') and 'correlation' in args:
                launches[args['correlation']] = pair
            if cat in ('cuda_runtime', 'cuda_driver') and 'Synchronize' in e.get('name', ''):
                sync_api.append(e)
    groups, kernels = {}, {}
    unattributed = 0
    timestamp_fallback = 0
    for e in complete:
        cat = e.get('cat', '')
        if cat not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        args = e.get('args', {})
        pair = launches.get(args.get('correlation'), external.get(args.get('External id'), (None, None)))
        phase, spectral = pair
        if phase is None:
            # Graph replays may omit a matching launch correlation. Because
            # every measured phase drains the device, actual GPU timestamps
            # inside its phase+drain interval can identify the parent phase.
            # Never infer a nested spectral subset this way.
            phase = enclosing(e['ts'],extended_phases)
            if phase is not None:
                timestamp_fallback += 1
        if phase is None:
            unattributed += 1
        for label in {phase or 'unattributed', spectral} - {None}:
            group = groups.setdefault(label, dict(kernel=[], gpu_memcpy=[], gpu_memset=[]))
            group[cat].append(e)
        if cat == 'kernel':
            key = (phase or 'unattributed', e['name'])
            count, dur = kernels.get(key, (0, 0.))
            kernels[key] = (count+1, dur+float(e['dur']))
    result = {}
    for label, cats in groups.items():
        result[label] = {cat:dict(count=len(rows), duration_sum_ms=sum(float(e['dur']) for e in rows)/1000,
                                 active_union_ms=_union_us(rows)/1000) for cat, rows in cats.items()}
    return dict(phases=result, unattributed_gpu_events=unattributed,
        timestamp_fallback_gpu_events=timestamp_fallback,
        gpu_kernel_count=sum(e.get('cat') == 'kernel' for e in complete),
        synchronization_api=dict(calls=len(sync_api), cpu_wait_wall_sum_ms=sum(e['dur'] for e in sync_api)/1000),
        top_kernels=[dict(phase=p, kernel=k, calls=n, cuda_duration_sum_ms=us/1000)
                     for (p,k),(n,us) in sorted(kernels.items(), key=lambda x:-x[1][1])[:40]],
        scopes=dict(kernel='CUPTI kernel activity durations, not CPU enqueue durations',
            duration_sum='Concurrent kernel durations may overlap; sums are not elapsed wall time',
            spectral='Nested spectral subsets also appear in their parent training phase; do not add both',
            synchronization='CPU API wait time overlaps GPU execution; do not add it to kernel time',
            attribution='CUDA launch correlation/external id, then serialized phase+device-drain timestamp for parent phases only; fallback/unattributed counts remain explicit'))
