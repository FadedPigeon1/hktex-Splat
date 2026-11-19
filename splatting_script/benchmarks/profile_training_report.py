"""Read synchronized training profiles without importing Torch or probing GPUs.

Benchmark/profiling report review.
Main API: review, main."""
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))
import argparse
import json
from pathlib import Path
import statistics

from hktex_splatting.utils.experiment_io import write_json


def review(run):
    run=Path(run)
    load = lambda name:json.loads((run/name).read_text())
    if (run/'pipeline_profile.json').is_file():
        result=load('pipeline_profile.json')
        return dict(result,run=str(run.resolve()),mode=result['backend'])
    manifest = load('manifest.json')
    if not manifest['arguments']['profile']:
        raise ValueError('Requires a synchronized --profile run, not enqueue-only timings')
    losses = load('losses.json')
    active = [r for r in losses if r.get('in_active_trace_window')]
    if not active or not all(r.get('phase_profile') for r in active):
        raise ValueError('Missing synchronized per-phase/window records')
    trace = load('profile_summary.json')
    if not trace.get('gpu_kernel_count',0):
        raise ValueError('No CUPTI GPU kernel activity was captured; enqueue/stream timing cannot replace a GPU execution profile')
    support_host={}
    for event in load('training_trace.json')['traceEvents']:
        if event.get('ph')=='X' and event.get('cat')=='user_annotation' and event.get('name','').startswith('support::'):
            support_host[event['name']]=support_host.get(event['name'],0)+event['dur']/1000/len(active)
    phases = {}
    for name in active[0]['phase_profile']:
        details = [r['phase_profile'][name] for r in active]
        kernel = trace['phases'].get('training::'+name,{}).get('kernel',{})
        copies = trace['phases'].get('training::'+name,{}).get('gpu_memcpy',{})
        phases[name] = dict(
            phase_wall_mean_ms=statistics.mean(r['phases'][name+'_ms'] for r in active),
            process_cpu_mean_ms=statistics.mean(d['process_cpu_ms'] for d in details),
            terminated_child_cpu_mean_ms=statistics.mean(d.get('terminated_child_cpu_ms',0) for d in details),
            host_enqueue_wall_mean_ms=statistics.mean(d['host_enqueue_wall_ms'] for d in details),
            explicit_sync_wait_mean_ms=statistics.mean(d['sync_before_wall_ms']+d['sync_after_wall_ms'] for d in details),
            cuda_kernel_duration_sum_per_step_ms=kernel.get('duration_sum_ms',0)/len(active),
            cuda_kernel_active_union_per_step_ms=kernel.get('active_union_ms',0)/len(active),
            cuda_memcpy_duration_sum_per_step_ms=copies.get('duration_sum_ms',0)/len(active),
            cuda_kernel_launches=kernel.get('count',0))
    return dict(run=str(run.resolve()),mode=manifest['arguments']['mode'],
        coefficient_backend=manifest['arguments'].get('coefficient_backend','eager'),
        runtime_counts=manifest['arguments'].get('runtime_counts',False),
        reuse_camera_geometry=manifest['arguments'].get('reuse_camera_geometry',False),
        dispatch_backend=manifest['arguments'].get('dispatch_backend','original'),
        reuse_coefficient_topology=manifest['arguments'].get('reuse_coefficient_topology',False),
        gpu_dispatch=manifest['arguments'].get('gpu_dispatch',False),
        support_backend=manifest['arguments'].get('support_backend','eager'),
        initial_model_sha256=manifest['initial_model_sha256'],precision=manifest.get('precision'),
        hardware={k:manifest[k] for k in ('gpu','gpu_uuid','torch_version','cuda_version')},
        setup=load('setup_stages.json'),active_steps=[r['step'] for r in active],phases=phases,
        nested_spectral_activity={k:v for k,v in trace['phases'].items() if k.startswith('spectral::')},
        nested_support_activity={k:v for k,v in trace['phases'].items() if k.startswith('support::')},
        nested_support_host_annotation_wall_ms_per_step=support_host,
        unattributed_gpu_events=trace['unattributed_gpu_events'],
        timestamp_fallback_gpu_events=trace.get('timestamp_fallback_gpu_events',0),
        synchronization_api=trace['synchronization_api'],top_kernels=trace['top_kernels'],
        interpretation='Phase and nested annotation walls include host waiting. CPU process time, API waits and CUDA kernel sums overlap; never add them or interpret enqueue/stream intervals as GPU execution.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    runs=[review(r) for r in args.runs]
    if len({r['initial_model_sha256'] for r in runs}) != 1 or any(r['precision']!=runs[0]['precision'] for r in runs):
        raise ValueError('Profile initialization/precision differs')
    result=dict(runs=runs,scope='Synchronized diagnostic profiles only; use separate unprofiled runs for step-time comparisons')
    if args.output:
        write_json(args.output,result)
    else:
        print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
