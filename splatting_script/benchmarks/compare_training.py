"""Compare HKTex (Original) and HKTex Splat (Ours) at measured image quality.

Benchmark reporting and workflow/scaling command routing; standard library only.
Main API: compare, native_split_psnr, main."""
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from splatting_script.benchmarks.training_research import backend_label



def native_split_psnr(cameras: list, heldout: bool) -> float:
    """Native-field PSNR from mean camera MSE, with equal camera weighting."""
    errors = [c['metrics']['native']['foreground_mse'] for c in cameras if c['heldout'] == heldout]
    if not errors:
        raise ValueError('Missing camera split')
    if any(not math.isfinite(error) or error < 0 for error in errors):
        raise ValueError('Native foreground MSE must be finite and nonnegative')
    return -10*math.log10(max(sum(errors)/len(errors), 1e-30))


def compare(runs, thresholds=(20., 25., 30.), require_same_gpu=False):
    records = []
    controls = None
    for run in map(Path, runs):
        load = lambda name: json.loads((run/name).read_text())
        manifest, measured, quality = load('manifest.json'), load('measurements.json'), load('experiments.converse_renderer_comparison.quality.json')
        if measured['status'] != 'MEASURED':
            raise ValueError('Requires a successfully completed measured run: '+str(run))
        if not manifest.get('inputs'):
            raise ValueError('Missing recorded input identities: '+str(run))
        args = manifest['arguments']
        shared = {k:args[k] for k in ('stage', 'iterations', 'resolution', 'seed', 'visibility', 'eval_every', 'profile')}
        shared.update(initial_model_sha256=manifest['initial_model_sha256'], config_sha256=manifest['config_sha256'],
                      gpu=manifest['gpu'], torch_version=manifest['torch_version'], cuda_version=manifest.get('cuda_version'),
                      kernel_count=manifest['kernel_count'],
                      sources_sha256=hashlib.sha256(json.dumps(manifest['sources'], sort_keys=True).encode()).hexdigest())
        shared.update(inputs=manifest['inputs'], precision=manifest.get('precision'),
                      evaluation_renderers=args.get('evaluation_renderers', 'all'),
                      support_cap=args.get('support_cap',128),
                      pair_chunk=args.get('pair_chunk',1024),
                      footprint_cutoff=args.get('footprint_cutoff',1e-5),
                      max_records=args.get('max_records',1048576),
                      profile_skip=args.get('profile_skip',4) if args['profile'] else None,
                      profile_steps=args.get('profile_steps',4) if args['profile'] else None)
        if 'compiler_runtime' in manifest:
            from hktex_splatting.utils.compiler_runtime import matched_policy
            shared['compiler_runtime'] = matched_policy(manifest['compiler_runtime'])
        if require_same_gpu:
            if not manifest.get('gpu_uuid') or manifest['gpu_uuid'] == 'unavailable':
                raise ValueError('A physical GPU UUID is required for a matched-device comparison')
            shared['gpu_uuid'] = manifest['gpu_uuid']
        if controls is None:
            controls = shared
        elif shared != controls:
            raise ValueError('Training controls/initial model differ: '+str(run))
        if not quality or quality[0]['step'] != 0 or quality[-1]['step'] != args['iterations']:
            raise ValueError('Quality history must include initialization and the final training step: '+str(run))
        history = []
        for point in quality:
            cameras = point['cameras']
            sample = dict(step=point['step'], optimization_seconds=point['optimization_seconds'], worker_seconds=point['worker_seconds'])
            for split, heldout in [('train', False), ('heldout', True)]:
                sample[split+'_native_psnr_db'] = native_split_psnr(cameras, heldout)
            history.append(sample)
        crossings = {}
        for target in thresholds:
            crossings[str(target)] = next((p for p in history if p['heldout_native_psnr_db'] >= target), None)
        records.append(dict(run=str(run.resolve()), mode=args['mode'],
                            implementation_label=backend_label(args['mode']),
                            coefficient_backend=args.get('coefficient_backend','eager'),
                            runtime_counts=args.get('runtime_counts',False),
                            reuse_camera_geometry=args.get('reuse_camera_geometry',False),
                            dispatch_backend=args.get('dispatch_backend','original'),
                            reuse_coefficient_topology=args.get('reuse_coefficient_topology',False),
                            gpu_dispatch=args.get('gpu_dispatch',False),
                            gpu_resident=args.get('gpu_resident',False),
                            resident_renderer=args.get('resident_renderer','baseline'),
                            support_backend=args.get('support_backend','eager'), measurements=measured,
                            sampled_quality=history, first_measured_heldout_crossing=crossings))
    return dict(controls=controls, runs=records,
        primary_metric='First measured held-out native-field PSNR crossing, equal camera weighting; worker_seconds includes setup/evaluation overhead',
        caveat='A missing crossing is unreached within the recorded budget. These image-space fixed-density controls do not establish speedup over the full original UV trainer.')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('workflow','report'):
        if argv[0]=='workflow':
            from splatting_script.benchmarks.pipeline_workflow import main as run
        else:
            from splatting_script.benchmarks.pipeline_report import main as run
        return run(argv[1:])
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='*', type=Path)
    p.add_argument('--psnr', nargs='+', type=float, default=[20., 25., 30.])
    p.add_argument('--output', type=Path)
    p.add_argument('--require-same-gpu', action='store_true')
    p.add_argument('--require-iterations', type=int, help='Research stage gate, e.g. 50 or 1000')
    p.add_argument('--plot-dir', type=Path, help='New directory for research PSNR curves in PNG/PDF')
    p.add_argument('--scale-plan', action='store_true', help='CPU-only geometry/capacity review; never probes/allocates a GPU')
    p.add_argument('--resolutions', nargs='+', type=int, default=[64,128,256,512])
    p.add_argument('--kernel-counts', nargs='+', type=int, default=[1000,5000,10000])
    p.add_argument('--max-buffer-mib', type=int, default=512)
    p.add_argument('--max-contributions', type=int, default=16777216)
    args = p.parse_args(argv)
    if args.scale_plan:
        from splatting_script.benchmarks.training_research import scaling_plan
        if args.runs or min(args.resolutions+args.kernel_counts+[args.max_buffer_mib,args.max_contributions]) < 1:
            p.error('Scale plan expects positive dimensions/budgets and no run paths')
        result = scaling_plan(args.resolutions,args.kernel_counts,args.max_buffer_mib,args.max_contributions)
    elif args.runs and all((r/'research_manifest.json').is_file() for r in args.runs):
        from splatting_script.benchmarks.training_research import compare_runs, plot_comparison
        result = compare_runs(args.runs, args.psnr, args.require_same_gpu, args.require_iterations)
        if args.plot_dir:
            plot_comparison(result, args.plot_dir)
    else:
        if not args.runs or args.plot_dir or args.require_iterations:
            p.error('Supply paired research runs for plotting/stage gates, or existing controlled runs')
        result = compare(args.runs, args.psnr, args.require_same_gpu)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
