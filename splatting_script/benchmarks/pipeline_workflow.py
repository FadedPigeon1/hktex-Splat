"""Benchmark HKTex (Original) versus HKTex Splat (Ours).

Benchmark/training workflow; previews are GPU-free and execution uses an allocation.
Main API: plan, execute, classify_failure, main."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import os
import tempfile
from functools import partial

from splatting_script.benchmarks.training_research import backend_label


def plan(args):
    resolutions=args.resolutions or ([64,128,256,512] if args.scaling else [256])
    kernels=args.kernel_counts or ([1000,5000,10000] if args.scaling else [5000])
    if min(resolutions+kernels+[args.iterations,args.profile_steps,args.pilot_iterations,args.max_buffer_mib,args.max_contributions]) < 1:
        raise ValueError('Dimensions, budgets and update counts must be positive')
    if args.profile_start < 0: raise ValueError('Profile start must be nonnegative')
    if args.initial_state and len(kernels)!=1:
        raise ValueError('An explicit initial state requires one kernel count')
    base=[sys.executable,'-B','-m','splatting_script.experiments.compare_backends']
    common=['--execute','--seed',str(args.seed),'--support-selector','fused','--coefficient-backend','triton',
            '--splat-backend','triton','--runtime-counts','--support-cap','128',
            '--visibility',args.visibility,'--matmul-precision',args.matmul_precision,
            '--warmup','8','--max-buffer-mib',str(args.max_buffer_mib),'--max-contributions',str(args.max_contributions)]
    jobs=[]
    allow_variation=getattr(args,'allow_native_variation',False)
    if allow_variation: common+=['--allow-native-variation']
    for n in kernels:
        if not args.initial_state:
            jobs.append(dict(id=f'initial_{n}',role='initialization',kernels=n,resolution=resolutions[0],
                command=base+['--backend','native',*common,'--research','--initialization-only',
                              '--kernels',str(n),'--resolution',str(resolutions[0]),'--iterations','1']))
        for resolution in resolutions:
            commands={}
            for backend in ('native','hybrid'):
                prefix=f'{n}_{resolution}_{backend}'
                init=str(args.initial_state.resolve()) if args.initial_state else '{initial_'+str(n)+'}'
                cmd=base+['--backend',backend,*common,'--kernels',str(n),'--resolution',str(resolution),'--initial-state',init]
                commands[backend]=cmd
                jobs.append(dict(id=prefix+'_preflight',role='preflight',backend=backend,
                    implementation_label=backend_label(backend),kernels=n,resolution=resolution,
                    command=cmd+['--preflight-only','--iterations','1']))
            gate=f'{n}_{resolution}_paired_gate'
            jobs.append(dict(id=gate,role='paired_gate',kernels=n,resolution=resolution,command=[],
                requires=[f'{n}_{resolution}_{b}_preflight' for b in ('native','hybrid')]))
            for backend,cmd in commands.items():
                prefix=f'{n}_{resolution}_{backend}'
                jobs.append(dict(id=prefix+'_pilot',role='pilot',backend=backend,kernels=n,resolution=resolution,
                                 requires=gate,command=cmd+['--iterations',str(args.pilot_iterations)]))
                jobs.append(dict(id=prefix+'_speed',role='speed',backend=backend,kernels=n,resolution=resolution,
                    requires=prefix+'_pilot',command=cmd+['--research','--iterations',str(args.iterations),
                        '--progress-every','100','--eval-every','250','--checkpoint-every','1000','--check-every','100']))
                jobs.append(dict(id=prefix+'_profile',role='profile',backend=backend,kernels=n,resolution=resolution,
                    requires=prefix+'_pilot',command=cmd+['--iterations',str(args.profile_start+args.profile_steps),
                        '--pipeline-profile',*(['--profile-validate'] if not (allow_variation and backend=='native') else []),
                        '--profile-start',str(args.profile_start),'--profile-steps',str(args.profile_steps)]))
    for job in jobs:
        if 'backend' in job: job['implementation_label']=backend_label(job['backend'])
    return dict(format='hktex_pipeline_workflow_v1',gpu_started=False,settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        jobs=jobs,correctness_protocol='paired_preflight_v1',
        implementations={b:backend_label(b) for b in ('native','hybrid')},
        scope='Separate processes: shared initialization, per-backend preflights, paired alignment gate, memory/capacity pilots, uninstrumented research speed runs, then diagnostic profiles. '
                       'Runtime guards decide feasibility; no silent budget increases or contribution truncation.')


def classify_failure(text):
    lower=text.lower()
    if 'out of memory' in lower: return 'gpu_out_of_memory'
    if 'duplicate or unordered' in lower or 'invalid hybrid support' in lower: return 'invalid_support'
    if any(s in lower for s in ('exceed capacity','exceeds capacity','exceed configured','exceeds configured',
                               'exceed configured budget','exceeds configured budget','requires one bounded record batch',
                               'contributions exceed','scratch exceed','support exceeds')): return 'capacity_failure'
    if any(s in lower for s in ('not supported','unsupported','requires an existing','requires --','not audited')): return 'unsupported_configuration'
    return 'failed'


def run_process(command, log, cwd, *, artifact_root=None, cache_root=None):
    started=time.perf_counter()
    with log.open('x') as stream:
        environment=os.environ.copy()
        if artifact_root is not None: environment['HKTEX_RUN_ROOT']=str(artifact_root)
        if cache_root is not None: environment['HKTEX_CACHE_ROOT']=str(cache_root)
        result=subprocess.run(command,cwd=cwd,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    data=log.read_text(errors='replace')
    paths=re.findall(r'^DIRECT_TILES_RUN=(.+)$',data,re.MULTILINE)
    failures=re.findall(r'digeo failed for (\d+) sample',data)
    row=dict(returncode=result.returncode,process_wall_seconds=time.perf_counter()-started,
             run=paths[-1].strip() if paths else None,log=str(log),
             status='complete' if result.returncode==0 else classify_failure(data),
             geodesic_failed_sample_events=sum(map(int,failures)),geodesic_warning_calls=len(failures),
             geodesic_scope='Whole-process emitted warnings, including warmup; repeated failures are events, not unique kernels. Missing warnings do not certify tracing success.')
    if row['run']:
        path=Path(row['run'])/'failure.json'
        if path.exists(): row['failure']=json.loads(path.read_text())
    if result.returncode!=0: row['error_tail']=data[-6000:]
    return row


def paired_gate(jobs, directory, allow_native_variation=False):
    """Reject mismatched inputs; report RGB differences without forcing equality."""
    import numpy as np
    from splatting_script.benchmarks.pipeline_report import controls_match
    receipts,manifests,paths=[],[],[]
    for job in jobs:
        path=Path(job['run']);paths.append(path)
        receipts.append(json.loads((path/'preflight.json').read_text()))
        manifests.append(json.loads((path/'research_manifest.json').read_text()))
    a,b=receipts
    controls=controls_match(*manifests)
    checks=dict(preflights_passed=all(r.get('passed') is True for r in receipts),
        distinct_implementations={r.get('backend_id') for r in receipts}=={'native','hybrid'},
        matched_controls=controls['matched'],
        saved_initialization=bool(a.get('initialization_input')) and a.get('initialization_input')==b.get('initialization_input'),
        model_parameters=bool(a.get('initial_model_sha256')) and a.get('initial_model_sha256')==b.get('initial_model_sha256'),
        geometry_targets_masks=bool(a.get('geometry_identity')) and a.get('geometry_identity')==b.get('geometry_identity'))
    differences={}
    agreements=[]
    for name in a['geometry_identity']:
        own=[np.load(p/(name+'_initial_rgb.npy'),allow_pickle=False) for p in paths]
        native=[np.load(p/(name+'_initial_native_rgb.npy'),allow_pickle=False) for p in paths]
        checks['common_evaluator_'+name]=native[0].shape==native[1].shape and all(np.isfinite(v).all() for v in native) and bool(np.allclose(*native,atol=3e-5,rtol=3e-4))
        agreements.append('common_evaluator_'+name)
        checks['finite_aligned_'+name]=native[0].shape==native[1].shape and all(np.isfinite(v).all() for v in native)
        if own[0].shape!=own[1].shape or not all(np.isfinite(v).all() for v in own):
            raise ValueError('Misaligned or nonfinite paired RGB: '+name)
        delta=np.abs(own[0]-own[1])
        differences[name]=dict(full_image_max_abs=float(delta.max()),full_image_mean_abs=float(delta.mean()),
            initial_loss_by_backend={r['backend_id']:r['cameras'][name]['trained_backend']['loss'] for r in receipts})
    passed=all(checks.values())
    diagnostic_allowed=bool(not passed and allow_native_variation and
        all(r.get('passed') is True or (r.get('backend_id')=='native' and r.get('diagnostic_allowed') is True) for r in receipts)
        and all(value for key,value in checks.items() if key not in ['preflights_passed',*agreements]))
    directory.mkdir(exist_ok=False)
    (directory/'paired_preflight.json').write_text(json.dumps(dict(passed=passed,diagnostic_allowed=diagnostic_allowed,checks=checks,controls=controls,
        initial_image_differences=differences,preflight_runs=list(map(str,paths))),indent=2,allow_nan=False)+'\n')
    return dict(status='complete' if passed or diagnostic_allowed else 'failed_paired_preflight',run=str(directory),
        validation_passed=passed,diagnostic_allowed=diagnostic_allowed,checks=checks,controls=controls)


def execute(workflow, directory, runner=run_process, on_update=None):
    from hktex_splatting.utils.experiment_io import ROOT
    directory.mkdir(parents=True,exist_ok=False)
    index=directory/'workflow.json'
    def save():
        index.write_text(json.dumps(workflow,indent=2,allow_nan=False)+'\n')
        if on_update is not None: on_update(index)
    workflow['gpu_started']=True
    save()
    initializations,completed={},{}
    for job in workflow['jobs']:
        n=job['kernels']
        if '{initial_'+str(n)+'}' in job['command'] and n not in initializations:
            job['status']='skipped_initialization_failed';save();continue
        prerequisites=job.get('requires',[])
        if isinstance(prerequisites,str): prerequisites=[prerequisites]
        if any(completed.get(key)!='complete' for key in prerequisites):
            job['status']='skipped_failed_prerequisite';save();continue
        command=[initializations.get(n,word) if word=='{initial_'+str(n)+'}' else word for word in job['command']]
        job['executed_command']=command
        try:
            if job['role']=='paired_gate':
                parents=[next(j for j in workflow['jobs'] if j['id']==key) for key in prerequisites]
                job.update(paired_gate(parents,directory/job['id'],workflow.get('settings',{}).get('allow_native_variation',False)))
            else:
                job.update(runner(command,directory/(job['id']+'.log'),ROOT))
                if job['role']=='preflight' and job['status']=='complete':
                    path=Path(job['run'])/'preflight.json' if job.get('run') else None
                    receipt=json.loads(path.read_text()) if path is not None and path.is_file() else {}
                    allowed=(workflow.get('settings',{}).get('allow_native_variation',False) and job.get('backend')=='native' and receipt.get('diagnostic_allowed') is True)
                    job['validation_passed']=receipt.get('passed') is True
                    job['diagnostic_allowed']=bool(allowed)
                    if receipt.get('passed') is not True and not allowed:
                        job['status']='failed_missing_preflight'
        except OSError as error:
            job.update(status='launcher_failure',error=repr(error))
        except (ValueError,KeyError) as error:
            job.update(status='invalid_verification_artifacts',error=repr(error))
        except BaseException:
            job.update(status='interrupted');save();raise
        if job['role']=='initialization' and job['status']=='complete':
            path=Path(job['run'])/'initial.pt' if job['run'] else None
            if path is not None and path.is_file(): initializations[n]=str(path.resolve())
            else: job['status']='failed_missing_initial_state'
        completed[job['id']]=job['status']
        save()
        print(job['id']+': '+job['status'],flush=True)
    return index


def run_compact(workflow, output, executor=execute):
    """Keep one Markdown report; isolate all newly generated artifacts/caches."""
    from splatting_script.benchmarks.pipeline_report import collect, compact_markdown
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        stream.write('# HKTex benchmark\n\nStarting; no measurements yet.\n')
    with tempfile.TemporaryDirectory(prefix='hktex-benchmark-',dir='/tmp') as scratch:
        scratch=Path(scratch)
        runner=partial(run_process,artifact_root=scratch/'runs',cache_root=scratch/'caches')
        def update(index):
            try:
                text=compact_markdown(collect(index),json.loads(index.read_text()))
            except Exception as error:
                # An interrupted child can leave a partial artifact. Preserve
                # the last report instead of masking the original failure.
                with output.open('a') as stream:
                    stream.write('\nReport update unavailable: '+type(error).__name__+': '+str(error)+'\n')
                return
            output.write_text(text)
        try:
            executor(workflow,scratch/'workflow',runner=runner,on_update=update)
        except BaseException as error:
            # Keep the latest completed statistics even on a normal interruption.
            with output.open('a') as stream:
                stream.write('\nStopped: '+type(error).__name__+': '+str(error)+'\n')
            raise
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--allow-native-variation',action='store_true',
                   help='Run diagnostic/unvalidated comparisons despite bounded native replay drift; input/finite/gradient checks stay mandatory')
    p.add_argument('--output',type=Path,
                   help='New .md report (bare filenames go under experiments/), or new report directory; explicit paths are respected')
    p.add_argument('--keep-artifacts',action='store_true',
                   help='Retain the full research logs, checkpoints, JSON/CSV, images, traces and plots instead of one Markdown file')
    p.add_argument('--scaling',action='store_true',help='All 4 resolutions x 3 kernel counts; dry plan unless --execute')
    p.add_argument('--resolutions',nargs='+',type=int,help='Image sizes; e.g. 128 or 256 (default)')
    p.add_argument('--kernel-counts',nargs='+',type=int,help='Counts; e.g. 1000 or 5000 (default, tiled GPU selector)')
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--pilot-iterations',type=int,default=4)
    p.add_argument('--profile-start',type=int,default=40)
    p.add_argument('--profile-steps',type=int,default=8)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--initial-state',type=Path)
    p.add_argument('--visibility',choices=('reuse','fresh'),default='reuse')
    p.add_argument('--matmul-precision',choices=('highest','high'),default='highest')
    p.add_argument('--max-buffer-mib',type=int,default=512)
    p.add_argument('--max-contributions',type=int,default=16777216)
    args=p.parse_args(argv)
    if args.output and not args.output.is_absolute() and args.output.parent==Path('.') and args.output.suffix.lower()=='.md':
        from hktex_splatting.utils.experiment_io import ROOT
        args.output=ROOT/'experiments'/args.output
    if args.keep_artifacts and args.output and args.output.suffix.lower()=='.md':
        p.error('--keep-artifacts requires a directory, not a .md filename')
    workflow=plan(args)
    if not args.execute:
        print(json.dumps(workflow,indent=2));return
    if args.output is None: p.error('--execute requires --output')
    if args.output.exists(): p.error('--output already exists; choose a new report path')
    from hktex_splatting.utils.experiment_io import gpu_guard
    gpu_guard()
    if args.keep_artifacts:
        index=execute(workflow,args.output.resolve())
        from splatting_script.benchmarks.pipeline_report import report
        report(index,args.output.resolve()/'report')
        output=args.output.resolve()/'report/report.md'
    else:
        output=args.output.resolve() if args.output.suffix.lower()=='.md' else args.output.resolve()/'report.md'
        run_compact(workflow,output)
    print('BENCHMARK_REPORT='+str(output),flush=True)


if __name__=='__main__': main()
