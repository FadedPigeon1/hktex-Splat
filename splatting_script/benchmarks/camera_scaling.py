"""Run paired adaptive camera training at matched resolutions and report scaling.

Preview uses only the standard library. Each executed resolution gets independent
native/HKS workers with shared initialization and identical raster/target policy.
Full-worker time includes setup and final evaluation; PSNR stays alongside time.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from hktex_splatting.utils.experiment_io import ROOT, gpu_guard


def markdown(rows):
    lines=['# Camera training resolution scaling','',
        'Fresh triangle rasterization each frame. Compare native and HKS within each resolution. '
        'Different resolutions are different workloads. Ratios do not establish equal quality or convergence. '
        'Diagnostic render allowances never establish a validated speedup.','',
        '| Resolution | Method | Status | Training seconds | Complete training worker seconds | Held-out PSNR | Final kernels | Peak allocated GiB |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        for method in ('native','hybrid'):
            result_path=Path(row['output'])/(method+'_train')/'result.json'
            result=json.loads(result_path.read_text()) if result_path.is_file() else {}
            workflow_path=Path(row['output'])/'workflow.json'
            workflow=json.loads(workflow_path.read_text()) if workflow_path.is_file() else {}
            job=next((j for j in workflow.get('jobs',[]) if j['id']==method+'_train'),{})
            # Field names follow the full-training worker's actual receipts.
            quality=result.get('heldout_surface_quality',{}).get('trained_backend',{})
            memory=result.get('memory',{})
            allocated=memory.get('allocated_bytes')
            lines.append('| '+ ' | '.join(str(x) for x in (row['resolution'],method,
                result.get('status',job.get('status',row['status'])),
                result.get('training_interval_seconds','—'),job.get('process_wall_seconds','—'),
                quality.get('psnr_db','—'),result.get('final_kernel_count','—'),
                round(allocated/1024**3,4) if allocated is not None else '—'))+' |')
    lines+=['','Detailed paired receipts and quality/verification evidence are retained in each resolution directory.','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--resolutions',type=int,nargs='+',default=[64,128,256])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args,extra=parser.parse_known_args(argv)
    # These belong to the sweep; reject overrides that would invalidate pairing.
    forbidden=('--output','--resolution','--backend','--worker','--run-dir','--initial-state','--supervision','--visibility','--execute','--verify-only')
    if any(arg.split('=')[0] in forbidden for arg in extra):parser.error('Sweep owns output, resolution, both backends, fresh camera policy and initialization')
    if len(set(args.resolutions))!=len(args.resolutions) or any(r<8 or r%8 for r in args.resolutions):
        parser.error('Use distinct positive resolutions divisible by 8')
    rows=[]
    for resolution in args.resolutions:
        output=args.output.resolve()/str(resolution)
        command=[sys.executable,'-B','-m','splatting_script.experiments.train_surface',
            '--backend','both','--supervision','camera','--visibility','fresh',
            '--resolution',str(resolution),'--output',str(output),*extra]
        if args.execute:command.append('--execute')
        # Validate all worker flags before any GPU job is launched.
        preview=subprocess.run([x for x in command if x!='--execute'],cwd=ROOT,capture_output=True,text=True)
        if preview.returncode:parser.error(preview.stderr.strip())
        planned=json.loads(preview.stdout)
        if (planned['supervision']!='camera' or planned['visibility']!='fresh' or planned['resolution']!=resolution or
                {j['id'] for j in planned['jobs']}!={'native_verify','hybrid_verify','native_train','hybrid_train'} or
                planned['initial_snapshot']!=str(output/'native_verify'/'initial.pt')):
            parser.error('Forwarded arguments changed the paired camera protocol')
        rows.append(dict(resolution=resolution,output=str(output),command=command,status='preview'))
    if not args.execute:print(json.dumps(rows,indent=2));return 0
    gpu_guard();args.output.mkdir(parents=True,exist_ok=False)
    failed=False
    for row in rows:
        result=subprocess.run(row['command'],cwd=ROOT)
        row['status']='complete' if result.returncode==0 else 'failed';failed|=result.returncode!=0
        (args.output/'comparison.md').write_text(markdown(rows))
        if failed:break
    (args.output/'workflow.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('CAMERA_SCALING_REPORT='+str(args.output.resolve()/'comparison.md'))
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
