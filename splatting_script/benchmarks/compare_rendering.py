"""Compare original Mitsuba rendering and fresh HKS rasterization as different workloads.

Preview is CPU-only. Explicit execution uses isolated workers, one shared trained
checkpoint and camera sequence. No optimizer step or GPU allocation is requested.
The original lit path renderer and unlit HKS outputs are intentionally not treated
as equivalent images or a validated algorithmic speedup.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from hktex_splatting.utils.experiment_io import ROOT, configure_cache, gpu_guard
from splatting_script.experiments.train_surface import sha256, sources


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def summary(workflow):
    lines=['# Original Mitsuba rendering versus HKS rasterization','',
        '**Different workloads; no equivalent-rendering speedup claim.** Original: lit principled material, '
        'configured path integrator/ground plane/sampling and requested denoising. HKS: unlit object RGB, '
        'black background, one center sample per pixel, fresh triangle rasterization each frame.','',
        'Both arms load the same checkpoint and use the same recorded object cameras and resolution. '
        'No PSNR between these differently shaded images is reported. The known HKS/eager RGB '
        'discrepancy at 128 resolution is unresolved; this benchmark does not replace that gate.','',
        '| Method | Status | First frame s | Median full frame s | FPS from median | Peak Torch allocated GiB |',
        '|---|---|---:|---:|---:|---:|']
    for job in workflow['jobs']:
        path=Path(job['output'])/'result.json'
        r=json.loads(path.read_text()) if path.is_file() else {}
        lines.append('| '+' | '.join(str(v) for v in (job['backend'],job.get('status','preview'),
            r.get('first_frame_seconds','—'),r.get('median_frame_seconds','—'),r.get('fps','—'),
            r.get('torch_peak_allocated_gib','—')))+' |')
        failure=Path(job['output'])/'failure.json'
        if failure.is_file():lines+=['','Failure: `'+str(failure)+'`','```text',json.loads(failure.read_text())['traceback'],'```']
    lines+=['','Full frame wall timing includes fresh geometry/scene preparation, rendering and GPU completion. '
        'Images are saved outside frame timing. Native follows the original renderer cache flush/reset lifecycle. '
        'Warmup frames are excluded from medians; first-frame time is separate. Worker setup/checkpoint loading '
        'and full process times remain in receipts. Torch memory excludes Mitsuba/Dr.Jit/OptiX allocations; '
        'it is not a complete renderer VRAM comparison. Frames cycle recorded views; these are not per-view medians.','']
    return '\n'.join(lines)


def check_camera(sensor,camera,mi,np):
    """Check native sensor rays against the recorded raster camera before timing."""
    from hktex_splatting.rendering.camera_raster import perspective_camera_basis
    origin,basis=perspective_camera_basis(camera['position'],camera['target'])
    ty=np.tan(np.deg2rad(camera['fov_y_degrees'])/2)
    maximum=0.
    for u,v in ((.5,.5),(.25,.5),(.5,.25)):
        ray,_=sensor.sample_ray(0.,.5,mi.Point2f(u,v),mi.Point2f(.5,.5))
        actual=np.asarray(ray.d).reshape(3,-1)[:,0]
        expected=basis@np.array([(2*u-1)*ty*camera['width']/camera['height'],(1-2*v)*ty,1.])
        expected/=np.linalg.norm(expected)
        error=float(np.abs(actual-expected).max());maximum=max(maximum,error)
        if error>2e-5:raise RuntimeError('Mitsuba/raster camera ray mismatch: '+str(error))
    return dict(passed=True,max_direction_error=maximum)


def worker(args):
    out=args.run_dir;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();before=sources()
    try:
        gpu_guard();cache=configure_cache('render_'+out.name+'_'+str(os.getpid()))
        import torch
        import numpy as np
        import mitsuba as mi
        import drjit as dr
        mi.set_variant('cuda_ad_rgb')
        import hktex
        from hktex.utils import seed_everything
        from omegaconf import OmegaConf
        from hktex_splatting.training.training_checks import tree_hash
        from hktex_splatting.rendering.cameras import cameras
        from hktex_splatting.utils.reference_assets import save_npy,save_png
        checkpoint_hash=sha256(args.checkpoint)
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if checkpoint.get('format')!='hktex_full_surface_checkpoint_v1':raise ValueError('Expected a full-training final.pt checkpoint')
        parent=args.checkpoint.parent
        metadata={name:sha256(parent/name) for name in ('resolved_config.json','manifest.json')}
        cfg=OmegaConf.create(json.loads((parent/'resolved_config.json').read_text()))
        original=json.loads((parent/'manifest.json').read_text())
        cfg.exp_root_dir=str(cache/'loader');cfg.use_timestamp=False
        cfg.data.use_importance_sampling=False;cfg.data.num_workers=0
        cfg.trainer.model.n_sources=checkpoint['kernel_count']
        if not cfg.trainer.use_knn_implementation or not cfg.trainer.eigen_albo.use_precomputed or not cfg.trainer.eigen_albo.error_if_not_precomputed:
            raise ValueError('Requires original KNN HKTex and existing spectral assets')
        mesh=Path(cfg.data.mesh_path).resolve()
        spectral=Path(str(Path(cfg.trainer.eigen_albo.mesh_path).resolve()).rsplit('.',1)[0]+'_'+cfg.trainer.eigen_albo.get('precomputed_name','eigen_albo')+'_'+cfg.trainer.eigen_albo.local_frames+'.pt')
        if sha256(mesh)!=original['controls']['mesh_sha256'] or sha256(spectral)!=original['controls']['spectral_sha256']:
            raise ValueError('Checkpoint mesh/spectral assets differ')
        seed_everything(args.seed)
        torch.set_float32_matmul_precision('high')
        data=hktex.find(cfg.data_type)(cfg.data);data.prepare_data();data.setup('fit')
        trainer=hktex.find(cfg.trainer_type)(cfg.trainer,data,renderer_cfg=cfg.renderer)
        if (not np.array_equal(data.mesh.faces,trainer.mesh.faces.detach().cpu().numpy()) or
                not np.allclose(data.mesh.vertices,trainer.mesh.verts.detach().cpu().numpy(),rtol=1e-5,atol=1e-6)):
            raise ValueError('Original/HKS object geometry alignment differs')
        trainer.model.load_state_dict(checkpoint['model'],strict=True);trainer.model.eval()
        identity=tree_hash(trainer.model.state_dict())
        if identity!=tree_hash(checkpoint['model']):raise ValueError('Checkpoint did not load exactly')
        views=cameras()
        selected=[dict(views[v],width=args.resolution,height=args.resolution,view=v) for v in args.views]
        for c in selected:c['near']=args.near
        camera_checks={}
        if args.backend=='native':
            from hktex.rendering.heat_kernels_renderer_knn import HeatKernelsRendererKNN
            rcfg=OmegaConf.to_container(cfg.renderer,resolve=True)
            if rcfg['camera_config']['camera_type']!='perspective':raise ValueError('Matched camera protocol requires perspective projection')
            rcfg['camera_config'].update(img_width=args.resolution,img_height=args.resolution,near_clip=args.near)
            if args.samples_per_pixel is not None:rcfg['camera_config']['sample_count']=args.samples_per_pixel
            def camera_dict(renderer,c):
                camera=renderer.set_centre_looking_camera(renderer.cfg.camera_config,to_world=mi.ScalarTransform4f().look_at(origin=c['position'],target=c['target'],up=c['up']))
                camera.update(fov=c['fov_y_degrees'],fov_axis='y')
                return camera
            probe=HeatKernelsRendererKNN(rcfg)
            for c in selected:camera_checks[c['view']]=check_camera(mi.load_dict(camera_dict(probe,c)),c,mi,np)
            def draw(c):
                renderer=HeatKernelsRendererKNN(rcfg)
                renderer.mega_kernel(cfg.trainer.get('renderer_mega_kernel',False),no_loops=True,no_opt_calls=True)
                renderer._camera_dict=camera_dict(renderer,c)
                trainer.prepare_knn(False)
                try:
                    model_mesh=renderer.mesh_to_mitsuba(data.mesh,trainer.mesh,trainer.model,trainer.eigalbo_interp)
                    scene=renderer.make_scene(model_mesh,with_params=False)
                    result=renderer.render(scene=scene,denoise=args.denoise)
                    dr.eval(result);image=result.torch();torch.cuda.synchronize()
                finally:
                    renderer.flush_cache();trainer.reset_knn()
                return image
            settings=dict(renderer=rcfg,denoise=args.denoise,material='original principled HKTex texture',
                original_calls='HeatKernelsRendererKNN.mesh_to_mitsuba -> make_scene -> render; prepare_knn/reset_knn; flush_cache')
        else:
            from hktex_splatting.rendering.cuda_mesh_raster import CudaMeshRaster
            from hktex_splatting.training.surface_adapter import surface_forward
            raster=CudaMeshRaster(data.mesh.vertices,data.mesh.faces,near=args.near)
            trainer.surface_options=dict(surface_adapter='direct',selector_merge='row-radix',support_pruning='none',
                support_cap=128,max_contributions=args.max_contributions,max_buffer_mib=args.max_buffer_mib,density_policy='original')
            def draw(c):
                ids,bary,_=raster.buffers(c)
                pixels=torch.where(ids.flatten()>=0)[0]
                if not len(pixels):raise ValueError('Camera has no visible object pixels')
                sample=dict(face_id=ids.flatten()[pixels],bary=bary.reshape(-1,3)[pixels],
                    _camera_ids=ids,_camera_bary=bary,_camera_device=True)
                rgb=surface_forward(trainer,sample)
                image=rgb.new_zeros((ids.numel(),3));image[pixels]=rgb
                return image.reshape(args.resolution,args.resolution,3)
            settings=dict(raster='nvdiffrast fresh each frame',shading='unlit RGB',samples_per_pixel=1,
                background='black',denoise=False,ground_plane=False,raster_validation=raster.validation)
        torch.cuda.synchronize();setup=time.perf_counter()-started
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        timings=[];first=None
        with torch.no_grad():
            for i in range(1+args.warmup+args.frames):
                c=selected[i%len(selected)]
                torch.cuda.synchronize();dr.sync_thread();begin=time.perf_counter()
                image=draw(c);dr.sync_thread();torch.cuda.synchronize()
                elapsed=time.perf_counter()-begin
                if i==0:first=elapsed
                elif i>args.warmup:timings.append(elapsed)
                if not bool(torch.isfinite(image).all()):raise RuntimeError('Nonfinite rendered RGB')
                if i<len(selected):
                    raw=image.detach().cpu().numpy()
                    save_npy(out/(c['view']+'_raw.npy'),raw);save_png(out/(c['view']+'_display.png'),raw)
        if any(sha256(parent/name)!=value for name,value in metadata.items()):raise RuntimeError('Checkpoint metadata changed while rendering')
        if checkpoint_hash!=sha256(args.checkpoint):raise RuntimeError('Checkpoint changed while rendering')
        if identity!=tree_hash(trainer.model.state_dict()):raise RuntimeError('Rendering changed model parameters')
        if before!=sources():raise RuntimeError('Sources changed during rendering')
        median=statistics.median(timings)
        write(out/'result.json',dict(status='COMPLETE_DIFFERENT_WORKLOADS',backend=args.backend,
            checkpoint_sha256=checkpoint_hash,checkpoint_metadata=metadata,model_sha256=identity,cameras=selected,camera_checks=camera_checks,
            settings=settings,sources=before,checkpoint_training_controls=original['controls'],
            first_frame_seconds=first,frame_seconds=timings,median_frame_seconds=median,fps=1/median,
            torch_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            torch_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            setup_seconds=setup,worker_seconds=time.perf_counter()-started,
            torch_version=torch.__version__,cuda=torch.version.cuda,gpu=str(torch.cuda.get_device_properties(0)),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
            seed=args.seed,frames=args.frames,warmup=args.warmup,model_unchanged=True,equivalent_workloads=False))
        return 0
    except Exception:
        write(out/'failure.json',dict(traceback=traceback.format_exc()));raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--resolution',type=int,default=256)
    p.add_argument('--views',nargs='+',default=['top','side','three_quarter','logo_closeup'])
    p.add_argument('--frames',type=int,default=12);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--samples-per-pixel',type=int,help='Native only; defaults to original saved renderer config')
    p.add_argument('--denoise',action=argparse.BooleanOptionalAction,default=True,help='Original renderer OptiX denoising; HKS remains unlit without denoising')
    p.add_argument('--near',type=float,default=.01)
    p.add_argument('--max-buffer-mib',type=int,default=512);p.add_argument('--max-contributions',type=int,default=16777216)
    p.add_argument('--output',type=Path,default=ROOT/'experiments'/('render_comparison_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--backend',choices=('native','hybrid'),help=argparse.SUPPRESS);p.add_argument('--run-dir',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv);a.checkpoint=a.checkpoint.resolve()
    if (not a.checkpoint.is_file() or any(not (a.checkpoint.parent/n).is_file() for n in ('manifest.json','resolved_config.json'))):p.error('Keep final.pt with its manifest.json and resolved_config.json')
    if a.resolution<8 or a.resolution%8 or a.frames<1 or a.warmup<0 or not 0<a.near<float('inf') or min(a.max_buffer_mib,a.max_contributions)<1:p.error('Invalid resolution/count/near/budget')
    if a.samples_per_pixel is not None and a.samples_per_pixel<1:p.error('Sample count must be positive')
    if not a.views or len(set(a.views))!=len(a.views) or any(v not in ('top','side','three_quarter','logo_closeup','heldout_left','heldout_right') for v in a.views):p.error('Invalid/repeated camera view')
    if a.worker:
        if not a.execute or a.backend is None or a.run_dir is None:p.error('Invalid internal worker')
        return worker(a)
    jobs=[]
    for backend in ('native','hybrid'):
        out=a.output.resolve()/backend
        command=[sys.executable,'-B','-m','splatting_script.benchmarks.compare_rendering','--execute','--worker',
            '--backend',backend,'--run-dir',str(out),'--checkpoint',str(a.checkpoint),'--resolution',str(a.resolution),
            '--frames',str(a.frames),'--warmup',str(a.warmup),'--seed',str(a.seed),'--near',str(a.near),
            '--max-buffer-mib',str(a.max_buffer_mib),'--max-contributions',str(a.max_contributions),
            '--denoise' if a.denoise else '--no-denoise','--views',*a.views]
        if a.samples_per_pixel is not None:command+=['--samples-per-pixel',str(a.samples_per_pixel)]
        jobs.append(dict(backend=backend,output=str(out),command=command))
    workflow=dict(jobs=jobs,gpu_started=False,equivalent_workloads=False,checkpoint=str(a.checkpoint))
    if not a.execute:print(json.dumps(workflow,indent=2));return 0
    gpu_guard();a.output.mkdir(parents=True,exist_ok=False)
    started_sources=sources();failed=False;identity=None
    for job in jobs:
        start=time.perf_counter()
        with (a.output/(job['backend']+'.log')).open('x') as log:r=subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        job.update(status='complete' if r.returncode==0 else 'failed',process_seconds=time.perf_counter()-start)
        if (Path(job['output'])/'result.json').is_file():
            receipt=json.loads((Path(job['output'])/'result.json').read_text())
            current=(receipt['checkpoint_sha256'],receipt['model_sha256'],receipt['cameras'],receipt['checkpoint_metadata'],receipt['gpu_uuid'])
            if identity is not None and current!=identity:
                job['status']='failed_shared_identity';failed=True
            identity=current
        if (Path(job['output'])/'result.json').is_file():workflow['gpu_started']=True
        elif not workflow['gpu_started']:workflow['gpu_started']=None
        (a.output/'comparison.md').write_text(summary(workflow));print(job['backend']+': '+job['status'],flush=True)
        if r.returncode:failed=True
        if failed:break
    workflow['sources_unchanged']=started_sources==sources()
    write(a.output/'workflow.json',workflow)
    print('RENDER_COMPARISON_REPORT='+str(a.output.resolve()/'comparison.md'))
    return int(failed or not workflow['sources_unchanged'])


if __name__=='__main__':raise SystemExit(main())
