"""Record matched training updates, quality, snapshots and source provenance.

Training/optimization bookkeeping and benchmark reproducibility.
Main API: run_training, provenance, aggregate_quality, compare_runs."""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time

TRAIN_VIEWS = ('top', 'side', 'three_quarter', 'logo_closeup')
HELDOUT_VIEWS = ('heldout_left', 'heldout_right')


def backend_id(value):
    """Resolve retained aliases without changing identities in historical files."""
    return {'original':'native', '2dgs':'hybrid', 'direct':'hybrid'}.get(value, value)


def backend_label(value):
    return {'native':'HKTex (Original)', 'hybrid':'HKTex Splat (Ours)'}.get(backend_id(value), value)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def model_hash(model):
    return state_hash(model.state_dict())


def capture_rng(torch, device):
    import numpy as np
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])


def restore_rng(torch, state):
    import numpy as np
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


@contextmanager
def preserve_rng(torch, device):
    state = capture_rng(torch, device)
    try:
        yield
    finally:
        restore_rng(torch, state)


def check_initialization(initial, args):
    if initial.get('format') == 'hktex_research_snapshot_v1':
        raise ValueError('Training snapshots are not initializations; resume is not implemented')
    if initial['kernel_count'] != args.kernels:
        raise ValueError('Initial state kernel count mismatch')
    seed = initial.get('initialization_seed')
    if seed is None and args.seed != 0:
        raise ValueError('Unlabelled historical initialization is allowed only for seed 0; create a matched seed initialization')
    if seed is not None and seed != args.seed:
        raise ValueError('Initialization seed differs from --seed')
    if initial.get('initial_model_sha256') is not None and initial['initial_model_sha256'] != state_hash(initial['model']):
        raise ValueError('Initialization tensor contents disagree with the saved model hash')


def provenance(args, root, initial_hash, cameras, trainer, torch):
    """Hash actual source/asset bytes outside warmup and measured steps."""
    from omegaconf import OmegaConf
    root = Path(root)
    def git(*words):
        return subprocess.check_output(['git', '-C', str(root), *words], text=True).strip()
    sources = {}
    for directory in ('hktex', 'hktex_splatting', 'splatting_script', 'scripts'):
        for path in sorted((root/directory).rglob('*.py')):
            sources[str(path.relative_to(root))] = file_identity(path)['sha256']
    for name in ('interactive_density_knn.py', 'optimisation.py'):
        sources[name] = file_identity(root/name)['sha256']
    assets = {name:file_identity(root/path) for name, path in dict(
        mesh='data/converse/prepared/converse__free.glb',
        spectral='data/converse/prepared/converse__free_eigen_albo_principal_curvatures.pt',
        cameras='experiments/converse_fidelity/cameras.json').items()}
    props = torch.cuda.get_device_properties(trainer.mesh.verts.device)
    free, total = torch.cuda.mem_get_info(trainer.mesh.verts.device)
    return dict(format='hktex_research_manifest_v1', git_commit=git('rev-parse', 'HEAD'),
        dirty_tree_status=git('status', '--porcelain', '--untracked-files=all').splitlines(),
        sources=sources, source_tree_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        assets=assets, config=file_identity(args.config), initial_model_sha256=initial_hash,
        initialization_input=file_identity(args.initial_state) if args.initial_state else None,
        seed=args.seed, kernel_count=args.kernels, resolution=args.resolution,
        cameras=cameras, training_views=list(TRAIN_VIEWS), heldout_views=list(HELDOUT_VIEWS),
        heldout_scope='Unseen synthetic camera views of the same textured mesh, not an independent object/image dataset',
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        backend='original' if args.mode == 'native' else '2dgs',
        backend_id=backend_id(args.mode), implementation_label=backend_label(args.mode),
        validation_status='diagnostic_unvalidated' if getattr(args,'allow_native_variation',False) else 'strict',
        selector='native_faiss' if args.mode == 'native' else args.support_selector,
        selection_policy=('pointwise_faiss_outer50_top30' if args.mode == 'native'
                                                       else 'triangle_cap128_kernel_tile_top30'),
        optimizers=OmegaConf.to_container(trainer.cfg.optimizers, resolve=True),
        model_config=OmegaConf.to_container(trainer.cfg.model, resolve=True),
        parameters={n:dict(shape=list(p.shape), dtype=str(p.dtype), requires_grad=p.requires_grad)
                    for n,p in trainer.model.named_parameters()},
        precision=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                       cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_allow_tf32=torch.backends.cudnn.allow_tf32),
        software=dict(torch=torch.__version__, torch_git=torch.version.git_version,
                      cuda=torch.version.cuda, triton=__import__('triton').__version__),
        hardware=dict(gpu=props.name, gpu_uuid=str(getattr(props, 'uuid', 'unavailable')),
                      total_bytes=total, free_bytes_after_setup=free),
        scope='Controlled fixed-density unlit encoded-RGB training; not full Mitsuba inverse rendering',
        checkpoint_policy='Archival snapshots only; no resume CLI. Do not pass a training snapshot as --initial-state')


def aggregate_quality(cameras, family, split):
    errors = [v[family]['mse'] for v in cameras.values() if v['split'] == split]
    if not errors:
        return None
    mse = statistics.mean(errors)
    return dict(mse=mse, psnr_db=-10*math.log10(max(mse, 1e-30)), camera_weighting='equal')


def first_crossings(history, thresholds, family='native_field', split='heldout'):
    result = {}
    for threshold in thresholds:
        point = next((p for p in history if p['aggregate'][family][split] is not None and
                      p['aggregate'][family][split]['psnr_db'] >= threshold), None)
        result[str(threshold)] = (dict(step=point['step'], training_seconds=point['training_seconds'],
            experiment_seconds=point['experiment_seconds'], sampled_psnr_db=point['aggregate'][family][split]['psnr_db'])
            if point is not None else None)
    return result


def parameter_health(torch, model, initial, require_gradients=True, mesh=None):
    values = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        values[name] = dict(finite=bool(torch.isfinite(parameter).all()),
            minimum=float(parameter.detach().min()), maximum=float(parameter.detach().max()),
            changed=not torch.equal(parameter.detach(), initial[name]),
            gradient_present=grad is not None,
            gradient_finite=bool(torch.isfinite(grad).all()) if grad is not None else None,
            gradient_nonzero=bool((grad != 0).any()) if grad is not None else None)
    cfg = getattr(model, 'cfg', None)
    if getattr(cfg, 'range_enforcement_type', None) == 'pgd':
        bounds = dict(_angles=(0.,float(model._angle_scale)), _anisotropies=(1.,200.),
            _thresholds=(.3,1.), _sharpnesses=(10.,200.), _mean_colour=(0.,1.),
            _kernel_colours=(-1. if cfg.allow_negative_colours else 0.,1.))
        for name, (low, high) in bounds.items():
            if name in values:
                v = values[name]
                v['within_configured_bounds'] = v['minimum'] >= low-1e-6 and v['maximum'] <= high+1e-6
    valid = all(v['finite'] and v.get('within_configured_bounds', True) and (not require_gradients or
                (v['gradient_present'] and v['gradient_finite'])) for v in values.values())
    face_valid = None
    if mesh is not None and hasattr(mesh, 'faces') and hasattr(model, 'kernel_face_ids'):
        ids = model.kernel_face_ids
        face_valid = bool(((ids >= 0) & (ids < len(mesh.faces))).all())
        valid = valid and face_valid
    return dict(passed=valid, any_parameter_changed=any(v['changed'] for v in values.values()),
                kernel_face_ids_valid=face_valid, parameters=values)


def geometry_identity(geometry):
    """Fingerprint actual raster samples, RGB targets and masks outside timers."""
    import numpy as np
    result = {}
    for name, values in geometry.items():
        ids, bary, target, mask = values
        target, mask = target.detach().cpu().numpy(), mask.detach().cpu().numpy()
        if target.shape != (*ids.shape, 3) or mask.shape != ids.shape or not np.array_equal(mask, ids >= 0):
            raise ValueError('Target/raster alignment differs: '+name)
        if not mask.any() or not np.isfinite(target).all() or not np.isfinite(bary[mask]).all():
            raise ValueError('Empty foreground or nonfinite geometry/target: '+name)
        if not np.allclose(bary[mask].sum(-1), 1., atol=1e-5):
            raise ValueError('Invalid barycentric samples: '+name)
        row = {}
        for label, value in zip(('face_ids','barycentric','target_rgb','mask'), (ids,bary,target,mask)):
            array = np.ascontiguousarray(value)
            row[label] = dict(shape=list(array.shape), dtype=str(array.dtype),
                              sha256=hashlib.sha256(array.tobytes()).hexdigest())
        result[name] = row
    return result


def compare_replay_tensors(torch, first, replay, atol=3e-5, rtol=3e-4):
    """Name replay differences using the existing elementwise tolerance test."""
    rows=[]
    for name in sorted(first.keys() | replay.keys()):
        a,b=first.get(name),replay.get(name)
        row=dict(name=name,passed=False)
        if a is None or b is None:
            row['reason']='missing_tensor'
        elif a.shape!=b.shape or a.dtype!=b.dtype:
            row.update(reason='shape_or_dtype',first_shape=list(a.shape),replay_shape=list(b.shape),
                       first_dtype=str(a.dtype),replay_dtype=str(b.dtype))
        else:
            close=torch.isclose(a,b,atol=atol,rtol=rtol)
            delta=(a.double()-b.double()).abs()
            maximum=float(delta.max()) if delta.numel() else 0.
            index=int(delta.argmax()) if delta.numel() and bool(torch.isfinite(delta).all()) else None
            def scalar(value):
                result=float(value)
                return result if math.isfinite(result) else None
            row.update(passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),shape=list(a.shape),dtype=str(a.dtype),
                elements=a.numel(),mismatched_elements=int((~close).sum()),
                max_abs=maximum if math.isfinite(maximum) else None,maximum_flat_index=index,
                first_at_maximum=scalar(a.flatten()[index]) if index is not None else None,
                replay_at_maximum=scalar(b.flatten()[index]) if index is not None else None)
        rows.append(row)
    return dict(passed=all(row['passed'] for row in rows),absolute_tolerance=atol,relative_tolerance=rtol,
                failed_names=[row['name'] for row in rows if not row['passed']],tensors=rows)


def run_preflight(args, out, torch, trainer, initial, render, geometry, reset, step, manifest, support_stats=None):
    """Gate real RGB/loss, finite VJPs and repeatable optimizer updates, untimed.

    This scalable smoke gate does not replace --validate-only's independent
    coefficient/support/backward oracle, or assert native/HKS equality.
    """
    import numpy as np
    from hktex_splatting.utils.experiment_io import write_json
    model, mesh = trainer.model, trainer.mesh
    reset()
    if model_hash(model) != manifest['initial_model_sha256']:
        raise RuntimeError('Preflight did not load the saved initialization exactly')
    if not parameter_health(torch, model, initial, require_gradients=False, mesh=mesh)['passed']:
        raise RuntimeError('Nonfinite or invalid initial parameters')
    rng = capture_rng(torch, mesh.verts.device)
    cameras = {}
    with preserve_rng(torch, mesh.verts.device), torch.no_grad():
        for name in TRAIN_VIEWS+HELDOUT_VIEWS:
            own = render(args.mode, name).detach().clone()
            selector = support_stats(name) if args.mode == 'direct' and support_stats else {}
            if not args.allow_selector_fallback and any(selector.get(k) for k in
                    ('selector_fallback','native_distance_fallback','native_topk_fallback')):
                raise RuntimeError('Selector fallback requires --allow-selector-fallback: '+str(selector))
            native = render('native', name) if args.mode != 'native' else own
            _, _, target, mask = geometry[name]
            entry = dict(selector=selector)
            for family, rgb in (('trained_backend',own), ('native_field',native)):
                if rgb.shape != target.shape or not bool(torch.isfinite(rgb).all()):
                    raise RuntimeError('Nonfinite or misaligned preflight RGB: '+name)
                mse = float((rgb[mask]-target[mask]).square().mean())
                loss = float((rgb[mask]-target[mask]).square().sum(-1).mean())
                if not math.isfinite(loss) or not math.isclose(loss, 3*mse, rel_tol=1e-5, abs_tol=1e-7):
                    raise RuntimeError('Invalid foreground RGB loss: '+name)
                entry[family] = dict(mse=mse, loss=loss, psnr_db=-10*math.log10(max(mse,1e-30)))
            delta = (own[mask]-native[mask]).abs()
            entry['vs_native_same_parameters'] = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()))
            cameras[name] = entry
            np.save(out/(name+'_initial_rgb.npy'), own.cpu().numpy())
            np.save(out/(name+'_initial_native_rgb.npy'), native.cpu().numpy())

    # Match the execution mode of step(), including compiled autograd forwards.
    # The six-camera no_grad renders above remain the evaluation diagnostics.
    reset(); restore_rng(torch, rng)
    with preserve_rng(torch, mesh.verts.device), torch.enable_grad():
        training_rgb = render(args.mode, TRAIN_VIEWS[0])
        _, _, target, mask = geometry[TRAIN_VIEWS[0]]
        training_loss = (training_rgb[mask]-target[mask]).square().sum(-1).mean()
        if training_rgb.shape != target.shape or not bool(torch.isfinite(training_rgb).all()) or not bool(torch.isfinite(training_loss)):
            raise RuntimeError('Nonfinite or misaligned gradient-enabled preflight reference')
        training_reference_loss = float(training_loss.detach())
        training_reference_rgb = training_rgb.detach().cpu().numpy().copy()
    del training_rgb, training_loss
    evaluation_loss = cameras[TRAIN_VIEWS[0]]['trained_backend']['loss']
    np.save(out/(TRAIN_VIEWS[0]+'_initial_training_rgb.npy'),training_reference_rgb)
    evaluation_rgb = np.load(out/(TRAIN_VIEWS[0]+'_initial_rgb.npy'),allow_pickle=False)
    foreground = mask.detach().cpu().numpy()
    evaluation_delta = np.abs(training_reference_rgb[foreground]-evaluation_rgb[foreground])
    diagnostics = dict(format='hktex_preflight_loss_v1',backend_id=backend_id(args.mode),
        camera=TRAIN_VIEWS[0],training_reference_grad_enabled=True,evaluation_grad_enabled=False,
        training_reference_loss=training_reference_loss,evaluation_loss=evaluation_loss,
        training_vs_evaluation=dict(loss_abs_difference=abs(training_reference_loss-evaluation_loss),
            foreground_rgb_max_abs=float(evaluation_delta.max()),foreground_rgb_mean_abs=float(evaluation_delta.mean()),
            scope='Recorded execution-mode difference; not the training-step loss consistency gate'),
        loss_tolerance=dict(atol=1e-7,rtol=1e-5),cameras=cameras)
    write_json(out/'preflight_diagnostics.json',diagnostics)

    def update_snapshot():
        loss = step(args.mode, TRAIN_VIEWS[0])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite preflight training loss')
        health = parameter_health(torch, model, initial, mesh=mesh)
        if not health['passed'] or not health['any_parameter_changed']:
            raise RuntimeError('Preflight gradients/parameters failed health checks')
        values = {'parameters/'+n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        values.update({'gradients/'+n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad})
        for i,opt in enumerate(trainer.optimizers):
            for n,p in model.named_parameters():
                for key,value in opt.state.get(p,{}).items():
                    if isinstance(value, torch.Tensor):
                        values[f'optimizer/{i}/{n}/{key}'] = value.detach().cpu().clone()
        if not all(bool(torch.isfinite(v).all()) for v in values.values()):
            raise RuntimeError('Nonfinite preflight VJP or optimizer state')
        return float(loss.detach()), health, values

    reset(); restore_rng(torch, rng)
    first_start=dict(model_sha256=model_hash(model),optimizer_state_empty=all(not o.state for o in trainer.optimizers),
                     learning_rates=[[float(group['lr']) for group in o.param_groups] for o in trainer.optimizers])
    first_loss, _, first = update_snapshot()
    loss_consistent = math.isclose(first_loss, training_reference_loss, rel_tol=1e-5, abs_tol=1e-7)
    diagnostics['training_step'] = dict(loss=first_loss,reference_loss=training_reference_loss,
        abs_difference=abs(first_loss-training_reference_loss),passed=loss_consistent)
    write_json(out/'preflight_diagnostics.json',diagnostics,update=True)
    reset(); restore_rng(torch, rng)
    replay_start=dict(model_sha256=model_hash(model),optimizer_state_empty=all(not o.state for o in trainer.optimizers),
                      learning_rates=[[float(group['lr']) for group in o.param_groups] for o in trainer.optimizers])
    replay_loss, replay_health, replay = update_snapshot()
    tensor_replay=compare_replay_tensors(torch,first,replay)
    replay_checks = dict(loss=math.isclose(first_loss,replay_loss,rel_tol=3e-4,abs_tol=3e-5),
        gradients_parameters_optimizer=tensor_replay['passed'])
    replay_report=dict(passed=all(replay_checks.values()),checks=replay_checks,
        first_loss=first_loss,replay_loss=replay_loss,loss_abs_difference=abs(first_loss-replay_loss),
        expected_initial_model_sha256=manifest['initial_model_sha256'],first_start=first_start,replay_start=replay_start,
        reset_restored_initial=(first_start==replay_start and first_start['model_sha256']==manifest['initial_model_sha256']
                                and first_start['optimizer_state_empty']),tensor_comparison=tensor_replay)
    write_json(out/'preflight_replay.json',replay_report)
    diagnostics['replay']=replay_report
    write_json(out/'preflight_diagnostics.json',diagnostics,update=True)
    variation = not loss_consistent or not all(replay_checks.values())
    diagnostic_allowed = bool(variation and args.mode=='native' and getattr(args,'allow_native_variation',False)
        and replay_report['reset_restored_initial'] and replay_checks['loss']
        and math.isclose(first_loss,training_reference_loss,rel_tol=3e-4,abs_tol=3e-5)
        and all(row['passed'] for row in tensor_replay['tensors'] if not row['name'].startswith('parameters/')))
    if not replay_report['reset_restored_initial']:
        raise RuntimeError('Preflight reset failed to restore the initial model/optimizer settings')
    if variation and not diagnostic_allowed:
        np.savez_compressed(out/'preflight_replay_tensors.npz',
            **{prefix+'/'+name:value.numpy() for prefix,snapshot in (('first',first),('replay',replay))
               for name,value in snapshot.items()})
        if not loss_consistent:
            raise RuntimeError('Actual training step disagrees with the gradient-enabled foreground loss reference: '
                f'step={first_loss:.10g}, reference={training_reference_loss:.10g}; '
                'replay failed tensors='+str(tensor_replay['failed_names'])+
                '; inspect preflight_diagnostics.json and preflight_replay.json')
        raise RuntimeError('Preflight optimizer/VJP replay differs after a complete reset; '
            'failed tensors='+str(tensor_replay['failed_names'])+'; inspect preflight_replay.json and preflight_replay_tensors.npz')
    health = [replay_health]
    for name in TRAIN_VIEWS[1:]:
        loss = step(args.mode,name)
        current = parameter_health(torch,model,initial,mesh=mesh)
        optimizer_finite = all(bool(torch.isfinite(v).all()) for opt in trainer.optimizers
            for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor))
        if not bool(torch.isfinite(loss)) or not current['passed'] or not optimizer_finite:
            raise RuntimeError('Preflight training failed on camera '+name)
        health.append(current)
    receipt = dict(format='hktex_preflight_v1', passed=not variation, diagnostic_allowed=diagnostic_allowed,
        validation_status='diagnostic_unvalidated' if diagnostic_allowed else 'passed',backend_id=backend_id(args.mode),
        implementation_label=backend_label(args.mode), initial_model_sha256=manifest['initial_model_sha256'],
        initialization_input=manifest['initialization_input'], geometry_identity=manifest['geometry_identity'],
        cameras=cameras,training_loss_check=diagnostics['training_step'],
        training_vs_evaluation=diagnostics['training_vs_evaluation'],
        replay_checks=replay_checks, update_health=health,
        scope='Untimed actual-renderer smoke and reset replay; independent --validate-only oracle remains separate. '
              'Different candidate policies may produce different initial RGB and loss.')
    write_json(out/'preflight.json',receipt)
    return receipt


def run_training(args, out, torch, trainer, initial, render, geometry, support_stats, jit,
                 manifest, process_start, setup):
    """Use the caller's unchanged training step and render functions."""
    import numpy as np
    from hktex_splatting.utils.experiment_io import write_json
    model, device = trainer.model, trainer.mesh.verts.device
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else lambda: None
    step = setup.pop('step_function')
    rows, history, health = [], [], []
    cumulative, evaluation_seconds, checkpoint_seconds, health_seconds, progress_io_seconds = 0., 0., 0., 0., 0.
    write_json(out/'research_manifest.json', manifest, update=True)
    log = (out/'iterations.jsonl').open('x')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    peaks = dict(allocated=0, reserved=0)

    def update_peak():
        if device.type == 'cuda':
            peaks['allocated'] = max(peaks['allocated'], torch.cuda.max_memory_allocated(device))
            peaks['reserved'] = max(peaks['reserved'], torch.cuda.max_memory_reserved(device))

    def evaluate(completed):
        nonlocal evaluation_seconds
        started = time.perf_counter()
        jit.set_phase('research_evaluation', completed)
        result = {}
        with preserve_rng(torch, device), torch.no_grad():
            for name in TRAIN_VIEWS+HELDOUT_VIEWS:
                rgb = render(args.mode, name)
                own = rgb.detach().clone()
                selector = support_stats(name) if args.mode == 'direct' else {}
                if args.mode == 'direct' and not args.allow_selector_fallback and any(
                        selector.get(k) for k in ('selector_fallback', 'native_distance_fallback', 'native_topk_fallback')):
                    raise RuntimeError('Selector fallback requires --allow-selector-fallback: '+str(selector))
                reference = render('native', name) if args.mode != 'native' else own
                _, _, target, mask = geometry[name]
                entry = dict(split='train' if name in TRAIN_VIEWS else 'heldout', selector=selector)
                for family, value in (('trained_backend', own), ('native_field', reference)):
                    if not bool(torch.isfinite(value).all()):
                        raise RuntimeError('Nonfinite evaluation RGB: '+name)
                    mse = float((value[mask]-target[mask]).square().mean())
                    entry[family] = dict(mse=mse, psnr_db=-10*math.log10(max(mse, 1e-30)))
                delta = (own[mask]-reference[mask]).abs()
                entry['vs_native_same_parameters'] = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()))
                result[name] = entry
                if completed == args.iterations:
                    np.save(out/(name+'_rgb.npy'), own.cpu().numpy())
                    np.save(out/(name+'_native_rgb.npy'), reference.cpu().numpy())
                    np.save(out/(name+'_mask.npy'), mask.cpu().numpy())
                    from PIL import Image
                    Image.fromarray(np.rint(own.cpu().numpy().clip(0, 1)*255).astype(np.uint8)).save(out/(name+'_rgb.png'))
        sync()
        point = dict(step=completed, training_seconds=cumulative,
            experiment_seconds=time.perf_counter()-process_start, cameras=result,
            aggregate={f:{s:aggregate_quality(result, f, s) for s in ('train','heldout')}
                       for f in ('trained_backend','native_field')})
        history.append(point)
        write_json(out/'quality.json', history, update=True)
        evaluation_seconds += time.perf_counter()-started
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(completed):
        nonlocal checkpoint_seconds
        start = time.perf_counter()
        state = dict(format='hktex_research_snapshot_v1', model=model.state_dict(),
            optimizers=[o.state_dict() for o in trainer.optimizers],
            schedulers=[s.state_dict() for s in trainer.schedulers], completed_updates=completed,
            next_camera=TRAIN_VIEWS[completed % len(TRAIN_VIEWS)], camera_schedule=list(TRAIN_VIEWS),
            rng=capture_rng(torch, device), kernel_count=args.kernels, seed=args.seed,
            initial_model_sha256=manifest['initial_model_sha256'], training_seconds=cumulative,
            model_sha256=model_hash(model), resume_supported=False)
        with (out/f'checkpoint_{completed:06d}.pt').open('xb') as stream:
            torch.save(state, stream)
        checkpoint_seconds += time.perf_counter()-start

    try:
        evaluate(0)
        for i in range(args.iterations):
            sync()
            jit.set_phase('measurement', i)
            before = jit.snapshot()
            cpu_start = time.process_time()
            start = time.perf_counter()
            loss = step(args.mode, TRAIN_VIEWS[i % len(TRAIN_VIEWS)])
            enqueue = time.perf_counter()-start
            sync()
            seconds = time.perf_counter()-start
            update_peak()
            cumulative += seconds
            after = jit.snapshot()
            row = dict(iteration=i, completed_updates=i+1, camera=TRAIN_VIEWS[i % len(TRAIN_VIEWS)],
                seconds=seconds, host_enqueue_seconds=enqueue, training_seconds=cumulative,
                process_cpu_seconds=time.process_time()-cpu_start,
                loss=float(loss.detach()), jit_cache_misses=after[0]-before[0],
                jit_cache_miss_wall_seconds=after[1]-before[1])
            if not math.isfinite(row['loss']):
                raise RuntimeError('Nonfinite training loss at update '+str(i+1))
            rows.append(row)
            io_start = time.perf_counter()
            log.write(json.dumps(row, allow_nan=False)+'\n')
            progress_io_seconds += time.perf_counter()-io_start
            if i == 0 or (i+1) % args.check_every == 0 or i+1 == args.iterations:
                start = time.perf_counter()
                check = parameter_health(torch, model, initial, mesh=trainer.mesh)
                check['step'] = i+1
                health.append(check)
                health_seconds += time.perf_counter()-start
                if not check['passed']:
                    raise RuntimeError('Nonfinite parameter/gradient or missing gradient: '+str(check))
            if (i+1) % args.eval_every == 0 or i+1 == args.iterations:
                evaluate(i+1)
            if (i+1) % args.checkpoint_every == 0 or i+1 == args.iterations:
                checkpoint(i+1)
            if (i+1) % args.progress_every == 0 or i+1 == args.iterations:
                start = time.perf_counter()
                log.flush()
                print('RESEARCH_PROGRESS='+json.dumps(dict(update=i+1, loss=row['loss'],
                    training_seconds=cumulative, last_quality_step=history[-1]['step'])), flush=True)
                progress_io_seconds += time.perf_counter()-start
        if not health[-1]['any_parameter_changed']:
            raise RuntimeError('No trainable parameters changed from initialization')
        write_json(out/'health.json', health)
        write_json(out/'iterations.json', rows)
        report = dict(format='hktex_research_report_v1', status='COMPLETE', completed_updates=len(rows),
            validation_status='diagnostic_unvalidated' if getattr(args,'allow_native_variation',False) else 'strict',
            initial_model_sha256=manifest['initial_model_sha256'], final_model_sha256=model_hash(model),
            steady_iteration_median_seconds=statistics.median(r['seconds'] for r in rows),
            steady_iteration_p95_seconds=sorted(r['seconds'] for r in rows)[math.ceil(.95*len(rows))-1],
            measured_training_seconds=cumulative, final_training_loss=rows[-1]['loss'],
            implementation_label=backend_label(args.mode), peak_allocated_bytes=peaks['allocated'],
            peak_reserved_bytes=peaks['reserved'], setup=setup,
            evaluation_seconds=evaluation_seconds, checkpoint_seconds=checkpoint_seconds,
            health_check_seconds=health_seconds, progress_io_seconds=progress_io_seconds,
            first_measured_psnr_crossings=first_crossings(history, args.psnr_thresholds),
            experiment_wall_before_final_report_seconds=time.perf_counter()-process_start,
            final_parameter_validity=health[-1], jit_cache=jit.report(),
            measurement_diagnostic=bool(getattr(args, 'debug_support', False)),
            timing_scope='Synchronized updates only; evaluation, health checks, checkpoint/progress I/O excluded. '
                         'Compilation encountered during updates remains included. Use external time for complete process wall.',
            quality_scope='Primary curves evaluate each learned model through the common native field; '
                          'trained_backend curves also report its deployed selection policy. No original-HKTex/HKS equivalence assumed.')
        write_json(out/'report.json', report)
        write_json(out/'completion.json', dict(status='COMPLETE',
            experiment_wall_seconds=time.perf_counter()-process_start,
            scope='Through final report writing; external /usr/bin/time additionally includes interpreter entry/exit'))
        return report
    except Exception as error:
        write_json(out/'failure.json', dict(status='FAILED', completed_updates=len(rows), error=repr(error)))
        raise
    finally:
        log.close()


def stage_activity(events, iterations):
    """Offline attribution to all enclosing named scopes, by launch correlation."""
    complete = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    threads, launches, external, stages = {}, {}, {}, {}
    unattributed, total = 0, 0
    for e in complete:
        if e.get('cat') in ('cpu_op','user_annotation','cuda_runtime','cuda_driver'):
            threads.setdefault((e.get('pid'), e.get('tid')), []).append(e)
    for values in threads.values():
        stack = []
        for e in sorted(values, key=lambda e:(e['ts'], -e['dur'])):
            while stack and stack[-1]['ts']+stack[-1]['dur'] <= e['ts']:
                stack.pop()
            if e['cat'] in ('cpu_op','user_annotation'):
                stack.append(e)
            labels = tuple(x['name'] for x in stack if x.get('cat') == 'user_annotation'
                           and x['name'].startswith(('training::','spectral::','support::','splat::')))
            args = e.get('args', {})
            if e['cat'] in ('cuda_runtime','cuda_driver') and 'correlation' in args:
                launches[args['correlation']] = labels
            if 'External id' in args:
                external[args['External id']] = labels
    for e in complete:
        if e.get('cat') != 'kernel':
            continue
        total += 1
        args = e.get('args', {})
        labels = set(launches.get(args.get('correlation'), external.get(args.get('External id'), ())))
        unattributed += not bool(labels)
        for label in labels:
            row = stages.setdefault(label, dict(kernel_launches=0, cuda_duration_sum_ms=0.))
            row['kernel_launches'] += 1
            row['cuda_duration_sum_ms'] += e['dur']/1000
    for row in stages.values():
        row['cuda_duration_sum_ms_per_iteration'] = row['cuda_duration_sum_ms']/max(1, iterations)
    return dict(profiled_iterations=iterations, stages=stages, total_kernel_launches=total,
        unattributed_kernel_launches=unattributed,
        scope='Launch-correlated CUPTI kernel duration sums. Nested scopes overlap; never sum parents and children. '
              'Includes HKS queue/emission/top30 and spectral forward/backward markers; diagnostic only.')


def compare_runs(runs, thresholds, require_same_gpu=False, require_iterations=None):
    """Compare paired seeds without interpolating PSNR or mixing timing regimes."""
    import numpy as np
    groups, initializations = {}, {}
    for run in map(Path, runs):
        manifest = json.loads((run/'research_manifest.json').read_text())
        report = json.loads((run/'report.json').read_text())
        history = json.loads((run/'quality.json').read_text())
        args = manifest['arguments']
        if args.get('allow_native_variation'):
            raise ValueError('Native-variation runs are diagnostic; use their workflow report, not a validated training comparison')
        if args.get('debug_support') or report.get('measurement_diagnostic'):
            raise ValueError('Diagnostic support checks cannot be used for timing comparisons: '+str(run))
        if (report.get('status') != 'COMPLETE' or not report['final_parameter_validity']['passed'] or
                not report['final_parameter_validity']['any_parameter_changed'] or
                report['completed_updates'] != args['iterations'] or
                history[0]['step'] != 0 or history[-1]['step'] != args['iterations']):
            raise ValueError('Incomplete/unhealthy research experiment: '+str(run))
        if require_iterations is not None and args['iterations'] != require_iterations:
            raise ValueError('Stage requires exactly '+str(require_iterations)+' updates')
        expected_quality = {0,args['iterations']} | set(range(args['eval_every'],args['iterations']+1,args['eval_every']))
        if [p['step'] for p in history] != sorted(expected_quality):
            raise ValueError('Missing or duplicated scheduled quality evaluations')
        for step in set(range(args['checkpoint_every'], args['iterations']+1, args['checkpoint_every'])) | {args['iterations']}:
            if not (run/f'checkpoint_{step:06d}.pt').is_file():
                raise ValueError('Missing scheduled snapshot: '+str(run))
        controls = {k:manifest[k] for k in ('initial_model_sha256','assets','config','source_tree_sha256',
                    'seed','kernel_count','resolution','cameras','training_views','heldout_views',
                    'optimizers','model_config','precision','software','parameters')}
        controls['settings'] = {k:args[k] for k in ('iterations','seed','warmup','visibility','eval_every',
            'checkpoint_every','progress_every','check_every','support_cap','footprint_cutoff',
            'coefficient_backend','runtime_counts','selection','splat_backend','max_contributions','max_buffer_mib')}
        controls['settings']['support_selector'] = args.get('support_selector','legacy')
        controls['gpu'] = manifest['hardware']['gpu']
        for name in ('initialization_input','geometry_identity'):
            if name in manifest: controls[name]=manifest[name]
        if require_same_gpu:
            uuid = manifest['hardware'].get('gpu_uuid', 'unavailable')
            if uuid == 'unavailable':
                raise ValueError('Missing GPU UUID')
            controls['gpu_uuid'] = uuid
        key = (manifest['kernel_count'], manifest['resolution'], manifest['seed'])
        group = groups.setdefault(key, dict(controls=controls, runs={}))
        if group['controls'] != controls:
            raise ValueError('Matched inputs/settings differ: '+str(run))
        backend = manifest['backend']
        if backend_id(backend) not in ('native','hybrid') or any(backend_id(b)==backend_id(backend) for b in group['runs']):
            raise ValueError('Supply exactly one original and one hybrid run per seed/workload')
        group['runs'][backend] = dict(run=str(run.resolve()), report=report, history=history,
            backend_id=backend_id(backend), implementation_label=backend_label(backend),
            hardware=manifest.get('hardware',{}),
            first_measured_psnr_crossings=first_crossings(history, thresholds))
        initializations.setdefault(key[0], {}).setdefault(manifest['initial_model_sha256'], set()).add(key[2])
    if any(len(seeds) > 1 for hashes in initializations.values() for seeds in hashes.values()):
        raise ValueError('Different seeds reused the same model initialization')
    output = []
    for key, group in sorted(groups.items()):
        by_id={backend_id(b):r for b,r in group['runs'].items()}
        if set(by_id) != {'native','hybrid'}:
            raise ValueError('Missing original/hybrid partner for '+str(key))
        original, hybrid = (Path(by_id[b]['run']) for b in ('native','hybrid'))
        a0,b0 = (by_id[b]['history'][0]['cameras'] for b in ('native','hybrid'))
        for name in TRAIN_VIEWS+HELDOUT_VIEWS:
            if not math.isclose(a0[name]['native_field']['mse'], b0[name]['native_field']['mse'],rel_tol=1e-4,abs_tol=1e-7):
                raise ValueError('Common evaluator disagrees at identical initial parameters')
        differences = {}
        for name in TRAIN_VIEWS+HELDOUT_VIEWS:
            a, b = (np.load(p/(name+'_rgb.npy'), allow_pickle=False) for p in (original,hybrid))
            ma, mb = (np.load(p/(name+'_mask.npy'), allow_pickle=False) for p in (original,hybrid))
            if a.shape != b.shape or not np.array_equal(ma, mb):
                raise ValueError('Final image shapes/visibility differ')
            delta = np.abs(a[ma]-b[mb])
            differences[name] = dict(mse=float(np.square(delta).mean()),
                max_abs=float(delta.max()), mean_abs=float(delta.mean()))
        from splatting_script.benchmarks.pipeline_report import paired_metrics
        metrics=[]
        for backend in ('native','hybrid'):
            run=by_id[backend]; summary=run['report']; final=run['history'][-1]['aggregate']
            native,deployed=final.get('native_field',{}),final.get('trained_backend',{})
            metrics.append(dict(backend=backend,implementation_label=backend_label(backend),
                completed_updates=summary['completed_updates'],total_training_seconds=summary['measured_training_seconds'],
                median_iteration_seconds=summary.get('steady_iteration_median_seconds'),
                p95_iteration_seconds=summary.get('steady_iteration_p95_seconds'),
                peak_allocated_bytes=summary.get('peak_allocated_bytes'),peak_reserved_bytes=summary.get('peak_reserved_bytes'),
                final_training_loss=summary.get('final_training_loss'),
                training_psnr_db=native.get('train',{}).get('psnr_db'),heldout_psnr_db=native.get('heldout',{}).get('psnr_db'),
                deployed_training_psnr_db=deployed.get('train',{}).get('psnr_db'),deployed_heldout_psnr_db=deployed.get('heldout',{}).get('psnr_db')))
        a,b=metrics
        uuids=[r['hardware'].get('gpu_uuid') for r in by_id.values()]
        matched_gpu=bool(uuids[0] not in (None,'','unavailable') and len(set(uuids))==1)
        differences_summary=paired_metrics(a,b)
        if not matched_gpu: differences_summary['gpu_memory_reduction_percent']=None
        output.append(dict(kernel_count=key[0], resolution=key[1], seed=key[2],
            controls=group['controls'], runs=group['runs'], final_image_differences=differences,
            performance=metrics,comparison=dict(**differences_summary,matched_physical_gpu=matched_gpu,
                timed_update_ratio_native_over_hybrid=a['total_training_seconds']/b['total_training_seconds']
                if matched_gpu and b['total_training_seconds']>0 else None)))
    return dict(groups=output, primary_metric='Common native-field held-out PSNR, equal camera weighting',
        crossing_scope='First actual sampled crossing only; null means unreached. No interpolation/extrapolation.',
        limitations='Controlled unlit, fixed-density same-object experiment. Pointwise HKTex outer50 and HKS triangle cap128 '
                    'selection differ; no exact-output, full Mitsuba, or unseen-object generalization claim.')


def plot_comparison(comparison, directory):
    """Export ordinary scientific plots; never invent unmeasured quality points."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for group in comparison['groups']:
        for family in ('native_field','trained_backend'):
            for axis, label in (('step','Optimizer updates'), ('training_seconds','Cumulative timed training (s)')):
                fig, ax = plt.subplots(figsize=(6.4,4.2))
                for backend, run in group['runs'].items():
                    for split, style in (('heldout','-'),('train','--')):
                        points = run['history']
                        ax.plot([p[axis] for p in points],
                            [p['aggregate'][family][split]['psnr_db'] for p in points],
                            style, marker='o', markersize=3, label=backend_label(backend)+' '+split)
                ax.set(xlabel=label, ylabel='PSNR (dB)', title=f'{family}: N={group["kernel_count"]}, {group["resolution"]}px, seed={group["seed"]}')
                ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
                stem=f'n{group["kernel_count"]}_r{group["resolution"]}_s{group["seed"]}_{family}_{axis}'
                fig.savefig(directory/(stem+'.png'), dpi=180)
                fig.savefig(directory/(stem+'.pdf'))
                plt.close(fig)


def scaling_plan(resolutions=(64,128,256,512), kernels=(1000,5000,10000),
                 max_buffer_mib=512, max_contributions=16777216):
    """CPU geometry and conservative budget estimates; no device probing."""
    import numpy as np
    from hktex_splatting.utils.experiment_io import ROOT, MESH
    from hktex_splatting.utils.reference_assets import load_cpu_mesh
    from hktex_splatting.rendering.mesh_visibility_raster import BVHRaster
    from hktex_splatting.rendering.cameras import cameras
    mesh = load_cpu_mesh(MESH)
    raster = BVHRaster(mesh.vertices, mesh.faces)
    base, cases = cameras(), []
    for resolution in resolutions:
        views = {}
        for name, camera in base.items():
            _, _, _, ids, _ = raster.all_buffers(dict(camera,width=resolution,height=resolution))
            y,x = np.nonzero(ids >= 0)
            faces = ids[y,x]
            tiles = (y//8)*math.ceil(resolution/8)+x//8
            views[name] = dict(foreground_pixels=len(faces), visible_faces=len(np.unique(faces)),
                face_tile_pairs=len(np.unique(np.stack((faces,tiles),1),axis=0)))
        for n in kernels:
            estimates = {}
            for name, view in views.items():
                p, v, patches = view['foreground_pixels'], view['visible_faces'], view['face_tile_pairs']
                pairs, tasks, fragments = 128*v, 128*patches, 128*p
                estimates[name] = dict(**view, maximum_pairs=pairs, maximum_tasks=tasks,
                    maximum_fragments=fragments,
                    hybrid_guard_bytes=24*fragments+112*tasks+128*pairs+512*p,
                    support_guard_bytes=52*pairs+40*tasks,
                    distance_scratch_bytes=8*min(128,v)*min(n,1024) if n > 4096 else 8*min(128,v)*n,
                    selector_candidate_state_bytes=12*min(128,v)*128 if n > 4096 else 0,
                    # Empirical projection from saved 64px peak: not a fit guarantee.
                    native_peak_projection_bytes=int(4237339648+max(0,p-3819)*(4237339648-2073838592)/3819))
            bounded = all(e['maximum_fragments'] <= max_contributions and
                          max(e['hybrid_guard_bytes'],e['support_guard_bytes']) <= max_buffer_mib*1024**2
                          for e in estimates.values())
            memory_risk = max(e['native_peak_projection_bytes'] for e in estimates.values()) > .8*40*1024**3
            cases.append(dict(kernels=n,resolution=resolution,views=estimates,
                selector='fused' if n <= 4096 else 'tiled_merge_distance_then_source_id',
                status='candidate_for_manual_memory_pilot' if bounded and not memory_risk else 'capacity_or_memory_review_required',
                budget_bounds_pass=bounded,native_memory_risk=memory_risk))
    spectral = ROOT/'data/converse/prepared/converse__free_eigen_albo_principal_curvatures.pt'
    return dict(gpu_started=False,cases=cases,mesh_faces=len(mesh.faces),spectral_bytes=spectral.stat().st_size,
        hardware_evidence='Saved reports: NVIDIA A100-PCIE-40GB; current free GPU memory unavailable in CPU preparation',
        bounds=dict(max_buffer_mib=max_buffer_mib,max_contributions=max_contributions),
        caveat='Conservative cap128 geometry bounds may overestimate actual support. Native memory projection '
               'uses saved 64px peaks and omits unknown compiler/library workspace and kernel-count growth. '
               'No configuration is GPU-validated here; run a 50-update memory/quality pilot before scaling.')
