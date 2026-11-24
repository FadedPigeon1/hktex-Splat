"""Compare native HKTex and HKS adaptive training without editing original HKTex.

Preview is standard-library only. Explicit execution uses fresh CUDA workers,
the inherited original optimisation loop, density controllers and optimizers.
Surface samples or rasterized camera pixels supervise both methods. Artifacts
are retained for verification; the older fixed-density camera benchmark is separate.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

from hktex_splatting.utils.experiment_io import ROOT, CAMERAS, configure_cache, gpu_guard


_PRESETS = {
    'compare-128': dict(
        backend='both', resolution=128, kernels=5000, iterations=5000, seed=0,
        config_overlay=ROOT/'splatting_script/configs/converse_boundary_stop.yaml',
        supervision='camera', raster_backend='nvdiffrast', target_sampling='gpu',
        visibility='fresh', shared_initial_model=True, surface_adapter='direct',
        support_domain='spectral', surface_support_hops=16, corner_evaluation='parallel',
        density_policy='original', optimizer_policy='original',
        max_buffer_mib=2048, max_contributions=67108864,
        lighting='ggx', material_roughness=.112565, allow_native_approximation=True,
        no_training_plots=True, verify_samples=1024),
}


def _preset_defaults(name, asset):
    """Expand a recorded comparison recipe; explicit CLI values still win.

    This only resolves arguments. It never prepares assets, initializes CUDA,
    or changes the original model/optimizer implementations.
    """
    defaults = dict(_PRESETS[name])
    defaults.update(asset=asset, config=ROOT/f'splatting_script/configs/{asset}_training.yaml',
                    cameras_file=CAMERAS if asset == 'converse' else ROOT/f'data/{asset}/cameras.json')
    return defaults


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def sources():
    paths = [ROOT/'optimisation.py', ROOT/'interactive_density_knn.py', ROOT/'pyproject.toml']
    for folder in ('hktex', 'hktex_splatting', 'splatting_script'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.suffix in ('.py', '.cu', '.cpp', '.h', '.hpp'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def write(path, data):
    with Path(path).open('x') as stream: json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')


def geodesic_log_counts(path):
    """Whole-worker warning events; repetitions are not unique failed kernels."""
    path = Path(path)
    counts = [int(value) for value in re.findall(r'digeo failed for (\d+) sample',
              path.read_text(errors='replace'))] if path.is_file() else None
    return dict(warning_calls=len(counts) if counts is not None else None,
        failed_sample_events=sum(counts) if counts is not None else None,
        scope='Whole-worker log, including any setup/density calls; not unique kernels or an iteration failure rate')


def lighting_settings(args):
    """Shared explicit-light controls; no learned illumination or hidden defaults."""
    return dict(mode=args.lighting, direction=list(args.light_direction),
                ambient=args.light_ambient, intensity=args.light_intensity,
                roughness=args.material_roughness, specular=args.material_specular,
                backend=args.lighting_backend)


def plan(args):
    from splatting_script.benchmarks.training_research import HELDOUT_VIEWS
    backends = ('native', 'hybrid') if args.backend == 'both' else (args.backend,)
    output = args.output.resolve()
    initial = args.initial_state.resolve() if args.initial_state else output/(backends[0]+'_verify')/'initial.pt'
    common = ['--config', str(args.config.resolve()), '--support-cap', str(args.support_cap),
        '--max-contributions', str(args.max_contributions), '--max-buffer-mib', str(args.max_buffer_mib),
        '--verify-samples', str(args.verify_samples), '--evaluation-seed', str(args.evaluation_seed)]
    if args.config_overlay:
        common += ['--config-overlay', str(args.config_overlay.resolve())]
    if args.cameras_file:
        common += ['--cameras-file', str(args.cameras_file.resolve())]
    for flag in ('iterations', 'kernels', 'seed'):
        if getattr(args, flag) is not None: common += ['--'+flag, str(getattr(args, flag))]
    common += ['--supervision', args.supervision, '--resolution', str(args.resolution),
               '--visibility',args.visibility,'--visibility-cache-mib',str(args.visibility_cache_mib),
               '--target-sampling', args.target_sampling, '--max-target-texture-mib', str(args.max_target_texture_mib),
               '--raster-backend',args.raster_backend,'--raster-near',str(args.raster_near),
               '--surface-support-hops',str(args.surface_support_hops),'--support-pruning',args.support_pruning,'--corner-evaluation',args.corner_evaluation,'--association-cache',args.association_cache,'--occupancy-threshold',str(args.occupancy_threshold)]
    common += ['--density-policy', args.density_policy, '--optimizer-policy', args.optimizer_policy,
               '--support-domain', args.support_domain]
    common += ['--lighting', args.lighting, '--light-direction', *map(str, args.light_direction),
               '--lighting-backend', args.lighting_backend,
               '--light-ambient', str(args.light_ambient), '--light-intensity', str(args.light_intensity),
               '--material-roughness', str(args.material_roughness), '--material-specular', str(args.material_specular)]
    if args.allow_render_variation: common.append('--allow-render-variation')
    if args.allow_native_approximation:common.append('--allow-native-approximation')
    common += ['--max-initial-psnr-drop-db',str(args.max_initial_psnr_drop_db)]
    if args.trace_diagnostics: common.append('--trace-diagnostics')
    if args.shared_initial_model: common.append('--shared-initial-model')
    if args.no_training_plots: common.append('--no-training-plots')
    common += ['--surface-adapter', args.surface_adapter, '--profile-start', str(args.profile_start),
               '--profile-steps', str(args.profile_steps), '--selector-merge', args.selector_merge]
    jobs = []
    for verification in (True, False):
        if args.verify_only and not verification: break
        for index, backend in enumerate(backends):
            name = backend+('_verify' if verification else '_train')
            command = [sys.executable, '-B', '-m', 'splatting_script.experiments.train_surface', '--worker', '--execute',
                       '--backend', backend, '--run-dir', str(output/name), *common]
            if verification: command.append('--verify-only')
            if args.initial_state or not (verification and index == 0):
                command += ['--initial-state', str(initial)]
            jobs.append(dict(id=name, backend=backend, role='verification' if verification else 'training',
                             command=command, run=str(output/name)))
    workflow = dict(format='hktex_full_surface_workflow_v1', gpu_started=False, output=str(output),
        jobs=jobs, initial_snapshot=str(initial),
        config_inputs=[str(args.config.resolve())]+([str(args.config_overlay.resolve())] if args.config_overlay else []),
        initialization_policy=('shared-camera-initial-model' if args.supervision == 'camera' else 'shared-kernel-placement') if args.shared_initial_model else 'independent-replay',
        allow_render_variation=args.allow_render_variation,
        raster_backend=args.raster_backend, raster_near=args.raster_near, support_pruning=args.support_pruning,corner_evaluation=args.corner_evaluation,association_cache=args.association_cache,occupancy_threshold=args.occupancy_threshold,support_domain=args.support_domain,surface_support_hops=args.surface_support_hops,allow_native_approximation=args.allow_native_approximation,max_initial_psnr_drop_db=args.max_initial_psnr_drop_db,
        visibility=args.visibility, visibility_cache_mib=args.visibility_cache_mib,
        target_sampling=args.target_sampling, max_target_texture_mib=args.max_target_texture_mib,
        supervision=args.supervision, resolution=args.resolution,
        cameras_file=str(args.cameras_file.resolve()) if args.cameras_file else None,
        lighting=lighting_settings(args),
        evaluation_views=list(HELDOUT_VIEWS) if args.supervision == 'camera' else [],
        compiled_cache_policy='per_backend_toolchain_artifacts_v1',
        training_progress=False, trajectory_logging='buffered_jsonl_v1',
        density_policy=args.density_policy, optimizer_policy=args.optimizer_policy,
        surface_adapter=args.surface_adapter, profile_steps=args.profile_steps,
        selector_merge=args.selector_merge,
        cache_surface_sampling=False, training_plots=not args.no_training_plots,
        protocol=('Rasterized camera pixels for both; visibility policy '+args.visibility+'; real screen tiles; no importance pool. ' if args.supervision == 'camera' else 'Original UV surface samples for both; HKS work tiles are not camera pixels. ')
                 +'Inherited original optimisation loop; native density/optimizers stay original. Requested HKS policies are recorded explicitly. '
                 'Adaptive sampling and density decisions may diverge after learning diverges. '
                 'No speedup claim is inferred from these experimental full-training records.')
    if getattr(args, 'preset', None):
        workflow.update(preset=args.preset, asset=args.asset)
    return workflow


def optimizer_alignment(trainer):
    """Check current optimizer/face-index bindings after original density operations."""
    parameters = {id(p) for p in trainer.model.parameters()}
    for optimizer in trainer.optimizers:
        for group in optimizer.param_groups:
            for index, parameter in enumerate(group['params']):
                if id(parameter) not in parameters: raise RuntimeError('Optimizer still references a replaced parameter')
                if 'face_ids' in group and group['face_ids'][index] is not trainer.model._kernel_face_ids:
                    raise RuntimeError('Geodesic optimizer face IDs are stale')
                for name, value in optimizer.state.get(parameter, {}).items():
                    if name != 'step' and hasattr(value, 'shape') and value.shape != parameter.shape:
                        raise RuntimeError('Optimizer tensor shape no longer matches parameter: '+name)
                    if hasattr(value, 'isfinite') and not bool(value.isfinite().all()):
                        raise RuntimeError('Nonfinite optimizer state: '+name)


def check_initial_controls(controls, initial):
    """Reject source/config/software drift before a worker starts its original loop."""
    if initial is None: return
    manifest = Path(initial).parent/'manifest.json'
    if not manifest.is_file(): raise FileNotFoundError('Keep the full-surface initial.pt with its manifest.json')
    reference = json.loads(manifest.read_text())
    if reference.get('format') != 'hktex_full_surface_run_v1':
        raise ValueError('Initial snapshot companion is not a full-surface manifest')
    expected, actual = dict(reference['controls']), dict(controls)
    # An explicitly supplied initialization can be reused for another update
    # budget. Both current arms still have to use the same requested budget.
    expected.pop('requested_updates', None); actual.pop('requested_updates', None)
    if expected != actual:
        raise RuntimeError('Source/config/asset/software/hardware controls differ from the shared initial snapshot')


def worker(args):
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    before = sources()
    start = time.perf_counter()
    startup_wall = {}
    phase_start = start
    def startup_phase(name):
        nonlocal phase_start
        now = time.perf_counter()
        startup_wall[name] = now-phase_start
        phase_start = now
    try:
        gpu_guard()
        cache = configure_cache('surface_'+directory.name+'_'+str(os.getpid()), compiled_namespace=args.backend)
        os.environ['MPLBACKEND'] = 'Agg'
        # Keep all trajectory records, without terminal refresh or per-step file opens.
        os.environ['TQDM_DISABLE'] = '1'
        import numpy as np
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA-enabled PyTorch and an NVIDIA GPU are required')
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        import hktex
        from hktex.utils import load_config, seed_everything
        from hktex.trainers.base import BaseTrainer
        from hktex_splatting.training.uv_training import (
            RecordedUvTrainer, HksUvTrainer, VerificationComplete, tree_hash,
            sampler_state, cpu_tree, native_forward,
        )
        from hktex_splatting.training.surface_adapter import surface_forward
        from splatting_script.benchmarks.training_research import capture_rng, preserve_rng, HELDOUT_VIEWS
        from omegaconf import OmegaConf
        # Match optimisation.py's precision policy, rather than the separate
        # camera benchmark's highest-precision recipe.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        overrides = ['use_timestamp=false', 'tag=surface_training', f'exp_root_dir={cache / "loader"}']
        for argument, key in (('seed', 'seed'), ('kernels', 'trainer.model.n_sources'), ('iterations', 'optim.iters')):
            if getattr(args, argument) is not None: overrides.append(f'{key}={getattr(args, argument)}')
        config_inputs = [str(args.config.resolve())]
        if args.config_overlay: config_inputs.append(str(args.config_overlay.resolve()))
        cfg = load_config(*config_inputs, cli_args=overrides, n_gpus=1)
        if cfg.data_type != 'data.uv-texture-sampler' or cfg.trainer_type != 'trainers.uv-texture' or not cfg.trainer.use_knn_implementation:
            raise ValueError('This adapter supports the original KNN UV surface-training recipe only')
        if cfg.data.num_workers != 0:
            raise ValueError('Verifiable sampler feedback/checkpoints require data.num_workers=0')
        if not cfg.trainer.eigen_albo.use_precomputed or not cfg.trainer.eigen_albo.error_if_not_precomputed:
            raise ValueError('Prepare the spectral archive first; this training command never regenerates it')
        mesh_path = Path(cfg.data.mesh_path).resolve()
        spectral = Path(str(Path(cfg.trainer.eigen_albo.mesh_path).resolve()).rsplit('.', 1)[0]+
                        '_'+cfg.trainer.eigen_albo.get('precomputed_name', 'eigen_albo')+'_'+cfg.trainer.eigen_albo.local_frames+'.pt')
        if not mesh_path.is_file() or not spectral.is_file(): raise FileNotFoundError('Mesh or precomputed spectral archive is missing')
        startup_phase('imports_cuda_config')
        if args.supervision == 'camera': cfg.data.use_importance_sampling = False
        if args.supervision=='camera' and args.target_sampling=='gpu' and cfg.data.num_workers!=0:
            raise ValueError('GPU camera supervision requires data.num_workers=0; do not fork CUDA datasets')
        seed_everything(cfg.seed)
        data = hktex.find(cfg.data_type)(cfg.data)
        data.prepare_data(); data.setup('fit')
        startup_phase('seed_and_mesh_data_setup')
        if args.supervision == 'camera':
            from hktex_splatting.training.camera_supervision import CameraSupervision
            original = data.train_dataset
            data.train_dataset = CameraSupervision(original, args.resolution, target_backend=args.target_sampling,
                raster_backend=args.raster_backend,raster_near=args.raster_near, lighting=lighting_settings(args), camera_path=args.cameras_file,
                visibility=args.visibility, visibility_cache_bytes=args.visibility_cache_mib*1024**2,
                max_texture_bytes=args.max_target_texture_mib*1024**2)
            data._camera_evaluation_dataset = CameraSupervision(original, args.resolution, views=HELDOUT_VIEWS,
                target_backend=args.target_sampling, target_sampler=data.train_dataset.target_sampler,
                raster_backend=args.raster_backend,raster_near=args.raster_near, lighting=lighting_settings(args), camera_path=args.cameras_file,
                visibility=args.visibility, visibility_cache_bytes=args.visibility_cache_mib*1024**2)
        startup_phase('sampler_cache_setup')
        cls = RecordedUvTrainer if args.backend == 'native' else HksUvTrainer
        assert cls.optimise is BaseTrainer.optimise
        trainer = cls(cfg.trainer, data, renderer_cfg=cfg.renderer)
        startup_phase('trainer_model_and_spectral_setup')
        if args.supervision == 'camera':
            if (not np.array_equal(trainer.mesh.faces.detach().cpu().numpy(), original.mesh.faces) or
                    not np.allclose(trainer.mesh.verts.detach().cpu().numpy(), original.mesh.vertices, atol=1e-6, rtol=1e-5)):
                raise ValueError('Raster target/model mesh alignment mismatch')
        fixed = (trainer.mesh.verts, trainer.mesh.faces, trainer.eigalbo_interp._eigen_vec_knn,
                 trainer.eigalbo_interp._iso_embeddings)
        fixed_versions = tuple(tensor._version for tensor in fixed)
        if any(tensor.requires_grad for tensor in fixed):
            raise ValueError('Mesh geometry and spectral assets must remain fixed')
        trainer.attach_run(directory, dict(support_cap=args.support_cap,
            lighting=lighting_settings(args),
            max_contributions=args.max_contributions, max_buffer_mib=args.max_buffer_mib,
            verify_samples=args.verify_samples, verify_only=args.verify_only, evaluation_seed=args.evaluation_seed,
            shared_initial_model=args.shared_initial_model, surface_adapter=args.surface_adapter,
            selector_merge=args.selector_merge,support_pruning=args.support_pruning,corner_evaluation=args.corner_evaluation,association_cache=args.association_cache,occupancy_threshold=args.occupancy_threshold,support_domain=args.support_domain,surface_support_hops=args.surface_support_hops,allow_native_approximation=args.allow_native_approximation,max_initial_psnr_drop_db=args.max_initial_psnr_drop_db, density_policy=args.density_policy,
            optimizer_policy=args.optimizer_policy, supervision=args.supervision, allow_render_variation=args.allow_render_variation, training_plots=not args.no_training_plots),
            expected_initial=args.initial_state)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        write(directory/'resolved_config.json', resolved)
        controls = dict(raster_backend=args.raster_backend,raster_near=args.raster_near,support_pruning=args.support_pruning,corner_evaluation=args.corner_evaluation,association_cache=args.association_cache,occupancy_threshold=args.occupancy_threshold,support_domain=args.support_domain,surface_support_hops=args.surface_support_hops,allow_native_approximation=args.allow_native_approximation,max_initial_psnr_drop_db=args.max_initial_psnr_drop_db,visibility=args.visibility, visibility_cache_mib=args.visibility_cache_mib, target_sampling=args.target_sampling, max_target_texture_mib=args.max_target_texture_mib, allow_render_variation=args.allow_render_variation, supervision=args.supervision, resolution=args.resolution,
            lighting=lighting_settings(args), lighting_target=('original unlit texture RGB' if args.lighting=='none' else 'known-light analytic shading of original texture; not captured or ray-traced ground truth'),
            camera_definitions=data.train_dataset.cameras if args.supervision == 'camera' else None,
            cameras_file_sha256=sha256(args.cameras_file) if args.cameras_file else None,
            training_views=list(data.train_dataset.views) if args.supervision == 'camera' else None,
            evaluation_view='heldout_left' if args.supervision == 'camera' else None,
            evaluation_views=list(HELDOUT_VIEWS) if args.supervision == 'camera' else None,
            compiled_cache_policy='per_backend_toolchain_artifacts_v1',
            training_progress=False, trajectory_logging='buffered_jsonl_v1',
            seed=cfg.seed, requested_updates=cfg.optim.iters, data=resolved['data'],
            config_sha256=sha256(args.config),
            trainer=resolved['trainer'], renderer=resolved['renderer'], evaluation_seed=args.evaluation_seed,
            mesh_sha256=sha256(mesh_path), spectral_sha256=sha256(spectral), sources=before,
            surface_options=dict(support_cap=args.support_cap, max_contributions=args.max_contributions,
                max_buffer_mib=args.max_buffer_mib, verify_samples=args.verify_samples,
                trace_diagnostics=args.trace_diagnostics, surface_adapter=args.surface_adapter,
                profile_start=args.profile_start, profile_steps=args.profile_steps),
            selector_merge=args.selector_merge, density_policy=args.density_policy,
            optimizer_policy=args.optimizer_policy,
            cache_surface_sampling=False, training_plots=not args.no_training_plots,
            precision=dict(matmul=torch.get_float32_matmul_precision(), tf32=torch.backends.cuda.matmul.allow_tf32),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        if args.config_overlay:
            controls['config_overlay_sha256'] = sha256(args.config_overlay)
        controls['tracer_policy'] = dict(type=cfg.trainer.tracer_type,
            avoid_holes=getattr(trainer.tracer.cfg, 'avoid_holes', None),
            max_iterations=getattr(trainer.tracer.cfg, 'max_iterations', None))
        controls['initialization_policy'] = ('shared-camera-initial-model' if args.supervision == 'camera' else 'shared-kernel-placement') if args.shared_initial_model else 'independent-replay'
        packages = {}
        for name in ('torch', 'triton', 'numpy', 'numba', 'mitsuba', 'digeo', 'faiss-gpu-cuvs', 'omegaconf'):
            try: packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name] = 'not recorded under this package name'
        if args.raster_backend=='nvdiffrast':
            distribution=importlib.metadata.distribution('nvdiffrast')
            packages['nvdiffrast']=distribution.version
            packages['nvdiffrast_source']=json.loads(distribution.read_text('direct_url.json') or 'null')
        import faiss
        packages['faiss_module'] = str(faiss.__version__)
        controls['software'] = packages
        controls['cuda'] = torch.version.cuda
        controls['torch_cpu_threads'] = torch.get_num_threads()
        check_initial_controls(controls, args.initial_state)
        write(directory/'manifest.json', dict(format='hktex_full_surface_run_v1', backend=args.backend,
            adapter=cls.__name__, inherited_loop='hktex.trainers.base.BaseTrainer.optimise',
            controls=controls, software=packages, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0), command=sys.argv, initial_snapshot=str(args.initial_state or directory/'initial.pt'),
            source_scope='Current working-tree files, including pre-existing local edits; not a claim of pristine upstream HKTex',
            compiler_cache=dict(policy='per_backend_toolchain_artifacts_v1', workspace=str(cache),
                paths={name:os.environ[name] for name in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
                    'TORCH_EXTENSIONS_DIR','NUMBA_CACHE_DIR','MPLCONFIGDIR','CUDA_CACHE_PATH','XDG_CACHE_HOME')}),
            checkpoint_resume_supported=False, sample_layout='rasterized_camera_pixels' if args.supervision == 'camera' else 'surface_sample_tiles' if args.backend == 'hybrid' else 'original_surface_batch'))
        torch.cuda.synchronize()
        startup_phase('controls_provenance_and_final_gpu_wait')
        optimise_start = time.perf_counter()
        trainer.loop_started_at = optimise_start
        verified_only = False
        diagnostic = None
        if args.trace_diagnostics:
            from splatting_script.benchmarks.legacy.geodesic_capture import TraceDiagnostics
            diagnostic = TraceDiagnostics(torch, trainer, directory)
        profiler = None
        if args.profile_steps:
            from hktex_splatting.training.surface_profile import SurfaceStageProfiler
            profiler = SurfaceStageProfiler(torch, trainer, args.profile_start, args.profile_steps)
        with ExitStack() as stack:
            trainer._trajectory_stream = stack.enter_context(
                (directory/'trajectory.jsonl').open('x', buffering=1024*1024))
            if diagnostic is not None: stack.enter_context(diagnostic.install())
            if profiler is not None: stack.enter_context(profiler.install())
            try:
                trainer.optimise(n_iter=cfg.optim.iters)
            except VerificationComplete:
                verified_only = True
        torch.cuda.synchronize()
        optimise_seconds = time.perf_counter()-optimise_start
        training_interval = None if verified_only else time.perf_counter()-trainer.training_start
        memory = dict(allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved(),
            scope=('Verification worker including gates; no training' if verified_only else 'Training after initial gates, before held-out evaluation/forward latency; PyTorch allocator only'))
        optimizer_alignment(trainer)
        current_fixed = (trainer.mesh.verts, trainer.mesh.faces, trainer.eigalbo_interp._eigen_vec_knn,
                         trainer.eigalbo_interp._iso_embeddings)
        if any(old is not new for old, new in zip(fixed, current_fixed)) or fixed_versions != tuple(tensor._version for tensor in fixed):
            raise RuntimeError('A fixed mesh/spectral tensor was replaced or modified in place')
        health = {name: dict(shape=list(parameter.shape), finite=bool(torch.isfinite(parameter).all()))
                  for name, parameter in trainer.model.named_parameters()}
        if not all(row['finite'] for row in health.values()): raise RuntimeError('Nonfinite final learned parameters')
        result = dict(status=('APPROXIMATE_' if args.allow_native_approximation and args.backend=='hybrid' else 'DIAGNOSTIC_' if args.allow_render_variation else '')+('VERIFIED' if verified_only else 'COMPLETE'), backend=args.backend,
            lighting=lighting_settings(args), ray_traced=False,
            initial_identity=trainer.initial_identity, completed_updates=trainer.step_index,
            approximation_accepted=getattr(trainer,'native_approximation_accepted',False),
            gpu_operator_checks_passed=getattr(trainer,'gpu_operator_checks_passed',None),
            initialization=trainer.initialization_receipt,
            original_optimise_call_seconds=optimise_seconds,
            memory=memory, final_kernel_count=trainer.model.N_sources,
            kernel_count_trajectory=[dict(step=r['step'],before=r['kernel_count_before'],after=r['kernel_count_after']) for r in trainer.records],
            density_parameter_change_steps=[r['step'] for r in trainer.records if r['kernel_count_before'] != r['kernel_count_after']],
            density_parameter_replacement_steps=[r['step'] for r in trainer.records if r['parameters_replaced']],
            optimizer_bindings_verified=True, controls_sha256=tree_hash(controls), initial_snapshot_verified=True,
            performance_claim_validated=False,
            diagnostic_unvalidated=args.allow_render_variation,
            strict_render_gate_passed=trainer.strict_render_gate_passed,
            diagnostic_render_variation_accepted=trainer.diagnostic_render_variation,
            fixed_tensor_versions_verified=True,
            final_parameter_health=health,
            tracer_policy=controls['tracer_policy'],
            trace_diagnostics=diagnostic.summary() if diagnostic is not None else None,
            stage_profile=profiler.summary() if profiler is not None else None,
            surface_optimization=getattr(trainer, 'surface_optimization', None),
            supervision=args.supervision,
            raster_backend=args.raster_backend,raster_near=args.raster_near,support_pruning=args.support_pruning,corner_evaluation=args.corner_evaluation,association_cache=args.association_cache,occupancy_threshold=args.occupancy_threshold,support_domain=args.support_domain,surface_support_hops=args.surface_support_hops,allow_native_approximation=args.allow_native_approximation,max_initial_psnr_drop_db=args.max_initial_psnr_drop_db,
            device_visibility=args.raster_backend=='nvdiffrast' and args.backend=='hybrid',
            batch_identity_scope='camera metadata only; no per-step pixel fingerprint' if args.raster_backend=='nvdiffrast' else 'sample face/bary/target tensor contents',
            raster_validation=getattr(data.train_dataset.raster,'validation',None) if args.supervision=='camera' else None,
            visibility=args.visibility if args.supervision == 'camera' else None,
            camera_geometry_builds=getattr(trainer,'camera_geometry_builds',0),
            camera_geometry_hits=getattr(trainer,'camera_geometry_hits',0),
            camera_workload=getattr(trainer,'camera_workload',None),
            occupancy_stats=getattr(trainer,'occupancy_stats',None),
            camera_selector_stats=getattr(trainer,'camera_selector_stats',None),
            target_sampling=args.target_sampling if args.supervision == 'camera' else None,
            target_texture_validation=data.train_dataset.target_sampler.validation if args.supervision == 'camera' and data.train_dataset.target_sampler is not None else None,
            target_texture_device_bytes=data.train_dataset.target_sampler.device_bytes if args.supervision == 'camera' and data.train_dataset.target_sampler is not None else 0,
            target_timing_scope='Host UV/target preparation and GPU enqueue; CUDA completion is covered by synchronized training/worker times',
            camera_data_stats=dict(frames=data.train_dataset.frames, raster_seconds=data.train_dataset.raster_seconds, target_seconds=data.train_dataset.target_seconds, raster_calls=data.train_dataset.raster_calls, visibility_hits=data.train_dataset.visibility_hits, visibility_invalidations=data.train_dataset.visibility_invalidations, visibility_bytes=data.train_dataset.visibility_bytes) if args.supervision == 'camera' else None,
            sampling_probability_cache=False,
            startup_wall_seconds=startup_wall,
            loop_before_receipt_seconds=trainer.initial_receipt_started_at-optimise_start,
            initial_receipt_and_gates_seconds=trainer.initial_receipt_seconds,
            density_policy=args.density_policy if args.backend == 'hybrid' else 'original',
            optimizer_policy=args.optimizer_policy if args.backend == 'hybrid' else 'original',
            density_configuration=[],
            density_events=[event for dc in trainer.density_controllers for event in getattr(dc, 'events', [])],
            training_plots=not args.no_training_plots,
            scope=('Camera rasterization (policy '+args.visibility+'), target lookup, image-space forward/backward, adaptive density and optimization. Complete worker includes setup/evaluation/saving; first frame and initial gates precede training interval. '+('Unlit RGB.' if args.lighting=='none' else 'Shared analytic '+args.lighting+' lighting; linear radiance, known lights, no shadows or indirect light. Not ray traced.') if args.supervision == 'camera' else 'Inherited original surface training lifecycle plus receipt/validation/plot overhead; no camera-render timing claim.'))
        if not verified_only:
            result['training_interval_seconds'] = training_interval
            result['loop_setup_and_initial_gates_seconds'] = trainer.training_start-optimise_start
            result['final_loss'] = trainer.records[-1]['loss']
            result['batch_sequence_sha256'] = tree_hash([row['batch_sha256'] for row in trainer.records])
            checkpoint = dict(format='hktex_full_surface_checkpoint_v1', kernel_count=trainer.model.N_sources,
                model=cpu_tree(trainer.model.state_dict()), lighting=lighting_settings(args),
                material_color_space='srgb', rendered_color_space='srgb' if args.lighting=='none' else 'linear_radiance',
                optimizers=cpu_tree([o.state_dict() for o in trainer.optimizers]),
                schedulers=cpu_tree([s.state_dict() for s in trainer.schedulers]),
                rng=capture_rng(torch, trainer.device), sampler=cpu_tree(sampler_state(data)),
                density_states=cpu_tree([dc._state for dc in trainer.density_controllers]),
                kernel_barycentric_coords=cpu_tree(getattr(trainer.model._kernel_locations, 'bary_coords', None)),
                completed_updates=trainer.step_index, resume_supported=False)
            with (directory/'final.pt').open('xb') as stream: torch.save(checkpoint, stream)
            result['final_model_sha256'] = tree_hash(checkpoint['model'])
            result['final_checkpoint_sha256'] = sha256(directory/'final.pt')
            evaluation = trainer.evaluation_batch
            quality = {}
            with preserve_rng(torch, trainer.device), torch.no_grad():
                evaluations = [('surface_samples', evaluation)]
                if args.supervision == 'camera':
                    iterator = iter(data._camera_evaluation_dataset)
                    evaluations = [(view, next(iterator)) for view in HELDOUT_VIEWS]
                for name in ('native_field', 'trained_backend'):
                    per_view = {}
                    for view, batch in evaluations:
                        rgb = (native_forward(trainer, batch) if name == 'native_field' or args.backend == 'native'
                               else surface_forward(trainer, batch))
                        target = batch['colour'].to(trainer.device)
                        mse = float((rgb-target).square().mean())
                        per_view[view] = dict(mse=mse, psnr_db=None if mse == 0 else -10*np.log10(mse),
                            pixels=len(target), values=target.numel())
                    total = sum(row['values'] for row in per_view.values())
                    mse = sum(row['mse']*row['values'] for row in per_view.values())/total
                    quality[name] = dict(mse=mse, psnr_db=None if mse == 0 else -10*np.log10(mse),
                        exact_match=mse == 0, per_view=per_view,
                        scope='Foreground-sample-weighted MSE across recorded held-out views of the same mesh; '+('sRGB texture' if args.lighting=='none' else 'linear radiance')+'; PSNR reference peak=1', reference_peak=1.)
            result['heldout_surface_quality'] = quality
            # Separate synchronized, unprofiled forward-only measurement. Input
            # visibility/targets are held fixed; training timing includes fresh raster.
            previous_profiler=getattr(trainer,'surface_profiler',None)
            trainer.surface_profiler=None
            renderer_profiles=[(r,r.profile) for r in getattr(trainer,'_camera_renderers',{}).values()]
            for r,_ in renderer_profiles:r.profile=False
            try:
                with preserve_rng(torch,trainer.device),torch.no_grad():
                    latency=[]
                    for repetition in range(6):
                        trainer.model.reset(trainer.eigalbo_interp)
                        torch.cuda.synchronize()
                        begin=time.perf_counter()
                        rgb=(native_forward(trainer,evaluation) if args.backend=='native'
                             else surface_forward(trainer,evaluation))
                        torch.cuda.synchronize()
                        if repetition:latency.append(time.perf_counter()-begin)
                result['forward_only_latency']=dict(median_seconds=float(np.median(latency)),
                    samples_seconds=latency,warmup_calls=1,profiled=False,
                    scope='Held-out camera/sample visibility fixed; fresh source preparation, support, coefficients, material RGB and '+args.lighting+' shading; excludes raster/target/loss/backward')
            finally:
                trainer.surface_profiler=previous_profiler
                for r,profile in renderer_profiles:r.profile=profile

        result['sources_unchanged_during_run'] = before == sources()
        if not result['sources_unchanged_during_run']: raise RuntimeError('Source files changed during this run')
        if controls['mesh_sha256'] != sha256(mesh_path) or controls['spectral_sha256'] != sha256(spectral):
            raise RuntimeError('Mesh or spectral asset changed during this run')
        if controls['config_sha256'] != sha256(args.config):
            raise RuntimeError('Source configuration changed during this run')
        if args.config_overlay and controls['config_overlay_sha256'] != sha256(args.config_overlay):
            raise RuntimeError('Configuration overlay changed during this run')
        if args.cameras_file and controls['cameras_file_sha256'] != sha256(args.cameras_file):
            raise RuntimeError('Camera definitions changed during this run')
        result['assets_unchanged_during_run'] = True
        result['worker_wall_seconds'] = time.perf_counter()-start
        write(directory/'result.json', result)
        print('SURFACE_TRAINING_RUN='+str(directory), flush=True)
        return 0
    except Exception:
        write(directory/'failure.json', dict(traceback=traceback.format_exc(), sources_unchanged=before == sources()))
        raise


def comparison_markdown(workflow):
    lines = ['# Original HKTex versus HKS: '+('rasterized camera training' if workflow.get('supervision') == 'camera' else 'full surface-sample training'), '', workflow['protocol'], '',
        '**Experimental adapter:** raw training/quality measurements only; no validated speedup claim. '
        'Full checkpoints and manifests are retained for verification.', '', '| Job | Status |', '|---|---|']
    if workflow.get('allow_render_variation'):
        lines[4:4] = ['**DIAGNOSTIC / UNVALIDATED:** bounded render variation is allowed. Strict failures and diagnostic acceptance remain recorded. No validated performance or numerical-equivalence claim.', '']
    if workflow.get('supervision') == 'camera':
        lines[4:4] = ['Visibility: `'+workflow.get('visibility','fresh')+'`; shared target sampling: `'+workflow.get('target_sampling','cpu')+'`. GPU target times measure host work/enqueue; synchronized whole-run times include completion.', '']
        lines[4:4] = ['Triangle rasterization backend: `'+workflow.get('raster_backend','cpu')+'`; fresh GPU mode retains visibility on device for HKS. Actual camera tiles, adaptive density and original-loop optimization. Lighting: `'+workflow.get('lighting',{}).get('mode','none')+'`; final quality aggregates recorded held-out views, while initial checks and forward latency use the saved first view.', '']
    if workflow.get('lighting', {}).get('mode', 'none') != 'none':
        lines[4:4] = ['**Analytic lighting, not ray tracing:** both fields reconstruct sRGB material albedo, then share Lambert/GGX shading in linear radiance, including lit targets and timed forwards. Known light, fixed face normals, roughness/F0; no shadows, indirect light or inverse-light estimation. PSNR reference peak is 1 even when radiance exceeds 1. Do not compare these timings/PSNR with unlit or Mitsuba results.', 'Light/material controls: `'+json.dumps(workflow['lighting'],sort_keys=True)+'`.', '']
    if workflow.get('density_policy', 'original') != 'original' or workflow.get('optimizer_policy', 'original') != 'original':
        lines[4:4] = ['**Custom HKS training experiment:** native uses original policies; HKS uses requested density/optimizer policies. This does not isolate renderer speed.',
                     'Requested HKS density: `'+workflow.get('density_policy','original')+'`; optimizer: `'+workflow.get('optimizer_policy','original')+'`.', '']
    if workflow.get('initialization_policy') in ('shared-kernel-placement','shared-camera-initial-model'):
        if workflow.get('initialization_policy') == 'shared-camera-initial-model':
            lines[4:4] = ['**Explicit shared camera initialization:** native initial positions/face IDs and associated residual/mean colors are copied before any update. All other model fields, RNG, evaluation and fixed assets must match. Rejected proposals are retained.', '']
        else:
            lines[4:4] = ['**Explicit shared placement:** all workers use the first original kernel positions/face IDs. '
            'Other model fields, RNG, sample pool and held-out batch must match exactly; placement receipts '
            'and any rejected local placement proposals are retained.', '']
    lines[4:4] = ['Surface adapter: `'+workflow.get('surface_adapter', 'reference')+'`.', '']
    lines[4:4] = ['Association: '+('global spectral pixel KNN inverted into kernel/triangle support'
        if workflow.get('support_domain','graph')=='spectral' else 'kernel-seeded mesh-edge graph')+
        '; legacy selector-merge setting is inactive.', '']
    if workflow.get('support_domain','graph') == 'spectral':
        lines[4:4]=['**Spectral-support HKS:** global per-pixel float64 KNN, triangle-union coefficients and per-pixel membership filtering; no hop truncation. GPU/eager and native-field checks remain required.','']
    else:
        lines[4:4]=['**Approximate kernel-centric HKS:** source-seeded edge-hop support, '+str(workflow.get('surface_support_hops',4))+' hops; strict native-field validation required.','']
    if workflow.get('allow_native_approximation',False):lines[4:4]=['**Approximation experiment:** strict GPU/eager operator checks remain required. Initial target PSNR may be up to '+str(workflow.get('max_initial_psnr_drop_db'))+' dB below native. Final quality and speed are unvalidated until measured.','']
    if workflow.get('occupancy_threshold',0):lines[4:4]=['**Changed evaluator:** native pointwise below '+str(workflow['occupancy_threshold'])+' pixels/face; triangle HKS otherwise. This is not all-triangle HKS.','']
    lines[4:4]=['Persistent candidate index: `'+workflow.get('association_cache','none')+'`; live coefficients are recomputed.','']
    lines[4:4] = ['Corner coefficient execution: `'+workflow.get('corner_evaluation','serial')+'`.', '']
    lines[4:4] = ['Conservative support pruning requested: `'+workflow.get('support_pruning','none')+'` (eligible shapes only).', '']
    lines[4:4] = ['HKS fixed sampling-probability cache: `'+str(workflow.get('cache_surface_sampling',False))+
        '`; training plots: `'+str(workflow.get('training_plots',True))+'`.', '']
    if workflow.get('profile_steps'):
        lines[4:4] = ['**Stage profiling enabled: diagnostic timings include synchronization overhead.**', '']
    for job in workflow['jobs']: lines.append('| '+job['id']+' | '+job.get('status', 'not_executed')+' |')
    for job in workflow['jobs']:
        validation=Path(job['run'])/'sample_validation.json'
        if job['role']=='verification' and validation.is_file():
            gate=json.loads(validation.read_text())
            if 'native_field' in gate:
                native=gate['native_field'];replay=gate.get('native_candidate_replay',{})
                lines += ['', 'Validation diagnostics for '+job['id']+' (not performance measurements):',
                    'GPU/eager implementation checks: `'+str(gate.get('operator_checks_passed'))+
                    '`; native-field equivalence: `'+str(native.get('equivalence_passed'))+'`.',
                    'Approximation accepted: `'+str(gate.get('approximation_accepted',False))+'`; initial PSNR drop dB: `'+str(gate.get('approximation_quality',{}).get('initial_psnr_drop_db'))+'`.',
                    'Support missing native selected weight fraction: `'+str(native.get('support_coverage',{}).get('missing_native_weight_fraction'))+'`.']
                if replay:
                    lines += ['Native-candidate replay: '+json.dumps({key:replay.get(key) for key in (
                        'samples','candidate_ids_equal','native_response_max_abs','native_rgb_max_abs',
                        'native_top_ids_equal','operator_rgb_max_abs','operator_contributions','operator_passed')})+'.',
                        'Replay/native parameter VJPs all close: `'+str(all(p['passed'] for p in replay['native_parameters']))+'`.']
        failure = Path(job['run'])/'failure.json'
        if failure.is_file():
            details = json.loads(failure.read_text())
            lines += ['', 'Failure in '+job['id']+':', '```text',
                      *details['traceback'].splitlines()[-12:], '```']
    runs = {}
    for job in workflow['jobs']:
        if job['role'] != 'training' or job.get('status') != 'complete': continue
        run = json.loads((Path(job['run'])/'result.json').read_text())
        run['process_wall_seconds'] = job.get('process_wall_seconds')
        run['geodesic_warnings'] = job.get('geodesic_warnings') or geodesic_log_counts(
            Path(job['run']).parent/(job['id']+'.log'))
        runs[job['backend']] = run
    if any(run.get('tracer_policy', {}).get('avoid_holes') is False for run in runs.values()):
        lines += ['', '**Boundary-stop experiment:** `avoid_holes=false` differs from the original '
                  'default boundary policy. The table records the actual setting for each backend.', '']
    if workflow.get('evaluation_views'):
        lines += ['', 'Final PSNR uses foreground-sample-weighted MSE across '+
            ', '.join(workflow['evaluation_views'])+'. Per-view scores remain in result.json; '
            'initial gates and forward latency use the saved first view.']
    lines += ['', '| Metric | Original HKTex | HKS '+('camera pixels' if workflow.get('supervision') == 'camera' else 'sample tiles')+' |', '|---|---:|---:|']
    for label, key in (('Completed updates', 'completed_updates'), ('Final kernel count', 'final_kernel_count'),
            ('Original optimise call seconds (includes setup/verification/plots)', 'original_optimise_call_seconds'),
            ('Training interval seconds (see scope)', 'training_interval_seconds'),
            ('Forward-only median seconds (fixed input visibility)', 'forward_only_latency.median_seconds'),
            ('Complete process wall seconds', 'process_wall_seconds'),
            ('Worker body seconds (excludes interpreter entry/initial source scan)', 'worker_wall_seconds'),
            ('Last sampled-batch loss', 'final_loss'),
            ('Held-out deployed surface PSNR dB', 'heldout_surface_quality.trained_backend.psnr_db'),
            ('Held-out common native-field PSNR dB', 'heldout_surface_quality.native_field.psnr_db'),
            ('Peak allocated PyTorch bytes', 'memory.allocated_bytes'), ('Peak reserved PyTorch bytes', 'memory.reserved_bytes'),
            ('Geodesic warning calls', 'geodesic_warnings.warning_calls'),
            ('Geodesic failed-sample events', 'geodesic_warnings.failed_sample_events'),
            ('Tracer avoid_holes (false = boundary stop)', 'tracer_policy.avoid_holes')):
        values = []
        for backend in ('native', 'hybrid'):
            value = runs.get(backend, {})
            for part in key.split('.'): value = value.get(part, 'N/A') if isinstance(value, dict) else 'N/A'
            values.append('N/A' if value is None else str(value))
        if workflow.get('supervision') == 'camera': label = label.replace('surface PSNR', 'camera PSNR')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    if len(runs) == 2:
        matched = runs['native']['controls_sha256'] == runs['hybrid']['controls_sha256'] and runs['native']['initial_identity'] == runs['hybrid']['initial_identity']
        lines += ['', 'Matched controls and initial model/RNG/pool/held-out receipts: '+str(matched)+'.']
        hashes=[runs[b].get('batch_sequence_sha256') for b in ('native','hybrid')]
        label=('Identical camera metadata trajectory (not pixel content)' if workflow.get('raster_backend')=='nvdiffrast'
               else 'Identical sampled-batch trajectory')
        lines += [label+': '+(str(hashes[0]==hashes[1]) if all(hashes) else 'not recorded')+'.']
    for backend, run in runs.items():
        lines += ['', '## '+backend, '', 'Held-out '+('camera' if workflow.get('supervision') == 'camera' else 'surface')+' quality: `'+json.dumps(run.get('heldout_surface_quality'))+'`.',
            'Peak PyTorch allocator bytes (excludes some driver/library memory): `'+json.dumps(run['memory'])+'`.',
            'Density count-change steps (zero-based): `'+str(run['density_parameter_change_steps'])+'`.']
        if run.get('target_texture_validation'):
            lines += ['', 'Target lookup validation: `'+json.dumps(run['target_texture_validation'])+'`; fixed target GPU bytes: `'+str(run.get('target_texture_device_bytes',0))+'`.']
        if run.get('camera_data_stats'):
            lines += ['', 'Camera dataset raster/target preparation totals (host wall, including first training frame): `'+json.dumps(run['camera_data_stats'])+'`.']
            lines += ['HKS prepared geometry builds/hits: `'+str(run.get('camera_geometry_builds',0))+' / '+str(run.get('camera_geometry_hits',0))+'`.']
            lines += ['Most recent HKS frame workload/selector: `'+json.dumps(run.get('camera_workload'))+'` / `'+json.dumps(run.get('camera_selector_stats'))+'`.']
            lines += ['Kernel-count trajectory: `'+json.dumps(run.get('kernel_count_trajectory',[]))+'`.']
            if run.get('occupancy_stats'):
                lines += ['Mixed evaluator occupancy: `'+json.dumps(run['occupancy_stats'])+'`.']
        if run.get('startup_wall_seconds'):
            lines += ['', 'Startup wall scopes (include CPU work and any waits, not GPU kernel durations):', '',
                      '| Startup scope | Seconds |', '|---|---:|']
            for name, seconds in run['startup_wall_seconds'].items():
                lines.append('| '+name+' | '+str(seconds)+' |')
            lines += ['| Original-loop setup before initial receipt | '+str(run.get('loop_before_receipt_seconds'))+' |',
                      '| Initial receipt and RGB/VJP gates | '+str(run.get('initial_receipt_and_gates_seconds'))+' |']
        if run.get('stage_profile'):
            profile = run['stage_profile']
            lines += ['', profile['scope'], '', '| Stage | Calls | Inclusive wall seconds |', '|---|---:|---:|']
            for name, row in profile['stages'].items():
                lines.append('| '+name+' | '+str(row['calls'])+' | '+str(row['wall_seconds'])+' |')
            lines += ['', 'Sample/triangle reuse: `'+json.dumps(profile['sample_reuse'])+'`.']
    lines += ['', 'The original loop is inherited. Original optimization policies are required; retired custom HKS policies are not supported. Adaptive importance errors, sampled batches, '
        'kernel counts and density decisions can diverge between methods. Training interval starts after initial '
        'receipts/gates and the first batch fetch; includes subsequent sampling, logging and original plots when enabled. '
        'No lit Mitsuba image rendering is timed. Checkpoints are verification archives, not an implemented resume API.', '']
    lines += ['Geodesic warnings count repeated failed-sample events across whole worker logs, not unique kernels. '
        'The original fallback retains failed positions while other updates continue. COMPLETE and finite final '
        'parameters do not imply every geodesic movement succeeded. Diagnostic trace-capture timings are not '
        'performance evidence.', '']
    if 'sources_unchanged' in workflow:
        lines += ['Sources unchanged during workflow: '+str(workflow['sources_unchanged'])+'.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=tuple(_PRESETS),
        help='compare-128 expands the lit spectral-support comparison recipe; explicit options override it')
    assets = sorted(p.stem.removesuffix('_training') for p in
                    (ROOT/'splatting_script/configs').glob('*_training.yaml'))
    parser.add_argument('--asset', choices=assets,
        help='Mesh/config/camera shortcut with --preset; defaults to converse for the preset')
    parser.add_argument('--backend', choices=('both', 'native', 'hybrid'), default='both')
    parser.add_argument('--config', type=Path, default=ROOT/'splatting_script/configs/converse_training.yaml')
    parser.add_argument('--config-overlay', type=Path,
        help='Optional YAML merged after --config for BOTH backends; values and file hash are recorded')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--kernels', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--evaluation-seed', type=int, default=1729)
    parser.add_argument('--initial-state', type=Path, help='A full-surface initial.pt; identity must match, no silent reset')
    parser.add_argument('--shared-initial-model', action='store_true',
        help='Explicitly share native placement; camera mode also shares its initial residual/mean colors; all other state stays strict')
    parser.add_argument('--surface-adapter', choices=('reference', 'optimized', 'aggressive', 'direct'), default='reference',
        help='Use direct for the current GPU camera/sample-tile HKS path; other adapters are retained references. Native forward is unchanged')
    parser.add_argument('--allow-native-approximation',action='store_true',help='Explicit approximate native-field comparison; same-method implementation checks stay strict')
    parser.add_argument('--max-initial-psnr-drop-db',type=float,default=.5,help='Allowed initial validation PSNR drop vs native; not a final-quality guarantee')
    parser.add_argument('--allow-render-variation', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--raster-backend',choices=('cpu','nvdiffrast'),default='cpu',help='Optional fresh CUDA raster; both arms use identical raster policy')
    parser.add_argument('--raster-near',type=float,default=.01,help='Explicit GPU clipping near plane, in mesh units; CPU reference keeps its original plane')
    parser.add_argument('--support-domain', choices=('graph','spectral'), default='graph',
        help='Graph uses approximate mesh hops; spectral uses global pixel KNN and shared triangle coefficients')
    parser.add_argument('--surface-support-hops',type=int,default=8,help='Approximate kernel-seeded edge-hop support; no monotone spectral cutoff')
    parser.add_argument('--occupancy-threshold',type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument('--association-cache',choices=('none',),default='none',help=argparse.SUPPRESS)
    parser.add_argument('--corner-evaluation',choices=('serial','parallel'),default='serial',help='Experimental parallel corner/Gram contraction; existing backward unchanged')
    parser.add_argument('--support-pruning',choices=('none',),default='none',help=argparse.SUPPRESS)
    parser.add_argument('--visibility', choices=('fresh','reuse'), default='fresh', help='Reuse fixed view raster/geometry; rerasterize changed views. Initial work stays timed')
    parser.add_argument('--visibility-cache-mib', type=int, default=64)
    parser.add_argument('--target-sampling', choices=('cpu','gpu'), default='cpu', help='Shared camera target backend; GPU uploads fixed input pixels once and validates exact lookups')
    parser.add_argument('--max-target-texture-mib', type=int, default=512)
    parser.add_argument('--supervision', choices=('surface','camera'), default='surface', help='Camera rasterizes fresh triangles each frame; disables the surface importance pool for both arms')
    parser.add_argument('--lighting', choices=('none','diffuse','ggx'), default='none',
        help='Optional shared analytic distant light; none preserves unlit RGB. Lit targets/loss use linear radiance, not ray tracing')
    parser.add_argument('--lighting-backend', choices=('auto','torch'), default='auto',
        help='Shared shader: auto uses fused CUDA float32/fixed geometry; torch retains the independent tensor reference')
    parser.add_argument('--light-direction', nargs=3, type=float, default=(.3,.8,.6), metavar=('X','Y','Z'), help='World-space surface-to-light direction')
    parser.add_argument('--light-ambient', type=float, default=.1, help='Diffuse linear fill multiplier, not indirect illumination')
    parser.add_argument('--light-intensity', type=float, default=3., help='White distant-light linear intensity')
    parser.add_argument('--material-roughness', type=float, default=.35, help='Fixed GGX perceptual roughness, .05 to 1')
    parser.add_argument('--material-specular', type=float, default=.04, help='Fixed GGX normal-incidence Fresnel F0, 0 to 1')
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--cameras-file', type=Path, help='Optional mesh-specific camera JSON; shared by both methods and recorded in provenance')
    parser.add_argument('--density-policy', choices=('original',), default='original',
        help='Current matched comparison requires original; historical hks policy is rejected')
    parser.add_argument('--optimizer-policy', choices=('original',), default='original',
        help='Current matched comparison requires original; historical hks policy is rejected')
    parser.add_argument('--selector-merge', choices=('reference', 'compact', 'radix', 'row-radix', 'row-topk'), default='reference',
        help=argparse.SUPPRESS)
    parser.add_argument('--no-training-plots', action='store_true',
        help='Disable optional histogram/loss plots for BOTH adapters; original HKTex files stay unchanged')
    parser.add_argument('--profile-start', type=int, default=0)
    parser.add_argument('--profile-steps', type=int, default=0,
        help='Bounded synchronized stage profiling for BOTH methods; zero disables it (diagnostic timings only)')
    parser.add_argument('--verify-samples', type=int, default=32)
    parser.add_argument('--support-cap', type=int, default=0,help='Optional per-face safety cap; zero keeps all support within record/byte limits')
    parser.add_argument('--max-contributions', type=int, default=16777216)
    parser.add_argument('--max-buffer-mib', type=int, default=512)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--trace-diagnostics', action='store_true',
        help='Capture bounded failed geodesic cases and per-step counts; diagnostic timings only')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments'/('full_surface_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.asset and not args.preset:
        parser.error('--asset requires --preset; use --config for a custom workflow')
    if args.preset:
        parser.set_defaults(**_preset_defaults(args.preset, args.asset or 'converse'))
        args = parser.parse_args(argv)
    if args.cameras_file and (args.supervision!='camera' or not args.cameras_file.is_file()):
        parser.error('--cameras-file requires camera supervision and an existing JSON file')
    if (not all(math.isfinite(x) for x in (*args.light_direction,args.light_ambient,args.light_intensity,args.material_roughness,args.material_specular))
            or sum(x*x for x in args.light_direction)<=0 or args.light_ambient<0 or args.light_intensity<0
            or not .05<=args.material_roughness<=1 or not 0<=args.material_specular<=1):
        parser.error('Light/material settings must be finite, direction nonzero, intensities nonnegative, roughness .05..1 and F0 0..1')
    if args.lighting!='none' and args.supervision!='camera':
        parser.error('Lighting requires camera supervision with surface normals and view directions')
    if not 0<=args.max_initial_psnr_drop_db<float('inf'):parser.error('Initial quality bound must be finite and nonnegative')
    if args.surface_support_hops<0:parser.error('Support hops must be nonnegative')
    if args.occupancy_threshold or args.association_cache!='none' or args.support_pruning!='none':
        parser.error('Occupancy routing, persistent association caches and support pruning are retired')
    if args.density_policy!='original' or args.optimizer_policy!='original' or args.allow_render_variation:
        parser.error('HKS comparisons require original policies and strict validation')
    if not 0 < args.raster_near < float('inf'):parser.error('GPU near plane must be finite and positive')
    if args.raster_backend == 'nvdiffrast' and (args.supervision != 'camera' or args.target_sampling != 'gpu' or args.visibility != 'fresh' or args.surface_adapter != 'direct'):
        parser.error('nvdiffrast requires camera supervision, GPU targets, fresh visibility and direct adapter')
    if args.raster_backend=='nvdiffrast' and args.resolution%8:parser.error('CUDA raster resolution must be a multiple of 8')
    if args.visibility_cache_mib < 1: parser.error('Visibility budget must be positive')
    if args.visibility == 'reuse' and args.supervision != 'camera':
        parser.error('Visibility reuse requires camera supervision')
    if args.max_target_texture_mib < 1: parser.error('Target texture budget must be positive')
    if args.target_sampling == 'gpu' and args.supervision != 'camera':
        parser.error('GPU target sampling requires camera supervision')
    if args.resolution < 8: parser.error('Resolution must be >=8')
    if args.selector_merge in ('compact', 'radix', 'row-radix','row-topk') and args.surface_adapter not in ('aggressive', 'direct'):
        parser.error('Optimized selector merge requires --surface-adapter aggressive or direct')
    if args.profile_start < 0 or args.profile_steps < 0: parser.error('Profile start/steps must be nonnegative')
    if args.execute and os.environ.get('TRITON_INTERPRET') == '1':
        parser.error('Unset TRITON_INTERPRET for real CUDA training/verification')
    if any(v is not None and v < 1 for v in (args.iterations, args.kernels, args.verify_samples, args.max_contributions, args.max_buffer_mib)) or not 0 <= args.support_cap <= 512:
        parser.error('Counts/budgets must be positive; support cap must be 0..512 (zero uses bounded chunked reduction)')
    if not args.config.is_file(): parser.error('Configuration file does not exist')
    if args.config_overlay and not args.config_overlay.is_file(): parser.error('Configuration overlay does not exist')
    if args.initial_state and not args.initial_state.is_file(): parser.error('Initial snapshot does not exist')
    if args.worker:
        if not args.execute or args.backend == 'both' or args.run_dir is None: parser.error('Invalid internal worker command')
        return worker(args)
    workflow = plan(args)
    if not args.execute:
        print(json.dumps(workflow, indent=2)); return 0
    gpu_guard()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    workflow['execution_requested'] = True
    workflow['gpu_started'] = None  # Unknown until a worker leaves CUDA execution evidence.
    initial_sources = sources()
    failed = False
    verification_identity = None
    for job in workflow['jobs']:
        if failed:
            job['status'] = 'skipped_after_failure'; continue
        job_start = time.perf_counter()
        with (output/(job['id']+'.log')).open('x') as log:
            completed = subprocess.run(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        job['process_wall_seconds'] = time.perf_counter()-job_start
        job['geodesic_warnings'] = geodesic_log_counts(output/(job['id']+'.log'))
        job['status'] = 'complete' if completed.returncode == 0 else 'failed'
        job['returncode'] = completed.returncode
        if (Path(job['run'])/'manifest.json').is_file() or (Path(job['run'])/'result.json').is_file():
            workflow['gpu_started'] = True
        failed = failed or completed.returncode != 0
        if not failed and job['role'] == 'verification':
            receipt = json.loads((Path(job['run'])/'result.json').read_text())
            identity = (receipt['controls_sha256'], receipt['initial_identity'])
            if verification_identity is not None and identity != verification_identity:
                job['status'] = 'failed_paired_initial_controls'; failed = True
            verification_identity = identity
        write(output/(job['id']+'_receipt.json'), job)
        (output/'comparison.md').write_text(comparison_markdown(workflow))
        print(job['id']+': '+job['status'], flush=True)
    workflow['sources_unchanged'] = initial_sources == sources()
    write(output/'workflow.json', workflow)
    (output/'comparison.md').write_text(comparison_markdown(workflow))
    print('SURFACE_TRAINING_REPORT='+str(output/'comparison.md'), flush=True)
    return int(failed or not workflow['sources_unchanged'])


if __name__ == '__main__':
    raise SystemExit(main())
