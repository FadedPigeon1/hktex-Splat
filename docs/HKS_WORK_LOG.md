# HKS development and validation work log

Session started: 2026-11-21 (America/Chicago).

## Authorization and constraints

The user authorized continued implementation and GPU testing using their existing
interactive Slurm allocation, with work recorded in this Markdown file. No new
allocation will be requested. Original `hktex/` code, research assets/results and
existing staged/unstaged changes must be preserved. No commits or pushes.
The explicit request for this log permits this Markdown file; existing Python
modules and entry points will be reused.

## Acceptance criteria

- Correct finite GPU forward/backward behavior against the same-method eager oracle.
- Distinguish harmless contribution ordering from changed selected membership;
  never waive RGB/VJP discrepancies under the label of a passing correctness test.
- Native and kernel-centric HKS are the only performance competitors.
- Approximation differences are reported separately from implementation errors.
- Short matched native/HKS training and held-out quality before long experiments.
- Original density/optimizers, initialization, cameras, loss and update budgets match.
- Report actual synchronized performance and quality; no promised speedup.
- A product is not declared validated while required checks remain unresolved.

## Starting evidence

Latest GPU evidence: `experiments/kernel_candidate_replay_fixed_20261121_155937/`.
The four-hop support excludes 87.0871% of native selected positive weight on 907
validation pixels. Ordinary HKS GPU/eager RGB/VJPs pass. With identical native
outer candidates, max native-response error is 2.01464e-5, max native RGB error
is 0.0120261, and native kernel-color VJP fails tolerance. Replay GPU/eager VJPs
all pass, weights are close, but ordered top IDs differ. Membership-versus-order
and the replay RGB error require investigation; ordering alone is not evidence
of a different rendered function.

## Actions

1. Inspected `git status --short`: extensive pre-existing staged, unstaged and
   untracked work. No reset, stash, staging or deletion performed.
2. Inspected host and allocation environment using `hostname`, selected Slurm/CUDA
   environment variables and command discovery: host `login5`, no inherited
   `SLURM_JOB_ID`/step/node list/CUDA device mask. `squeue`, `srun`, `nvidia-smi`
   are available. This shell is not the user's interactive compute-node shell.
3. Inspected package metadata and local instruction-file discovery. No `AGENTS.md`
   found in repository. No new Python module or dependency is planned.
4. Queried existing allocations with
   `squeue --user "$USER" --noheader --format='%i|%T|%N|%b|%L'`.
   Result will be recorded below before running anything on a compute node.

## Current status

Continued optimization under existing allocation20651097, no new request.
Retained two verified changes: fused analytic shader/albedo backward and exact
fresh GPU camera/tile packing. Original89native Python files remain identical
to the saved starting tree (not a claim of pristine upstream source).

Latest full5000update lit piggy pairs:128² native/HKS112.27/62.55s updates,
133.83/90.68s workers,33.33/33.19dB;320²516.73/106.84s updates (4.84x),
543.51/233.59s workers (2.33x),34.54/34.71dB. At320²both finish5000kernels;
allocated28.69/1.68GiB. Matched initialization/controls/camera metadata sequences,
original policies, strict HKS/eager RGB/VJPs and source/asset checks pass. Native
field equality remains false, explicitly approximate. One scene/seed/GPU only.

5x full pipeline NOT achieved; no ray-traced comparison implemented. Lighting
was explicitly authorized separately. Histories and failed trials retained.
Final320frozen-checkpoint numeric assertions and previews completed, but Slurm
terminated the diagnostic at its4minute bound during/after final reporting;
clean process exit is not validated (srun143). Both measured full-training
workflows exited normally. Allocation subsequently expired naturally; no user
job cancelled, limit changed, commit/push or file deletion.

## Baseline provenance

Original `hktex/` source digest (89 files): `448848a66bbce87b61e1f11a0ddfacc4fb9186ec74c05aa64e7ac55b40038c68`.
This is the initial working-tree state, including pre-existing changes; it is not a claim of pristine upstream source.

Initial revision: `a61c19caa19cd235f3c188d6606e131c79bee7d7`.

Relevant HKS file hashes before new implementation edits:

```text
780aed23a01a06de983e35f8e65c232099e66b303e22dd62d01019bc9afcfed2  hktex_splatting/training/training_checks.py
532d1568bfa88b106e97a5a3043bc96b0d3d40e1a0736a4c62201ec7537eef70  hktex_splatting/training/surface_adapter.py
22b7436cb1aed05956dbc9d12753350a7b2ed37c73d4777cfd5e94e694b0a9fb  hktex_splatting/knn_heat/triangle_kernel_support.py
7998cccc0dd8c175087128870db9719688b1893d967e0bec66078306b5938f4a  hktex_splatting/rendering/heat_kernel_splatting.py
e527414d5ec3a454503b346e86416d45b4b41121607caf748478a06fc3b24696  hktex_splatting/triton/heat_kernel_splatting.py
108e6303a411dabf752dba42f5e74c66d377ee3e96f1503f40371ee8cb7a9980  splatting_script/experiments/train_surface.py
```

Initial Git status (captured after creating this log):

```text
MM .gitignore
MM README.md
 D Splatting_script/README.md
AD Splatting_script/validation/run_retained_cpu.py
AM data/converse/README.md
A  data/converse/prepared/.gitkeep
A  data/converse/prepared/textures/.gitkeep
A  data/converse/raw/.gitkeep
AD docs/CONSOLIDATION.md
AD docs/CONSOLIDATION_STAGING_PATHS.txt
AM docs/ORIGINAL_HKTEX_README.md
 A docs/RESEARCH_CLEANUP.md
AD docs/RESEARCH_LAYOUT.md
 D docs/research_cleanup/abandoned_renderers_plan.json
 D docs/research_cleanup/plan.json
 D docs/research_cleanup/storage_inventory.json
 D docs/research_cleanup/validation.json
 D docs/research_cleanup/visualization_plan.json
 D docs/three_backends_cleanup/RETAINED_DEPENDENCIES.md
AD experiments/converse_fidelity/README.md
A  experiments/converse_fidelity/cameras.json
AD experiments/converse_fidelity/common.py
AD experiments/converse_fidelity/evaluate_metrics.py
AD experiments/converse_fidelity/render_hktex_baseline.py
AD experiments/converse_fidelity/render_original_mesh.py
AD experiments/converse_fidelity/rois.json
AD experiments/converse_heat_kernel_splatting/AUDIT.md
AD experiments/converse_heat_kernel_splatting/KERNEL_CENTRIC.md
AD experiments/converse_heat_kernel_splatting/PERFORMANCE_AUDIT.md
AD experiments/converse_heat_kernel_splatting/README.md
AD experiments/converse_heat_kernel_splatting/RUN_500.md
AD experiments/converse_heat_kernel_splatting/audit_checkpoint.py
AD experiments/converse_heat_kernel_splatting/benchmark_500.py
AD experiments/converse_heat_kernel_splatting/benchmark_kernel_centric.py
AD experiments/converse_heat_kernel_splatting/benchmarks/README.md
AD experiments/converse_heat_kernel_splatting/benchmarks/forward.py
AD experiments/converse_heat_kernel_splatting/benchmarks/preprocessing.py
AD experiments/converse_heat_kernel_splatting/consolidation/audit_saved.py
AD experiments/converse_heat_kernel_splatting/consolidation/dependencies.json
AD experiments/converse_heat_kernel_splatting/consolidation/inventory.json
AD experiments/converse_heat_kernel_splatting/consolidation/restore.py
AD experiments/converse_heat_kernel_splatting/consolidation/verify.py
AD experiments/converse_heat_kernel_splatting/cpu_native.py
AD experiments/converse_heat_kernel_splatting/docs/REFERENCE_PROTOTYPE_README.md
AD experiments/converse_heat_kernel_splatting/experiment_io.py
AD experiments/converse_heat_kernel_splatting/fast_raster.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/MATHEMATICS.md
AD experiments/converse_heat_kernel_splatting/forward_splatting/README.md
AD experiments/converse_heat_kernel_splatting/forward_splatting/VALIDATION.md
AD experiments/converse_heat_kernel_splatting/forward_splatting/__init__.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/benchmark.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/core.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/gpu.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/renderer.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/report.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/test_forward.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/test_harness.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/validate_compile.py
AD experiments/converse_heat_kernel_splatting/forward_splatting/validate_saved_cpu.py
AD experiments/converse_heat_kernel_splatting/full_report.py
AD experiments/converse_heat_kernel_splatting/kernel_centric.py
AD experiments/converse_heat_kernel_splatting/kernel_centric_gpu.py
AD experiments/converse_heat_kernel_splatting/kernel_centric_raster.py
AD experiments/converse_heat_kernel_splatting/kernel_centric_report.py
AD experiments/converse_heat_kernel_splatting/optimized_kernel_splat.py
AD experiments/converse_heat_kernel_splatting/preprocessing/README.md
AD experiments/converse_heat_kernel_splatting/preprocessing/__init__.py
AD experiments/converse_heat_kernel_splatting/preprocessing/audit_saved.py
AD experiments/converse_heat_kernel_splatting/preprocessing/batched.py
AD experiments/converse_heat_kernel_splatting/preprocessing/benchmark.py
AD experiments/converse_heat_kernel_splatting/preprocessing/lightweight.py
AD experiments/converse_heat_kernel_splatting/preprocessing/profiling.py
AD experiments/converse_heat_kernel_splatting/preprocessing/report.py
AD experiments/converse_heat_kernel_splatting/preprocessing/test_preprocessing.py
AD experiments/converse_heat_kernel_splatting/rebuild_report.py
AD experiments/converse_heat_kernel_splatting/recovery/MEASURED_RESULTS.md
AD experiments/converse_heat_kernel_splatting/recovery/README.md
AD experiments/converse_heat_kernel_splatting/recovery/__init__.py
AD experiments/converse_heat_kernel_splatting/recovery/fixed_benchmark.py
AD experiments/converse_heat_kernel_splatting/recovery/recover.py
AD experiments/converse_heat_kernel_splatting/recovery/test_recovery.py
AD experiments/converse_heat_kernel_splatting/surface_kernel_splat.py
AD experiments/converse_heat_kernel_splatting/test_kernel_centric.py
AD experiments/converse_heat_kernel_splatting/test_optimized.py
AD experiments/converse_heat_kernel_splatting/test_reference.py
AD experiments/converse_heat_kernel_splatting/tests/README.md
AD experiments/converse_heat_kernel_splatting/tests/preserved_suite.py
AD experiments/converse_heat_kernel_splatting/tests/run_cpu.py
AD experiments/converse_heat_kernel_splatting/tests/test_consolidation.py
AD experiments/converse_heat_kernel_splatting/validate_gpu.py
AD experiments/converse_heat_kernel_splatting/validate_optimized_cpu.py
AD experiments/converse_heat_kernel_splatting/validation_report.py
AD experiments/converse_renderer_comparison/README.md
AD experiments/converse_renderer_comparison/adaptive.py
AD experiments/converse_renderer_comparison/adaptive_report.py
AD experiments/converse_renderer_comparison/allocation.py
AD experiments/converse_renderer_comparison/benchmark.py
AD experiments/converse_renderer_comparison/checkpoint_metadata.py
AD experiments/converse_renderer_comparison/configs/benchmark_config.json
AD experiments/converse_renderer_comparison/quality.py
AD experiments/converse_renderer_comparison/renderers.py
AD experiments/surface_splatting/README.md
AD experiments/surface_splatting/optimization/assets.py
AD experiments/surface_splatting/optimization/benchmark_backends.py
AD experiments/surface_splatting/optimization/color_atlas.py
AD experiments/surface_splatting/optimization/live_inference/appearance.py
AD experiments/surface_splatting/optimization/live_inference/benchmark.py
AD experiments/surface_splatting/optimization/live_inference/consistent_knn.py
AD experiments/surface_splatting/optimization/live_inference/coverage.py
AD experiments/surface_splatting/optimization/live_inference/geometry.py
AD experiments/surface_splatting/optimization/live_inference/runtime.py
AD experiments/surface_splatting/optimization/resident_backend.py
AD experiments/surface_splatting/optimization/results/gpu_benchmark_/20261020T015737_846857Z/case_0/fresh_gpu/runtime_cache/triton/L2UUZSXBNKUGG3DJFPM5GFSVTGAPY2ZY26KVVVPZ7JD3P4F7SSFQ/triton_poi_fused_div_exp_mul_neg_0.ttgir
AD experiments/surface_splatting/optimization/torch_backend.py
 M hktex/modules/tracer.py
 A hktex_splatting/README.md
AD hktex_splatting/modules/optimized_kernel_splat.py
AD hktex_splatting/modules/surface_kernel_splat.py
 D hktex_splatting/rendering/frame.py
AD hktex_splatting/rendering/kernel_centric.py
AD hktex_splatting/rendering/renderer.py
AD hktex_splatting/triton/kernel_centric_gpu.py
AD hktex_splatting/utils/_legacy_paths.py
 D scripts/benchmark_image_gs/README.md
 D scripts/benchmark_image_gs/evaluate_image_gs.py
 D scripts/benchmark_image_gs/for_hktex_comparison.patch
 D scripts/benchmark_image_gs/image_gs_fitting.py
 D scripts/compare_triangle_splat_backends.py
 D scripts/extract_video_frames.py
 D scripts/find_common_valid_files.py
 D scripts/gather_benchmark_videos.py
AD scripts/surface_splatting.py
 D scripts/test_numba_raster.py
 D scripts/test_renderers.py
 D scripts/test_triangle_splat_backends.py
MD scripts/triangle_splat_renderer.py
 D scripts/visualise_all_kernels.py
 D scripts/visualise_bh_weights.py
 D scripts/visualise_eigenvectors.py
 D scripts/visualise_hktex_steps.py
 D scripts/visualise_kernel_split.py
 D scripts/visualise_local_frames.py
 D scripts/visualise_single_kernel.py
 D scripts/visualize_kernel_faces.py
?? configs/texture_hktex_knn_converse_alpha30.yaml
?? configs/texture_hktex_knn_converse_hr_alpha30.yaml
?? docs/BENCHMARK_PROFILER.md
?? docs/CAMERA_RASTER_IMPORT_FIX.md
?? docs/COMPACT_SELECTOR_MERGE.md
?? docs/CUSTOM_HKS_TRAINING.md
?? docs/DIRECT_SURFACE_ADAPTER.md
?? docs/DIRECT_TILES.md
?? docs/DIRECT_TILES_VALIDATION_FIX.md
?? docs/FINAL_CLEANUP_REPORT.md
?? docs/FULL_SURFACE_TRAINING.md
?? docs/GEODESIC_TRACE_AUDIT.md
?? docs/GPU_CAMERA_PIPELINE.md
?? docs/GPU_TARGET_SAMPLING.md
?? docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md
?? docs/HKS_AGGRESSIVE_OPTIMIZATION.md
?? docs/HKS_FILE_RENAME_REPORT.md
?? docs/HKS_OPTIMIZATION_REPORT.md
?? docs/HKS_PIPELINE_ONLY_CLEANUP.md
?? docs/HKS_SHARED_MODULE_RENAME.md
?? docs/HKS_UNUSED_CODE_CLEANUP.md
?? docs/HKS_WORK_LOG.md
?? docs/HKTEX_RESEARCH_TRAINING.md
?? docs/HKTEX_VS_HKS_BENCHMARK.md
?? docs/NATIVE_ADAM_REPLAY_ANALYSIS.md
?? docs/ORIGINAL_RENDER_VS_HKS.md
?? docs/PREFLIGHT_LOSS_MODE_FIX.md
?? docs/PROFILE_VALIDATION_FIX.md
?? docs/RADIX_AND_STARTUP_OPTIMIZATION.md
?? docs/RASTERIZED_CAMERA_TRAINING.md
?? docs/ROW_SELECTOR_AND_TEXTURE_CACHE.md
?? docs/SPARSE_ASSOCIATION.md
?? docs/SPECTRAL_DISTANCE_STUDY.md
?? docs/STATIC_CAMERA_VISIBILITY.md
?? docs/TEST_CLEANUP_REPORT.md
?? docs/THREE_BACKENDS_CONSOLIDATION.md
?? docs/TILED_SELECTOR.md
?? docs/hks_pipeline_only/
?? docs/hktex_vs_hks_benchmark/
?? experiments/aggressive_profile_20261120_182046/
?? experiments/aggressive_verify_1000_20261120_180609/
?? experiments/aggressive_verify_5000_20261120_181541/
?? experiments/association_smoke_20261121_135254/
?? experiments/blocked_distance_full_20261120_231000/
?? experiments/blocked_distance_verify_20261120_225647/
?? experiments/blocked_distance_verify_20261120_225804/
?? experiments/compact_merge_verify_20261120_184209/
?? experiments/custom_hks_full_20261120_220334/
?? experiments/custom_hks_smoke_20261120_214229/
?? experiments/custom_hks_smoke_retry_20261120_214905/
?? experiments/direct_adapter_full_20261120_211416/
?? experiments/direct_adapter_verify_20261120_210106/
?? experiments/direct_render_profile_20261120_222405/
?? experiments/distance_layouts_20261120_225112.md
?? experiments/distance_layouts_reverse_20261120_225306.md
?? experiments/full_training_compact_20261120_185110/
?? experiments/full_training_compact_20261120_191324/
?? experiments/full_training_radix_20261120_200047/
?? experiments/gpu_full_20261121_021837/
?? experiments/gpu_full_20261121_022432/
?? experiments/hks_density_smoke_20261121_120536/
?? experiments/hks_only_profile_20261121_114205/
?? experiments/hktex_vs_custom_hks_20261121_121523/
?? experiments/hktex_vs_custom_hks_256_20261121_123114/
?? experiments/hktex_vs_hks_128_diagnostic_20261121_130406/
?? experiments/hktex_vs_hks_256_diagnostic_20261121_125141/
?? experiments/kernel_candidate_replay_20261121_155129/
?? experiments/kernel_candidate_replay_fixed_20261121_155937/
?? experiments/kernel_centric_verify_64_20261121_151430/
?? experiments/kernel_centric_verify_64_20261121_153321/
?? experiments/optimized_camera_64_20261121_104112/
?? experiments/original_render_smoke_20261121_110757/
?? experiments/original_render_smoke_20261121_111413/
?? experiments/radix_startup_verify_20261120_195129/
?? experiments/raster_camera_diagnostic_20261121_001441/
?? experiments/raster_camera_diagnostic_smoke_20261121_002730/
?? experiments/raster_camera_smoke_20261121_000024/
?? experiments/raster_gpu_targets_20261121_005748/
?? experiments/row_selector_verify_20261120_205032/
?? experiments/surface_boundary_stop_320_new/
?? experiments/surface_boundary_stop_shared_1001/
?? experiments/surface_boundary_stop_shared_320/
?? experiments/surface_boundary_stop_shared_verify/
?? experiments/surface_boundary_stop_verify_retry/
?? experiments/surface_density_check_new/
?? experiments/surface_full_5000_20261120_161953/
?? experiments/surface_optimized_profile/
?? experiments/surface_optimized_profile_20261120_173732/
?? experiments/surface_optimized_verify/
?? experiments/surface_smoke_new/
?? experiments/surface_verify_new/
?? experiments/training_bottleneck_20261121_112201/
?? experiments/training_topk_parallel_20261121_114049/
?? hktex_splatting/__init__.py
?? hktex_splatting/__main__.py
?? hktex_splatting/cli.py
?? hktex_splatting/knn_heat/
?? hktex_splatting/modules/__init__.py
?? hktex_splatting/modules/spectral_heat_coefficients.py
?? hktex_splatting/modules/spectral_heat_field.py
?? hktex_splatting/prepare_assets.py
?? hktex_splatting/rendering/__init__.py
?? hktex_splatting/rendering/camera_raster.py
?? hktex_splatting/rendering/cameras.py
?? hktex_splatting/rendering/cuda_mesh_raster.py
?? hktex_splatting/rendering/device_camera.py
?? hktex_splatting/rendering/fast_raster.py
?? hktex_splatting/rendering/gpu_dispatch.py
?? hktex_splatting/rendering/heat_kernel_splatting.py
?? hktex_splatting/rendering/mesh_visibility_raster.py
?? hktex_splatting/rendering/native_reference.py
?? hktex_splatting/rendering/prepared_geometry.py
?? hktex_splatting/rendering/reference_mesh.py
?? hktex_splatting/rendering/resident.py
?? hktex_splatting/rendering/splat_autograd.py
?? hktex_splatting/rendering/triangle_tile_layout.py
?? hktex_splatting/training/
?? hktex_splatting/triton/__init__.py
?? hktex_splatting/triton/heat_kernel_backward.py
?? hktex_splatting/triton/heat_kernel_splatting.py
?? hktex_splatting/triton/resident_gpu.py
?? hktex_splatting/triton/spectral_heat_coefficients.py
?? hktex_splatting/triton/spectral_support_distances.py
?? hktex_splatting/triton/splat_contributions.py
?? hktex_splatting/triton/splat_reconstruction.py
?? hktex_splatting/utils/__init__.py
?? hktex_splatting/utils/compiler_runtime.py
?? hktex_splatting/utils/experiment_io.py
?? hktex_splatting/utils/metrics.py
?? hktex_splatting/utils/reference_assets.py
?? hktex_splatting/utils/support_debug.py
?? hktex_splatting/utils/training_profile.py
?? pyproject.toml
?? scripts/inspect_converse.py
?? scripts/setup_converse_experiment.py
?? splatting_script/
```

## Allocation access and contribution comparison

- Sandboxed Slurm query failed: `Unable to contact slurm controller (connect failure)`.
- Retried with explicit outside-sandbox Slurm access. Existing job found:
  `20644106|RUNNING|g051|gres/gpu:a100:1|1:57:34`.
- Started a five-minute-bounded CUDA probe step with `srun --jobid=20644106
  --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
  --time=00:05:00 env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib
  /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -c ...`.
  Probe prints host, CUDA availability, GPU type, free/total memory and Torch version;
  no model/training computation. Result pending below.
- Added source-ID canonicalization for replay contribution comparisons in
  `training_checks.py`. A rank permutation is accepted only when exact selected
  membership, per-source weights, padding, RGB and all VJPs pass their existing
  checks. Different membership is never excused. Original ordered-ID equality
  remains reported. Added membership-row count and replay GPU/eager RGB error.
  No production heat calculation or source support changed in this edit.

### GPU probe result and diagnostic 01

Probe succeeded: host g051; CUDA=True; NVIDIA A100-PCIE-40GB; free/total
bytes 41959292928/42404806656; Torch 2.10.0+cu129. This verifies compute-node
access inside the existing allocation, not a new allocation.

Read-only step listing used an unsupported `%T` step-format field; Slurm printed
a format warning but showed steps .0 (interactive shell), .1 (probe), extern.
No job/step was canceled or changed.

CPU canonical contribution checks passed for rank permutation, changed
membership, mismatched source-weight assignments and nonzero/valid-ID padding.
All HKS/script source syntax passed. `git diff --check` passed.
Updated existing comparison Markdown generation to include replay RGB error and
canonical contribution diagnostics.

Next GPU action: one HKS-only validation workflow (not a performance competitor),
reusing the last paired native initial state, 907/1024 bounded held-out samples,
no training updates. Full command:

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 \
  --output=experiments/hks_work_session_20261121/diag01_step.log \
  --error=experiments/hks_work_session_20261121/diag01_step.log \
  env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
  /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B \
  -m splatting_script.experiments.train_surface \
  --backend hybrid --supervision camera --resolution 64 \
  --kernels 5000 --iterations 40 --seed 0 \
  --raster-backend nvdiffrast --target-sampling gpu \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --initial-state experiments/kernel_candidate_replay_fixed_20261121_155937/native_verify/initial.pt \
  --shared-initial-model --surface-adapter direct \
  --surface-support-hops 4 --support-cap 512 \
  --density-policy original --optimizer-policy original \
  --no-training-plots --verify-samples 1024 --verify-only --execute \
  --output experiments/hks_work_session_20261121/diag01
```

No source edits will be made while this source-provenance-checked worker runs.

### Diagnostic 01 progress

- GPU probe step exited successfully (code 0).
- CPU contribution comparison test executed the new helper from source and passed:
  equal source/weight pairs with rank permutations; changed membership rejected;
  swapped per-source weights rejected; valid-ID/nonzero tail padding rejected.
  All HKS/script syntax passed.
- Read-only source search initially referenced nonexistent `hktex/utils/functions.py`;
  corrected search located `rescaled_soft_step` in `hktex/utils/misc.py:232`. No edit
  to original HKTex. Torch sigmoid/rescaling and Triton exp/division rounding are
  a possible remaining selection issue, not yet a confirmed cause.
- Ran a read-only process inspection inside the existing allocation:
  `srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1
  --cpus-per-task=1 --gres=none --time=00:02:00 /usr/bin/ps
  -u franciscotorres2124 -o pid,ppid,etime,pcpu,pmem,comm --sort=-pcpu`.
  It showed workflow PID 1746656 and worker PID 1746666 active, about 2:30 elapsed;
  worker CPU 7.3%, memory 0.5% of node. Existing interactive shell remained intact.
- Worker log currently contains the known Mitsuba LLVM initialization warning.
  No traceback or validation result yet; no speed/quality conclusion drawn.

### Diagnostic 01 result: rejected initial controls (not a rendering result)

GPU step exited 1. Existing `check_initial_controls` rejected the old initial
snapshot because current source/control provenance differs. The worker reached
model setup but did not run candidate replay or training. Evidence:
`experiments/hks_work_session_20261121/diag01/hybrid_verify/failure.json`.
No provenance or initialization checks were removed.

Diagnostic 02 repeats the same bounded Slurm command and flags, except:
`--backend both`, remove `--initial-state ...`, output root
`experiments/hks_work_session_20261121/diag02`, step output/error
`experiments/hks_work_session_20261121/diag02_step.log`. This creates a fresh native
initial snapshot with current source provenance, then shares it with HKS. Both
jobs are verification-only. No source changes between the two workers.

### Diagnostic 02 result: GPU implementation passes; native field differs

Fresh native verification passed. HKS worker ran its GPU checks, then exited 1
only on `native_field_equivalence`. Full evidence in
`experiments/hks_work_session_20261121/diag02/hybrid_verify/sample_validation.json`.
Replay candidate IDs match exactly. Ordered top IDs differ, but membership is
identical for all 907 rows; per-source weights pass (max difference 3.33786e-6).
GPU/eager RGB max error is 1.19209e-7; all seven parameter VJPs pass. Thus the
old replay rejection was a rank-order comparison error, not changed contributions.
Native replay RGB max difference remains 0.0120261, response difference 2.01464e-5;
color VJP differs. Four-hop production support still omits 87.0871% of native
selected weight. No claim of native-field equivalence is made.

### Next change: explicit approximate quality experiment

Keep strict mode as default. Add opt-in `--allow-native-approximation` with an
explicit `--max-initial-psnr-drop-db` bound (0.5 dB default experiment setting).
This accepts no failed same-method GPU/eager RGB, membership, weights or VJPs.
It permits a different finite native field only when initial target quality is
within the requested bound. Initial sampled quality does not prove final/held-out
quality; paired 40-update unprofiled runs will measure it before longer runs.
Original density/optimizers, kernel initialization, camera sequence and source
provenance checks remain. Approximation acceptance is recorded separately from
strict equivalence; no strict failures are relabeled as passes.

### Exact fixed-geometry support lookup optimization

The fixed-hop relation depends only on mesh connectivity, not on heat, camera,
source coordinates within a face or learned shape. Added a bounded precomputed
face-to-hop-neighbor table to `KernelSurfaceSupport` using existing SciPy CSR.
Rows are sorted; current source-face lookup reconstructs source-major pairs each
forward, followed by fresh visibility intersection. This removes repeated
frontier sort/unique work without changing the support relation. No learned
coefficients, lighting, colors or visibility are cached. Mesh/table mutations
invalidate geometry; source-face/count changes are immediately reflected.
A degree-based worst-case bound limits table allocation to half the configured
byte budget (including adjacency); unsupported/high-memory cases retain fresh
BFS. No kernel primitive/equation, top-30 behavior or backward operator changed.
CPU oracle tests and the next strict GPU gates must validate this optimization.

Approximation-policy CPU tests passed: explicit acceptance; disabled acceptance;
GPU operator failure rejection; replay failure rejection; quality-bound rejection;
nonfinite native rejection. Native equivalence remains `passed=false` even when
an explicit approximation experiment is accepted. Paired preview/flag propagation
and all source syntax passed. HKS result labels now use `APPROXIMATE_*` when
requested, with separate approximation/implementation acceptance fields.

Fixed-geometry lookup CPU test initially hit an environment loader error:
SciPy/NumPy required `GLIBCXX_3.4.29` but the login-shell system libstdc++ was
selected. No source algorithm executed past the SciPy import. Rerun uses
`env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib`,
the same environment already used for successful GPU steps. No dependency was
installed/upgraded and no system library was changed. Final source syntax and
`git diff --check` passed.

### Short paired unprofiled 64x64 experiment

Fixed-hop lookup CPU tests passed against both fresh frontier traversal and an
independent BFS oracle for hops 0/1/2/4, nonmanifold/disconnected faces, changed
source faces/counts, ordering and table-mutation invalidation. This was the rerun
with the correct Conda library path.

Next GPU action: fresh paired verification followed, only if accepted, by 40
updates per method. Same original density/optimizer policies; no stage profiling
or trace diagnostics; explicit approximation experiment with initial 0.5 dB bound.
The purpose is to measure training/render time and FINAL held-out quality rather
than infer usefulness from initial field parity. No source edits during workers.

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:25:00 --output=experiments/hks_work_session_20261121/smoke64_step.log --error=experiments/hks_work_session_20261121/smoke64_step.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --resolution 64 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 4 --support-cap 512 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121/smoke64
```

### Short 64x64 experiment: capacity failure exposed before update zero

Fresh native and HKS verification succeeded (HKS labeled APPROXIMATE_VERIFIED;
strict operator checks true, native equivalence still false). Native completed
40 updates. HKS failed before its first update because the first training view
had a face with more than the explicit 512 contributor safety cap. No silent
truncation occurred. Files: `smoke64/comparison.md`,
`smoke64/hybrid_train/failure.json`. Native interval 2.29192 s; forward-only median
6.58856 ms; held-out PSNR 19.5480 dB. There is NO paired speed result from this run.
GPU verified static hop lookup is active (12,900,000 bytes) and reproduces support.

### Capacity correction

Kernel-centric default per-face cap becomes zero (uncapped), while total pair,
fragment, task, int32 and byte-budget checks remain. Explicit positive caps still
raise errors. Original triangle-selector/reference max-512 checks remain by
default; only kernel-centric packing opts out of that selector-specific limit.
For more than 512 entries/pixel, reuse the existing GPU merge with a runtime
record origin and a 512-record bound. Merge the previous top 30 with each next
chunk, keeping the original exact float-weight/global-entry-ID key. Only pixels
with remaining records are processed in subsequent chunks. Mathematical top-k
associativity guarantees the same global strongest 30 including stable ties.
No heat/filter/color/backward equation changed. Need CPU/GPU correctness at
0/1/30/31/511/512/513/1300/5000 records and a fresh paired smoke before longer runs.

### Capacity GPU checks passed

Two short correctness steps completed inside the existing allocation (no
performance claims, no training). Each used the same Slurm prefix/environment,
an eight-minute step bound, and `python -B -c` with in-memory test code; no new
Python module or test script was written.

- `chunk_check.log`: exact CUDA merge vs stable Torch top-30 for record counts
  0/1/30/31/32/511/512/513/1300/5000, random weights, equal weights and all-zero
  weights. IDs and weights matched exactly in every case.
- `chunk_renderer_check.log`: actual DirectTileRenderer CUDA forward/custom
  backward vs existing eager renderer, synthetic coefficients and 600 sources
  on one face. Selected sources beyond entry 512 were exercised. Membership
  matched; RGB max error 2.98023e-8. Five parameter VJP max errors:
  [3.72529e-9, 2.23517e-8, 0, 2.32831e-9, 9.31323e-10]. All passed unchanged
  tolerances. A scalar-reporting requires-grad warning occurred in the inline
  test only; all gradient comparisons completed successfully.
- Zero-cap CLI preview/source syntax and diff checks passed. A CPU test additionally
  checks explicit positive caps and legacy limits remain active.

Allocation query after these checks: 20644106 running on g051, 1:26:29 remaining.

### Short paired 64x64 retry

Same paired unprofiled 40-update protocol, with support-cap zero to retain every
pair within global record/byte limits. Fresh initialization/provenance; no old
checkpoint control reuse. Full command:

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:25:00 --output=experiments/hks_work_session_20261121/smoke64_retry_step.log --error=experiments/hks_work_session_20261121/smoke64_retry_step.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --resolution 64 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 4 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121/smoke64_retry
```

### 64x64 retry outcome: functional, quality/speed target not met

All four jobs completed. Both same-method GPU/eager implementation gates passed;
HKS was explicitly APPROXIMATE_COMPLETE. Source provenance and optimizer/fixed
tensor checks passed. Both ended with 5000 kernels; no density event at 40 steps.

| Metric | Native | Kernel-centric HKS |
|---|---:|---:|
| 40-update training interval s | 0.633563 | 5.031967 |
| Worker body s | 65.4394 | 79.5476 |
| Held-out forward-only median ms | 6.58900 | 5.46488 |
| Held-out deployed PSNR dB | 19.55149 | 17.40934 |
| Peak allocated bytes | 5061141504 | 2387240960 |
| Peak reserved bytes | 12150898688 | 12178161664 |
| Median host step, steps 8 onward s | 0.0102917 | 0.0113486 |

Evidence: `smoke64_retry/comparison.md`, both `*_train/result.json`, and raw
`trajectory.jsonl`. The host-step statistic is diagnostic and is not a complete
synchronized training latency. Cold compilation contributes to this short-run
interval; no measured training speedup or equal-quality speedup is established.
HKS common native-field PSNR is 17.16561 dB, not the deployed HKS metric. The
lower HKS quality fails the provisional final 0.5 dB comparison target.

CPU safety tests also passed: complete 600-entry support at zero cap, positive
cap error, unchanged legacy 512 limit, and global record-budget rejection.

### Eight-hop quality/cost diagnostic

The four-hop relation misses substantial native contribution weight and trained
quality is worse. Test the existing eight-hop parameter (same algorithm), which
the fixed-geometry lookup can precompute when bounded. Profile only 8 updates
after step 8. This run is diagnostic, not a performance result. Keep all shared
training controls and initial quality bound unchanged. Full command:

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:25:00 --output=experiments/hks_work_session_20261121/support8_profile64_step.log --error=experiments/hks_work_session_20261121/support8_profile64_step.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --resolution 64 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --profile-start 8 --profile-steps 8 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121/support8_profile64
```

### Exact inline GPU regression programs used above

These are historical test bodies executed with `python -B -c` inside job
20644106, using the Slurm prefix and Conda library environment already recorded.
They create no repository Python modules. Preserve existing logs when reproducing
in another allocation; select fresh output paths and the current allocated job ID.

<details><summary>Chunk merge regression body</summary>

```python
import torch,triton
from hktex_splatting.triton.splat_contributions import merge
torch.manual_seed(7)
counts=torch.tensor([0,1,30,31,32,511,512,513,1300,5000],device="cuda",dtype=torch.long)
offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
total=int(offsets[-1]);p=len(counts)
entries=torch.arange(total,device="cuda",dtype=torch.int32)
identity=entries.long()
for kind in ("random","ties","zeros"):
 weights=torch.rand(total,device="cuda") if kind=="random" else torch.ones(total,device="cuda") if kind=="ties" else torch.zeros(total,device="cuda")
 selected=torch.full((p,32),2147483647,device="cuda",dtype=torch.int32)
 picked=weights.new_zeros((p,32))
 for start in range(0,int(counts.max()),512):
  active=torch.where(counts>start)[0].int() if start else torch.arange(p,device="cuda",dtype=torch.int32)
  merge[(len(active),)](offsets,identity,entries,weights,active,selected,picked,p,1024,30,32,start,512,num_warps=4,enable_fp_fusion=False)
 expected=selected.clone().fill_(2147483647);expected_weight=picked.clone().zero_()
 for row,count in enumerate(counts.cpu().tolist()):
  begin=int(offsets[row]);order=weights[begin:begin+count].argsort(descending=True,stable=True)[:30]
  expected[row,:len(order)]=(order+begin).int();expected_weight[row,:len(order)]=weights[begin+order]
 torch.cuda.synchronize()
 assert torch.equal(selected,expected),(kind,selected,expected)
 assert torch.equal(picked,expected_weight)
 print("PASS CUDA chunked top30 vs stable Torch:",kind,flush=True)
print("All record counts 0/1/30/31/32/511/512/513/1300/5000 passed.",flush=True)
```

</details>

<details><summary>Public CUDA renderer/backward regression body</summary>

```python
import torch,numpy as np
from types import SimpleNamespace as NS
from hktex_splatting.rendering.heat_kernel_splatting import DirectTileRenderer,TileSurface
torch.manual_seed(13)
device="cuda"
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.N_sources=600;self.kernel_face_ids=torch.zeros(600,device=device,dtype=torch.long)
  self.cfg=NS(power_diffused_diracs=1,knn_inner_k=30)
  heat=torch.rand(600,3,device=device)*.5;heat[580:]+=.35
  for name,value in [("heat",heat),("kernel_colours",torch.rand(600,3,device=device)*.2),("_mean_colour",torch.ones(3,device=device)*.2),("thresholds",torch.ones(600,device=device)*.4),("sharpnesses",torch.ones(600,device=device)*2)]:
   setattr(self,name,torch.nn.Parameter(value))
 def forward(self,x):return x
 def kernel_filter_func(self,x,epsilon,sharpness):
  y=torch.sigmoid(sharpness*(x-epsilon));y0=torch.sigmoid(-sharpness*epsilon);y1=torch.sigmoid(sharpness*(1-epsilon))
  return ((y-y0)/(y1-y0+1e-8)).clamp(0,1)
m=Model();faces=torch.tensor([[0,1,2]],device=device);embedding=torch.rand(3,2,device=device)
ids=np.array([[0,0],[0,-1]],dtype=np.int64);bary=np.zeros((2,2,3),np.float32);bary[0,0]=[1,0,0];bary[0,1]=[0,1,0];bary[1,0]=[.2,.3,.5]
surface=TileSurface(faces,embedding,ids,bary)
e=NS(_iso_embeddings=embedding,_eigen_vec_knn=torch.zeros(1,device=device),use_weighting=True,
 faiss_index=NS(config=NS(metric="l2",index_type="flat",use_float16=False)),
 heat_weighting=NS(config=NS(kernel="gaussian",normalize=False,gaussian_std=.05,eps=1e-8),compute=lambda d:torch.exp(-(d+1e-8)/(.005))))
field=NS(model=m,mesh=NS(faces=faces,verts=torch.rand(3,3,device=device)),eigen=e)
field.sources=lambda t:NS(embedding=m.heat[:,:2],normalizer=torch.ones(600,device=device),phi=m.heat)
field.coefficients=lambda mapping,state,topology:(m.heat[mapping.kernels.long()],torch.zeros(len(mapping.kernels),6,device=device))
renderer=DirectTileRenderer(field,surface,splat_backend="triton",support_cap=0)
image,tape=renderer(contributions=True)
a=torch.autograd.grad(image.sum(),tuple(m.parameters()))
reference,control=renderer(contributions=True,oracle=True)
b=torch.autograd.grad(reference.sum(),tuple(m.parameters()))
torch.cuda.synchronize()
assert torch.allclose(image,reference,atol=3e-5,rtol=3e-4),(image,reference)
assert int(tape["top"].max())>=512
assert torch.equal(tape["top"][:,:30].sort(1).values,control["top"].sort(1).values)
assert all(torch.allclose(x,y,atol=3e-5,rtol=3e-4) for x,y in zip(a,b)),[float((x-y).abs().max()) for x,y in zip(a,b)]
print("PASS actual CUDA renderer >512 contributors, exact selected membership across chunks, RGB and five backward VJPs vs eager control",flush=True)
print("RGB_MAX_ABS="+str(float((image-reference).abs().max())),flush=True)
print("VJP_MAX_ABS="+str([float((x-y).abs().max()) for x,y in zip(a,b)]),flush=True)
```

</details>


### Eight-hop profiled diagnostic result

All four jobs completed, both strict implementation gates true, original
training policies/5000 final kernels retained. Held-out PSNR: native 19.55682 dB,
HKS 19.82542 dB (+0.26860 dB). This meets the provisional 0.5 dB short-run final
quality condition on this view. Initial HKS target quality was 1.20066 dB above
native; native selected weight missing from graph support fell from 87.09% to
59.54%. These fields are different; native equivalence is NOT claimed.

Unprofiled held-out forward-only medians within the completed workers: native
6.60700 ms, HKS 6.16147 ms. Training interval native 2.65656 s/HKS 5.94390 s is
PROFILED and includes cold compilation; it cannot establish a speed ratio.
Allocated peak native 5061141504/HKS 2466390016 bytes. Static geometry table
69,300,000 bytes.

Per-call synchronized inclusive profile ms (nested rows overlap): native
prepare_knn 1.5984, prepare_batch 5.6315, forward .9387, backward 4.1499; HKS
prepare_knn .4239, prepare_batch .0663, forward 7.4642, backward 1.8574. HKS
forward subsets: sample geometry 1.5165, source state 1.3482, support 1.6297
(discovery .7023/intersection .7760), coefficients .5654, tile queue .9337, splat
.4974. Common raster/target ~1.9 ms and optimizers ~2.1 ms. Never sum nested rows.
This points to geometry/support/Python scheduling, not the emitter alone.

Selected eight hops as the provisional main default, replacing the measured
poor-quality four-hop default. Updated current README command; still explicitly
approximate, no general radius or speed guarantee. Corrected verification-only
memory scope text (its peaks include gates, not a nonexistent training interval).
No measured numerical operator changed in this default/documentation edit.

Next: existing `splatting_script.benchmarks.camera_scaling` entry point at 128 and
256, 40 updates, unprofiled, original policies, fresh shared initialization at
each resolution. Native 256 may exceed this A100's 40 GiB; record OOM rather
than modify native math or claim a speed ratio on a failed baseline.

Full scaling command (no new source edits while workers run):

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:30:00 --output=experiments/hks_work_session_20261121/scaling_step.log --error=experiments/hks_work_session_20261121/scaling_step.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.benchmarks.camera_scaling --resolutions 128 256 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121/scaling
```

### Resolution sweep stopped correctly on one 128x128 operator pixel

Native 128 verification passed. HKS 128 operator RGB failed at one of 1024
samples, max 0.001568675; all seven VJPs passed; corner heat max error
3.57628e-7; Gram matrices identical. Native-candidate replay GPU/eager membership,
weights, RGB (max 1.19209e-7), and all VJPs passed. Approximation quality was
1.66555 dB initially above native, but the independent operator RGB failure
correctly prevented acceptance/training. The existing sweep stopped; 256 has
not run. No GPU/eager tolerance was loosened.
Evidence: `scaling/128/hybrid_verify/sample_validation.json` and saved
`render_gate_failure.pt`.

At failing sample 598, GPU selects IDs 4344/4523 where eager selects 3551/4280.
Weights near the cutoff differ by float32 ULPs around 0.9999995. Responses for
3551 and 4280 are identical across saved controls; responses for 4344/4523 differ
by ~6e-8. Thus inspect filter/emit arithmetic, not only spectral coefficient
rounding. Added filtered-record weights to TRACE output only (no training math
change), with restoration to original sample order.

Started an eight-minute-bounded GPU single-pixel replay inside the same allocation,
using `python -B -c` (no new module). Fixed saved heat/Gram and barycentric inputs;
compare CUDA to Torch with identical coefficients, then Torch with saved eager
coefficients. This isolates emit/filter rounding from coefficient evaluation.
Output: `experiments/hks_work_session_20261121/pixel598.log`.

### Pixel replay isolates filter arithmetic; precision correction

Fixed-input replay reproduced the original RGB and selected IDs exactly. With
identical coefficients, GPU vs Torch still differs by 0.001568675 RGB. Native
saved eager coefficients give the same Torch selection, ruling out coefficient
rounding as this pixel's cause. For kernels 4344 and 4523, identical raw response
inputs produce GPU weight 0.9999995827674866 vs Torch 0.9999995231628418. These
one-ULP differences displace 3551/4280 at the top-30 boundary. Full output in
`pixel598.log`.

Changed only the main Triton filter's numerical implementation: CUDA libdevice
exp and round-to-nearest division for the three sigmoid expressions and final
rescaling. Equations, thresholds, sharpnesses, heat, color and backward math
remain the same. No ranking quantization, tie tolerance or failed test bypass.
Started identical fixed-input replay into `pixel598_precise.log` to validate.
All source syntax passed; full preflight will still be required afterward.

### Corrected pixel passes exactly

`pixel598_precise.log` reports GPU RGB exactly equal to Torch with either saved
fast or eager coefficients: [0.55633986, 0.34601310, 0.36771244]. All selected IDs
now match Torch, including 3551/4280 instead of 4344/4523. Filter weights computed
from the GPU responses match Torch exactly (max error 0); raw response difference
remains 1.78814e-7 but no longer changes this selection. This is a demonstrated
rounding bug correction, not a threshold relaxation. Previously affected output
pixels may change to the Torch result; mathematical heat/filter equations remain.
Started the prior 600-contributor forward/backward regression again using the
corrected filter, log `chunk_renderer_precise.log`.
Updated support module description to identify kernel-centric main machinery
and legacy internal spectral utilities accurately; no functional edit there.

Corrected-filter 600-contributor CUDA renderer/backward regression also passed:
RGB max 2.98023e-8; five VJP max errors [2.79397e-9,1.49012e-8,0,2.32831e-9,
9.89530e-10]. Evidence `chunk_renderer_precise.log`. Same original tolerances.
Updated existing architecture documentation to distinguish current kernel-centric
path, exact geometry lookup/chunk merging, numerical filter fix, explicit
approximation policy, and historical triangle selection. No new documentation
file beyond the user-requested work log. Allocation had 46:47 remaining at the
last read-only query. Next retry the existing 128/256 sweep with corrected shader.

Command construction was checked before execution: output paths change to
scaling_precise; the existing module remains camera_scaling (no new module).

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:30:00 --output=experiments/hks_work_session_20261121/scaling_precise_step.log --error=experiments/hks_work_session_20261121/scaling_precise_step.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.benchmarks.camera_scaling --resolutions 128 256 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121/scaling_precise
```

### Corrected 128 paired test complete; native 256 OOM

Corrected shader: both strict same-method gates pass at 128 and 256. At 128,
40 unprofiled updates complete for both methods. Native training interval
1.236127 s, HKS 9.121417 s (includes cold per-view compilation); worker bodies
65.5252 / 86.1378 s. Held-out deployed PSNR 19.572516 / 19.870699 dB;
forward-only medians 16.189570 / 7.519064 ms. Peak allocated bytes
13286290432 / 2477384704. Both final counts 5000. Forward is faster in this
measurement; training is slower. Do not infer warmed full-training throughput
from unsynchronized host step medians.

At 256, native training failed during loss.backward with torch.OutOfMemoryError:
requested 11.63 GiB, 2.77 GiB free on 39.49 GiB A100; process 36.71 GiB in use.
Sources unchanged, per failure.json. Driver consequently did not run HKS train.
This is not a matched 256 performance result. No allocator settings changed.
Allocation remaining time was 25:41. Next run matched 1105 updates at 128 to
exercise original adaptive density after update 1000. Reuse only current-source
paired initial state, with unchanged controls except requested update count.

Read-only follow-ups: all current HKS/script Python sources compile without
writing pyc. Compared all 89 original hktex Python hashes to the first fresh
GPU run's controls.sources: no changes. The earlier aggregate digest's encoding
was not recorded, so direct per-file manifest comparison is used instead of
claiming an unmatched aggregate hash. Two inspection corrections: provenance is
in training_research.py (no source_provenance.py); surface manifest source hashes
are nested under controls.sources, not the top level. No source edits resulted.
Slurm step format does not support %T (read-only query warning); use %i/%M/%N.

Updated HKS README to remove stale statements that GPU validation has not run,
explain strict-default versus opt-in approximation, include the opt-in in the
illustrative command, and state actual measured results/256 OOM. Documentation
edits only while source-provenance training run is active.

Matched 1105 command (same allocation, 18-minute maximum step):

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:18:00 --output=experiments/hks_work_session_20261121/matched1105_step.log --error=experiments/hks_work_session_20261121/matched1105_step.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 1105 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --initial-state experiments/hks_work_session_20261121/scaling_precise/128/native_verify/initial.pt --execute --output experiments/hks_work_session_20261121/matched1105
```

Final review in progress: full git diff --check reports only pre-existing
trailing whitespace at hktex/modules/tracer.py:285,290. Original source remains
untouched; these warnings were not automatically fixed. Scoped HKS/script/doc
whitespace check will be reported separately. Read-only worker process inspection
confirms a live native verification process inside step 20644106.15; initialization
and imports are outside measured training time. A production import check is
also running on the login host without GPU operations. Its Mitsuba LLVM warning
and Matplotlib temporary cache notice will be distinguished from errors.

Login-node production import check: rendering, surface_adapter imports succeeded,
then training_checks imports original HKTex, whose Mitsuba utility explicitly
selects cuda_ad_rgb. It failed with cuInit 'no CUDA-capable device is detected'.
This is an expected GPU requirement, not a passing CPU import result. The same
training imports execute on the allocated GPU in the verification/training
workers. No native fallback or variant change was introduced to hide this.
All 89 native file hashes match diag02/native_verify manifest controls.sources.
Scoped git diff --check for HKS/scripts/edited docs passes.

1105 native verification complete with exact saved initialization. This attempt's
startup is unusually slow: imports_cuda_config 154.803 s, sampler 20.739 s,
trainer/spectra 23.712 s; gates 56.614 s; worker body 283.302 s. These are measured
setup costs, not renderer throughput. No hypothesis about filesystem or system
load is treated as proven. A no-GRES nvidia-smi probe found no devices because
that inspection step intentionally had no GPU visibility; a subsequent GPU-visible
read-only query ran between worker exits and reported no active compute process.
The driver then started hybrid verification normally. No experiment was overlapped
with another GPU performance run.

CLI preview passed without CUDA and planned exactly native_verify, hybrid_verify,
native_train, hybrid_train with original policies and explicit approximation.
Preview printed only JSON (gpu_started=false); no preview output directory created.

1105 native training complete: synchronized interval 37.492964 s; worker body
102.266818 s; held-out deployed PSNR 20.346213 dB. Original adaptive density
changed kernel counts at updates 1000 and 1100, returning to 5000 kernels at
completion. This confirms the run crosses actual density transactions rather
than merely reaching a nominal update budget. HKS training now running; paired
conclusion awaits its result. Allocation remaining 16:26 at last query.

### Adaptive-density 1105 run complete

Both methods complete all 1105 updates; fixed tensor versions, final parameter
finiteness and optimizer bindings pass. Native strict field gate passes; HKS
same-method GPU gates pass under explicit approximation mode, native equality
remains false. Native/HKS synchronized training 37.492964 / 49.660046 s;
worker bodies 102.266818 / 128.298103 s. Deployed held-out PSNR 20.346213 /
19.596410 dB (HKS is 0.749804 dB lower). Thus HKS is 1.3245 times slower for
this interval and misses a provisional final 0.5 dB comparability target. Initial
quality acceptance was not a final-quality guarantee. No training speedup claim.

At update1000 native pruning 5000→3928, HKS5000→3630; update1100 densifies each
back to5000. Counts diverge despite identical original policies, as expected for
different learned fields. Peak allocated native13286290432 / HKS2502251520 bytes.
Receipts preserve all trajectories. No source files changed during the run.

Allocation remaining14:19; start a ten-minute-bounded matched5000 update run,
now that implementation and adaptive-density checks passed. Purpose: establish
the full-update outcome, not assume speed/quality. Same initial snapshot, cameras,
loss, policies and unprofiled timing. Short-run quality deficit remains disclosed.

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:10:00 --output=experiments/hks_work_session_20261121/matched5000_step.log --error=experiments/hks_work_session_20261121/matched5000_step.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --initial-state experiments/hks_work_session_20261121/scaling_precise/128/native_verify/initial.pt --execute --output experiments/hks_work_session_20261121/matched5000
```

### Session source-change inventory (not the whole dirty Git diff)

All changes use existing Python modules; no implementation deletion or native
edit in this session. Pre-existing Git removals/renames are not attributed to this
work. Files edited here:

| File | Change |
|---|---|
| hktex_splatting/knn_heat/triangle_kernel_support.py | Bounded exact fixed-hop topology lookup, mutation guards, default8, uncapped kernel-centric faces with global budgets; legacy512 guard preserved. |
| hktex_splatting/triton/heat_kernel_splatting.py | Exact chunked strongest30 above512; precise filter sigmoid/division; trace-only filtered weights. |
| hktex_splatting/triton/splat_contributions.py | Optional chunk origin/limit, original callers retain defaults. |
| hktex_splatting/rendering/heat_kernel_splatting.py | Main cap0 default and diagnostic filtered-record response. |
| hktex_splatting/modules/spectral_heat_field.py | Kernel support default8, fresh learned state retained. |
| hktex_splatting/training/training_checks.py | Canonical source/weight replay checks; strict implementation versus explicit bounded approximate-native acceptance. |
| hktex_splatting/training/uv_training.py | Record/accept explicit approximate experiment while retaining strict gate receipt. |
| hktex_splatting/training/surface_adapter.py | Restore full diagnostic tensors to original query order. |
| splatting_script/experiments/train_surface.py | Explicit approximation/quality flags, cap0/default8, propagation, report/verification-memory labels. |
| hktex_splatting/README.md | Current protocol, working command, limitations and measured outcomes. |
| docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md | Current kernel-centric pipeline above clearly marked historical architecture. |
| docs/HKS_WORK_LOG.md | Authorized new Markdown log, commands, evidence, failures and conclusions. |

No new package dependency, Python module, benchmark entry point, checkpoint class
path or native optimizer/density implementation. The fixed-hop graph relation is
precomputed, not learned heat, coefficients, visibility, camera lighting or RGB.
Math equations are retained; affected GPU filter outputs intentionally change to
the Torch reference rounding. Eight versus four hops intentionally changes the
approximate support experiment and is explicitly recorded, not numerical parity.

Independent current-source CPU topology regression passed for hops0/1/2/4/8:
20 triangles forming an edge-connected chain; every precomputed row equals the
independent clipped integer hop interval. Same-value in-place table mutation
invalidates geometry. Actual KernelSurfaceSupport class extracted from its AST
to avoid original HKTex's mandatory CUDA import on the login node. This isolates
geometry only; GPU render/VJP results are reported separately. No test file,
fixture, package or pyc was created.

Saved-checkpoint movement audit (CPU torch.load, no training): corrected 128
40-update runs have no density/reindexing events, so source indices align.
Native3730/5000 and HKS3433/5000 kernel location vectors differ from exact shared
initial values. Native1128 and HKS433 source-face IDs change. This demonstrates
actual on-mesh updates/source-face crossings, beyond merely configuring an
optimizer. Existing GPU short-run/full-density tests use fresh support each step.

Retained entry-point checks passed on CPU: compare_training workflow preview
(--kernel-counts5000 --resolutions128 --iterations40, without --execute) and
compare_backends --help. Both exit0 and start no GPU workload. The older pointwise
image-fitting workflow remains a separate historical protocol (its default
support cap128/visibility reuse were not silently rewritten). Only native and
kernel-centric HKS are competitors in the current train_surface experiment.
A CLI preview is not claimed as GPU execution of that older workflow.

5000 native training complete; HKS full worker has started. To avoid interrupting
its final worker purely because the original step bound was too short, changed
ONLY step20644106.19 TimeLimit from10 to13 minutes:
`scontrol update StepId=20644106.19 TimeLimit=00:13:00` (exit0).
The job/allocation remains the same unextended two-hour allocation; step deadline
is still before that allocation expires. No new allocation requested, no source
change or verification bypass. This adjustment is separate from measured training.

### Full matched 5000-update result — research goal not met

Both workers complete all5000 updates, original policies and shared initialization,
fresh GPU rasterization and GPU targets at128×128, seed0, unprofiled. Fixed mesh/
spectral versions, optimizer bindings, source provenance and finite learned
parameters pass. Native equality is strict; HKS is explicitly approximate.

| Metric | Native HKTex | Kernel-centric HKS |
|---|---:|---:|
| Synchronized training interval, seconds | 150.166016 | 166.938579 |
| Complete training worker body, seconds | 219.189078 | 248.573332 |
| Held-out deployed PSNR, dB | 21.885996 | 20.920457 |
| Forward-only median, ms | 14.511572 | 8.160524 |
| Peak allocated bytes (training) | 13286290432 | 2518844928 |
| Peak reserved bytes (training) | 18402508800 | 12209618944 |
| Final kernel count | 5000 | 5000 |

HKS is11.17% slower in the training interval and13.41% slower in worker body,
with0.965539 dB lower deployed quality. Forward-only is1.7783× faster for these
held-out inputs, but quality is not matched; this is NOT a matched-quality speedup
or a faster training result. Five-times training target is not achieved. Different
protocols/timers must not be substituted to claim success.

Native density: update1000 5000→3943; update1100→5000. HKS: update1000
5000→3626; update1100→5000. All trajectories retained. Native's pruning outcome
differs slightly between independent1105 and5000 runs despite matching initial
controls, consistent with previously observed native GPU nondeterminism, although the
cause of this cross-run difference was not isolated; no reproducibility gate was relaxed or bitwise training determinism claimed.

Evidence: experiments/hks_work_session_20261121/matched5000/comparison.md and
native_train/result.json, hybrid_train/result.json, both trajectory.jsonl files.
Forward excludes rasterization/target/loss/backward; training uses fresh camera
rasterization. Memory is PyTorch allocator peak, not all device memory. Current
benchmark is unlit mesh texture fitting, not view-dependent lighting training.

### Final HKS256 smoke

Native256 OOM is retained as failure evidence. Run only the existing HKS training
worker at256×256 for40 updates, with the current-source256 initial snapshot.
The worker itself executes all1024-sample initial correctness/quality gates;
no gate is skipped. A separate preliminary256 verification already passed.
This saves a duplicate external verification process in the last allocation
minutes; it is not a paired performance comparison. Step bounded to3 minutes,
inside unchanged existing allocation. No source edits during this worker.

```bash
srun --jobid=20644106 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:03:00 --output=experiments/hks_work_session_20261121/hks256_step.log --error=experiments/hks_work_session_20261121/hks256_step.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend hybrid --worker --run-dir experiments/hks_work_session_20261121/hks256 --supervision camera --visibility fresh --resolution 256 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --initial-state experiments/hks_work_session_20261121/scaling_precise/256/native_verify/initial.pt --execute --output experiments/hks_work_session_20261121/hks256
```

### Final 256×256 result

Existing worker exits0, status APPROXIMATE_COMPLETE, all40 updates. Strict
same-method GPU operator checks, fixed-tensor-version checks, optimizer bindings
and finite parameters pass. Training interval9.666167s; worker92.229037s;
forward-only median11.793504ms; held-out deployed PSNR20.032552dB. Peak allocated
2531587072 bytes; peak reserved12207521792 bytes; final5000kernels. Do NOT compare
this PSNR with native_field in the same HKS receipt as if it were a separately
native-trained model. Native256 training OOM means no paired256 speed/quality
claim. Evidence: hks256/result.json and hks256_step.log.

### Final verification / limitations / recovery

- All77 current HKS/script Python sources compile. Scoped whitespace check passes.
- All89 original hktex Python files match the first fresh GPU-run hashes exactly.
  Original pre-existing tracer edits were preserved, not restored or overwritten.
- Main128 native/HKS gates and full5000 updates complete. HKS25640 updates also
  complete. CPU graph0/1/2/4/8, cap/budget/order, exact GPU5000-contribution merge,
  GPU600-contributor backward and saved-pixel filter regressions pass as above.
- Training/benchmark CLI previews pass. Login-node full native imports require
  CUDA and therefore failed there; actual compute-node workflows pass imports.
- No new allocation, package installation, source deletion, Python module, commit,
  push, reset, stash, staging, or Git history change. Only the existing allocation
  was used. The short step was extended within its unchanged allocation, as logged.
- Git remains broadly dirty from prior work. Reviewed final status: MM2, D27,
  AD105, AM2, A4, intent-add2, M1, MD1, untracked1882 paths (including historical
  results and this session's outputs). Combined tracked diff against HEAD:
  31files,1108insertions,5280deletions. This is NOT the session diff; most HKS
  sources are already untracked and pre-existing deletions are unrelated. Session
  file inventory above is the authoritative scoped list. Full diff --check only
  warns about pre-existing native tracer whitespace; scoped edits pass.
- SciPy>=1.10 is already declared in pyproject.toml:15. No new dependency added.
- PSNR in this harness is foreground samples of ONE heldout_left synthetic view,
  not averaged across all available held-out views or an independent dataset.
  Multi-view/multi-seed quality evaluation remains necessary before broad claims.
- Eight-hop support is approximate, not a conservative influence bound. Its extent
  does not adapt to learned anisotropy; shape changes affect freshly computed
  coefficients, not the fixed hop radius. Increasing hops can add work without
  recovering native outer-KNN membership. Learned heat/lighting is not cached.
- Native equivalence remains false. Explicit approximation does not mark that
  check passed and does not waive GPU/eager RGB/VJP/membership checks. Final HKS
  quality misses the provisional0.5dB target by about0.47dB in the5000-update test.
- Current matched experiment uses original controllers/optimizers and camera
  texture supervision; it is not the complete original Mitsuba experiment or a
  view-dependent radiance-field training benchmark. Five-times training target
  is not achieved; no faster-training product is claimed.

Retained evidence: every experiment receipt, failure snapshot, final checkpoint
and trajectory remains under experiments/hks_work_session_20261121/. Do not use
Git reset/clean to recover selectively: much source was untracked before this
session. Review this log's session file inventory, current contents and first-run
manifest hashes before reverting anything. Native per-file hashes are in
`diag02/native_verify/manifest.json` under controls.sources; final source receipts
are in matched5000/*/manifest.json. Existing cleanup recovery records remain.

Next research priorities: (1) repair the quality/support mismatch before claiming
matched quality; (2) profile current128 hot training separately from startup,
including density transitions; (3) investigate measured geometry/source/visibility
and allocation synchronization costs without caching learned values; (4) evaluate
multiple held-out views/seeds and a reproducible memory-safe native256 protocol.
Do not substitute forward-only latency or remove strict checks to manufacture a
training speedup. No more GPU work is left running by this session.

## Second allocation — continued optimization (2026-11-21)

User authorized continuing with another allocation. Read the prior results and
inspected current source/Git status before changing anything. Existing job20645021,
nodeg086, A100 allocation, 1:59:13 remaining at discovery. No allocation requested.
Prior working-tree changes and original hktex files remain preserved. Same log.

Start a fresh paired diagnostic profile at128×128,5000initial kernels,1120updates,
profile window980..1107 inclusive: covers original pruning1000 and densification
1100, with both native/HKS. No code edits while provenance run is active.
New node requires fresh controlled initialization; no old snapshot controls bypass.
Profiled interval is diagnostic only and cannot establish a training speedup.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:20:00 --output=experiments/hks_work_session_20261121_round2_profile.log --error=experiments/hks_work_session_20261121_round2_profile.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 1120 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --profile-start 980 --profile-steps 128 --execute --output experiments/hks_work_session_20261121_round2_profile
```

Second-allocation design review (before edits): current fixed-hop selection
materializes all graph neighbors, compacts twice, performs histogram size queries,
then sorts visible pairs. Proposed exact optimization: GPU per-kernel fixed-hop
count+visibility histogram, one bounded metadata read, prefix/pack visible keys,
then existing deterministic face/kernel sort. Preserve exact hop sets, full
source-major support as a lazy diagnostic view, all budgets, Torch fallback and
native-candidate replay. Existing eager renderer must force the Torch selector
so GPU selection is independently checked. No learned coefficients reused.
This is a hypothesis until profiling and GPU timing demonstrate its effect.

Potential cleanup, reviewed BEFORE removal: DeviceCameraSupport class in
rendering/device_camera.py has no imports/call sites anywhere in current Python
sources; broad code/doc search only finds its definition. It wraps the old
FusedResidentSupportGeometry, which remains preserved independently. Current
DirectTileRenderer constructs KernelSurfaceSupport instead. If removed, delete
only that unused class and its exclusive FusedResidentSupportGeometry import;
keep DeviceCameraSurface, DeviceCameraDispatch and all shared support kernels.
No files are proposed for deletion without further dependency review. Legacy
persistent_association/occupancy code has historical docs/shared selection links;
keep for now rather than assume it is safely removable.

Saved a pre-edit recovery archive because many HKS sources were already untracked:
experiments/hks_work_session_20261121_round2_before_sources.tar.gz, SHA256
c6136eaaec2a0632574b966bb16bb86add3d47180d0103465a5c04bcd3ef770a.
Contains eight planned target modules, original paths, before second-allocation
edits. This is a recovery artifact, not another imported implementation.

Additional exact-overhead candidates from call-site review: DeviceCameraSurface
repeats foreground torch.where already done by CameraSupervision, uses dynamic
bincount for known histogram sizes and sorts pixels solely to pack tile lanes.
Can reuse same-frame pixel indices with strict checks, fixed-size histograms,
and physical tile lanes (all consumers honor -1 sentinels) without caching views.
HKS.prepare_knn and LiveHeatField.sources also repeat identical source barycentric
geometry; any sharing must be one-forward-only, gradient-connected and guarded
against parameter mutation/replacement. No changes implemented during profile.
Custom density/optimizer modules contain unique prior research and checkpoint/API
considerations: retained for now; disabling their unnecessary default-path imports
is preferable to deleting that work solely for cosmetic cleanup.

Paired1120 diagnostic profile completes. Window980..1107 (128calls) includes
prune/densify and cold shape recompilation. Inclusive meansms/call:
HKS forward43.594, splat28.251, source9.742, backward4.104,
importance pre15.010, error-density pre8.721, sample geometry1.568,
selector1.253, coefficient0.967, tilequeue0.875. Native prepare_batch13.028,
prepare_knn11.929, backward10.809; includes compile costs. Do not sum nested
rows or use this profiled interval as a speed claim. Exact receipts in round2_profile.
Profiler's optimizer wrapper causes a StepLR warning about method replacement;
wrapper still calls the original step before scheduler; no ordering changed.

Higher-priority exact optimization identified: the padding bridge calls native
controllers once for each distinct valid contribution width. Both native pre
methods only update named accumulator buffers. Use one shallow controller copy
with a scratch model containing those accumulators plus one discarded padding
row; invoke the ORIGINAL native pre method on full rectangular data once, then
copy real rows back to original buffers. Keep original objects, optimizer links,
post methods, policies and valid zero-weight hits. Exact-controller-type guard;
legacy grouping remains fallback. Compare accumulator states before enabling.
New scratch row is not a learned kernel or an optimizer/model parameter.

Density prototype implemented in existing bridge; inactive callbacks can omit
contribution tapes, active exact-native callbacks use scratch padding. Native
source untouched. GPU invalid-padding check becomes asynchronous assertion;
CPU retains ValueError, so safety checks remain without a device read per callback.
First CPU fixture failed because it stored a pre-bound method on SimpleNamespace:
copying that object retains the old self, unlike real controller class descriptors.
Corrected fixture to an actual class with the extracted native method; no production
change made to hide that fixture error. Tests pending. Also made unused pixel-count
N runtime/non-specializing in shared merge and runtime RGB reconstruction, removing
unnecessary new-camera JIT variants; shader arithmetic and call signatures retained.

Density component results: CPU and real CUDA compare extracted unchanged native
pre callbacks through grouped versus scratch routing. Hit counters EXACT; maximum
CPU contribution error5.72e-6, error accumulator4.77e-7; CUDA9.54e-6 /7.15e-7.
All within original3e-5 absolute/3e-4 relative thresholds. Includes empty/partial/
full rows, valid zero weights,1/37/5000sources, buffer-identity and inactive checks.
CPU invalid nonzero padding is rejected. GPU synthetic4096-point two-controller
benchmark: grouped19.171/18.382ms versus scratch0.610/0.603ms, forward and reverse
measurement order. Component-only speed; not a training claim. Evidence round2_density.log.

Implemented exact GPU graph count/visible pack in existing triton/resident_gpu.py;
normal CUDA selection uses this, while existing eager/Torch renderer forces the
original tensor selector for independent comparison. Full source-major graph
support is a lazy inspection view with fresh source-face snapshot; all pair,
fragment/byte/cap guards retained. No source-count limit introduced. Device camera
geometry reuses validated same-frame pixel indices, uses fixed-size histograms,
and physical8×8tile lanes with -1holes (all consumers mask -1). Fresh rasterization
still occurs every frame; no camera/heat/lighting cache. Unused DeviceCameraSupport
class and exclusive import removed after audited proposal; shared implementation
and its fallback retained. No source file deleted.

Geometry regression caught a real physical-lane bug at irregular width23: flat
pixel%8 is not column%8 unless width is divisible by8. Corrected to pixel%width%8.
Normal benchmark widths were unaffected, but general geometry contract is preserved.
First component run stops at this CPU failure, so no GPU correctness claim for it.
Retry all CPU/CUDA geometry/support tests with the correction. Explicit axis0 added
to Triton cumsum before first GPU compilation; no tolerance change.

Corrected component suite now passes CPU and real CUDA: geometry empty/full/
irregular frames, identical pixels/faceIDs/barycentrics/query embeddings/dispatch
counts and tile membership, rejected incomplete prepared pixels, mutation guards;
exact support CSR/source-major views/counters for hops0/2/8 and0/1/37/5000sources,
invisible/mixed visibility and disconnected component. Eager selector remains
independent tensor oracle. Evidence round2_components_retry.log.
Synthetic50×50triangle-grid5000source selection: tensor1.628/1.516ms,
fused0.571/0.571ms (bothorders, same505846discovered/100920visible pairs).
Fresh256geometry: old1.824/1.306ms, new1.095/1.155ms; variability disclosed.
No component figure is a full-training speed claim. Start unprofiled paired1120
fresh initialization with all strict GPU gates before updates, then original
pruning/densification. No further source edits during this provenance run.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261121_round2_optimized1120.log --error=experiments/hks_work_session_20261121_round2_optimized1120.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 1120 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121_round2_optimized1120
```

<details><summary>Second-allocation GPU density regression and timing command</summary>

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:08:00 --output=experiments/hks_work_session_20261121_round2_density.log --error=experiments/hks_work_session_20261121_round2_density.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -u - <<'PY'
DEVICES=['cuda']
import ast,copy,types,torch
from pathlib import Path
from types import SimpleNamespace as NS
from hktex_splatting.training.surface_adapter import ValidDensityContributions,valid_contribution_groups,contribution_validity

def native_method(path):
 tree=ast.parse(Path(path).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='pre_backward_step')
 fn.returns=None
 for arg in (*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs):arg.annotation=None
 if fn.args.vararg:fn.args.vararg.annotation=None
 if fn.args.kwarg:fn.args.kwarg.annotation=None
 ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),path,'exec'),ns)
 return ns['pre_backward_step']

for device in DEVICES:
 torch.manual_seed(17)
 for count,points in ((1,1),(37,128),(5000,1024)):
  width=30
  ids=torch.randint(count,(width,points,1),device=device)
  weights=torch.rand((width,points,1),device=device)
  lengths=torch.arange(points,device=device)%31
  valid=torch.arange(width,device=device)[:,None]<lengths
  ids[~valid]=-1;weights[~valid]=0;weights[0,::3]=0
  rgb=torch.rand((points,3),device=device);gt=torch.rand_like(rgb)
  for kind,names in [('importance_pruning',('_hit_accumulator','_contrib_accumulator')),('error_densification',('_error_accumulator',))]:
   method=native_method('hktex/density_controllers/'+kind+'.py')
   model=NS(N_sources=count,**{n:torch.rand((count,1),device=device) for n in names})
   other=copy.deepcopy(model)
   cfg=NS(start_iter=100,stop_iter=2000,accumulation_interval=50,error_accumulation_interval=50)
   def controller(m):
    out=type('NativeCallbackFixture',(),{'pre_backward_step':method})();out.cfg=cfg;out._model=m;return out
   ref=controller(other)
   for mask,idx,w in valid_contribution_groups(ids,weights,count,compact_widths=True):
    ref.pre_backward_step(step=1000,rendered_colours=rgb[mask],gt_colours=gt[mask],topk_kernel_idxs=idx,topk_kernel_contribs=w,kernel_contributions=None)
   bridge=ValidDensityContributions(controller(model),model);bridge.skip_inactive=True;bridge.accumulation_option='accumulation_interval';bridge.accumulator_names=names
   original_ids={n:id(getattr(model,n)) for n in names}
   bridge.pre_backward_step(step=1000,rendered_colours=rgb,gt_colours=gt,topk_kernel_idxs=ids,topk_kernel_contribs=weights)
   errors={n:float((getattr(model,n)-getattr(other,n)).abs().max()) for n in names}
   for n in names:
    assert id(getattr(model,n))==original_ids[n]
    assert torch.allclose(getattr(model,n),getattr(other,n),atol=3e-5,rtol=3e-4),(kind,count,n,errors)
    if n=='_hit_accumulator':assert torch.equal(getattr(model,n),getattr(other,n))
   bridge.pre_backward_step(step=0,rendered_colours=rgb,gt_colours=gt,topk_kernel_idxs=None,topk_kernel_contribs=None)
   print(device,kind,count,points,errors,'PASS')
 bad=ids.clone();bad[0,0,0]=-1;bad_weights=weights.clone();bad_weights[0,0,0]=1
 if device=='cpu':
  try:contribution_validity(bad,bad_weights,count)
  except ValueError:print('CPU invalid-padding guard PASS')
  else:raise AssertionError('Invalid padding accepted')
print('DENSITY_REGRESSION_PASS')
import time,statistics
from hktex_splatting.training.surface_adapter import ContributionGroups
count,points,width=5000,4096,30
torch.manual_seed(23)
ids=torch.randint(count,(width,points,1),device='cuda');weights=torch.rand((width,points,1),device='cuda')
valid=torch.arange(width,device='cuda')[:,None]<(torch.arange(points,device='cuda')%31)
ids[~valid]=-1;weights[~valid]=0
rgb=torch.rand((points,3),device='cuda');gt=torch.rand_like(rgb)
sets={}
for mode in ('grouped','scratch'):
 model=NS(N_sources=count,**{n:torch.zeros((count,1),device='cuda') for n in ('_hit_accumulator','_contrib_accumulator','_error_accumulator')})
 bridges=[]
 for kind,names in [('importance_pruning',('_hit_accumulator','_contrib_accumulator')),('error_densification',('_error_accumulator',))]:
  method=native_method('hktex/density_controllers/'+kind+'.py')
  ctrl=type('NativeCallbackFixture',(),{'pre_backward_step':method})();ctrl.cfg=NS(start_iter=100,stop_iter=2000,accumulation_interval=50,error_accumulation_interval=50);ctrl._model=model
  b=ValidDensityContributions(ctrl,model);b.skip_inactive=True;b.accumulation_option='accumulation_interval';b.accumulator_names=names if mode=='scratch' else ();bridges.append(b)
 sets[mode]=(bridges,ContributionGroups())
def run(mode):
 bridges,groups=sets[mode];groups.clear()
 for b in bridges:
  b.group_cache=groups;b.pre_backward_step(step=1000,rendered_colours=rgb,gt_colours=gt,topk_kernel_idxs=ids,topk_kernel_contribs=weights)
for mode in sets:
 for _ in range(3):run(mode)
 torch.cuda.synchronize()
for order in [('grouped','scratch'),('scratch','grouped')]:
 for mode in order:
  torch.cuda.synchronize();start=time.perf_counter()
  for _ in range(20):run(mode)
  torch.cuda.synchronize();print('DENSITY_PAIR_MS',mode,(time.perf_counter()-start)*1000/20)
print('DENSITY_BENCH_COMPLETE')
PY
```
</details>

<details><summary>Corrected geometry and support regression/timing command</summary>

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:08:00 --output=experiments/hks_work_session_20261121_round2_components_retry.log --error=experiments/hks_work_session_20261121_round2_components_retry.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -u - <<'PY'
DEVICES=['cpu','cuda']
import ast,tarfile,time,torch
from pathlib import Path
from types import SimpleNamespace as NS
from hktex_splatting.rendering.device_camera import DeviceCameraSurface
from hktex_splatting.knn_heat.triangle_kernel_support import KernelSurfaceSupport
archive='experiments/hks_work_session_20261121_round2_before_sources.tar.gz'
with tarfile.open(archive) as ar:old=ar.extractfile('hktex_splatting/rendering/device_camera.py').read().decode()
tree=ast.parse(old);nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('DeviceCameraSurface','DeviceCameraDispatch')]
ns={'torch':torch,'SimpleNamespace':NS};exec(compile(ast.Module(body=nodes,type_ignores=[]),'previous_device_camera','exec'),ns)
Old=ns['DeviceCameraSurface']
for device in DEVICES:
 for h,w,kind in ((8,16,'empty'),(8,16,'full'),(17,23,'mixed')):
  torch.manual_seed(12);faces=torch.randint(100,(41,3),device=device);emb=torch.randn((100,5),device=device)
  ids=torch.randint(-1,41,(h,w),device=device)
  if kind=='empty':ids.fill_(-1)
  if kind=='mixed':ids[torch.rand((h,w),device=device)<.4]=-1
  if kind=='full':ids.clamp_min_(0)
  bary=torch.rand((h,w,3),device=device);bary/=bary.sum(-1,keepdim=True)
  pixels=torch.where(ids.flatten()>=0)[0]
  before=Old(faces,emb,ids,bary);after=DeviceCameraSurface(faces,emb,ids,bary,pixels=pixels)
  for name in ('pixels','face_ids','bary','active_ordinals','tiles','tile_lengths','active','vertices','queries'):
   assert torch.equal(getattr(before,name),getattr(after,name)),(device,kind,name)
  for a,b in zip(before.tile_pixels,after.tile_pixels):assert torch.equal(a[a>=0],b[b>=0])
  for name in ('lengths','patch_tiles','face_counts','face_starts','pixel_counts_device'):
   assert torch.equal(getattr(before.dispatch_geometry,name),getattr(after.dispatch_geometry,name)),name
  assert before.valid() and after.valid()
  after.pixels.add_(0);assert not after.valid()
  print('GEOMETRY',device,h,w,kind,'PASS')
 if device=='cpu':
  try:DeviceCameraSurface(faces,emb,ids,bary,pixels=pixels[:-1])
  except RuntimeError:print('Prepared subset rejection PASS')
  else:raise AssertionError('Bad prepared pixels accepted')
 for hops in (0,2,8):
  faces=torch.tensor([[i,i+1,i+2] for i in range(24)]+[[100,101,102]],device=device)
  emb=torch.zeros((103,3),device=device);support=KernelSurfaceSupport(faces,emb,hops=hops)
  for count in (0,1,37,5000):
   src=torch.arange(count,device=device)%len(faces)
   for visible in (False,True):
    pix=torch.arange(len(faces),device=device)%4 if visible else torch.zeros(len(faces),device=device,dtype=torch.long)
    dispatch=NS(pixel_counts_device=pix,face_counts=(pix>0).long(),valid=lambda:True)
    actual,stats=support.select(src,dispatch_geometry=dispatch,max_records=16777216)
    all_faces=support.support_faces.clone();all_offsets=support.source_offsets.clone()
    ref,refstats=support.select(src,dispatch_geometry=dispatch,max_records=16777216,reference=True)
    for name in ('counts','offsets','kernels','entry_faces'):
     assert torch.equal(getattr(actual,name),getattr(ref,name)),(device,hops,count,visible,name)
    for name in ('maximum','tasks','fragments'):assert getattr(actual,name)==getattr(ref,name)
    assert torch.equal(all_faces,support.support_faces) and torch.equal(all_offsets,support.source_offsets)
    assert stats['discovered_pairs']==refstats['discovered_pairs']
   print('SUPPORT',device,hops,count,'PASS')
print('GEOMETRY_SUPPORT_REGRESSION_PASS')
if 'cuda' in DEVICES:
 device='cuda';torch.manual_seed(31)
 side=50;face_list=[]
 for y in range(side):
  for x in range(side):
   a=y*(side+1)+x;b=a+1;c=a+side+1;d=c+1;face_list.extend(((a,b,c),(b,d,c)))
 faces=torch.tensor(face_list,device=device);emb=torch.zeros(((side+1)**2,64),device=device)
 support=KernelSurfaceSupport(faces,emb,hops=8);src=torch.randint(len(faces),(5000,),device=device)
 pix=torch.randint(1,17,(len(faces),),device=device)*((torch.arange(len(faces),device=device)%5)==0)
 dispatch=NS(pixel_counts_device=pix,face_counts=(pix>0).long()*2,valid=lambda:True)
 for ref in (False,True):
  for _ in range(3):support.select(src,dispatch_geometry=dispatch,max_records=16777216,reference=ref)
  torch.cuda.synchronize()
 for order in ((True,False),(False,True)):
  for ref in order:
   torch.cuda.synchronize();start=time.perf_counter()
   for _ in range(20):mapping,stats=support.select(src,dispatch_geometry=dispatch,max_records=16777216,reference=ref)
   torch.cuda.synchronize();print('SUPPORT_MS','tensor' if ref else 'fused',(time.perf_counter()-start)*1000/20,stats)
 h=w=256;ids=torch.randint(-1,len(faces),(h,w),device=device);ids[torch.rand((h,w),device=device)<.4]=-1
 bary=torch.rand((h,w,3),device=device);bary/=bary.sum(-1,keepdim=True);pixels=torch.where(ids.flatten()>=0)[0]
 for cls in (Old,DeviceCameraSurface):
  for _ in range(3):cls(faces,emb,ids,bary)
  torch.cuda.synchronize()
 for order in ((Old,DeviceCameraSurface),(DeviceCameraSurface,Old)):
  for cls in order:
   torch.cuda.synchronize();start=time.perf_counter()
   for _ in range(20):cls(faces,emb,ids,bary,**({'pixels':pixels} if cls is DeviceCameraSurface else {}))
   torch.cuda.synchronize();print('GEOMETRY_MS','old' if cls is Old else 'new',(time.perf_counter()-start)*1000/20)
 print('COMPONENT_BENCH_COMPLETE')
PY
```
</details>

Optimized unprofiled1120 result: both gates and all updates complete, optimizer
bindings/fixed tensor versions/finite parameters verified. Native/HKS intervals
33.871185/19.625955s (1.7258× for this run); worker bodies104.354932/93.140280s.
Single-heldout-left PSNR21.059885/20.622454dB, gap0.437431dB. Forward median
15.564768/6.781695ms. Peak allocated13286290432/2503299072bytes. Both final5000.
Prune1000 native5000→3950, HKS5000→3627; dens1100 both→5000. This is a short
matched camera-training result, not5× or a full-update/multi-view quality claim.
Keep source unchanged and run paired unprofiled5000 with the same current-source
initial snapshot. No need to relax any GPU/eager tolerances or use custom policies.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261121_round2_optimized5000.log --error=experiments/hks_work_session_20261121_round2_optimized5000.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 8 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --initial-state experiments/hks_work_session_20261121_round2_optimized1120/native_verify/initial.pt --execute --output experiments/hks_work_session_20261121_round2_optimized5000
```

Optimized5000 unprofiled comparison COMPLETE, same source and original policies.
Native/HKS training intervals141.014885/68.950211s (2.0452×), worker bodies
205.771055/148.329978s (1.3873×). Forward medians15.661020/6.841896ms.
Peak allocated13286290432/2542422016bytes; final5000each. Strict operator/fixed
mesh/optimizer-binding/finite-state checks pass. Prune1000 native3945/HKS3626,
dens1100 both5000. BUT single-heldout-left PSNR21.881719/20.462841dB:1.418878dB
lower HKS quality. Thus no matched-quality speedup, no5× claim. Short-run0.44dB
gap did not predict full-training gap. Source/controller accumulator equations
remain same within measured float tolerances; bitwise learned trajectories are
not claimed. Evidence round2_optimized5000.

Next: existing support-hops option above8 currently falls back to repeated graph
traversal because precomputation uses an exponential degree bound. Extend fixed
lookup construction with actual sparse reachability, bounded host workspace and
existing device-table budget, without changing any requested hop set. Then test
12hops explicitly (default8 unchanged), rather than relax quality checks. Also
correct reporting to score BOTH existing heldout_left/right views, weighted by
foreground sample count; keep per-view values so historical left-only scores
remain interpretable. Initial strict gates still use the original saved left
batch; final multi-view quality is a separate, stronger measurement.

Extended fixed-hop component checks pass CPU and CUDA for hops0/2/8/12/16 and
0/1/37/5000/20000sources, including independent chain/disconnected BFS oracle,
exact GPU/tensor CSR/counters and full source-major inspection. No arbitrary
source-count cap.12hop synthetic5000source selection tensor1.838/1.825ms,
fused0.592/0.592ms in opposite orders, identical1050091discovered209618visiblepairs.
Device-table budget unchanged; host sparse preprocessing has explicit conservative
workspace bound, and exact traversal fallback remains if lookup cannot fit.
Default support8 remains unchanged. Start a paired12hop40-update pilot with
strict gates and new two-view final reporting; do not jump to5000 before it passes.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261121_round2_hops12_smoke.log --error=experiments/hks_work_session_20261121_round2_hops12_smoke.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 12 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121_round2_hops12_smoke
```

To preserve the EXACT completed optimized8hop implementation despite untracked
sources, saved experiments/hks_work_session_20261121_round2_verified8_sources.tar.gz
(SHA2560238a9462a73eb679af04ab18a9a595b94e106ea0e7652bd8083eaa8dbffdc2c).
All169source members match completed round2_optimized5000 controls.sources hashes.
For five files with subsequent lookup/report/docstring edits, reconstructed old
bytes IN MEMORY and asserted their saved SHA256 before archiving. Live files were
not overwritten. SOURCE_MANIFEST.json inside records evidence and every hash.
Configs/assets stay in the repository with hashes in the original run manifest.
This archive provides a byte-verified fallback/reproduction point, not an imported
alternate implementation or a deletion of current work.

12hop40 pilot COMPLETE: strict GPU/eager gates and both training workers pass.
Actual Converse fixed lookup189000000bytes fits unchanged device budget; fused
engine active. Native/HKS training1.799587/0.567233s (short-run/cold variability,
not a full-training claim), worker66.825854/81.090931s. Two-view weighted deployed
PSNR18.559726/19.053103dB; per-view native left19.572240,right17.092319,
HKS left19.821997,right17.868340. All finite/fixed/optimizer checks pass.
Start full5000 paired12hop with unchanged source and same controlled initial.
PREDECLARED provisional final-quality target: weighted two-view PSNR no more than
0.5dB below native; report each view and do not conceal larger per-view deficits.
Initial acceptance still checks its original saved bounded left sample only.
No training-policy tuning or quality-tolerance relaxation.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261121_round2_hops12_full.log --error=experiments/hks_work_session_20261121_round2_hops12_full.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 12 --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --initial-state experiments/hks_work_session_20261121_round2_hops12_smoke/native_verify/initial.pt --execute --output experiments/hks_work_session_20261121_round2_hops12_full
```

Read-only connectivity diagnosis executed the unchanged native load_mesh function
via AST on CPU (same merge_tex=True/normalization), then independently built
edge-connected components including all incident faces. Actual native mesh:
75000faces,37691vertices, ONE component, zero nonmanifold edges. Initial and
trained8hop5000source models both seed that component; zero held-out-left pixels
lie on unseeded components. Thus disconnected parts/UV seams do not explain this
case's support omissions. No welding, topology edit or initializer change made.
The diagnosis informs support-radius experiments; it is not a performance test.

12hop5000 COMPLETE. Native/HKS intervals144.600663/77.756360s (1.8597×),
workers209.450850/154.426628s. Weighted two-view PSNR20.911839/20.073590dB,
gap0.838249 (FAIL provisional0.5 target). Native left21.883336,right19.489671;
HKS left21.462401,right18.229383. Left gap0.420935 passes alone; right gap1.260288
must not be concealed by selecting only left. Peak allocated13286290432/
2709870080bytes; final5000each. Prune1000 native3942/HKS3677, dens1100→5000.
Operator, optimizer and fixed asset checks all pass. Sources remain unchanged.
Next16hop pilot will use explicit common max-buffer-mib768 to permit its larger
fixed lookup; no silent budget change. First run isolated kernel execution-choice
checks: emitter2vs4warps (bitwise outputs) and existing serial/parallel corner
layout (forward/VJP tolerances, both timing orders). No new renderer competitor.

GPU execution-choice tests PASS.2/4-warp emitters produce BITWISE identical
entries/weights across6400200records; timing4warps0.904/0.903ms vs2warps0.673/
0.673ms, bothorders. Promoted only emitter to2warps. Existing parallel corner
layout passes forward and five source-state VJP checks; max errors[4.77e-7,
2.38e-7,5.96e-8,1.91e-6,3.81e-6]. Synthetic10000/100000/300000pair forward
serial≈0.263/2.19..2.48/4.44..4.51ms; parallel≈0.134/1.141/2.97ms.
Use existing --corner-evaluation parallel for next pilot, subject to REAL RGB/VJP
gates. Component gains are not whole-training speed claims. Evidence kernel_layout.log.

Also guard scratch density path against instance-level pre-callback overrides:
those retain old grouping because a copied bound closure may target the original
model (as the first fixture demonstrated). Exact class-descriptor callbacks keep
fast scratch routing; original post/optimizer behavior unchanged.

Found repeated setup work: configure_cache previously gave EVERY worker a new
compiler/extension/font/CUDA cache directory. Main surface workers now reuse
backend-isolated toolchain artifacts outside the repo; each run's loader/tmp
workspace remains unique. Legacy callers default to old isolation. This stores
compiled code/IR/font metadata only: kernel values, visibility, gradients and
lighting are recomputed. Gates still run each worker. Record shared cache policy
in equal controls and actual backend paths separately in manifests. Both methods
use identical policy; do not compare old cold-worker setup costs as algorithm gains.
CPU reuse/backend-isolation/workspace/legacy-policy/namespace tests and syntax pass.
Saved exact12hop source/config/docs recovery archive before these edits:
round2_verified12_sources.tar.gz,169source hashes verified, SHA256
4cee349dc36ca1acd91c0aa1a585cbc23b8ca96b4194841c6f97d76c4e0c9166.

Started16hop40 pilot, explicit max-buffer-mib768 for BOTH arms and existing
corner-evaluation parallel. Default hop/layout flags unchanged. Shared toolchain
artifact policy is recorded in equal controls; backend paths remain separate in
manifest metadata so initial-control equality stays meaningful. No camera,
coefficient or learned-value cache introduced. CPU16hop two-view CLI preview
passes; six invalid/retired-option cases still reject with exit2 after removal
of unreachable duplicate validation branches. Effective safety guards remain.
Updated retired-option help to match actual parser behavior. Profiler now uses
functools.wraps to preserve scheduler wrapper metadata, avoiding its false step
replacement warning without changing optimizer/scheduler order.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261121_round2_hops16_smoke.log --error=experiments/hks_work_session_20261121_round2_hops16_smoke.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121_round2_hops16_smoke
```

16hop40 pilot COMPLETE with parallel corners/2warp emission and explicit768MiB
budget. Strict implementation RGB/VJP and finite/optimizer/fixed-geometry checks
pass. Converse lookup362700000bytes, fused graph engine active (not BFS fallback).
Native/HKS intervals2.595264/0.561428s (short/cold, not a reliable speed claim),
workers52.881703/60.019038s. Two-view PSNR18.559729/18.772173dB; HKS left19.659390,
right17.445855. Shared compiler policy cold/warm costs are recorded, not described
as learned-field caching. CPU density recheck and instance-override fallback
30-group routing pass. Start unchanged-source5000 paired16hop using this controlled
initial snapshot. Final provisional0.5dB aggregate criterion unchanged; report both
views. No further source edits until this run finishes.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261121_round2_hops16_full.log --error=experiments/hks_work_session_20261121_round2_hops16_full.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --initial-state experiments/hks_work_session_20261121_round2_hops16_smoke/native_verify/initial.pt --execute --output experiments/hks_work_session_20261121_round2_hops16_full
```

### Completed 16-hop 5,000-update result

Both verification workers and both training workers complete. Sources and assets
unchanged during each worker; shared controls/initial receipts and camera metadata
trajectories match. Native-field equality is false, recorded separately from
strict passing GPU/eager RGB and parameter VJPs; approximation is explicit.

| Metric | Native | HKS |
|---|---:|---:|
| Training interval seconds | 140.812883 | 72.889431 |
| Worker body seconds | 186.874186 | 129.880781 |
| Process wall seconds | 189.465017 | 132.620486 |
| Two-view deployed PSNR dB | 20.930270 | 20.709824 |
| Left / right PSNR dB | 21.811895 / 19.610453 | 21.587389 / 19.394739 |
| Forward median ms, fixed supplied visibility | 10.709615 | 6.155899 |
| Peak allocated bytes | 13286290432 | 2936951296 |
| Peak reserved bytes | 18402508800 | 12211716096 |
| Final kernels | 5000 | 5000 |

Measured training interval ratio1.93187; worker1.4388; process1.4286.
Predeclared final aggregate quality deficit<=0.5dB passes (0.22045dB), and
both individual deficits~0.22dB. It is not native field/gradient equivalence.
HKS checkpoint evaluated with the native field gives19.836935dB, a counterfactual
that must NOT replace the independently trained native baseline20.930270dB.
Both density count changes occur at steps1000/1100. Live graph discovery handles
current source faces and count changes; final retained615933/discovered2403623
pairs, fixed geometry lookup362700000bytes. Original density and optimizers stay.
This run is unprofiled, fresh128rasterization each frame. Forward latency excludes
rasterization/targets/loss/backward. Cache policy is common but separate backend
compiler artifacts; no learned field/visibility cache. No5x claim.

Archive round2_verified16_sources.tar.gz verifies169source hashes from this run;
SHA256 bc711f92229915f6bf2108c5059ee90efec1a6f1d10b13708b7d604b99d589de.
CPU syntax77modules and all89native file hashes pass after run. Updated HKS README
and current architecture explanation; original README untouched. Starting64x64
5000pair with strict gates first; no new allocation requested.

64x64 follow-up exact command (same source and original policies):
```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:10:00 --output=experiments/hks_work_session_20261121_round2_hops16_64_full.log --error=experiments/hks_work_session_20261121_round2_hops16_64_full.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 64 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121_round2_hops16_64_full
```

Final CPU preview exits0; syntax77 modules and scoped diff --check pass.
Both edited guides have no broken relative Markdown links. A direct native import
on the login host fails because the original hktex initializer requests Mitsuba
cuda_ad_rgb and cuInit finds no GPU. This is an environment limit, not a passing
CPU import check; no native source was changed. Actual native/HKS GPU verification
and training workers imported successfully. No CPU CUDA mocks added.

### Second-allocation change inventory and cleanup scope

This supplements the first-allocation inventory above; it is not a clean-tree
Git diff. Pre-existing staged/deleted/untracked work remains untouched.

| Existing file | Work performed |
|---|---|
| training/surface_adapter.py | Original density callback scratch routing, inactive-tape omission contract, GPU async safety check, guarded instance-override fallback. |
| training/uv_training.py | Configure only audited original density callback classes; omit inactive detached tape; propagate fresh camera pixel indices. |
| training/camera_supervision.py | Supply current frame foreground indices to geometry. |
| rendering/device_camera.py | Fixed-size histograms and physical masked tile lanes; validate supplied frame indices; remove unused DeviceCameraSupport class/exclusive import. |
| knn_heat/triangle_kernel_support.py | Exact GPU count/pack selection, lazy diagnostic full support, wider bounded fixed mesh hop lookup. |
| triton/resident_gpu.py | Current-source GPU graph count and visible-pair pack kernels. |
| rendering/heat_kernel_splatting.py | Keep tensor selector for independent eager reference; fast GPU selector in live path. |
| triton/splat_contributions.py | Remove unused runtime length specialization, preserve signatures/math. |
| triton/splat_reconstruction.py | Runtime non-specialized length, preserve RGB math. |
| triton/heat_kernel_splatting.py | Two-warp emitter launch, bitwise checked. |
| utils/experiment_io.py | Optional backend-isolated toolchain artifact reuse; unique loader/tmp workspace; old callers retain isolation. |
| training/surface_profile.py | Preserve wrapped optimizer metadata in temporary profiler wrappers. |
| splatting_script/experiments/train_surface.py | Both held-out views and weighted aggregate quality, common compiler policy receipts, accurate retired-option help/duplicate-check cleanup. |
| hktex_splatting/README.md | Current setup, measured results, full pipeline and reproducible commands. |
| docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md | Current GPU support/geometry/density behavior; historical section retained. |
| docs/HKS_WORK_LOG.md | Actual commands, evidence, failures, archives and limitations. |

Paths without a leading package are relative to hktex_splatting/.
Files deleted in this session: NONE. The sole removed class was not imported,
called or named anywhere else in repository source/docs; its complete old module
is in the before_sources recovery archive. Shared resident coefficient/selector,
GPU utilities, eager reference and fallback code stay. Unique research code was
not removed merely because it was not used in this run. No new Python module,
package dependency, checkpoint class rename, native source edit, commit or push.
Serial/parallel corner operators already existed; parallel is an explicit measured
experiment setting, not a replacement equation. Fixed mesh lookup and toolchain
artifact reuse do not retain learned coefficients or camera visibility.

Additional CPU profiler regression extracts the existing SurfaceStageProfiler
class from its AST (no mock module/test file added). Temporary wrappers preserve
_wrapped_by_lr_sched metadata and return values, restore instance/class method
lookup after exit, and remove the temporary trainer.surface_profiler attribute.
All assertions pass. GPU profile run below will check actual original optimizers.

Remaining risks: final quality bound is provisional and measured only for seed0,
one Converse mesh and two synthetic views. Native-field gradients are not claimed
equal under graph support approximation. More seeds/scenes/view trajectories and
actual view-dependent lighting are untested. Native256 failed CUDA memory capacity
in the earlier allocation; no native/HKS256 speed ratio is available. The current
128study is not the complete original Mitsuba rendering/denoising experiment.
Runtime support/byte/index/mutation checks and the safe graph fallback remain.
GPU density accumulator differences are within recorded tolerances; identical
bitwise adaptive training trajectories are not promised. Original policies match.

### Completed 64x64 follow-up (5,000 updates)

Both strict GPU gates and training workers complete, sources/assets unchanged.
Native/HKS training intervals80.353280/73.297217s (1.0963x); worker bodies
123.647158/132.171553s (HKS slower including setup). Aggregate PSNR
18.174587/17.750057dB (0.42453deficit; provisional aggregate0.5target passes).
Left19.765185/19.257054 (0.50813deficit), right16.152240/15.799345
(0.35290deficit). This is NOT a claim every view satisfies0.5dB.
Forward4.984777/5.925208ms (HKS slower), peak allocated5061141504/2921401856bytes,
reserved12150898688/12194938880; final kernels5000/5000.
Result demonstrates low-resolution fixed support/setup overhead; do not generalize
128speedup to all resolutions or conceal the slower64worker/render measurements.
README now includes this result. Started separate128profile over steps980..1107
including native pruning1000/densification1100; instrumented times diagnostic only.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:06:00 --output=experiments/hks_work_session_20261121_round2_hops16_profile.log --error=experiments/hks_work_session_20261121_round2_hops16_profile.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 1120 --profile-start 980 --profile-steps 128 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --initial-state experiments/hks_work_session_20261121_round2_hops16_smoke/native_verify/initial.pt --execute --output experiments/hks_work_session_20261121_round2_hops16_profile
```

The first16hop profile attempt rejected the unprofiled initial snapshot because
profiling controls differ (source unchanged=True). This is a correct provenance
rejection, not a renderer failure; no GPU training started and checks were not
waived. Retrying in a fresh directory WITHOUT external initial state, allowing
this workflow to create its own common native/HKS snapshot. Before retry, corrected
one stale report-only sentence saying final quality uses heldout_left; it now
states aggregate recorded views and first-view gate/latency. No numerical code
changed. The previous169source archive preserves exact measured-run source.

```bash
srun --jobid=20645021 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:04:30 --output=experiments/hks_work_session_20261121_round2_hops16_profile_retry.log --error=experiments/hks_work_session_20261121_round2_hops16_profile_retry.log env LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 1120 --profile-start 980 --profile-steps 128 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261121_round2_hops16_profile_retry
```

Final formatter regression renders the completed16hop workflow in memory using
the current reporting function: correct aggregate-view wording and existing
20.709823887619514dB result asserted; no archived result overwritten. Pass.
Current77module syntax, scoped git diff --check, and89native hashes pass.
Whole-workspace git status counts (including pre-existing unrelated changes):
MM2, unstaged-deleted27, added-then-deleted105, AM2, staged-added4,
intent-to-add2, unstaged-modified1, MD1, untracked162. Do not attribute these
pre-existing deletions/additions to this session. No Git index/history mutation.
Recovery: use the listed tar archives in a separate review directory and compare
SOURCE_MANIFEST hashes; do not extract over this dirty workspace blindly. Original
before_sources and verified8/12/16 source records and all results remain intact.

### Final separate profile — completed

Both gates and1120update native/HKS workers COMPLETE; sources/assets unchanged.
128 synchronized calls per stage over updates980..1107, including density1000/1100.
Native/HKS average milliseconds per call:

| Inclusive stage | Native ms | HKS ms |
|---|---:|---:|
| Camera rasterization | 1.3248 | 1.3969 |
| Target lookup | 0.5614 | 0.5598 |
| prepare_knn | 14.2477 | 0.3917 |
| prepare_batch (nested) | 13.0891 | 0.0611 |
| forward_knn | 1.3896 | 19.8766 |
| Backward | 10.8262 | 4.7902 |
| First density pre-callback | 0.1767 | 0.5110 |
| Second density pre-callback | 0.2218 | 0.5193 |
| Adam optimizer | 0.8413 | 0.8367 |
| Geodesic optimizer | 0.9988 | 0.9946 |

HKS nested stages: source_state14.1668ms, sample_geometry1.3468,
corner_coefficients1.1655, tile_queue0.8964, supportCSR0.7018,
splat_forward0.6432, visibility_intersection0.3722, support_discovery0.1914,
training_tape0.1508. Density pre callbacks fell from the earlier grouped profile
15.010+8.721ms to0.511+0.519ms. This is an observed stage reduction, not a claim
that whole training accelerates by that ratio. Nested rows overlap: DO NOT SUM.
The profile spans dynamic count replacement and possible compiler specialization;
source_state average14.17ms is not established steady-state time. Current eager
source preparation/recompilation is the clearest remaining measured target,
followed by geometry/corner/tile overhead. Candidate graph discovery is no longer
the largest measured stage. Optimizers retain native costs/policies. Compiler
cache reuse is not a learned-field or camera cache.

No more GPU runs are started near allocation expiry. No allocation requested,
extended or cancelled; all our steps exited, leaving the user's allocation alone.
No5x speedup achieved or claimed. Best measured unprofiled128training1.93x,
completeworker1.44x, two-view deficit0.22dB. At64, training1.10x but worker/render
slower; at256 native OOM means no paired claim. Next research validation should
repeat seeds/scenes/views, examine source preparation with compilation separated,
and repeat full unprofiled timing after any further change. Do not silently reduce
support or remove native correctness checks to manufacture speed.

Session complete: all changes logged; no source files/data/checkpoints/results
deleted, one verified-unused class removed with recovery archive, original89native
Python files unchanged from session start. No commit/push/stash/reset/index change.

Final receipt-check helper initially asserted status==complete, but worker status
uses its existing training-result vocabulary (completed update count and workflow
job status are the completion proof). Corrected this helper assertion, without
changing production or result files; both1120 counts, unchanged source/assets and
workflow completion records pass. Documentation links and scoped diffcheck pass.

Final Slurm step query initially used a job-only format field T (invalid for steps); default step listing used instead. No mutation or job cancellation. Final scoped diff --check exits0.

## Third allocation — 2026-11-22

User authorizes continued optimization on their new existing allocation20646979 (A100nodeg087), with lighting/material separation conditional on measured5x. No lighting implementation or ray-traced speed claim until that criterion. Preserve native source/data/history and existing dirty worktree. Initial allocation remaining1h58m. Source recovery archive experiments/hks_work_session_20261122_before_sources.tar.gz SHA256 e7254f43151cc2ece8c032bf930c5f9f9821a96795a656f21d8e289da16d49ad. 166 Python files archived; native hashes will be checked after edits. No new allocation requested.

Source experiment: existing spectral coefficient modules now contain a prototype
fused source gather/decay/normalizer/embedding forward and source-local adjoints.
Not enabled in production yet. Original source formula remains the reference.
The backward receives upstream gradients for ALL four outputs, including source
normalization and intrinsic embedding, and differentiates barycentrics, grid
weights and optional diffusion time. Archive tensors remain fixed. No learned
state cache, illumination code, new Python module or original HKTex edit.

Warm baseline probe runs on step20646979.1 with8minute limit,4CPUs/1GPU;
log experiments/hks_work_session_20261122_source_probe.log. CPU interpreter
forward/VJP probe is also running. Imports on this allocation/login host are
unusually slow; no timing from import/setup is called GPU kernel latency.
Exact warm-probe Python (launched through existing allocation srun):

```python
import os,time,json,statistics
from pathlib import Path
from hktex_splatting.utils.experiment_io import configure_cache
cache=configure_cache('source_probe_20261122',compiled_namespace='hybrid')
os.environ['MPLBACKEND']='Agg'
import torch
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import hktex
from hktex.utils import seed_everything
from omegaconf import OmegaConf
from hktex_splatting.training.uv_training import HksUvTrainer
from hktex_splatting.modules.spectral_heat_field import LiveHeatField
cfg=OmegaConf.create(json.loads(Path('experiments/hks_work_session_20261121_round2_hops16_full/hybrid_train/resolved_config.json').read_text()))
cfg.exp_root_dir=str(cache/'loader');cfg.trial_dir=str(cache/'loader'/'probe');cfg.exp_dir=str(cache/'loader')
seed_everything(0)
data=hktex.find(cfg.data_type)(cfg.data);data.prepare_data();data.setup('fit')
trainer=HksUvTrainer(cfg.trainer,data,renderer_cfg=cfg.renderer)
trainer.surface_options=dict(surface_support_hops=16,max_buffer_mib=768)
field=LiveHeatField(trainer,coefficient_backend='triton')
m,e,mesh=trainer.model,trainer.eigalbo_interp,trainer.mesh
torch.cuda.synchronize()
print('GPU',torch.cuda.get_device_name(),'sources',m.N_sources,'modes',e._eigen_vec_knn.shape[-1],flush=True)
def bench(name,fn,reps=40):
 for _ in range(5):fn()
 torch.cuda.synchronize();cpu=[];gpu=[]
 for _ in range(reps):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start=time.perf_counter();a.record();out=fn();b.record();b.synchronize()
  cpu.append((time.perf_counter()-start)*1000);gpu.append(a.elapsed_time(b))
 print(name,'cpu_median_ms',statistics.median(cpu),'gpu_median_ms',statistics.median(gpu),flush=True)
with torch.no_grad():
 vertices=mesh.get_face_vertices(m.kernel_face_ids).long();bary=mesh.cartesian_to_barycentric(m.kernel_locations,vertices)
 def select():
  c=e.cfg
  return e.knn_gather.select_grid(e._smp_coords_angle,e._smp_coords_scale,m.angles,m.anisotropies,abs_sin=c.grid_abs_sin,scale_map=c.grid_scale_map,grid_layout=e.GRID_CONFIGURATION,eps=c.grid_eps,clamp_unit=c.grid_clamp_unit)
 gi,gw=select();vals,phi=e.knn_gather.gather_src(gi,gw,vertices,bary,e._eigen_val_knn,e._eigen_vec_knn)
 def decay_norm():
  decay=torch.exp(-vals*m.diff_times[:,None]);return decay,(decay*phi*phi).sum(-1)+e.knn_heat.config.eps
 bench('bary',lambda:mesh.cartesian_to_barycentric(m.kernel_locations,vertices))
 bench('select_grid',select)
 bench('gather_src',lambda:e.knn_gather.gather_src(gi,gw,vertices,bary,e._eigen_val_knn,e._eigen_vec_knn))
 bench('decay_norm',decay_norm)
 bench('embedding',lambda:(e._iso_embeddings[vertices]*bary[:,:,None]).sum(1))
 bench('whole_source_no_grad',field.sources)
bench('whole_source_with_grad',field.sources)
params=[m._kernel_locations,m._angles,m._anisotropies]
def vjp():
 s=field.sources();loss=s.phi.square().mean()+s.decay.mean()+s.normalizer.mean()+s.embedding.square().mean()
 return torch.autograd.grad(loss,params)
bench('whole_source_forward_vjp',vjp,20)
print('DONE',flush=True)
```

Actual CUDA synthetic source gate PASSES N0/1/37/5000, modes11/31/256,
embedding5/7/64 and all four output-adjoint branches. Max forward errors at5000:
decay1.19e-7, phi2.38e-7, norm1.53e-5, embedding0. VJPs pass existing3e-5/3e-4
tolerances; no loosened checks. Empty source case passes. Log source_gate.log.
CPU interpreter nonempty test cannot execute CUDA libdevice extern; an attempted
interpreter-only monkeypatch also failed. Not claimed as CPU numerical pass.
CUDA kernel/math unchanged to accommodate interpreter limitations.
Automatic approval review REJECTED scontrol step20646979.1 limit extension8to15m
as unauthorized scheduler-limit modification. No limit changed, no retry/bypass;
leave original8minute diagnostic deadline. Allocation itself is unmodified.
SourceState production path now selects fused source kernel with Triton coefficients;
eager spectral source formula remains for the independent oracle. Next renderer
verification must pass before any long training. No source cache or detached
learned coefficient. Original89native sources remain unmodified.

Warm probe completed bary0.2785ms, grid selection0.2268ms, source gather0.3123ms
CUDA medians, then its diagnostic decay fixture failed indexing scalar model.diff_times
as a vector. Production scalar diffusion time is supported; this is a helper bug.
No complete-source latency or speedup inferred from this incomplete probe. CUDA
synthetic gate overlapped startup, so isolated whole-pipeline timing still required.
Same-update source geometry is now passed once from prepare_knn into the renderer;
identity/version/grad-mode mismatches fall back to fresh geometry, and the field
consumes it once. No reuse across optimizer steps or camera visibility cache.
Started full1285000source strict verification (not5000training) with fresh shared
initialization. No further source edits until worker source-provenance checks finish.
Syntax77/native89 preserved hash checks and scoped diffcheck pass.

Full128/5000source renderer verification COMPLETE: same-method RGB max2.384e-7,
all7parameter VJPs pass. Native field equality remains false and explicitly
approximate, with unchanged quality gate. CPU one-use geometry/regeneration tests
pass for repeatedcalls, coordinate version change, face version change, replaced
location Parameter and grad-mode mismatch; live location VJP stays connected.
No source edits while the following1120step density-transition trial runs.

Verified-renderer command:
```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_fused_source_verify.log --error=experiments/hks_work_session_20261122_fused_source_verify.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --verify-only --execute --output experiments/hks_work_session_20261122_fused_source_verify
```
1120step trial command:
```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_fused_source1120.log --error=experiments/hks_work_session_20261122_fused_source1120.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 1120 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --initial-state experiments/hks_work_session_20261122_fused_source_verify/native_verify/initial.pt --execute --output experiments/hks_work_session_20261122_fused_source1120
```

Source-only1120 trial complete: native/HKS training35.328424/18.567488s,
workers96.479225/81.504199s, aggregate PSNR20.286402/20.189563. All controls,
optimizer bindings, finite state and density1000/1100 checks pass; finalN5000.
This is a short trial, not a5000update or5x claim. Exact169sources archived in
hks_work_session_20261122_source_only_verified.tar.gz before further edits.

Investigated fusing selected RGB reconstruction and emission adjoints. GPU1/137/
4096pixels,129/257/32000pairs,37/73/5000sources and partial-gradient masks pass
against retained original two-kernel backward, tolerance unchanged. Inclusive
zero-allocation stage timings old/new .2673/.2642 and .2632/.2601ms; isolated
kernels old/new .2058/.2253 and .2089/.2028ms in opposite order. No consistent
meaningful gain; REJECTED prototype. Two affected backward modules restored
byte-for-byte from source-only archive; prototype preserved in
hks_work_session_20261122_rejected_backward_sources.tar.gz, not silently deleted.
The main implementation continues using its original validated backward.

Common benchmark instrumentation now disables tqdm progress refresh for BOTH
methods (original loop itself unchanged) and buffers the same JSONL trajectory
records. Flush at each128steps or parameter replacement; close on handled
exceptions. Metadata records training_progress=false and buffered_jsonl_v1.
CPU260-row buffered/fallback output byte equality and17-row exception flush pass;
TQDM_DISABLE still invokes loop calls but emits no terminal progress. No loss,
optimizer, gradient, density or checkpoint data removed. Subsequent timing must
label this COMMON instrumentation change separately from source fusion.

Started5000update unprofiled paired128comparison after source-only1120 passed; currentsource+common buffered instrumentation, fresh common initialization, same16hop support, original density/optimizers and original0.5dB initial gate. No GPU interference. No source edits until run finishes.
```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_fused_source5000.log --error=experiments/hks_work_session_20261122_fused_source5000.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261122_fused_source5000
```

Lighting condition remains NOT satisfied: no5x whole-training result exists.
Do not trigger it from a microkernel benchmark or an OOM native baseline. Report
unprofiled synchronized update interval and complete worker/process separately.
Any future separate-material lighting extension must be compared against original
ray tracing with matching scene, cameras, material/lighting targets, ray/sample
budget, bounce/shadow policy, denoising and quality evaluation. A fast unlit HKS
pass or raster primary visibility alone is not a matched lit/path-traced result.
No illumination/material implementation or new dependency added in this session.

Current source-only1120 trajectory still has a3.501second update1001 after
pruning; window median9.942ms. Native corresponding transition2.214s and median
17.210ms. Source gather fusion has NOT established that recompilation was removed.
Next candidate is the remaining compiled bilinear orientation/anisotropy grid
selector, but any replacement must preserve grid IDs/weights and boundary VJPs.
Buffered logging is common to both backends; its gains are not renderer-only gains.

Source fusion + common buffered logging5000update128pair COMPLETE.
Native/HKS intervals138.056581/64.917092s (2.12666x), worker187.537644/128.053529
(1.4645x). Two-view PSNR20.986603/20.516738 (0.469865dBdeficit, narrowly within
provisional0.5aggregate). Left21.886144/21.428682, right19.646018/19.161883.
Peak allocated13286290432/2828678144bytes, finalN5000. All four jobs complete;
sources unchanged throughout. This is a NEW paired node/initial receipt/control
run; do not attribute differences against the previous day's single run solely
to source fusion. Quality has shifted within accepted floating/adaptive trajectories.
No5x claim and no lighting added. Archive169 exact current sources in
hks_work_session_20261122_source_full_verified.tar.gz before grid-selector trial.

Prototype compiled source-grid replacement passes32CUDA cases: exact integer
four-grid IDs against original bilinear function, weights and angle/scale VJPs
within unchanged3e-5/3e-4 tolerance. Includes N0/1/37/5000, both layouts,
log1p/identity mapping, clamped/unclamped weights and actual axis endpoints.
Original native bilinear function AST is used as a torch.compile(dynamic=True)
reference without importing original CUDA-heavy HKTex initializer. No original
source edited. Reference derived verbatim math, annotations removed only for
standalone loading. Grid prototype not yet activated in renderer. Separate
opposite-order component benchmark now running, no other GPU job active.

Grid component opposite-order GPU medians at5000sources: native/fused forward
.28979/.09011ms and .28672/.08909ms; forward+VJP .46899/.26931 and
.46438/.26726ms. Consistent component gain, not a whole-training5x result.
Enabled fused source grid only for existing compiled bilinear4 recipe. Eager
source oracle and other native grid implementations still use original function.
No original HKTex code changed. Started paired1120 instrumented trial spanning
980..1107 for post-prune behavior, with fresh strict gates and shared initialization.
No source modifications while provenance workers run; stage timings diagnostic.

```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_grid_profile1120.log --error=experiments/hks_work_session_20261122_grid_profile1120.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 1120 --profile-start 980 --profile-steps 128 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261122_grid_profile1120
```

Combined spectral adjoint prototype: forward unchanged; combine shared
sum(grad_heat*mixed/norm) once, derive decay/phi adjoints afterwards, keeping
normalizer/grid/embedding adjoints. CUDA13pair independent Torch VJP passes;
600000pair/5000source synthetic test passes at unchanged tolerance. Synthetic
performance NOT consistent: separate/combined3.8799/3.0239ms then2.3265/3.0239ms
in reversed order. No improvement claimed or promoted from those numbers.
A watchdog confirms startup120s spent in torch imports; not GPU math latency.

Next is a frozen trained-checkpoint operator gate plus actual coefficient-adjoint
replay on real geometry/spectral data and loss adjoints. No training updates,
checkpoint overwrite, concurrent GPU timing or native-source edits. This is the
loaded final model treated as initial state of a new diagnostic process, not a
resume or bypass of a training gate. This will determine whether to retain the
adjoint prototype; source_full/grid_verified archives permit exact restoration.
Time override compatibility retained: per-mode, CPU-scalar and non-float32
explicit time overrides use the original eager source formula rather than failing
the narrow fast-source ABI. Default configured float32 scalar fast path unchanged.

Frozen trained-checkpoint GPU/eager operator gate PASSES with new grid/source
kernels, RGB max1.0133e-6 and all7parameter VJPs passing. Native equality remains
false (expected approximation), explicitly recorded. Actual coefficient replay:
106651pairs,63439nonzero heat-gradient entries. Direct separate/combined adjoints
match existing tolerances. Actual timing separate/combined1.9026/1.5437ms then
1.2575/1.5421ms; no stable gain and warm separate faster. This direct replay also
runs outside autograd.Function.backward's no-grad context, so it is not a precise
production backward latency measurement. Synthetic no-grad inputs showed the
same lack of consistent gain. REJECTED combined-adjoint prototype; restored only
two affected spectral modules byte-for-byte from grid_verified_sources archive.
Prototype saved hks_work_session_20261122_rejected_coefficient_sources.tar.gz.
Current code retains original separate coefficient adjoints. No optimizer/density
policy change, loss change, new module or lighting. Source/time compatibility
fallback remains. Only verified source/grid fusion and common instrumentation
are retained. Next unprofiled5000update128 and bounded192capacity/quality smoke.

Unprofiled grid5000 command (fresh source/GPU receipt):
```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_grid_full5000.log --error=experiments/hks_work_session_20261122_grid_full5000.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261122_grid_full5000
```

Grid-profile1120completed with all gates/source checks and density lifecycle:
HKS source_state0.4708ms, forward6.1950, backward2.5483, geometry1.3532,
coefficients1.1673, tilequeue0.9006, splat0.6357, support0.6883 (nested discovery
0.1799/intersection0.3725), camera1.2306, target0.5606, Adam0.8402,
Geodesic0.9960, trajectory/reset0.0724. Update1001 now15.019ms versus prior3.501s.
Both source grid and gathers are now runtime source-count kernels. These are
instrumented inclusive diagnostic values, not additive rows or an overall5x claim.
Separate grid5000 unprofiled comparison runs with fresh shared receipts; no source
edits while its workers execute. Frozen checkpoint probe passes all7VJPs too.

Grid5000 pair finished; all169source hashes match. Exact source snapshot grid_full_verified_sources archived before192trial. Starting40update192capacity/correctness trial, same settings and fresh native/HKS receipt; no OOM ratio or unvalidated resolution speed claim.
```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_grid_192_smoke.log --error=experiments/hks_work_session_20261122_grid_192_smoke.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 192 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261122_grid_192_smoke
```

Integer-only tile-task expansion replaces repeated entry/patch tensors with a
fixed-face prefix scan and one Triton expansion. Original stable ordering/sort,
tile grouping, bounds, emission and backward remain. GPU bitwise keys/ranges/
count/entries/pixel-offset checks pass for zero entries, zero tasks, variable1..8
patches and75000faces/619776pairs/1560106tasks. Interleaved old/new benchmark
medians1.93894/1.37062ms GPU, wall1.96618/1.39646; synthetic workload, not total
training. No floating field values/selection/gradients changed by this scheduling.

192smoke40updates completed both methods before long run; native allocated25.123GiB,
HKS2.530GiB. Short intervals2.221216/.611369s, PSNR18.58327/18.87397. These short
cold/compile-sensitive intervals are not advertised as a final speedup. Starting
192full5000 with exact scheduling optimization, same hyperparameters, fresh
receipts and mandatory verification jobs before training. Native unchanged.

Source autograd dependency flags now match original formulas under frozen inputs:
decay depends on weights/time, phi on barycentrics/weights, norm on all3, embedding
on barycentrics. No default-training dependency/math change. CPU shape-only
stub checks all8requires-grad combinations, NOT a GPU numerical test. Existing
CUDA all-input forward/VJP and frozen-checkpoint full7parameter checks stand.
Extended time override CPU value/VJP equivalence checks PASS. No new tests/modules.

```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_tile_192_full5000.log --error=experiments/hks_work_session_20261122_tile_192_full5000.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 192 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261122_tile_192_full5000
```

192full5000 COMPLETE: native/HKS training275.505030/70.415079s (3.9126x),
worker330.060373/136.895602s (2.4111x). PSNR22.143969/21.854598 (0.289371deficit),
left22.932769/22.566468, right20.934772/20.741155. BothfinalN5000 and density
count changes1000/1100; source/data checks pass. Peak allocated26976602624/
2914848768bytes; forward20.227708/7.849580ms with saved supplied visibility.
169source hashes checked and archived in tile_192_verified_sources. This is a
real matched unlit192result, not5x or a ray-traced comparison. Larger resolution
shows improved relative scaling; do not generalize to64/128or all scenes.

Starting bounded22440update capacity/quality smoke to map the upper fitting
resolution between192(success) and256(previous native OOM). Same field, policies,
sourcecount, quality gate, byte/record guards andGPU. No OOM treated as speedup,
no limits weakened. Only after both full-pixel40step workers pass may a long
224comparison start, subject to remaining allocation. No illumination added.

```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_tile_224_smoke.log --error=experiments/hks_work_session_20261122_tile_224_smoke.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 224 --kernels 5000 --iterations 40 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261122_tile_224_smoke
```

224smoke correctly stops: native full-pixel training fails at step3 backward
with CUDA OOM requesting8.90GiB, free8.45GiB, allocated17.86GiB plus11.12GiB
reserved/unallocated. Failure sources_unchanged=True. This is allocator/memory
capacity evidence, NOT a finite speed ratio. HKS224training was not launched;
no5x or successful full224comparison claimed. Default allocator unchanged.
A future common expandable-segment allocator trial could investigate fragmentation,
but must record that setting for BOTH arms and rerun gates. Not attempted here.

Final source polish: module descriptions name source/grid operators accurately;
source/camera adapter doc corrected. Integer expansion retains original vectorized
builder for unusually coarse task incidence (>8tasks per pair), without dropping
support or adding any kernel-count cutoff. GPU fast-path arrays already bitwise
validated; CPU interpreter checks include empty, ordinary and coarse fallback.
Final1285000paired run starts with these final source files and all usual gates.
No further Python source modifications until all jobs finish.

```bash
srun --jobid=20646979 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=00:15:00 --output=experiments/hks_work_session_20261122_tile_128_final5000.log --error=experiments/hks_work_session_20261122_tile_128_final5000.log env PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -m splatting_script.experiments.train_surface --backend both --supervision camera --visibility fresh --resolution 128 --kernels 5000 --iterations 5000 --seed 0 --raster-backend nvdiffrast --target-sampling gpu --config-overlay splatting_script/configs/converse_boundary_stop.yaml --shared-initial-model --surface-adapter direct --surface-support-hops 16 --max-buffer-mib 768 --corner-evaluation parallel --support-cap 0 --density-policy original --optimizer-policy original --allow-native-approximation --max-initial-psnr-drop-db 0.5 --no-training-plots --verify-samples 1024 --execute --output experiments/hks_work_session_20261122_tile_128_final5000
```


## Third allocation: final results and handoff (2026-11-22)

Final1285000 completed all four verification/training jobs. All169source hashes,
initial model/RNG/sampler/evaluation identities, controls hashes and camera-metadata
sequence hashes match the paired protocol. Sources/assets unchanged during runs.
Native/HKS training149.365777/61.052881s (2.4465x), workers222.417807/125.115782s
(1.7777x). Own-renderer PSNR20.997739/20.568081dB (0.429657deficit); left21.884894/
21.522568 and right19.671492/19.164941. Right deficit0.506551dB must not be hidden
by the aggregate metric. Native evaluation of the HKS-trained parameters gives
19.438914dB; it is a counterfactual field evaluation, not the baseline result.
Allocated memory10966796800/2828041216bytes; reserved18402508800/12209618944.
Forward medians10.726859/5.739701ms exclude raster/targets/loss/backward.
Both5000updates/final5000kernels, original policies. Native and HKS verification
pass, but HKS status APPROXIMATE_COMPLETE and performance_claim_validated=false
because strict native-field equality remains false. Timings are measured evidence
for this explicitly approximate workload, not proof of exact equivalence.

Reports: experiments/hks_work_session_20261122_tile_128_final5000/comparison.md
and experiments/hks_work_session_20261122_tile_192_full5000/comparison.md. Raw
result.json, manifest.json, receipts, JSONL trajectories and checkpoints retained.
No timing from the instrumented1120trial was used as an unprofiled speed claim.
The192comparison is a pre-final-polish snapshot; it is not attributed to identical
final128source bytes. No additional frozen-checkpoint experiment was launched at
the end of the allocation; the earlier trained-checkpoint GPU/VJP gate and final
initial-state gates are the actual checks performed.

### Retained source changes in this allocation

- hktex_splatting/triton/spectral_heat_coefficients.py: runtime-source-count
  bilinear grid/source preparation and matching adjoints; original coefficient
  kernels retained.
- hktex_splatting/modules/spectral_heat_coefficients.py: autograd wrappers,
  input-gradient dependency flags and narrow fast-path validation.
- hktex_splatting/modules/spectral_heat_field.py: fast source dispatch, single-use
  live geometry with identity/version guards, original eager time-override fallback.
- hktex_splatting/rendering/heat_kernel_splatting.py: backend-aware source oracle,
  integer task expansion with original coarse-task fallback.
- hktex_splatting/triton/heat_kernel_splatting.py: integer task expansion;
  existing splat emission and backward unchanged.
- hktex_splatting/training/surface_adapter.py: consume current-update geometry.
- hktex_splatting/training/uv_training.py: prepare guarded geometry; buffered
  unchanged trajectory records, original append fallback preserved.
- splatting_script/experiments/train_surface.py: same disabled terminal progress
  and buffered trajectory policy recorded for both competitors; original loop
  and optimizers unchanged.

Documentation modified: hktex_splatting/README.md,
docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md, docs/HKS_WORK_LOG.md. No source file
was deleted or moved; no new Python module/package dependency. Rejected pixel
backward fusion and combined coefficient-adjoint prototypes are archived and
restored out of production because timing gains were not consistent.

### Verification and recovery

CUDA source and grid forwards/VJPs pass existing tolerances, including5000sources;
grid IDs exact. Integer tile task queues match the old implementation bitwise on
empty, variable-patch and1.56million-task cases. CUDA renderer seven-parameter
VJP gates pass. CPU mutation/invalidation, time-override value/VJP compatibility,
logging byte-equivalence and exception flush pass. Integer Triton CPU interpreter
passes ordinary/coarse/empty scheduling cases. Autograd freeze-mask CPU checks
use a shape-only stub and are not GPU/math evidence. CUDA libdevice CPU interpreter
attempts failed unsupported extern execution; not reported as numerical passes.
Final CPU syntax77modules, native89byte hashes, CLI/help/workflow preview and local
README/architecture link checks pass. Actual native/HKS production imports and
training executed on A100; a CPU-only native import requires CUDA Mitsuba and is
not claimed as passed. No persistent test infrastructure was recreated.

Exact final169-source archive:
experiments/hks_work_session_20261122_final_verified_sources.tar.gz
SHA256 00d4fc3d44aec95ac6d2ea490f3629a82dfe9618de05a0534e9468d74ea6e193.
Contains SOURCE_MANIFEST.json and matches final128manifest source hashes. Earlier
archives, results, failed trials and recovery records preserved. To inspect or
recover, extract archives into a separate temporary directory first and diff the
specific file against the working tree; do not extract over existing dirty work.
The session-start before_sources archive preserves pre-edit Python bytes.
Git remains extensively dirty from earlier work; no staged changes discarded.

Remaining priorities: independent repeated seeds/meshes and trained-checkpoint
quality/VJP checks; optimize measured coefficient/geometry/backward work without
changing the field; investigate native memory fragmentation under a common,
recorded allocator configuration before higher-resolution comparisons. No5x
claim, lighting feature or ray-traced performance comparison is established.

Automatic approval review rejected an attempted8-to15minute diagnostic-step
extension earlier in this allocation as a scheduler-limit change. No scheduler
limit was changed, and the rejected action was not retried or bypassed.

Final handoff verification: first syntax recheck accidentally used system Python3.6,
which cannot parse supported Python3.11 starred return syntax; this was an
interpreter mismatch, not a source regression. Re-ran with the actual HKTex
Python3.11.13: all77modules syntax PASS, native89bytes PASS, both documentation
link scans PASS, all169final source hashes PASS. Scoped git diff --check PASS.
Final squeue lists only the user shell and extern step; all our GPU steps exited.
The existing allocation was left untouched. No final Python edits after the
verified benchmark or archive.


## Explicit lighting request (2026-11-22)

User subsequently requested “okay lets add lighting”, removing the earlier5x
condition for implementation. Implemented optional known-light analytic shading
using existing files only. Original hktex/89files byte-identical; no source/math,
optimizer, density, saved model-class path or dependency changes under hktex/.
No commit/push, new allocation request, GPU run, file deletion or test suite added.
The prior allocation had only41seconds left when checked; no new work was started
there. GPU lighting integration remains unvalidated.

### Implementation and changed files

- hktex_splatting/rendering/reference_mesh.py: shade_material/shade_batch.
  sRGB material -> linear albedo, Lambert diffuse, optional isotropic GGX with
  separable Smith masking and Schlick Fresnel; distant white world-space light,
  ambient diffuse fill. Roughness uses alpha=r^2, fixed F0/roughness controls.
  No HDR clipping/tone mapping. `none` returns the exact original tensor.
- hktex_splatting/training/camera_supervision.py: precompute fixed outward face
  normals, transfer fixed normals once per device/dtype, recompute current view
  directions each camera frame, shade original texture targets with known light.
  Dataset target timing includes shader host/enqueue time; synchronization stays
  at existing training/worker scopes. No learned shading or visibility cache.
- hktex_splatting/training/uv_training.py: native adapter post-reconstruction
  shader, retaining original auxiliary outputs and inherited optimizer loop.
- hktex_splatting/training/surface_adapter.py: HKS post-reconstruction shader
  after original sample order restoration, trace and normal paths alike;
  material_lighting stage nested inside forward_knn during profiling.
- hktex_splatting/training/training_checks.py: independent native/eager oracles
  and native candidate replay apply the same shader before RGB/VJP checks;
  no tolerance/acceptance change. Contribution weights/top30 remain material
  field operations; shader gradients feed existing field backward operators.
- splatting_script/experiments/train_surface.py: none/diffuse/ggx flags; finite
  directional light, ambient/intensity, roughness/F0 validation; camera supervision
  required for lighting. Propagate settings to both verification/training workers,
  recorded controls, result.json, saved checkpoint and Markdown report. Quality
  and forward timing include shading. Lit loss/PSNR use unclipped linear radiance,
  PSNR reference peak1. Old unlit snapshot provenance deliberately not reusable.
- hktex_splatting/README.md: configuration, full short paired smoke command,
  color space, fixed material controls, assumptions and GPU-validation status.
- docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md: optional shader/data/oracle stage.
- docs/HKS_WORK_LOG.md: this record; previous measurements preserved as unlit.

### Checks actually performed

CPU PyTorch2.10.0+cu129, Python3.11.13:
- Shader none identity, analytic Lambert front/back values, double-precision
  gradcheck through albedo for diffuse/GGX, empty tensors, grazing/back-facing/
  degenerate geometry finite values/VJPs: PASS.
- GGX highlight changes for a changed camera direction: PASS.
- Actual CameraSupervision.shade_target method on synthetic geometry: target/
  prediction parity, target no-grad, finite positive albedo gradient, updated view
  direction, fixed-normal transfer reuse, validation_subset lighting geometry,
  default-mode identity/no added geometry: PASS. Constructor/raster/GPU path
  not executed by this CPU method probe.
- Native adapter actual forward_knn method extracted with AST and a minimal
  parent stub: shading once, unchanged auxiliary output ABI, unlit identity PASS.
  This is not a CUDA native-production import or full trainer execution test.
- Existing reference mesh code AST unchanged except module docstring and two
  shader functions: PASS. Initial harness assertion included the intentionally
  updated docstring and failed; corrected harness to exclude only that docstring,
  then the implementation comparison passed. No source behavior fix was needed.
- Three CPU CLI/workflow previews none/diffuse/ggx propagate identical light
  settings to all4native/HKS verification/training workers, gpu_started=false.
  Lit report labels analytic/not ray tracing, linear radiance: PASS.
- Five bad-control cases (surface supervision with lights, zero direction,
  NaN intensity, zero roughness, F0>1) reject at argparse with exit2: PASS.
- All77HKS/script modules parse; docs local links and scoped git diff --check
  pass. Native89Python bytes match original session-start recovery archive.
- Compared169source manifest hashes against final unlit128run: exactly the6
  expected Python files above differ. Heat equations, selectors, Triton spectral/
  splat/backward code and optimizer implementations remain byte-identical to that
  final unlit snapshot. No performance timing from this CPU work is claimed.

### Recovery and remaining work

Before-lighting6-file archive:
experiments/hks_lighting_before_sources_20261122.tar.gz
SHA256 69558873987f96517325b2b93931f85581a80f4bbdde690a596ea5ad03728ea5.
Original full169source unlit archives/results retained. Inspect recovery archives
in a separate temp directory before selectively restoring; do not overwrite the
existing dirty worktree. No reset/stash/staging/commit/push.

Run the README64x64paired four-update smoke in an existing CUDA allocation;
verification jobs must pass before training. Then use independent lit runs/seeds
and held-out cameras/lights for quality/performance. Light and material controls
are fixed/user-selected, not estimated from photographs. The original texture is
assumed albedo and may itself contain baked light. Flat mesh winding normals may
show facets or inward-facing regions; no automatic normal repair was introduced.
There are no shadow rays, environment map, secondary illumination, learnable
roughness/lighting, metallic material or inverse-rendering objective. This is
not a complete radiance-field or ray-traced experiment. Previously measured2.45x/
3.91x gains remain unlit and do not establish lit performance. No5x claim.


## Piggy mesh GPU smoke tests (2026-11-22)

User requested “ok test”. Found existing A100 allocation20648652, nodeg093;
no allocation requested. Ran two bounded steps using srun --jobid20648652
--overlap --exact --nodes1 --ntasks1 --cpus-per-task4 --gresgpu:1 --time00:05:00,
HKTex Python3.11, OMP_NUM_THREADS1 and environment library path. No spectral
preprocessing, HKS/native training, source edits or long benchmark launched.
Both steps exit0. Original piggy GLB SHA256 remains
3a3007dfdaddf9c3879a0fc9f3fb8296daec7ab06218c76e5f086c46eb41b091.

1. experiments/piggy_lighting_smoke_20261122/result.json and matching .log:
   PASSED_MESH_SHADER_ONLY. Actual transformed scene geometry31123vertices/
   45238triangles. Normalize only in memory; fresh CUDA rasterization of two
   64x64views,886/950visible samples. Existing synthetic raster setup check passes.
   GGX source material roughness0.112565 for body; ideal-smooth black material
   roughness0 is clamped to shader minimum0.05 for this diagnostic. Source glTF
   linear base-color factors converted to sRGB for shader input; F0 fixed0.04.
   RGB/material VJPs finite and nonzero, changed light direction changes output.
   Diagnostic previews view_0.png and view_1.png saved; linear output clipped to
   unit range only when saving display PNG. Peak linear radiance37.93 in first
   view is a sharp glossy highlight, not clipped in the math; this does not prove
   HDR training quality. Viewed first PNG; actual piggy silhouette and highlights
   visible. No HKS/eigensystem/model execution in this diagnostic.
2. experiments/piggy_camera_smoke_20261122/result.json and matching .log:
   PASSED_CAMERA_TARGET_LIGHTING. Actual original loader (existing isolated CPU
   load_cpu_mesh helper) yields22639welded vertices/45238faces and preserved UVs.
   CameraSupervision and TargetTexture CUDA constructors execute; two fresh views,
   889/936samples, CPU/GPU raw target RGB EXACT; shader prediction/target EXACT;
   material-color VJP finite/nonzero;32sample validation_subset retains matching
   normal/view/screen geometry. Camera frames2/raster calls2. Global fixed shader
   roughness0.112565/F0.04; not automatic per-material BRDF loading or fitting.

Correction to earlier assumption: although raw piggy has no image textures, the
unchanged original loader's Trimesh concatenation creates a16x8PBR texture atlas.
No new material-color loader is required for this asset. CPU atlas inspection
found RGBA[231,148,152,255] pink, black, and transparent padding; baseColorFactor
None on merged material. This is consistent with converted source linear factors.
Source material roughness is not automatically preserved by the training shader;
choose fixed material controls explicitly. No need to edit original HKTex loader.

Full piggy HKS training still requires its OWN eigensystem and a matching config,
cameras, source initialization and field forward/backward gates. Never reuse
Converse spectral data. These tests exercise material/shader gradients, not
kernel-position/anisotropy VJPs, optimizers, density or timing parity. No scene
reflections, shadow rays, ray tracing or speed claim. GPU shader/target checks do
not validate full native/HKS lit training. README and architecture validation
wording updated accordingly. Research assets/source files untouched; only this
log/docs and diagnostic output artifacts written. No commit/push.


## Piggy independent spectral/training setup (2026-11-22, in progress)

User requested “continue” after mesh/shader tests. Existing allocation20651097
on A100nodeg011 had1h59m remaining; no new allocation requested. Inspected dirty
Git status; no reset/stash/staging/deletion/commit/push. Source recovery archive:
experiments/piggy_training_setup_before_20261122.tar.gz, four pre-edit Python files.

Created splatting_script/configs/piggy_training.yaml from the existing original-
policy Converse recipe, changing mesh/spectral paths only. New camera JSON:
data/piggy/cameras.json, four views on unit-normalized piggy, same historic view
names as command aliases; two held-out views derived by the existing rotations
before training. No hyperparameter/quality-threshold tuning on held-out results.
Original raw GLB kept in data/converse/raw/piggy_bank.glb; exact byte copy created
at data/piggy/prepared/piggy_bank.glb. Converse assets untouched.

Modified existing prepare_assets.py with explicit --asset converse|piggy,
independent fixed checksum/config/camera/spectral paths, same atomic non-overwrite
copies and same original EigenAlboInterpolation calculation. Default Converse
paths/checksum remain. rendering/cameras.py and CameraSupervision accept an
optional camera file. train_surface --cameras-file propagates to both workers,
records its contents as camera definitions and checks its file hash for mutation.
Old camera defaults remain byte-for-byte equivalent as dictionaries.

CPU checks passed: existing Converse inputs/checksum, default camera equality
against pre-edit source, six piggy views finite and valid, both worker commands
use the requested piggy config/camera JSON, preview gpu_started=false, all77
Python modules syntax. Only missing piggy prerequisite was its spectral archive.

Launched bounded preprocessing (45minute step limit, existing allocation only):

```bash
srun --jobid=20651097 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=00:45:00 --output=experiments/piggy_precompute_20261122.log --error=experiments/piggy_precompute_20261122.log env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 LD_LIBRARY_PATH=/scratch/user/franciscotorres2124/.conda/envs/hktex/lib /scratch/user/franciscotorres2124/.conda/envs/hktex/bin/python -B -u -m hktex_splatting.prepare_assets --asset piggy --precompute --execute
```

The original configured256eigenmodes,30degree orientation grid and anisotropies
[1,5,15,30,60,100,200] are retained. No coarse test archive masquerades as the
full configured eigensystem. Preprocessing writes privately and publishes the
final archive atomically only after success. Waiting for result before training.
No long5000update run is authorized by this continuation's short-check plan.

Piggy preprocessing completed successfully. Archive finite/shape/mass checks PASS:
49anisotropic entries,256modes,22639vertices, float32 eigenarrays and float64
local directions. Archive SHA256 bb1d5d492eb4f5b2a04212650b55f8afd437794aab076a87b51365255fc2a1eb.
Mesh SHA256 3a3007dfdaddf9c3879a0fc9f3fb8296daec7ab06218c76e5f086c46eb41b091.
Camera SHA256 6879ec4e24119b083ae08709bf4a0f3e0b8525069143f32542a4359c29af07a2.
Config SHA256 783ce381a167cf9001347e6aae907cc0a21eb7768063d02df05ef0ed1813f1c8.
Receipt data/piggy/prepared/preprocessing.json retained locally. Added .gitignore
rule for new prepared binary/archive directory; piggy camera/config remain visible.

Topology probe:22639vertices/45238triangles,3connected components, not watertight,
zero near-zero-area faces, all atlas UV triangles constant/degenerate (material
lookup still exact as prior shader test demonstrated). Original UV-based placement
initialization returns without placement changes when no valid UV triangles;
no remeshing or change to native initialization is introduced. NumPy probe emitted
a2D-cross deprecation warning, not a correctness failure.

Started fresh paired64x64/1000kernel/four-update GGX smoke, source body roughness
0.112565, F0.04, original density/optimizers, fresh CUDA cameras/targets, strict
same-method gates and explicit native approximation bound0.5dB. Log/output:
experiments/piggy_lit_smoke_20261122.log and matching directory. Steps run
verification for both methods before either training worker. No source edits
until all provenance workers finish.

First piggy full-trainer smoke FAILED in unchanged native initialization before
verification/forward: np.random.choice raises “Fewer non-zero entries in p than
size” on the16x8constant-material atlas. No HKS worker or training launched.
Failure sources_unchanged=True. Previous note about native no-valid-UV return
occurs only after weighted candidate sampling; this earlier exception is the
actual first failure. No assertion or tolerance was loosened.

Archived failed-run config/cameras in
experiments/piggy_initialization_failure_sources_20261122.tar.gz. Changed only
piggy's original supported data_initialisation_random_ratio from0.5to1.0 so both
methods use surface-random placement; no native Python modification. Canonical
config comparison confirms only that field differs. Spectral settings identical,
so existing archive remains valid; its generation receipt retains ORIGINAL
config hash783ce381a167cf9001347e6aae907cc0a21eb7768063d02df05ef0ed1813f1c8.
Current training config SHA256
61a08fbcde537d14bb41efe79e5e1dc73cf6edad531cc38f86e908c648f51790.
Fresh retry output/log experiments/piggy_lit_smoke_retry_20261122; all ordinary
initial identity and strict operator gates remain. No old snapshot reused.

Piggy retry four-update smoke COMPLETE: native VERIFIED/COMPLETE, HKS
APPROXIMATE_VERIFIED/APPROXIMATE_COMPLETE; all four jobs exit0. Strict HKS
GPU/eager RGB max1.758337e-6, seven learned-parameter VJPs pass (locations max
6.705523e-8); candidate-replay GPU/eager max1.251698e-6 passes. Native field
identity remains FALSE, including same-candidate native replay RGB max0.242904;
this is not hidden under an exact-equivalence claim. Initial target PSNR drop
-0.008902dB is within explicit0.5dB bound. Both run4updates/final1000kernels,
all finite parameters/optimizer bindings/source+asset preservation pass. Shared
initial identities, controls hash8f22c5a1757c7d4a2544281a1a8a975b996d170a845ebcc8b3a4193faa6891d1
and camera sequence match. Held-out linear PSNR native12.005322/HKS11.999109dB:
these are barely trained parameters, not final quality results. Training intervals
2.043336/.133525s are short cold-start-sensitive numbers, not a validated15x or
5x benchmark. Original .log/result.json/manifest/checkpoints preserved.

To cover actual original density/pruning lifecycle, launched fresh paired1120
update64x64check under same settings/strict gates. No5000update run, no renderer
or threshold change. Output/log experiments/piggy_lit_lifecycle_20261122. Both
native and HKS must verify before training; source files remain frozen throughout.


### Piggy lifecycle and frozen-checkpoint results (completed)

experiments/piggy_lit_lifecycle_20261122 completes native/hybrid verification
and1120training updates. Source/asset integrity, all finite parameters and
optimizer binding checks PASS. Shared initial identities, controls and camera
metadata sequence match. Both original policies remain; no custom optimizer or
controller installed. Native count changes at400/500/600/1000; HKS at400/500/600/
700/800/1000. Final counts816/808. These demonstrate both growth/pruning/rebinding
lifecycle paths, not a fixed-count renderer microbenchmark.

Native/HKS synchronized intervals21.861877/14.303084s; complete workers44.481693/
42.074935s. Preliminary ratios1.52847/1.05720, NOT5x. Held-out own-renderer
linear-radiance PSNR30.057704/29.256024dB, deficit0.801679dB: FAILS provisional
final quality target0.5dB. Left32.130215/30.042771, right28.408267/28.458011dB.
HKS parameters evaluated with native field give29.377912dB, a counterfactual,
not the independent baseline. Thus the aggregate gap is not solely the final
querying renderer; no causal split or matched-quality performance claim follows.
Peak allocated2426573824/1386439680bytes (2.26/1.29GiB); reserved2478833664/
2434793472bytes, PyTorch only. Raw source-controlled report/result JSON retained;
there was no profiling instrumentation. This is1120updates on one scene/seed/
GPU, not the full5000update benchmark or a ray-tracing comparison.

Frozen post-training diagnostic:
experiments/piggy_frozen_check_20261122/result.json, matching .log, per-HKS
sample_validation.json, heldout_left.png and heldout_right.png. Fresh trainer
objects load final checkpoints with exact model-state hashes verified, including
source counts816/808 and saved source barycentric metadata. No optimizer or
training updates. Native/HKS per-view MSE reproduced saved evaluation exactly;
parameters unchanged after calls. HKS GPU/eager RGB max3.576279e-7, all7learned
parameter VJPs pass (locations max4.097819e-8). Strict native-field equality remains
false and is recorded separately; no tolerance or native check was disabled.
The loaded trained HKS model is treated as initial state only of this separate
diagnostic object for candidate replay, never as a resumed training run.

PNG panels: Target | native | hybrid. Linear HDR radiance is converted to sRGB
and clipped only for display, not the quality calculation. Viewed left PNG to
confirm actual piggy rendering. Fixed light/F0/roughness, flat mesh normals,
known analytic targets; no fitted lighting, environment reflections, shadow rays
or indirect illumination. An unused extra dataset frame is fetched during probe
setup; there are no optimizer updates or timing claims from this diagnostic.

### Final changes, validation and recovery

Modified four existing Python files in this continuation: prepare_assets.py,
rendering/cameras.py, training/camera_supervision.py,
splatting_script/experiments/train_surface.py. New source inputs only:
splatting_script/configs/piggy_training.yaml and data/piggy/cameras.json.
.gitignore now excludes data/piggy/prepared/** binary/generated assets, leaving
camera JSON visible. Updated HKS README, architecture and this work log.
No deleted/moved file, new Python module, native HKTex edit, new dependency,
commit/push, reset/stash, staging change or allocation request.

CPU checks: all77Python syntax, default Converse input/checksum and camera
identity, custom six-camera geometry, native/HKS worker flag propagation and
CPU preview, identical spectral settings after initialization config fix,
atomic copy/idempotence/checksum rejection/existing-target preservation PASS.
Piggy archive finite/all shapes and positive masses PASS. Original89native
Python bytes are checked against original session-start archive. Scoped
whitespace/link checks finish at handoff. Earlier failed initialization logs
and before-edit recovery archives remain; no failed run overwritten.

Exact verified169-source snapshot plus piggy config/cameras and boundary overlay:
experiments/piggy_verified_sources_20261122.tar.gz
SHA25693c5809fd1ecb42fed0ce76267abd642f7821c9212a6bce596f817132c23dcf2.
Includes SOURCE_MANIFEST.json. Original source recovery archive for this
continuation is piggy_training_setup_before_20261122.tar.gz. Failed-init config
is piggy_initialization_failure_sources_20261122.tar.gz. Inspect/extract into a
separate temporary directory before selective restoration, preserving current
staged/unstaged work. All raw models, mesh copies, spectral data and research
results remain. User allocation left intact after our steps exit.

Remaining: reduce the observed held-out quality gap before claiming matched-
quality lit gains; independent seeds/resolutions and a full5000update run only
after the short quality target is satisfied. In particular, same-candidate native
replay differs even though GPU/eager agrees, so candidate coverage alone cannot
be assumed to explain the gap. No spectral/mathematical change was made to hide
that issue. Full ray-traced comparisons require matched BRDF/light/shadow/
integrator and color-space controls; this direct-light shader does not supply
those features.


## Continued optimization: fused material lighting (2026-11-22, in progress)

User requested continued optimization. Existing allocation20651097 still used;
no new allocation, native edit, cleanup deletion or source-history change.
Added diagnostic camera_target_lighting wrapper in surface_profile.py so target
shader dispatch is separately measured. HKS-only1120profile,64steps980..1043,
completed at experiments/piggy_lighting_profile_before_20261122. Mean inclusive
ms/call: targetlighting0.9043, material_lighting0.8597 (nested in forward5.7426),
backward1.8186, samplegeometry1.2243, coefficients0.4394, source0.4511,
support0.7241 (nested discovery0.1802/intersection0.4080), tilequeue0.7512,
splat0.3518, raster1.1109, targetlookup0.5634, Adam0.8557, Geodesic0.9814.
Do not sum nested stages or use instrumented whole-run time as a speed claim.

Profiled HKS PSNR30.170944/final811sources differs from prior unprofiled HKS
29.256024/final808 despite EXACT same initial model/RNG/sampler/evaluation
identities. No field operation changed, only instrumentation. This demonstrates
training/density sensitivity/nondeterminism; one-run quality changes cannot be
causally attributed to a new optimization. Repeated unprofiled matched trials
and frozen-model operator comparisons are needed.

Recovery before new shader: experiments/piggy_lighting_optimization_before_20261122.tar.gz.
Implemented in existing reference_mesh.py, splat_autograd.py and
triton/splat_reconstruction.py: one fused CUDA forward, one analytic albedo VJP
kernel; same Lambert/GGX/Smith/Schlick/sRGB equations. Lighting/roughness/F0 are
runtime inputs, not learned-state caches or source-count specializations. CUDA
float32 with contiguous FIXED normal/view inputs uses auto fast path. CPU/double,
geometry derivatives/noncontiguous geometry retain original Torch graph. Fused
path first-order only, same active field limitation; --lighting-backend torch
preserves full Torch behavior. Both competitors receive the SAME backend flag
and its setting is recorded in controls/checkpoints. Eager HKS oracle explicitly
forces Torch shading, ensuring new shader math is independently checked.
No heat-kernel, selector, top30, optimizer or density algorithm was changed.

Initial fused test FAILED at original tolerance for sharp roughness0.05,N1024:
3RGBvalues differed, max0.000997901, relative0.00115994. Preserved failed JSON/log
and kernel snapshot piggy_fused_lighting_failed1_sources_20261122.tar.gz. Did NOT
loosen tolerance. GPU reduction diagnostic found Torch uses (x+z)+y, not (x+y)+z,
for contiguous three-component norm/dot reductions. Tested1/37/1024/5000/65536rows;
Torch norm021 mismatch0 for all, sum021 mismatch0. Matched this reduction tree,
rounded square-root/division and disabled FMA contraction in shader launch.

Retry experiments/piggy_fused_lighting_gate_retry_20261122 PASSES50cases at
unchanged atol3e-5/rtol3e-4: empty/1/37/1024/5000/65536pixels, diffuse/GGX,
roughness0.05/0.112565/0.35/1, black/sRGBthreshold/white, degenerate/backfacing
geometry, random VJPs and actual piggy normals/views. Actual piggy RGB max
2.384186e-7. Max all-test RGB6.103516e-5 occurs at HDR values within relative
bound; max albedo VJP1.430512e-6. Geometry-gradient auto fallback matches Torch
exactly. Warm4096pixel component medians in reversed orders: Torch forward
0.81247/0.80737ms vs fused0.07172/0.06604; fwd+VJP1.22888/1.21822 vs
0.27708/0.27279ms. These are COMPONENT timings, not whole training speedup.

experiments/piggy_fused_lighting_frozen_20261122 PASSES new shader on existing
trained808source checkpoint. Fresh diagnostic explicitly records six intentional
source changes; never bypasses benchmark initial-state provenance or resumes
training. Seven parameter GPU/eager VJPs pass; exact native field equivalence
still false. Frozen per-view MSE0.000990200206/0.001426261035 vs original
0.000990199973/0.001426260569 (within5e-10). Model/source bytes unchanged during
probe. Later guard polish routes noncontiguous normal/view geometry to Torch.
Initial layout probe incorrectly assumed transposed1x3tensor was noncontiguous;
zero-tolerance assertion failed7.45e-9 on a valid fused path. Corrected fixture to
slice [N,6][:,::2], preserving EXACT assertions for genuinely strided fallback;
this is a harness correction, not a relaxed shader tolerance.


### Controlled shader trials and exact camera packing

CPU shader identity, float64 albedo/geometry gradcheck, Torch higher-order fallback
and both CLI backend propagation previews PASS. Layout retry with truly strided
geometry passes bit-exact Tensor fallback and original fused RGB/VJP tolerances.
No CUDA interpreter numerical test is claimed for libdevice operations.

Sequential unprofiled1120update64x64/1000initial-source pairs, same source snapshot:
- piggy_shader_torch_1120_20261122: native/HKS intervals25.831015/14.459537s,
  workers47.642220/41.980077s, PSNR30.265303/30.022579, final829/784kernels.
- piggy_shader_auto_1120_20261122: intervals21.795279/12.739735s, workers42.829499/
  40.045376s, PSNR30.109694/30.294795, final831/796kernels.
Both methods receive the same shader setting WITHIN each pair. Across shader
settings, initial parameter/RNG/sampler identities match; evaluation hashes may
differ by tested floating shader rounding. No old initial manifest was reused
across different controls. Adaptive count/quality divergence and nondeterminism
prevent attributing every timing/quality difference purely to the shader.
Measured shader-only HKS interval reduction is1.135x (~11.9% less elapsed time),
not a5x whole-training improvement. Separate fixed-geometry shader benchmarks
provide the direct component evidence. All raw reports/JSON preserved.
Exact shader-only source snapshot:
experiments/piggy_fused_shader_verified_sources_20261122.tar.gz
SHA2560c9f24f6b61c0d17d65a9b2df36377e8a91d4aa647f10fbbdd5db44888bcbb9f.

Second retained optimization: fresh device-camera packing in device_camera.py and
triton/heat_kernel_splatting.py. One pixel pass validates full-image triangle IDs,
ordered/in-bounds/covered prepared pixels, finite/bounded/unit-sum barycentrics,
and counts total foreground. It copies visible IDs/barycentrics, ordinals, tile
lanes/counts and face pixel histograms. Exact same fresh immutable snapshot,
unsupported-input Tensor path and Torch async assertion remain. Device dispatch
reuses its just-computed face histogram but retains unique/sort/prefix operations
and allocation readbacks. No cached camera visibility or learned field values.
Before-geometry two-file recovery:
experiments/piggy_camera_pack_before_20261122.tar.gz.

piggy_camera_pack_gate_20261122 PASS: original archived constructor vs new exact
arrays (IDs,pixels,faceIDs,bary,ordinals,tiles,lengths,active,tilePixel,vertices,
queries and every dispatch array), sizes1x1,8x16,64x64,128x192,256x256 and empty
foreground. Mutation guards and double/no-prepared-pixel fallbacks pass. Direct
GPU kernel flag probes reject12invalid classes without deliberately aborting the
CUDA process: negative/large face, negative/large pixel, unordered/duplicate/
missing foreground, background pixel, negative/NaN/Inf/bad-sum barycentrics.
Unused background NaN barycentrics remain accepted, matching original semantics.
Warm component old/new construction1.13482/.61480ms and1.05603/.59700ms reversed.
Shader integer control values also pass (coerced to floats at launch as Torch's
albedo.new_tensor did). Runtime count alignment specialization disabled to avoid
new variants per camera/count. These metadata changes do not alter field values.

Combined trained-checkpoint gate piggy_final_frozen_20261122 PASS. Same808source
model, intentional8source-file delta recorded, no optimizer updates; all7parameter
GPU/eager VJPs pass and fixed held-out MSE matches previous within5e-10. Native
field equality still false. Scalars/layout helpers preserve the Tensor graph;
only supported float32/fixed geometry uses the new first-order shader.

Final combined64x64/1120update pair piggy_final64_1120_20261122:
native/HKS intervals21.353194/12.194338s, workers42.945117/39.796562s, PSNR
30.039533/30.011266, final832/800kernels, allocated2.26007/1.29199GiB. This
passes provisional aggregate0.5dB quality target in this trial. HKS vs controlled
Torch14.459537s interval is1.18576x throughput (~15.7% less time), not5x.
Native vs HKS is1.75107x updates,1.07912x workers. Raw per-view data retained.
Count/quality variation across repeats means no statistical/general claim.

### Larger workload validation and full lit result

128x128/5000source40update pilot piggy_final128_5000_smoke_20261122 PASS,
all strict same-method gates, finite train parameters, PSNR13.483764/13.451617,
allocated5.62928/1.32532GiB. Pilot intervals.95745/.61709s NOT final-speed claims.

Full5000update128pair piggy_optimized128_full5000_20261122 COMPLETE:
native/HKS intervals112.272612/62.553980s (1.79481x), workers133.833749/90.679567s
(1.47590x). PSNR33.331616/33.188671, aggregate deficit0.142944dB, left35.931056/
34.918542 and right31.446404/31.738223. Left1.01251dB deficit is NOT hidden by
aggregate pass. HKS parameters in native field33.138200dB (counterfactual only).
Final4098/3983kernels; original policies, same initial state/controls/camera
metadata hashes, optimizer/source/asset/finite-parameter checks PASS. Count
changes1000/1100/1200/2000 for both but actual count trajectories differ. Peak
allocated6044395008/1515505664bytes; reserved8084520960/2468347904bytes.
Supplied-visibility forward medians12.43545/5.71900ms include shader/source/
support/coefficients and exclude raster/target/backward. Approximate HKS status
and performance_claim_validated=false remain because strict native equivalence
is false; this is a disclosed comparable-quality approximate-field trial.
No source edit while workers run; exact source manifests available.

Post-optimization128x128/5000source diagnostic500update pair completed32profiled
steps420..451. Native mean ms prepare_knn1.5531, prepare_batch9.6335,
forward1.0398, backward6.5527. HKS source.4909, coefficients1.6686, geometry.8704,
support.7008 (nested discovery.1812/intersection.3828), tilequeue.7644,
splat.4694, material_lighting.1278 inside forward6.1059, backward3.1908.
Both raster~1.1,targetlookup~.56,targetlighting~.20,Adam~.85,Geodesic~1.0.
Nested inclusive profile rows are not summed; no performance claim uses its
whole-run timings. Larger-input profile not a size-matched comparison with the
initial64profile. Current coefficient backward already skips zero-adjoint pairs;
inspection confirmed this, so that optimization was not duplicated/reimplemented.

320x320/5000source40update capacity pilot piggy_optimized320_smoke_20261122
PASS both methods. Allocated28.69219/1.42648GiB, reserved33.92773/2.27539GiB.
PSNR13.495164/13.453992, intervals4.27181/.84860s. The roughly5x SHORT/COLD
ratio is NOT a full-training claim. Allocation had22m36remaining and only user
shell/extern steps remained. Launched bounded20minute full5000update320pair
piggy_optimized320_full5000_20261122, same settings and mandatory fresh gates,
no time extension or new allocation. Result pending; no claims before completion.

### Completed full320 result and final handoff (2026-11-22)

The pending full320 pair above completed normally: native/HKS training intervals
516.734526/106.840708 seconds, ratio4.83649x; complete worker bodies
543.509995/233.590551 seconds, ratio2.32676x. Thus neither a5x update claim nor
a5x whole-pipeline claim is supported. Aggregate held-out own-renderer PSNR
34.540306/34.714096dB; left37.891201/38.557147, right32.367976/32.387108.
HKS parameters evaluated using native field score34.700517dB (counterfactual).
Final kernel counts5000/5000, original density/optimizer policies, count-changing
steps1000/1100 for both. Initial identities, controls, camera metadata sequence,
optimizer bindings, finite parameters, source and asset preservation pass.
Allocated native/HKS30809103872/1803658752 bytes; reserved39910899712/2594177024.
Forward-only supplied-visibility medians64.313607/9.195819ms exclude raster,
target generation, loss, backward and optimizer work. Native/HKS startup
imports+CUDA+config6.204996/85.777125seconds; the observed HKS delay remains
included in worker totals, and its precise cause is not isolated.

Both full-training workers exited normally. A subsequent frozen320 diagnostic
completed its numerical assertions, seven-parameter checks, JSON and both PNG
exports, but Slurm terminated the process at its four-minute step limit (exit143).
It is NOT a clean-process pass; execution_status.json records this explicitly.
Frozen held-out MSE exactly matches the saved320 result: left.00013940724602434784,
right.0005771506694145501. Earlier frozen-model probes exited normally.
No allocation extension/new allocation was requested; allocation expired naturally.

These are one-scene/seed GPU measurements with known analytic direct lighting,
NOT ray tracing. HKS remains APPROXIMATE_COMPLETE, strict native-field equivalence
false and performance_claim_validated=false. The128 aggregate deficit.14294dB
and left-view deficit1.01251dB remain disclosed; the320 result is not a universal
quality guarantee. No performance claim uses instrumented run totals.

Exactly eight Python files differ from piggy_verified_sources_20261122.tar.gz:
- hktex_splatting/rendering/reference_mesh.py: shader dispatch and safe Torch fallback.
- hktex_splatting/rendering/splat_autograd.py: fused first-order albedo VJP.
- hktex_splatting/triton/splat_reconstruction.py: fused material lighting kernels.
- hktex_splatting/training/surface_adapter.py: independent Torch shader oracle.
- hktex_splatting/training/surface_profile.py: target-lighting profiling stage.
- splatting_script/experiments/train_surface.py: shared auto/torch shader CLI control.
- hktex_splatting/rendering/device_camera.py: fresh validated GPU camera packing.
- hktex_splatting/triton/heat_kernel_splatting.py: exact camera metadata pack kernel.
Documentation updated: hktex_splatting/README.md,
docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md and this work log.
No native HKTex file, heat equation, selector, optimizer or density policy changed.
No files deleted/moved, new modules/dependencies, commits, pushes or staging.
Pre-existing staged/unstaged changes remain intact.

Final source recovery archive: experiments/piggy_optimized_final_verified_sources_20261122.tar.gz,
SHA2569c37c330353f175cb5aca21f2fe7e30358544d7edfd52a10fb71e415cbb6d7dc.
Extract archives into a separate temporary directory and compare individual files;
do not overwrite the dirty working tree wholesale. All failure artifacts retained.
AST checks cover77 HKS/script modules; original89 HKTex files match session-start
snapshot byte-for-byte. CUDA shader forward/VJP, exact camera/dispatch metadata,
invalid-input flag probes, trained-model gates and full lifecycle comparisons ran.
CPU double shader gradcheck/gradgradcheck, identity and CLI controls passed.
The final320 frozen-process timeout remains the explicit validation limitation.

Remaining measured HKS work includes triangle coefficients (~1.67ms), backward
(~3.19ms), camera geometry (~.87ms) and tile queue (~.76ms) in the128 instrumented
profile, with shared raster and optimizer costs. Inclusive nested rows must not
be added. Direct face-CSR/tile-patch emission could avoid redundant pair-to-tile
expansion/sorting, but is only a next-step hypothesis; it was NOT implemented.

## Additional meshes and exact scheduling optimization (2026-11-22)

User requested further optimization after adding meshes; existing allocation20653414,
A100-PCIE-40GB on g010, was confirmed idle. No new allocation requested. The local
shell has no CUDA driver; GPU checks run through bounded srun steps in that allocation.
Baseline recovery: experiments/mesh_queue_optimization_20261122/before_sources.tar.gz,
profiler_before.tar.gz, assets_before.tar.gz and gitignore_before.tar.gz. Preserve
pre-existing dirty/staged tree and all prior results; no commit/push/staging/deletions.

New raw GLBs inventoried with SHA256 and GLB primitive metadata in
experiments/mesh_queue_optimization_20261122/mesh_inventory.json. Seven distinct files:
teapot19378triangles; basketball560; bowl18944; light_saber155138;
light_saber (1)4552; skull763077; punga chips384256. Counts are GLB primitive totals,
not necessarily post-weld original loader topology. The two saber assets have distinct
hashes and geometry, and were NOT treated as duplicates. All inputs retained.
CPU geometry probe uses the unchanged original load_mesh AST with its CPU dependencies
and an inert logger, because importing full hktex locally attempts CUDA initialization
and fails. This is explicitly not a production package import pass. Teapot post-weld
9689vertices/19378faces, basketball282/560, bowl9474/18944; all finite, watertight,
no zero-area faces detected. Teapot1024x1024 and basketball4096x4096 base-color images
and preserved UVs exist. Bowl has no UV/image and uses a specular/glossiness extension;
its texture-supervision suitability needs review. Large assets not preprocessed blindly.

Added teapot/basketball recorded asset paths and exact checksums to existing prepare_assets
CLI, isolated YAML recipes and camera JSON, and exact-byte prepared copies. Existing
converse/piggy paths and checksums unchanged. Unknown asset names now reject explicitly.
Both new recipes reuse piggy's original-supported random placement and numerical settings,
changing mesh paths only. Camera trajectory fixed before quality evaluation. Spectral
archives and real GPU training remain pending until their bounded checks complete.
Raw inputs/prepared binaries are ignored by Git; recipes/cameras remain visible.

Scheduling changes:
- triton/heat_kernel_splatting.py: block-local sorted-key boundary scan, compact
  block prefix and deterministic pack, replacing separate per-task Tensor booleans,
  concatenations and int64 group scan in the active Triton queue path.
- rendering/heat_kernel_splatting.py: selects new grouping only for Triton; eager
  reference unchanged. Key sort and within-key coefficient entry order unchanged.
- triton/splat_contributions.py: optional compile-time identity-index branch (default
  false for reference callers); active HKS merge uses direct segmented record addresses.
  Eliminates one int64 arange allocation, write and gather per contribution (8bytes
  per record of avoided permutation storage). Top30 tie rule and chunked overflow remain.
- benchmarks/pipeline_profile.py: recognizes both new scheduling kernels.
No heat values, corner interpolation, filters, support, density, optimizers or backward
formulas changed. No caching of learned tensors/visibility; support remains per-forward.

GPU component gate retry PASSED, clean step exit0:65 queue cases, sizes0 through1048576,
all-identical/unique/random/padded/all-sentinel keys; queue, ranges and counts bit-exact.
11 top30 cases0..1500maximum contributions, including zero/ties/sentinels and overflow
above512; IDs and float weights bit-exact vs explicit identity array and independent
CPU ordering. Initial probe's CPU oracle incorrectly compared signed sentinel IDs;
old/new GPU arrays already agreed. Corrected CPU oracle to existing unsigned tie rule,
without changing implementation or loosening tolerances; failed log retained.
Warm synchronized grouping medians, old/new ms, forward/reversed test order:
65536 .26873/.15773 and .26609/.15642;
262144 .26254/.15673 and .26221/.15487;
1048576 .27796/.16316 and .27729/.16137.
These are component measurements only, NOT whole training speedups.
Artifacts: gpu_components.log (initial oracle failure), gpu_components_retry.log,
gpu_components.json. Full pipeline/gradient/training validation pending below.

### Integrated and trained-model checks

piggy_smoke40 completed normally: native_verify,hybrid_verify,native_train,hybrid_train.
Report experiments/mesh_queue_piggy_smoke_20261122/comparison.md. Native/HKS40update
intervals2.541689/1.694886s and initial-stage quality13.482903/13.451957dB are smoke
measurements, not steady-state or full-training claims. HKS all7 parameter VJPs pass;
GPU/eager RGB maximum3.42727e-7; compact IDs exact and weights/RGB within original
strict tolerances. Native-field equality remains false; explicit approximate acceptance
is retained and not represented as an exact-native gate pass.

Trained320 replay against this turn's archived prior queue/identity code PASSED,
clean exit0, on checkpoint piggy_optimized320_full5000_20261122/hybrid_train/final.pt.
Model SHA2560baa1da8379a1f97dd2218faed3ab221f329928d8064ae3d79cbae2e611f7cb5
unchanged. Both full held-out RGB images, selected top30 IDs and selected weights
are BIT EXACT. Seven parameter VJPs pass unchanged3e-5/3e-4 tolerances; largest
absolute delta9.46e-11. Held-out MSE remains left.00013940724602434784 and
right.0005771506694145501. Warm synchronized forward+loss+VJP medians old/new
14.43933/14.32551ms and reverse order14.50796/14.30682ms (~0.8-1.4% benefit).
Scope excludes raster/targets/optimizers/density; this is NOT an end-to-end training
speedup claim. Prior trained snapshot removes initialization-quality confounding.
Old functions loaded from archived text with dedicated linecache filenames to
preserve Triton's source introspection; production source files never swapped.
Initial custom replay import failed GLIBCXX_3.4.29. Retry uses the environment's
libstdc++.so.6 via process-local LD_PRELOAD for BOTH old/new measurements;
no installed package or global environment changed. Both logs retained.

Original89 HKTex files still match previous complete source snapshot byte-for-byte.
AST77 modules and CPU teapot workflow preview pass; old assets --check unchanged,
new --check correctly reports only missing spectral archives at initial setup.
New spectral preprocessing now runs sequentially through unchanged original solver
in a bounded existing-allocation step; results pending below.

### New-mesh preprocessing

Original EigenAlboInterpolation preprocessing completed for both new recipes,
without solver/grid/mode changes: teapot110.744878s (post-weld9689vertices),
basketball11.727659s (282vertices). Both archives are finite and --check passes.
Original input bytes remain unchanged. Exact mesh/config/camera/archive SHA256,
bytes and archive keys are recorded in mesh_precompute.json. No old archive replaced.
Teapot solver progress1m35s, basketball9.9s; elapsed above also includes import/loading,
validation and SHA work. This is one-time preprocessing, not training-time savings.
Short64x64/1000source/40update native/HKS workflows on each now run sequentially,
with unchanged strict GPU/eager gates and explicit native-field approximation mode.

### New mesh training smoke results (completed)

Sequential64x64,1000initial kernels,40updates,seed0,shared initialization/original
policies,GGX known-light supervision,16hops and parallel corners. All four workers
verify/train completed with exit0 for EACH asset. All7 GPU/eager VJPs pass on both.
Explicit native approximation remains disclosed; not exact-native equivalence.

| Mesh | Native interval s | HKS interval s | Native/HKS PSNR dB | Final counts |
|---|---:|---:|---|---|
| Teapot | .587417 | .949620 | 11.015293 / 10.985164 |1000/1000|
| Basketball | .691935 | .540804 |24.227943 /24.227220 |1000/1000|

These SHORT/COLD runs validate integration, not converged quality or steady-state
speed. Teapot HKS is SLOWER here; do NOT claim universally faster across meshes.
At this resolution many small triangles have little pixel reuse, a plausible
explanation requiring a matched profile/scaling study. It is not a measured causal
bottleneck conclusion. Both finish with finite parameters and bound optimizers;
mesh/source/asset preservation checks remain in normal receipts. Teapot allocated
1545147904/598620160bytes; basketball2079046656/114370048bytes(native/HKS).
Teapot final support432048discovered/38654visible pairs, maximum face support55.
Basketball352267discovered/113335visible pairs, maximum face support716: real
GPU end-to-end coverage of top30 chunk overflow beyond512, without capping support.
Recipes/camera paths and exact commands: new_mesh_smokes.json and workflow.json.
Reports experiments/mesh_queue_teapot_smoke_20261122/comparison.md and
experiments/mesh_queue_basketball_smoke_20261122/comparison.md. Research data preserved.

HKS-only piggy128x128,5000initial kernels,1120updates now checks original density
transitions and fresh queue rebuilds after parameter/count changes; not a native
performance comparison. Pending completion below. No5000update job was launched.

### Final density and preservation checks

HKS-only piggy128,5000initial sources,1120updates completed normally with strict GPU/eager implementation gates.
Final count3944; original count transitions[{"step": 1000, "before": 5000, "after": 3347}, {"step": 1100, "before": 3347, "after": 3944}].
Optimizer bindings True, GPU implementation checks True, source/asset unchanged True/True.
Training interval13.475051604000328s; this single-arm run is a density correctness exercise, not a native speedup comparison.
Report experiments/mesh_queue_density_1120_20261122/comparison.md. All77 HKS/script modules parse, source/config snapshot171files unchanged during checks, all89 original HKTex files preserved and all7 new raw GLBs retain inventory hashes.
No new Python modules, dependencies, commits/pushes/staging or deletions. Five existing Python files changed (listed in summary.json), plus .gitignore, this log and HKS README. Added only teapot/basketball YAML recipes and camera JSON as production configuration. Prepared exact mesh copies and spectral archives stay local. All failed probes and prior research artifacts retained.
No new5000update benchmark was launched. Latest full5000update piggy table predates this scheduler change; no5x whole-pipeline, cross-mesh, converged-quality or ray-traced performance claim is established. Existing separate original-Mitsuba render comparison still selects its default Converse camera definitions; do not treat it as validated for these new mesh camera files. New-asset validation here uses train_surface with explicit camera files.
Recovery: extract before_sources.tar.gz, profiler_before.tar.gz, assets_before.tar.gz and gitignore_before.tar.gz into a separate directory, review each delta, and selectively restore only this turn's files. New meshes remain raw at their original paths. Do not reset/stash or overwrite pre-existing staged/unstaged changes. Candidate source/config archive verified_candidate_sources.tar.gz and source_manifest.json reproduce tested code.

## Frozen-code full128 new-mesh study (2026-11-22)

User explicitly stopped optimization and requested ALL seven new GLBs at128,
then selected full5000initial-kernel/5000-update native vs HKS comparisons after
short correctness checks. No renderer, backward, optimizer, selector, training
or benchmarking source is edited. Source freeze166implementation files in
experiments/new_meshes_128_20261122/frozen_sources.json; all meshes/aliases and
SHA256 inputs in assets.json, post-original-loader counts in loaded_geometry.json.
Five remaining asset-specific YAML recipes and fixed camera JSON added; raw GLBs
preserved. Boundary-stop overlay,seed0,known GGX light,16hops,parallel corners,
fresh GPU raster/targets,shared initial model and original density/optimizers
match across both arms. Built-in native_verify and hybrid_verify must complete
before either full training job starts. No gate bypass/tolerance change.
Contribution allocation guard raised to67108864 records for BOTH arms, from
16777216, to accommodate coarse basketball at5000sources; support not capped,
max-buffer remains768MiB. This changes safety capacity only, not heat math.
Only existing allocation20653414(g010,A10040GB) is used; no new allocation or
extension requested. About79minutes initially remained; large skull/chips may
require more preprocessing time/memory than available, without simplification.
Teapot/basketball full workflows launched first, sequentially, with25minute
step bound. Bowl lacks a color texture; user asked whether to derive a documented
constant atlas from original material or mark it unrun. Work on other meshes
continues while that question is pending. No learned color target invented.

User explicitly selected LEAVE BOWL UNRUN: no generated color texture, no bowl preprocessing/training. Remaining six assets requested. No raw/prepared geometry altered.

Teapot128/5000source initial native and HKS verification complete; full training then starts under the existing gated workflow. Frozen source hashes unchanged. Bowl exclusion confirmed; no bowl atlas generated.

Basketball native full5000 completed; HKS passed initial gates but step1 hit MemoryError: Hybrid coefficients/queue/pixel scratch exceed configured budget. Prior attempt preserved in basketball/. Retry BOTH arms in basketball_buffer2048/ with max-buffer2048MiB, same64M record guard and all other settings unchanged. No source/gate/support calculation changed.

Teapot full128 pair completed: native/HKS intervals108.730131/78.193452s,
training-worker129.904666/111.341172s; own PSNR28.744441/29.533725dB;
HKS parameters through native field29.060779dB; final5000/5000kernels.
Basketball retry full128 pair completed: native/HKS164.700/122.344s;
training-worker190.919/150.671s; own PSNR29.277/29.183dB, native-field HKS29.167dB;
final5000/5000. Full precision remains in result JSON. Original failed768MiB
attempt retained. Current table uses documented2GiB retry for both methods.
All strict implementation gates remain intact, exact-native equivalence still false.

Remaining four meshes now run sequential original precompute then full gated
comparison in existing49minute Slurm step, with47minute application deadline and
safe process-group termination only for its own subprocesses if time expires.
Source freeze rechecked before each asset. Same study settings,2GiB scratch guard
and64M contribution allocation guard shared by both arms. No geometry/eigen/grid/
iteration simplification. Deadline is existing-allocation capacity, not a completed
result. Status JSON/logs retain each stage and incomplete work explicitly.
No new allocation, optimizer change, source edit, commit or push.

User explicitly instructed USE ONLY CURRENT ALLOCATION. Do not request/extend allocation. Finish what fits; retain incomplete statuses for remaining assets with stopping stage. Smaller light-saber original preprocessing completed51.032772s, including original solver regularization retries (unchanged built-in eps fallback). Native/HKS initial checks pass; full run ongoing.

Smaller light-saber native worker PID568877: elapsed5m41s,CPU5m30s,GPU100%,trajectory0bytes (flush every128). First128 updates not recorded; possible geodesic trace issue, cause NOT verified. Original max_iterations defaults2**63-1; no cap/optimizer edit applied. Stop only own native worker to preserve remaining allocation, retain diagnostic/failed workflow; HKS is not a completed competitor. runtime_stop JSON records observation, not a speed result.

Smaller light-saber native worker terminated bySIGTERM only after explicit PID/cmdline verification. Parent recorded native_train failed and skipped HKS; workflow remains incomplete, not a completed speed comparison. Driver continued to larger light-saber original preprocessing. All166 frozen source hashes still checked before each asset. Teapot/basketball/smaller-saber completed archives pass independent CPU-mmap finite checks; no spectral data altered.

Larger light-saber original precompute exitedSIGSEGV(returncode-11) after131.544171s, before an archive or training. Cause not verified; no native/HKS speed or quality result. Crash log retained, any private temporary directory left in place. Driver continues chips(originalprecompute) with settings unchanged. No solver repair, mesh repair or caps applied.

During CPU-heavy chips preprocessing (no timed training), attempted disjoint-core
skull preprocessing. Initial step4CPU affinity request failedEINVAL before launch;
no skull solver ran in that attempt. Retry step inside same allocation requests
access to allocated8cores, checks physical core IDs, pins skull to[1,3,5,7]
while chips is pinned[0,2,4,6]; actual active threads4+4within allocation.
Skull original solver starts with Python faulthandler only (diagnostic traces,
no math changes). Supervisor terminates this subprocess if chips workflow log
appears, before any timed native/HKS training can begin;27/25minute bounds remain
below allocation end. Precompute wall times under overlap are not standalone
performance claims. First chips angle287.03s; remaining solver/training may not
fit. Only current allocation allowed by explicit user reply.

Local standard-library archive watcher preserves a hardlink only after torch-save ZIP footer is present in a private chips/skull preprocessing directory. Recovery-only paths in cpu_archive_backups/ are not automatically published or claimed finite/GPU-validated. No storage copy, no source/target modification, no learned-state caching. Captures retained for CPU finite validation if helper setup later fails/timeouts.

### Explicit user stop

User requested STOP HERE and plans a larger allocation next time. Only own Slurm
steps20653414.17(chips/main study supervisor) and20653414.23(skull parallel prep)
were cancelled. Allocation20653414 and interactive shell20653414.0 are preserved.
Completed full128/5000update pairs:teapot and basketball(shared2GiB retry).
Smaller light-saber:initial gates pass, original native training stopped for no
flushed128-update progress, HKS skipped. Larger light-saber:original precompute
SIGSEGV(-11), no archive/training. Chips/skull:preprocessing stopped explicitly
by user; no full training result. Bowl excluded explicitly by user(no texture).
No implementation modifications or new optimization during study. Prepared
copies/configs/cameras, completed archives, failed attempts and all logs retained.
Exact commands in remaining_plan.json/workflow.json are restart instructions;
checkpoints are archival, not a supported training resume interface. Unfinished
spectral computation requires regeneration; no claim of partial-grid resumability.
Stop record/summary at experiments/new_meshes_128_20261122/.

Stop verified: squeue shows only interactive shell.0 and extern. Both own steps exited137 after cancellation, local archive watcher exited0. All166 implementation hashes unchanged. No complete chips/skull archive or recovery CPU archive was produced; preprocessing cannot resume from its partial in-memory grid. Restart with original settings in a new output directory; exact commands preserved. No further experiments run after stop request.


### 2026-11-23 — exact per-source GPU support traversal

Implemented per-source GPU breadth-first graph traversal in the two existing
modules `hktex_splatting/triton/resident_gpu.py` and
`hktex_splatting/knn_heat/triangle_kernel_support.py`. No module was added or
deleted. Current hop count, source-major diagnostic support, face/source CSR
ordering, visibility intersection, spectral equations, top-30 reconstruction,
backward kernels, optimizers and density policies are preserved.

One GPU program owns each source's hash table and queue. Mesh adjacency stays
fixed, but traversal uses current source faces every forward, including invisible
intermediate triangles. The 1,024-entry queue is bounded scratch, not a source
count limit. Scratch-budget failure or row overflow invokes the complete old
tensor traversal. Existing fixed-hop lookup remains preferred when available.
Scratch queue/hash/flags total approximately 58.6 MiB at 5,000 sources, plus
existing support/dispatch allocations. Existing budgets and explicit errors remain.

## Measurements

Scene: trained punga chips, 384,256 triangles, 5,000 kernels, 128², A100 40 GiB,
existing 16-hop support. Same in-memory parameters and camera sequence for A/B.

| Measurement | Old HKS | New HKS | Scope |
|---|---:|---:|---|
| Fresh raster + targets + forward + loss + VJP, median ms | 23.802 | 11.029 | Unprofiled, 20 observations |
| Reversed-order trial, median ms | 23.240 | 11.262 | Unprofiled, 20 observations |
| Support discovery, ms | 16.363 | 0.524 | Instrumented, inclusive stage |
| Visibility intersection, ms | 0.809 | 0.445 | Instrumented, nested in support |
| Entire support stage, ms | 17.317 | 1.664 | Inclusive; do not add child timings |

The unprofiled measured pipeline gain is 2.06–2.16× against previous HKS.
Raw observations, including outliers, remain in `pipeline_ab.json`.
Instrumented source-state preparation was about 0.90 ms before this change:
the historical 14.17 ms source-state measurement is not this bottleneck.
The optimized profile has a 29.38 ms geometry outlier; it is not a reliable
whole-step estimate. Profile stages overlap and must not be summed.

Eight warmed 64-update trials also executed the inherited original training loop
with fresh original optimizers/schedulers, camera supervision and source-face
movement. New times: 0.854–0.920 s; old: 1.582–3.485 s. The wide old variation
precludes a precise optimizer-inclusive speedup claim. These trials start at the
trained checkpoint and reset iteration numbering; density intervals are inactive,
so they do not validate a complete densification cycle or resumed optimizer state.
All camera hashes match and losses are finite. Final learned parameters differ
beyond frozen VJP tolerances both between old/new and between repeats of the same
implementation (`short_training_drift.json`). Nondeterminism is a possible cause,
not an established explanation. Long-trajectory equivalence remains unproven.

No new 5,000-update native-versus-HKS comparison, held-out PSNR or peak-memory
comparison was run. Do not infer a 5× full-training result from these numbers.
The existing completed chips/skull native comparisons remain historical results.

## Correctness actually executed on GPU

- 49 graph cases against independent CPU BFS, including hops 0–32, disconnected
  cases, duplicate seeds, self loops, hash collisions and intentional overflow.
- Four real camera views: ordered CSR, corner heat, Gram matrices, top IDs,
  weights and final RGB bit exact against pre-edit HKS; all seven parameter VJPs
  pass unchanged tolerances (atol 3e-5, rtol 3e-4).
- Eight batches of 5,000 random source faces on the actual chips graph: ordered
  visible CSR and discovered-pair totals exactly match the old traversal.
- End-to-end overflow fallback: 48-hop grid support, 4,854 discovered pairs;
  fallback flag `row_overflow`, all CSR/source-major arrays exactly match the
  reference traversal. No truncated support published.

Raw records: `graph_units.json`, `trained_gates.json`, `moving_source_gate.json`,
`fallback_gate.json`, `baseline_profile.json`, `optimized_profile.json`.

## Quality finding and remaining work

On 256 identical surface samples using frozen HKS-trained chips parameters,
16-hop support contains 19.79% of native positive top-30 kernel IDs and 25.02%
of their total native weight. Mean candidates: 7.10; 98.83% of these pixels have
fewer than 30 candidates. This is evidence of support mismatch on this sampled
view, not a quantified explanation of the entire PSNR gap. The new traversal
preserves this same approximation. It does not solve native-field quality or
training-trajectory differences. Coverage/quality across views and a fresh full
paired training run remain necessary before making a native speed/quality claim.
Skull full rendering and active density events were not validated in this sweep.

## Harness costs and exclusions

An initial production-worker setup spent about eight minutes hashing the 9 GB
spectral archive for provenance; only our diagnostic Python was interrupted.
Production hashing/validation was not edited or loosened. The already constructed
trainer was reused for the independent in-memory checks above. Therefore the
`setup/` directory is NOT a completed production preflight receipt.

Early `old_updates_*`/`new_updates_*` trials restored fixed-mesh buffers, bumped
their tensor versions and forced expensive adjacency reconstruction inside timing.
Those 24–42 s intervals are invalid for steady updates and excluded; one old trial
was interrupted in SciPy adjacency preprocessing. The corrected restore skips
fixed geometry; all failed trials/tracebacks are retained in `gpu.log`.
First-call GeoOpt/JIT and remaining timing variation are retained, not erased.

## Source safety and recovery

Before edits, 176 source/config files were hashed and archived in
`before_manifest.json` and `before_sources.tar.gz`. All 89 original `hktex/`
files are unchanged versus that snapshot. Only the two implementation modules
above changed in that manifest. All 77 HKS/script Python modules parse under
Python 3.11. The system Python 3.6 parser rejects existing modern syntax;
that first check was repeated successfully in the repository's Python 3.11 env.
No standalone testing infrastructure, dependency, CLI or asset was added/removed.
Documentation changed: HKS README and `docs/HKS_WORK_LOG.md`, plus this result.
Per-file pre/post diffs are saved alongside these measurements.

The worktree already contained staged/unstaged/untracked research changes.
Those were retained; global Git status is not this task's diff. No staging,
reset, stash, commit or push. Recover an individual pre-edit source with
`tar -xOf experiments/hks_targeted_20261123/before_sources.tar.gz PATH` to inspect
its exact previous content before choosing a restoration; avoid Git reset.

Only diagnostic Slurm step 20664166.2 was stopped after these checks. Allocation
20664166 and the user's interactive shell were preserved; no new allocation
was requested. No long training run was launched.


### 2026-11-23 — spectral-support accuracy and graph-overflow optimization

User authorized further optimization for speed AND quality. Existing allocation
20664166 only; no new allocation. Original hktex source remains unchanged (89
files verified against a fresh 176-file source snapshot). Recovery archive and
per-file hashes: experiments/hks_accuracy_20261123/before_sources.tar.gz and
before_manifest.json. Existing staged/unstaged research work was preserved.

Verified bottleneck: latest skull result used tensor_frontier with
frontier_fallback=row_overflow. Graph discovery now retries 1024/2048/4096-face
rows within min(max_dispatch_bytes/4,256MiB) scratch, then retains complete tensor
fallback. The row size is NOT a source-count cap. Grid GPU checks pass: 4854
pairs via 4096-row GPU retry and 6809 via full tensor fallback, exact ordered CSR
and source-major support in both. No support is silently dropped.

Opt-in --support-domain spectral removes edge-hop truncation. It computes fresh
canonical source/pixel spectral embeddings, bounded float64 matrix KNN (native
outer-k policy), unions kernel/triangle incidences, evaluates shared three-corner
heat/Gram coefficients and uses existing kernel/tile emission, top-30, RGB and
custom backward. Each emitted contribution is masked by that pixel's OWN KNN
membership, preventing union leakage across pixels. Eager selection uses direct
float64 cdist; eager shading/adjoints remain independent. Only discrete selection
is detached; floating coefficients retain existing autograd connections. No live
heat, lighting, support or source-state caching across updates. CLI/provenance,
reference adapter and reports record the new option; graph stays default.

A first FAISS-based trial failed strict compact consistency despite matching
RGB/VJPs: repeated equal query/source embeddings gave different cutoff candidates
in 5/256 rows. It was rejected without relaxing checks; its failed gates and
snapshots are retained with faiss_failed_ prefixes. The replacement uses bounded
float64 matrix distances and canonical native interpolation for selection.

Actual GPU checks after replacement:
- Strict trained-skull 256-sample gate PASS, including all seven parameter VJPs,
  compact/full consistency, native-field RGB/VJPs and native replay operator.
  Missing positive native top-30 weight fraction=0 on that sample.
- Synthetic shared-triangle union (7 candidates/pixel, union up to16): RGB and
  six heat/Gram/color/base/filter adjoints pass unchanged 3e-5/3e-4 tolerances;
  zero-response RGB bit exact. No zero-response gradient claim.
- Source counts1,17,8193 and empty view pass independent direct cdist comparison.
- 64 inherited-loop skull updates pass with finite parameters and original
  optimizer bindings. This begins at a trained checkpoint with fresh optimizer
  state and includes first-call setup:11.5809s, NOT a steady timing claim or
  optimizer resume/densification-cycle validation.
-77 Python modules parse; CLI preview propagates spectral support into all four
  native/HKS verification/training commands without starting a GPU.

Frozen trained-skull parameters, full128 camera frames, same input views:
unprofiled synchronized forward+MSE+VJP medians native13.9995/spectral9.0396ms;
reversed order spectral9.0163/native13.5717ms,24 observations per arm/trial.
This EXCLUDES rasterization/target generation, optimizer, density, setup and
held-out evaluation. It is not a full-training speed claim. Four training views
have native/spectral target-PSNR differences under0.001dB at the SAME HKS-trained
weights. Individual pixel differences reach0.04189 RGB, so no dense equivalence
claim: float64 KNN differs from native float32 FAISS near cutoffs. No claim of
better converged quality follows from frozen weights or a bounded initial gate.

Harness setup was slow in shared-filesystem framework imports; first process
was interrupted during torch import and restarted. Diagnostic step .12 was
stopped only after saving checks; user allocation/shell preserved. Two synthetic
harness attempts failed before verification: omitted device patch_tiles argument,
and an intentionally too-small dispatch budget. Corrected tests passed; failure
logs retained. Diagnostic commands (including failures) in diagnostic_commands.json.

After the strict gates and short optimizer smoke, launched one fresh production
native-versus-spectral-HKS skull128/5000-initial/5000-update pair through existing
train_surface workflow, seed0, fresh nvdiffrast visibility, GPU targets, original
density/optimizers, shared initialization, boundary-stop overlay, GGXroughness
0.112565. Existing approximation opt-in retains all strict GPU/eager/replay gates.
Step .15 bounded40minutes, no concurrent GPU diagnostics. Raw workflow command,
source/control receipts and results are under skull_full/ and skull_full.log.
Full-run results must be read there; completion is not assumed in this entry.


Completed fresh paired skull result: native114.578191s versus HKS89.547803s training (1.27952x); complete processes248.091910/225.932927s (1.098x). Held-out deployedPSNR26.281164/26.263656dB, deltaHKS-native=-0.017508dB; near parity, not better quality. HKS common-native-fieldPSNR26.263626. Initial/control/camera hashes match. Both passed strict initial implementation/native checks, fixed assets and optimizer bindings; density parameter replacements at1000/2000, final5000sources. No profiling/concurrentGPUwork. Rawreport:experiments/hks_accuracy_20261123/skull_full/comparison.md. Full audit and limitations:experiments/hks_accuracy_20261123/comparison.md. All169 validated source hashes archived exactly; final2Python comment/docstring edits preserve executableAST. Native89files unchanged. No commit/push or data deletion.


### 2026-11-23 — HKS-only results cleanup

User clarified ONLY HKS source may be cleaned and requested per-file/function review. Original hktex, author scripts/configs and root README/metadata preserved (89 native,154 protected files; Git index unchanged). Reviewed77 HKS/workflow Python modules; removed8 retired custom-policy/occupancy/cache modules, simplified8 others,29 class/function names removed, source+31/-593 net-562. Unique source preimages verified in archive/hks_results_cleanup_20261123/preimages.tar.gz BEFORE removal. Written deletion plan and dependency evidence precede edits in docs/HKS_RESULTS_CLEANUP.md. Source/equations/backward files byte-identical; all47 active GPU kernel definitions unchanged; graph/spectral support class unchanged; exact dead-branch equivalence for retained reference selection and full-spectrum oracle. Actual CPU/interpreter17 arrays bit-exact,132 trainer callback scenarios,4 attachment cases,6 command/workflow previews,6 retired-interface guards.69 modules parse/no broken local imports. No GPU or training/benchmark job launched for cleanup; no new performance claim. Retired whole-object APIs need archived source. Existing results/recovery/data retained, no code moved, no new Python module/test runner, no dependency uninstall, no original source edits, no staging/commit/push/reset/stash. All-file review and verification data linked in cleanup report.


### 2026-11-23 — shorter initialization/comparison commands

Added opt-in `train_surface --preset compare-128 --asset NAME` in the existing
entry point, without new Python modules or wrapper scripts. Named recipe expands
the latest lit128 spectral-support comparison settings (5000sources/updates,
seed0, fresh GPU raster/targets, shared initialization, original density/optimizers,
boundary-stop overlay, parallel corners, GGXroughness0.112565,2GiB/64M budgets,
1024verification samples, bounded native approximation, no training plots).
`--verify-only --execute` initializes and verifies both arms; the full workflow
already initializes before training, so a separate INITIAL shell variable is not
required. No execution occurs without --execute. Preset/asset labels are recorded
in workflow JSON; actual expanded values reach worker commands/manifests. Explicit
value flags override defaults. Existing no-preset CLI defaults remain unchanged.

CPU-only checks: shortcut expands exactly to the archived long skull command,
explicit long command still matches,4 legacy previews unchanged,9 configured
asset previews resolve configs/cameras without GPU imports,2 verification jobs
share the native initial-state path, value overrides work independent of argument
order,4 invalid shortcut/policy requests reject. Torch/Triton were not imported.
Worker, optimizer-binding, initial-control and lighting function ASTs unchanged;
all89 native Python files unchanged. No assets/preprocessing/GPU runs launched,
no index/staging/commit/push changes. Source/docs preimages and expanded recipe/
verification JSON in archive/hks_command_shortcuts_20261123/. HKS README now leads
with the short initialization command and documents the full recipe and limits.
