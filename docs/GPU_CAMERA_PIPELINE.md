# Fresh GPU camera training

The [distance optimization and cleanup report](HKS_OPTIMIZATION_REPORT.md)
documents the subsequent compact exact reduction, removal of unused pixel
embedding work, module splits and CPU/offline checks. Those changes have not
been timed on a GPU.

This opt-in pipeline rasterizes mesh triangles **on every frame**, including when
a view repeats. It uses the existing triangle-supported HKS renderer; it does not
cache learned heat values, colors, support IDs, or visibility. CPU rasterization
remains the default/reference. No speedup has been measured for this new path.

## Pipeline and implementation

1. `rendering/cuda_mesh_raster.py`: project the fixed mesh using the recorded
   camera and rasterize with nvdiffrast's CUDA context. Convert bottom-up raster
   rows to top-down rows. Output device triangle IDs and perspective barycentrics.
2. `training/camera_supervision.py` and `training/target_texture.py`: keep these
   tensors on the GPU, interpolate fixed UVs in float64, and look up unlit input
   texture RGB. Native HKTex receives these same visible surface samples.
3. `rendering/device_camera.py`: build visible triangle bounds, face/tile patches,
   and padded screen-tile pixel lists using device tensors. The existing fused
   selector, safe fallbacks, three-corner coefficients, barycentric heat evaluation,
   tile emission, top-30 RGB reconstruction, backward operators, density callbacks,
   and optimizers remain in use. No original `hktex/` file is changed.
4. `splatting_script/benchmarks/camera_scaling.py`: launch paired full adaptive
   camera training at matched resolutions. Its Markdown report includes training
   time, complete training-worker time, held-out PSNR, kernel count and memory.
   Per-resolution receipts preserve initialization, source hashes and gate results.

The raster-to-HKS geometry path does not download visibility arrays. Bounded
setup correctness checks and existing initialization/verification receipts still
transfer diagnostic data. Existing support allocation reads four device scalars
(32 bytes), and dynamic-shape Torch operations can synchronize. GPU resident does
not mean synchronization-free. Dataset raster/target counters measure host
work/enqueue; synchronized worker/training times include GPU completion.
For this GPU raster mode only, per-update trajectory fingerprints record camera
metadata instead of downloading/hash-checking all sample tensors. Matching
trajectory hashes therefore establish matching view sequences, not pixel tensor
identity at every update. Initial/evaluation assets and bounded RGB/VJP gates
retain their strict checks; original CPU/surface fingerprint behavior is unchanged.

## Optional installation

Use the repository's existing CUDA/PyTorch environment. nvdiffrast requires a
matching CUDA toolkit (`nvcc`) and C++ compiler in addition to the NVIDIA driver.
See the [official installation instructions](https://nvlabs.github.io/nvdiffrast/#installation).

```bash
HK_PY=/path/to/your/hktex/environment/bin/python
"$HK_PY" -m pip install setuptools wheel ninja
"$HK_PY" -m pip install -e '.[native,gpu-raster]' --no-build-isolation
```

The optional extra fetches nvdiffrast from its official Git repository. For
reproduction, record/pin the tested commit rather than assuming a later Git HEAD
is equivalent. Worker controls record its distribution version and pip VCS
`direct_url.json` when available. This environment has not built or tested that
optional dependency; an allocated CUDA GPU is required for validation. Missing
nvdiffrast produces an explicit error; it does not silently use the CPU raster.

## Verify before training

Prepare mesh/spectral assets using the existing instructions first. On your GPU:

```bash
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --supervision camera --resolution 128 \
  --raster-backend nvdiffrast --target-sampling gpu --visibility fresh \
  --kernels 5000 --iterations 4 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct --selector-merge row-radix \
  --no-training-plots --verify-samples 1024 --verify-only --execute \
  --output experiments/gpu_camera_verify
```

The rasterizer runs a bounded synthetic GPU check for perspective barycentrics,
row orientation, interior visibility/occlusion, and near clipping at setup. Target
lookup and the first bounded device-interpolation batch are checked against CPU
RGB exactly. HKS rendering/VJP/support gates stay strict. The previously observed
camera RGB/top-30 gate discrepancy remains unresolved; do not assume this change
fixes it. `--allow-render-variation` is still a separate diagnostic-only option
and cannot waive target or raster checks or establish validated performance.

GPU raster edge/depth precision differs from the float64 CPU raster. The GPU near
plane is explicitly `--raster-near 0.01` in mesh units; adjust it for your scene.
The CPU reference retains its original scale-dependent near plane. Thus these
are distinct raster policies, not bit-exact interchangeable outputs. Both arms
of a comparison always use the same policy. Use resolutions divisible by 8 and
`data.num_workers=0`; CUDA camera datasets must not be forked.
Adding `--profile-start 10 --profile-steps 4` records synchronized rasterization
and GPU target-lookup stages alongside the existing forward/backward stages.
These diagnostic runs include synchronization overhead and must remain separate
from unprofiled speed measurements.

## Conservative support pruning experiment

`--support-pruning prefix` is optional and defaults off. It applies only to the
audited CUDA float32 64-D, 4097–8192-source row-radix path. Other shapes/counts
retain the existing selector and fallbacks. Support stats record requested/used
pruning; selecting the flag is not proof that a workload used it.

For each triangle, full distances to the first K sources provide an upper bound
on the true Kth distance. Apply the unchanged radius/guard expression to that
bound. A 16-coordinate distance, computed with the same float64 reduction tree
and RN square root as the full distance, is a conservative lower bound. Sources
whose lower bound exceeds the upper support limit cannot affect Kth selection
or support. Their distance slots become infinity; admitted pairs retain the
exact full calculation. Nonfinite seed distances disable pruning for that row.
Source IDs, support ordering, corner math and backward code remain unchanged.
Whole rejected face/source blocks skip the full norm calculation; partially
admitted blocks retain full arithmetic and mask unnecessary source loads.

There is seed, allocation, and prefix overhead. Loose bounds may reject nothing;
this can be slower. Benchmark on the actual GPU before adopting it:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.spectral_distances \
  --pruning --execute --output experiments/gpu_pruning_distances.md
```

This compares retained distances and support IDs against unpruned calculations
and includes seed/bound overhead in pruning timings. It is a microbenchmark,
not whole-training evidence (synthetic rank 10 / cap 16, rather than the usual
training rank 50 / cap 128). Add `--support-pruning prefix` to the verification
command above before measuring full training with that flag.

## Resolution scaling

Preview first by omitting `--execute`. Use a fresh output directory:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.camera_scaling \
  --resolutions 64 128 256 --kernels 5000 --iterations 601 --seed 0 \
  --raster-backend nvdiffrast --target-sampling gpu \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct --selector-merge row-radix \
  --no-training-plots --verify-samples 1024 --execute \
  --output experiments/gpu_camera_scaling
```

601 updates exercise early density changes, not the complete default training
schedule. Set `--iterations 5000` for that schedule once smoke checks pass.
Compare native/HKS **within** each resolution and inspect held-out PSNR and final
kernel counts. Increasing resolution alone does not establish equal quality or
convergence. Complete worker time includes setup, gates, rasterization, targets,
learning, final evaluation and saving; training time excludes initial gates and
the first batch. Do separate unprofiled timing and bounded profiling runs.

This remains unlit texture fitting with view-dependent visibility. It does not
add view-dependent appearance or physically based relighting. No 5× end-to-end
claim is supported until paired GPU runs pass gates and measure matched quality.

## Local verification and recovery (2026-11-21)

The first real GPU attempt (`gpu_full_20261121_021837`) stopped during the
synthetic raster barycentric check, before native training. The camera transform
used GEMM under the trainer's TF32 policy. It now uses explicit float32 products
and sums, independently of matmul precision; the 2e-5 check is unchanged and
failures now include the measured error. This addresses a precision risk; a GPU
verification rerun is required to confirm the cause and fix.

Passed without GPU execution:

- All HKS/script Python syntax, package TOML, paired surface/camera CLI previews,
  GPU flag guards, 64/128/256 scaling previews and report receipt-field checks.
- Real HKS renderer in the Triton CPU interpreter: device-tensor geometry versus
  the existing camera geometry at 1×1, 9×17 and 16×16; matching RGB, selected IDs,
  weights and six parameter VJPs with synthetic coefficients. Empty views,
  mutation checks, gate subsets, adapter wiring and saved CPU metadata were checked.
- Existing real two-triangle CPU camera raster/target/DataLoader checks; exact
  CPU target checks across RGB/RGBA/palette/grayscale and multiple materials;
  device-interpolation code exercised on CPU against reference RGB over 16,384
  further samples. This is not a CUDA interpolation validation result.
- Conservative pruning kept exact retained distances and final support IDs for
  51, 257 and 5000 sources, including fully rejected blocks, unpruned large-radius
  rows and nonfinite seed guards. The device support wrapper matched existing
  selection at 12, 5000 and 8209 sources, including legacy/tiled fallback cases.
- Offline SM80 compilation of unpruned/pruned blocked distances and seed bounds.
  Compilation establishes neither runtime GPU correctness nor performance.
- Optional camera profiling wrappers restored original methods after exit.
  Original `hktex/` Python bytes and the pre-existing Git index were unchanged.

GPU raster/context checks, real spectral-data CUDA forward/backward, target
interpolation on CUDA, post-density GPU behavior, memory feasibility and speed
remain pending. nvdiffrast and `nvcc` are absent here; `nvidia-smi` cannot contact
a driver. A CPU projection check initially hit a system `libstdc++` mismatch and
passed using the existing Conda environment's library directory. No dependency
installation or GPU training was launched.

New source modules: `rendering/cuda_mesh_raster.py`, `rendering/device_camera.py`,
and `splatting_script/benchmarks/camera_scaling.py`. Integration changes are in
`training/{camera_supervision,target_texture,surface_adapter,uv_training,surface_profile}.py`,
`rendering/heat_kernel_splatting.py`, `knn_heat/triangle_kernel_support.py`,
`triton/resident_gpu.py`, `splatting_script/{experiments/train_surface,benchmarks/spectral_distances}.py`,
and the optional extra in `pyproject.toml`. The main/HKS/script READMEs and camera
training guide link this document. No files, research results or assets were
deleted. No commit, push, index rewrite or Git history change was performed.

Task-start Python/package copies and temporary validation scripts are in
`/tmp/hks-gpu-pipeline-fsqw_aev` on this machine; `/tmp` is temporary recovery,
not a durable research archive. Compare against those copies to isolate these
edits from the repository's extensive pre-existing staged/unstaged cleanup work.
Whole-tree `git diff --check` still reports pre-existing whitespace in
`hktex/modules/tracer.py` at lines 285/290; that original file was left untouched.
