# Heat-Kernel Splatting with material lighting

Heat-Kernel Splatting (HKS) evaluates learned spectral heat kernels on a fixed
mesh using triangle support and GPU screen tiles. Material RGB is reconstructed
first; a separate analytic shader supplies diffuse lighting and optional
view-dependent GGX highlights. Original pointwise HKTex remains in `../hktex/`;
its paper, authors, original experiments and citation are in the
[main README](../README.md).

## What is implemented

| Component | Current implementation |
| --- | --- |
| Geometry | Fixed triangular mesh and precomputed Laplace–Beltrami eigensystem. |
| Learned field | Trainable source positions on the mesh, anisotropy, orientation, colors and filter parameters. |
| Recommended comparison | `train_surface --preset compare-128`: camera-supervised native HKTex versus spectral-support HKS. |
| Spectral support | Fresh global spectral nearest neighbors per visible pixel, inverted into kernel–triangle support. |
| Graph support | Retained approximate kernel-seeded adjacency traversal, with bounded GPU retries and a complete tensor traversal fallback. |
| Heat evaluation | Three-corner spectral heat and intrinsic-distance coefficients, reused across pixels on a triangle. |
| Rendering | Barycentric interpolation, GPU tile scheduling, Triton contribution emission and strongest-30 weighted RGB reconstruction. |
| Training | Custom renderer backward; inherited original trainer, Adam/GeodesicOpt, schedules, density controllers and optimizer-binding checks. |
| Lighting | Shared analytic distant light: `none`, Lambert `diffuse`, or `ggx`; fixed roughness and Fresnel F0 controls. |
| Validation | Independent eager checks, RGB/parameter-gradient gates, initialization/control receipts and parameter-health checks. |

`hybrid` is the retained CLI name for HKS; `native` selects original HKTex.
The older renderer harness also accepts `2dgs` and `original` aliases. HKS uses
heat kernels and HKTex weighted color reconstruction, rather than Gaussian
primitives or depth-sorted alpha compositing.

The current recommended spectral-support mode is **pixel-driven spectral KNN
inverted into triangle support**. It is not connectivity-only kernel-centric
splatting. Graph mode retains kernel-to-triangle discovery for research comparisons.
Without a preset, the existing CLI defaults remain surface supervision, graph
support and unlit RGB; use the preset explicitly for the workflow below.

## How rendering and training work

1. Load the textured mesh, matching spectral archive and recorded cameras.
   Geometry and the eigensystem stay fixed throughout training.
2. Rasterize triangles freshly for each training camera frame using nvdiffrast.
   Both methods receive the same visible triangle IDs, barycentrics and targets.
3. Prepare current learned source positions, anisotropy, orientation and spectral
   state. Recompute live values each forward; do not cache learned coefficients.
4. In spectral mode, find each pixel's nearest kernels in spectral space, then
   union those candidates into kernel–triangle pairs. Each pixel retains its own
   candidate membership so triangle-wide unions cannot introduce extra neighbors.
5. Evaluate heat at each supported triangle's three corners and prepare six
   intrinsic-distance coefficients for that kernel–triangle pair.
6. Build kernel/tile work. Triton interpolates heat with pixel barycentrics,
   evaluates distance weighting and filtering, and emits scalar contributions.
7. Keep the strongest 30 filtered contributions and reconstruct material RGB
   using the original weighted-color formula, learned base color and clamp.
8. Apply the shared material shader, compute loss and backpropagate through
   reconstruction and spectral coefficients. Original optimizers move kernels
   along the fixed mesh; original density callbacks can prune/split kernels.
   Support is refreshed as source parameters and counts change.

The spectral candidate selector uses bounded float64 matrix distances and an
independent direct-`cdist` oracle. It removes graph-hop truncation, but close
cutoff choices can differ from native float32 FAISS. Three-corner interpolation
also remains an approximation. Discrete visibility, neighbor membership and
strongest-30 membership are fixed during each backward pass.

## Lighting and material behavior

The learned heat-kernel RGB field acts as material albedo. `shade_material` in
[reference_mesh.py](rendering/reference_mesh.py) applies a separate world-space
white distant light, diffuse fill and optional isotropic GGX reflection with
Smith masking and Schlick Fresnel. Camera direction affects the specular term.
The same shader and settings are applied to both methods and texture targets.
Lit loss/evaluation use linear RGB radiance; `--lighting none` preserves unlit RGB.

The light direction, intensity, fill, roughness and F0 are explicit **fixed
controls**, not learned illumination or material estimates. Different light and
camera inputs can be shaded without baking lighting into the kernel definition.
This is not a complete learned radiance field: there are no traced shadows,
reflected scene images, indirect illumination or ray-traced lighting. No claim
of ray-traced end-to-end acceleration follows from these measurements.

`--lighting-backend auto` uses the fused Triton shader on supported CUDA float32
inputs with fixed geometry, including first-order albedo gradients. `torch`
retains the reference graph for other layouts or geometry/higher-order derivatives.
Lighting is separate from the heat-kernel equations and top-30 reconstruction.

## Setup

### 1. Install the environment

Use a Linux source checkout, Python 3.11 and a compatible NVIDIA CUDA GPU.
A100 40 GiB is the recorded device for the latest skull comparison; memory needs
depend on the scene and spectral archive. Slurm and Grace are not requirements.
On a cluster, obtain GPU resources using its scheduler before execution.

Reuse a compatible original-HKTex environment or install the recorded stack:

```bash
mamba create -n hktex-hks python=3.11
mamba activate hktex-hks
mamba install -c pytorch -c nvidia -c rapidsai -c conda-forge \
  libnvjitlink=12.9.86 faiss-gpu-cuvs=1.13.1 "numpy<2.3"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install ninja
python -m pip install -e '.[native,gpu-raster]' --no-build-isolation
python -m pip check
python -c "import faiss; assert hasattr(faiss, 'StandardGpuResources'); import nvdiffrast.torch"
```

Run installation commands from the repository root after cloning. nvdiffrast
builds require a compatible **full CUDA toolkit**, `nvcc`, C++ compiler and
`CUDA_HOME`; PyTorch wheels alone do not supply the full build toolkit. Put the
toolkit's `bin` directory on `PATH` and check `nvcc --version` before installing.
GPU FAISS is a separate native prerequisite; `faiss-cpu` is not a substitute.
Triton must match the declared PyTorch stack. Numba remains required for retained
CPU visibility/reference utilities. Optional neural-texture baselines are not
needed for this comparison. See [pyproject.toml](../pyproject.toml) for declarations.

Other CUDA stacks/devices must pass the correctness gates before benchmarking.
A clean installation, regenerated eigensystem and all non-A100 devices have not
been validated by the latest result. No dependencies or assets download merely
by previewing a workflow.

### 2. Prepare the inputs

Meshes, textures, spectral archives and experiment outputs are not bundled.
Camera metadata and training configurations are source-controlled. The Converse
example uses:

| Input | Path |
| --- | --- |
| Textured mesh | `data/converse/prepared/converse__free.glb` |
| Spectral archive | `data/converse/prepared/converse__free_eigen_albo_principal_curvatures.pt` |
| Cameras | [data/converse/cameras.json](../data/converse/cameras.json) |
| Configuration | [converse_training.yaml](../splatting_script/configs/converse_training.yaml) |

Obtain the exact GLB from the source/attribution link in the
[Converse asset guide](../data/converse/README.md). Then, from the repository root:

```bash
python -B -m hktex_splatting.prepare_assets --mesh /path/to/converse__free.glb --execute
# Potentially expensive, explicit one-time preprocessing; no training:
python -B -m hktex_splatting.prepare_assets --precompute --execute
# Read-only CPU path/checksum check:
python -B -m hktex_splatting.prepare_assets --check
```

Existing different meshes are never overwritten; existing spectral archives are
preserved. The renderer requires the matching archive and never regenerates it.
The CPU check is not a numerical spectral/gradient correctness certificate.
Preparation also has recorded `--asset piggy|teapot|basketball` recipes. Other
training assets need matching prepared inputs supplied separately; their presence
in the training CLI does not imply a bundled or redistributable mesh.

## Run

Use your activated environment's `python` from the repository root. If it is not
on `PATH`, replace `python` with the environment's absolute interpreter path.
No `HK_PY` or `INITIAL` variable is required. Each execution needs a new output
directory; omitting `--output` generates a timestamped one.

Preview the recommended comparison without importing Torch or launching GPU work:

```bash
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse
```

Initialize and verify both methods on a GPU before a long run:

```bash
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse --verify-only --execute
```

Run a short training smoke check, then the full comparison:

```bash
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse --iterations 40 --kernels 1000 \
  --execute --output experiments/converse_lighting_smoke

python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse \
  --execute --output experiments/converse_lighting_full
```

The full command initializes and verifies both methods before training; it does
not depend on running the separate verify command first. A failed gate stops the
workflow. Change `converse` to a prepared asset such as `skull`, `piggy`, `teapot`,
`basketball` or `punga_chips`. The bowl is unrun because the provided input lacks
a color texture; it is not a validated training recipe. Light-saber configs also
require their own prepared assets and correctness checks.

For HKS only, add `--backend hybrid`. Explicit options override the preset:
`--resolution 256`, `--iterations 1000`, `--lighting none`, or
`--material-roughness 0.2`, for example. Increasing resolution can increase
candidate scratch, tile records and GPU memory; a successful 128² run does not
validate a larger one. `--support-domain graph` selects the older approximate
hop-limited mode; it is not the policy used for the latest spectral result.

`compare-128` expands to 128², 5,000 initial kernels, 5,000 updates, seed 0,
shared native initialization, fresh nvdiffrast visibility/GPU texture targets,
spectral support, direct adapter, parallel corners, original density/optimizers,
the shared boundary-stop overlay, GGX roughness 0.112565, 2 GiB dispatch budget,
64M contribution limit, 1,024 initial validation samples and no training plots.
The light defaults are direction `(0.3, 0.8, 0.6)`, fill `0.1`, intensity `3.0`
and F0 `0.04`. This recipe explicitly enables bounded native approximation;
independent implementation, gradient and replay gates remain strict. Its initial
0.5 dB allowance is not a final-quality guarantee. The overlay changes original
HKTex's default boundary policy for both competitors and is recorded in results.

## Compare original HKTex and HKS

| Workflow | Purpose | Entry point |
| --- | --- | --- |
| Recommended adaptive camera training | Full training lifecycle with matched rasterized camera supervision, density policies and analytic lighting. | `experiments.train_surface --preset compare-128` |
| Original HKTex experiment | Original supervision/configuration and Mitsuba workflows, where configured. | `python optimisation.py --config ...` in the [main README](../README.md#quick-start) |
| Earlier fixed-kernel camera benchmark | Renderer/update microbenchmarks, memory/PSNR and separate pipeline profiles; not full adaptive training. | `benchmarks.compare_training workflow` |
| Original-render comparison | Render-only comparison with its own visibility policies; not training speed. | `benchmarks.compare_rendering` |

Module paths in this table are relative to `splatting_script`. Both competitors
in the recommended camera comparison use original HKTex model/optimizer code,
shared initial parameters, views, targets, lighting, loss and density policies.
HKS replaces field evaluation and adapts density contribution bookkeeping.
Adaptive trajectories can diverge; compare recorded kernel counts and held-out
quality, rather than assuming equal final parameters or constant kernel counts.
The camera adapter is not a run of the complete original Mitsuba experiment.

## Outputs and timing

`train_surface` retains `comparison.md`, `workflow.json`, worker logs, provenance,
verification receipts, `result.json`, initial/final checkpoint archives and
kernel-count trajectories under the output directory. These verification archives
are **not a resume API**. This workflow does not automatically export comparison
PNGs; `--no-training-plots` also disables original training plots. Image-export
capabilities in the separate renderer benchmark do not imply image exports here.

The training interval includes recurring camera work, field evaluation, loss,
backward, optimizer and density operations. It excludes initial gates/setup and
final evaluation/saving. Complete-process time also includes startup, provenance,
validation and final work. Forward-only latency uses supplied visibility and
excludes rasterization, targets, loss, backward and optimizer. Report these scopes
separately. GPU memory figures are PyTorch allocated/reserved memory and can omit
external allocations. Held-out PSNR is foreground-weighted across the recorded
held-out views, using the selected unlit or lit color space.

The older `compare_training workflow` defaults to a compact Markdown report;
`--keep-artifacts` retains its diagnostics. That single-file policy does not apply
to full adaptive `train_surface` runs, which preserve reproducibility receipts.

## Measured results and limits

Latest completed matched spectral-support comparison: skull, 128², 5,000 initial
kernels, 5,000 updates, seed 0, A100 40 GiB, original density/optimizers, shared
initialization, fresh GPU raster/targets, boundary-stop overlay and analytic GGX.

| Metric | Native HKTex | Spectral-support HKS |
| --- | ---: | ---: |
| Training interval (s) | 114.58 | 89.55 |
| Complete training process (s) | 248.09 | 225.93 |
| Forward median, visibility supplied (ms) | 10.329 | 6.764 |
| Held-out own-renderer PSNR (dB) | 26.281164 | 26.263656 |
| HKS parameters evaluated with native field (dB) | — | 26.263626 |
| Peak allocated PyTorch memory (GiB) | 22.928 | 20.443 |
| Peak reserved PyTorch memory (GiB) | 37.227 | 37.273 |
| Final kernel count | 5000 | 5000 |

This measured **1.28× training speedup** and **1.10× complete-process speedup**
with HKS 0.018 dB lower PSNR. It is one scene/seed, near quality parity, not a
universal improvement or a 5× training result. Earlier 7× renderer/component
figures concern different workloads and do not measure this full training loop.
The raw local report is `experiments/hks_accuracy_20261123/skull_full/comparison.md`;
raw experiment directories are intentionally not published in the source branch.
The work log preserves settings, checks and historical measurements.

Float64 KNN can be expensive on other GPUs. Large triangle unions and spectral
archives can require substantial memory. Graph-mode support can miss distant
spectral influences and hurt quality. There is no 4,096-source model limit: graph
workspace retries/fallbacks and older tiled-selector paths preserve their safety
checks. Configured record/byte budgets still bound feasible workloads.

## Profiling

Use a separate diagnostic run, not its timing for a speedup claim:

```bash
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse --iterations 160 \
  --profile-start 100 --profile-steps 8 --execute \
  --output experiments/converse_lighting_profile
```

Nested synchronized stage timings overlap: do not add parent and child rows.
The latest full spectral comparison was unprofiled; its current stage ranking
has not been measured. Spectral candidate distances/top-k/union are a plausible
next bottleneck, not an established measured share. Older 14.17 ms source-state
measurements included transition/compilation costs; later diagnostics were
sub-millisecond. See [the profiler guide](../docs/BENCHMARK_PROFILER.md).

## Code map

| Responsibility | Main files |
| --- | --- |
| Shared training recipe, initialization and reporting | [train_surface.py](../splatting_script/experiments/train_surface.py) |
| Fresh camera rasterization and target batches | [cuda_mesh_raster.py](rendering/cuda_mesh_raster.py), [camera_supervision.py](training/camera_supervision.py), [target_texture.py](training/target_texture.py) |
| Live spectral source parameters | [spectral_heat_field.py](modules/spectral_heat_field.py) |
| Graph/spectral support and triangle CSR | [triangle_kernel_support.py](knn_heat/triangle_kernel_support.py) |
| Three-corner coefficients and adjoints | [spectral_heat_coefficients.py](modules/spectral_heat_coefficients.py), [Triton coefficients](triton/spectral_heat_coefficients.py) |
| Triangle/tile queues and dispatch | [device_camera.py](rendering/device_camera.py), [heat_kernel_splatting.py](rendering/heat_kernel_splatting.py) |
| Pixel interpolation and contribution emission | [Triton splatting](triton/heat_kernel_splatting.py) |
| Strongest-30 RGB and backward | [splat_contributions.py](triton/splat_contributions.py), [splat_reconstruction.py](triton/splat_reconstruction.py), [splat_autograd.py](rendering/splat_autograd.py) |
| Separate material shader | [reference_mesh.py](rendering/reference_mesh.py), [fused lighting](triton/splat_reconstruction.py) |
| Original-loop adapters and density integration | [uv_training.py](training/uv_training.py), [surface_adapter.py](training/surface_adapter.py) |
| Strict implementation/gradient checks | [training_checks.py](training/training_checks.py) |

## Research history

See the [architecture](../docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md),
[work log](../docs/HKS_WORK_LOG.md), [cleanup audit](../docs/HKS_RESULTS_CLEANUP.md)
and [command reference](../splatting_script/README.md). Historical reports preserve
retired commands and earlier graph experiments; use this README for current runs.
Retained resident/eager helpers are correctness oracles, not public training modes.
Retired whole-object pickle paths need their archived source version; state-dict
archives alone do not implement training resume.
