# Heat-Kernel Splatting

This extension investigates **kernel-centric, triangle-supported, tile-based
Heat-Kernel Splatting (HKS)**. Original pointwise HKTex remains in `hktex/`.
Its authors, original experiments, installation and citation are in the
[main README](../README.md).

The current comparison uses the original trainer loop, Adam/GeodesicOpt,
schedulers and density policies with shared camera supervision. It freshly
rasterizes the fixed mesh for every training frame. The current piggy experiment
uses explicit direct lighting; earlier Converse measurements used unlit RGB.
Neither is the complete original Mitsuba/path-traced experiment or a learned
view-dependent radiance-field benchmark.

## Short initialization and training commands

From the repository root, with the environment activated and assets already
prepared, initialize the shared model and verify both methods:

```bash
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset skull --verify-only --execute
```

Remove `--verify-only` to run the full comparison. The full command already
initializes and verifies both methods before training, so a separate initialization
command or `INITIAL` shell variable is optional. Omit `--execute` to preview all
expanded commands without GPU work. Output uses a new timestamped folder under
`experiments/`; `--output experiments/my_run` chooses another new directory.

`compare-128` reproduces the lit 128² comparison recipe: both backends, 5,000
initial kernels, 5,000 updates, seed0, shared initialization, fresh nvdiffrast
visibility/GPU targets, spectral-support HKS, parallel corners, original
density/optimizers, the boundary-stop overlay, fixed GGX roughness0.112565,
2GiB dispatch budget, 64M contribution limit, 1,024 verification samples and no
training plots. Bounded native-field approximation is enabled as in the recorded
comparison; strict GPU/eager/replay checks remain mandatory. It does not download
assets or compute an eigensystem automatically.

Replace `skull` with another configured asset, such as `piggy`, `teapot`,
`basketball`, `punga_chips` or `converse`. Configs/cameras resolve automatically.
Explicit value options override the preset: for example `--iterations 40
--kernels 1000` for a short run, `--backend hybrid` for HKS only, or
`--support-domain graph` for the older graph policy. Expanded values are retained
in worker commands/manifests; the workflow records the preset and asset names.
The existing long-form CLI and its defaults remain unchanged without a preset.

## Measured status

The **2026-11-23 chips diagnostic** optimized support discovery on large meshes:
a per-kernel Triton graph traversal now builds the same hop-limited support as
the previous tensor traversal. It runs automatically through existing commands.
Each forward uses current source faces and visibility. The 1,024-face scratch
row is not a kernel-count limit: overflow uses the complete tensor traversal,
without silently dropping pairs. Existing fixed-mesh lookup tables remain in use
when available. No learned heat values or lighting are cached.

On the trained chips mesh at 128², unprofiled synchronized fresh-camera
rasterization, targets, HKS forward, loss and VJP measured 23.24–23.80 ms before
versus 11.03–11.26 ms after (two reversed-order trials, 20 samples each).
These are **short HKS-versus-previous-HKS measurements**, excluding optimizer and
density updates, not a new full-training comparison against native HKTex.
Four-view RGB/support/top-30 checks were bit exact; gradients passed existing
tolerances. The existing approximation remains: a 256-sample chips diagnostic
found only 19.8% of native positive top-30 kernel IDs inside 16-hop support.
This speed optimization does not resolve that quality limitation.
Details and raw measurements: `experiments/hks_targeted_20261123/comparison.md`.

### Spectral-support accuracy experiment

Add `--support-domain spectral` to `splatting_script.experiments.train_surface`
commands to investigate global spectral candidates instead of mesh-edge hops.
The default remains `--support-domain graph` for existing experiment reproduction.
The native backend ignores the HKS selection policy; both workers record it in
their shared controls. The original `hktex/` implementation is unchanged.

The spectral variant selects each visible pixel's nearest configured number of
sources (50 in these recipes), builds the union of kernel/triangle pairs, evaluates
three triangle corners, then uses the existing kernel/tile emitter and top-30
RGB reconstruction. The emitter checks each pixel's own candidate membership:
a candidate from another pixel on the triangle cannot leak into its top-30.
This removes graph-hop truncation, including across disconnected components.
Every forward refreshes source embeddings, candidates and differentiable
coefficients; positions, shapes, density and optimizer updates remain live.

Selection uses bounded float64 matrix distances; the eager oracle uses direct
float64 `cdist`. This has the native Euclidean nearest-neighbor policy, but is
not bitwise equivalent to native float32 FAISS at close cutoffs. Keep strict
GPU/eager checks enabled. `--allow-native-approximation` can explicitly permit
native-field differences within the existing initial quality gate, without
waiving implementation checks. GPU throughput for float64 varies by device.

On frozen trained skull parameters at 128², synchronized forward+loss+VJP with
supplied visibility measured 13.57–14.00 ms native versus 9.02–9.04 ms spectral
HKS. Four training views' target PSNR differed by less than 0.001 dB. Individual
pixels still differed by up to 0.042 RGB; this is not a dense bitwise-equivalence
claim. A strict 256-sample RGB/VJP gate, synthetic triangle-union adjoints,
8,193-source selection, an empty view and a 64-update inherited-loop smoke
test passed. These component checks do not establish better converged quality
or a full-training speedup. Results and the fresh training comparison are kept
under `experiments/hks_accuracy_20261123/`.

The subsequent fresh skull 128² pair completed 5,000 updates per method with
matching initial/control/camera receipts and original density/optimizer policies:

| Metric | Native HKTex | Spectral-support HKS |
|---|---:|---:|
| Synchronized training interval (s) | 114.58 | 89.55 |
| Complete training process (s) | 248.09 | 225.93 |
| Held-out deployed PSNR (dB) | 26.281 | 26.264 |
| Peak allocated PyTorch memory (GiB) | 22.93 | 20.44 |

This pair measured **1.28× faster updates** and **1.10× faster complete processes**,
with 0.018 dB lower aggregate PSNR: near parity, not better quality than native.
HKS's own-renderer/native-field evaluation of its final weights differed by
only 0.00003 dB in aggregate. Both replaced density parameters at steps 1,000 and
2,000 and ended with 5,000 kernels; optimizer bindings and fixed assets passed
verification. Reserved PyTorch memory was slightly higher for HKS (37.27 versus
37.23 GiB), so allocated-memory savings do not imply lower reserved memory.
This is one scene/seed; the other meshes require new global-support comparisons.
There is no 5× full-training claim. See
[the full report](../experiments/hks_accuracy_20261123/skull_full/comparison.md).

Graph mode now retries overflowing GPU neighborhoods with bounded 2,048/4,096
face scratch rows before invoking the exact tensor fallback. These are per-source
workspace sizes, not kernel-count limits. No overflow is silently truncated.

### Earlier lit piggy comparisons

The latest **lit piggy** comparisons use 5,000 updates, 5,000 initial kernels,
16-hop support, fresh GPU rasterization and the original optimizers/density
policies on one A100 40 GiB. Both methods share initialization, camera sequence,
loss and fixed GGX lighting. These are preliminary single-scene/seed results:

| Metric | Native 128² | HKS 128² | Native 320² | HKS 320² |
|---|---:|---:|---:|---:|
| Synchronized training interval (s) | 112.27 | 62.55 | 516.73 | 106.84 |
| Complete training worker body (s) | 133.83 | 90.68 | 543.51 | 233.59 |
| Foreground linear-radiance PSNR (dB) | 33.33 | 33.19 | 34.54 | 34.71 |
| Peak allocated PyTorch memory (GiB) | 5.63 | 1.41 | 28.69 | 1.68 |
| Final kernels | 4,098 | 3,983 | 5,000 | 5,000 |
| Forward median (ms; visibility supplied) | 12.44 | 5.72 | 64.31 | 9.20 |

At 128×128, HKS measured **1.79× faster updates** and **1.48× faster complete
workers**, with a 0.14 dB aggregate PSNR deficit. Left/right scores were
35.93/31.45 dB native and 34.92/31.74 dB HKS: the left view remains 1.01 dB lower.
At 320×320, HKS measured **4.84× faster updates**, **2.33× faster workers**, and
0.17 dB higher aggregate PSNR in this pair. Left/right were 37.89/32.37 native
and 38.56/32.39 HKS. Supplied-visibility forward timing was about 7× faster at
320², but it excludes rasterization, targets, loss, backward and optimization.

**The requested 5× full-training/pipeline result is not established.** Do not
round 4.84× to 5× or substitute render-only/component timings for complete-worker
timing. This run's HKS import/CUDA/config setup took 85.8 s versus native's
6.2 s; those waits are retained in worker totals, not erased. Startup cause is
not fully isolated. Training excludes initial gates/first-frame preparation;
worker time includes setup, internal checks, evaluation and saving. Neither
includes one-time spectral preprocessing or separate workflow verification jobs.

Reports: `experiments/piggy_optimized128_full5000_20261122/comparison.md` and
`experiments/piggy_optimized320_full5000_20261122/comparison.md`.
The [work log](../docs/HKS_WORK_LOG.md) retains source snapshots, commands,
checks, failed trials and earlier results. Historic **unlit Converse** comparisons
measured 3.91× updates / 2.41× workers at 192² and 2.45× / 1.78× at 128². Those
use different scenes/supervision and older source snapshots; do not combine them
with lit piggy results or call them ray-traced lighting speedups.

Current GPU source/grid, analytic material lighting and fresh camera/tile packing
recompute learned state and visibility each update. No heat equations, top-30
reconstruction, mesh geometry, optimizer or density algorithm were changed.
Original `hktex/` files match the saved starting tree. Strict HKS GPU/eager
RGB/VJP checks pass; strict native-field equivalence remains false and is
reported separately as an accepted approximation. Both full comparisons exited
normally. The final 320² checkpoint diagnostic wrote passing numerical results
and previews before Slurm terminated its process at a four-minute limit; clean
exit is not validated for that diagnostic. A matched ray-traced comparison remains
unimplemented: this is known analytic direct lighting with no shadows/indirect light.

## Rendering and training pipeline

1. Load the fixed mesh and matching precomputed Laplace–Beltrami eigensystem.
2. Refresh learned kernel locations, anisotropy/orientation, spectral source state,
   colors and filter parameters. Triton preserves the native bilinear four-grid
   mixture and fuses source spectral gathers, decay, normalization and embedding.
   Source geometry is reused only once within its current update, with mutation
   guards; it is not a cross-update learned-state cache.
3. Starting from each kernel's current source triangle, discover its fixed-hop
   mesh neighborhood. Intersect it with the current visible triangles and pack
   ordered kernel–triangle support on the GPU.
4. Evaluate spectral heat at each supported triangle's three corners and prepare
   its intrinsic-distance coefficients.
5. Group supported triangle patches into screen tiles. Triton interpolates corner
   values with pixel barycentrics, applies the original distance weighting/filter,
   and emits contributions.
6. Keep the strongest 30 filtered contributions and reconstruct RGB with HKTex's
   weighted sum, denominator floor, learned mean color and clipping.
7. Custom backward propagates to the kernel parameters. Original density callbacks,
   Adam, geodesic movement, schedules and clamps perform training updates.

Optional lighting shades reconstructed material albedo before loss/evaluation;
it leaves the heat-kernel contributions and top-30 reconstruction unchanged.

The graph hop limit is **approximate support**, not a conservative spectral bound.
Heat is not assumed to decrease along adjacency. Wider support can improve
coverage while increasing work; it does not guarantee native outer-KNN parity.
No Gaussian primitives or 3DGS alpha compositing replace the heat kernels.

Fixed connectivity and a bounded neighborhood lookup are reused. Learned heat,
coefficients, gradients and camera visibility are recomputed. Backend-specific
compiler/extension/font artifacts may be reused outside the repository; loader
and temporary run workspaces remain unique. There is no source-count cutoff at
4,096: exact support checks cover 20,000 sources, subject to memory/index limits.
If a geometry lookup cannot fit, exact graph traversal remains available.

GPU support is checked against the retained tensor selector; RGB and gradients
are checked against eager spectral/splat operations. Native-field equality is
reported separately. It is required by default; explicit
`--allow-native-approximation` permits a research experiment only when strict
implementation checks and the declared initial quality bound pass. It never
marks failed native equality as passing, and final quality must be measured.

## Optional lighting, separate from material

Kernel colors and mean color still reconstruct **sRGB material albedo**. The
optional shader converts it to linear RGB and applies a white world-space distant
light and diffuse ambient fill. `--lighting diffuse` uses Lambert reflection;
`--lighting ggx` adds a view-dependent isotropic GGX highlight with Smith masking
and Schlick Fresnel. Camera motion changes the highlight through the current
surface-to-camera direction. Light direction/intensity can change independently
of the learned albedo. Fixed face normals follow mesh winding; no camera-facing
normal flip is applied. Roughness and Fresnel F0 are fixed user controls, not new
learned kernel parameters. Default `--lighting none` preserves unlit RGB.

Both training backends use the same shader **after** material reconstruction,
with original optimizers/density policies and gradients through albedo. Synthetic
lit targets use the same known lighting applied to the original texture. This
assumes that texture is albedo: it does not remove baked illumination from a GLB,
recover unknown lights, or fit roughness. The shader returns unclipped linear
radiance, which can exceed 1; loss/PSNR use this space with reference peak 1.
Those PSNR values are not comparable with the old unlit sRGB scores.

Use this short paired correctness/training smoke command in a CUDA environment
with the prepared Converse assets. Omit `--execute` for a CPU workflow preview:

```bash
python -B -m splatting_script.experiments.train_surface \
  --backend both --supervision camera --resolution 64 \
  --kernels 1000 --iterations 4 --seed 0 \
  --raster-backend nvdiffrast --target-sampling gpu \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct \
  --surface-support-hops 16 --corner-evaluation parallel --max-buffer-mib 768 \
  --density-policy original --optimizer-policy original \
  --lighting ggx --light-direction 0.3 0.8 0.6 \
  --light-intensity 3 --light-ambient 0.1 \
  --material-roughness 0.35 --material-specular 0.04 \
  --allow-native-approximation --no-training-plots --verify-samples 1024 \
  --execute --output experiments/lighting_smoke
```

Add `--verify-only` to stop after verification. Use a fresh output directory and
fresh initial state: old unlit receipts are intentionally rejected by lighting
and source provenance checks. Do not start a long lit run until the GPU gates and
short training smoke pass. CPU shader numerical/gradient and CLI checks pass.
A100 piggy-mesh raster/shader and actual CameraSupervision/TargetTexture checks
pass on two fresh 64×64 views, including finite material-color gradients and
exact CPU/GPU target agreement. Piggy native/HKS lit training also completes a 1,120-update lifecycle check, and
trained-checkpoint GPU/eager kernel-parameter VJPs pass. Its quality deficit
exceeds the provisional target; no matched-quality lit speedup is established.

The reusable API is `shade_material` / `shade_batch` in
`rendering/reference_mesh.py`; camera geometry/targets are prepared in
`training/camera_supervision.py`. Saved training checkpoints retain light,
roughness/F0 and color-space settings. Benchmarks record the same settings and
include shading in forward/training times. Existing unlit benchmark commands
retain their original defaults. No new package dependency is required.

This is **analytic direct lighting, not ray tracing**: no shadow rays, indirect
illumination, environment maps, metallic material or learned BRDF. A future
Mitsuba/ray-traced comparison must match lights, BRDF, color space, shadows,
cameras, quality and timing scopes before making a lit-rendering speed claim.
The GGX/Smith reference is [PBRT's microfacet chapter](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory);
our perceptual roughness convention is alpha = roughness squared, and Fresnel
uses Schlick's approximation rather than PBRT's exact dielectric BSDF.

## Piggy asset setup and short lit check

The piggy is a separate experiment. Its source GLB currently lives at
`data/converse/raw/piggy_bank.glb`; this original is preserved. `--asset piggy`
copies the exact recorded bytes to `data/piggy/prepared/piggy_bank.glb` and builds
its own spectral archive there. Never rename or reuse the Converse eigensystem.
The original loader creates a texture atlas from the piggy's constant material
colors, so a new material-color loader is unnecessary. Its constant atlas cannot
support the original texture-guided weighted initialization; the piggy config
selects HKTex's existing surface-random initialization for BOTH methods. Spectral
settings, density policies and optimizers retain the original recipe. The embedded GLB metadata
credits octopuslover and declares CC-BY-SA-4.0; preserve that source metadata.

Prepare and check inputs (spectral generation can take several minutes and needs
CUDA plus the full original HKTex preprocessing dependencies):

```bash
python -B -m hktex_splatting.prepare_assets --asset piggy \
  --mesh data/converse/raw/piggy_bank.glb --execute
python -B -m hktex_splatting.prepare_assets --asset piggy --precompute --execute
python -B -m hktex_splatting.prepare_assets --asset piggy --check
```

The following is a **short integration check**, not a performance benchmark.
It must pass native/HKS verification before either training worker starts:

```bash
python -B -m splatting_script.experiments.train_surface \
  --config splatting_script/configs/piggy_training.yaml \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --cameras-file data/piggy/cameras.json \
  --backend both --supervision camera --resolution 64 \
  --kernels 1000 --iterations 4 --seed 0 \
  --raster-backend nvdiffrast --target-sampling gpu \
  --shared-initial-model --surface-adapter direct \
  --surface-support-hops 16 --corner-evaluation parallel --max-buffer-mib 768 \
  --density-policy original --optimizer-policy original \
  --lighting ggx --material-roughness 0.112565 --material-specular 0.04 \
  --allow-native-approximation --no-training-plots --verify-samples 1024 \
  --execute --output experiments/piggy_lit_smoke
```

Omit `--execute` for a CPU command preview, or add `--verify-only` to stop after
verification. Use a fresh output directory each time. The roughness is the pink
body's source value; this command applies fixed global roughness/F0, not the GLB's
full per-material BRDF. Default Converse commands/cameras remain unchanged.
`top`, `side`, `three_quarter`, `logo_closeup` are retained view-name aliases for
the piggy camera file; held-out views use the existing fixed rotations.

Current A100 piggy checks pass camera/texture targets, initial heat-field
GPU/eager checks, original density/pruning transactions, optimizer bindings and
trained-checkpoint RGB/VJPs for all seven learned parameter groups.

Controlled 64×64 / 1,000-initial-kernel / 1,120-update shader trials measured HKS
training intervals of **14.46 s** with the Torch shader and **12.74 s** with the
fused shader. Adding exact GPU camera packing measured **12.19 s** (about 19%
higher update throughput than the Torch control). Native/HKS in that final pair
measured 21.35/12.19 s, worker bodies 42.95/39.80 s and PSNR 30.04/30.01 dB.
These small-run results do not substitute for the full 5,000-update table above.
Quality/count trajectories vary across repeats; an earlier 0.80 dB deficit and
all failed initialization logs remain recorded. Do not attribute stochastic
quality changes to this numerically equivalent optimization.

`--lighting-backend auto` uses two fused CUDA kernels for fixed contiguous
float32 camera geometry (forward and first-order albedo backward).
`--lighting-backend torch` preserves the tensor reference for both competitors.
CPU/double inputs, noncontiguous normal/view geometry and geometry derivatives
retain Torch; use the Torch backend when higher-order derivatives are needed.
Light values/pixel counts remain runtime inputs, so moving cameras/lights do not
cache learned values or specialize every frame. The independent eager HKS oracle
always uses Torch shading. Fifty shader cases, real piggy geometry, integer
settings and strided dispatch were validated at unchanged tolerances.

The camera fast path packs fresh IDs/barycentrics, tile lanes/counts and face
pixel counts in one pass. It preserves immutable snapshots, version guards and
ID/order/barycentric/foreground-count validation. Integer arrays and barycentrics
match the old builder bit-for-bit; unsupported layouts retain it. Triangle/tile
sorting and allocation readbacks still exist: this is not a zero-synchronization
or cached-visibility renderer. Microbenchmarks and exact source revisions are in
the work log; component gains are not whole-training speedups.

## Setup

### 1. Install the environment

Use Linux, Python 3.11 and an NVIDIA GPU supported by CUDA PyTorch/Triton.
A100 is the recorded validation device; other CUDA GPUs must pass the correctness
gate before use. CPU-only machines can inspect command previews and asset checks.

The GPU camera path requires nvdiffrast. Set `CUDA_HOME` to a compatible full
CUDA toolkit root and put its `bin` directory on `PATH`; verify `nvcc --version`
and provide a compatible C++ compiler before building. PyTorch CUDA wheels alone
do not provide the full build toolkit.

The following Conda/Mamba example installs GPU FAISS and the recorded CUDA 12.9
PyTorch build. Use an NVIDIA driver compatible with those wheels. If you already
have a compatible original-HKTex environment, reuse it and install the project
extra instead of creating another environment. Run project commands from the
repository root.

```bash
mamba create -n hktex-hks python=3.11
mamba activate hktex-hks
mamba install -c pytorch -c nvidia -c rapidsai -c conda-forge \
  libnvjitlink=12.9.86 faiss-gpu-cuvs=1.13.1 "numpy<2.3"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install ninja
python -m pip install -e ".[native,gpu-raster]" --no-build-isolation
python -m pip check
python -c "import faiss; assert hasattr(faiss, 'StandardGpuResources')"
HK_PY=${HK_PY:-python}
```

`pyproject.toml` declares HKS runtime dependencies, including Numba. The `native`
extra installs the original-HKTex Python dependencies required by the shared
loader; optional tiny-cuda-nn texture baselines are not required. Triton must
match PyTorch, so do not substitute an unrelated Triton version. Native extensions
may need their upstream compiler/CUDA toolkit prerequisites.

**GPU FAISS is a separate native prerequisite**, not a dependency pip can safely
replace with `faiss-cpu`. Mamba is one way to install it, not a runtime requirement.
For another environment manager or CUDA driver/toolkit, install compatible CUDA
PyTorch wheels and GPU FAISS using upstream instructions, then run the correctness
gate. The example versions preserve the tested Torch selector policy. Installation
does not download mesh/spectral assets. Use an editable source checkout; experiment
commands depend on repository configurations and provenance.

### 2. Prepare the inputs

The current comparison harness is configured for the Converse experiment:

| Required input | Repository-relative path |
| --- | --- |
| Textured mesh | `data/converse/prepared/converse__free.glb` |
| Matching spectral archive | `data/converse/prepared/converse__free_eigen_albo_principal_curvatures.pt` |
| Recorded cameras | `data/converse/cameras.json` |
| Training configuration | [converse_training.yaml](../splatting_script/configs/converse_training.yaml) |

Follow the [asset guide](../data/converse/README.md) for mesh and texture layout.
Use the source link, attribution and checksum in the asset guide. Prepare the
exact downloaded GLB, explicitly generate the original spectral archive once,
and check inputs before training:

```bash
"$HK_PY" -B -m hktex_splatting.prepare_assets --mesh /path/to/converse__free.glb --execute
"$HK_PY" -B -m hktex_splatting.prepare_assets --precompute --execute
"$HK_PY" -B -m hktex_splatting.prepare_assets --check
```

Precomputation can be expensive. It runs only with both `--precompute` and
`--execute`, uses the unchanged original HKTex eigensystem code and performs no
training. Existing meshes/archives are preserved. Preview preparation by omitting
`--execute`. The renderer itself requires the archive and never regenerates it.

The recorded [camera JSON](../data/converse/cameras.json) contains
the `top`, `side`, `three_quarter` and `logo_closeup` views. The two held-out views
are derived by the camera helper. Keep these definitions unchanged when reproducing
the recorded comparison. CLI previews work without loading these assets.

Execution requires a usable NVIDIA GPU and `nvidia-smi`; **Slurm is not required**.
On a workstation, run directly in your activated environment. On a cluster, use
its GPU allocation procedure. Tools never allocate resources automatically.
The harness warns on GPU models other than the validated A100 and records actual
hardware for comparisons. CPU previews require no GPU.

### Setup verification and limits

Dependency declarations were checked against the existing research environment;
an offline wheel build, CPU command previews, input checks and file-preservation
checks passed. Preprocessing wrapper checks used a stub writer, not a numerical
eigensystem computation. A fresh environment installation, regenerated spectral
archive and non-A100 GPU execution have not been validated. Run the GPU gate
below before starting a full comparison in a new environment.

## Validate, train and compare

Activate the environment and set `HK_PY` to its Python interpreter. Run commands
from the repository root. No command requests a GPU allocation automatically.
Each output directory must be new.

This uses the completed 16-hop study's settings. Set `--resolution 192` to
repeat the latest completed scaling study. Library defaults remain
8 hops and serial corners; choose and report these approximation settings
explicitly. The 768 MiB budget is a per-buffer/support guard, not a cap on total
GPU memory.

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --supervision camera --visibility fresh \
  --resolution 128 --kernels 5000 --iterations 40 --seed 0 \
  --raster-backend nvdiffrast --target-sampling gpu \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct \
  --surface-support-hops 16 --support-cap 0 --corner-evaluation parallel \
  --max-buffer-mib 768 \
  --density-policy original --optimizer-policy original \
  --allow-native-approximation --max-initial-psnr-drop-db 0.5 \
  --no-training-plots --verify-samples 1024 \
  --verify-only --execute --output experiments/hks_verify
```

Omit `--execute` for a CPU-only JSON workflow preview. After verification,
remove `--verify-only` and choose a fresh output directory for the 40-update
smoke test. After it passes, set `--iterations 5000` for the full comparison.
Use `--backend hybrid` for HKS alone; use `both` for matched native/HKS studies.
The installed `hktex-splatting-surface-train` command calls this same entry point.

The boundary-stop overlay sets `avoid_holes=false` for BOTH methods. This differs
from the original default boundary policy and is labeled in the report. Kernel
counts adapt under the same original density policy; trajectories can diverge
when the learned fields differ. Locations move throughout training, not only
when density changes occur.

Read `comparison.md` in the output directory. JSON receipts, logs, trajectories
and final checkpoints are retained for verification. Shared initialization is
created automatically. Do not reuse an initial snapshot after source, GPU,
configuration or asset changes: the strict provenance check rejects that drift.
Snapshots are archival; there is no resume CLI.

The 0.5 dB command-line bound applies ONLY to initial validation samples. It is
not a guarantee about final quality. Final PSNR uses foreground-sample-weighted
MSE across `heldout_left` and `heldout_right`; per-view values remain in receipts.
These are views of the same textured mesh, not an independent scene dataset.
Forward latency uses the saved first view and excludes rasterization, targets,
loss and backward. Training timing includes fresh rasterization and updates;
worker/process timing additionally includes setup, checks, evaluation and saving.
Memory reports cover PyTorch allocations, excluding some driver/FAISS memory.

## Profiling

Use a separate run with the same flags, `--iterations 1120 --profile-start 980
--profile-steps 128`, and a new output directory to examine pruning and
densification. Nested stage times overlap and must not be summed. Instrumented
times are diagnostic; use unprofiled runs for speed claims.

The latest diagnostic window across density changes measured source preparation
at 0.47 ms and backward at 2.55 ms. Earlier source preparation averaged 14.17 ms
across a compilation-heavy transition; that was not its steady-state cost. These
profiled figures are not the unprofiled training times above. Integer tile-task
expansion was checked bitwise against the retained original scheduler.

HKS routes padding into one discarded scratch accumulator row and invokes each
original density pre-callback once. Original model buffers, controller objects,
post-callbacks and optimizer bindings remain intact. Valid zero-weight hits still
count. Compared accumulator values pass the original float tolerances; identical
bitwise training trajectories are not promised. The grouped path remains for
unsupported/controller-override cases.

## Code map

Paths below are relative to `hktex_splatting/` unless stated otherwise.

| Stage | Files / main API |
|---|---|
| Current training/benchmark entry point | `splatting_script/experiments/train_surface.py` (repository root) |
| Native loop and optimizers | Original `hktex/trainers/base.py`, `hktex/modules/geodesic_opt.py` (repository root) |
| Camera rasterization and targets | `rendering/cuda_mesh_raster.py`, `training/camera_supervision.py`, `training/target_texture.py` |
| Source spectral state | `modules/spectral_heat_field.py` — `LiveHeatField.sources`; `modules/spectral_heat_coefficients.py` — `fused_source_grid`, `fused_source_state`; matching kernels in `triton/spectral_heat_coefficients.py` |
| Kernel-centric support | `knn_heat/triangle_kernel_support.py` — `KernelSurfaceSupport` |
| GPU support count/pack | `triton/resident_gpu.py` — `count_kernel_surface_support`, `pack_kernel_surface_support` |
| Fresh screen geometry | `rendering/device_camera.py` — `DeviceCameraSurface`, `DeviceCameraDispatch` |
| Corner heat and Gram terms | `modules/spectral_heat_coefficients.py`, `triton/spectral_heat_coefficients.py` |
| Tile work and splatting | `rendering/heat_kernel_splatting.py` — `triangle_tile_work`; `triton/heat_kernel_splatting.py` — `expand_triangle_work`, `group_sorted_triangle_work`, `_hybrid_emit` |
| Strongest contributions and RGB | `triton/splat_contributions.py`, `triton/splat_reconstruction.py` |
| Backward | `rendering/splat_autograd.py`, spectral coefficient/source adjoints |
| Training bridge and safety checks | `training/uv_training.py`, `training/surface_adapter.py`, `training/training_checks.py` |
| Stage profiling | `training/surface_profile.py` |

The historical `resident` filename/class names identify shared utilities and
retained correctness references, not another main benchmark competitor.
See the [architecture guide](../docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md).

## Retained workflows and limitations

- Original author training remains in [`optimisation.py`](../optimisation.py).
- `splatting_script.benchmarks.compare_training workflow` is the older fixed-density
  camera benchmark. `python -m hktex_splatting`, `hktex-splatting-train` and
  `hktex-splatting-compare` retain that older rendering/fitting entry point;
  historical `2dgs`/`original` aliases select HKS/native there. Their timings are
  not the adaptive-training measurements above.
- The [original-render comparison](../docs/ORIGINAL_RENDER_VS_HKS.md) separates
  native lit/denoised Mitsuba rendering from HKS unlit RGB; they are different
  rendering workloads.
- Old occupancy/certified-cache/custom-policy CLI experiments are rejected by the
  current kernel-centric training comparison. Historical custom-policy/occupancy/cache sources are archived for reproducibility;
  their unused live modules are removed. See [the HKS-only cleanup](../docs/HKS_RESULTS_CLEANUP.md).
- Mesh/camera/visibility gradients and higher derivatives are unsupported. Original
  model/optimizer state dictionaries remain; old whole-object pickles may need
  their original source. No original HKTex files are edited by the HKS adapter.
- Fresh installation, regenerated eigensystems and non-A100 execution have not
  been validated in this optimization session. Correctness gates remain required.

Historical results and recovery: [work log](../docs/HKS_WORK_LOG.md),
[pipeline cleanup](../docs/HKS_PIPELINE_ONLY_CLEANUP.md),
[unused-code cleanup](../docs/HKS_UNUSED_CODE_CLEANUP.md).
See [LICENSE](../LICENSE) and the main README for original attribution/citation.

## Additional local meshes

The added GLBs are preserved in `data/`. The teapot and basketball now have
independent asset recipes and fixed camera files:

| Asset | Training config | Camera file | Original-loader faces |
|---|---|---|---:|
| Teapot | `splatting_script/configs/teapot_training.yaml` | `data/teapot/cameras.json` | 19,378 |
| Basketball | `splatting_script/configs/basketball_training.yaml` | `data/basketball/cameras.json` | 560 |

Check inputs with `python -B -m hktex_splatting.prepare_assets --asset teapot --check`
(or `--asset basketball`). Missing spectral archives require explicit preprocessing:
`python -B -m hktex_splatting.prepare_assets --asset teapot --precompute --execute`.
Run preprocessing on a CUDA device; it uses the original HKTex solver. Then use
the chosen config and camera file with the existing `train_surface` command,
starting with `--verify-only` and a short smoke run. Keep all comparison settings
shared between native and HKS. A reflective-looking GLB does not enable ray tracing:
current supervision uses its base color and explicitly configured analytic lighting,
not imported roughness/metallic/normal maps or indirect reflections.

The bowl has no texture UV/image in the original loader and needs supervision review.
The skull and chips contain roughly 763,000 and 384,000 triangles; their full spectral
archives may be costly. The two light-saber files are distinct meshes. None was deleted.
See `experiments/mesh_queue_optimization_20261122/mesh_inventory.json` for exact inputs.

The active Triton scheduler now uses a blocked sorted-key run scan and avoids a
redundant identity permutation in top-30 selection. Queue keys, coefficient ordering
and contribution tie rules are preserved; reference renderers retain their old index
path. Component GPU checks pass, but component timings alone are not training claims.

Both new assets have finite original-solver spectral archives and completed GPU
verification plus 40-update native/HKS smoke runs at 64×64 with 1,000 kernels.
Reports: `experiments/mesh_queue_teapot_smoke_20261122/comparison.md` and
`experiments/mesh_queue_basketball_smoke_20261122/comparison.md`. These are integration
checks, not converged-quality or steady-state speed benchmarks. HKS was slower
on the teapot's short small-resolution run, so the piggy speed result must not
be generalized to every mesh. The basketball exercised 716 contributions on one
face, including the retained top-30 overflow handling above 512.

The 5,000-update piggy table above predates this scheduler optimization. Its new-code
checks include exact trained-model RGB/top-30 replay, seven parameter VJPs and a
1,120-update density-transition run. The measured trained forward+loss+VJP benefit
is approximately 0.8–1.4%; a new complete 5,000-update timing has not been measured.
