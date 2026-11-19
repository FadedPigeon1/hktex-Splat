# Heat-Kernel Splatting

This extension renders and trains **triangle-supported, tile-based Heat-Kernel
Splatting (HKS)** on a fixed mesh. It reuses original HKTex's learned spectral
heat kernels and optimizers, evaluates heat at triangle corners, and shares
those values across visible pixels using GPU screen tiles.

For the original HKTex paper, authors, installation, training and citation, see
the [main README](../README.md).

## Original HKTex versus HKS

| Implementation | How it evaluates visible pixels | Command identifier |
| --- | --- | --- |
| Original HKTex | Finds pointwise kernel neighbors and evaluates spectral heat at each query point. | `native` |
| Heat-Kernel Splatting | Selects kernel–triangle pairs, evaluates three corner values, then interpolates and splats contributions in screen tiles. | `hybrid` (default) |

`hybrid` is the historical command name for HKS. `original` and `2dgs` are
compatibility aliases for `native` and `hybrid`, respectively. The `2dgs` alias
does not select conventional 2D Gaussian Splatting.

Both use spectral heat kernels and HKTex-style weighted RGB reconstruction.
Triangle support and interpolation are approximations, so HKS images and gradients
need not equal pointwise HKTex's. HKS does not replace the kernels with Gaussian
primitives or use depth-sorted alpha compositing.

## How it works

1. **Load the surface and spectral archive.** The mesh and its precomputed
   Laplace–Beltrami eigensystem remain fixed throughout training.
2. **Find visible surface samples.** Rasterization records the visible triangle
   and barycentric coordinates for each foreground pixel.
3. **Prepare the learned kernels.** Current source locations, anisotropy and
   orientation determine the spectral source state; learned colors and filters
   determine each source's appearance.
4. **Select kernel–triangle support.** Triangle spectral centroid/radius bounds
   find candidate kernels. Ordered support records which kernels can contribute
   to each visible triangle. The matched benchmark caps this pool at 128 kernels
   per triangle.
5. **Evaluate three corners.** GPU coefficient computation evaluates spectral
   heat at the triangle's three vertices and prepares six intrinsic-distance
   coefficients for every retained kernel–triangle pair.
6. **Schedule screen tiles.** Supported visible triangle patches are grouped
   into kernel/tile work. Support comes from mesh triangles; no independent
   projected kernel footprint is used.
7. **Emit pixel contributions with Triton.** The emitter uses barycentric
   coordinates to interpolate corner values, computes intrinsic-distance
   weighting, applies the learned filter, and writes scalar contributions.
8. **Reconstruct RGB.** Keep the strongest 30 filtered contributions per pixel.
   Combine their learned colors using HKTex's weighted reconstruction, including
   its denominator floor, learned mean color and clipping.
9. **Train through the renderer.** Custom backward operators propagate gradients
   through selected contributions and spectral coefficients. Original Adam,
   GeodesicOpt, schedulers and clamps update the parameters. Kernel locations
   move along the fixed mesh.

The eigensystem is reused rather than recomputed every step. Configured diffusion
time stays fixed. Discrete support, visibility and top-30 membership are held
fixed during each backward pass; kernel parameters remain trainable.

See the [architecture and call flow](../docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md)
for equations, function names and the full dependency map.

## Setup

### 1. Install the environment

Use Linux, Python 3.11 and an NVIDIA GPU supported by CUDA PyTorch/Triton.
A100 is the recorded validation device; other CUDA GPUs must pass the correctness
gate before use. CPU-only machines can inspect command previews and asset checks.

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
python -m pip install -e ".[native]" --no-build-isolation
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
| Recorded cameras | `experiments/converse_fidelity/cameras.json` |
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

The recorded [camera JSON](../experiments/converse_fidelity/cameras.json) contains
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

## Run HKS

Start with a preview, which prints the configuration without importing Torch or
starting GPU work:

```bash
"$HK_PY" -B -m hktex_splatting --backend hybrid \
  --kernels 1000 --resolution 128 --iterations 40 --support-selector fused
```

With the inputs ready and a CUDA GPU available, add `--execute` to
render and perform the configured training updates:

```bash
"$HK_PY" -B -m hktex_splatting --backend hybrid \
  --kernels 1000 --resolution 128 --iterations 40 --support-selector fused \
  --execute
```

This is a short single-renderer training/timing run. Its `DIRECT_TILES_RUN` output
identifies a unique directory under
`experiments/converse_heat_kernel_splatting/results/` with detailed artifacts.
Use the matched workflow below for research comparisons, periodic quality
assessment and separate profiles. Single-renderer commands keep detailed outputs;
the one-Markdown output mode belongs to the matched workflow.

To use original HKTex through the same harness, change `--backend hybrid` to
`--backend native`. Original author training remains available through
[`optimisation.py`](../optimisation.py), as described in the main README.

The installed commands `hktex-splatting-train` and `hktex-splatting-compare` route
to the same harness. Use `--help` to inspect its controls.

## Compare original HKTex and HKS

### Validate and run the 5,000-kernel experiment

Activate the environment installed above, open a terminal in the repository
root, and run the independent GPU correctness gate. Use your institution's
normal GPU allocation procedure when working on a cluster; no account-specific
filesystem layout or environment name is required.

```bash
HK_PY=${HK_PY:-python}

"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --kernels 5000 --resolution 32 --iterations 1 \
  --support-selector fused --validate-only --execute
```

Continue only after `DIRECT_TILES_VALIDATION=PASSED`. This gate passed on an
allocated A100 at 5,000 kernels and 32×32; the saved artifact confirms support,
RGB, coefficient, gradient and optimizer checks. That result establishes this
small-image gate, not completion or performance of the full workload.

Then run 1,000 updates per renderer at 256×256, with a unique report filename:

```bash
OUT="experiments/comparison_full_$(date +%Y%m%d_%H%M%S).md"

"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 5000 --resolutions 256 --iterations 1000 --seed 0 \
  --allow-native-variation --execute --output "$OUT"

cat "$OUT"
```

The workflow generates shared initialization, preflights, capacity pilots,
training runs and separate profiles. It retains one Markdown report and does not
automatically allow selector fallback. Because this command permits the known
native repeatability variation, the report is **DIAGNOSTIC / UNVALIDATED**;
HKS checks remain strict. Remove `--allow-native-variation` for a strict run,
which stops if native repeatability checks fail. Preserve the configured common
matmul precision even if PyTorch prints a performance suggestion.

### General workflow commands

Preview the complete plan first:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 5000 --resolutions 256 --iterations 1000 --seed 0
```

Execute the same plan on your CUDA GPU:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 5000 --resolutions 256 --iterations 1000 --seed 0 \
  --execute --output comparison.md
```

The workflow creates one shared initialization, runs both correctness preflights
and their paired input gate, then runs capacity pilots, training measurements and
separate profiles. You do not need to find or supply an `initial.pt` yourself.

**The retained output is `experiments/comparison.md`.** It includes job/gate
statuses, failure summaries, training time, update statistics, PyTorch peak GPU
memory, loss and training/held-out PSNR. Deployed-renderer quality and quality
from the common original-HKTex evaluator are reported separately.

Bare `.md` names go under `experiments/`; explicit paths are respected. Choose a
new filename for each run because existing reports are not overwritten. Temporary
logs, initialization, snapshots, images, traces and compiler caches are removed
on normal exit or handled interruption. Forced termination may leave temporary
files. Existing research results and explicitly supplied initialization files
are preserved.

For a four-update smoke run, use a separate report:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 1000 --resolutions 128 --iterations 4 --seed 0 \
  --pilot-iterations 4 --profile-start 0 --profile-steps 4 \
  --execute --output comparison_smoke.md
```

To retain checkpoints and full profiling/debugging records, replace the output
arguments with `--keep-artifacts --output experiments/comparison_artifacts`.
That directory must be new. A compact Markdown report alone does not retain
trained models.

### Replay failures and diagnostic runs

The native 5,000-kernel workload has shown optimizer replay variability. Strict
preflight failures block measurements; do not interpret a failed workflow as a
completed comparison. HKS's interpolation approximation does not explain a
native-versus-native replay mismatch.

For diagnostic measurements despite that known native replay variation, add
`--allow-native-variation` and choose a new report filename. Reset, input,
finite-value and gradient checks remain enforced. Limited native loss/Adam replay
variation may proceed, and the native observer-parity profile gate is skipped;
HKS checks remain strict. Reports are labeled **DIAGNOSTIC / UNVALIDATED**, with
validated speedup and reduction claims withheld. See the
[native replay analysis](../docs/NATIVE_ADAM_REPLAY_ANALYSIS.md).

## Correctness checks and profiling

These are explicit GPU diagnostics, separate from speed measurements:

```bash
# Initial rendering, support, gradient and optimizer comparisons.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --execute --validate-only --support-selector fused

# Separate instrumented profile with an observer-on/off consistency gate.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --execute --pipeline-profile --profile-validate \
  --profile-start 40 --profile-steps 8 --iterations 48
```

Fused selection keeps the existing row selector through **4,096 sources**.
Above that, an all-Triton tiled selector examines every source in groups of 1,024
and merges bounded candidate lists. Large source counts alone no longer trigger
legacy candidate or CSR fallback. The wide path breaks equal-distance ties by
source ID; this may differ from the former PyTorch fallback at exact ties.
See the [selector implementation and validation notes](../docs/TILED_SELECTOR.md).
The 5,000-source, 32×32 A100 correctness gate passed as recorded above. Larger
workloads and performance need their own GPU runs; CPU interpreter checks do not
establish CUDA parity. Other input, capacity and fallback guards remain.

Detailed instructions: [command guide](../splatting_script/README.md),
[benchmark protocol](../docs/HKTEX_VS_HKS_BENCHMARK.md),
[profiler guide](../docs/BENCHMARK_PROFILER.md) and
[training protocol](../docs/HKTEX_RESEARCH_TRAINING.md).

## Where to read the code

Paths in this table are relative to `hktex_splatting/` unless indicated otherwise.

The [shared-module rename guide](../docs/HKS_SHARED_MODULE_RENAME.md) maps the
previous triangle-splat filenames to the current layout, contribution and RGB
modules. They remain parts of the same HKS pipeline.

| File | Responsibility |
| --- | --- |
| [rendering/heat_kernel_splatting.py](rendering/heat_kernel_splatting.py) | `DirectTileRenderer`, visible-surface payload and kernel/tile scheduling. The API retains `selection='hybrid'`. |
| [modules/spectral_heat_field.py](modules/spectral_heat_field.py) | Current learned source state and corner coefficient preparation. |
| [knn_heat/triangle_kernel_support.py](knn_heat/triangle_kernel_support.py), [packed_support.py](knn_heat/packed_support.py) | Triangle candidates, ordered support, fused selection and safe fallback. |
| [modules/spectral_heat_coefficients.py](modules/spectral_heat_coefficients.py), [Triton coefficients](triton/spectral_heat_coefficients.py) | GPU spectral coefficients and their backward operator. |
| [triton/heat_kernel_splatting.py](triton/heat_kernel_splatting.py) | Group tile work, interpolate heat and emit weighted pixel contributions. |
| [rendering/fast_raster.py](rendering/fast_raster.py), [mesh_visibility_raster.py](rendering/mesh_visibility_raster.py) | Numba CPU visibility: winning triangle IDs and barycentric coordinates. |
| [rendering/camera_raster.py](rendering/camera_raster.py) | Shared camera basis, clipping and retained reference rasterization utilities. |
| [rendering/triangle_tile_layout.py](rendering/triangle_tile_layout.py) | Fixed visible triangle/tile patches and bounded record layout used during geometry preparation. |
| [triton/splat_contributions.py](triton/splat_contributions.py) | Shared top-30 selection; generic emission and reconstruction for correctness comparisons. |
| [triton/splat_reconstruction.py](triton/splat_reconstruction.py) | Main HKS weighted RGB reconstruction and selected contribution backward kernels with runtime pair bounds. |
| [rendering/splat_autograd.py](rendering/splat_autograd.py), [triton/heat_kernel_backward.py](triton/heat_kernel_backward.py) | Shared custom backward for contributions, filters and colors. |
| [rendering/resident.py](rendering/resident.py), [prepared_geometry.py](rendering/prepared_geometry.py), [gpu_dispatch.py](rendering/gpu_dispatch.py) | Shared fixed geometry, coefficient topology and internal correctness reference. |
| [renderer_comparison.py](../splatting_script/benchmarks/renderer_comparison.py) | Matched training, timing, memory, quality and correctness gates. |
| [pipeline_workflow.py](../splatting_script/benchmarks/pipeline_workflow.py) | Shared initialization, comparison jobs and compact reports. |

`resident.py` is a required shared implementation module, not a public alternate
training backend. Original parameter optimizers remain in `hktex/`.

## Scope and limitations

- The comparison is fixed-density, unlit RGB fitting on the same mesh with
  held-out camera views. It is distinct from original HKTex's full physically
  based inverse-rendering experiments.
- Mesh/camera/visibility gradients and higher derivatives are outside the
  supported first-order kernel training path described here.
- GPU memory reports cover PyTorch allocations, not all FAISS/driver allocations.
  Short smoke runs establish execution, not convergence or a reliable speedup.
- No new performance result is implied by documentation or cleanup. Historical
  measurements belong to their recorded workload and source version.
- Standalone `splat`/`resident` commands, projected-footprint pruning and renamed
  module aliases are retired. Canonical model/optimizer state dictionaries remain
  supported; old whole-object HKS pickles may require their original source.
  Snapshots are archival; the workflow has no resume CLI.

For historical changes and source recovery, see the
[pipeline cleanup report](../docs/HKS_PIPELINE_ONLY_CLEANUP.md) and
[unused-code cleanup report](../docs/HKS_UNUSED_CODE_CLEANUP.md).
Original HKTex attribution and citation remain in the [main README](../README.md#citation);
see [LICENSE](../LICENSE) for licensing.
