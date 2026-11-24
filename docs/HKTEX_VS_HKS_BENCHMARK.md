# HKTex (Original) versus HKTex Splat (Ours)

## Simple output (current workflow default)

### Explicit native-variation diagnostic mode

Add `--allow-native-variation` to a workflow to collect raw statistics while
reporting native repeatability failures as **diagnostic/unvalidated**. This does
not change either renderer, optimizer, equations or recorded strict tolerances.
The native receipt's `passed` stays false when its strict checks fail.

The exception is narrowly scoped: start state/settings must match, loss must be
within the existing replay loss tolerance, gradients and optimizer moments must
pass their existing tensor tolerances, and all values must be finite. Only the
native path may continue when the stricter reference loss or post-Adam parameter
repeatability test fails. Hybrid replay remains strict. The paired gate still
requires matching mesh/targets/cameras/source/initialization/device, but records
finite common-native-evaluator differences rather than requiring identical RGB.
Native observer-parity profiling is skipped in this diagnostic mode; hybrid's
observer gate remains enabled. Raw timing/memory/PSNR remain available, but
validated paired speedup, memory-reduction and PSNR-delta claims are withheld.
The standalone validated comparison reader rejects diagnostic runs.

This option is separate from approximation: triangle-supported HKS can differ
from pointwise HKTex, but the observed failed replay compared native with itself.

`compare_training workflow --execute --output comparison.md` keeps just one
Markdown report, containing performance/quality statistics, job/gate status,
initial/source hashes when available and failure details. The report is refreshed
after each job. Internal initialization, checkpoints, images, logs, raw traces,
JSON/CSV working records and per-run compiler caches are created under a private
temporary directory and removed on normal exit or handled interruption. An
abrupt process kill can leave OS temporary files; it does not intentionally
create new per-renderer result folders in the repository in this mode.

Use `--keep-artifacts --output NEW_DIRECTORY` for the full archival workflow
described below, including rebuilding reports and plots later. Existing results
and explicitly supplied initialization files are never cleaned up. Without
retained artifacts, the one-page statistics and hashes cannot replace a full
training checkpoint, raw profile, image-quality history or exact-state archive.
The same renderers, updates, measurements and correctness gates execute in both
output modes; this change does not resolve the native replay failure.

The existing `splatting_script.benchmarks.compare_training workflow` compares
exactly two implementations. It does not implement a replacement baseline or
change rendering, selection, gradients or optimizers. This is a controlled,
fixed-density, unlit texture-fitting benchmark on the prepared Converse mesh,
not the full lit Mitsuba inverse-rendering experiment.

## Actual execution paths

| Label | Internal identifier | Actual implementation |
|---|---|---|
| HKTex (Original) | `native` | `renderer_comparison.py:render` calls the original model's `prepare_kernels`, `prepare_points`, `diffuse_heat_kernels` and RGB forward; original FAISS flat-L2 pointwise outer-50 queries, query chunks of 256, filtering and top-30 reconstruction. |
| HKTex Splat (Ours) | `hybrid` (harness mode `direct`) | `DirectTileRenderer` in `hktex_splatting/rendering/heat_kernel_splatting.py`: triangle spectral centroid/radius candidates, kernel–triangle support, shared three-corner spectral heat, barycentric interpolation, Triton tile scheduling/emission and filtered top-30 weighted RGB reconstruction. |

The original methods are in
[`HeatKernelTextureKNN`](../hktex/modules/heat_kernel_texture_knn.py), which calls
[`EigenAlboInterpolationKNN.build_kernel_graph`](../hktex/modules/eigen_albo_knn.py)
and inherits RGB reconstruction from
[`HeatKernelTexture.forward`](../hktex/modules/heat_kernel_texture.py).
Its class defaults are not the benchmark recipe: the shared
[`converse_training.yaml`](../splatting_script/configs/converse_training.yaml)
selects outer-50/inner-30. HKS dispatch starts in
[`heat_kernel_splatting.py`](../hktex_splatting/rendering/heat_kernel_splatting.py),
using [`triangle_kernel_support.py`](../hktex_splatting/knn_heat/triangle_kernel_support.py),
[`spectral_heat_field.py`](../hktex_splatting/modules/spectral_heat_field.py) and
the unchanged [`Triton splatting kernels`](../hktex_splatting/triton/heat_kernel_splatting.py).

Both paths use the actual original optimizer parsing, Adam and geodesic kernel
movement, schedulers and post-step parameter enforcement. The mesh and
precomputed Laplace–Beltrami eigensystem remain fixed. Source positions,
orientation, anisotropy, weighting and colors remain trainable. These are heat
kernels, with HKTex weighted color reconstruction, not Gaussian alpha compositing.
`original`/`2dgs` are compatibility aliases for the same two paths; `2dgs` does not
mean conventional 2D Gaussian splatting.

## Matched controls and correctness gates

One initialization process saves a seed-labelled `initial.pt` per kernel count.
Both arms load that same file. New manifests retain the historical `backend`
values `original`/`2dgs` and add `backend_id` and `implementation_label`; historical
result files are never rewritten.

Matched controls include mesh/spectral/camera asset hashes, config and actual
source hashes, initial state hash and initialization-file identity, kernel count,
seed, resolution, all cameras, optimizer/model settings, trainable parameter
shapes, precision/software and physical GPU UUID. Actual face IDs, barycentric
coordinates, foreground masks and target RGB arrays are fingerprinted outside
measurement. Target topology and mesh vertex alignment are checked at setup.

Each workload executes these stages in order:

1. Generate or load the shared initialization.
2. Run one separate `--preflight-only` process per renderer at the requested
   kernel count and resolution. Render four training and two held-out cameras;
   check finite RGB, valid foreground samples and the actual loss. Execute the
   training-step loss check against a separately rendered gradient-enabled
   reference under the same saved state/RNG. Evaluation renders remain under
   `no_grad()`; their differences from the training reference are recorded in
   `preflight_diagnostics.json`, rather than conflated with this consistency check.
   Execute the
   real backward/optimizer step, reset parameters, gradients, optimizers,
   schedulers and RNG, and replay the first update. Check finite gradients,
   parameter movement and optimizer tensors, and repeatability with the existing
   `atol=3e-5`, `rtol=3e-4` comparison tolerances. Continue through the other three
   training cameras with finite parameter/gradient/loss checks.
3. Require both preflight receipts and a paired gate: identical saved initial
   parameters/file, assets/settings/source/GPU, exact raster/target/mask receipts,
   and agreement of the common pointwise evaluator at initialization. Record
   deployed-renderer RGB and loss differences without demanding equality.
4. Run capacity pilots (four updates after eight warmup updates). Failed pilots
   block their dependent speed/profile jobs. Failed initialization or preflight
   blocks measurements; errors, logs and skipped jobs remain in `workflow.json`.
5. Run uninstrumented research training and separate diagnostic profiles.
   Research warmup is followed by exact model restoration, fresh optimizers and
   schedulers, and RNG restoration; optimizer state and model hash are checked.
   Both arms start measured updates at the same camera.

The scalable preflight checks finite VJPs and reset/replay behavior; it is not
an independent mathematical proof of the backward implementation. Existing
`--validate-only` still independently compares support ordering, coefficients,
RGB, all parameter VJPs and optimizer moments against internal triangle and
full-spectrum controls. Its dense full-spectrum oracle may exceed configured
capacity for large workloads; run it separately at a feasible diagnostic size.
`--profile-validate` gates the observer-enabled same-renderer RGB/VJPs using
the actual foreground training loss. It also checks an independent unprofiled
repeat, retains the existing tolerances and saves named diagnostics on failure.
The helper's legacy RGB-squared-sum default remains for compatibility; see
[the smoke-run gate investigation](PROFILE_VALIDATION_FIX.md).
Neither gate is removed or weakened. Triangle cap-128 and native outer-50
candidate policies differ, so cross-renderer output equality is not required.

Above 4,096 sources, the workflow uses the bounded tiled Triton selector and no
longer passes `--allow-selector-fallback` automatically. Its explicit tie rule is
distance followed by source ID; this can differ from historical PyTorch fallback
results at exact ties. Preflights and quality records include the actual selector
algorithm, tie policy and fallback diagnostics. Support and rendering memory
budgets remain enforced. See [selector scope and validation](TILED_SELECTOR.md).

## Reproduce the two workloads

Run from the repository root with the installed environment's Python. Omit
`--execute` to preview commands without importing Torch or touching the GPU.
Execution requires an NVIDIA CUDA GPU; Slurm is optional. On clusters, obtain
resources through your scheduler. The recorded validation hardware is A100; run
the correctness gate on other supported devices. Use a new output directory.

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 1000 --resolutions 128 --iterations 1000 --seed 0
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 5000 --resolutions 256 --iterations 1000 --seed 0

# In an existing allocation only; use a different NEW_DIR for each invocation.
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 1000 --resolutions 128 --iterations 1000 --seed 0 \
  --execute --output /path/to/NEW_DIR

# Short matched smoke workflow before full experiments (also dry by default).
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 1000 --resolutions 128 --iterations 4 \
  --pilot-iterations 4 --profile-start 0 --profile-steps 4
```

The 5,000/256/1,000 workload is also the default. Custom controls are
`--kernel-counts`, `--resolutions`, `--iterations`, `--seed`, `--initial-state`,
`--visibility`, `--matmul-precision`, buffer budgets and profile windows.
Multiple counts/resolutions produce their Cartesian product. The default
single workload plans ten jobs; `--scaling` plans 111. Actual feasibility is
decided by existing runtime capacity/OOM guards; a supported configuration is
not a guarantee that a particular memory budget will fit.

## Measurements and report artifacts

The archival workflow (`--keep-artifacts`) automatically generates a detailed
report; rebuild those retained records offline with:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_training report \
  /path/to/workflow.json --output /path/to/NEW_REPORT_DIR
# Add --no-plots to avoid requiring Matplotlib.
```

`report.md`, `overall.csv`, `pairs.csv` and `benchmark_report.json` contain:

| Measurement | Definition |
|---|---|
| Training iterations | Completed updates; paired ratios require equal requested/completed budgets and final quality steps. |
| Total measured training time | Sum of synchronized forward, loss, backward, optimizer, scheduler and post-step times. Excludes setup, warmup, quality evaluation, checks, progress/checkpoint I/O. Any JIT misses during measured updates remain included. |
| Complete process wall time | Parent's subprocess timer through child exit, including setup, warmup, evaluation, snapshots and process overhead. Distinct from the child's shorter `completion.json` wall. |
| Median / P95 iteration time | Distribution of synchronized update times; P95 uses nearest rank. |
| Peak GPU memory | PyTorch allocated and reserved allocator high-water marks for measured training; evaluation peaks are excluded. Does not include all FAISS, driver or other library allocations. |
| Final training loss | Last update's pre-optimizer, single-camera loss: channel-summed squared RGB error averaged over foreground pixels. It is three times per-channel foreground MSE, not an all-camera post-training loss. |
| Training / held-out PSNR | Both common-native-field and actual deployed-renderer PSNR, described below. |
| Training speedup | Original total measured update seconds / Ours seconds, only for complete, healthy, matched pairs. |
| GPU memory reduction | `100 * (Original allocated peak - Ours allocated peak) / Original allocated peak`. Negative means Ours used more PyTorch allocator memory. |
| Quality differences | Ours minus Original training and held-out PSNR, separately for each evaluator family. |

Four training cameras are `top`, `side`, `three_quarter`, `logo_closeup`; held-out
cameras are `heldout_left`, `heldout_right`. Both splits use the same mesh texture
sampled with the same nearest lookup, encoded RGB/255 targets, foreground masks
and predictions without evaluation clipping. Camera MSEs have equal weight;
PSNR is `-10 log10(mean camera MSE)` with the existing `1e-30` floor, not mean dB.
Evaluation includes initialization, every 250 updates by default, and the final
update. Held-out cameras are unseen views of the same object, not new objects.

Historical `training_psnr_db`/`heldout_psnr_db` remain the **common native-field**
metrics: each learned parameter state is evaluated with actual pointwise HKTex.
New `deployed_*` fields evaluate each renderer's actual selection/interpolation
policy. Neither family is silently substituted for the other. Quality CSVs
include a `family` field. The report plots both families and side-by-side total
training time and memory; detailed profiling CSVs/plots are retained.

Reports expose missing/differing controls and withhold paired ratios/deltas
when required controls, health, budget or the new paired gate fail. Historical
workflows remain readable and explicitly say `not_recorded_historical` for the
new preflight protocol; historical measurements are not recertified. Full source,
asset and configuration provenance is retained in manifests and report JSON.
Profiles are separate diagnostics and never enter training speed ratios. Host,
CUDA event and kernel-duration sums can overlap and must not be added together.

## Current validation status

The implementation update is checked with GPU-free command previews, syntax and
imports, archived CPU benchmark checks executed only from temporary storage,
new preflight/failure/report checks and before/after CPU optimizer-trajectory
comparisons. Exact results are recorded in
[verification.json](hktex_vs_hks_benchmark/verification.json).
No A100 allocation was available during that initial update: no real-renderer GPU smoke or
full training benchmark was executed, and no new performance improvement is
claimed. First run the short matched workflow in an existing allocation, review
the receipts and deeper correctness gates, then launch the full workloads.

A subsequent user-run A100 smoke in `hktex_vs_hks_smoke` completed both
four-update speed runs and the paired preflight. Both diagnostic profiles failed
the old RGB-squared-sum gradient gate. Those original artifacts remain untouched.
At that stage the profiler-gate change had CPU verification only and its GPU
outcome was pending a fresh matched rerun. See [the investigation](PROFILE_VALIDATION_FIX.md).

The user's subsequent `hktex_vs_hks_smoke_v2` run on an A100-PCIE-40GB completed
all ten jobs, including both profiler RGB/VJP gates and their repeat checks.
The 5,000-kernel/256px run then passed hybrid preflight with the required
`source_width_exceeds_4096` fallback, but native preflight rejected the mixed
inference/training loss comparison before any measured training. The correction
uses a gradient-enabled reference and retains the loss/replay tolerances;
its 5,000-kernel GPU outcome is pending. See
[the preflight loss investigation](PREFLIGHT_LOSS_MODE_FIX.md).

The initial update changed the six benchmark/entry modules listed in the request
and current guides. The subsequent gate fix also updates `pipeline_profile.py`.
No standalone test infrastructure is restored; no renderer,
original HKTex, GPU kernel, optimizer or selector source is changed. Existing
research data, results and cleanup/recovery records remain untouched.
