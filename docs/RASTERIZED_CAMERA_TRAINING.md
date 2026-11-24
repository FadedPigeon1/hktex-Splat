# Full adaptive training on rasterized camera pixels

For fresh GPU rasterization on every camera frame, see the opt-in
[GPU camera pipeline](GPU_CAMERA_PIPELINE.md). It retains device visibility for
HKS and includes conservative support-pruning and resolution-scaling experiments.
GPU execution and performance validation for this new path are still pending.
`--visibility fresh` remains the default. The older
[`--visibility reuse`](STATIC_CAMERA_VISIBILITY.md) option is a separate fixed-view
experiment, not the fresh-camera protocol.

Optional [GPU target-texture sampling](GPU_TARGET_SAMPLING.md) is now available
with `--target-sampling gpu` for both methods. CPU remains the default/reference;
CPU triangle rasterization remains the default. The optional nvdiffrast backend
also moves UV interpolation to the GPU with a bounded exact RGB check. Target
checks stay strict.

The intended camera pipeline is now available in the full-loop worker:
`train_surface --supervision camera`. The historical command name is retained;
the explicit supervision mode, report title and manifests identify this workload.
The earlier surface runs did not rasterize camera triangles and cannot establish
camera-rendering speed. The separate old camera benchmark uses fixed density;
it is not this full adaptive training integration.

Each update freshly rasterizes the fixed mesh for one of four recorded training
views. Visible triangle IDs, depth and barycentrics are computed by the existing
CPU BVH/Numba rasterizer. The target is unlit texture RGB sampled at those pixels.
Native HKTex evaluates its pointwise field at the same visible pixels; HKS uses
the actual 2D visibility and screen tiles, selecting support, evaluating corners,
interpolating heat and reconstructing RGB. No sample-grid packing or surface
importance pool is used in camera mode. Triangle IDs/barycentrics are supervision
metadata, not trainable geometry. The mesh and eigensystem stay fixed.

The original `BaseTrainer.optimise` loop, loss, configured schedulers, geodesic
movement and adaptive density callbacks execute. Original policies remain the
default for both arms; HKS custom policies can be selected independently. Original
HKTex source is unchanged, but camera supervision changes the experiment from its
original UV-sampling recipe. This is an image-space training comparison.

Camera evaluation uses a full `heldout_left` frame. It is a different quality
metric from previous held-out surface samples and the old multi-view camera
benchmark. RGB/VJP gates retain the first requested visible pixels at their actual
screen positions; the eager reference still uses the independent sample adapter.
Camera definitions, mode, resolution, sources and initialization receipts are
recorded and compared. Both arms cycle the same training views from the same seed.

Complete process/worker wall times include setup, first rasterized frame, gates,
all subsequent rasterization/target lookup, updates, final evaluation and saving.
The training interval retains its existing definition: it starts after initial
gates and the first batch fetch. Dataset counters separately record raster and
target-preparation host wall totals. Startup/raster counters are not CUDA times.
No material shading or relighting is implemented by this mode.

## Validate before timing

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --supervision camera --resolution 64 \
  --backend both --kernels 5000 --iterations 1 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct --selector-merge row-radix \
  --density-policy original --optimizer-policy original \
  --no-training-plots --verify-samples 1024 --verify-only --execute \
  --output "experiments/raster_camera_verify_$(date +%Y%m%d_%H%M%S)"
```

After that passes, use fresh output, `--kernels 1000 --iterations 601`, and remove
`--verify-only` to exercise actual optimizer/density transitions. Inspect measured
raster costs and memory limits before attempting 5000 updates at a chosen
resolution. The rasterizer and target texture lookup are CPU operations and may
dominate complete time; a fast spectral kernel alone does not remove their cost.
Fresh target lookup currently uses the original Trimesh conversion, without a
new target-pixel cache. Allocation guards fail explicitly rather than truncating
visible pixels or support. No expensive GPU runs are launched automatically.

Changed files: `training/camera_supervision.py` (new), `training/uv_training.py`,
`training/surface_adapter.py`, `splatting_script/experiments/train_surface.py` and
this note. The surface-mode default remains available for reproducing earlier
experiments. Recovery copies are in `/tmp/hks-camera-training-lrfbgnh9`.

CPU validation passed on a real two-triangle mesh: BVH/Numba rasterization,
fresh frame storage, foreground ID/order, barycentrics, constant texture targets,
DataLoader's NumPy-to-tensor conversion and screen-preserving validation subsets.
An isolated execution of the actual adapter body verified that real 2D visibility
is passed through and RGB/contribution outputs keep visible-pixel order. That
adapter check substitutes renderer boundaries; it is not a GPU rendering check.
Surface/camera CLI previews, report labels and Python syntax passed. Original
HKTex sources and Git staging were checked unchanged. No GPU execution or full
training was performed here. GPU RGB/VJP checks and density-transition smoke
training remain required. Historical result JSON retains its existing
`heldout_surface_quality` key for compatibility; camera reports label its actual
camera evaluation scope. Whole-tree diff checking still finds only pre-existing
whitespace in original `hktex/modules/tracer.py`, which was left unchanged.

## First GPU smoke: unresolved eager-reference disagreement

`raster_camera_smoke_20261121_000024` completed native verification but stopped
HKS before updates. All parameters were finite; GPU camera/sample paths agreed
on selected IDs/weights/RGB, and candidate rows matched. The eager reference RGB
disagreed by up to 0.019477, and the mean-color VJP failed. This is not a passing
camera training validation. No tolerance or check was relaxed.

Failed HKS gates now write `render_gate_failure.pt` with bounded gate inputs,
actual/reference RGB, ordered support, corner coefficients, interpolation responses
and selected contribution data. `sample_validation.json.failure_stages` reports
coefficient and selected-ID discrepancies and the first affected pixels. Extra
tracing executes only after a failed gate. It does not change normal rendering.
Syntax, all trace-mode dispatch combinations, original HKTex and staging checks
passed. A fresh GPU `--verify-only` retry is needed to collect this evidence and
identify the numerical/selection disagreement; this change does not claim to fix
its cause. Source recovery copies are in `/tmp/hks-camera-gate-cmtow6qu`.

The diagnostic retry `raster_camera_diagnostic_20261121_001441` instead stopped
at strict initialization matching before the renderer gate. RNG, sampler and
evaluation matched, but independently initialized kernel placement, residual
colors and mean color differed. Camera mode's explicit `--shared-initial-model`
now shares these four fields together from native initialization, before any
gradients or optimizer state. All other model fields must match exactly; no mesh
or spectral data is copied. The original surface-mode placement-only behavior is
unchanged. Policy receipts/reports label this `shared-camera-initial-model` and
retain the rejected local proposal. This is not a relaxation of the RGB/VJP gate.

A CPU check loaded the actual failed snapshots: camera sharing made the entire
model exact while unrelated fields/RNG/evaluation stayed unchanged. Surface mode
still rejected the same color mismatch; unrelated angle drift and populated
optimizer state were rejected. CLI/report policy checks passed. Recovery/check
files: `/tmp/hks-camera-initial-pw6eguse`. A fresh camera verification must now
reach the rendering gate and, if necessary, save the still-needed failure trace.

## Optional diagnostic continuation

At the researcher's request, `--allow-render-variation` is available only with
camera supervision and the direct adapter. Defaults remain strict. It does not
fix or establish the cause of the eager-reference disagreement, and is distinct
from the historical native optimizer replay allowance.

Diagnostic continuation requires finite RGB/VJPs, max absolute RGB error <=0.02,
all parameter VJPs within atol 1e-4 / rtol 3e-4, exact GPU camera/sample selected
IDs, original strict weight/RGB agreement between those GPU layouts, and exact
candidate rows. Initialization, assets, mesh, optimizer bindings and ordinary
training health checks stay strict. Missing safety evidence rejects continuation.
The 0.02 RGB bound is an explicit debugging allowance, not a numerical-equivalence
or acceptable image-quality guarantee; it may conceal a renderer bug.

The original `passed:false` remains in the validation report; separate
`diagnostic_accepted`, bounds and failure snapshots record the allowance. Workflow
reports are prominently DIAGNOSTIC / UNVALIDATED; worker statuses are prefixed
DIAGNOSTIC_, and results record strict-gate status and whether variation was
accepted. No validated speedup or equivalence claim is justified by these runs.
CPU policy tests checked the recorded failure against bounds, rejection of
nonfinite/over-bound/missing safety evidence, strict-default behavior, CLI
forwarding and report labels. GPU execution remains pending for this option.
