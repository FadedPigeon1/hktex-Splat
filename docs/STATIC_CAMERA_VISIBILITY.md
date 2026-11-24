# Reuse fixed camera visibility while learning fresh heat kernels

`--supervision camera --visibility reuse` rasterizes a fixed view on its first
visit, then reuses its triangle IDs/barycentrics. This is valid only for fixed mesh
geometry and unchanged cameras. The earlier fresh mode unnecessarily paid the
CPU raster cost every update for the four recorded, static training views.
`fresh` remains the default/reference so existing commands keep their semantics.

## What changes

`CameraSupervision.visibility_buffers` owns at most four immutable raster frames,
within `--visibility-cache-mib` (default 64 MiB, host raster arrays). Camera
definitions are compared per visit; changed definitions invalidate the view and
cause another rasterization. Mesh changes are rejected before using the existing
BVH; rebuild the dataset to change geometry safely. CPU arrays are copied on
delivery so DataLoader/caller writes cannot corrupt retained visibility. If one
frame exceeds the budget, execution fails rather than silently truncating it.

HKS also keeps at most four prepared `TileSurface`/renderer objects, keyed by
camera, model/mesh/eigensystem identity, backend, policy and buffer settings.
Incoming visibility is checked against the snapshot; existing tensor/version
guards remain active. This avoids rebuilding immutable GPU geometry/tile layout.
The GPU cache is bounded by four entries; the host visibility budget is not a
claim about total GPU memory. Existing per-forward support/buffer guards remain.

Every forward still runs source preparation, triangle support, corner coefficient
evaluation, contribution queue construction, splatting and backward as applicable.
Kernel movement and parameter replacement during densification do not reuse old
source values or coefficient tables. Target colors still use the selected CPU/GPU
sampler every frame. Full-trace diagnostics and cropped validation inputs bypass
renderer reuse so an old full-view layout cannot leak into a subset comparison.
Original HKTex implementation and original optimizer/density defaults are unchanged.

## Timing and accounting

First-use rasterization, GPU geometry construction and final evaluation/saving
remain inside complete process/worker wall time. The existing training interval
still starts after first-batch fetch and initial gates: do not use it alone as a
start-to-finish measurement. Native and HKS use the same visibility policy.
The held-out dataset has its own first-use raster, included in worker time.

Reports record raster calls, visibility hits, invalidations, host resident bytes,
and HKS GPU geometry builds/hits. For 601 updates cycling four unchanged training
views, expect four training-dataset raster calls and 597 visibility hits, instead
of 601 raster calls. These are expected operation counts, not measured speed gains.
RNG/model/evaluation receipts remain strict; use fresh initialization/output when
changing visibility policy. Fixed asset reuse is separate from learned-state reuse.

## A100 smoke

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --supervision camera --resolution 64 --visibility reuse --target-sampling gpu \
  --backend both --kernels 1000 --iterations 601 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct --selector-merge row-radix \
  --density-policy original --optimizer-policy original --allow-render-variation \
  --no-training-plots --verify-samples 1024 --execute \
  --output "experiments/raster_reuse_smoke_$(date +%Y%m%d_%H%M%S)"
```

This retains the explicit diagnostic allowance for the unresolved eager-reference
RGB mismatch; target-lookup and other safety gates remain strict. It is not a
validated-equivalence/speedup claim. Compare unprofiled full process/training
times, quality, density transitions and geometry counters. Use `--visibility fresh`
as the control. No GPU training/benchmark jobs were launched automatically here.

New documentation is this note; modified source is `training/camera_supervision.py`,
`training/surface_adapter.py`, `training/uv_training.py` and
`splatting_script/experiments/train_surface.py`. Source/index recovery copies and
temporary checks are in `/tmp/hks-static-visibility-92_4rbsz`. No research data,
checkpoints or results were removed; no new dependency, commit or push was added.

## Completed validation

Real CPU BVH/Numba raster tests produced exact fresh/reused samples, colors,
triangle IDs and barycentrics; camera edits rerasterized with a new key. DataLoader
writes did not corrupt retained buffers. Budget rejection, fixed-mesh mutation
rejection, four-view LRU and resident-byte accounting passed. Cropped validation
inputs discard full-view reuse keys.

The actual adapter body was checked with isolated renderer boundaries: reuse did
not freeze outputs, stale keys were rejected, cache ownership stayed bounded and
full-trace diagnostics bypassed reuse. A separate real renderer CPU-interpreter
check changed source embeddings, grew from 12 to 13 sources and replaced appearance
parameters: reused/fresh renderers matched RGB, selected IDs, weights and six
splat/coefficient-input VJPs using synthetic coefficients. It does not replace
full-model A100 gradient/density validation.

Syntax, fresh/reuse CLI forwarding, report labels, original HKTex/Triton source
preservation and Git staging checks passed. No GPU jobs or performance measurements
ran here; the next A100 smoke must validate the integrated reuse path. Whole-tree
`git diff --check` still reports pre-existing whitespace in original
`hktex/modules/tracer.py`, lines 285 and 290; that file was left unchanged.
