# Direct GPU surface-sample adapter

`--surface-adapter direct` is an opt-in HKS training adapter. Original HKTex and
the default/reference adapter are unchanged. The goal is to remove sample-packing
overhead, not to change supervision or claim an unmeasured speedup.

## Data flow

Original face IDs and barycentrics → stable GPU face grouping → flat 64-sample
work tiles → existing triangle support and corner spectral coefficients →
existing Triton splatting/top-30 reconstruction → original sample order.

The new `hktex_splatting/training/sample_tile_geometry.py` builds surface and
face/tile dispatch metadata directly with Torch tensor operations. There is no
square sample grid, NumPy barycentric packing, background padding in the output,
or `PreparedGeometry`/`ResidentDispatchGeometry` construction for this path.
Only the last work tile contains masked padding slots. `(P,1)` is an output view,
not a camera image. GPU sorting and consecutive unique-key operations can still
have allocation/synchronization costs; this is not a CUDA Graph implementation.

The audited selector still needs host face IDs for visibility and geometry
checks. Normal training retains these from the original CPU sampler, rather
than downloading the moved batch again. Diagnostic calls with GPU-only inputs
may copy their IDs to the host. This is not a fully GPU-resident training loop.

Every forward creates fresh sample geometry and learned support. The direct
mode inherits the aggressive mode's fixed spectral-bound optimization, selector
options, shared density contribution grouping and inactive-controller checks.
It does not enable the optional sampling-probability cache or cache target pixels.

## Preserved behavior

- Original samples, targets, feedback indices and sample order at the loss.
- Original optimizer, scheduler and density-controller implementations.
- Spectral equations, support ordering, corner interpolation and kernel motion.
- Existing Triton forward, top-30 weighted RGB reconstruction and custom backward.
- Allocation bounds, fixed-geometry mutation checks and selector fallbacks.
- Initial-state, RGB/VJP, selected-ID/weight and candidate-row gates. Direct mode
  runs these against the existing ungrouped adapter; it does not validate itself.

New files are this note and `training/sample_tile_geometry.py`. Modified files:
`training/surface_adapter.py`, `training/uv_training.py`,
`rendering/heat_kernel_splatting.py`, `splatting_script/experiments/train_surface.py`
and `docs/FULL_SURFACE_TRAINING.md`. No source files or results were deleted.

## GPU verification and timing

Use an allocated CUDA GPU with the project environment active:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 5000 --iterations 1001 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct --selector-merge row-radix \
  --no-training-plots --verify-samples 1024 --verify-only --execute \
  --output "experiments/direct_adapter_verify_$(date +%Y%m%d_%H%M%S)"
```

After both gates pass, use fresh output with `--kernels 1000 --iterations 5000`
and remove `--verify-only` to measure the original adaptive training lifecycle.
For an isolated adapter comparison, also run aggressive mode with exactly the
same selector, seeds, update budget, original sampling and plot policy. Use fresh
initialization/receipts for each workflow, as source/control checks remain strict.
Do not compare an old cached-sampler run against the new uncached run and
attribute the whole difference to this adapter. Compare training and total worker
wall times, quality and final kernel counts; repeated runs are needed.

Random surface samples currently have little triangle reuse (about 1.05 samples
per triangle in the recorded diagnostic window). This adapter removes setup
overhead but does not remove the three-corner evaluation cost. It is not camera
rendering, material shading or a complete relit pipeline benchmark.

## Recovery and validation

Pre-change Python source copies and the unchanged Git index listing are in
`/tmp/hks-direct-adapter-tz24zyi3`. This temporary directory is local recovery
material, not a committed archive. Existing staged/unstaged work was preserved.
Checks and GPU availability are reported after implementation; offline/interpreter
execution does not establish GPU correctness or performance.

Completed checks:

- All HKS and script Python files parsed; the direct CLI preview forwards both
  workers with sampling caching disabled and starts no GPU work.
- CPU/interpreter batches of 1, 63, 64, 65 and 129 samples matched the reference
  for support CSR, sample order, query embeddings, selected entries, weights,
  RGB and six splat-input VJPs. Real selector/queue/emitter/reconstruction/backward
  code executed through Triton's CPU interpreter.
- The actual `DirectTileRenderer` constructor and compact contribution interface
  matched the old adapter for a 65-sample synthetic scene, including six VJPs.
  Its spectral coefficients were synthetic; this is not a full asset/eigensystem
  gradient check. The existing full-model GPU gates remain required.
- Mutation rejection, invalid IDs/barycentrics and the geometry budget were checked.
- Original HKTex Python files, all existing Triton kernels and the Git index
  match their pre-change copies. No tests were added to the repository.
- NVIDIA-SMI could not communicate with a driver; no GPU execution, training or
  performance measurement ran. `git diff --check` reports pre-existing trailing
  whitespace in `hktex/modules/tracer.py` at lines 285 and 290; it was not edited.

Temporary check scripts are alongside the source recovery copies. No performance
improvement or A100 correctness is claimed from these CPU checks.
