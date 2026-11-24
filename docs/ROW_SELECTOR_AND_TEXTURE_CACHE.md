# Whole-row selection: focus on HKS computation

This pass targets repeated tile merges during kernel candidate selection. The
proposed target-texture cache has been withdrawn at the researcher's request.
Faster data preparation would not demonstrate faster heat-kernel splatting.
Default/reference implementations remain available. No training speedup is claimed.

## Measured workload and priority

The recorded [5000-update surface run](../experiments/full_training_radix_20261120_200047/comparison.md)
took 135.1469 seconds for native HKTex and 215.7067 seconds for HKS in the
training interval. This HKS implementation was about 1.60 times slower in that
run. The new whole-row selector has not yet been measured on a GPU.

In the earlier [5000-kernel profile](../experiments/aggressive_profile_20261120_182046/comparison.md),
steps 900-919 averaged 21.0765 ms in triangle support, 4.2383 ms in corner
coefficients and 0.6081 ms in splat forward. These are synchronized inclusive
wall times; nested stages overlap and must not be summed. Random surface
batches averaged only about 1.05 samples per triangle, so three-corner work
has little reuse there. Camera images and random surface batches are different
workloads; a camera speedup would not establish faster full surface training.

The next comparison uses the original sampling path for both arms, with the
selector change isolated. RGB, gradient, support-order and initialization gates
stay enabled. A complete relit camera pipeline requires a separate measurement.

## Withdrawn target-texture cache: historical investigation

The prepared Converse asset contains an 8192x8192 RGB texture. The installed
`trimesh.visual.color.uv_to_color` converts the entire image to RGBA and exposes
it as a NumPy array on every call. The original million-point pool draws batches
of 4096: approximately 245 lookups, each reconverting a 256-MiB image.

An optional RGBA-pixel cache was implemented and checked, then removed before
GPU training validation. `CachedSurfaceSampler` has been restored to its earlier
face-probability-only implementation. The module and its old `--cache-surface-sampling` flag have since been removed.
Old cached-sampler results require their recorded source version; current commands
use the original sampler. Native HKTex remains unchanged.

An isolated CPU check on the actual decoded texture measured three 4096-point
lookups (seconds):

| Scope | Measurements |
|---|---|
| Original lookup | 0.35873, 0.35794, 0.35572 |
| Cached lookup | 0.000469, 0.000246, 0.000248 |
| One-time cache construction | 0.35760 |

Colors matched exactly. These are agent-environment CPU texture-lookup timings,
not GPU measurements, whole-training timings or a claimed end-to-end speedup.
JPEG decoding was warmed before measurement. The withdrawn cache used
268435456 CPU bytes. These historical measurements do not describe the current
sampler or justify a splatting-performance claim. Kernel positions, anisotropy,
orientation, heat, contribution weights and RGB continue to refresh as usual.

Changing scene lights or cameras does not require changing that fixed surface
texture. Original HKTex's `HeatKernelsRendererKNN.mesh_to_mitsuba` supplies the
learned texture as a principled material's base color; Mitsuba handles shading.
The current HKS camera benchmark instead reconstructs unlit RGB, and the surface
training workflow compares surface colors. Neither measures a complete relit
HKS pipeline. Relightable HKS needs its surface-color evaluation connected to a
shading stage; camera visibility and screen-tile layouts must refresh for a new
view. This optimization pass does not add that integration or cache shaded output.

## Wide candidate selection: cutoffs instead of tile merging

Opt-in `--selector-merge row-radix` retains the original small-row policy through
4096 sources. For 4097-8192 sources it computes each visible triangle's complete
distance row using the original audited float64 distance kernel, then finds only
the neighbor-rank and support-cap cutoffs with exact byte-radix selection. A
stable scan emits admitted IDs in ascending order. No per-source-tile candidate
lists or cross-tile merges are needed in this bounded case.

The radix loop is GPU-resident and bounded to eight bytes. Identical keys and
identical current prefix bytes skip work exactly; no approximate termination or
statistical truncation is used. A GPU loop avoids code replication for all bytes.
CUDA float32 source embeddings at dimension 64 now use the measured four-face,
16-source distance layout; see the [layout study](SPECTRAL_DISTANCE_STUDY.md).
Other <=64-dimensional row cases retain 32 source pairs per program; wider
embeddings retain 8. Explicit reduction arithmetic and RN square root are unchanged.

The row-distance scratch budget is at most 32 MiB and no larger than the
configured buffer budget. Face batches shrink to fit. Source counts above 8192,
or budgets unable to hold a single row, retain the existing scalable tiled radix
path; this is not a new kernel-count limit. The Torch revision/dtype/neighbor
rank/index-span audit fallbacks are preserved. The report records the actual
algorithm and `row_radix_fallback`, so a fallback cannot masquerade as a tested
whole-row path. Learned support and heat still refresh every forward.

## Startup accounting

Existing result JSON and comparison Markdown now separate environment/config,
mesh/data preparation, sampler-cache construction, model/eigensystem setup and
provenance/final GPU waits. They also record original-loop setup before the first
state receipt, and the initial state/RGB/VJP gate time. These are host wall scopes,
not CUDA kernel times; GPU work may finish in later synchronization scopes.
Target-texture cache reporting was removed along with that cache.

## Changed files

- `hktex_splatting/training/cached_sampling.py`: restored the earlier face-probability-only implementation; no pixel cache.
- `hktex_splatting/triton/resident_gpu.py`: new exact radix-rank helper and row selector; old kernels retained.
- `hktex_splatting/knn_heat/triangle_kernel_support.py`: bounded row driver and tiled fallbacks.
- `hktex_splatting/rendering/heat_kernel_splatting.py`: forwards scratch budget.
- `hktex_splatting/training/uv_training.py`: gate scratch settings/fallback receipts and initial-gate timing.
- `splatting_script/experiments/train_surface.py`: opt-in selector mode and startup reporting; historical sampling flag retained.

## Validation

CPU interpreter checks matched an independent lexicographic reference through
8192 sources, including cap/rank order variations, ties, NaNs, infinities and
all-equal rows. Real row/tiled driver outputs matched with forced sub-batching;
small-source dispatch and budget/count fallbacks were checked. Distance results
matched bit-for-bit for 8/32 pairs on dimensions 2/64/256, strided inputs and tails.

Before withdrawal, original/cached UV batches, importance pools, feedback indices
and RNG matched exactly using original sampler source bodies. Texture lookup
also matched Trimesh. These checks cover the withdrawn implementation, not a
new retained optimization. The earlier sampler was restored byte-for-byte.

The final row selector and 32-pair distance variants compiled offline for SM80.
This is not CUDA execution evidence. Original HKTex, spectral equations, backward
operators, original GPU kernel bodies and Git staging were preserved. No testing
infrastructure was added to the repository and no GPU jobs were launched.
The environment could not communicate with an NVIDIA driver, so GPU execution
remains pending. Whole-tree `git diff --check` reports pre-existing trailing
whitespace in `hktex/modules/tracer.py` (lines 285 and 290); that original HKTex
file was left unchanged by this pass.

## Run

First run correctness gates with fresh output and initialization:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 5000 --iterations 1001 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter aggressive --selector-merge row-radix \
  --no-training-plots --verify-samples 1024 \
  --verify-only --execute \
  --output "experiments/row_selector_verify_$(date +%Y%m%d_%H%M%S)"
```

Require actual algorithm `row_radix`, no row/native fallback, exact candidate rows,
passing RGB/VJP checks and identical model/RNG/pool/evaluation receipts. Next use
the same options for a fresh 5000-update run starting from 1000 kernels, without
profiling or trace capture. Compare complete worker and training intervals, and
inspect startup scopes instead of attributing all differences to rendering.
Repeated runs with matched quality are still needed for a speed claim.
