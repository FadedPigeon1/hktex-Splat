# Compact exact merge for the tiled HKS selector

`--selector-merge compact --surface-adapter aggressive` is an opt-in optimization
of the wide selector, used above 4096 sources when existing audit guards allow
tiled selection. The default remains `--selector-merge reference`. The original
full-sort kernel remains intact as a correctness reference and rollback path.

## Change

Previously each 1024-source tile was concatenated with a retained candidate list
and padded to 2048 slots, then fully bitonic-sorted. The new kernel sorts only the
1024 new distances, retains their best KEEP entries, and bitonic-merges those
with the already sorted KEEP old entries. The first tile directly initializes
the list, removing sentinel-fill writes and their GPU launches. Private scratch
lists are allocated once per selection call and reused across sequential face
batches, after each batch's final rows are emitted on the same stream. No shared
scratch survives into another forward or autograd graph. KEEP is unchanged: the next power
of two covering the existing neighbor rank and support cap, at most 512 on the
audited path. Every source is still evaluated.

For the usual KEEP=128, later tiles use a 1024-slot sort followed by a 256-slot
merge instead of a 2048-slot full sort. This reduces sorting work, not spectral
precision or candidate coverage. Sorting each tile down to KEEP is exact: a
discarded entry already has KEEP better entries in that same tile and cannot
enter the global best KEEP. Both operations compare the same float64 distance
bits followed by ascending source IDs. Padding, NaNs, cap/threshold evaluation,
final ascending-ID support order, guards and legacy fallbacks are retained.

The distance kernel and all heat, coefficient, backward, reconstruction and
optimizer calculations are unchanged. Distance fusion is deliberately deferred:
combining 1024 sources with 64+ dimensions in one sorting program may increase
register pressure and must preserve the audited float64 reduction order.

## Files

- `hktex_splatting/triton/resident_gpu.py`: new `merge_support_tile_compact`; existing kernels unchanged.
- `hktex_splatting/knn_heat/triangle_kernel_support.py`: selectable merge and nested diagnostic timings for distances, merging, scans and packing.
- `hktex_splatting/rendering/heat_kernel_splatting.py`: forwards the merge policy to the fused selector.
- `hktex_splatting/training/surface_adapter.py`: propagates the explicit policy and records it.
- `hktex_splatting/training/uv_training.py`: compares new candidate rows exactly against the original full-sort selector.
- `splatting_script/experiments/train_surface.py`: CLI, controls, worker forwarding and report labeling.

## GPU validation

Implementation checks passed bit-for-bit interpreter comparisons of retained
distance keys, source IDs and final support rows after every source tile, plus
an independent NumPy lexicographic oracle. Cases covered 17 through 5000 sources,
KEEP values 1/8/32/128/512, cross-tile ties, NaNs, infinities and partial tails.
The real selector driver also matched across multiple face chunks above 4096
sources with scratch reuse. Both KEEP=128 and KEEP=512 kernels compiled offline
for CUDA SM80/A100. This is compilation evidence, not GPU execution evidence.
Original GPU kernel bodies, original HKTex and optimization/adjoint sources and
Git staging were preserved. Checks used temporary fixtures outside the repository.

Run a fresh verification on an allocated GPU before training. Existing results
are preserved. The larger source count exercises the new wide merge:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 5000 --iterations 1001 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter aggressive --selector-merge compact \
  --verify-samples 1024 --verify-only --execute \
  --output "experiments/compact_merge_verify_$(date +%Y%m%d_%H%M%S)"
```

Inspect `aggressive_support`: the actual algorithm should be
`tiled_compact_merge`, the reference `tiled_merge`, candidate rows exactly equal,
and no fallback. RGB, selected contribution and all parameter-VJP gates must
also pass. A passing gate with a fallback does not exercise the new kernel.

After correctness, use a fresh 1001-update run starting from 1000 kernels, with
`--profile-start 900 --profile-steps 20`. Nested `selector::` rows now separate
distance evaluation, tile merging, threshold emission, scans and CSR packing.
Their synchronized inclusive times overlap parent rows and include diagnostic
overhead. Do not compare the old aggregate 21 ms directly with nested-profile
totals. Compare full-sort and compact policies with profiling/trace capture off
for speed claims, on the same GPU with repeated runs and quality measurements.

No speedup is established by implementing this kernel. GPU execution remains
required even if interpreter checks and offline compilation succeed.
