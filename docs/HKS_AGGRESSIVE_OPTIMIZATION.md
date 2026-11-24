# Aggressive HKS preparation and dispatch optimization

An additional opt-in [compact tiled merge](COMPACT_SELECTOR_MERGE.md) is available
with `--selector-merge compact`. It retains the original full-sort kernel as the
default reference and adds finer selector stage timings.

This is an opt-in implementation pass, not a measured speedup. Use
`--surface-adapter aggressive`; `reference` remains the default and `optimized`
remains available for comparisons. Original HKTex source and assets are unchanged.

## Implemented changes

| File | Change |
|---|---|
| `hktex_splatting/knn_heat/packed_support.py` | Cache fixed float64 triangle spectral centers/radii once; gather visible bounds in one operation instead of rebuilding per chunk. |
| `hktex_splatting/knn_heat/triangle_kernel_support.py` | Dispatch existing CUDA selector kernels in up to 512-face batches; retain reference 128-face chunks for native-policy audits, CPU ties and fallback. Guard new execution views against mutation. |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | Replace the Python loop over visible samples with stable vectorized tile-slot construction; pass optional cached bounds/dispatch settings. |
| `hktex_splatting/training/surface_adapter.py` | Own the bounded fixed-scene cache; upload the inverse sample permutation once per forward rather than once per returned tensor; avoid grouping/masking for inactive original density callbacks while retaining input validation. |
| `hktex_splatting/training/uv_training.py` | Verify all candidate rows against the original uncached/unbatched selector, in addition to existing RGB, VJP, selected-ID and weight checks. |
| `splatting_script/experiments/train_surface.py` | Expose the explicit mode and retain cache/dispatch settings in result metadata. |

The triangle-bound cache has a maximum allocation of 64 MiB, also limited by the
configured working-buffer budget. If it cannot fit, the previous bound-building
path is used and the budget fallback is recorded. It retains no learned kernel
values. Mesh/embedding identity, tensor versions and cached-bound mutations are
checked. Dynamic support, heat coefficients, topology and autograd buffers remain
fresh for each forward and after density changes.

Larger CUDA execution batches reduce host launches and scratch allocations across
face chunks. Each row still uses the same distance, threshold, cap and tie-order
operations. Source tiles, neighbor ranks, the 4096-source dispatch boundary,
arbitrary-count tiled selection and safety fallbacks are unchanged. Execution
batching is disabled when the configured working-buffer budget is below 32 MiB.
CPU selection retains the original chunks and native cutoff-tie repair.

No Triton kernel, heat/Gram contraction, custom adjoint, filtering equation,
top-30 reconstruction, optimizer, pruning or densification code was rewritten.
The vectorized tile layout preserves exactly the old per-tile ordinal order,
including holes and partial edge tiles. Sample order is restored before loss and
density accumulation, as in the preceding optimized adapter.

Inactive-controller shortcuts are enabled only for the two exact original
controller classes, using their existing accumulation start and stop conditions.
Subclasses retain normal delegation. Input shape and invalid-weight checks still
run even when accumulation is inactive. Active accumulation and post-backward
callbacks, including pruning/splitting, are unchanged.

## Validation and interpretation

CPU checks passed exact cached/reference centers, radii and candidate rows on
random geometry, ties, empty visibility and chunk tails. Mutation,
trainable-input and memory-budget guards were checked. Vectorized tile slots and
lengths matched the serial implementation, including holes and partial tiles.
Actual eager renderer and surface-forward checks matched RGB, selected IDs,
weights and synthetic-coefficient VJPs; original controller method bodies matched
accumulators through start/stop schedules with the inactive shortcut enabled.
CLI previews, syntax, documentation links and unchanged original/GPU source and
Git-index checks passed. Fixtures were temporary, outside the repository.

CPU selector checks retain the CPU-native 128-face policy; they do not validate
512-face CUDA execution. No usable NVIDIA driver was available in the
implementation session, so no CUDA jobs or real-spectrum GPU gates were run.
All candidate rows must pass the new GPU gate before performance measurements.

The prior four-step profile crossed a density change. Its large source/selection
times may include compilation; they are not steady-state kernel timings. This
pass targets avoidable repeated geometry work and launches found in source. It
does not establish that these are the largest steady-state costs.

Run correctness gates on an allocated GPU first. A 1024-sample gate covers more
than one execution batch, and both kernel counts cover small and tiled selectors:

```bash
HK_PY=${HK_PY:-python}
for KERNELS in 1000 5000; do
  "$HK_PY" -B -m splatting_script.experiments.train_surface \
    --backend both --kernels "$KERNELS" --iterations 1001 --seed 0 \
    --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
    --shared-initial-model --surface-adapter aggressive --verify-samples 1024 \
    --verify-only --execute \
    --output "experiments/surface_aggressive_verify_${KERNELS}_$(date +%Y%m%d_%H%M%S)"
done
```

Inspect `sample_validation.json`, including `aggressive_support`, and its recorded
actual selection algorithm/fallback. Exact candidate IDs must match; no gate is
relaxed. The comparison shares the existing fused kernels with the reference;
independent selector validation remains necessary for kernel-algorithm changes.

After gates pass, run 1001 updates to exercise density changes and profile a
stable count (`--profile-start 900 --profile-steps 20`) in a fresh directory.
Use `--verify-samples 1024` for the gates in that workflow as well. Disable stage
profiling and geodesic diagnostics for full 5000-update speed runs. Compare
reference, optimized and aggressive HKS on the same GPU with repeated seeds;
include quality and complete training intervals. Existing research results are
not overwritten, and initial snapshots from older source revisions are not
silently reused.

## Deliberately preserved costs

Source spectral gathering still calls the original differentiable operators.
Rewriting it or changing compilation behavior needs steady-state evidence and
separate GPU VJP validation. No shared temporary buffer is reused across live
autograd graphs. No support caps, spectral dimensions, precision, safety checks,
sampling distributions, density schedules or boundary policies are weakened to
obtain faster numbers. Learned support/heat caching across updates is unsafe
because kernels move and their parameters change.
