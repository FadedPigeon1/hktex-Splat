# Tiled GPU kernel selection

Date: November 18, 2026 (America/Chicago).

## Scope

`--support-selector fused` retains the previous row selector at up to 4,096
sources. Wider source tables now use a bounded, all-Triton selection path.
The default 5,000-source workflow no longer opts into legacy fallback simply
because of source count. This is a selection implementation change, not a
change to spectral heat equations or the original pointwise HKTex baseline.

## Algorithm

For each existing batch of visible destination triangles:

1. Read at most 1,024 sources at a time. Reuse `support_distances`, including
   its existing float64 CUDA reduction tree and explicit rounded square root.
2. `merge_support_tile` combines that tile with a rolling nearest-source list.
   Keep a power-of-two capacity sufficient for both the neighbor rank and the
   triangle support cap. A lexicographic bitonic network compares full float64
   distance keys and source IDs separately; it does not quantize distances.
3. Repeat until every source has been examined. Retaining the nearest L from
   each prefix preserves the global nearest L under the same total ordering.
4. `finish_support_tiles` recovers the global kth distance and applies the
   existing centroid/radius bound, guard, support cap and ascending source-ID
   output ordering.
5. Feed the existing GPU count/scan/CSR packing. Its large-face-table guard
   remains, but total source count no longer forces the legacy metadata path.

For the usual neighbor rank 50 and cap 128, the rolling list has 128 entries.
The merge program handles at most 2,048 lanes; its width does not grow with total
source count. Float64 distance scratch is at most `face_chunk * 1024 * 8` bytes,
plus `face_chunk * L * 12` bytes for rolling keys and IDs. Candidate output rows
still scale with visible triangles and the support cap. No dense triangle-by-all-
sources distance table is allocated by the wide CUDA path.

The loop schedules multiple GPU launches. It does not invoke native Torch
cdist/topk for supported CUDA inputs and does not download candidate IDs. The
existing small allocation-metadata read remains. CPU controls retain native CPU
cdist, whose reduction order differs from CUDA's.

## Compatibility and limits

- Wide selection uses **distance, then source ID** for exact ties. This matches
  a stable total order; it may change selected kernels at tied cutoffs compared
  with the former PyTorch large-count fallback. Such changes can affect RGB and
  gradients at those ties. This is explicitly recorded in selector diagnostics.
- The existing path through 4,096 sources retains its previous policy. The
  original pointwise HKTex baseline and `--support-selector legacy` are unchanged.
- `StableSupportReference` is an independent Torch full-sort correctness oracle
  for the wide path. The validation harness compares ordered support, allocation
  metadata, RGB, gradients and optimizer state against that control. It does not
  silently weaken the previous support-equality checks.
- Current wide-path neighbor rank and triangle cap are bounded by 512. Normal
  rank 50/cap 128 are unchanged. Unsupported ranks, dtypes, dimensions, pinned
  Torch revisions and index spans retain explicit fallback diagnostics and checks.
- Source/face/record indexing and rendering memory budgets still apply. There
  is no promise of unlimited kernel count, constant runtime or unlimited support
  per triangle. Every source is examined afresh when the learned field changes.
- Corner coefficients, interpolation, tile emission, top-30 RGB reconstruction,
  backward operators, kernel movement and optimizer equations are unchanged.

## Benchmark and profiling

Preflight/quality selector records include `selection_algorithm` and
`selection_tie_policy`. Normal wide CUDA runs should report `tiled_merge`,
`distance_then_source_id`, and null selector/native-operation fallback reasons.
Profiles expose `support.tiled_selection` and `support.tiled_threshold_output`
alongside distance, scans and packing. Source provenance changes; rerun matched
comparisons rather than mixing old and new source hashes.

The workflow no longer passes `--allow-selector-fallback` automatically above
4,096 kernels. A fallback caused by another unsupported input remains a gate
failure unless explicitly acknowledged. `--allow-native-variation` concerns the
original baseline's repeatability; it does not authorize selector fallbacks.

## Verification

| Check | Result |
| --- | --- |
| Small-source compatibility | Completed before/after interpreter cases at 127, 1,000 and 4,095 sources matched the saved implementation exactly. The 4,096 case did not complete within the interpreter timeouts; its unchanged kernel bodies were checked separately. |
| Production wide-path orchestration | 4,097, 5,000, 8,193, 10,001 and 65,537 sources matched the independent stable full-sort oracle without source-width fallback. |
| Selection edge cases | Passed exact ties, selection of the final source in a partial tile, separate triangle batches and multiple rows per batch, strided inputs, float32/float64 sources, cap below/above neighbor rank, non-power-of-two capacities, rank/cap 512 and nonfinite distance ordering. |
| Complete CSR contract | Passed source IDs, face IDs, counts, offsets, maximum, task/fragment counts and support validity against the independent oracle. |
| Empty/unsupported input | Empty visibility remained valid; the unsupported neighbor-rank guard remained explicit. |
| CUDA compilation | Merge/finalization kernels compiled to A100 `sm_80` for capacities 128 and 512 without a GPU. Merge shared-memory metadata was 16,384 bytes. |
| Source preservation | 176 Python modules parsed; original HKTex and author scripts were byte-identical; all 21 pre-existing decorated Triton definitions were byte-identical. |
| API/command checks | Renderer, profiler and report imports passed; inherited backward stayed unchanged; eight native/HKS/workflow previews passed with no automatic large-count fallback acknowledgement. |
| Compact reporting | Actual selector algorithm, tie policy and fallback reasons survive in the single Markdown report; historical records still load and incomplete data does not produce validated speedup ratios. |

Numerical cases used small synthetic geometry (four embedding dimensions) and
**Triton's CPU interpreter**, not CUDA execution. Offline compilation establishes
compilability, not runtime CUDA correctness, real-scene equivalence or speed.
The implementation agent had no A100 allocation and ran no GPU jobs. A subsequent
user-run A100 correctness gate at 5,000 sources and 32×32 **passed**; its saved
`validation.json` was inspected on November 18, 2026. Both selector controls
passed, as did the direct renderer's image, gradient, parameter, optimizer-moment,
support-set, triangle-coefficient, spectral-response and positive top-30 checks.
The local artifact is under
`experiments/converse_heat_kernel_splatting/results/direct_tiles_direct_20261119T022539_785698Z_d2313bf146/validation.json`
(the run directory uses UTC). It is a local ignored result, not a bundled fixture.
No completed 256×256 full-run result or performance improvement is claimed.

Before a full training comparison, run the independent GPU gate on a small image
inside an existing A100 allocation:

```bash
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --kernels 5000 --resolution 32 --iterations 1 \
  --support-selector fused --validate-only --execute
```

That command retains existing rendering, coefficient and gradient checks. A
failure remains a failure; neither approximate support nor native replay variation
is permission to disable those checks. Then run a fresh 5,000-source, four-update
workflow at the intended resolution before increasing the update budget. Existing
native-repeatability diagnostics remain separate from HKS selector validation.

## Files and recovery

Implementation changes are in `hktex_splatting/knn_heat/triangle_kernel_support.py`
and `hktex_splatting/triton/resident_gpu.py`. Benchmark entry, workflow, profile,
trace, capacity-plan and report readers were updated under
`splatting_script/benchmarks/`. Current README, architecture, command and benchmark
guides describe the new policy; historical results/recovery records are untouched.

No test suite was added to the repository, and no mesh, archive, model or result
was deleted. The starting Git index and existing worktree edits were preserved;
nothing was staged, committed, stashed, reset or pushed. Before-images and local
verification scripts/receipts are under `/tmp/hks-tiled-selector-vfau_uw3/`;
that directory is temporary storage, not a permanent recovery archive.

The wide tie rule is an intentional behavior change at exact tied cutoffs.
No performance improvement or GPU correctness is claimed by this implementation.
