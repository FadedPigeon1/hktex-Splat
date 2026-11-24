> Source retirement (2026-11-23): the uncalled occupancy router and certified
> cache implementation are now removed from live HKS. Exact preimages are kept
> in `archive/hks_results_cleanup_20261123/preimages.tar.gz`. The historical
> descriptions and results below remain; use the current run guide instead.

> Historical experiment: occupancy routing and certified candidate caching are
> superseded by the kernel-centric path in `hktex_splatting/README.md`.
> Current training comparisons reject these old flags. Preserve this record to
> interpret prior results; it is not the current run guide.

# Sparse association and occupancy ablations

These opt-in experiments implement the first architectural steps in the supplied
triangle–kernel association proposal. Original HKTex and default HKS behavior
remain available. No GPU was requested or used during implementation; no speedup
is claimed. Both experiments keep fresh camera rasterization.

## Occupancy-aware evaluator

`--occupancy-threshold N` counts current visible pixels per face on the GPU.
Faces with fewer than N pixels use original pointwise HKTex; other faces use
triangle HKS. All pixels retain their original order and selected contributions
are merged for the existing density callback interface. Low-occupancy faces are
removed from HKS visibility before support/corner work. Threshold zero disables
routing. Threshold 4 is an ablation setting, not a measured crossover or a
permanent recommendation. Test thresholds independently at matched quality.

**This changes the evaluator:** native pointwise heat and triangle-interpolated
HKS need not produce identical values. Results are explicitly labeled as mixed
evaluation. The reference gate uses the same partition, with native pointwise
evaluation and the existing eager triangle reference. This establishes branch
consistency, not equivalence to all-triangle HKS. Original density is currently
required; custom support-cost density is not enabled for mixed contributions.

Implementation: `training/occupancy_evaluation.py`, called from
`training/surface_adapter.py`. Counts/bins are reported in `occupancy_stats`.
Only diagnostic occupancy summaries are retained per view; routing always uses
the current raster's counts. It does not retain visibility images or predictions.
Profiles expose `occupancy_partition`, `occupancy_pointwise` and
`occupancy_triangle`. Gradient gate reports record `branches_exercised`: a small
validation subset may contain only low-occupancy faces, which does not validate
the triangle branch. Increase validation coverage when testing mixed evaluation.

## Certified persistent candidate index

`--association-cache certified` retains a bounded, face-major candidate list
with source anchors and a distance certificate. Heat, colors, anisotropy,
orientation and corner coefficients are recomputed from live parameters.

The present support rule is spectral KNN plus triangle radius/guard and a cap;
it is not a physical fixed-radius kernel support. Consequently this version
uses spectral bounds rather than inventing a Euclidean footprint or adopting
a locality radius from another kernel formulation.

For each visible face, retain its nearest L=256 source IDs at an anchor step,
the old distance to source L+1, and the old Mth distance for
M=max(neighbors, cap). Let D bound the embedding displacement of **every**
source, including omitted ones. Reuse is permitted only when

```
old_excluded_distance - old_Mth_distance > 2*D + numerical_margin
```

holds for every face. The numerical margin is deliberately loose for the
audited <=256-dimensional float64 distance arithmetic. A valid gap guarantees
that current nearest-M sources are inside the stored list. Exact current
indexed distances, Kth/cap cutoffs, radius/guard filtering and ascending source
ID packing then rebuild fresh support CSR from that superset. On failure, the
existing full selector runs and refreshes the index. No contribution is dropped
solely for being small, and no gradient/heat/color is reused.

Every candidate generation's **first successful reuse is independently audited
against fresh selection**; mismatch raises an error. Parameter/face-buffer
replacement, source count, geometry/version, visible-face set, neighbor/cap or
guard changes invalidate reuse. Thus pruning/densification cannot silently
reindex old source IDs. Source motion is bounded globally, so a formerly distant
source moving into the query neighborhood also invalidates the certificate.

Limits: nvdiffrast camera mode, existing audited 4097–8192-source row selection, no prefix pruning,
neighbors/cap <=256, at most four retained camera-key entries with a total
retained budget up to min(128 MiB, the configured working-buffer limit).
Temporary sorting/indexed-distance workspaces are additional. Unsupported
counts/policies or budget failures keep the original selector/fallbacks. Cached
candidate validity still performs GPU reductions/scalar checks; this is not a
zero-synchronization implementation. A cache miss adds capture/top-k overhead.

Implementation: `knn_heat/persistent_association.py`, the support driver and
`rendering/heat_kernel_splatting.py`. Stats include hits, misses, audits,
invalidation reason and candidate distance pairs. Profile scopes distinguish
lookup, capture and audit costs. The cache is disabled during initial receipts;
its first-hit audit validates actual reuse during subsequent training.

## Preserved work lists and remaining research

Existing GPU face/tile counts, scans, stable incidence ordering, queue emission,
corner evaluation and backward operators are reused. This is not a new 3DGS
alpha-compositing renderer. No proxy mesh, physical-radius spatial hash/BVH,
per-source incremental update queue or persistent fused worker is included.
Those require additional conservative bounds or measured justification. Cache
fallback is currently whole-face-set, not incremental dirty-source rebuilding.
There is no automatically calibrated cost LUT yet; occupancy threshold sweeps
must establish a crossover before enabling that behavior by default.

## Validation and manual ablations

CPU interpreter tests passed cached versus fresh ordered support, cache hits
and first-hit audit, ordinary source motion, an omitted source moving into the
query, parameter generation changes, byte-budget fallback and occupancy split.
Both row-radix and row-topk passed 64-dimensional, top-128 cases, including
moving sources, embedding-version and unsupported-source-count invalidation.
Routing/merge tests with isolated renderer boundaries checked complete pixel
partition, original pixel order, contribution IDs, exclusion of low faces from
the HKS geometry and parameter gradients, including all-pointwise/all-triangle
cases with an empty opposite branch. Public renderer imports, four CLI previews
(baseline/occupancy/cache/combined), incompatible-option guards, and syntax of
all 77 HKS/script Python modules passed. The indexed distance kernel compiled
offline for SM80. Original HKTex itself cannot execute in this CPU-only session;
the isolated boundary tests are not full CUDA correctness/performance results.

Use original density/optimizer policies in both arms first. Preview is the
default and creates no rendering job:

```bash
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --supervision camera --resolution 64 \
  --kernels 5000 --iterations 40 --seed 0 \
  --raster-backend nvdiffrast --target-sampling gpu \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct \
  --selector-merge row-topk --corner-evaluation parallel \
  --support-pruning none --density-policy original --optimizer-policy original \
  --association-cache certified --occupancy-threshold 4 \
  --profile-start 8 --profile-steps 8 --no-training-plots --verify-samples 1024 \
  --output experiments/association_ablation
```

For manual GPU execution, add `--execute` in your existing allocation, using a
fresh output directory. Test baseline (no cache/threshold zero), occupancy only,
cache only and combined separately. Inspect cache hit/miss/audit costs and
occupancy distributions; a cache with no hits may be slower. Strict gates remain
enabled. Repeat final candidates unprofiled at full training budget and matched
quality. A faster mixed-evaluation branch is not proof of faster pure HKS.

The cited paper claims in the pasted proposal were not used as numerical
guarantees. This implementation's certificate is derived from the current
repository's spectral KNN support rule.
