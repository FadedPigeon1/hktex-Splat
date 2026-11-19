# Triangle-supported kernel-driven spectral Heat-Kernel Splatting

> Historical validation record: standalone test/validation paths and commands
> below were retired on November 15, 2026. Recorded measurements and source
> identities are preserved as provenance; those commands require archived-source
> restoration. Use the current [benchmark diagnostics](../splatting_script/README.md)
> and [test cleanup/recovery report](TEST_CLEANUP_REPORT.md) for maintained commands.

> Current source paths use lowercase `splatting_script`. Historical run
> manifests, source hashes and cleanup recovery records retain their original
> identities; updating this guide's paths does not claim a new GPU validation.

## Completed fused candidate path: manual A100 validation required

`--support-selector legacy|fused` remains the only selector interface and
`legacy` remains default. This update replaces the previous partial fusion
inside the existing fused class. No renderer mathematics or legacy selector
operation is changed. GPU equivalence and speed remain unmeasured by the agent.

### Audited inputs, arithmetic and selection contract

The measured configuration uses 1,000 live sources in the fixed 64-dimensional
spectral embedding. Each immutable visible-face chunk has at most 128 float64
centroids `[M,64]` and radii `[M]`; sources `[1000,64]` are float32 or float64,
converted to float64 before distance arithmetic. Source locations/anisotropic
spectra, visible destinations and hidden-source eligibility retain their
existing meaning. All source IDs are evaluated for every visible destination.

The CUDA policy is pinned to installed PyTorch revision
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`. The inspected primary sources are
[CUDA distance](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/DistanceKernel.cu),
[CUDA top-k](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/TensorTopK.cu)
and the installed `ATen/native/cuda/block_reduce.cuh` header.

The distance is the square root of summed squared differences in that spectral
embedding. For dimensions up to 256, each native CUDA lane contributes one
product, followed by two warp reductions. The replacement explicitly reproduces
their addition order and uses float64 round-to-nearest CUDA square root. It
does not use a generic reassociated sum, matrix multiply or float32 arithmetic.
FMA contraction is disabled. CPU cdist has a different reduction order, so CPU
selector controls explicitly retain CPU-native cdist instead of weakening a
comparison tolerance. The separate distance-program tests use an independent
scalar CUDA-tree oracle on CPU and compare finite bits to native cdist on GPU.
In the 510-value CPU fixture corpus, the CUDA-tree program matched its oracle
exactly, while native CPU cdist differed in 229 values by up to six ULP, including
the expanded subnormal-product cases. This
is why CPU-native cdist remains an explicit control fallback. No CUDA tolerance
is widened to hide a reduction-order difference, and no CPU result is treated
as proof of CUDA cdist equivalence.

The threshold is still `kth + 2*radius + guard*(1+kth+radius)`, with
`kth` the `min(neighbors,N)`th distance and default guard `1e-5`. Equality is
admitted by `<=`. Pool cap remains a limit on kernels **per destination face**.
The audited single-block CUDA top-k chooses ascending source IDs at a cutoff
tie; sorting its selected IDs afterwards leaves that set unchanged. CPU top-k
has different tied identities and receives a CPU-only native tie repair.
Neither policy is substituted silently for the other.

### Existing fused implementation after this update

| Stage | Current implementation |
| --- | --- |
| Distances | One tiled `support_distances` launch per chunk, eight source/face pairs per program, direct strided loads and float64 conversion |
| Threshold/cap | One `select_support_rows` launch per chunk. One uint64 ordering of nonnegative distance values supplies both order statistics; NaNs order last |
| Filtering/ties/output | In that same row kernel, apply the unchanged bound, reproduce masked-infinity cutoff ties by source-ID prefix counts, then emit count and ascending IDs |
| Global counts/offsets | Three integer kernels: local scan/statistics, block-total scan/metadata reduction, and global offset addition |
| Compact CSR | Existing `pack_support` program, unchanged downstream representation and consumers |
| Allocation sizes | Existing single 32-byte host read, followed by unchanged overflow/budget checks before compact allocations |

Filtering is monotone in distance. The raw cap order statistic is reused when
admitted; otherwise its masked cutoff becomes infinity. Prefix tie counts
include rejected entries that native masking also changes to infinity, and
only then discard rejected IDs. This matters for infinite bounds and NaNs.
Selected destinations and padding occupy disjoint row slots, avoiding a race
between padding stores and selected-ID stores. Integer scans preserve exact
counts/offsets/tasks/fragments; no floating-point reduction is introduced there.

There is no global masked-distance matrix, boolean selection matrix, native
kthvalue/topk output, or ID-sort temporary on the audited CUDA path. A float64
score workspace of at most `[128,1000]` (1,024,000 bytes) remains. It is allocated
once per forward and reused only between that forward's sequential chunks.
Each row consumes all scores to select globally, and splitting distance work
from row sorting avoids a single program holding the entire `[N,D]` product.
This two-stage resource tradeoff is unmeasured on A100; no claim is made that a
giant kernel would be faster or that this tiling is optimal.

Source counts, caps, neighbors, strides, chunk offsets, face counts and block
counts are non-specialized runtime values. Block capacities, immutable embedding
dimension and guard are compilation settings. The earlier changing-queue-count
JIT fix is untouched.

### Exact fallbacks and allocation synchronization

Full legacy row construction is used for uncapped/cap-covering cases, more
than 4,096 sources, embedding dimensions outside 1–256, unsupported input
dtypes, unaudited CUDA/PyTorch revisions and shapes selecting native multiblock
top-k. Extremely wide/sparse strides or buffers exceeding int32 relative index
spans also retain native indexing. The native dispatch predicate is checked using the actual chunk lengths,
including the last partial chunk. CPU keeps native cdist and repairs only rows
with a cap-spanning tie using CPU-native topk; native kthvalue is replaced there
too. These CPU controls exercise row selection while preserving CPU-reference
semantics. Reports expose `selector_fallback`, `native_distance_fallback` and
`native_topk_fallback`; an audited fast CUDA run requires all three to be null
and nonzero `fused_chunks`. Extremely large face tables retain legacy metadata
construction rather than launching an unbounded scan program.

The 32 bytes are four int64 values: exact retained pairs, largest face support,
expanded face/tile tasks and pixel fragments. Resident coefficient/dispatch
construction and hybrid queue/record allocation consume these Python sizes.
They change with live parameters. Reusing their previous values would be stale;
moving the read to setup would make later allocations incorrect. A persistent
mutable CSR/record workspace would also need ownership across outstanding
autograd graphs. Fixed-capacity output would require every coefficient,
dispatch, sorting, reduction and backward consumer to mask padding and track
live counts. That is a broader representation change with additional work and
safety obligations, not a removable read under the current exact CSR contract.

The read is therefore retained. Its preceding scatter, casts, cumsum/cat,
maximum, weighted sums and metadata stack are replaced by three integer
kernels. Each forward still owns its counts, offsets, CSR, selected tape and
metadata. Only immutable face-to-visible-row geometry is cached and version
checked. No learned association, size, score or source state is cached.
Small scan temporaries remain per-forward allocations. Pooling them would
require exclusive leases and stream-lifetime handling; it was not added without
GPU evidence that this overhead warrants that ownership machinery. The existing
CUDA caching allocator can reuse physical allocations while Tensor ownership
stays separate.

Code inspection predicts **two candidate kernels per chunk plus three scan
kernels and compact packing** on the fast CUDA path. Native cdist/kthvalue/topk,
the separate filtering/row-packing kernels and the source-double conversion
launch are absent. This is a launch-structure prediction, not an observed GPU
count or speedup. Native fallback and diagnostic costs remain explicitly labeled.

### Validation, reporting and manual commands

The existing RGB, loss, VJP, finite-difference, Adam, visibility, hidden-source,
disconnected-surface, cap, overflow and changing-parameter tests remain.
Additional checks cover CUDA distance-tree finite bits in dimensions 1–256,
strided float32/float64 sources, local threshold/cap ties, infinite thresholds,
NaNs, chunk lengths 1/7/128/513/1024, native multiblock dispatch fallback, and
integer scan boundaries at 0/1/255/256/257/521 faces. No tolerances are loosened.
The retained two-outstanding-backward-graphs regression also runs on one shared
fused geometry, checking both old and new CSR/tape ownership and source VJPs.
CUDA precision, tie membership and compilation must still pass user-run tests.

Recorded CPU results: the fresh full regression process passed **83 tests in
1,192.690 seconds**. The subsequently added fused two-graph ownership regression
and the stride fallback rerun passed (two focused tests, 10.033 seconds).
The expanded subnormal-product distance corpus passed finite-bit comparisons
against the scalar CUDA-tree oracle. All seven edited Python files passed
static syntax checks. The current combined command discovers 84 distinct tests;
the added ownership test was verified separately from the 83-test full run.
No tolerance was weakened. No GPU test, CUDA benchmark or allocation was run.

Preservation checks found all 180 protected Python files unchanged, including
original HKTex, resident rendering/coefficient/backward code and the previous
JIT fix. The legacy support class and original resident GPU kernel definitions
are unchanged. All non-edited files retain size/mtime, and the repository path
inventory is identical: no new or deleted repository files/directories, reports,
checkpoints, scientific assets or recovery archives.

`candidate_activity.json` now separates distance, native threshold selection,
native top-k, masking/output, integer compaction and allocation metadata. On
the fast path, threshold/top-k/masking/output are **one joint GPU stage**;
their separate times are null, not fabricated estimates. Correlated GPU
activities, launch counts, active union, host selection wall, allocator-operator
wall and synchronization waits remain distinct. Normal timing adds no profile
events or extra device synchronization. Median/p95 and measured JIT misses
remain in the normal reports. Profile, validation and host-profile runs remain
diagnostic processes.

The verified checkpoint, correctness commands, three alternating four-case
benchmark rounds and separate late-profile commands below remain applicable
unchanged: `--support-selector fused` now selects this completed implementation.
Check null fallback fields and the recorded Torch git revision before labeling
a result fully fused. Compare correctness first, then matched median/p95 and
kernel counts. Remaining bottlenecks may include FP64 loads/reduction, row-local
uint64 sorting, the required allocation read, queue construction and spectral
coefficient/backward work. Only A100 measurements can rank them or establish
a meaningful end-to-end improvement.

Exactly nine existing files change in this completion:

- `hktex_splatting/knn_heat/resident_support.py`
- `hktex_splatting/triton/resident_gpu.py`
- `splatting_script/benchmarks/direct_tiles.py`
- `splatting_script/validation/checks/resident_checks.py`
- `splatting_script/validation/forward_splatting/test_resident.py`
- `splatting_script/validation/validate_resident_gpu.py`
- `splatting_script/validation/test_direct_validation_cpu.py`
- `docs/DIRECT_TILES.md`
- `splatting_script/README.md`

## Experimental exact support selector: legacy remains default

The following section records the preceding partial-fusion audit and protocol.
Its implementation description and CPU counts are historical; the current
implementation is described above. Its checkpoint and manual commands are kept.

The latest supplied A100 results are original 21.118 ms, resident 10.984 ms and
hybrid 11.038 ms/iteration (1,000 sources, 64x64, cap 128). The previous dynamic
queue-count JIT fix remains intact. The older JIT audit below is historical.
This selector change has **no agent-run GPU validation or speed measurement**.

`compare_backends --support-selector legacy|fused` selects candidate-row
construction for both resident and hybrid training. The default is `legacy`.
`LiveHeatField(..., support_selector='fused')` exposes the same opt-in to API
callers. The standalone validated resident trainer keeps its existing default;
no renderer backend is added, and original HKTex is unchanged.

### Implementation audit and exact contract

The legacy selector runs the following operations freshly every forward:

| Stage | Operations and storage |
| --- | --- |
| Fixed visibility | Sorted unique visible destination face IDs; float64 centroid/radius in the fixed spectral embedding, in chunks of 128 faces. All source kernels remain eligible, including hidden sources |
| Source conversion | One live float64 source tensor and an ID arange (the capped branch does not use that arange) |
| Scoring | Per chunk, direct float64 `torch.cdist(..., donot_use_mm_for_euclid_dist)` allocates `[faces_in_chunk,sources]` distances |
| Threshold ranking | `kthvalue(min(neighbors,sources))` returns the row's kth distance |
| Filtering | Exactly `distance <= kth + 2*radius + guard*(1+kth+radius)`, with guard `1e-5`; six arithmetic tensor operations plus comparison, then masked fill to infinity |
| Cap ranking | When `0<cap<sources`, native `topk(cap, largest=False)` on masked float64 distances. Cutoff ties retain PyTorch's same-device behavior; they are not replaced by an ID tie rule |
| Row construction | Gather selected validity, sum counts, replace invalid IDs by `sources`, sort integer IDs ascending, copy counts and IDs into `[visible_faces,cap+1]` int32 staging |
| Output | Scatter counts into all-face int64 counts, cumsum to offsets; maximum, face/tile task and pixel-fragment reductions; pack ascending IDs into compact face-major int32 CSR |
| Synchronization | Download four int64 allocation sizes (32 bytes, one existing CUDA allocation read). No support IDs are downloaded. Safety checks precede compact allocations |

Every arithmetic/comparison/mask/gather/reduction/sort/copy above is a potential
CUDA launch; cdist, kthvalue and topk can themselves launch multiple kernels.
The previously supplied approximately 238 launches/iteration is profile
evidence, not a portable count derived from the number of Python operations.
The new correlated trace report lists the measured counts by stage.

Read-only reanalysis of the existing resident trace
`direct_tiles_resident_20261106T050519_985213Z_817e14b694/trace.json` attributes
2,014 kernels and 68.602 ms of kernel activity to the complete selection/CSR
contract over eight iterations (251.75 launches and 8.575 ms/iteration).
This broader scope includes count/scan/output beyond the original candidate
scope. Its 82 native cdist calls consume 41.556 ms (5.194 ms/iteration), and
native kth/topk ranking consumes 15.719 ms (1.965 ms/iteration). There are eight
32-byte D2H copies and eight synchronization API calls. The trace also records
100,158,464 bytes of device-to-device copies in selection, including native
masked/row sorting paths. These are historical GPU activities parsed offline;
no GPU operation was executed to obtain these figures. New profile output
lists every correlated kernel type with counts and durations in each stage.

Downstream `DeviceSupport` requires all-face counts/offsets, ascending source
IDs for each face, entry face IDs, and exact maximum/tasks/fragments host sizes.
Coefficients and both dispatchers consume that CSR. The cap limits **kernels
per destination triangle**. A proposed `[num_kernels,support_cap]` triangle
table would impose a different limit and could truncate valid associations.
It is therefore not used. Existing face-stride staging followed by CSR packing
preserves consumers and avoids a transpose/new compaction contract.

### Fused operations and retained semantics

The opt-in subclass inherits legacy geometry checks, CSR packing, allocation
guards and metadata read unchanged. The original `ResidentSupportGeometry`
class, `PreparedSupport.candidate_rows`, and existing `resident_gpu` programs
remain intact. Two appended Triton programs replace only capped row staging:

1. `filter_support_scores` fuses six float64 bound arithmetic operations,
   comparison, mask inversion and masked fill into one launch per existing face chunk. One
   program handles one row, using at most 4,096 score lanes. No spectral or
   embedding-distance reduction is reimplemented.
2. `pack_support_rows` gathers original scores at the native topk indices,
   evaluates the identical bound, counts valid choices, sorts integer IDs and
   writes the count/ID staging row in one launch per chunk. It eliminates the
   full boolean candidate tensor and separate gather/sum/where/ID-sort/copy
   temporaries. Scores are rechecked rather than treating masked infinity as
   invalid, preserving the legacy behavior for infinities and NaNs too.

The native float64 cdist, kthvalue and masked topk keep exactly the same input
shape, chunk size and ordering. Their reduction order and cutoff tie policy
are retained rather than approximated. `enable_fp_fusion=False` and an explicit
float64 guard constant preserve the threshold's arithmetic sequence. Source
count, cap, output stride and chunk offset are non-specialized runtime inputs;
power-of-two block capacities and the fixed guard are compile-time settings.

Uncapped selection, caps covering all sources, and source counts above 4,096
retain legacy row construction. Reports explicitly label `fused_chunks` and
`selector_fallback`; no unsupported case changes semantics silently. The
4,096 bound limits resource use rather than candidate coverage. Selected IDs
and all mutable CSR/tape storage remain freshly owned by each forward.

Code inspection establishes two new launches per chunk replacing the nine
filtering operations and six row-construction operations, plus removal of the
unused capped-branch ID arange. Actual launch savings and occupancy must be
measured. The native reductions, compact CSR scan/packing and **existing
32-byte synchronization remain**. Removing that read without changing
downstream dynamic allocation is outside this exact row fusion. No new host
read, synchronization, dynamic upload or learned-support cache is introduced.

### Validation and measurement

Existing tests remain. Added CPU-interpreter and user-run GPU checks compare
entire padded rows, counts, offsets, ordered IDs, entry faces, maximum, tasks
and fragments at zero tolerance. Cases include production-width ties, partial
face chunks, below/at/above cap, empty views/sources, single sources, irregular
visibility, strided source tensors, updates and explicit legacy fallbacks.
Nextafter tests straddle float64 threshold boundaries for several guards and
caps, including masked infinities and NaNs. Resident and hybrid checks compare
RGB/loss, nine source/appearance VJPs and three Adam steps; CPU VJPs use zero
tolerance and CUDA atomics retain the existing `3e-5/3e-4` tolerances. Native
parameter finite differences keep the existing `1e-6` step and tolerances.
Hidden-source and disconnected-visible-region controls remain explicit.

The pre-change run passed 70 existing CPU tests. The complete extended run
passed **78 CPU tests in 497.448 seconds**, with CUDA hidden, CUDA initialization
forbidden and actual Triton programs interpreted on CPU. The expanded
infinite-threshold boundary rerun also passed (4.628 seconds). All 11 edited
Python files passed static syntax checks. These are CPU results, not CUDA
compiler/occupancy or A100 performance validation.

Preservation checks found no changes to 176 protected Python files, including
all original HKTex source and the prior dynamic task-count JIT fix. AST checks
confirmed the original resident support class and every existing `resident_gpu`
definition are unchanged; the new class/programs are appended. All non-edited
repository files retain their size/mtime and the complete path inventory is
unchanged. No files/directories were added or deleted; historical reports,
scientific inputs and recovery archives remain untouched.

Exactly these 13 existing files changed:

- `hktex_splatting/knn_heat/resident_support.py`
- `hktex_splatting/triton/resident_gpu.py`
- `hktex_splatting/modules/live_field.py`
- `hktex_splatting/rendering/direct_tiles.py`
- `splatting_script/benchmarks/direct_tiles.py`
- `splatting_script/validation/checks/resident_checks.py`
- `splatting_script/validation/forward_splatting/test_resident.py`
- `splatting_script/validation/validate_resident_gpu.py`
- `splatting_script/validation/test_direct_tiles_cpu.py`
- `splatting_script/validation/test_direct_validation_cpu.py`
- `splatting_script/validation/test_direct_tiles_command.py`
- `splatting_script/README.md`
- `docs/DIRECT_TILES.md`

Run the complete relevant CPU suites without invoking their artifact-writing
standalone entry points:

```bash
CUDA_VISIBLE_DEVICES='' "$HK_PY" -B -m unittest \
  splatting_script.validation.forward_splatting.test_resident \
  splatting_script.validation.test_direct_tiles_cpu \
  splatting_script.validation.test_direct_validation_cpu \
  splatting_script.validation.test_direct_tiles_command \
  splatting_script.validation.test_retained_commands -v
```

`--validate-only --support-selector fused` additionally compares resident and
hybrid against an independent **legacy resident** run from the same initial
parameters, including support CSR and allocation metadata, RGB, gradients,
updates and moments. It retains the full-coverage spectral diagnostics.

Normal timing adds no candidate timers or profiling synchronizations.
`report.json` now includes whole-iteration median and nearest-rank p95. Candidate
counts, visible face IDs, selected pairs per face and per kernel are gathered
only after timing and labeled accordingly. Existing `jit_cache.json` and
per-iteration miss counters remain. In separate `--profile` processes,
`candidate_activity.json` attributes CUPTI GPU activities by launch correlation
and CPU thread to scoring/filtering, ranking, combined integer ranking/row
output, compact output and metadata reads. It reports launch counts, kernel
duration sums, GPU activity union, timeline span, copies/bytes and host sync
API waits. Union excludes idle time; timeline span includes it; CPU waits
overlap GPU activity. Legacy stages use native-op attribution buckets; combined
row sorting/output is not falsely split. Missing correlations remain explicit.

Remaining risks are CUDA compiler handling of strict float64 arithmetic,
platform/version-specific topk ties (the native operation is retained), filter
memory bandwidth and local integer-sort occupancy. The GPU-ready boundary and
same-device tie tests are required before any correctness/speed claim. Remaining
costs include unchanged cdist/kthvalue/topk, compact scans/metadata sync and
coefficient/backward work. Success requires passing the A100 gates and improving
matched median/p95; neither is inferred from fewer source operations.

### Manual Grace A100 protocol

Run in the existing user allocation. These commands do not allocate resources
and were not executed by the agent. Reuse the exact same checkpoint in every
process; no profiled/validated run supplies normal speed evidence.
The path below is verified from all three saved 21.118/10.984/11.038 ms reports;
their common initial-state hash is
`babff9646c9019180225b9fdbfac3a8a53b5bd7cbe5722122e53557f9cc4df20`.

```bash
HK_PY=/scratch/user/"$USER"/.conda/envs/hktex/bin/python
export LD_LIBRARY_PATH=/scratch/user/"$USER"/.conda/envs/hktex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
INITIAL=experiments/converse_heat_kernel_splatting/results/direct_tiles_direct_20261105T044854_349740Z_0529a6a42c/initial.pt

# Same-device policies, nextafter boundaries, resident/hybrid VJPs and Adam.
"$HK_PY" -B -m splatting_script.validation.validate_resident_gpu --execute --compare-selectors

# Real trained-field gate against legacy resident, plus full spectral diagnostics.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --validate-only --support-selector fused \
  --initial-state "$INITIAL" --selection hybrid --splat-backend triton \
  --footprint-cutoff 0 --kernels 1000 --resolution 64 --support-cap 128 \
  --coefficient-backend triton --runtime-counts --visibility reuse

# Three rounds with alternating/reversed process order: four configurations each.
for order in 'resident:legacy resident:fused 2dgs:legacy 2dgs:fused' \
             '2dgs:fused 2dgs:legacy resident:fused resident:legacy' \
             'resident:legacy resident:fused 2dgs:legacy 2dgs:fused'; do
  for case in $order; do
    backend=${case%:*}
    selector=${case#*:}
    /usr/bin/time -p "$HK_PY" -B -m splatting_script.experiments.compare_backends \
      --backend "$backend" --support-selector "$selector" --execute \
      --initial-state "$INITIAL" --selection hybrid --splat-backend triton \
      --footprint-cutoff 0 --kernels 1000 --resolution 64 --support-cap 128 \
      --coefficient-backend triton --runtime-counts --visibility reuse \
      --warmup 8 --iterations 100
  done
done

# Separate late traces for all four configurations, after the same 40 updates.
for backend in resident 2dgs; do
  for selector in legacy fused; do
    "$HK_PY" -B -m splatting_script.experiments.compare_backends \
      --backend "$backend" --support-selector "$selector" --execute \
      --initial-state "$INITIAL" --selection hybrid --splat-backend triton \
      --footprint-cutoff 0 --kernels 1000 --resolution 64 --support-cap 128 \
      --coefficient-backend triton --runtime-counts --visibility reuse \
      --warmup 8 --iterations 48 --profile --profile-start 40
  done
done
```

## Optimization audit before editing

The three repeated 100-iteration runs in the supplied evidence have medians
74.153/74.789/75.877 ms for hybrid versus 10.738/10.686/10.675 ms for resident.
In `direct_tiles_direct_20261106T053334_297693Z_c8afe7c7ed`, hybrid's median
host enqueue wall is 74.761 ms, process CPU is 57.686 ms and final device drain
is 0.026 ms. These wall/CPU measurements overlap GPU work and mid-step waits;
they cannot be added to CUPTI durations or called pure GPU execution time.

The first eight measured hybrid iterations take 11.69–32.40 ms; iteration 8
onward is mostly 70–91 ms. Other repeats likewise have only 13/14 fast samples.
The eight-iteration profiles therefore sample the short warmed trajectory,
not the sustained slow regime. Their apparent speed is not evidence that the
profiler itself fixes the regression. Resident remains fast through all 100
iterations, and both methods have very similar early losses.

The matched hybrid profile `direct_tiles_direct_20261106T051315_938925Z_b1bd5f57ee`
records 67.404 ms in candidate GPU kernels, 17.005 ms in coefficient GPU kernels,
14.039 ms in backward and 5.242 ms in optimizer kernels over eight iterations.
The actual `_group_triangle_work` GPU kernel totals **0.047137 ms** and
`_hybrid_emit` **0.555810 ms** over those eight iterations. These are CUPTI
durations, not host enqueue, compilation or whole-iteration latency.

| Audited operation | Resident versus hybrid before optimization |
| --- | --- |
| Visibility | Same fixed raster/mesh samples; reused in timed iterations |
| Destination support | Already limited to `unique(face_ids[face_ids>=0])` by `PreparedSupport`; all sources stay eligible. No second visible-face optimization is needed |
| Source state / coefficients | Same `LiveHeatField.sources`, `ResidentSupportGeometry.select`, `ResidentTopology`, fused coefficients; no repeated per-pixel spectral sums |
| Shared synchronization | Resident selector reads four int64 allocation sizes (32 bytes) in both paths. Candidate indexing and optimizer trace also have shared synchronization; final drain alone does not capture them |
| Hybrid-only work | One stable kernel/tile sort, group masks/scan, queue packing, per-pixel supported-triangle lookup, extra validation kernels, scratch and diagnostic response storage |
| Reduction / backward | Hybrid already avoids emitted pixel sorting/histogram; reuses resident merge/reconstruction and inherited VJP. Its compact record contract still needs selected tape owned by each outstanding forward |
| Changing launch count | **Hybrid `_group_triangle_work` declares `N: tl.constexpr`**, whereas resident's analogous `identify_ranges` uses a runtime count. Every distinct task count changes hybrid's JIT key |
| Profiling | Eight warmup steps are reset before measurement. The next eight repeat that trajectory; a longer run visits new task counts. Existing short traces have no cache-miss census |

The installed Triton JIT binder includes every constexpr value in its
specialization key and calls compilation/cache loading on an in-memory miss.
Thus the changing-N specialization hazard is confirmed by source inspection.
Repeated JIT misses are the leading explanation of the delayed host regression,
not a measured attribution of all 74 ms: the old traces are too short, and the
GPU node's local compilation caches are unavailable here. A long run with
cache-miss wall/CPU counters and a late profile is required to quantify it.

Semantics-preserving priorities are to make N non-specialized runtime data,
avoid allocating/writing raw responses outside diagnostic traces, and reuse
immutable active-pixel ordinals. Reusing mutable selected/autograd buffers
across outstanding forwards would be unsafe. Changing support, caps, ties,
pruning, precision or using nearby triangles alone could change semantics and
is excluded from this optimization. Calling resident's complete renderer is
also excluded: kernel/tile splat scheduling remains the experimental path.

Before editing, all **54 existing CPU tests and static syntax checks passed**.
No GPU job, CUDA kernel or A100 benchmark was executed by the agent.

## Applied optimization and verification protocol

`_group_triangle_work` now receives a runtime `N`, with both value and alignment
specialization disabled. Grid size still follows the actual work count, and
the same masks delimit every partial block. New task counts no longer create
distinct specializations through this argument. This fixes a confirmed JIT-key
hazard; its share of the old 74.789 ms remains an inference until measured.

Training no longer allocates, zero-fills or writes the raw heat-response
diagnostic buffer. `trace=True` retains that buffer and its unchanged values.
The visible-pixel identity ordinals are immutable per camera and now cached in
`TileSurface`, with the same stale-buffer checks as other geometry. By code
inspection, each nonempty ordinary hybrid forward removes one response
zero-fill and one ordinal-generation tensor operation (normally two CUDA
launches), two allocations and the emitter's diagnostic response stores.
Actual launch counts await a CUDA trace. No synchronization is added or removed;
the shared 32-byte allocation read remains. Mutable top-30 tape/record buffers
remain owned by their individual forwards so outstanding autograd graphs are safe.

The original resident support, spectral coefficients, top-30/reconstruction,
backward and optimizer integration remain reused. Visibility already restricts
destination support, so no extra filter or candidate-set change was introduced.
Kernel/tile scheduling stays in the experimental renderer. Cutoffs, ties,
normalization, source eligibility, selection modes and tolerances are unchanged.

The existing benchmark now writes `jit_cache.json`: Triton in-memory cache
misses with kernel name, constexpr arguments, phase, iteration, wall time and
process CPU time. Hooks chain/restore existing hooks, observe misses only and
never read Tensor values or launch/synchronize CUDA. A miss may compile **or
load the disk cache**; neither these intervals nor process CPU time are pure
compiler/GPU time. Unsupported Triton hook APIs are labeled unavailable.
`iterations.json` includes matching miss deltas; `report.json.latency_windows`
separates the first eight, subsequent and last eight iterations and each camera.
Any misses during measurement remain part of reported iteration time.

`--profile --profile-start 40 --iterations 48` executes the actual preceding
40 optimizer steps and then traces eight steps. This tests the sustained
trajectory missing from the old profiles. The complete profiled-run median
mixes traced and untraced steps and is diagnostic only; `profiled` marks rows.
Separate `--host-profile` runs write the top cProfile host calls, including
blocking native calls, to `host_profile.json`. Nested cumulative durations
overlap; these also are diagnostic rather than speed evidence.

After editing, CPU tests cover varying runtime counts across 256-lane block
boundaries, empty queues, unchanged traced/training RGB and all VJPs, cached
ordinals, hook chaining/error restoration, safe constant reporting and the
short-versus-sustained latency summary. The full CPU command below retains all
previous correctness, oracle, finite-difference and optimizer checks.

No post-change GPU validation or speed measurement has been run. The target is
resident's **10.686 ms/iteration** under matched 100-step conditions. Remaining
suspects after the JIT fix are shared support/synchronization and coefficient
VJPs, and hybrid's queue sort/scan, temporary allocations, pixel triangle lookup
and top-30 work. Their relative cost needs the new long/late diagnostics;
the old short trace does not establish a sustained bottleneck ranking.

The current `--backend 2dgs` default is `--selection hybrid`. Its pipeline is:

```
spectral kernels → resident triangle support → shared corner/Gram coefficients
→ visible projected regions → unique (kernel,tile) work → compact pixel records
→ resident top-30/reconstruction primitives → original appearance
```

This changes the acceleration structure while keeping heat kernels as the
rendering primitives. Neither FAISS nor pointwise nearest-neighbor selection
is used in this primary path. The renderer does not call resident `build_scene`,
`splat`, `render_ranges` or the complete resident renderer. `original`,
`resident` and `2dgs` remain the only backend names. `--selection splat` and
`--selection native` retain the earlier full-positive spectral and native
outer-50 controls within 2dgs.

## Reused resident contracts

| Existing function/type | Contract and reuse |
| --- | --- |
| `LiveHeatField.sources()` | Fresh differentiable source barycentrics, grid weights, decay, eigenfunctions, self-heat, embedding and GeodesicOpt metadata |
| `ResidentSupportGeometry.select()` | Fresh float64 triangle-centroid/radius KNN-union selection and optional pool cap; returns device CSR counts/offsets/kernel IDs/entry faces |
| `PreparedGeometry`, `ResidentDispatchGeometry` | Immutable visibility and face/tile incidence geometry only; no dynamic resident scene or renderer |
| `ResidentTopology` | CSR-entry-to-triangle-vertices/kernel association for coefficient evaluation |
| `LiveHeatField.coefficients(..., topology=...)` | One heat row `[pair,3]` and distance Gram row `[pair,6]` per supported kernel/triangle; eager or existing fused implementation |
| `fused_coefficients`, `coefficient_gpu` | Existing forward and atomic source/grid/normalizer/embedding VJP; no new spectral backward implementation |
| `footprint.projected_bounds()` | Existing support-derived intrinsic conic and perspective interval bounds; optional CPU reference fallback, never an ordinary source Gaussian ellipse |
| `gpu.merge`, `runtime_counts.reconstruct` | Existing segmented positive top-30, CSR-entry tie order, normalized colors and clamping, fed this renderer's own compact pixel records |
| `_Splat.backward` | Inherited unchanged by the experimental `_HybridSplat` forward; reuses reconstruction/emit backward kernels and their gradient accumulation |
| Existing trainer optimizer/parser/scheduler/post-step | Adam and GeodesicOpt behavior unchanged |

The support selector retains GPU arrays. Its existing allocation contract reads
four int64 values to the host: pairs, maximum support, raw task count and pixel
fragments (**32 bytes, one allocation synchronization on CUDA**). No support IDs
or dynamic dispatch arrays are downloaded/uploaded in the default hybrid path.
The existing `check_sizes` guard rejects support wider than 512, unsafe integer
sizes, record overflow or dispatch budget overflow. The hybrid additionally
checks coefficient/queue/record scratch before allocating those buffers.

## Kernel-driven work and coefficient reuse

Only source-independent geometry is cached. Each forward selects new support,
constructs the resident coefficient topology and evaluates its spectral corners
once. Every pixel on a supported triangle reuses that row. Per-pixel work is
barycentric interpolation of three heat values and six Gram terms, followed by
the original distance attenuation and heat filter; no complete spectral sum is
repeated at every pixel.

The static raster's visible face/tile incidences are the exact projected regions
needed when pruning is disabled. Expand supported triangle incidences and sort
once by `(kernel,tile)` with stable face ordering. Compact live keys and entry
ranges remain on device. A work item represents one kernel and one screen tile;
its range can contain several supported triangles. Queue capacity is bounded by
the smaller of raw incidence count and `sources * active_visible_tiles`, with
its live count read on device. There is no Python loop over kernels.

Each GPU work item visits that tile's visible pixels. A small bounded lookup in
its sorted triangle-entry range finds the coefficient row for the pixel's
winning triangle. The tile rows partition visible pixels; CSR incidences are
unique; each queue key is unique. Therefore each `(kernel,pixel)` contribution
owns exactly one compact slot:

```
pixel_offsets[pixel] + coefficient_entry - face_CSR_offsets[visible_face]
```

Records are already grouped by pixel, avoiding the resident forward's emitted
pixel sort and histogram. This renderer owns emission, record storage and the
reduction schedule; it calls the resident segmented top-30 primitive with an
identity permutation. It retains the resident tie rule: lower CSR entry first
for equal nonnegative weights. Reconstruction divides by
`max(sum(selected_weights),1)`, adds the mean and applies the original clamps
and appearance. No alpha compositing is introduced.

The differentiable heat/Gram tensors stay attached to source state. The custom
forward saves the same selected tape contract as resident, allowing its exact
backward implementation to be inherited. Gradients from many pixels accumulate
into shared triangle coefficients, then into source parameters through the
existing fused coefficient VJP. Threshold, sharpness, colors and mean use the
same reconstruction/emit VJPs. Visibility, support, queue membership and top-k
selection remain discrete. First-order fixed-mesh training is the scope;
mesh/camera gradients and higher derivatives remain unsupported.

## Visible footprints and explicit pruning limits

The default `--footprint-cutoff 0` disables footprint pruning **within selected
triangle support**. It does not make support complete for all positive heat.
Supported triangles can form several disconnected screen regions. Hidden
sources are never rejected because their source point is occluded. Only
winning visible triangle IDs/barycentrics are shaded, so hidden surfaces cannot
paint independently through foreground geometry.

Positive cutoffs invoke the unchanged resident `projected_bounds` routine. Its
intrinsic quadratic bound is built from actual heat corners, filter inversion
and the source-relative Gram matrix; its perspective interval, rank-deficient
and near-plane fallbacks are retained. Expanded kernel/triangle/tile incidences
are admitted when the tile intersects the returned screen bound, then grouped
into the same kernel/tile queue. Stale mesh/archive/tile geometry is rejected;
projection camera changes require a new visibility snapshot.

**This optional projected-bound implementation is currently a CPU diagnostic
fallback.** It downloads support/coefficient/filter arrays and uploads bounds,
and loops over coefficient entries in the existing CPU routine. The fixed
mesh is downloaded once per camera snapshot during setup. Its
extra dynamic D2H bytes and projection stats are reported. It is explicitly
outside the GPU-resident default and is not a demonstrated speed optimization.
`bounds=full` disables that pruning. Cutoff validation still compares to the
unpruned resident result at unchanged tolerances. A failed cutoff gate must not
be described as equivalence. A future GPU port would require separate review.

## Selection semantics, validation and measured limitations

The hybrid uses resident triangle-union/cap semantics (default cap 128) and its
entry-order tie policy. It may differ from native pointwise FAISS outer-50 and
from the full-positive spectral oracle. Validation now:

1. Starts native, resident, hybrid and full spectral reference from identical
   parameters with gradients/optimizers reset independently.
2. Gates hybrid RGB, support IDs, corner/Gram coefficients, positive top-30 sets,
   parameter VJPs, updates and Adam moments against resident. Numerical gates
   retain `atol=3e-5, rtol=3e-4`; set checks remain exact.
3. Checks heat responses for every admitted source against the full spectral
   oracle, and reports omitted positive contributions and positive top-30
   contributors, affected pixels, and RGB differences separately.
4. Does not claim native or full-coverage equivalence. Even complete observed
   support on the initial camera does not prove general coverage or identical
   full-reference tie behavior.

The prior full-coverage A100 result (13.26 ms versus resident 10.94 ms and
original 20.36 ms) is historical. The supplied current hybrid result is
74.789 ms versus resident 10.686 ms; post-optimization performance is unmeasured.
Possible bottlenecks include fresh
support selection and its 32-byte allocation synchronization, coefficient
gathers/VJPs, queue expansion/sort, capacity launches, pixel triangle lookup,
top-30 reduction and atomic accumulation. The optional CPU conic routine may
dominate an iteration and must be measured separately. Memory now scales with
supported pairs, expanded incidences and supported pixel fragments, rather
than all-source spectral evaluations for every visible pixel.

Positive pruning currently reduces active queue work, not coefficient or
record capacity: all selected support coefficients are needed to build the
bounds, and supported pixel slots stay allocated at their safe unpruned size.
Trace statistics distinguish record capacity from actually emitted records.

Profiler scopes distinguish `triangle_support`, `triangle_coefficients`,
`footprint_projection`, `kernel_triangle_tile_queue`, `hybrid_pixel_emission`,
`hybrid_top30_reduction`, `training::backward` and `training::optimizer`.
Use CUPTI activity correlation from the separate profile run for kernel time;
do not sum overlapping parent/child scopes or treat CPU enqueue walls as GPU
durations. Whole training iteration times and allocator peaks remain reported.
Setup, first camera visits, remaining warmup and reset are separate from
measured iterations. Isolated compiler duration is still unmeasured (`null`);
compilation is included in first-use visits alongside training work.

Quality evaluation now includes the existing `heldout_left` and `heldout_right`
camera definitions after timing. They are excluded from the four-camera
training/warmup schedule. Reports label training/held-out splits and provide
`heldout_quality`; earlier reports are preserved and are not relabeled.

## CPU verification and exact manual A100 commands

The numerical suites hide CUDA, forbid CUDA initialization and use Triton's
CPU interpreter. Previous native/full-coverage regressions remain intact as
explicit controls. New tests cover grouped kernel/tile dispatch, coefficient
reuse, matched resident RGB/VJPs, hidden/curved/folded/disconnected/occluded
samples, unique records, ties/normalization/clamping, support omissions,
finite differences for every fixture parameter, independent Adam moments,
the original projected-bound routine, duplicate/oversized support rejection,
memory capacities, source movement, empty visibility and stale snapshots.
No new repository file/directory or GPU job is needed for these tests.

The pre-edit run passed **54 CPU tests**. The complete post-edit run passed
**61 CPU tests in 51.819 seconds**, with CUDA hidden and initialization forbidden:
17 retained native/direct tests, 10 retained full-coverage tests, 15 hybrid
tests, ten comparison/instrumentation regressions, six command checks and
three retained-command/input checks.
Static syntax checks passed for all six edited Python files. A separate
stdlib-only smoke check passed for the cProfile report without importing
Torch/Triton. Preservation checks found no content changes in 181 protected
Python files and no size/mtime changes in any non-edited repository file;
existing reports, inputs and recovery archives remain untouched.
Focused reruns also cover empty-view Adam moment decay and projection/cache
invalidation. On the small float32 matched fixture, hybrid/resident maximum
RGB difference was **0**, and maximum differences for all eight fixture
parameter VJPs were **0**. This is interpreted CPU evidence, not an A100 claim.
Finite differences use step `1e-6` and tolerance
`max(2e-7, 2e-5*abs(gradient))`; three independent Adam steps plus an empty
view compare parameter and moment state at the retained float32 tolerances.

```bash
CUDA_VISIBLE_DEVICES='' "$HK_PY" -B -m unittest \
  splatting_script.validation.test_direct_tiles_cpu \
  splatting_script.validation.test_direct_validation_cpu \
  splatting_script.validation.test_direct_tiles_command \
  splatting_script.validation.test_retained_commands -v
```

The commands below are for the user's existing A100 session, not allocation
requests. They were not executed by the agent. Use one exact saved initial
state for all methods:

```bash
HK_PY=/scratch/user/"$USER"/.conda/envs/hktex/bin/python
export LD_LIBRARY_PATH=/scratch/user/"$USER"/.conda/envs/hktex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
INITIAL=experiments/converse_heat_kernel_splatting/results/direct_tiles_direct_20261105T044854_349740Z_0529a6a42c/initial.pt

# Hybrid versus matched resident, plus full-positive omission diagnostics.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --validate-only --selection hybrid --splat-backend triton \
  --footprint-cutoff 0 --support-cap 128 --coefficient-backend triton \
  --initial-state "$INITIAL" --kernels 1000 --resolution 64

# Retained full-coverage and native reference gates.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --validate-only --selection splat --splat-backend triton \
  --footprint-cutoff 0 --initial-state "$INITIAL" --kernels 1000 --resolution 64
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --validate-only --selection native --nearest-backend faiss \
  --initial-state "$INITIAL" --kernels 1000 --resolution 64

# Separate late diagnostic profile; repeat with resident for matched traces.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --profile --profile-start 40 --selection hybrid --splat-backend triton \
  --footprint-cutoff 0 --support-cap 128 --coefficient-backend triton \
  --initial-state "$INITIAL" --kernels 1000 --resolution 64 --warmup 8 --iterations 48 \
  --runtime-counts --visibility reuse

# Separate long host diagnostic; repeat with resident if latency still regresses.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --host-profile --initial-state "$INITIAL" \
  --selection hybrid --splat-backend triton --footprint-cutoff 0 \
  --kernels 1000 --resolution 64 --warmup 8 --iterations 100 \
  --support-cap 128 --coefficient-backend triton --runtime-counts --visibility reuse

# Matched complete training iterations, with permuted process orders.
for order in 'original resident 2dgs' '2dgs original resident' 'resident 2dgs original'; do
  for backend in $order; do
    /usr/bin/time -p "$HK_PY" -B -m splatting_script.experiments.compare_backends \
      --backend "$backend" --execute --initial-state "$INITIAL" \
      --selection hybrid --splat-backend triton --footprint-cutoff 0 \
      --kernels 1000 --resolution 64 --warmup 8 --iterations 100 \
      --support-cap 128 --coefficient-backend triton --runtime-counts --visibility reuse
  done
done
```

Compare initialization hashes, all matching arguments, iteration medians,
allocator peaks, held-out PSNR, measurement cache misses and semantic diagnostics.
The first-eight and after-first-eight summaries should both stay fast. If misses
remain, inspect kernel names/constants and correlate their intervals with slow
iterations before attributing latency. Validation/CUDA-profile/host-profile
processes are excluded from speed evidence. Optional positive-cutoff runs must
name their cutoff and first pass the same hybrid/resident gate; their CPU
projection cost and approximation must not be hidden.

The previous JIT/hybrid optimization changed eight existing files: the two experimental
`direct_tiles.py` files; `splatting_script/benchmarks/direct_tiles.py`;
`splatting_script/validation/test_direct_tiles_cpu.py`,
`test_direct_validation_cpu.py`, `test_direct_tiles_command.py`;
`splatting_script/README.md`; and this document. No files are added or deleted.
Original HKTex, resident code, scientific inputs, historical reports and
recovery archives are protected. No new backend is introduced.

## Retained full-coverage and native-reference architecture notes

The following material describes the earlier full-coverage renderer and its
native control. It does not describe the default hybrid's support or dispatch.

# Kernel-driven spectral Heat-Kernel Splatting (full-coverage reference)

`--backend 2dgs --selection splat` uses **kernel → visible tile footprint → compact
kernel/tile queue → spectral pixel contributions → normalized top-30 RGB**.
It does not construct or consume dynamic kernel/triangle support CSR and does
not call FAISS. `--selection native` retains the previous pointwise outer-50
renderer as an explicit control within this backend. There are still exactly
three maintained backends: original, resident and 2dgs.

## Work, bounds and visibility

`TileSurface` reuses resident `BVHRaster` visibility: visible triangle IDs,
barycentrics, positions, depth and intrinsic sample embeddings. Source-independent
tile rows contain visible pixel ordinals, once each, with -1 padding. Hidden
triangles never become painted samples. A hidden source is not culled: its
intrinsic diffusion is evaluated at whichever surface samples are visible.
Curved/folded surfaces and disconnected image regions need no single ellipse.
Source-conditioned anisotropic spectra remain represented by the fixed archive
plus visible vertices/barycentrics, and are interpolated during shading rather
than caching a large pixel/grid/mode tensor.

`kernel_footprints` builds a kernel-major membership mask and a compact queue
of admitted `source * tile_count + tile` work items. Prefix sums and masked
compaction stay on the device. The Triton launch uses fixed queue capacity and
only executes the live prefix, whose count is read on device. Each kernel/tile
program processes groups of visible pixels, gathers the actual four-grid
spectral basis and source decay/self-normalization, and emits heat responses.
The spectral-mode dimension has contiguous/coalesced archive reads; response
columns have strided writes to favor the subsequent pixel reduction. One task
owns each kernel/pixel slot, so emission has no write races. Reconstruction
combines the separate columns after selecting top contributors.

Exact heat can be positive over an entire connected surface. Consequently,
**cutoff=0 conservatively retains every visible tile for every kernel**; this
is the default and makes no locality claim. Dynamic outer-KNN radius bounds
are not used for splat semantics. `bounds=full` is also an explicit coverage
control. With a positive `--footprint-cutoff`, an independent upper bound is:

```
H(source) = ||decay * source_phi||_2 * B * sum(abs(grid_weights)) / self_heat
B >= max_grid,vertex ||eigenfunctions(grid, vertex)||_2
d_min(source,tile) = max(||source_embedding - tile_center|| - tile_radius, 0)
response_upper = H * exp(-(d_min² + distance_eps)/(2*sigma²))
```

Apply the original monotone heat filter to this upper bound and retain tiles
whose upper filtered contribution can exceed the cutoff. Nonfinite bounds or
nonpositive sharpness retain coverage. The implementation pads the spectral
norm, barycentric norm and distance calculations outwards. These are engineering
guards, not a formal floating-point interval proof. Positive cutoffs approximate
all-positive semantics; their output and selected sets must still pass the full
oracle's unchanged tolerances. A failed cutoff check must not be called parity.

The global archive norm requires a setup pass and a small cached scalar per
pruned camera snapshot. Runtime bounding costs O(kernels * tiles), with no
triangle enumeration. Bounds refresh on source/angle/anisotropy/time/filter
changes; archive, embedding or mesh mutation invalidates the snapshot. The
exact default avoids the global archive-bound pass but performs more shading.

## Resident code reused and footprint limits

| Existing utility | Use in 2dgs |
| --- | --- |
| `LiveHeatField.sources` | Source faces/barycentrics, grid mixture, decay, eigenfunctions, self-heat, embedding and optimizer metadata |
| Resident `BVHRaster` / `StaticRaster`, shared `perspective_camera_basis` | Camera transforms, frustum/clip bounds, occlusion, pixel centers, depth and barycentrics |
| `LiveHeatField._chunk` | Eager spectral oracle and selected-record coefficient reference |
| `modules.coefficients.fused_coefficients` | Differentiable selected kernel/pixel heat/Gram coefficients |
| `triton.coefficient_gpu.coefficients_backward` | Existing atomic source/grid/decay/normalizer/embedding VJPs through that autograd wrapper |
| `heat_weighting.compute` | Original intrinsic Gaussian distance attenuation during differentiable recomputation |
| Model `kernel_filter_func`, `diffuse_heat_kernels`, `forward` | Original rescaled filter, positive weights, top-k, clamped normalization, kernel/mean colors and appearance |
| Trainer optimizer/parser/scheduler/post-step | Existing Adam and GeodesicOpt updates, unchanged |

The full resident `projected_bounds` routine cannot be reused independently:
its conic, Gram matrix, heat maximum and perspective intervals are indexed by
kernel/face support entries. Its conic inversion, `kernel_face_order` and
`prune_dispatch` remain read-only references, exercised by the retained old
footprint test. No independent projection helper exists inside that routine.
The camera projection/bounding operations shared through `perspective_camera_basis`
and the resident visibility raster are reused. The new footprint is a union of
visible image tiles bounded by the spectral distribution, not an ordinary
Gaussian disk projected from the source. The intrinsic Gaussian attenuation
above is HKTex's existing distance weighting, not a replacement heat field.

The resident full `splat` scene contract requires face patches and support
entries, so feeding it the new queue would reintroduce triangle dispatch.
Instead, the new emitter produces kernel/pixel responses. Discrete top-k IDs
are selected under `no_grad`; only those records are recomputed with resident
coefficients and native reconstruction under autograd. The forward emitter
streams the same spectral contraction without storing per-face coefficients.
The shared resident coefficient VJP carries gradients to locations, grid
angles/anisotropies and diffusion time; original filtering/reconstruction carry
threshold, sharpness, color and mean gradients. No separate full renderer or
custom copy of the resident backward was added.

## Semantics, memory and numerical gates

Splat mode considers all positive spectral responses before top-30 selection.
It deliberately omits native outer-50 preselection. It agrees with native where
all relevant sources are admitted and ties/order permit, but it is not a claim
of exact native equivalence. The native control keeps FAISS batch dimensions,
native search/distance operations and the earlier strict native gate. Legacy
float64 tile KNN controls are selectable only with `--selection native` and
still have the documented FAISS rounding/tie limitation.

Splat reduction uses Torch top-k over source-ID columns, then original
`diffuse_heat_kernels` with selected spectral responses. The denominator is
`max(sum(selected_filtered_weights), 1)`; add mean color and apply the original
clamps/appearance. Selection, visibility and footprints are discrete. Float32
reduction order can change near-tied selected IDs; the full-oracle gate reports
that rather than relaxing its tests. First-order fixed-geometry training is
supported; higher derivatives and mesh/camera derivatives are not audited.

Current buffers include a dense visible-pixel/kernel response/filter scratch
and a padded compact queue. This costs O(pixels * kernels + tiles * kernels);
this implementation is not yet a sparse contribution allocator. Static
dimensions are checked against `--max-contributions` (default 16,777,216) and
`--max-buffer-mib` (default 512) before allocation. Exceeding either budget
raises explicitly, never drops contributions. The scratch estimate covers
direct buffers, not full allocator/optimizer/archive memory. Device queue
prefix counts cannot overflow the allocated queue by construction; inactive
slots never shade. Selected-record autograd costs O(pixels * top_k * modes).

Default exact emission may be slower than resident: global heat coverage,
capacity-based launches, dense scratch, spectral gathers and selected-record
recomputation remain significant. Positive-cutoff bounds offer a practical
alternative, subject to full-oracle checks. No A100 test or benchmark of the
new emitter has run here. User-reported earlier medians (21.12 ms original,
10.92 ms resident, 20.73 ms old FAISS experiment) are historical comparisons;
**beating 10.92 ms is an unverified target**.

Validation compares kernel emission against the full eager spectral oracle,
including spectral responses, RGB, positive top-30 sets, VJPs, independent
optimizer updates and Adam moments. Numerical tolerances stay `3e-5/3e-4`
(absolute/relative), and set checks remain exact. Native RGB differences are
reported separately as semantic evidence. All comparison resets clear gradients.

## CPU evidence and manual A100 commands

40 CPU tests passed across the direct, comparison-isolation, command and
retained-command suites: all 17 prior direct regressions, 10 new kernel-splat
tests, six comparison regressions, four command checks and three retained
command/input checks. They cover real Triton programs in CPU interpreter,
resident coefficient backward, hidden/visible/folded/disconnected surfaces,
occlusion, no-support/no-FAISS architecture, sparse/empty queue tasks, top-30
accumulation and clamps, source movement, capacity failure, stale archives,
finite differences, every fixture parameter VJP, and three independent Adam
steps/moments. Existing tests were retained; the old parity cases explicitly
select native mode. CPU tests create no repository output files or directories.

```bash
CUDA_VISIBLE_DEVICES='' "$HK_PY" -B -m unittest \
  splatting_script.validation.test_direct_tiles_cpu \
  splatting_script.validation.test_direct_validation_cpu \
  splatting_script.validation.test_direct_tiles_command \
  splatting_script.validation.test_retained_commands -v
```

The following commands are for manual execution in an existing user-owned A100
session; no resource allocation command is used. They were not executed here.
Reuse the exact initial state from the successful earlier native-control run:

```bash
HK_PY=/scratch/user/"$USER"/.conda/envs/hktex/bin/python
export LD_LIBRARY_PATH=/scratch/user/"$USER"/.conda/envs/hktex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
INITIAL=experiments/converse_heat_kernel_splatting/results/direct_tiles_direct_20261105T044854_349740Z_0529a6a42c/initial.pt

# New kernel emitter versus full spectral oracle; no warmup/timing afterward.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --validate-only --selection splat --splat-backend triton \
  --footprint-cutoff 0 --initial-state "$INITIAL" --kernels 1000 --resolution 64

# Earlier native outer-50 parity control, within 2dgs.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --validate-only --selection native --nearest-backend faiss \
  --initial-state "$INITIAL" --kernels 1000 --resolution 64

# Optional approximate footprint: validate against full coverage before timing.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --validate-only --selection splat --splat-backend triton \
  --footprint-cutoff 0.0001 --initial-state "$INITIAL" --kernels 1000 --resolution 64

# Matched complete training iterations; repeat in permuted backend orders.
for order in 'original resident 2dgs' '2dgs original resident' 'resident 2dgs original'; do
  for backend in $order; do
    /usr/bin/time -p "$HK_PY" -B -m splatting_script.experiments.compare_backends \
      --backend "$backend" --execute --initial-state "$INITIAL" \
      --selection splat --splat-backend triton --footprint-cutoff 0 \
      --kernels 1000 --resolution 64 --warmup 8 --iterations 40 \
      --support-cap 128 --coefficient-backend triton --runtime-counts --visibility reuse
  done
done
```

Keep validation/profiling separate from speed runs. Compare initial-state
hashes, config, camera mix, iteration times, allocator peaks and PSNR/native
differences; splat/native trajectories need not coincide. The new spectral
emitter and selected-record work have separate profiler scopes. Compilation,
startup and warmup remain separate from steady state. Reports separate setup,
first camera visits, remaining warmup and reset. Cold-cache compilation is
included in those first visits alongside training work; isolated compiler
duration is not measured (`compilation_seconds=null`). External `time`
includes the whole process. Exact full-coverage timing is the initial control;
approximate timing must name its cutoff and include its correctness/quality
evidence. Original and resident source, assets, results and recovery archives
remain unchanged; all edits are in existing experimental files.

Exact modified files for this refactor:

- `hktex_splatting/rendering/direct_tiles.py`
- `hktex_splatting/triton/direct_tiles.py`
- `splatting_script/benchmarks/direct_tiles.py`
- `splatting_script/experiments/compare_backends.py`
- `splatting_script/validation/test_direct_tiles_cpu.py`
- `splatting_script/validation/test_direct_validation_cpu.py`
- `splatting_script/validation/test_direct_tiles_command.py`
- `splatting_script/README.md`
- `docs/DIRECT_TILES.md`
- `docs/DIRECT_TILES_VALIDATION_FIX.md`

## Earlier pointwise native-control audit

The following source/math audit describes the retained `selection='native'`
control and its original CPU evidence, not the new default splat traversal.
See [the earlier failure/fallback analysis](DIRECT_TILES_VALIDATION_FIX.md).

Consolidation update: the maintained selector and current comparison commands
are in [splatting_script/README.md](../splatting_script/README.md). Its resident
defaults use the validated fused coefficients and runtime counts. Retired
standalone experiments are recoverable through the
[cleanup inventory](three_backends_cleanup/INVENTORY.md). The CPU JSON below
remains the original audit snapshot; its earlier benchmark-harness source is
preserved in the cleanup source-context archive. Native/resident numerical
Python and the direct tile mathematics were not changed by consolidation.

The existing projected footprints **depend on dynamic kernel/triangle support**.
They cannot simply replace its construction. They are per-face intrinsic conic
bounds followed by perspective interval AABBs, not per-source Gaussian disks.

An opt-in support-free prototype is in
[`direct_tiles.py`](../hktex_splatting/rendering/direct_tiles.py). It uses new,
conservative tile bounds for the real-arithmetic outer-KNN domain, then executes the
original spectral gather, heat, filtering, top-30 and reconstruction methods at
the visible surface point. It supports hidden sources and disconnected image
regions without projecting hidden surface values onto the image. It does not
call the old projected conics. No speedup or real-CUDA correctness is claimed.

Only new files were added for this work. The pre-existing dirty working tree,
`hktex/`, current resident implementation and archived results were preserved.
See `DIRECT_TILES_CPU.json` for measured CPU evidence and preservation checks.

## Source map and actual execution order

Locations below refer to the inspected working-tree sources, not historical
copies under the archived experiments.

| Operation | Exact implementation |
| --- | --- |
| Spectral asset generation and anisotropic LBO basis | `hktex/modules/eigen_albo.py:96`, `:172`, `:219` |
| Native source grid selection and barycentric source gather | `hktex/modules/eigen_albo_knn.py:119` (`build_kernel_graph`), `hktex/knn_heat/knn_heat/knn_gather.py:64`, `:106` |
| Query grid/barycentric gather | `hktex/modules/eigen_albo_knn.py:158` (`query_points`), `hktex/knn_heat/knn_heat/knn_gather.py:141`, `_ops/gather.py:63` |
| Spectral sums and self-normalization | `hktex/knn_heat/knn_heat/_ops/heat.py:46`, `:61`; `knn_heat.py:74`, `:87`, `:146` |
| Outer source selection / differentiable distances | `hktex/knn_heat/knn_heat/faiss_knn.py:225`, `:256`; `hktex/modules/eigen_albo_knn.py:158` |
| Spectral-distance Gaussian attenuation | `hktex/knn_heat/knn_heat/post_weights.py:108`, `_ops/weights.py:30` |
| Original rescaled sigmoid | `hktex/utils/misc.py:232` |
| Native top-K and normalized RGB | `hktex/modules/heat_kernel_texture_knn.py:117` (`diffuse_heat_kernels`), especially `:161`, `:172`, `:181`; appearance postprocess `heat_kernel_texture.py:446` |
| Native rendered surface samples | `hktex/rendering/heat_kernels_renderer_knn.py:28`, `:40`: Mitsuba passes `si.p` and `si.prim_index`; `diffhk_renderer.py:48` supplies the differentiable bridge |
| Experimental full-mesh visibility | `hktex_splatting/rendering/fast_raster.py:13`, `:130`; `kernel_centric_raster.py:36` adds conservative frustum BVH traversal |
| Native unlit camera reference wrapper | `hktex_splatting/rendering/native_reference.py` |
| Live source state and coefficients | `hktex_splatting/modules/live_field.py:51`, `:75`, `:95`; optional fused version `modules/coefficients.py:10`, `:36`, `triton/coefficient_gpu.py:12`, `:51` |
| Dynamic kernel/triangle support | `modules/live_field.py:142`; real-arithmetic union `rendering/kernel_centric.py:17`; packed policy `knn_heat/packed_support.py:50`; resident selector `knn_heat/resident_support.py:72` |
| Existing projected footprints | `hktex_splatting/rendering/footprint.py:13` (`projected_bounds`), `:103` (`prune_dispatch`) |
| Geometry and dispatch | `rendering/core.py:53`, `prepared_geometry.py:15`, `gpu_dispatch.py`, `resident.py:68`, `:83` |
| GPU emission, selection and reconstruction | `hktex_splatting/triton/gpu.py:7`, `:34`, `:61`; cooperative option `triton/resident_gpu.py:35` |
| Forward/backward integration | `rendering/differentiable.py:102`, `:147`, `:172`; explicit VJPs `triton/backward.py:12`, `:43`; runtime-count versions `triton/runtime_counts.py` |
| Controlled training order | `hktex_splatting/trainers/forward.py`, local `prepare`, `project`, `native` and training loop |
| New tile bounds, selection, native shading | `rendering/direct_tiles.py`: `TileSurface`, `bin_footprints`, `nearest_sources`, `DirectTileRenderer`; `triton/direct_tiles.py:_nearest` |

The paths in shortened table cells are relative to `hktex_splatting/`.
`hktex/modules/tracer.py` traces learned source movement for GeodesicOpt; it is
**not** the camera occlusion pass. It already had local changes before this
work and was not edited.

For native KNN inference the dependency sequence is:

1. Read fixed mesh and precomputed spectral arrays.
2. Convert every source location to barycentrics on its current source face;
   choose its angle/anisotropy grid mixture; gather its eigenvalues and source
   eigenfunctions; build its self-heat cache and FAISS embedding index.
3. Obtain surface samples and triangle identities from Mitsuba intersections,
   or from the shared mesh visibility prepass in the unlit comparison harness.
4. Interpolate each query's intrinsic embedding; find pointwise outer neighbors;
   recompute their distances differentiably; gather source-conditioned query
   eigenfunctions; evaluate normalized, distance-weighted spectral heat.
5. Filter responses, select inner contributors, reconstruct and postprocess RGB.

Original UV training samples mesh points rather than camera pixels. The
controlled camera-space training harness shares an explicit raster prepass
between methods. It compares unlit texture reconstruction, not the complete
lit Mitsuba path-integrated image.

For current GPU-resident camera-space training:

1. Rasterize the full mesh on CPU to nearest triangle ID, perspective-correct
   barycentrics and depth. Fixed geometry can be reused for fixed cameras.
2. Rebuild `LiveHeatField.sources()` from current source parameters.
3. Select a conservative outer-KNN union on **visible triangles**, optionally
   cap that union, count/scan and pack fresh device kernel/face CSR. The current
   resident path reads 32 allocation-metadata bytes to the host; source IDs
   remain on device. This still synchronizes once for allocation sizes.
4. Sort by kernel/face, expand to visible face/tile tasks, sort integer
   `[patch | CSR entry]` keys, identify ranges and construct coefficient topology.
5. Compute spectral heat at the three corners of each retained kernel/face
   pair, plus six Gram distance coefficients. These tensors retain parameter
   gradients. Interpolation of corner heat equals interpolation of query
   eigenfunctions in real arithmetic, with finite-precision reassociation.
6. Emit responses for visible pixels of each task; sort/merge top contributors
   per pixel, or use the opt-in cooperative patch-range implementation.
7. Normalize and reconstruct RGB. Backward replays selected contributions and
   scatters VJPs to coefficients, colors, base, thresholds and sharpnesses.
   Coefficient/source autograd continues to location, grid mixture and decay.

The old `--mode projected` branch adds **after steps 3–5**: download live
coefficients/properties, compute CPU per-entry conics, build the existing
dispatch, prune it and upload a reduced scene. Its production trainer flags
cannot be combined with `--gpu-resident`; these are separate existing modes.
The conics do not participate in source selection or spectral evaluation.

## Three concepts that must remain separate

The intrinsic spectral function is, with the source's grid-interpolated basis,

\[
H_t(s,q)=\sum_j e^{-\lambda_jt}\phi_j(s)\phi_j(q),\qquad
h(s,q)=\frac{H_t(s,q)}{H_t(s,s)+\epsilon_h}.
\]

The configured KNN renderer additionally multiplies by
`exp(-(max(d2,0)+distance_eps)/(2*sigma**2))`, where `d2` is distance in
`_iso_embeddings`, not camera XY or mesh XYZ under the audited configuration.
`EigenAlboInterpolationKNN.configure` defaults to isotropic eigenvectors
divided by eigenvalues, dropping the constant mode. Its separate
`use_euclidian_distance=true` configuration intentionally changes that
embedding; it is not enabled by the Converse recipe.

Native preparation retains only the nearest `min(knn_outer_k, N_sources)`
sources at each query. Converse uses outer=50 and inner=30; class defaults and
other configurations differ. Both finite spectral modes/grid interpolation and
outer KNN truncation are existing reference behavior. A significant spectral
value outside native outer-50 can already be omitted by native HKTex. The new
path preserves this rule; it is not a renderer of the infinite, untruncated
heat equation or of all sources regardless of KNN.

For response `r` (including original configured power), native filtering is

\[
w=\operatorname{clamp}_{[0,1]}
\frac{\operatorname{sigmoid}(a(r-e))-\operatorname{sigmoid}(-ae)}
{\operatorname{sigmoid}(a(1-e))-\operatorname{sigmoid}(-ae)+10^{-8}}.
\]

Native chooses `filtered.topk(min(inner_k,K))`, then computes
`clamp(base + sum(w*residual_color)/clamp_min(sum(w),1),0,1)`.
Weights below total one are **not** rescaled to sum to one. There is no native
`1e-5` heat cutoff in this sequence. The native model's activation properties
and final optional appearance processing remain relevant.

A footprint only determines which sources could be candidates at a pixel.
Visibility independently determines the one surface point to evaluate at that
pixel. A source hidden behind the surface must not be culled on that basis.
Neither native query selection nor the inspected resident support selector
tests source camera visibility: all sources remain available. Resident support
limits the **query faces** to those seen by the camera. Hidden faces never emit
independent fragments into foreground pixels. An intrinsically distant hidden
fold must not gain influence because it projects near a query.

## What the old footprints actually contain

`projected_bounds(mapping, parameters, vertices, faces, camera, cutoff)` returns
one pixel-center AABB and one diagnostic 2x2 covariance per **CSR entry**.

| Question | Finding |
| --- | --- |
| Information defining a footprint | Entry source ID and face identity from CSR; three normalized corner heat values; six source-relative embedding Gram coefficients; source threshold/sharpness; sigma, distance epsilon, power restriction, cutoff; face vertices and camera. |
| Center, support or spectral origin? | A face-local quadratic level-set bound combines spectral `max(corner_heat)` and the metric. Its conic center is solved on that face's barycentric plane, not the projected source location. |
| Independent of dynamic support? | No. `len(mapping.kernels)` defines the outputs; `kernel_face_order(mapping)` recovers entry faces; `mapping.heat` and `mapping.metric` were computed for those pairs. Constructing all kernel/face pairs instead would only replace sparse support with a larger Cartesian product. |
| Conservative coverage? | Intended outward coverage of `w > cutoff` on already-retained face entries under power=1. It cannot restore omitted support, capped sources or previously invisible query faces. Its engineering floating-point guards and sampled tests are not a universal proof for arbitrary ill-conditioned input. |
| Curved/folded surfaces? | Handles them piecewise through actual faces and intrinsic coefficients. A single global source ellipse would fail. |
| Multiple projected regions? | Yes, via the collection of entries/faces. Each entry has one AABB. The diagnostic ellipse is never a global footprint for an entire kernel. |
| Parameter changes? | Location changes source eigenfunctions, embedding and Gram terms; angle/scale change the mixed spectrum; time changes decay and corner heat; threshold/sharpness change the inverse-filter bound; sigma changes distance attenuation; camera changes perspective projection. Bounds must be refreshed after relevant changes. Native time is a config constant, not a learned parameter. |
| Approximation/truncation? | Positive cutoff can discard nonzero responses; earlier support/caps can also omit terms. The local Jacobian ellipse is a first-order diagnostic. The actual pruning AABB uses numerator/depth intervals plus padding, with full-image fallback for rank deficiency, bad conditioning, nonpositive sharpness, near-plane/depth ambiguity and other uncertain cases. |

The derivation inverts the actual rescaled sigmoid at `cutoff` to obtain
`r_min`. With `H = max(corner_heat)`, `w > cutoff` requires
`b^T G b <= -2*sigma**2*log(r_min/H)-distance_eps`. The bound solves that conic
on the face plane and projects a 3D interval enclosure. The ellipse can extend
outside the physical triangle; this makes the bound looser. `prune_dispatch`
tests task AABBs against bounds; it does not replace response evaluation with
an ellipse. Dropping tasks can also change normalization and gradients through
their omitted small weights; cutoff is not an exact native contract.

The new curved-source footprint test checks above-cutoff recall on the existing
supported entries. It retained all 144 emitted records in that fixture (12
entries, 10 finite conics and 2 fallbacks). This supplies no speedup evidence
and does not prove universal conservativeness.

## New opt-in path and its guarantees

```python
from hktex_splatting.rendering.direct_tiles import TileSurface, DirectTileRenderer

# ids/bary/depth come from the unchanged full-mesh visibility prepass.
surface = TileSurface(trainer.mesh.faces, trainer.eigalbo_interp._iso_embeddings,
                      ids, bary, tile_size=8)
renderer = DirectTileRenderer(field, surface, selection="native", nearest_backend="triton")
rgb = renderer()  # Fresh source state, bounds and bins on every forward.
loss = (rgb[mask] - target[mask]).square().mean()
loss.backward()
```

The API is separate. Existing imports, CLI modes and resident algorithms were
not changed. The current audited `LiveHeatField` RGB/power=1 recipe remains the
scope. Fixed geometry/camera and first-order source/appearance derivatives are
supported; camera/mesh differentiation, density changes, antialiasing and
cross-face GeodesicOpt behavior require their own validation.

1. `TileSurface` interpolates each visible sample's embedding from its actual
   triangle/barycentrics. Each foreground pixel belongs to exactly one tile.
   Mesh/embedding/buffer mutation invalidates the snapshot.
2. Fixed center `c` and radius `R` enclose **all visible query embeddings** in
   each tile, even if several folds or disconnected triangles occur there.
3. Each forward computes source-to-center distances in float64, finds
   `d_K(c)` and retains sources satisfying
   `||s-c|| <= d_K(c)+2R+guard*(1+d_K(c)+R)`, with default guard `1e-5`.
   If a source violates the real-arithmetic bound, triangle inequalities give
   `||s-q|| > d_K(c)+R >= d_K(q)` for every query `q` in that tile.
   It cannot belong to real-arithmetic pointwise outer K. The guard is an engineering
   allowance for rounding; `bounds="full"` disables this pruning completely.
4. Device boolean membership is the projected footprint: a union of tiles,
   potentially disconnected. Sorting IDs produces padded device candidate rows
   and counts. There is no dynamic host size read, CSR compaction, source
   occlusion test or tile capacity truncation.
5. A new Triton program loops over each query's tile candidates in 32-source
   blocks, selecting exact outer K by float64 squared distance, then source ID
   for ties. A Torch exhaustive path is available as a CPU/small-case oracle.
6. Selected source/query eigenfunctions are gathered using native methods.
   Differentiable distances use the same native naive/bmm/db-norm operator as
   the configured reference. A private copy of the source heat cache prevents
   overwriting another forward's autograd state.
7. The actual model's `diffuse_heat_kernels` and `forward` execute the original
   heat/filter/Torch-topk/normalization/appearance math. Native autograd handles
   spectral/source/appearance derivatives. Source-face barycentric metadata is
   refreshed through `LiveHeatField.sources()` as before.

This really eliminates dynamic **kernel-to-triangle support CSR**, per-pair
corner/Gram coefficient construction, face/tile task expansion and the resident
emit-record sort in this path. The explicit `torch`/`triton` selectors also
avoid rebuilding a FAISS index. The default `faiss` fallback rebuilds the native
index and bypasses tile selection. The tile variants still perform candidate
selection and pointwise outer-KNN selection;
claiming all selection has been eliminated would be incorrect. No heat-value
cutoff or candidate cap was added. Tile bounds can stay identical when angle,
anisotropy, time or filter parameters change: native outer-K membership depends
only on the source embedding, while shading still responds to all parameters.

There are deliberate limits:

- The old footprints cannot be reused independently. These are new embedding
  bounds for finite raster samples, not spectral conic footprints for a
  continuous surface. Resampling, MSAA or a new camera requires rebuilding.
- GPU bins are padded `tiles * sources` arrays, not a compact production 2DGS
  radix-sort implementation. Allocation is estimated at 48 bytes per tile/source
  including temporaries and rejected above the configured budget (512 MiB by
  default). The Torch nearest oracle additionally has `pixel_batch*N*D`
  working memory; the Triton path avoids that tensor. No silent capping.
- Direct selection is float64 exhaustive L2 with source-ID distance ties;
  native GPU FAISS flat search uses float32 arithmetic and unspecified tie
  order. Very close rankings can differ. The full bounds fallback resolves
  candidate-bound concerns, **not** FAISS tie/order differences. Real GPU parity
  is a release gate, not assumed from CPU tests.
- The new path follows native Torch `topk` semantics. The unchanged resident
  renderer's deterministic entry-order ties and face-union candidate domain
  can differ from native; simultaneous exact agreement with both is impossible
  on the demonstrated counterexamples.
- Native autograd also preserves its derivatives at zero response/clamp
  boundaries. Resident backward masks `w > 0`, so degenerate zero-weight
  boundary gradients are another reason not to assume universal equality.
- Dense bounds and padded ID sorting cost `O(tiles*N)`, with distance work
  proportional to embedding dimension. Per-pixel KNN, float64 sorting, Python
  batch scheduling and repeated native spectral gathers may dominate. Native
  distance recomputation still has its configured float32 cancellation behavior.

## CPU evidence and tolerances

Run the commands below without any GPU allocation. The tests hide CUDA and
patch its initializer to fail. Triton programs execute through their CPU
interpreter; no CUDA compilation or launch is involved. The environment's
Conda `libstdc++` is needed for the existing Numba rasterizer.

```bash
HK_PY=/scratch/user/"$USER"/.conda/envs/hktex/bin/python
export LD_LIBRARY_PATH=/scratch/user/"$USER"/.conda/envs/hktex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$HK_PY" -B -m splatting_script.validation.test_direct_tiles_cpu -v
"$HK_PY" -B -m unittest splatting_script.validation.test_direct_tiles_command -v
```

The suite distinguishes three evidence levels:

The completed run passes **17 numerical/geometry tests and 2 dry-command
tests**, with `cuda_initialized=false`.

- **Intrinsic mathematics:** full cotangent-Laplacian spectra with lumped mass
  on a cylinder and an isometrically folded ribbon; unrolled/folded heat
  operators agree to `1.16e-14`. Two points .08 apart in XYZ but over 4 units
  apart along the ribbon have effectively zero short-time cross influence
  (`-6.49e-16`, numerical roundoff), versus `4.0974` for an intrinsic neighbor.
- **Original source-function comparisons:** unchanged AST-extracted native
  mesh conversions, grid/source/query gathers, heat, filter, topk and RGB
  methods. GPU-only FAISS search is replaced by exhaustive CPU L2, explicitly
  not a FAISS validation. Visible/hidden sources, folded surfaces, two
  disconnected projected lobes, outer-50/top-30, under/over-unit normalization
  and clamp cases compare to native operations. Image checks use float64
  `atol=2e-12, rtol=2e-11`; observed geometry-case RGB error is zero. Hidden
  back-source normalized/weighted response reaches `0.78153` at visible points;
  long-time folded response reaches `0.65154`; short-time folded response max
  is `2.87e-10`. An opaque central strip gives exactly two visible influenced
  components and no heat projected from the occluded cylinder onto that strip.
- **Actual resident/direct Triton programs in CPU interpreter:** unchanged
  resident support/dispatch/forward/backward versus native and new rendering on
  an all-in-outer-KNN, distinct-weight fixture. RGB max difference `5.96e-8`,
  largest parameter VJP difference `1.91e-6`; acceptance `3e-6/3e-5` for RGB
  and `4e-5/5e-4` for VJPs (absolute/relative). New nearest-program tests cover
  ties, multiple source blocks, K=50, partial blocks and candidate recall.

Synthetic grid-varying spectra are used **only for derivative/optimizer tests**,
separately labeled. Every source/appearance parameter has a nonzero derivative:
source XYZ, angle, anisotropy, threshold, sharpness, residual color, base color,
and a test-only live diffusion time. Native-vs-new VJPs use `2e-11/2e-9`;
observed max error is `2.43e-17`. Central finite differences use step `1e-6`
and tolerance `max(2e-7,2e-5*abs(gradient))`. Three Adam steps compare parameters
and first/second moments (`2e-12/2e-11` for parameters, `2e-12/2e-10` for
moments). The original KNN model's diffusion time remains a configuration
constant; the prototype does not silently register a learned time parameter.
An additional float32 three-way Adam comparison uses the actual resident
interpreter, native methods and direct path for three steps: max parameter
difference `1.20e-7`, with `2e-6/2e-5` parameter tolerance and `2e-6/2e-4`
moment tolerance.

Visibility is checked independently with exhaustive ray/triangle intersections,
including triangle ID, depth and barycentrically recovered 3D hit positions.
The fixture also checks zero background. Tests enforce absence of calls to
`support` and `coefficients` in the new path, and FAISS index building in the
explicit tile variant. They reject
stale snapshots, nonfinite sources and excessive bin budgets.

The resident/native counterexamples are measured behavior, not new errors:

| Fixture | Measured consequence |
| --- | --- |
| Native outer=1, three colocated sources admitted by face union | Resident/native max RGB difference `0.061346`; direct agrees with native. |
| Forty tied positive responses, inner=30 | Native CPU Torch selects a different set from resident entries 0–29; max RGB difference `0.015415`; direct agrees with CPU native. GPU Torch/FAISS tie order still needs validation. |

No Converse checkpoint PSNR, native GPU FAISS, CUDA numerical atomic behavior,
Mitsuba intersection parity, GPU GeodesicOpt transitions, real optimizer
trajectory or end-to-end speed was measured here. CPU synthetic PSNR equality
is not evidence of real-workload fidelity or training speed.

## Manual A100 commands and interpretation

These commands are for the user to execute **inside an existing user-owned
A100 allocation**. They never request one. No `srun`/`salloc` command is needed
in this protocol. All output directories are fresh and exclusive.

First run the existing resident gate if that environment has not passed it:

```bash
"$HK_PY" -B -m splatting_script.validation.validate_resident_gpu --execute
```

The new harness is dry unless `--execute` is supplied:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.direct_tiles --mode direct --selection native
```

Create a native run and use its printed `DIRECT_TILES_RUN` directory's
`initial.pt` as the exact common initial state for subsequent runs. Set the
path explicitly; do not automatically pick an arbitrary previous checkpoint.

```bash
/usr/bin/time -p "$HK_PY" -B -m splatting_script.benchmarks.direct_tiles \
  --execute --mode native --warmup 8 --iterations 40 --kernels 1000 --resolution 64

INITIAL=/absolute/path/from/the/native/run/initial.pt

# Separate correctness diagnostic. Stops if direct/native gates fail.
"$HK_PY" -B -m splatting_script.benchmarks.direct_tiles \
  --execute --mode direct --selection native --validate-only --nearest-backend faiss --initial-state "$INITIAL"

/usr/bin/time -p "$HK_PY" -B -m splatting_script.benchmarks.direct_tiles \
  --execute --mode resident --initial-state "$INITIAL" --warmup 8 --iterations 40

/usr/bin/time -p "$HK_PY" -B -m splatting_script.benchmarks.direct_tiles \
  --execute --mode direct --selection native --initial-state "$INITIAL" --warmup 8 --iterations 40

# Separate instrumented runs, one per mode; do not mix these timings with above.
"$HK_PY" -B -m splatting_script.benchmarks.direct_tiles \
  --execute --mode direct --selection native --profile --initial-state "$INITIAL" --iterations 8
```

Repeat the last profiling command with `--mode native` and `--mode resident`.
Repeat matched speed runs in a permuted order and at the intended resolution,
source count and camera mix. Compare `initial_sha256`, config, visibility mode,
initialization, source count, outer/inner K and optimizer settings. Native and
resident are unchanged implementations called by the new harness, not rewritten
surrogates. Resident default support cap is explicitly 128; `--support-cap 0`
tests the uncapped union when it fits the existing 512-per-face limit. Neither
setting makes the union equal native pointwise outer-50.

The `--validate`/`--validate-only` run loads the real HKTex trainer, real spectral asset and real
FAISS. On the first training camera, it records initial images, actual selected
native/direct IDs and weights, actual resident selection tape, raw weighted
heat responses, all source/appearance gradients, original optimizer updates
(Adam and GeodesicOpt as configured) and Adam moments. All methods reset to the
same model/optimizer state. Native/direct outer and positive top-30 sets must
agree; image/gradient/update/moment gates use `atol=3e-5, rtol=3e-4`. The raw
arrays and numerical differences remain available if the gate fails. Query and
source embeddings and native search distances are now saved as well.
`--validate-only` stops after this comparison, before warmup or timing. This is
an initial single-camera gate, not a claim of full training equivalence.
Resident tapes additionally save the actual corner heat/Gram inputs,
kernel/face identities and selected entry IDs for numerical reconstruction;
native/direct tapes save pointwise normalized, distance-weighted heat.

| Measurement | Harness evidence |
| --- | --- |
| Footprint bounds | Profile scope `support::direct_tiles_footprint_bounds`; CPU launch spans and correlated CUPTI kernels. |
| Candidate generation | `support::direct_tiles_tile_candidate_generation`; padded bin bytes and full candidate count per tile in report. |
| Pointwise KNN | `support::direct_tiles_pointwise_outer_knn`; actual new Triton kernel appears in trace. |
| Spectral evaluations | Direct count `visible_pixels*min(outer_k,N)`; actual gather/heat kernels under `spectral::direct_tiles_spectral_shading`; extra work for optional trace is outside measured iterations. |
| Forward/backward | `training::direct_forward`, `training::native_source_and_forward`, resident support/coefficient scopes, and `training::backward`. |
| CUDA execution | `cuda_activity.json` correlates actual CUPTI kernel durations, memory operations and launch phases; parent and nested scopes overlap and must not be added. |
| CPU scheduling/synchronization | Whole host enqueue wall, process CPU, final device-drain wait per iteration; profiler CPU/driver calls for finer attribution. These are not isolated GPU kernel durations. |
| Memory | Per-iteration allocator deltas and training peak allocated/reserved bytes; static setup and post-timing diagnostics are separated. |
| End-to-end iteration | Boundary-synchronized iteration walls include source state, dynamic support/bounds/bins, spectral work, forward/loss/backward and configured optimizer/scheduler steps. |
| Full process | External `/usr/bin/time -p` includes imports, setup, warmup, output and exit; report's elapsed field excludes final report serialization and is labeled accordingly. |
| Correctness/PSNR | Final RGB arrays and PSNR against the original UV target for all four matched cameras, plus same-state native image differences. Visibility arrays saved as IDs/depth/barycentrics. |

Warmup covers all four cameras, performs backward and optimizer steps, then
resets model and optimizer state before measurement. Camera snapshots are
rebuilt after `load_state_dict` writes mesh buffers. This warms ordinary kernel
specializations and allocator caches; inspect traces for compilation/first-use
events and increase warmup if parameter-dependent support sizes still introduce
new specializations. The default uses fixed visibility reuse. Run **all three**
with `--visibility fresh` to include per-iteration CPU rasterization/preparation
instead. Compilation/initialization belong in full-process time, never silently
in steady-state claims. `--validate`/`--profile` runs are diagnostic processes.

The expected opportunity is removal of face support/CSR, coefficient tables,
dynamic face/tile expansion and record sorting. The main risks to performance
are looser tile bounds over folds, padded bins, float64 nearest selection and
more spectral evaluation: the baseline reuses three corner evaluations per
retained kernel/face, while direct shading gathers spectra per visible
point/outer-neighbor pair. This may be slower. Only matched, repeated real
training iteration and full-process measurements with quality/gradient gates
can decide whether support elimination is a net gain.
