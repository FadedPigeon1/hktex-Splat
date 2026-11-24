# Exact spectral-distance layout study

The current direct-adapter A100 profile attributes about 14.92 ms/update to
distance tables, 3.35 ms to cutoff/output and 18.94 ms to overall triangle
support. Corner evaluation is 4.19 ms and splatting 0.54 ms. These are
synchronized inclusive diagnostic timings; nested rows must not be summed.

This study compares source batches of 8, 32 and 64 pairs with 2-face/16-source,
4-face/8-source and 4-face/16-source tiles. Multiple faces can reuse loaded source
values, trading fewer redundant loads/programs against additional live registers
and layout conversions. Neither larger blocks nor lower code size guarantees
better GPU time. The techniques follow the data-reuse guidance in the
[Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
and [NVIDIA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
This implementation is explicit elementwise FP64 distance reduction, not a
matrix-multiplication distance identity or a claimed implementation of GEMM.

`support_distances_blocked` in `triton/resident_gpu.py` is new. The old kernels,
heat mathematics, training, optimizer and backward are unchanged. The production
whole-row selector now calls the blocked kernel in the narrow case below.
`splatting_script/benchmarks/spectral_distances.py` provides six matched layouts,
CPU interpreter checks, offline SM80 compilation and an explicit short CUDA
microbenchmark. No new external dependencies or standalone test suite was added.

## Run on A100

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.benchmarks.spectral_distances \
  --execute --output "experiments/distance_layouts_$(date +%Y%m%d_%H%M%S).md"
```

This bypasses the importance pool, mesh loading and training. It benchmarks
128x5000 and 512x5000 tables at spectral dimension 64, using contiguous inputs
as in the renderer. Each layout must match reference distances and candidate
rows exactly before timing. CUDA events cover ten launches per sample, seven
samples, with compilation excluded. Only one Markdown report is written;
normal compiler caches may be populated. GPU outputs are validated independently
for each layout. No auto-tuning or default change occurs.

Use the same command with `--cpu` for strided-input/tail checks across dimensions
2, 31, 32, 64, 65 and 256, or `--compile-sm80` for driver-free compilation/resource
inspection. CPU speed and compile-resource counts do not predict A100 speed.
CPU sqrt tests are not CUDA RN sqrt execution; actual CUDA checks remain required.

## Selection criteria

Select a candidate only after repeatable CUDA improvement at both table sizes.
Repeat with `--reverse` and fresh output to check timing-order effects. A microbenchmark
winner still needs integration into the selector plus full-model RGB/gradient
gates and unprofiled training measurements. Check scratch/occupancy costs; retain
the original path for unsupported dimensions or constraints. This study does
not claim a winner without GPU measurements.

Recovery: `/tmp/hks-distance-study-_x3v8sxi` contains the previous GPU source and
Git index listing. Historical results, original HKTex and staging were preserved.
The current session has no accessible NVIDIA driver and launched no GPU jobs.

## Completed checks

All 36 CPU interpreter combinations passed exact distance and candidate-row
comparisons with the existing source8 kernel, including strided inputs and tails.
All six dimension-64 variants compiled offline for SM80:

| Layout | Cubin bytes | Compiler-reported shared bytes |
|---|---:|---:|
| source8 | 32608 | 2048 |
| source32 (current common path) | 96864 | 8192 |
| source64 | 180576 | 16384 |
| face2_source16 | 94816 | 4096 |
| face4_source8 | 93792 | 2048 |
| face4_source16 | 176736 | 4096 |

The face4/source8 candidate reduces reported shared storage relative to source32,
but actual register allocation, occupancy and GPU timings are not established by
this table. No candidate was selected as the production default.

An additional NumPy cancellation check rejected the squared-norm/dot-product
identity as a bit-exact replacement: almost coincident vectors near one in 64
dimensions produced differing distances, including zeros. This adversarial check
does not measure real workload frequency; it demonstrates that algebraic
equivalence does not preserve the required floating point result.

The first offline compilation attempted the environment's read-only home cache.
The benchmark now defaults its Triton cache to a user-specific temporary directory
while honoring an explicitly configured cache directory. Compilation then passed.
Original GPU function bodies, syntax and the Git index were checked unchanged.
The no-execution preview starts no GPU job. CPU reports/compiled resources are in
the recovery directory above. Full GPU distance/support checks and timing remain
pending; after integration, full-model gradient gates must run again.

## Integration after two A100 microbenchmarks

User-run forward/reverse-order reports passed exact GPU distance/support checks
for every candidate. For 512x5000 tables, source32 measured 2.411725/1.900954 ms;
face4/source16 measured 0.839782/0.817254 ms. At 128x5000, source32 measured
0.885453/0.489984 ms and face4/source16 0.368947/0.368538 ms. Source-only timing
was order-sensitive. Blocked timings were more consistent; the lowest-shared-
memory candidate was not consistently fastest. These results select face4/source16
for integration, not a full-training performance claim.

`FusedResidentSupportGeometry._row_radix_candidate_rows` now uses that kernel
only for CUDA float32 sources with a 64-dimensional embedding, after all existing
row-radix audit/capacity guards. It retains FP64 intermediates, explicit reduction,
RN sqrt and disabled FP fusion. CPU, other source dtypes/dimensions, small source
tables and the scalable tiled path retain their previous dispatch. Buffer budgets,
candidate cutoff/order, spectral heat and backward are unchanged.

The validation receipt now records `aggressive_support.actual_distance_layout`;
require `face4_source16` when verifying the eligible 5000-kernel setup. Support
stats also identify the distance layout. A CPU launch-spy regression executed the
actual row-driver body for CPU and CUDA-shaped float32/float64 inputs and dimensions
32/64/65, checking unchanged fallbacks and budget-limited 3/3/1-row launches.
Those are dispatch tests, not GPU execution. Existing GPU function bodies were
left unchanged. Full-model A100 RGB/VJP verification and unprofiled training
timing remain required for this integrated launch.
