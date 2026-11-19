# HKTex (Original) versus HKTex Splat (Ours) benchmark profiler

**Output default:** workflows now retain one Markdown report. Use
`--output comparison.md`, or a new directory containing just `report.md`.
Add `--keep-artifacts` to execution commands below when you need the raw
profiles, images, checkpoints, JSON/CSV and plots described in this guide.
Compact mode removes its private temporary files/caches after collecting the
statistics and failure summaries; all correctness gates remain enabled.

See [the matched benchmark protocol](HKTEX_VS_HKS_BENCHMARK.md) for the two
retained workloads, metric definitions and current verification status.

> Current interface: native/original and hybrid/2dgs only. Projected cutoff,
> alternative selection modes and the historical forward trainer are retired.
> See [the current command guide](../splatting_script/README.md) and
> [pipeline-only cleanup](HKS_PIPELINE_ONLY_CLEANUP.md) for compatibility/recovery.

This extends `compare_backends`, `renderer_comparison`, `training_research`, and the
existing report readers. The renderer, heat-kernel equations, native FAISS
implementation, support assertions, conservative 5,000/10,000-source fallback,
backward kernels, and geodesic optimizer are unchanged.

## One workflow on Grace's A100

Use the existing user-owned Grace A100 Slurm allocation. Run these commands on
its allocated compute node, with `SLURM_JOB_ID` inherited. The tools check the
allocation/GPU and never submit `sbatch`, request `salloc`, or enlarge budgets.

```bash
cd /path/to/your/hktex/checkout
HK_PY=${HK_PY:-python}
# If your Conda CUDA libraries require it:
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset TRITON_INTERPRET

# Dry preview: standard library only, no Torch import, output files, or GPU work.
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow

# Embedded real-scene RGB/gradient/optimizer gate, in the existing allocation.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --execute --validate-only --support-selector fused

# Choose a new directory. Defaults prioritize 256px, 5,000 sources, seed 0.
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --execute --output experiments/converse_heat_kernel_splatting/results/profile_256_5000_new
```

Each kernel count gets one seed-labelled initial state, shared by both backends
and all requested resolutions. Before any timed job, each renderer executes an
untimed RGB/loss/finite-gradient/optimizer-reset-replay preflight. A paired gate
checks saved initialization, source/assets/settings, physical GPU UUID, exact
camera/raster/target/mask receipts and common-native-evaluator agreement. Initial
RGB/loss differences between the two deployed renderers are recorded, not rejected.
Missing or failed preflights block both arms of that workload. The existing
loss-consistency gate compares the actual step with a gradient-enabled reference
at the same initialization/RNG. The no-grad evaluation comparison is a separate
recorded diagnostic; see [the execution-mode fix](PREFLIGHT_LOSS_MODE_FIX.md).
The existing
independent `--validate-only` oracle remains a separate, capacity-bounded diagnostic.
Each backend then gets a four-update capacity/
memory pilot (after eight warmup updates). A failed pilot is recorded and its
long jobs are explicitly marked skipped. The other backend/configurations can
still run; missing partners never produce a speedup ratio. Runtime capacity and
OOM guards remain authoritative; conservative geometry estimates are not proof
that actual live support will fit.

For each feasible backend, the workflow then runs:

1. **Speed/quality:** 1,000 synchronized research updates, without profiling or
   saved-tensor instrumentation; training/held-out quality every 250 updates
   through the common native field, archival snapshots, parameter/gradient
   health, source/config/model hashes, and JIT miss accounting.
2. **Diagnostic profile:** a separate process from the identical initial state,
   with a same-backend observer-enabled RGB/all-parameter-VJP correctness gate,
   eight warmup updates, model/optimizer/scheduler/RNG reset, 40 unprofiled
   trajectory updates, then eight actively profiled updates. No optimizer step
   is added or removed from either trajectory. The active sample covers all
   four training cameras twice. These measurements do not enter speed ratios.

The defaults keep cap 128, Triton coefficients/emission,
runtime counts, reused visibility, and highest float32 matmul precision.
Projected footprint pruning is unavailable; zero-cutoff fields in new records
are fixed historical-comparison metadata.
Above 4,096 sources, plans use the new tiled Triton selector without automatically
allowing a legacy fallback. Profiles distinguish tiled merging and threshold/output
from distance and CSR work. See [the selector notes](TILED_SELECTOR.md) for the
deterministic wide-path tie rule and GPU validation still required.
`--debug-support` remains a separate diagnostic; it cannot be combined with
the pipeline profiler because its extra downloads would confound attribution.

The installed training/compare aliases and historical backend names still
work. Individual pipeline profiles can also use `--backend original`/`2dgs`.

## Individual profiles and scaling

For an existing matching seed-0, 5,000-source initialization:

```bash
INITIAL=/absolute/path/to/initial.pt
COMMON=(--execute --initial-state "$INITIAL" --seed 0 --kernels 5000 --resolution 256
  --support-selector fused --coefficient-backend triton
  --runtime-counts --support-cap 128 --matmul-precision highest
  --pipeline-profile --profile-validate --warmup 8 --iterations 48 --profile-start 40 --profile-steps 8)
"$HK_PY" -B -m splatting_script.experiments.compare_backends --backend native "${COMMON[@]}"
"$HK_PY" -B -m splatting_script.experiments.compare_backends --backend hybrid "${COMMON[@]}"

# Preview the full 64/128/256/512 × 1000/5000/10000 matrix.
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow --scaling
# Execution is explicit and potentially long; use a new directory.
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow --scaling \
  --execute --output experiments/converse_heat_kernel_splatting/results/profile_scaling_new
```

Use `--resolutions` and `--kernel-counts` to choose a subset, `--iterations` to
change the speed budget, and `--profile-start`/`--profile-steps` to select a later
representative diagnostic window. `--visibility fresh` includes CPU raster and
camera preparation per update; the default explicitly reports visibility reuse.
Fixed triangle embedding/centroid/radius preparation is measured separately in
setup, not invented as a recurring update cost. Budgets change only through
explicit `--max-buffer-mib`/`--max-contributions` arguments, recorded for both arms.

## Artifacts and offline reports

`workflow.json` records commands, process walls, return codes, run directories,
and logs. Initialization failures, OOM, capacity guards, invalid support, failed
pilots, unsupported configurations, and interruptions remain explicit. Geodesic
warnings are counted from whole-process logs as failure **events**, including
warmup; they are not counts of unique failed sources or a guarantee of tracing
success when no warning appears. Each child starts a fresh process so a CUDA
assert cannot poison the next configuration's context.

Each diagnostic child writes:

| Artifact | Contents |
|---|---|
| `profile_validation.json` | Actual same-device observer-enabled RGB/VJP gate, differences and existing CUDA tolerances |
| `trace.json` | CPU operators/ranges, CUPTI kernels/copies, shapes and allocator events |
| `pipeline_observations.json` | Host/process CPU durations, CUDA event intervals, allocator snapshots, setup, saved storage and work counts |
| `pipeline_profile.json` | Correlation-based inclusive/exclusive component analysis, top-five CPU/GPU costs, memory and work data |
| Existing `cuda_activity.json`, `candidate_activity.json`, `stage_activity.json`, `jit_cache.json` | Preserved legacy readers and diagnostics |

The workflow automatically creates `report/report.md`,
`report/benchmark_report.json`, and `overall.csv`, `component_timing.csv`,
`memory.csv`, `work_counts.csv`, `quality.csv`, `scaling.csv`, `failures.csv`.
Standalone PNG plots show update time and peak allocated memory against each
scaling axis, forward/backward/optimizer CUDA event intervals, candidate/spectral
GPU costs, and native-field training/held-out PSNR against updates. A single
size produces points rather than an invented scaling trend. Missing data is
marked N/A and no measured-data line is fabricated.

Regenerate a report offline without importing CUDA or running training:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_training report \
  /path/to/workflow.json --output /path/to/new_report_directory
# --no-plots keeps this report path standard-library only.
"$HK_PY" -B -m splatting_script.benchmarks.profile_training_report \
  /path/to/native_profile_run /path/to/hybrid_profile_run --output /path/to/profiles.json
```

The older controlled-run/research comparison commands are unchanged. Profile
controls are checked against the corresponding speed run: initialization,
source/asset/config hashes, precision, cameras, optimizer configuration, seed,
resolution, kernel count, common capacity/workload settings, and GPU UUID.
Unmatched or absent profiles cannot populate a speed row's component results.
Profiles and support-debug runs are rejected as speed evidence.
Speed ratios additionally require matching completed update budgets, quality
evaluation steps and passing parameter/gradient health checks. Scaling rankings
check shared controls while allowing only the selected size axis (and its
necessary initialization/shape or camera changes) to differ.

## Interpreting measurements

| Measurement | Meaning and limits |
|---|---|
| Synchronized update wall | Existing full-update timer through final device synchronization; includes encountered JIT misses, excludes evaluation/checkpoints/setup |
| Host enqueue wall | Host interval scheduling the update, including blocking APIs; not GPU execution or pure CPU utilization |
| Process CPU time | CPU across process threads; it can exceed elapsed wall time |
| CUDA event interval | Current-stream elapsed interval, including host-induced idle gaps; external FAISS streams require CUPTI attribution |
| CUPTI kernel-duration sum | Actual kernel execution intervals; concurrent streams can overlap |
| Activity union | Union of kernel/copy/memset intervals, counting overlap once |
| Observed GPU gaps | Span minus activity union within correlated updates; does not measure whole-device utilization or other processes |
| CPU bottleneck ranking | Exclusive self wall from nested trace ranges/operators, including waits; observer bookkeeping is separately labelled/excluded |
| GPU bottleneck ranking | Exclusive leaf kernel durations, excluding diagnostic reductions; parent/child ranges are not added |

Native FAISS construction/search, source gathering/heat, and distance weighting
are observed around original callables. Native top-k and RGB costs use CPU
operator/launch ownership inside original shading: RGB is its residual weighted
color preparation/reconstruction after heat/filter/top-k. Native spectral
backward is attributed only where autograd sequence and forward-thread IDs
link to measured forward operations; unmatched work remains in total backward.
Fused support threshold/cap/mask/output remains one indivisible GPU stage.
Unavailable subdivisions are N/A, not zero. No GPU ownership is guessed from a
later timestamp, and uncorrelated launches remain visible as unattributed.

Stage GPU percentages divide exclusive kernel sums by diagnostic update wall;
they are not an additive wall-time decomposition. Full distributions provide
sample count, mean, median, nearest-rank p95 and maximum. JIT miss walls include
compiler **or** disk-cache loading; isolated compiler time remains unmeasured.

Memory values distinguish measured unique tensor-storage bytes, live allocator
snapshots, global peaks, allocator events, and logical buffer bytes derived from
actual shapes/dtypes. Model parameters, gradients, source IDs, spectral assets,
optimizer state, native index tensors and camera/target storage are reported.
Saved-tensor hooks report live storage retained for backward at the forward/loss
boundary, including its overlap with known resident components. They do not
copy tensor data or retain a second graph. Categories can alias; neither these
bytes nor per-stage global high-water marks should be summed. FAISS/driver/JIT
allocations outside PyTorch's allocator remain unattributed rather than being
assigned to spectral assets or activations. Memory rankings are measured known
components, not a complete whole-device ownership claim.

Hybrid counts include visible faces, support pair counts/mean/max, cap
utilization, CSR bytes, expanded face/tile incidences, queue count/capacity,
actual emitted records, contributions per visible pixel, selected/positive
weights and top-30 discarded percentage. Counts are sampled from real buffers
with separately labelled diagnostic reductions. Discarded percentage compares
emitted valid records to retained valid entries; positive-weight counts are
reported separately, so zero-weight suppression is not confused with ranking.

The automatic optimization targets name the largest measured exclusive GPU
components. They do not predict speedups or infer cache/bandwidth causes from
timings alone. Scaling rankings require at least two completed sizes. The
existing 3.03× preliminary result is historical; this implementation introduces
no new GPU performance claim.

## Benchmark correctness gates

The workflow retains `--profile-validate`: its embedded observer-on/off gate
checks RGB and parameter VJPs before collecting a diagnostic profile. The
gate now uses the exact foreground training loss supplied by the harness,
including its target, mask and pixel-mean reduction. The previous unnormalized
`RGB.square().sum()` remains the validation helper's compatibility default.
An independent unprofiled repeat must pass the same tolerances. Failures save
`profile_validation.json` with parameter names, shapes, mismatched-element
counts, observer differences and baseline-repeat differences before raising;
the gate never accepts baseline noise as permission for a larger error.
See [the smoke-run failure investigation](PROFILE_VALIDATION_FIX.md).
The separate `compare_backends --validate-only` gate checks renderer comparisons
and optimizer updates. Both execute only with `--execute` in an allocation;
neither supplies uninstrumented speed evidence.

Standalone CPU/synthetic GPU suites were retired after a fresh CPU run. Their
results and recovery instructions are in the [test cleanup report](TEST_CLEANUP_REPORT.md).
Command previews do not establish CUDA event accuracy, CUPTI/FAISS attribution
coverage or GPU numerical parity. Those require actual A100 gates and profiles.
