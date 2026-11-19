# Native HKTex versus hybrid spectral heat-kernel splatting

> Current interface: native/original and hybrid/2dgs only. Projected cutoff,
> alternative selection modes and the historical forward trainer are retired.
> See [the current command guide](../splatting_script/README.md) and
> [pipeline-only cleanup](HKS_PIPELINE_ONLY_CLEANUP.md) for compatibility/recovery.

This experiment compares native HKTex texture calculations with the experimental
kernel-driven hybrid renderer. It is a controlled **unlit, fixed-density,
encoded RGB/255** optimization on Converse, not full Mitsuba inverse rendering.
The internal resident control remains a correctness/debug reference, not an
exposed backend or research arm.
No GPU experiment in this protocol was executed during preparation.

## The two unchanged calculation paths

`compare_backends --backend original` maps to `renderer_comparison --mode native`.
The measured step calls `HeatKernelTextureKNN.prepare_kernels`,
`EigenAlboInterpolationKNN.build_kernel_graph`, native FAISS flat L2 pointwise
outer-50 search, native spectral gathers/diffusion/filtering, and native top-30
normalized RGB. Queries are still batched in 256-point chunks. Autograd and the
configured Adam plus GeodesicOpt updates execute normally.

`--backend 2dgs --support-selector fused` calls
`DirectTileRenderer._hybrid`: live source state, current triangle support,
`ResidentTopology` indices and shared spectral coefficients,
`triangle_tile_work`, `_hybrid_emit`, top-30 reduction/reconstruction, and the
existing inherited resident VJP plus coefficient VJP. Its forward owns a
kernel/triangle/tile queue; it does not call the resident scene renderer.
The word "2dgs" names this scheduling experiment. There are no ordinary Gaussian
primitives, opacity, alpha compositing, or replacement of intrinsic diffusion.

Both arms use the same saved parameters, fixed mesh/eigen archive, four training
cameras, RGB targets, channel-summed squared-error loss averaged over foreground
pixels, optimizer configuration and round-robin schedule. Research mode uses an
explicit common matmul precision (`highest` by default), fixed 1,000 kernels,
cap128, cutoff0, Triton coefficients and runtime counts. The common KNN diffusion
time remains a configuration constant; positions, colours, angles, anisotropies,
thresholds, sharpnesses and mean colour are trained. Density controllers and
importance sampling are disabled by the existing controlled loader.

Warmup performs real updates across all four views, then reloads the initial
model, clears gradients, recreates optimizers/schedulers, and restores Python,
NumPy and Torch CPU/CUDA RNG states. The measured schedule restarts with top view.
Initial and periodic evaluation also preserve RNG states. Evaluation and health
checks do not change optimizer steps or scheduler steps.

## Research mode and outputs

`--research` is opt-in. Ordinary benchmark defaults/calculations remain intact.
Research mode requires original or hybrid and rejects mixing validation or
profiling into a convergence run. Use separate ordinary `--profile` runs.

Defaults: progress every100 updates, quality every250, snapshots every1000,
parameter/gradient checks every100 plus first/final updates. Initial/final
quality and a final snapshot are always saved, including a 50-update pilot.
`--check-every 1` checks every pilot update outside its timer.

Every update synchronizes before timing and after the complete optimizer step.
The timer includes forward/loss/backward/updates and any newly encountered
compilation. Evaluation, checkpoint writing, health checks and log I/O execute
outside it. Warmup wall includes compilation; isolated compilation is unknown.
`completion.json` includes setup and overhead through final report writing;
external `/usr/bin/time -p` also measures interpreter entry/exit. Do not add
overlapping JIT wall/CPU/CUDA intervals to training time.

Each `DIRECT_TILES_RUN=...` directory contains:

| File | Contents |
|---|---|
| `research_manifest.json` | Commit/dirty status, actual source and asset SHA256s, initial hash, seed, cameras, configs, optimizer/LRs, precision, Torch/CUDA/Triton, GPU UUID and live memory at manual execution |
| `initial.pt` | Initial model, kernel count, initialization seed and model hash |
| `iterations.jsonl`, `iterations.json` | Update/camera, loss, synchronized seconds, cumulative training seconds, JIT miss accounting; JSONL streams during training |
| `quality.json` | Step0, every250 and final quality for six cameras; MSE/PSNR per camera and equal-camera aggregate |
| `health.json` | Finite parameters/gradients, missing/nonzero gradients, actual parameter changes, configured PGD bounds |
| `checkpoint_001000.pt`, etc. | Model, optimizer and scheduler states, completed update, next camera/schedule, Python/NumPy/Torch CPU/CUDA RNG, seed, hashes and cumulative training time |
| `*_rgb.npy`, `*_rgb.png` | Final image using the trained arm's renderer |
| `*_native_rgb.npy`, `*_mask.npy`, `*_visibility.npz` | Final common native-field images and shared masks/visibility |
| `report.json`, `completion.json`, `jit_cache.json` | Performance/quality crossings, overhead, peak training GPU memory, validity and completion |
| `failure.json` | Failure and number of successfully logged updates if a gate fails |

Snapshots are **archival only**. No resume interface is advertised. Research
mode rejects passing a snapshot as an initialization; doing so would discard
optimizer/scheduler progress. A failed experiment must restart from its shared
initialization. There is no partial resume implementation.

Quality has two families. `native_field` evaluates both learned models through
the same native field implementation and is the primary common evaluator.
`trained_backend` measures the image delivered by each arm's own selection
policy. Report both. Training loss equals three times per-channel MSE on the
same samples; equal-camera aggregate PSNR is computed from mean camera MSE,
not mean PSNR or unequal foreground-pixel pooling.

The two additional views (`heldout_left`, `heldout_right`) are fixed, unseen
camera views with targets sampled from the same mesh texture. They are not an
independent dataset or unseen-object generalization test. Native outer50/top30
and hybrid triangle cap128 selection differ. Passing fused/legacy and
hybrid/resident correctness gates does not establish native parity.

## Grace commands: common shell setup

Run manually in an existing Grace A100 allocation. These commands allocate no
resources or submit jobs. Keep paired runs on the same GPU where possible.

```bash
set -euo pipefail
cd /path/to/your/hktex/checkout
HK_PY=${HK_PY:-python}
# If your Conda CUDA libraries require it:
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset TRITON_INTERPRET
RESULTS=experiments/converse_heat_kernel_splatting/results
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
INITIAL=$RESULTS/direct_tiles_direct_20261105T044854_349740Z_0529a6a42c/initial.pt
COMMON=(--execute --research --support-selector fused
  --splat-backend triton --coefficient-backend triton --runtime-counts
  --support-cap 128 --visibility reuse
  --matmul-precision highest --warmup 8 --progress-every 100
  --eval-every 250 --checkpoint-every 1000 --psnr-thresholds 20 25 30)

# Arguments: backend updates initialization seed label [extra flags].
# Sets LAST_RUN without capturing the process's complete stdout in a variable.
run_case() {
  local backend=$1 updates=$2 init=$3 seed=$4 label=$5
  shift 5
  local log="$RESULTS/${label}_${backend}_${STAMP}.log"
  /usr/bin/time -p "$HK_PY" -B -m splatting_script.experiments.compare_backends \
    --backend "$backend" "${COMMON[@]}" --iterations "$updates" \
    --initial-state "$init" --seed "$seed" --kernels 1000 --resolution 64 \
    "$@" | tee "$log"
  LAST_RUN=$(awk -F= '/^DIRECT_TILES_RUN=/{p=$2} END{print p}' "$log")
  test -s "$LAST_RUN/report.json"
}

# One labelled initialization per seed/kernel count; both arms reuse its file.
make_initial() {
  local seed=$1 kernels=$2
  local log="$RESULTS/init_n${kernels}_s${seed}_${STAMP}.log"
  "$HK_PY" -B -m splatting_script.experiments.compare_backends \
    --backend original "${COMMON[@]}" --initialization-only \
    --seed "$seed" --kernels "$kernels" | tee "$log"
  NEW_INITIAL=$(awk -F= '/^RESEARCH_INITIAL_STATE=/{p=$2} END{print p}' "$log")
  test -s "$NEW_INITIAL"
}
```

The historical unlabelled initialization is accepted only with seed0. Other
seeds require labelled files. The comparator rejects different seeds that share
the same model hash. `make_initial` is GPU initialization only when run manually;
it does not train, warm up, or request a GPU.

## Stage A: 50-update pilot

```bash
run_case original 50 "$INITIAL" 0 pilot --check-every 1
PILOT_NATIVE=$LAST_RUN
run_case 2dgs 50 "$INITIAL" 0 pilot --check-every 1
PILOT_HYBRID=$LAST_RUN
"$HK_PY" -B -m splatting_script.benchmarks.compare_training \
  "$PILOT_NATIVE" "$PILOT_HYBRID" --require-iterations 50 --require-same-gpu \
  --output "$RESULTS/pilot_comparison_${STAMP}.json" \
  --plot-dir "$RESULTS/pilot_plots_${STAMP}"
```

Before continuing check: COMPLETE, exactly50 updates, matching manifests and
initial hashes, restored warmup state, finite losses/parameters/gradients,
actual parameter changes, initial/final PSNR, six camera results, final snapshot,
and no hybrid selector fallback. Examine deployed-versus-native differences;
do not gate on exact native/hybrid output equality.

## Stage B: 1,000 updates, after the pilot passes

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_training \
  "$PILOT_NATIVE" "$PILOT_HYBRID" --require-iterations 50 --require-same-gpu >/dev/null
run_case original 1000 "$INITIAL" 0 preliminary --check-every 100
PRE_NATIVE=$LAST_RUN
run_case 2dgs 1000 "$INITIAL" 0 preliminary --check-every 100
PRE_HYBRID=$LAST_RUN
"$HK_PY" -B -m splatting_script.benchmarks.compare_training \
  "$PRE_NATIVE" "$PRE_HYBRID" --require-iterations 1000 --require-same-gpu \
  --output "$RESULTS/preliminary_comparison_${STAMP}.json" \
  --plot-dir "$RESULTS/preliminary_plots_${STAMP}"
```

Expect quality at0/250/500/750/1000 and snapshot1000. The comparator checks
completion, actual updates, validity, paired inputs/settings and scheduled
snapshot existence. It intentionally sets no arbitrary PSNR acceptance target;
inspect quality before authorizing the long experiment.

## Stage C: 10,000 updates and three matched seeds

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_training \
  "$PRE_NATIVE" "$PRE_HYBRID" --require-iterations 1000 --require-same-gpu >/dev/null
FULL_RUNS=()
for seed in 0 1 2; do
  if [ "$seed" -eq 0 ]; then
    SEED_INITIAL=$INITIAL
  else
    make_initial "$seed" 1000
    SEED_INITIAL=$NEW_INITIAL
  fi
  order='original 2dgs'
  if [ "$seed" -eq 1 ]; then order='2dgs original'; fi
  for backend in $order; do
    run_case "$backend" 10000 "$SEED_INITIAL" "$seed" "full_s${seed}" --check-every 100
    FULL_RUNS+=("$LAST_RUN")
  done
done
"$HK_PY" -B -m splatting_script.benchmarks.compare_training \
  "${FULL_RUNS[@]}" --require-iterations 10000 --require-same-gpu \
  --output "$RESULTS/full_comparison_${STAMP}.json" \
  --plot-dir "$RESULTS/full_plots_${STAMP}"
```

Plots are PNG and PDF, one seed/workload per figure: PSNR versus optimizer
updates and PSNR versus cumulative timed training, each for the common native
evaluator and deployed renderer. Lines connect observed samples only. PSNR
threshold times are first actual sampled crossings; unreached thresholds are
`null`, never extrapolated. Report seed spread instead of choosing the best seed.

## Scaling review and commands

CPU preparation inspected the actual Converse mesh (75,000 faces), spectral
asset (1,930,889,350 bytes), and rasterized all six cameras at every resolution.
Saved 64px reports identify A100-PCIE-40GB and peaks of about3.95GiB native and
1.93GiB hybrid allocated memory. Current free device memory is unknown without
your manual run; research manifests record it after setup.

| Resolution | Conservative hybrid guard maximum | Native peak projection | Preparation status |
|---|---:|---:|---|
|64|96MiB|3.9GiB|Memory pilot candidate|
|128|196MiB|10.0GiB|Memory pilot candidate|
|256|412MiB|34.1GiB|Memory review required; limited A100 headroom|
|512|1163MiB|130.6GiB|Not prepared as a matched native/hybrid run|

The guard is an existing logical buffer check, not total GPU memory. Native
projections scale the observed native-versus-hybrid 64px peak difference by
foreground pixels and are **not measurements or fit guarantees**; unknown
workspace and kernel-count growth remain. The default512MiB/16,777,216-record
budgets also fail conservative512px bounds. Do not silently raise budgets or
truncate contributions. A256px pilot needs a memory review;512px needs a design
for native autograd memory before a fair matched run can be prepared.

The current selector uses bounded Triton source tiles above 4,096 kernels;
source count alone no longer triggers a legacy fallback. Exact distance ties
use source-ID order on that path, which can differ from historical PyTorch
fallback results. Earlier measurements in this guide describe their recorded
source versions. See [the tiled selector](TILED_SELECTOR.md) for the new path,
CPU checks and outstanding GPU validation. Unsupported-input fallbacks still
require explicit acknowledgement; source count no longer adds it automatically.

```bash
# CPU-only live geometry/capacity plan; no Torch import or GPU probing.
"$HK_PY" -B -m splatting_script.benchmarks.compare_training --scale-plan \
  --output "$RESULTS/scaling_plan_${STAMP}.json"

# Six candidate workloads: memory/correctness pilots first, then inspect results.
for kernels in 1000 5000 10000; do
  SCALE_INITIAL=$INITIAL
  if [ "$kernels" -ne 1000 ]; then
    make_initial 0 "$kernels"
    SCALE_INITIAL=$NEW_INITIAL
  fi
  for resolution in 64 128; do
    run_case original 50 "$SCALE_INITIAL" 0 "scale_n${kernels}_r${resolution}" \
      --kernels "$kernels" --resolution "$resolution" --check-every 1
    SCALE_NATIVE=$LAST_RUN
    run_case 2dgs 50 "$SCALE_INITIAL" 0 "scale_n${kernels}_r${resolution}" \
      --kernels "$kernels" --resolution "$resolution" --check-every 1
    SCALE_HYBRID=$LAST_RUN
    "$HK_PY" -B -m splatting_script.benchmarks.compare_training \
      "$SCALE_NATIVE" "$SCALE_HYBRID" --require-iterations 50 --require-same-gpu \
      --output "$RESULTS/scale_n${kernels}_r${resolution}_${STAMP}.json"
  done
done
```

For each workload that passes, use the same `--kernels`, `--resolution`,
initialization and fallback acknowledgement with StageB/C. Do not infer scaling
benefit from the pilot or extrapolate64px speedups. Report median/p95, cumulative
time, peak allocated/reserved memory and PSNR for every matched workload.

Collect stage CUDA timing in **separate** ordinary runs, using the same
initialization and preceding update schedule. Example for the primary workload:

```bash
for backend in original 2dgs; do
  "$HK_PY" -B -m splatting_script.experiments.compare_backends \
    --backend "$backend" --execute --initial-state "$INITIAL" --seed 0 \
    --kernels 1000 --resolution 64 --warmup 8 --iterations 100 \
    --support-selector fused --splat-backend triton \
    --coefficient-backend triton --runtime-counts --support-cap 128 \
    --visibility reuse --matmul-precision highest --profile --profile-start 40
done
```

For scaled profiles substitute the paired workload's count/resolution and
initialization in both processes. `stage_activity.json` adds launch-correlated
kernel durations/counts for support selection, queue construction, hybrid splat
emission/top30 and spectral coefficient forward/backward. `trace.json`,
`candidate_activity.json` and `cuda_activity.json` remain available. Nested
scopes overlap. Native has no triangle support/queue/coefficient kernel stage;
compare its pointwise native forward/backward rather than treating missing
stages as free equivalent work. Profiles are diagnostic and are not convergence
or steady-state speed evidence.

## What this can establish

These experiments can measure convergence and time to sampled quality for two
selection/scheduling architectures on a controlled textured mesh. They can
test whether hybrid scaling helps at the configurations actually completed.
They cannot establish exact native-gradient parity, preserved quality before
measurement, full Mitsuba training acceleration, alpha-composited Gaussian
splatting performance, or generalization beyond the tested scene and cameras.
