# HKTex (Original) versus HKTex Splat (Ours) commands

Workflow execution defaults to **one Markdown file**: `--output comparison.md`.
Bare Markdown filenames are saved under `experiments/`; explicit paths keep
their specified location.
New logs, snapshots, images, traces and compiler caches are temporary and cleaned
up afterward. Existing results are preserved. Add `--keep-artifacts` with a new
output directory for full research/debugging artifacts. Rendering, measurements
and correctness gates are the same in both output modes.

The main command is `experiments.compare_backends`; execution uses
[`benchmarks/renderer_comparison.py`](benchmarks/renderer_comparison.py).
The two implementations are `native` (HKTex Original) and `hybrid` (HKTex Splat Ours).
`original`/`2dgs` remain compatibility aliases for these same implementations.
Preview is the default.

The separate `experiments.train_surface` command runs the original UV training
lifecycle with native or HKS surface-sample adapters, preserving configured
importance sampling and density controllers. See [full surface training](../docs/FULL_SURFACE_TRAINING.md).
This experimental protocol retains verification checkpoints and is distinct
from the camera-image comparison documented below.

`train_surface --supervision camera` instead runs adaptive training on rasterized
camera pixels. The optional [fresh GPU camera pipeline](../docs/GPU_CAMERA_PIPELINE.md)
uses nvdiffrast and device-resident HKS geometry. `benchmarks.camera_scaling` pairs
native/HKS at matched resolutions and reports training/worker time with PSNR.
These full-training protocols retain verification receipts; the single-Markdown
artifact policy above applies to `benchmarks.compare_training`.

Use the lowercase `splatting_script` package for every module command. The former
mixed-case command namespace has been removed; older cleanup reports retain its
historical paths as recovery records.

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.compare_backends --backend native
"$HK_PY" -B -m splatting_script.experiments.compare_backends --backend hybrid
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow --scaling
```

These commands do not import Torch or request/launch GPU work. The standard workflow
previews ten jobs; scaling previews 111. Both renderer preflights and their paired
alignment gate must pass before capacity pilots or measurements. Existing GPU and capacity checks
control actual feasibility. The main training YAML and default numeric controls remain.

## Execute on a CUDA GPU

For the 5,000-source GPU validation and 1,000-update comparison commands,
follow the [HKS run guide](../hktex_splatting/README.md#validate-and-run-the-5000-kernel-experiment).
Slurm is optional; use your cluster scheduler when applicable. A100 is the
validated hardware; run the correctness gate on other supported CUDA GPUs too.
Run the correctness gate before the full workload. The documented diagnostic
workflow allows native repeatability variation and keeps HKS validation strict.

```bash
# Create matched initialization, preflights, pilots, paired speed runs and late profiles.
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --execute --output /path/to/a/new/workflow_directory

# Separate RGB/support/gradient/optimizer gate; never speed evidence.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --execute --validate-only --support-selector fused

# Separate diagnostic profile, with observer-on/off RGB/VJP gate.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --execute --pipeline-profile --profile-validate \
  --profile-start 40 --profile-steps 8 --iterations 48
```

Native/hybrid comparisons share initialization, seed, camera, loss and optimizer
controls. Research speed runs retain quality/PSNR, parameter-health checks and
archival checkpoints; profiling is separate. The runtime profile/memory/quality
readers remain compatible with historical records. Source matching is never bypassed.

`--support-selector legacy|fused` is preserved. Above 4,096 sources, fused selection
uses bounded Triton source tiles and deterministic distance/source-ID ties.
Source count alone no longer enables a legacy fallback or causes the workflow
to pass `--allow-selector-fallback`. Unsupported input guards remain explicit.
See [selector validation and compatibility](../docs/TILED_SELECTOR.md).

`--debug-support` downloads/checks live candidate rows and CSR, writes a failure
snapshot if necessary, then raises before invalid work reaches tile emission.
It does not repair bad support or weaken assertions. Diagnostic timing is excluded
from research speed evidence. Snapshots are not optimizer/RNG resume checkpoints.

The benchmark retains internal resident and full-spectrum eager controls solely for
its correctness comparisons. Their VJP/optimizer tolerances and the observer gate
remain. They cannot be selected as standalone training backends.

## Retired interfaces

The old forward trainer, `splat`/`resident` backend commands, alternative direct
selection modes and projected footprint pruning are discontinued. There is no
`--footprint-cutoff`, `--bounds`, `--nearest-backend` or `--selection` CLI option.
Renamed-module aliases such as `benchmarks.direct_tiles` and `modules.live_field`
are retired; use canonical filenames. Original native/HKS state-dictionary
checkpoints remain supported. Arbitrary object pickles using retired module paths
require the archived code version.

Use [the profiler guide](../docs/BENCHMARK_PROFILER.md),
[the matched benchmark protocol](../docs/HKTEX_VS_HKS_BENCHMARK.md),
[research protocol](../docs/HKTEX_RESEARCH_TRAINING.md),
[current architecture](../docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md), and
[cleanup/recovery report](../docs/HKS_PIPELINE_ONLY_CLEANUP.md).
Previous cleanup and rename reports preserve historical commands and results.
