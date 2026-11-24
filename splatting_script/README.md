# HKS training and benchmark commands

Use lowercase `splatting_script` module names from the repository root with the
HKS environment activated. See the [HKS README](../hktex_splatting/README.md) for
installation, required mesh/spectral assets, lighting and implementation details.
Original HKTex's own experiment entry point remains `optimisation.py`.

## Recommended: adaptive camera training with lighting

The current `compare-128` preset compares native HKTex with spectral-support HKS
using fresh GPU triangle rasterization, shared targets and analytic GGX lighting,
original optimizers and original density policies. `hybrid` means HKS; it does
not select conventional 2D Gaussian Splatting. The field evaluation changes,
while original HKTex source files remain intact.

```bash
# Preview only: no Torch import or GPU work.
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse

# Initialize and verify both implementations; no training loop.
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse --verify-only --execute

# Short training smoke check.
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse --iterations 40 --kernels 1000 \
  --execute --output experiments/converse_lighting_smoke

# Full comparison: 5,000 initial kernels, 5,000 updates per method.
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse \
  --execute --output experiments/converse_lighting_full
```

Use a new output directory for each execution. Initialization and gates already
run before full training; a separate manual `INITIAL` variable is unnecessary.
Replace `converse` with a prepared configured asset. The preparation CLI supports
fewer assets than the training CLI; selecting an asset does not obtain its mesh.
The provided bowl is unrun because it lacks the required color texture.

Useful explicit overrides:

| Option | Effect |
| --- | --- |
| `--backend hybrid` | Train/verify HKS only. |
| `--backend native` | Train/verify native HKTex only. |
| `--resolution 256` | Override image resolution; check GPU memory and correctness first. |
| `--lighting none` | Unlit material RGB instead of analytic GGX. |
| `--lighting diffuse` | Analytic diffuse-only lighting. |
| `--material-roughness 0.2` | Change fixed GGX roughness, not a learned material parameter. |
| `--light-direction 0.3 0.8 0.6` | Set the world-space surface-to-light direction. |
| `--support-domain graph` | Older approximate hop-limited support; spectral is selected by the preset. |

The preset uses a shared boundary-stop overlay and bounded initial native
approximation, both recorded in receipts. Independent implementation/gradient
checks remain strict. An initial quality allowance does not guarantee final PSNR.
Without the preset, legacy CLI defaults remain surface supervision, graph support
and no lighting. Do not infer the preset from the entry point's filename.

## Reports, checkpoints and images

Adaptive `train_surface` runs retain `comparison.md`, workflow/worker JSON,
provenance, verification receipts, logs, kernel-count trajectories and checkpoint
archives. These are reproducibility records, not a resume interface. They are
kept under the selected output directory, which is excluded from source publishing.
This workflow does not automatically export comparison PNGs. The preset disables
original training plots; renderer-benchmark image exports are a separate workflow.

Use training interval, whole-process time and forward-only latency for their
respective scopes. Compare PSNR and final kernel counts alongside time/memory.
A completed run alone is not evidence of a speedup or dense native equivalence.
Shared analytic lighting is not ray tracing or the complete original Mitsuba run.

## Profile separately

```bash
python -B -m splatting_script.experiments.train_surface \
  --preset compare-128 --asset converse --iterations 160 \
  --profile-start 100 --profile-steps 8 --execute \
  --output experiments/converse_lighting_profile
```

This records synchronized stage diagnostics with overhead. Nested stages overlap;
do not add them or use the profiled run for speedup claims. The above short
window does not profile later density events. To examine those, choose enough
updates and a window covering the configured controller schedule.

## Earlier fixed-kernel camera benchmark

This separate workflow isolates renderer/update costs with fixed kernel counts.
It is not the recommended full adaptive training comparison above. Preview:

```bash
python -B -m splatting_script.benchmarks.compare_training workflow
```

Execute only after assets and correctness checks pass:

```bash
python -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 5000 --resolutions 128 --iterations 1000 --seed 0 \
  --execute --output experiments/comparison_fixed_kernels.md
```

Its default compact mode produces a Markdown report and removes private temporary
artifacts. Add `--keep-artifacts` and a new output directory for retained debugging
records. This policy applies to `compare_training`, not adaptive `train_surface`.
The workflow includes initialization, strict preflights, pilots, unprofiled speed
runs and separate profiles. Do not bypass a failed gate to obtain a speed claim.

For a standalone renderer preview and bounded GPU validation:

```bash
python -B -m splatting_script.experiments.compare_backends --backend hybrid
python -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --execute --validate-only --support-selector fused
```

The older harness accepts `original`/`2dgs` aliases for `native`/`hybrid`.
`--support-selector legacy|fused` and bounded large-source tiled selection belong
to that harness. They are different from `train_surface --support-domain
spectral|graph`. There is no 4,096-source model limit. Shared resident/eager
controls remain internal correctness references, not standalone training modes.

## Other retained workflows

| Entry point | Scope |
| --- | --- |
| `splatting_script.benchmarks.camera_scaling` | Paired adaptive training at multiple resolutions; use its preview/help to inspect propagated controls. |
| `splatting_script.benchmarks.compare_rendering` | Original-render versus HKS render-only comparison, not full training. |
| `splatting_script.benchmarks.spectral_distances` | Selector-distance microbenchmarks, not end-to-end training. |
| `splatting_script.experiments.train_surface --supervision surface` | Original mesh-sample supervision; no camera triangle rasterization. |

All execution needs a usable NVIDIA CUDA GPU; Slurm is optional. Tools never
request allocations automatically. CPU previews need neither GPU nor prepared
assets. The former mixed-case namespace and retired `splat`/`resident` training
backends, projected-footprint options and custom density/optimizer policies are
not supported. Runtime safety checks and original density policies remain.

See the [current implementation/run guide](../hktex_splatting/README.md),
[architecture](../docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md),
[profiler guide](../docs/BENCHMARK_PROFILER.md),
[work log](../docs/HKS_WORK_LOG.md) and
[cleanup/recovery record](../docs/HKS_RESULTS_CLEANUP.md). Earlier reports retain
historical commands; use the current READMEs for new runs.
