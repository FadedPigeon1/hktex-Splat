# Native 5,000-kernel preflight: loss execution modes

The user's A100 run `hktex_vs_hks_smoke_v2` completed all ten jobs, including
both profiler gates. The subsequent `hktex_vs_hks_5000` workflow completed
initialization and hybrid preflight, but native preflight failed with:

```text
RuntimeError: Actual training step disagrees with the foreground loss check
```

This happened before the paired gate, pilots, measured training or profiling. Both arms'
measurement jobs were correctly skipped. No 5,000-kernel performance comparison
was produced. The passing hybrid receipt records `fused_chunks: 0` and
`selector_fallback: source_width_exceeds_4096`, confirming that the safe legacy
fallback was used; this is not a fused-selector limit error or reported OOM.
The native code reached this assertion after its first untimed update had
passed the finite-gradient, parameter-movement and optimizer-state checks.

## Finding and change

`run_preflight` previously evaluated its six initial cameras inside
`torch.no_grad()`, then compared the top-camera loss against the gradient-enabled
training step. Both used the same foreground formula, but different execution
modes. Compiled forward/autograd paths can differ numerically, so a strict loss
consistency test should compare matching modes. The original failure did not
save the two scalar losses, so its magnitude and exact compiled operation
responsible cannot be recovered from the log alone.

The preflight now renders a separate **gradient-enabled** top-camera reference
at the original model state and RNG, calculates the foreground loss, and compares
that to the actual training step. The consistency tolerance remains
`abs_tol=1e-7, rel_tol=1e-5`. The existing reset/replay checks for gradients,
parameters and optimizer state remain `atol=3e-5, rtol=3e-4`; finite-value and
kernel-face guards remain active.

The six no-grad evaluations remain unchanged for quality diagnostics and the
paired common-evaluator check. Their difference from the gradient-enabled
reference is recorded, not used as a substitute for training-step consistency.
Extra reference rendering restores RNG and is confined to untimed preflight;
the actual training step, baseline, HKS, optimizers and numerical equations
are unchanged.

Future preflights write `preflight_diagnostics.json` with:

- Gradient-enabled reference loss and no-grad evaluation loss.
- Foreground RGB maximum/mean differences between these execution modes.
- Actual first-step loss, absolute difference, tolerance and pass/fail result.
- Existing per-camera initial quality diagnostics.

The gradient-enabled reference image is saved as `top_initial_training_rgb.npy`.
A real training-step mismatch still raises, now including both scalar values
in the error. Original failed/successful runs and initialization files remain
untouched. This is a correction to the comparison setup, not proof that every
remaining GPU mismatch is harmless.

## Validate before the full benchmark

In the existing A100 allocation, first rerun only native preflight using the
same 5,000-source initialization. This does not launch the 1,000-update benchmark:

```bash
HK_PY=${HK_PY:-/scratch/user/$USER/.conda/envs/hktex/bin/python}
# If the saved initialization is missing, create a new seed-labelled one only.
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend native --execute --research --initialization-only \
  --kernels 5000 --resolution 256 --iterations 1 --seed 0 \
  --support-selector fused --matmul-precision highest

# Set this to the absolute path printed as RESEARCH_INITIAL_STATE above.
INITIAL=/absolute/path/printed/by/RESEARCH_INITIAL_STATE/initial.pt
test -f "$INITIAL"
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend native --execute --preflight-only --initial-state "$INITIAL" \
  --kernels 5000 --resolution 256 --iterations 1 --seed 0 \
  --support-selector fused --allow-selector-fallback \
  --coefficient-backend triton --splat-backend triton --runtime-counts \
  --matmul-precision highest
```

Inspect `preflight_diagnostics.json` in the printed `DIRECT_TILES_RUN` directory
if it fails. If it passes, run a fresh **matched four-update** 5,000/256 smoke
before the full experiment. Existing output directories are never resumed or
overwritten; source hashes must match both arms:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 5000 --resolutions 256 --iterations 4 --seed 0 \
  --initial-state "$INITIAL" --profile-start 0 --profile-steps 4 \
  --execute --output experiments/converse_heat_kernel_splatting/results/hktex_vs_hks_5000_smoke_v2

# Only after the smoke and gates pass; this is a full experiment.
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 5000 --resolutions 256 --iterations 1000 --seed 0 \
  --initial-state "$INITIAL" \
  --execute --output experiments/converse_heat_kernel_splatting/results/hktex_vs_hks_5000_v2
```

## Current verification

Temporary CPU fixtures reproduce the old mixed-mode rejection, check that the
new same-mode reference passes while recording the difference, and ensure
wrong training losses, nonfinite gradients and optimizer replay failures are
still rejected. They also check RNG and bitwise Adam trajectory preservation.
See [verification.json](hktex_vs_hks_benchmark/preflight_loss_verification.json).
The agent has no allocated GPU session; the corrected 5,000-source GPU check
has not been run. The user's earlier passing 1,000-source GPU smoke remains
valid evidence for its recorded source version.

## Subsequent replay failure: unresolved

The user's native run `direct_tiles_native_20261117T024719_509957Z_b442c03dff`
passed the new same-mode loss check, then failed optimizer/VJP replay. Saved
reference and first-step losses are 0.0978575125336647 and 0.09785667061805725;
their difference, 8.419156074523926e-7, passes the existing loss tolerance. This
does not establish gradient or optimizer repeatability. The old replay error
did not identify its failed tensor, so the cause is not yet established.

The diagnostic update retains the same loss and elementwise tensor predicates
and tolerances. `preflight_replay.json` now names every failed tensor, its shape,
mismatch count, maximum absolute difference and values at that difference.
It also records initial model hashes, empty optimizer state and learning rates
before each step. `preflight_diagnostics.json` embeds the same replay report.
On failure, `preflight_replay_tensors.npz` preserves both complete CPU snapshots
under `first/` and `replay/` keys for inspection. No failure is reclassified as a
pass, and the production training/rendering/optimizer code is unchanged.

Rerun the same native preflight command above and inspect the printed run's
`preflight_replay.json`. This is added diagnostic coverage, **not a claimed fix
for the replay mismatch**. GPU validation of this replay-report change remains
pending.

The later `hktex_vs_hks_full_20261116_214138` run reached preflight normally.
Hybrid passed with identical first/replay loss and all compared tensors.
Native failed even between gradient-enabled reference and training forwards:
0.09785506129264832 versus 0.09785808622837067 (difference
3.0249357223510742e-6). The execution-mode change alone therefore did not resolve
the native variability. Its exact cause is still unproven.

Loss-consistency failures now defer their exception until the second finite
update has produced replay diagnostics and saved snapshots. The same loss and
replay tolerances still apply, and any failed check still blocks training.
This avoids losing tensor-level evidence when the earlier scalar check fails;
it is diagnostic collection, not a bypass or a claimed numerical fix.

## Missing initialization paths

If `--initial-state` names a missing file, no loss or replay check has run.
Create an initialization using the command above and use its printed absolute
path; do not substitute a trained checkpoint. The CLI now resolves and checks
the file before importing Torch or setting up GPU/model data. Preview commands
still accept example paths without requiring those files. Historical timestamped
paths only work while their original result directories remain present.
