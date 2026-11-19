# Profiler gradient gate: failed smoke-run investigation

The user's existing `hktex_vs_hks_smoke` workflow completed initialization,
both preflights, paired input checks, capacity pilots and both four-update
speed runs. Only its two diagnostic profiles failed. Their logs and report
are preserved under
`experiments/converse_heat_kernel_splatting/results/hktex_vs_hks_smoke/`.

Both tracebacks end in `pipeline_profile.validate_observer` comparing parameter
VJPs, after the RGB comparison passed:

| Existing GPU evidence | Native | Hybrid |
|---|---:|---:|
| Mismatched gradient entries | 5 / 3000 | 4 / 3000 |
| Largest absolute difference | 0.0005366802 | 0.0014858246 |
| Existing absolute / relative tolerances | 0.00003 / 0.0003 | 0.00003 / 0.0003 |

This was not a native-versus-hybrid equality assertion. Each renderer was
compared with itself, first unprofiled, then with profiler scopes and saved-tensor
hooks enabled. The original gate used `RGB.square().sum()`, whereas the real
training loss is `(RGB[mask] - target[mask]).square().sum(-1).mean()`.
The unnormalized probe can amplify floating-point reduction differences relative
to the actual training objective. That is a plausible contributor, not a proven
root cause: the old logs did not record repeated unprofiled gradients or their
parameter names. They cannot distinguish an observer effect from ordinary
same-backend GPU reduction variability. The preceding LLVM warning was not the
exception that stopped these processes.

## Change

- `renderer_comparison.py` now supplies the exact training loss, including the
  foreground mask and target, to the profiler gate. Training, renderers and
  optimizers are unchanged.
- `pipeline_profile.py:validate_observer` keeps its historical RGB-squared-sum
  default for existing callers. The harness explicitly selects the actual
  training objective, recorded in the receipt.
- A second unprofiled forward/VJP checks repeatability before the profiled
  forward/VJP. All three calls preserve and use the same RNG state, and no
  parameters or optimizers are updated. Reference outputs/VJPs are cloned so
  later scratch reuse cannot overwrite comparison evidence.
- RGB, scalar loss, gradient presence, finiteness and every parameter VJP are
  checked. CUDA tolerances remain **atol=3e-5, rtol=3e-4**; CPU tolerances remain
  exact zero. An unstable unprofiled repeat also fails; tolerances are not
  inflated to cover observed noise. This changes the diagnostic objective to
  the real training objective, not the tolerance or renderer mathematics.
- Failures now save `profile_validation.json` before raising, with parameter
  names/shapes, maximum differences, mismatch counts and baseline-repeat
  comparisons. `failure_reason` distinguishes `observer_mismatch` from
  `unprofiled_repeatability_failed`. Reports explicitly display failed gates.

The deeper support/coefficient/VJP/optimizer `--validate-only` gate, all original
HKTex/HKS sources, selectors and GPU kernels remain unchanged. The additional
unprofiled evaluation happens only in the diagnostic validation process and
does not enter speed measurements.

## Validation and rerun

CPU checks use actual Torch profiler and saved-tensor hooks on small synthetic
RGB models. They cover the supplied training loss, legacy API, unused parameters,
RNG preservation, unchanged Adam updates, named gradient failures, repeatability
failures and serializable nonfinite diagnostics. Verification details are in
[profile_gate_verification.json](hktex_vs_hks_benchmark/profile_gate_verification.json).
No A100 allocation was available to the agent; this fix has not passed the GPU
gate yet. A successful GPU rerun must be observed, not inferred from CPU tests.

Source hashes change when benchmark code changes. Keep the old smoke results
and run a fresh matched workflow in a **new directory**; a new profile must not
be relabelled as matching an older speed run's source hash. Inside the existing
A100 allocation:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.benchmarks.compare_training workflow \
  --kernel-counts 1000 --resolutions 128 --iterations 4 \
  --pilot-iterations 4 --profile-start 0 --profile-steps 4 \
  --execute --output experiments/converse_heat_kernel_splatting/results/hktex_vs_hks_smoke_v2
```

If a gate still fails, inspect the new `profile_validation.json`. An unprofiled
repeat failure establishes insufficient repeatability for this check; an
observer-only mismatch warrants investigation of the named parameters and
profiler/hooks. Neither is treated as a successful profile. Four updates remain
a smoke test, not a steady-state performance or convergence result.

## Subsequent user GPU verification

The user's `hktex_vs_hks_smoke_v2` workflow completed all ten jobs on an
NVIDIA A100-PCIE-40GB. Both saved `profile_validation.json` receipts report
`passed: true`, objective `foreground_rgb_squared_error_pixel_mean`, and passed
unprofiled-repeat checks. This is observed GPU evidence for the 1,000-kernel,
128px, four-update smoke workflow, not validation of a full training budget.
The later 5,000-kernel failure is in a different preflight loss check; see
[its investigation](PREFLIGHT_LOSS_MODE_FIX.md).
