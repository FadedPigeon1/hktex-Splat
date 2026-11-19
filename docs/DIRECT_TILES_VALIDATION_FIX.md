# Saved direct/native validation failure

> Historical validation record: standalone test/validation paths and commands
> below were retired on November 15, 2026. Recorded measurements and source
> identities are preserved as provenance; those commands require archived-source
> restoration. Use the current [benchmark diagnostics](../splatting_script/README.md)
> and [test cleanup/recovery report](TEST_CLEANUP_REPORT.md) for maintained commands.

Historical context: the current refactor defaults to triangle-supported
hybrid splatting, described in [DIRECT_TILES.md](DIRECT_TILES.md). Full-coverage
spectral splatting remains `--selection splat`. The FAISS fallback
below remains available through `--selection native --nearest-backend faiss`.
Its user-run GPU gate passed in run
`direct_tiles_direct_20261105T044854_349740Z_0529a6a42c`: RGB error 0, all six
original checks passed. That result validates the earlier native control,
not the new kernel/tile emitter, whose GPU validation remains pending.

The run `direct_tiles_direct_20261105T041437_067896Z_6f4eeef0d4` failed
the correctness gate. Its original files are preserved unchanged. This update
fixes the comparison harness and adds a native-selection fallback; it does not
claim that a subsequent GPU run passed. No GPU jobs or allocations ran during
this investigation.

> Historical evidence availability: the six `direct_tiles_validation_fix/`
> artifacts listed below were already absent when the final sweep began.
> Their recorded paths are retained as provenance, with no fabricated contents.
> Use [the final cleanup report](FINAL_CLEANUP_REPORT.md) for current CPU results;
> historical GPU measurements below are not new validation.

## Findings

The old comparison reset called `load_state_dict` and created new optimizers
without clearing parameter gradients. Neither operation clears `.grad`.
The native, resident and direct backward passes therefore accumulated
gradients across comparisons. The saved gradient, optimizer update and moment
differences cannot establish backend correctness. The reset now calls
`model.zero_grad(set_to_none=True)` for every model parameter, including any
parameters outside an optimizer group, and validation asserts that gradients
are empty before each backward pass.

There is also a separate forward selection difference in the saved tapes:

| Saved observation | Result |
| --- | --- |
| Foreground queries | 711 |
| Queries with different outer-50 sets | 1, ordinal 11 (zero based) |
| Different source | Native includes 534; direct includes 555 |
| Queries with different ordered outer lists | 16 |
| Common query/source heat-response pairs | 35,549 of 35,550 |
| Maximum heat-response difference on common pairs | Exactly 0 |
| Maximum filtered-weight difference with identical ordered outer lists | Exactly 0 |
| Maximum RGB difference in the saved GPU report | 0.0066356957 |

The old diagnostic sorted each outer list and compared response slots even
when they represented different sources. The new diagnostic matches source
identities and reports missing pairs separately. Missing selections still fail
the strict selection gate.

A CPU reconstruction using the saved source checkpoint, isotropic spectral
basis and top-camera visibility includes both sources in the tile candidate
list. It gives squared distances of about `2.62400436e-5` (534) and
`2.62378817e-5` (555), a gap of `2.16190e-9`. The source locations are distinct.
This is consistent with native float32 FAISS rounding/ranking differing from
the experimental float64 selector. It is an inference: the old GPU tapes do
not contain the actual GPU query/source embeddings, so the CPU reconstruction
does not prove their exact floating-point values. Future tapes now save both
embeddings and native search distances.

The resident comparison retains its existing face-union/cap/tie semantics.
It is reported for context and is not changed to implement native pointwise
outer-50 selection.

## Current selection policy

`2dgs` and `DirectTileRenderer` now default to `nearest_backend='faiss'`.
This fallback builds a private copy of the native index, interpolates queries
in the native batch dimensions (256 in the harness, including its final
partial batch), uses native search order and differentiable distance
recomputation, and shades with the original spectral heat/filter/top-30/color
methods. It does not construct kernel-to-triangle support or coefficient
tables, and does not mutate the original index or renderer implementation.

The fallback bypasses tile pruning. Its ordinary forward skips unused tile
bins; trace mode computes bins for diagnostics only and labels them as such.
Reports explicitly record the selected policy, FAISS rebuild and whether tile
candidates affected selection. This mode provides a correctness control, not
evidence of faster tile selection.

Explicit `--nearest-backend triton` or `torch` keeps the original float64 tile
ranking. That ranking is still experimental and may reproduce the saved
failure. No threshold was relaxed: native/direct images, parameter gradients,
updates and Adam moments retain `atol=3e-5, rtol=3e-4`; outer-50 and positive
top-30 sets must still match. No TF32 policy or native/resident numerical code
was changed.

Metrics detach tensors before converting them to scalars, addressing the
autograd scalar-conversion warning. The run path and argument JSON are now
written before setup so a failed run remains identifiable. The final exception
in the supplied log came from the correctness gate; investigation of the
earlier LLVM message is separate from these measured discrepancies.

## CPU evidence and preservation

**29 CPU tests passed:** six new comparison/fallback regressions, 17 existing
direct geometry/spectral/gradient/interpreted-Triton tests and six dry-command
and retained-input checks. The new fallback test deliberately uses a CPU
index with a different tie policy from the tile selector, then checks native
IDs, responses, weights, image and source/appearance gradients. It checks
delegation to the native policy, not GPU FAISS arithmetic. Another regression
checks independent Adam updates and moments across three comparison resets.
The full retained CPU runner includes the new regression module.

Original HKTex's 89 Python files and all 44 retained resident Python files
match the sealed pre-cleanup source snapshot, including preexisting user
edits. Preserved GPU source hashes and existing JSON reports were checked
separately. Historical GPU reports and the failed run were not edited.

- Saved-tape analysis: `direct_tiles_validation_fix/analysis.json` (not present in this checkout)
- CPU tests and current source hashes: `direct_tiles_validation_fix/tests.json` (not present in this checkout)
- Native/resident/report preservation: `direct_tiles_validation_fix/preservation.json` (not present in this checkout)
- New regression log: `direct_tiles_validation_fix/regressions.log` (not present in this checkout)
- Existing direct suite log: `direct_tiles_validation_fix/geometry.log` (not present in this checkout)
- Command checks: `direct_tiles_validation_fix/commands.log` (not present in this checkout)

## Manual correctness rerun

For the user's existing GPU session, this reuses the failed run's exact saved
initial state. It requests no allocation. Match any nondefault config, source
count or resolution used in the failed invocation; the command below uses
the 1,000-source, 64-pixel defaults.

```bash
RUN=experiments/converse_heat_kernel_splatting/results/direct_tiles_direct_20261105T041437_067896Z_6f4eeef0d4
"$HK_PY" -B -m splatting_script.experiments.compare_backends \
  --backend 2dgs --execute --validate-only --selection native --nearest-backend faiss \
  --initial-state "$RUN/initial.pt"
```

`--validate-only` compares the three backends, records tapes/metrics and exits
before warmup and timing. A failure still raises and preserves the evidence.
A pass is an initial single-camera check; full matched training and speed
measurements remain separate work. The three-backend comparison protocol is
in [splatting_script/README.md](../splatting_script/README.md).
