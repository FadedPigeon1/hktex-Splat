# Three-backend repository consolidation

> Historical validation record: standalone test/validation paths and commands
> below were retired on November 15, 2026. Recorded measurements and source
> identities are preserved as provenance; those commands require archived-source
> restoration. Use the current [benchmark diagnostics](../splatting_script/README.md)
> and [test cleanup/recovery report](TEST_CLEANUP_REPORT.md) for maintained commands.

> Current source paths use lowercase `splatting_script`. Historical run
> manifests, source hashes and cleanup recovery records retain their original
> identities; updating this guide's paths does not claim a new GPU validation.

This is a historical consolidation report. The current main renderer is the
triangle-supported hybrid, and package/training commands now default to it.
Native, direct spectral, and resident controls remain available; `original`
and `2dgs` remain compatibility names. See the [current README](../README.md)
and [legacy trainer entry](../splatting_script/benchmarks/legacy/README.md).
The preservation results and measurements below describe the earlier cleanup.

Cleanup is complete. The maintained configurations are original HKTex, the
preserved GPU-resident implementation, and the opt-in 2DGS-inspired direct tile
experiment. No GPU jobs, allocations, CUDA launches or CUDA benchmarking ran.
The new renderer still needs real-CUDA/FAISS and end-to-end training validation.

Subsequent validation fix: the experimental renderer and comparison harness
were updated after a user-run GPU failure. The cleanup evidence below remains
the original snapshot. Native and resident numerical source remain unchanged;
see [the failure analysis](DIRECT_TILES_VALIDATION_FIX.md) for current defaults
and new CPU evidence.

## Removed

Removed **10,360 files and 1,755 empty directories**,
after publishing the full inventory, verifying every recovery payload, passing
the retained CPU suites, checking source dependencies, and rechecking protected
files. This includes 118 Python files (including
retired snapshots/tests), obsolete preprocessing, independent old splatting and
experiment workflows, their exclusive benchmark/figure/migration tools, an
unreferenced image-GS comparison implementation, loose run logs, superseded
migration evidence, repeated CPU reports and obsolete rendered outputs.

- [Exact proposed inventory and reasons](three_backends_cleanup/INVENTORY.md)
- [Actual removed files and empty directories](three_backends_cleanup/removed.json)
- [Removed Python sources](three_backends_cleanup/REMOVED_SOURCE_FILES.md)
- [Pre-removal gate](three_backends_cleanup/removal_gate.json)

No wildcard research-directory deletion was used. Every removal was an explicit
in-repository file from the sealed inventory; directories were removed only
when empty. Existing source deletions visible at task start are not counted as
this cleanup's removals.

## Retained

- Entire `hktex/`, original root training tools, native helpers/configurations,
  `data/`, `outputs/`, meshes, spectral assets and checkpoint inputs.
- Existing resident trainer, support/dispatch/geometry, coefficients, shader,
  reconstruction/backward and numerical regression tests.
- Direct tile renderer and Triton nearest-selection prototype, unchanged from
  the prior CPU mathematical audit.
- Three-way benchmark/selector, current report readers and GPU correctness
  validators. Resident benchmark defaults now use the validated fused Triton
  coefficients and runtime counts through the existing numerical operators.
- Every GPU report and `forward_training_*` run, including raw traces,
  checkpoints, quality reports and source manifests; latest CPU report per
  family. [Essential baseline report index](../experiments/baselines/README.md).

Some old-named containers are required dependencies. For example,
`resident.py` imports classes from `gpu_dispatch.py`, and the unchanged trainer
imports `ForwardRenderer`, whose timer/frame helpers live in older splatting
files. Those files were retained byte-for-byte instead of changing validated
source. Their standalone obsolete runners are retired. See the
[explicit dependency exceptions](three_backends_cleanup/RETAINED_DEPENDENCIES.md).
Only the three canonical configurations are maintained; compatibility branches
inside protected files remain as dependencies/control code.

The camera and ROI JSON inputs were already missing at task start. Only those
required data files were recovered from Git's index, with exact byte checks:
[recovered inputs](three_backends_cleanup/recovered_inputs.json). No deleted
historical implementation tree was restored to active use.

## Recovery and version control

Original removed payload: **7,560,566,549 bytes**.
Verified compressed payload: **3,517,711,590 bytes**.
Recovery directory: `archive/three_backends/20261101_2db94d6b/`.

Every removed file has its original path, SHA256, size, mode and recorded mtime
in `manifest.json`. `seal.json` records checksums of the containers. All members were
stream-verified; representative source/JSON/image files were restored inside the
repository and independently checked before cleanup. A separate 374-file source
context includes pre-cleanup native/resident code and existing user edits.
Git working/staged patches, index entries and refs were captured read-only.
HEAD, the index and existing refs were not rewritten. Large tar payloads remain
local and are ignored by Git; manifests and instructions remain visible.

[Recovery instructions](../archive/three_backends/README.md) explain integrity
checks and explicit selective restoration. Restoration refuses overwrite,
external paths, duplicate/nonregular members and corruption. Older archive
folders referenced by historical reports were already absent before this task;
this cleanup does not claim to reconstruct unavailable research data.

## Preservation and CPU results

| Check | Result |
| --- | --- |
| Original native Python source | 89 files unchanged by SHA256 |
| Retained package Python source | 46 pre-existing files unchanged, including the direct tile math |
| Latest A100 record's package files | 44 retained unchanged; five orphaned preprocessing/provenance files also match inside the recovery archive |
| Protected scientific files | 4,227 files preserved: size/mtime; reports/configs/source additionally SHA256 |
| CPU before cleanup | 107 tests in 14 isolated suites, passed |
| CPU after cleanup | 107 tests in the same suites, passed |
| Recovery safety | 7 tests passed, before and after |
| Retained package syntax/local imports | 186 files checked, no errors or missing local import modules |
| Git HEAD/index | Unchanged |
| GPU execution/allocation | None |

The pre-existing local change in `hktex/modules/tracer.py` remains unchanged.
No numerical assertion or tolerance in retained baseline tests was modified.
Retired experiment-only assertions survive in the archive; they were not
weakened to obtain a passing suite. The new recovery test uses the filesystem's
recordable timestamp precision rather than inventing subsecond precision.
A development-only relative-output-path issue in the new runner was corrected
before the final pre-removal gate; the logs are retained separately.

The baseline remains functional under its CPU interpreter/control checks,
including source gradients, selection, normalization, outstanding graphs,
face transitions and Adam parameters/moments. Its prior A100 `PASSED` record
and performance evidence are preserved. No new GPU certification is implied.

Evidence: [before tests](three_backends_cleanup/cpu_before_final/tests.json),
[after tests](three_backends_cleanup/cpu_after/tests.json),
[preservation](three_backends_cleanup/preservation.after.json),
[structure](three_backends_cleanup/structure.after.json),
[archive verification](three_backends_cleanup/archive_verified.json),
[final machine-readable result](three_backends_cleanup/final.json).

## Commands for the three configurations

From the repository root:

```bash
HK_PY=/scratch/user/"$USER"/.conda/envs/hktex/bin/python
export LD_LIBRARY_PATH=/scratch/user/"$USER"/.conda/envs/hktex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CUDA_VISIBLE_DEVICES='' "$HK_PY" -B splatting_script/validation/run_cpu.py
"$HK_PY" -B -m splatting_script.experiments.compare_backends --backend original
"$HK_PY" -B -m splatting_script.experiments.compare_backends --backend resident
"$HK_PY" -B -m splatting_script.experiments.compare_backends --backend 2dgs
```

The previews are dry. For actual matched A100 runs, manually use `--execute`
inside an existing user-owned allocation with the same `--initial-state`,
resolution, source count, cameras and budget. Run the correctness gate before
speed measurements; separate profiles from steady-state timing. Full commands
are in [splatting_script/README.md](../splatting_script/README.md). The optional
installed selector is `hktex-splatting-compare`; module invocation needs no
installation or dependency changes.
