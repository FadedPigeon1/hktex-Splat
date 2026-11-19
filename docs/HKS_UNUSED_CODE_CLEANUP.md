# HKS unused-code cleanup

Scope: retain original HKTex, Hybrid HKS, preprocessing, native comparisons, correctness gates and benchmark/profiler workflows. Existing staged/unstaged changes are preserved.

## Proposed changes (recorded before deletion)

| File | Evidence and action |
|---|---|
| `hktex_splatting/utils/paths.py` | Delete after moving its only active value (ROOT) to the existing experiment_io import. Its five callable helpers have no active callers; historical source/archive readers in compare_training and recovery.py remain. |
| `hktex_splatting/utils/_legacy_paths.py` | Delete obsolete migration table; its only consumer is the unused research_path helper being removed. No imports, CLI, configuration or lazy exports depend on it afterward. |
| `hktex_splatting/utils/experiment_io.py` | Remove unused digest/input_identities/protected_snapshot/verify_snapshot functions and exclusive hashlib/CHECKPOINT/PARSED/ROIS metadata. Current provenance is recorded by training_research, and archive safety checks remain in maintenance/recovery.py. |
| `hktex_splatting/utils/reference_assets.py` | Remove unused input_identity/library_versions wrappers. Keep camera, mesh, texture, original reference rendering and alignment helpers. |
| `hktex_splatting/rendering/prepared_geometry.py` | Remove unused cpu_bytes property; retain all buffer/version checks and device memory reporting. |
| `hktex_splatting/triton/resident_gpu.py` | Remove unused filter_support_scores JIT kernel (no Python, dynamic-string, profiler or CLI call site). Keep _support_limit, fused selection, packing and all selector fallbacks unchanged. |
| `splatting_script/benchmarks/pipeline_workflow.py` | Consolidate two ROOT imports; existing constant value is identical. |

No deletion is based on filename alone. Original hktex modules are registered through eager imports/configuration; they are retained. Native reference/metrics/profile readers are executable workflows and are retained. Recovery utilities and previous cleanup records are retained.

The unused score-filter kernel is unique retired GPU code; its exact source is preserved in the archive before removal. _support_limit and all active kernels remain.

## Recovery

Exact current working-tree preimages, including uncommitted edits: `archive/hks_unused_20261117/preimages.tar.gz`.

Archive SHA256: `65bd5afefcc9b7516a950f69ed98bfd074d9a904c37dfb483b78504c9ff30250`. Verified every archived member against the original bytes. Extract into a separate directory for review; copying selected files back restores this task's preimages. No Git history/index changes are needed.

## Validation

Completed CPU/static checks; no GPU jobs were launched.


## Completed cleanup

Deleted only:

- `hktex_splatting/utils/paths.py` (its only active constant, ROOT, now comes from the existing experiment_io module).
- `hktex_splatting/utils/_legacy_paths.py` (exclusive obsolete migration table).

Removed 13 unused definitions: five historical path/source resolvers, four old asset/snapshot functions, two reference metadata helpers, PreparedGeometry.cpu_bytes, and filter_support_scores. The retained _support_limit and selector fallback functions are unchanged. The corresponding unused hashlib import and three checkpoint/ROI constants were removed. Two ROOT imports were updated. No renderer, model, optimizer, camera/raster helper, preprocessing command, benchmark gate or original HKTex implementation was deleted.

| File | Added lines | Removed lines |
|---|---:|---:|
| `hktex_splatting/utils/paths.py` | 0 | 96 |
| `hktex_splatting/utils/_legacy_paths.py` | 0 | 146 |
| `hktex_splatting/utils/experiment_io.py` | 1 | 65 |
| `hktex_splatting/utils/reference_assets.py` | 1 | 17 |
| `hktex_splatting/rendering/prepared_geometry.py` | 0 | 3 |
| `hktex_splatting/triton/resident_gpu.py` | 0 | 12 |
| `splatting_script/benchmarks/pipeline_workflow.py` | 2 | 2 |

Net runtime-source reduction: **337 lines**, across two deleted and five modified files, relative to the starting working tree. The 18 KiB archive is recovery data, not an executable dependency. No scientific assets, checkpoints, results, historical cleanup reports or existing recovery utilities were deleted.

## Verification results

- Parsed all 184 remaining Python sources in the audited original/HKS/script roots.
- Imported all 52 remaining HKS/workflow module paths on CPU; CUDA remained uninitialized.
- Compared all 39 retained functions in the edited utility/geometry/GPU modules against their original ASTs: identical.
- Four original-versus-cleaned CPU geometry cases matched bitwise, including device buffers, layouts and byte reporting. Buffer mutation/version checks still reject invalid snapshots.
- Main DirectTileRenderer, TileSurface/LiveHeatField dependencies, fused selector and active packing kernels remain importable.
- Native/hybrid, compact-report and full-artifact workflow previews passed without importing Torch/Triton or starting GPU work.
- No remaining Python call/import references to the deleted modules or definitions were found.
- Original hktex sources, original author scripts and all other source files outside the seven-file change list match the starting bytes. Git index/staged content is unchanged; no commit, push, reset or stash was performed.
- GPU numerical rendering, backward execution and full training were not run. Retained active function bodies are unchanged; CPU/static checks do not certify GPU behavior.

## Already-missing inputs and manual review

`experiments/converse_fidelity/cameras.json` and `rois.json` were already deleted in Git before this task. The camera file is required by current Converse rendering/benchmark setup; the ROI file supports the standalone regional-quality workflow. Both mesh and spectral assets remain present. These prior deletions were not overwritten. Real-scene execution needs the camera configuration supplied/restored by the owner; synthetic imports and geometry checks do not establish that the incomplete data scaffold runs.

Original HKTex's registry imports other author models/trainers (including MLP/PCL/Mitsuba/vertex workflows). Removing these would require changing the original registration/loading contract, so they remain. Standalone native_reference, image-quality metrics and profile_training_report modules are retained executable research workflows, not classified unused just because they lack internal importers. Historical result readers, strict/diagnostic validation modes, screen tile scheduling, and the 4096-source legacy selector fallback are retained.

The native 5000-source repeatability issue discussed in earlier reports is not addressed by this cleanup. Old source-resolver imports are retired; external clients of those internal utilities need the archived preimages. Native/HKS model class paths and state-dictionary checkpoints are unchanged.

## Restore/review instructions

Review archive membership with `tar -tzf archive/hks_unused_20261117/preimages.tar.gz`. Extract into a separate review directory and copy back only the files you intend to restore. Do not use git reset/checkout over the existing staged or unstaged work. Some starting files, including paths.py, were untracked, so the preimage archive and the task-local table above are necessary in addition to Git diff. Temporary audit scripts stayed under /tmp and were not installed as a test suite.
