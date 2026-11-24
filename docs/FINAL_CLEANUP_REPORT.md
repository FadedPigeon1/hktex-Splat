# Final cleanup report — HKTex + Hybrid HKS

Date: 2026-11-14. This report describes the final sweep only. The earlier
[research cleanup](RESEARCH_CLEANUP.md), its [plans](research_cleanup/plan.json),
[visualization plan](research_cleanup/visualization_plan.json),
[renderer retirement plan](research_cleanup/abandoned_renderers_plan.json), and
recovery records remain unchanged. Their earlier deletions and test results are
not relabelled as new work.

## Result and scope

The repository retains original pointwise HKTex and triangle-supported Hybrid
HKS, including source preparation, spectral centroid/radius selection, fused
selection and its safe large-source fallback, CSR ordering, three-corner heat
coefficients, barycentric interpolation, tile queues, Triton emission, top-30
weighted RGB reconstruction, custom backward operators and optimizers. The
`splat` and `resident` controls remain actual dependencies of validation and
comparison workflows; they were not deleted just to reduce file count.

The important finding was a pre-existing directory-case mismatch: the user had
already renamed the working tree to `splatting_script/`, but imports, configuration
paths, source hashing, console metadata and guides still used `Splatting_script`.
The lowercase tree was preserved. Current references now use it consistently.
A small root `Splatting_script.py` compatibility namespace retains historical
module commands without duplicating implementations or creating directories that
differ only by case. A regression test compares old/new command previews.
No directory was renamed or moved by this sweep.

The cleanup removes three unreferenced postprocessing utilities and 111 derived
`.pyc` files (1,704,691 bytes). It removes unused imports in recovery/report tests
and an obsolete package-data glob. No installed dependencies were uninstalled,
and no environment was changed. The native package's broad registered imports
were preserved rather than treating uncalled modules as safe dependency removals.

## Pre-deletion review and evidence

[The deletion plan](final_cleanup/plan.json) was produced and described before
removal. The three source candidates had no incoming imports, symbol callers,
required command/configuration references, or runtime dependency in retained
native/HKS, preprocessing, benchmark or validation workflows. Their standalone
CLI purpose was checked, not inferred from import counts alone. Each source was
verified byte-for-byte against Git's existing index before deletion.

| Deleted source | Why it was unnecessary |
| --- | --- |
| `scripts/extract_video_frames.py` | General MP4 frame extraction; independent output postprocessing, never called by the retained rendering/training or reporting pipelines. |
| `scripts/gather_benchmark_videos.py` | Moves legacy presentation videos; current benchmark writes its own report artifacts and does not invoke this script. No video outputs were touched. |
| `scripts/find_common_valid_files.py` | Intersects legacy CSV filename sets; matched native/HKS comparisons use shared initialization and recorded workload controls instead. Existing CSV/results were retained. |

The exact bytecode file list is in the appendix and plan. Only regular `.pyc`
files in live-source `__pycache__` directories were removed. Archive, data,
checkpoint/output, result and runtime-cache trees were excluded. Derived
bytecode can be regenerated; it is not scientific evidence. No prior benchmark
runtime cache or result was removed as a space-saving shortcut.

Unique original research was identified rather than silently retired. Native
PBR, dataset, ablation/tuning, spectral-analysis and paper-comparison commands
retain distinct work and are listed for manual review below.

## Footprint decision

`hktex_splatting/rendering/footprint.py` is still genuinely used:

- `DirectTileRenderer._hybrid` calls `projected_bounds` for positive cutoffs,
  non-full bounds and non-oracle rendering. Default cutoff zero remains unchanged.
- The retained forward trainer calls projection and `prune_dispatch`.
- Direct-rendering and forward-training tests check actual spectral values,
  projection fallback and above-cutoff recall.
- `bin_footprints` and `kernel_footprints` support retained reference policies.
  `triangle_tile_work` is required by the main hybrid emitter independently of
  optional conic pruning.

Consequently no remaining footprint implementation, flag, recall assertion,
triangle-to-tile scheduling or support correctness check was removed. The
obsolete cached/visualization/Gaussian approaches retired previously remain
retired; no archived implementation is imported by the live renderer.

## Documentation and portability

The main README now covers original versus hybrid rendering, the spectral/tile
pipeline, installation/dependency guidance, assets/preprocessing, portable
training and benchmark commands, CPU/GPU validation and current limitations.
Maintained guides use `HK_PY=${HK_PY:-python}` and a caller-selected checkout;
Grace-specific home directories are no longer needed to copy those commands.
The existing A100/Slurm execution guard remains a deliberate reproducibility
policy, not a newly removed safety restriction.

The Converse scaffold checker required templates and an output directory retired
in an earlier cleanup. It now verifies the actual camera/ROI/configuration
scaffold; `--check` passes without creating files or processing assets. Its
directory setup never overwrites existing files.

Six `docs/direct_tiles_validation_fix/` artifacts were already absent when this
sweep began: `analysis.json`, `tests.json`, `preservation.json`, `regressions.log`,
`geometry.log`, and `commands.log`. The historical guide now labels those recorded
paths as unavailable provenance rather than linking to absent files or claiming
them as fresh validation. Their contents were not fabricated. Current-source
links were repaired; earlier raw cleanup manifests and source hashes were not
rewritten.

The project is an editable research checkout. Pyproject discovers 17 current
packages plus the compatibility module; it does not bundle assets, install the
native CUDA stack, or promise a self-contained wheel. CUDA PyTorch, FAISS,
Mitsuba/Dr.Jit and geodesic/spectral dependencies still require the documented
native environment. Ray/Optuna/Tiny CUDA NN are used by separate retained paper
workflows, so no arbitrary replacement dependency set was introduced.

## Remaining structure

```text
README.md, LICENSE, pyproject.toml
Splatting_script.py                 # one-file historical command compatibility
hktex/                             # original native implementation, unchanged
hktex_splatting/
  knn_heat/                        # live triangle support and safe fallback
  modules/                         # live source state and spectral coefficients
  rendering/                       # hybrid orchestration, shared geometry/oracles
  triton/                          # forward, backward, scan and reduction kernels
  trainers/, utils/                # retained controls, profiling and recovery paths
splatting_script/
  configs/                         # current recorded training/source settings
  experiments/                     # native/hybrid command and legacy compatibility
  benchmarks/                      # matched training, workflow, profiler and reports
  validation/                      # full CPU suite, numerical and GPU gates
  maintenance/                     # verified archive recovery
scripts/                           # required preprocessing/raster and original tools
configs/, optimisation.py, interactive_*.py
  # original registered workflows; data-dependent demos need user input paths
data/, outputs/, experiments/       # assets, checkpoints and research evidence
archive/, docs/research_cleanup/, docs/three_backends_cleanup/
  # preserved recovery/evidence; never imported as rendering implementations
```

## Mathematical and research preservation

The audit compares the exact starting working tree, including existing user
edits. All original `hktex/` files, original root training loaders, and previous
cleanup/recovery inputs are preserved. For 114 native/HKS numerical source
modules, the entire AST is unchanged after normalizing only the script namespace
spelling; core shaders/support/coefficients are byte-identical. No masks,
spectral operations, arithmetic ordering, parameter shapes, gradients, optimizer
steps, capacity limits, fallback conditions or eigensystem calculations were
rewritten. Metadata/config path repairs do not substitute another renderer.

Source provenance now includes the actual lowercase tree and compatibility
module. A new run's source hash therefore changes; comparisons should continue
requiring matched source/workload manifests. Earlier runs and their original
source identities are unchanged, not silently relabelled.

## Validation results

The full CPU runner passed **208 tests across 20 suites**, with every suite's source unchanged during execution. The 42-case synthetic raster comparison passed. Early command/report checks passed 30 tests; the root-module compatibility regression also passed separately. Package discovery (17 packages plus compatibility module), the read-only scaffold check, the seven-job GPU-free workflow preview, Python syntax, current documentation links and source-reference checks passed.

Full logs: `experiments/converse_heat_kernel_splatting/results/final_cleanup_cpu_20261114/`.

| CPU suite | Tests | Result |
| --- | ---: | --- |
| `splatting_script.validation.forward_splatting.test_forward` | 9 | passed |
| `splatting_script.validation.forward_splatting.test_training` | 12 | passed |
| `splatting_script.validation.forward_splatting.test_coefficients` | 7 | passed |
| `splatting_script.validation.forward_splatting.test_runtime_counts` | 4 | passed |
| `splatting_script.validation.forward_splatting.test_packed_support` | 7 | passed |
| `splatting_script.validation.forward_splatting.test_prepared_geometry` | 6 | passed |
| `splatting_script.validation.forward_splatting.test_dynamic_dispatch` | 8 | passed |
| `splatting_script.validation.forward_splatting.test_coefficient_topology` | 10 | passed |
| `splatting_script.validation.forward_splatting.test_gpu_dispatch` | 8 | passed |
| `splatting_script.validation.forward_splatting.test_resident` | 19 | passed |
| `splatting_script.validation.forward_splatting.test_profile_report` | 5 | passed |
| `splatting_script.validation.test_direct_tiles_cpu` | 44 | passed |
| `splatting_script.validation.test_direct_tiles_command` | 11 | passed |
| `splatting_script.validation.test_direct_validation_cpu` | 12 | passed |
| `splatting_script.validation.test_support_debug` | 4 | passed |
| `splatting_script.validation.test_pipeline_report` | 9 | passed |
| `splatting_script.validation.test_pipeline_observer_cpu` | 2 | passed |
| `splatting_script.validation.test_retained_commands` | 10 | passed |
| `splatting_script.validation.test_training_research_cpu` | 14 | passed |
| `splatting_script.validation.test_cleanup_recovery` | 7 | passed |

The CPU suite uses original numerical methods in isolated fixtures and the
Triton interpreter. Its native fixture substitutes deterministic exhaustive L2
for GPU FAISS; it does not establish real CUDA/FAISS numerical parity. No Slurm
allocation was available, so no GPU correctness, CUDA training or GPU benchmark
was launched or claimed. Those remain explicit gates inside an allocation.

## Remaining risks and manual review

- **GPU validation:** real A100 execution, FAISS stream behavior, GPU backward
  atomics and complete on-mesh training still require the retained manual gates.
- **Historical missing artifacts:** the six direct-validation files noted above
  were missing before this sweep. Recover them from the owner's original copy
  if that historical investigation must be independently rechecked.
- **Original private paths:** `scripts/render_hktex_pbr.py` contains original
  scene/environment-map search roots under `/homes/`, `/data/home/` and `/data2/`.
  Original `interactive_*.py` demos and some `scripts/time_*`, `tune_hparams*`,
  `run_*` tools default to `/data2/objaverse`. Two original MLP/vertex ray YAMLs
  reference `/data/objaverse`. These independent native/paper workflows require
  caller-supplied inputs; changing their defaults would change existing behavior.
  [The path audit](final_cleanup/paths_review.json) lists exact files/lines.
  The maintained matched workflow uses repository-relative assets instead.
- **Distinct original work:** manually review `scripts/kernel_modulation.py`,
  `albo_diffuse_from_matlab.py`, `albo_diffuse_single_kernel.py`, `evecs_interps.py`,
  `render_hktex_pbr.py`, `run_benchmark_parallel*.py`, `tune_hparams*.py`, MLP/vertex
  comparison scripts, and UV storage/low-resolution experiments before retiring
  original paper functionality. No file was kept solely as an imagined future
  dependency; these are existing standalone research/preprocessing commands.
- **Reference policies:** resident/splat and positive-cutoff projection remain
  supported controls with active callers/tests. Retiring their interfaces needs
  a separate decision rather than deleting shared code during a final sweep.
- **Untracked source:** much of the essential lowercase script/HKS tree was already
  untracked at the start. A clean working tree is not implied: the owner must
  review and explicitly include those files when preparing the sharing commit.
  No broad staging operation was performed.
- **Historical relocation metadata:** `utils/paths.py` and `_legacy_paths.py`
  retain source-identity/recovery helpers and retired target strings. Those
  archival identities are not advertised as current executable entry points.
- **Distribution:** assets and several native dependencies are supplied outside
  pip. Use a checkout/editable install; preserve the recorded native environment.

## Git state and recovery

The branch remains `cleanup/hybrid-main-renderer`. The tree was already heavily
dirty: prior deletions, staged source recovery copies, untracked packages, and
user edits (including the lowercase directory rename and native tracer edit).
This sweep does not attribute those changes to itself. Git's index remains
byte-for-byte unchanged; no staged input was overwritten. There was no reset,
stash, commit, push, history rewrite or branch operation.

The three new source deletions can be restored from their existing index bytes:

```bash
git restore --worktree -- scripts/extract_video_frames.py
# Substitute another path from this sweep's three-file source deletion list.
```

Exact pre-sweep working files, hashes, initial Git status and the original index
were saved under `/tmp/hktex-final-sweep-ked5yl87/`. Files already untracked in
the lowercase tree cannot be restored by a Git command against HEAD; use the
matching `before/<relative-path>` snapshot for selective recovery. Keep a durable
copy of that snapshot if long-term recovery of uncommitted work is needed.
Previous cleanup archives/plans and their Git recovery preimages were not changed.
Avoid a blanket restore, which would also discard unrelated user changes.

This sweep changes 68 existing files, deletes 3 source files and adds the compatibility/report/audit files listed below, plus 111 ignored bytecode deletions. The Python changes against the exact captured working inputs contain 277 insertions and 465 deletions; additions include one small compatibility module. Namespace spelling repairs account for most touched Python files.

Tracked-file Git diff for this sweep's paths (includes pre-existing edits to those same tracked files; untracked lowercase content changes are inventoried separately):

```text
 README.md                              | 174 ++++++++++++++++++++++----
 data/converse/README.md                |  10 +-
 docs/ORIGINAL_HKTEX_README.md          | 100 +++------------
 hktex_splatting/README.md              |  57 +++++++++
 hktex_splatting/utils/_legacy_paths.py | 221 ++++++++++++++++-----------------
 scripts/extract_video_frames.py        |  58 ---------
 scripts/find_common_valid_files.py     |  84 -------------
 scripts/gather_benchmark_videos.py     |  63 ----------
 8 files changed, 339 insertions(+), 428 deletions(-)
```

The complete current Git status is saved in [git-status.txt](final_cleanup/git-status.txt). The original status/index and exact before-files remain in the external snapshot. No index entry was added, removed or replaced by this sweep.

## Complete file inventory for this sweep

The following table is computed against the starting working-tree snapshot,
including modifications to already-untracked files. Git's ordinary diff alone
cannot show all those content changes. Generated audit/report files are listed
separately; all 111 derived bytecode deletions appear in the following appendix.

| File | Change | Reason |
| --- | --- | --- |
| `README.md` | modified | Clear installation, pipeline, native/hybrid, portable commands, limitations and validation guide. |
| `pyproject.toml` | modified | Lowercase entry points/package discovery, compatibility module, removal of unused package-data glob. |
| `experiments/baselines/README.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `hktex_splatting/README.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `hktex_splatting/cli.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `hktex_splatting/trainers/forward.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `hktex_splatting/utils/paths.py` | modified | Actual lowercase script/config paths; recognize both historical/current root spellings without rewriting identities. |
| `hktex_splatting/utils/_legacy_paths.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `hktex_splatting/utils/experiment_io.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `hktex_splatting/utils/compiler_runtime.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/__init__.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/README.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `splatting_script/experiments/__init__.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/experiments/compare_backends.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/experiments/train_forward.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/configs/__init__.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_retained_commands.py` | modified | Lowercase namespace and regression coverage for old/new native/hybrid/workflow command parity. |
| `splatting_script/validation/validate_gpu_dispatch.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/run_cpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_direct_tiles_cpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/run_retained_cpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_pipeline_report.py` | modified | Canonical imports and removal of confirmed unused imports; existing checks and calculations preserved. |
| `splatting_script/validation/__init__.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/validate_coefficient_topology_gpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/validate_geometry_gpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_direct_validation_cpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/validate_resident_gpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/validate_support_gpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_direct_tiles_command.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_support_debug.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_cleanup_recovery.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/validate_runtime_counts_gpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_training_research_cpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/run_training_cpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/test_pipeline_observer_cpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/workflow/__init__.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/workflow/run_training_cpu.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/__init__.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/test_packed_support.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/test_coefficient_topology.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/test_runtime_counts.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/test_gpu_dispatch.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/test_dynamic_dispatch.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/test_resident.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/test_coefficients.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/forward_splatting/test_prepared_geometry.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/checks/__init__.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/checks/gpu_dispatch_checks.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/checks/geometry_checks.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/validation/checks/resident_checks.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/maintenance/recovery.py` | modified | Canonical imports and removal of confirmed unused imports; existing checks and calculations preserved. |
| `splatting_script/benchmarks/pipeline_workflow.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/benchmarks/__init__.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/benchmarks/profile_training_report.py` | modified | Canonical imports and removal of confirmed unused imports; existing checks and calculations preserved. |
| `splatting_script/benchmarks/training_research.py` | modified | Canonical imports and provenance hashing of the actual lowercase source tree plus compatibility module. |
| `splatting_script/benchmarks/compare_training.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/benchmarks/direct_tiles.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/benchmarks/pipeline_report.py` | modified | Canonical script namespace/path spelling only; numerical behavior and existing assertions retained. |
| `splatting_script/benchmarks/legacy/README.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `scripts/find_common_valid_files.py` | deleted | Prelisted unreferenced postprocessing utility; exact Git-index preimage verified. |
| `scripts/setup_converse_experiment.py` | modified | Replace requirements for retired templates/output scaffold with current camera/ROI/configuration inputs. |
| `scripts/extract_video_frames.py` | deleted | Prelisted unreferenced postprocessing utility; exact Git-index preimage verified. |
| `scripts/gather_benchmark_videos.py` | deleted | Prelisted unreferenced postprocessing utility; exact Git-index preimage verified. |
| `docs/HKTEX_RESEARCH_TRAINING.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `docs/ORIGINAL_HKTEX_README.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `docs/THREE_BACKENDS_CONSOLIDATION.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `docs/DIRECT_TILES_VALIDATION_FIX.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `docs/DIRECT_TILES.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `docs/BENCHMARK_PROFILER.md` | modified | Current lowercase/portable command paths and repaired links; missing historical artifacts explicitly labelled where applicable. |
| `data/converse/README.md` | modified | Portable relative asset path and links/check command for the current scaffold. |
| `Splatting_script.py` | added | Historical module-command compatibility. |
| `docs/FINAL_CLEANUP_REPORT.md` | added | Requested report or machine-readable deletion, environment, path, validation and Git audit. |
| `docs/final_cleanup/audit.json` | added | Requested report or machine-readable deletion, environment, path, validation and Git audit. |
| `docs/final_cleanup/environment.json` | added | Requested report or machine-readable deletion, environment, path, validation and Git audit. |
| `docs/final_cleanup/git-status.txt` | added | Requested report or machine-readable deletion, environment, path, validation and Git audit. |
| `docs/final_cleanup/package_check.json` | added | Requested report or machine-readable deletion, environment, path, validation and Git audit. |
| `docs/final_cleanup/paths_review.json` | added | Requested report or machine-readable deletion, environment, path, validation and Git audit. |
| `docs/final_cleanup/plan.json` | added | Requested report or machine-readable deletion, environment, path, validation and Git audit. |

## Derived bytecode deletion appendix

```text
scripts/__pycache__/test_reference_ray_bvh.cpython-311.pyc
scripts/__pycache__/inspect_converse.cpython-311.pyc
scripts/__pycache__/test_nvdiffrast_raster.cpython-311.pyc
scripts/__pycache__/test_color_evaluation_trace.cpython-36.pyc
scripts/__pycache__/cloudrunner_importance_ablation.cpython-311.pyc
scripts/__pycache__/test_cloudrunner_benchmark.cpython-311.pyc
scripts/__pycache__/cloudrunner_reference_surface.cpython-311.pyc
scripts/__pycache__/diagnose_color_evaluation.cpython-36.pyc
scripts/__pycache__/test_cloudrunner_hktex_v2.cpython-311.pyc
scripts/__pycache__/test_triangle_splat_backends.cpython-36.pyc
scripts/__pycache__/test_reference_ray_pruning.cpython-311.pyc
scripts/__pycache__/audit_cloudrunner_reconstruction.cpython-311.pyc
scripts/__pycache__/compare_cloudrunner_results.cpython-311.pyc
scripts/__pycache__/test_numba_raster.cpython-311.pyc
scripts/__pycache__/test_cloudrunner_importance_ablation.cpython-311.pyc
scripts/__pycache__/inspect_converse.cpython-36.pyc
scripts/__pycache__/reference_ray_pruning.cpython-311.pyc
scripts/__pycache__/triangle_splat_renderer.cpython-36.pyc
scripts/__pycache__/compare_nvdiffrast_splat.cpython-36.pyc
scripts/__pycache__/triangle_splat_renderer.cpython-311.pyc
scripts/__pycache__/compare_triangle_splat_backends.cpython-311.pyc
scripts/__pycache__/compare_triangle_splat_backends.cpython-36.pyc
scripts/__pycache__/benchmark_cloudrunner.cpython-311.pyc
scripts/__pycache__/diagnose_color_evaluation.cpython-311.pyc
scripts/__pycache__/visualize_kernel_faces.cpython-36.pyc
scripts/__pycache__/benchmark_cloudrunner_hktex_v2.cpython-311.pyc
scripts/__pycache__/test_audit_cloudrunner_reconstruction.cpython-311.pyc
scripts/__pycache__/visualize_kernel_faces.cpython-311.pyc
scripts/__pycache__/reference_ray_bvh.cpython-311.pyc
scripts/__pycache__/test_nvdiffrast_raster.cpython-36.pyc
scripts/__pycache__/compare_nvdiffrast_splat.cpython-311.pyc
hktex/__pycache__/__init__.cpython-311.pyc
hktex/knn_heat/__pycache__/__init__.cpython-311.pyc
hktex/knn_heat/knn_heat/__pycache__/knn_heat.cpython-311.pyc
hktex/knn_heat/knn_heat/__pycache__/post_weights.cpython-311.pyc
hktex/knn_heat/knn_heat/__pycache__/__init__.cpython-311.pyc
hktex/knn_heat/knn_heat/__pycache__/faiss_knn.cpython-311.pyc
hktex/knn_heat/knn_heat/__pycache__/knn_gather.cpython-311.pyc
hktex/knn_heat/knn_heat/__pycache__/config.cpython-311.pyc
hktex/knn_heat/knn_heat/_ops/__pycache__/grid_select.cpython-311.pyc
hktex/knn_heat/knn_heat/_ops/__pycache__/heat.cpython-311.pyc
hktex/knn_heat/knn_heat/_ops/__pycache__/__init__.cpython-311.pyc
hktex/knn_heat/knn_heat/_ops/__pycache__/weights.cpython-311.pyc
hktex/knn_heat/knn_heat/_ops/__pycache__/distance.cpython-311.pyc
hktex/knn_heat/knn_heat/_ops/__pycache__/gather.cpython-311.pyc
hktex/rendering/__pycache__/torch_texture_renderer.cpython-311.pyc
hktex/rendering/__pycache__/base.cpython-311.pyc
hktex/rendering/__pycache__/heat_kernels_density_renderer_knn.cpython-311.pyc
hktex/rendering/__pycache__/__init__.cpython-311.pyc
hktex/rendering/__pycache__/ray_renderer.cpython-311.pyc
hktex/rendering/__pycache__/util.cpython-311.pyc
hktex/rendering/__pycache__/uv_texture_renderer.cpython-311.pyc
hktex/rendering/__pycache__/heat_kernels_renderer.cpython-311.pyc
hktex/rendering/__pycache__/heat_kernels_renderer_knn.cpython-311.pyc
hktex/rendering/__pycache__/ray_integrator.cpython-311.pyc
hktex/rendering/__pycache__/vertex_colours_renderer.cpython-311.pyc
hktex/trainers/__pycache__/uv_texture.cpython-311.pyc
hktex/trainers/__pycache__/base.cpython-311.pyc
hktex/trainers/__pycache__/vertex_colours.cpython-311.pyc
hktex/trainers/__pycache__/__init__.cpython-311.pyc
hktex/trainers/__pycache__/mitsuba_ray_trainer.cpython-311.pyc
hktex/trainers/__pycache__/uv_texture_pcl.cpython-311.pyc
hktex/trainers/__pycache__/mitsuba_trainer.cpython-311.pyc
hktex/trainers/__pycache__/vertex_ray_trainer.cpython-311.pyc
hktex/trainers/__pycache__/stat_heat_kernels.cpython-311.pyc
hktex/trainers/__pycache__/utils.cpython-311.pyc
hktex/density_controllers/__pycache__/error_densification.cpython-311.pyc
hktex/density_controllers/__pycache__/base.cpython-311.pyc
hktex/density_controllers/__pycache__/importance_pruning.cpython-311.pyc
hktex/density_controllers/__pycache__/__init__.cpython-311.pyc
hktex/density_controllers/__pycache__/utils.cpython-311.pyc
hktex/utils/__pycache__/config.cpython-311.pyc
hktex/utils/__pycache__/mesh.cpython-311.pyc
hktex/utils/__pycache__/base.cpython-311.pyc
hktex/utils/__pycache__/typing.cpython-311.pyc
hktex/utils/__pycache__/diffusion.cpython-311.pyc
hktex/utils/__pycache__/image.cpython-311.pyc
hktex/utils/__pycache__/color.cpython-311.pyc
hktex/utils/__pycache__/__init__.cpython-311.pyc
hktex/utils/__pycache__/video.cpython-311.pyc
hktex/utils/__pycache__/callback.cpython-311.pyc
hktex/utils/__pycache__/misc.cpython-311.pyc
hktex/utils/__pycache__/local_reference_frames.cpython-311.pyc
hktex/utils/__pycache__/pcl_losses.cpython-311.pyc
hktex/utils/__pycache__/cart_bary.cpython-311.pyc
hktex/utils/__pycache__/mitsuba.cpython-311.pyc
hktex/utils/__pycache__/sampling.cpython-311.pyc
hktex/utils/__pycache__/repr_patches.cpython-311.pyc
hktex/utils/__pycache__/geodesics.cpython-311.pyc
hktex/utils/__pycache__/lbo.cpython-311.pyc
hktex/utils/__pycache__/distances.cpython-311.pyc
hktex/utils/__pycache__/metrics.cpython-311.pyc
hktex/triton/__pycache__/__init__.cpython-311.pyc
hktex/triton/__pycache__/distances.cpython-311.pyc
hktex/modules/__pycache__/mesh.cpython-311.pyc
hktex/modules/__pycache__/pcl_texture.cpython-311.pyc
hktex/modules/__pycache__/base.cpython-311.pyc
hktex/modules/__pycache__/heat_kernel_density_knn.cpython-311.pyc
hktex/modules/__pycache__/heat_kernel_model_knn.cpython-311.pyc
hktex/modules/__pycache__/__init__.cpython-311.pyc
hktex/modules/__pycache__/eigen_albo_knn.cpython-311.pyc
hktex/modules/__pycache__/tracer.cpython-36.pyc
hktex/modules/__pycache__/eigen_albo.cpython-311.pyc
hktex/modules/__pycache__/tracer.cpython-311.pyc
hktex/modules/__pycache__/pcl_model.cpython-311.pyc
hktex/modules/__pycache__/geodesic_opt.cpython-311.pyc
hktex/modules/__pycache__/mlp_net.cpython-311.pyc
hktex/modules/__pycache__/heat_kernel_texture.cpython-311.pyc
hktex/modules/__pycache__/heat_kernel_model.cpython-311.pyc
hktex/modules/__pycache__/heat_kernel_texture_knn.cpython-311.pyc
hktex/modules/__pycache__/utils.cpython-311.pyc
```

These paths were derived bytecode, excluded from scientific result/archive trees, and are regenerable from retained source.
