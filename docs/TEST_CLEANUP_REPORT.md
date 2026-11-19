# Test infrastructure cleanup report

Date: November 15, 2026 (America/Chicago). Scope: retirement of standalone testing infrastructure after the previous repository cleanup. No renderer, numerical, optimizer, training, benchmarking or profiling implementation was rewritten. No commit, push, reset or stash was performed.

Read first: the preserved [previous final cleanup report](FINAL_CLEANUP_REPORT.md). Its recorded CPU suite and package counts describe the earlier tree; this report records the subsequent testing-only retirement.

## Deletion gate and research preservation

The [reviewable pre-deletion plan](test_cleanup/plan.json) identified every candidate, reason and SHA-256 before removal. Imports, call sites, CLI entry points, benchmark/profiler execution paths and synthetic fixture dependencies were inspected. No production, preprocessing, training or benchmark import targeted these modules. The only declared console entry targeting them was `hktex-splatting-validate`; its registration was retired with its exclusive runner.

The synthetic derivative, support-ordering and selector fixtures contain unique validation work. Their retirement was explicitly identified before deletion. Exact source, modes, timestamps and hashes are preserved in a sealed recovery archive as data, never imported as an implementation. The existing recovery utility verified all **47 files** before deletion. Removal proceeded only after the fresh full CPU run succeeded. No broad directory/data deletion was used.

Preserved byte-for-byte relative to the starting working tree: all remaining Python source, including original `hktex/`, its registered modules, geodesic optimizers and root training loaders; all live HKS spectral calculations, support construction/fallbacks, three-corner evaluation, barycentric interpolation, tile scheduling, forward/backward kernels, weights, top-30 reconstruction and optimizers; every training, benchmark, profiler, metrics/reporting and preprocessing module; runtime assertions, overflow guards and support-debug checks. Existing user edits are part of that starting tree.

Meshes, eigensystems, cameras, checkpoints, benchmark/research results, previous cleanup plans/recovery manifests and previous result logs were not removed or rewritten. Fresh CPU logs are preserved as research evidence. Source provenance hashes for *new* benchmark runs will change because the source inventory no longer contains the test-only CPU adapter; existing source manifests/results are not relabelled. Compare runs made with matching source/workload manifests.

## Every deleted file

Removed **47 files**, **336,765 source bytes**, **5,961 source lines**. This includes all 45 Python files in `splatting_script/validation/`, their exclusive CPU adapter, and the unused manual Mitsuba test/demo. Exclusive package markers and empty directories were removed too. No active renderer/reference implementation was removed.

| Deleted path | Why removal was safe |
| --- | --- |
| `splatting_script/validation/__init__.py` | Package marker used only by the removed test/check/runner modules. |
| `splatting_script/validation/checks/__init__.py` | Synthetic fixture/numerical comparison helpers; callers are exclusively the retired test/GPU-check tree. |
| `splatting_script/validation/checks/coefficient_topology_checks.py` | Synthetic fixture/numerical comparison helpers; callers are exclusively the retired test/GPU-check tree. |
| `splatting_script/validation/checks/direct_tiles_checks.py` | Synthetic fixture/numerical comparison helpers; callers are exclusively the retired test/GPU-check tree. |
| `splatting_script/validation/checks/geometry_checks.py` | Synthetic fixture/numerical comparison helpers; callers are exclusively the retired test/GPU-check tree. |
| `splatting_script/validation/checks/gpu_dispatch_checks.py` | Synthetic fixture/numerical comparison helpers; callers are exclusively the retired test/GPU-check tree. |
| `splatting_script/validation/checks/resident_checks.py` | Synthetic fixture/numerical comparison helpers; callers are exclusively the retired test/GPU-check tree. |
| `splatting_script/validation/checks/runtime_count_checks.py` | Synthetic fixture/numerical comparison helpers; callers are exclusively the retired test/GPU-check tree. |
| `splatting_script/validation/checks/support_checks.py` | Synthetic fixture/numerical comparison helpers; callers are exclusively the retired test/GPU-check tree. |
| `splatting_script/validation/forward_splatting/__init__.py` | Package marker used only by the removed test/check/runner modules. |
| `splatting_script/validation/forward_splatting/test_coefficient_topology.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_coefficients.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_dynamic_dispatch.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_forward.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_gpu_dispatch.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_packed_support.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_prepared_geometry.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_profile_report.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_resident.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_runtime_counts.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/forward_splatting/test_training.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/run_cpu.py` | Test orchestration/bootstrap/worker only; no production or benchmarking caller. |
| `splatting_script/validation/run_retained_cpu.py` | Test orchestration/bootstrap/worker only; no production or benchmarking caller. |
| `splatting_script/validation/run_training_cpu.py` | Test orchestration/bootstrap/worker only; no production or benchmarking caller. |
| `splatting_script/validation/test_cleanup_recovery.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/test_direct_tiles_command.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/test_direct_tiles_cpu.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/test_direct_validation_cpu.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/test_pipeline_observer_cpu.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/test_pipeline_report.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/test_retained_commands.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/test_support_debug.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/test_training_research_cpu.py` | Automated unittest file, used only by test suites and their fixture dependencies. |
| `splatting_script/validation/validate_coefficient_topology_gpu.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/validate_coefficients_gpu.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/validate_compile.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/validate_geometry_gpu.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/validate_gpu_dispatch.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/validate_resident_compile.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/validate_resident_gpu.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/validate_runtime_counts_gpu.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/validate_support_gpu.py` | Standalone synthetic GPU/offline-compile correctness program; not called by the benchmark/profiler. |
| `splatting_script/validation/workflow/__init__.py` | Test orchestration/bootstrap/worker only; no production or benchmarking caller. |
| `splatting_script/validation/workflow/run_training_cpu.py` | Test orchestration/bootstrap/worker only; no production or benchmarking caller. |
| `splatting_script/validation/workflow/training_case_worker.py` | Test orchestration/bootstrap/worker only; no production or benchmarking caller. |
| `hktex_splatting/utils/cpu_native.py` | Test-only AST adapters and mocked CPU-native loader; all callers are retired fixtures/tests. |
| `scripts/test_renderers.py` | Standalone manual Mitsuba renderer demo/check; no training, preprocessing, benchmark or implementation caller. |

## Retained utilities with validation/testing roles

| Path | Why it remains |
| --- | --- |
| `scripts/test_numba_raster.py` | Production NumPy/Numba rasterizer imported by `hktex_splatting/utils/reference_assets.py:raster_surface`; the filename alone is misleading. Its embedded 42-case parity diagnostic remains with the required module, unchanged. |
| `splatting_script/benchmarks/direct_tiles.py` | Real-scene benchmark correctness gates (`--validate`, `--validate-only`) compare RGB/contributions/VJPs and optimizer state; all timing, GPU memory, quality/PSNR, rendering and training remain. |
| `splatting_script/benchmarks/pipeline_profile.py` | Workflow observer instrumentation and `validate_observer` gate, selected with `--profile-validate`. |
| `splatting_script/benchmarks/training_research.py` | Training health checks, matched quality evaluation, archival state and speed/quality reporting. |
| `hktex_splatting/utils/support_debug.py` | Opt-in diagnosis of live support rows/CSR without weakening assertions or silently repairing bad support. |
| `splatting_script/maintenance/recovery.py` | Safe, non-overwriting recovery of archived source; required for previous cleanup recovery and this retirement. |
| `hktex_splatting/utils/_legacy_paths.py` | Historical path-identity mappings used in provenance/recovery. Old validation strings are metadata, not live imports; retained unchanged. |

`unittest.mock` remains a standard-library dependency of actual benchmark/profiler instrumentation. No pytest/dev-only dependency or separate test configuration was declared in the current project metadata, so none was invented or uninstalled. Dataset validation/held-out quality calculations are production evaluation and were preserved.

## Modified files and metadata

Only documentation and the obsolete test console registration changed among pre-existing retained files:

| File | Task-scoped line changes |
| --- | --- |
| `README.md` | +21 / −18 |
| `pyproject.toml` | +0 / −1 |
| `hktex_splatting/README.md` | +1 / −1 |
| `splatting_script/README.md` | +12 / −13 |
| `docs/THREE_BACKENDS_CONSOLIDATION.md` | +6 / −0 |
| `docs/DIRECT_TILES_VALIDATION_FIX.md` | +6 / −0 |
| `docs/DIRECT_TILES.md` | +6 / −0 |
| `docs/BENCHMARK_PROFILER.md` | +15 / −16 |

`pyproject.toml` retains train/compare entry points, the historical command compatibility module, package data and all runtime metadata. Package discovery now finds **13 packages** instead of 17, because four validation packages are absent. Current README/command/profiler guides use retained embedded benchmark gates. Historical technical reports label their standalone test commands as archived provenance. The earlier `FINAL_CLEANUP_REPORT.md`, `RESEARCH_CLEANUP.md` and raw evidence/recovery records are unchanged.

Added records: this report; `docs/test_cleanup/plan.json`, `verification.json`, `scope_changes.json`, `preservation.json`, `archive_verified.json`, `documentation_checks.json`, `git_status_before.txt`, `git_status_after.txt`, `git_diff_stat.txt`, and per-check logs in that directory; `archive/three_backends/test_cleanup_20261115/{manifest.json,seal.json,retired.tar.gz}`; fresh pre-removal CPU results below. These are evidence and recovery records, not a new test suite/runner. Verification helper scripts were kept outside the repository in `/tmp`.

## Verification results

**Before deletion:** a fresh full CPU run passed **208 tests across 20 isolated suites**, with all executed suite sources unchanged. CUDA was hidden; numerical suites forbid CUDA initialization and use the Triton interpreter. These cover rendering, VJPs, optimizers, support ordering/fallbacks, command/workflow/reporting and recovery. The production raster utility separately passed **42 parity cases**. These are the actual new results, not copied from the previous report.

- [Full CPU result JSON](../experiments/converse_heat_kernel_splatting/results/test_cleanup_before_20261115/tests.json) and sibling individual suite logs.
- [Raster pre-removal log](test_cleanup/raster_before.log).

**After deletion:** 189 remaining source files passed syntax parsing and static local import resolution, with no imports of retired modules. All remaining Python sources match their pre-cleanup bytes. The Git index matches its initial bytes. Runtime imports covered 62 HKS/command/benchmark modules, the hybrid default and inherited backward API. Entry-point previews asserted that Torch/Triton were not imported; workflow/scaling previews did not execute GPU commands. Expanded package metadata and its installed-callable targets passed. The read-only Converse scaffold check and repeated 42-case raster diagnostic passed.

| Check | Result | Evidence |
| --- | --- | --- |
| `preview_native` | passed | [`log`](test_cleanup/preview_native.log) |
| `preview_hybrid` | passed | [`log`](test_cleanup/preview_hybrid.log) |
| `preview_original` | passed | [`log`](test_cleanup/preview_original.log) |
| `preview_2dgs` | passed | [`log`](test_cleanup/preview_2dgs.log) |
| `preview_splat` | passed | [`log`](test_cleanup/preview_splat.log) |
| `preview_resident` | passed | [`log`](test_cleanup/preview_resident.log) |
| `preview_default` | passed | [`log`](test_cleanup/preview_default.log) |
| `preview_compatibility` | passed | [`log`](test_cleanup/preview_compatibility.log) |
| `preview_legacy` | passed | [`log`](test_cleanup/preview_legacy.log) |
| `preview_workflow` | passed | [`log`](test_cleanup/preview_workflow.log) |
| `preview_scaling` | passed | [`log`](test_cleanup/preview_scaling.log) |
| `preview_profile_native` | passed | [`log`](test_cleanup/preview_profile_native.log) |
| `preview_profile_hybrid` | passed | [`log`](test_cleanup/preview_profile_hybrid.log) |
| `preview_benchmark_gate` | passed | [`log`](test_cleanup/preview_benchmark_gate.log) |
| `production_imports` | passed | [`log`](test_cleanup/production_imports.log) |
| `package_metadata` | passed | [`log`](test_cleanup/package_metadata.log) |
| `scaffold` | passed | [`log`](test_cleanup/scaffold.log) |
| `raster_after` | passed | [`log`](test_cleanup/raster_after.log) |
| `original_import` | unavailable; see limitation below | [`log`](test_cleanup/original_import.log) |

Original eager registry import reached the unchanged `hktex/utils/mitsuba.py`, where `mi.set_variant("cuda_ad_rgb")` failed with `no CUDA-capable device is detected`. The exact error is retained in [the import log](test_cleanup/original_import.log). Native command routing/import resolution passed, and the pre-removal suites exercised original numerical methods through CPU fixtures. This does not establish a successful native CUDA scene/training run.

The maintained-guide link audit checked **161 local links** across ten documents, with no broken links or active commands referencing retired test modules. Historical reports, raw evidence and migration mappings retain their recorded identities as described above. See [the documentation check record](test_cleanup/documentation_checks.json).

No allocated Slurm/A100 GPU was available. **No GPU smoke test, CUDA correctness run, training job or GPU benchmark was launched or claimed.** Real native CUDA/FAISS/Mitsuba construction, hybrid GPU forward/backward, atomics/stream behavior and on-mesh optimizer updates still require the retained manual benchmark gates inside an existing allocation. The benchmark's gates do not substitute for every retired synthetic selector/component check.

## Remaining structure

```text
hktex/                          original registered HKTex implementations
hktex_splatting/
  knn_heat/, modules/            candidate support, live kernels, coefficients
  rendering/, triton/            hybrid/shared references, tiles, GPU VJPs
  trainers/, utils/              optimizers, runtime controls, required utilities
splatting_script/
  configs/, experiments/         training configuration and command entry points
  benchmarks/                    timing, memory, quality, profiling and reports
  maintenance/                   sealed archive recovery
scripts/                         required original/preprocessing/raster tools
configs/, optimisation.py, interactive_*.py
README.md, pyproject.toml, Splatting_script.py
data/, outputs/, experiments/     protected assets, models and research results
archive/, docs/                  protected history, current guides and evidence
```

There is no live standalone `splatting_script/validation/` package or test-only `cpu_native.py` adapter. The production raster module retains its historical `test_` name for import compatibility.

## Git diff summary and recovery

Branch: `cleanup/hybrid-main-renderer`. The repository already contained substantial staged/unstaged changes and untracked source before this task. No index contents were changed. [Initial status](test_cleanup/git_status_before.txt), [ending status](test_cleanup/git_status_after.txt) and [ordinary Git diff stat](test_cleanup/git_diff_stat.txt) preserve that context. **The ordinary Git diff includes prior cleanup/user changes and is not this task's diff.** Newly added audit files may appear after the recorded status capture; the report and inventory describe them explicitly.

The task-scoped [change inventory](test_cleanup/scope_changes.json) reports 47 deletions and 8 modifications relative to the exact starting working tree: 67 added lines in modified files and 6,010 removed lines including retired source. Added audit/recovery records are listed above. Many lowercase canonical files were already untracked, so plain `git diff` cannot show all of their removals. The sealed manifest plus task-scoped inventory is authoritative. No Git history was rewritten, and original staged/unstaged edits were not discarded.

Recovery archive: `archive/three_backends/test_cleanup_20261115/`. It contains every deleted file's original bytes and can be verified/listed with the retained tool:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.maintenance.recovery archive/three_backends/test_cleanup_20261115
"$HK_PY" -B -m splatting_script.maintenance.recovery archive/three_backends/test_cleanup_20261115 --list
# Example: restore a selected missing file, never overwrite an existing file.
"$HK_PY" -B -m splatting_script.maintenance.recovery archive/three_backends/test_cleanup_20261115 \
  --restore splatting_script/validation/run_cpu.py
```

Restoring a runner alone is insufficient to execute its suites: restore the corresponding package markers, tests and fixture dependency closure, or explicitly select all manifest paths. The recovery tool checks seals, paths, regular-file types, membership and byte hashes. It refuses existing destinations. Do not use `git restore HEAD` for files that were untracked or whose original working copy differs from HEAD. The temporary starting-tree snapshot is `/tmp/hktex-test-cleanup-m9_elw_e`; it is supplemental and may not persist.

The archive payload is covered by the existing tarball ignore rule. Preserve/copy it alongside the manifest and seal for durable recovery; metadata alone cannot restore source. Do not reset/stash the repository to recover tests. No environment packages or installed launchers were altered: refresh the editable installation when desired to remove a previously installed `hktex-splatting-validate` launcher, which is intentionally no longer supported.

## Remaining risks

- Standalone automated regression coverage has intentionally been retired. Future numerical changes should restore the archived suites or use equivalent reviewed checks before research claims.
- CPU fixture success and unchanged production source do not establish actual GPU correctness or performance. No new performance improvement is claimed.
- Historical path mappings, result manifests, raw logs and earlier cleanup reports intentionally retain retired module identities. Maintained commands no longer call those modules; historical commands are labelled as requiring recovery.
- Original eager import and real GPU entry-point execution depend on the native environment/allocation; consult the exact check log and limitation above.
- Untracked canonical source and ignored recovery payloads need deliberate owner review when preparing a shareable Git snapshot. No automatic staging, commit or push was performed.
