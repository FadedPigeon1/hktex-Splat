# HKS pipeline-only cleanup

Date: November 16, 2026 (America/Chicago). Scope: retain original HKTex, current triangle-supported Hybrid HKS and reproducible native-versus-hybrid training/benchmark/profiling workflows; retire unrelated optional implementations. No commit, push, reset, stash or Git-index modification was performed.

## Result and dependency gate

Read before changes: the current architecture, [prior final cleanup](FINAL_CLEANUP_REPORT.md), [test cleanup](TEST_CLEANUP_REPORT.md), [rename report](HKS_FILE_RENAME_REPORT.md) and earlier research cleanup evidence. These reports remain unchanged historical records; this document supersedes their supported-interface descriptions.

The [reviewable deletion plan](hks_pipeline_only/PLAN.md) and [full dependency inventory](hks_pipeline_only/plan.json) were recorded before deletion. Imports, nested calls, CLI roots, lazy exports, benchmark/reference calls, original checkpoint writers and recovery dependencies were checked. The inventory covers **115 Python files** across the HKS packages and related scripts. [Every module's disposition](hks_pipeline_only/module_disposition.json) records why it remains or is retired. Native author scripts were treated as executable original workflows, not marked unused because no module imports them.

Removed **26 files**: fifteen renamed-module aliases, the optional projected-footprint module, historical forward renderer/trainer and wrappers, exclusive payload/timer/topology modules and the obsolete CPU dynamic dispatcher. No original HKTex component, native author script, configuration, mesh, spectral asset, trained checkpoint, scientific result or prior cleanup/recovery record was deleted or rewritten.

Unique research work was identified before retirement: tile-KNN/direct-native controls, full-spectral Triton emission, projected cutoff pruning and old forward/cooperative approaches. Exact preimages of 89 source/guide files were sealed and verified before removal in `archive/three_backends/hks_pipeline_only_20261116/`. These are archival data, never imported by the live pipeline. See [archive verification](hks_pipeline_only/archive_verified.json).

## Every deleted file

| Deleted path | Why removal was safe / caller disposition |
| --- | --- |
| `hktex_splatting/rendering/direct_tiles.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/triton/direct_tiles.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/modules/live_field.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/modules/coefficients.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/triton/coefficient_gpu.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/knn_heat/resident_support.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/rendering/core.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/rendering/differentiable.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/triton/gpu.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/triton/backward.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/triton/runtime_counts.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/rendering/kernel_centric.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/rendering/renderer.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/rendering/kernel_centric_raster.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `splatting_script/benchmarks/direct_tiles.py` | Retire renamed-module compatibility alias. Canonical imports are used by every retained runtime; old experimental module imports/object pickles are intentionally discontinued. |
| `hktex_splatting/rendering/footprint.py` | Exclusive optional projected-bound pruning; remove positive-cutoff callers/flags first. Default cutoff zero never calls it. |
| `hktex_splatting/rendering/triangle_splat_renderer.py` | Standalone ForwardRenderer used only by the retired forward trainer; hybrid owns a different kernel/tile queue. |
| `hktex_splatting/rendering/frame.py` | SurfaceFrame/StageTimer used only by the retired standalone forward renderer. |
| `hktex_splatting/trainers/forward.py` | Separate forward/resident training experiment, including footprint pruning; main native/hybrid matched trainer is renderer_comparison.py. |
| `hktex_splatting/trainers/__init__.py` | Package marker for the exclusively retired historical trainer. |
| `splatting_script/experiments/train_forward.py` | Compatibility command for the exclusively retired historical trainer. |
| `splatting_script/benchmarks/legacy/train_forward.py` | Legacy wrapper for the exclusively retired historical trainer. |
| `splatting_script/benchmarks/legacy/__init__.py` | Package marker exclusively for the retired trainer entry. |
| `splatting_script/benchmarks/legacy/README.md` | Instructions exclusively for the retired trainer command; exact text archived. |
| `hktex_splatting/modules/coefficient_topology.py` | PreparedCoefficientTopology used only by the retired field.prepare_topology/legacy trainer; active ResidentTopology is preserved. |
| `hktex_splatting/rendering/dynamic_dispatch.py` | Stable CPU task dispatcher used only by retired PreparedGeometry/legacy topology methods; active hybrid grouping and selector fallback are separate and preserved. |

No production source was removed on the basis of filename alone. `scripts/visualize_kernel_faces.py` supplies active camera-basis helpers and `scripts/test_numba_raster.py` supplies the native reference rasterizer; both remain unchanged. The original `hktex/` registry includes other native model/author workflows, so their scripts/dependencies remain with the requested original implementation. Removing those original workflows would require a separate scope decision.

## Removed functions, classes and features

The [complete removed-definition inventory](hks_pipeline_only/removed_definitions.json) enumerates **82 classes/functions/methods**, including nested functions in deleted source. Definitions belonging to entire deleted files are covered by the file table and that inventory. These definitions were removed from surviving files:

| Surviving file | Removed definition |
| --- | --- |
| `hktex_splatting/cli.py` | `main.options` |
| `hktex_splatting/knn_heat/packed_support.py` | `PreparedSupport.select` |
| `hktex_splatting/modules/spectral_heat_field.py` | `LiveHeatField.prepare_topology` |
| `hktex_splatting/modules/spectral_heat_field.py` | `LiveHeatField.render` |
| `hktex_splatting/rendering/gpu_dispatch.py` | `StepCoefficientTopology` |
| `hktex_splatting/rendering/gpu_dispatch.py` | `StepCoefficientTopology.__init__` |
| `hktex_splatting/rendering/gpu_dispatch.py` | `StepCoefficientTopology.device_bytes` |
| `hktex_splatting/rendering/gpu_dispatch.py` | `StepCoefficientTopology.matches` |
| `hktex_splatting/rendering/gpu_dispatch.py` | `build_scene` |
| `hktex_splatting/rendering/gpu_dispatch.py` | `build_scene.context` |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | `TileBins` |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | `TileBins.device_bytes` |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | `bin_footprints` |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | `nearest_sources` |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | `kernel_footprints` |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | `DirectTileRenderer._native` |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | `DirectTileRenderer._native.diffuse` |
| `hktex_splatting/rendering/prepared_geometry.py` | `PreparedGeometry.matches` |
| `hktex_splatting/rendering/prepared_geometry.py` | `PreparedGeometry.dispatch` |
| `hktex_splatting/rendering/prepared_geometry.py` | `PreparedGeometry.active_pixels` |
| `hktex_splatting/rendering/resident.py` | `WorkBins` |
| `hktex_splatting/rendering/resident.py` | `WorkBins.device_bytes` |
| `hktex_splatting/rendering/resident.py` | `bin_tasks` |
| `hktex_splatting/rendering/splat_autograd.py` | `PreparedScene.build` |
| `hktex_splatting/rendering/splat_autograd.py` | `PreparedScene.build.tensor` |
| `hktex_splatting/rendering/splat_autograd.py` | `reference` |
| `hktex_splatting/rendering/triangle_heat_map.py` | `face_candidates` |
| `hktex_splatting/rendering/triangle_heat_map.py` | `polynomial_distance` |
| `hktex_splatting/rendering/triangle_heat_map.py` | `shade_numpy` |
| `hktex_splatting/rendering/triangle_splat_reference.py` | `weights` |
| `hktex_splatting/rendering/triangle_splat_reference.py` | `weights.sigmoid` |
| `hktex_splatting/rendering/triangle_splat_reference.py` | `render_numpy` |
| `hktex_splatting/triton/heat_kernel_splatting.py` | `_sort_pairs` |
| `hktex_splatting/triton/heat_kernel_splatting.py` | `_nearest` |
| `hktex_splatting/triton/heat_kernel_splatting.py` | `nearest` |
| `hktex_splatting/triton/heat_kernel_splatting.py` | `_compact` |
| `hktex_splatting/triton/heat_kernel_splatting.py` | `compact_work` |
| `hktex_splatting/triton/heat_kernel_splatting.py` | `_splat_heat` |
| `hktex_splatting/triton/heat_kernel_splatting.py` | `splat_responses` |
| `hktex_splatting/triton/resident_gpu.py` | `identify_ranges` |
| `hktex_splatting/triton/resident_gpu.py` | `render_ranges` |
| `hktex_splatting/utils/compiler_runtime.py` | `source_paths` |
| `hktex_splatting/utils/compiler_runtime.py` | `compiler_environment` |
| `hktex_splatting/utils/compiler_runtime.py` | `configure_compiler_cache` |
| `hktex_splatting/utils/compiler_runtime.py` | `configure_compiler_cache.digest` |
| `hktex_splatting/utils/compiler_runtime.py` | `runtime_record` |
| `hktex_splatting/utils/training_profile.py` | `children_cpu_seconds` |
| `hktex_splatting/utils/training_profile.py` | `timed_phase` |

Feature-level changes:

- **Projected pruning intentionally discontinued:** all `projected_bounds`/`prune_dispatch` callers, camera/mesh projection state, positive-cutoff norm bounds and per-tile embedding balls used only by the old footprint/KNN paths are gone. `footprint.py` was removed only after those runtime callers were removed. The triangle's spectral centroid/radius bounds, visibility patches and triangle/tile queue remain.
- **Only two public backends:** native/original and hybrid/2dgs. Standalone resident/splat selection, direct-native tile KNN, alternative `selection` modes and the historical forward trainer are retired.
- **Removed CLI options:** `--footprint-cutoff`, `--bounds`, `--nearest-backend`, `--selection` and legacy trainer-only flags/commands. They fail before importing GPU libraries, even with `--execute`. Positive `footprint_cutoff` is no longer supported; this is an intentional optional-feature removal, not a silent reinterpretation.
- **Removed redundant implementations:** old packed/eager standalone support builders, duplicate NumPy/torch shading oracles, legacy coefficient topology/device task builders, the cooperative patch renderer and its exclusive range/bin kernels, and the standalone full-spectral Triton emitter. The independent controls actually used by the benchmark remain.
- **Removed exclusive legacy infrastructure:** forward-frame timers/payloads and compiler cache setup/runtime-record functions used only by the retired trainer. `summarize_trace`, `_union_us` and historical `matched_policy` comparison calculations remain unchanged.
- **Retired import aliases intentionally:** old filenames from the November 15 rename no longer proxy canonical modules. Current canonical imports are supported. Root `Splatting_script` spelling compatibility for primary commands remains; it carries no alternative renderer.

The Python API `DirectTileRenderer(selection='hybrid')` remains valid. Its optional `bounds`, `nearest_backend` and `footprint_cutoff` constructor arguments are removed. The CLI no longer exposes a selection flag. New reports retain fixed `selection='hybrid'` and `footprint_cutoff=0` metadata so existing result-matching readers preserve their semantics; these fields do not enable removed features. Historical records retain their original values and matching rules. The empty footprint-projection trace range disappears with the feature; retained timing/event accounting and primary pipeline ranges remain.

## Shared dependencies preserved

| Component | Why it remains |
| --- | --- |
| Original `hktex/`, root loaders, `configs/`, author scripts | Original registered implementation, training, optimizers, preprocessing and baseline command behavior; bytes unchanged. |
| `knn_heat/packed_support.py` | Immutable triangle embedding bounds and the exact `candidate_rows` used by both the normal selector and fallback. Only its exclusive packed-to-CPU `select` adapter was retired. |
| `knn_heat/triangle_kernel_support.py` | Entire support/selector/fallback file byte-identical, including 4,096 limit, source ordering, allocations and version/overflow guards. |
| `modules/spectral_heat_field.py` | `sources`, `_chunk` and `coefficients` math unchanged; old support/trainer-only adapters removed. |
| `modules/spectral_heat_coefficients.py`, Triton coefficient kernels | GPU corner heat/Gram forward and first-order source adjoints preserved. |
| `rendering/prepared_geometry.py`, `gpu_dispatch.py`, `resident.py` | Actual fixed visibility/device layout and `ResidentTopology` used by hybrid. No replacement algorithm or unnecessary relocation was introduced. |
| `triangle_splat_reference.py`, `triangle_heat_map.py` | Keep only needed layout planning and coefficient payloads; neither is a standalone renderer now. |
| `triton/heat_kernel_splatting.py` | The real grouped queue, `_hybrid_emit` and `hybrid_forward` remain. |
| `triton/triangle_splat_forward.py`, `runtime_splat_kernels.py`, `heat_kernel_backward.py` | Actual weighting/top-30/RGB formulas and shared backward remain. |
| `splat_autograd.py`, internal resident baseline scene | Independent RGB/support/VJP/optimizer comparisons still execute the original baseline contribution forward. Hybrid inherits the same backward. This is a required diagnostic dependency, not an exposed third training implementation. |
| `_splat` / `_torch_splat_responses` eager full-spectrum oracle | Required by existing candidate-omission and spectral-response correctness diagnostics; bounded by unchanged capacity limits. No projected support/pruning remains. |
| Benchmark/profile/report modules | Native/hybrid timing, memory, PSNR, source matching, observer checks and historical result readers preserved. |
| Asset/camera/recovery utilities | Native rendering, reference images, preprocessing, support debugging and prior source/result recovery remain usable. |

Shared names such as `ResidentDispatchGeometry` identify historical origins; removing them would break current hybrid rendering. The retained internal resident/full-spectrum controls are necessary to preserve existing benchmark correctness checks, tolerances and independence. They are not kept merely for possible future use.

## Before and after structure

Before: native HKTex + hybrid + standalone resident/direct-spectral/direct-native controls + a separate projected-footprint forward trainer, cooperative prototype, duplicate CPU oracles and fifteen alias files.

After:

```text
hktex/                              original implementation and registered workflows
hktex_splatting/
  knn_heat/                         shared triangle candidates, ordered CSR, fused fallback
  modules/                          current source state and spectral coefficients
  rendering/                        fixed visibility/layout, hybrid and required controls
  triton/                           active support/coefficient/splat/backward kernels
  utils/                            assets, diagnostics, provenance and historical readers
splatting_script/
  configs/, experiments/            unchanged recipe and native/hybrid command
  benchmarks/                       matched training/profiling/reporting and numerical gates
  maintenance/                      verified source archive recovery
scripts/, configs/, root loaders    original native/preprocessing workflows retained
archive/, data/, outputs/, experiments/  historical evidence/assets/checkpoints preserved
```

The HKS packages now contain **53 Python files**, down from 78; package discovery finds **11 packages**, down from 13. There is no historical trainer directory or legacy benchmark entry area. Current READMEs and architecture describe only supported commands. The detailed native/hybrid protocol/profiler guides had obsolete command tokens removed and retain their measured-method/report semantics. Previous reports remain dated provenance rather than being rewritten as current results.

## Numerical and operational verification

| Check | Actual result | Evidence |
| --- | --- | --- |
| Actual hybrid before/after | **231 arrays exactly identical**: visibility, RGB, corner heat/Gram values, support IDs, selected entries/weights, parameter gradients, six Adam updates and first/second moments; empty-view images included | [before](hks_pipeline_only/probe_before.json), [after](hks_pipeline_only/probe_after.json) |
| Selector coverage | Legacy and requested fused paths exercised; actual 5,000-source candidate/CSR fallback matched exactly and reported `legacy_selector_fallback` | [probe log](hks_pipeline_only/probe_after.log) |
| Internal benchmark controls | **16 additional RGB/VJP arrays exactly identical** for eager full-spectrum and baseline resident controls, comparing archived pre-cleanup source against current source | [before](hks_pipeline_only/oracles_before.json), [after](hks_pipeline_only/oracles_after.json) |
| Retained Triton math | **22 complete decorated definitions byte-identical**; retired kernels belong only to discontinued features | [kernel fingerprints](hks_pipeline_only/kernel_preservation.json) |
| Source/backward/optimizer preservation | Source contractions, coefficient autograd, shared backward, whole selector/fallback module, original native loop and measured loss/optimizer steps preserved | [verification](hks_pipeline_only/verification.json) |
| Original implementation/configuration | Original HKTex, author scripts, root configurations and previous cleanup records byte-identical; original optimizer/tracer code unchanged | [verification](hks_pipeline_only/verification.json) |
| Syntax/local/runtime imports | 53 remaining HKS Python files parsed; 52 module paths imported; no missing local imports | [audit](hks_pipeline_only/audit_after.json), [log](hks_pipeline_only/verification.log) |
| Native/hybrid commands and profiles | 11 before/after previews matched after excluding only retired default options and documentation fields; original/2dgs and root namespace aliases passed | [command checks](hks_pipeline_only/command_checks.json) |
| Workflow previews | Seven default jobs and 75 scaling jobs planned; all 82 argument sets parse with execution removed; no Torch/Triton imported | [command checks](hks_pipeline_only/command_checks.json) |
| Retired interfaces | Eight retired backend/option/trainer forms rejected before GPU imports, including when `--execute` is present | [command checks](hks_pipeline_only/command_checks.json) |
| Packaging | 11 packages; existing train/compare console targets resolve; pyproject unchanged | [verification](hks_pipeline_only/verification.json) |
| Raster/preprocessing | Production raster passed 42 CPU parity cases; read-only Converse scaffold check passed | [raster](hks_pipeline_only/raster.log), [scaffold](hks_pipeline_only/scaffold.log) |
| Original eager import | Blocked at unchanged Mitsuba CUDA initialization: no CUDA-capable device detected | [exact log](hks_pipeline_only/original_import.log) |
| Documentation | Final current-guide link/reference audit recorded separately | [documentation checks](hks_pipeline_only/documentation_checks.json) |

Total before/after numerical arrays: **247 exact matches**. These are fresh tests for this cleanup, not reused results from previous reports. Independent [before/after hashes](hks_pipeline_only/protected_sources.json) record preservation of 209 original source/configuration/historical-document files; the retained legacy `candidate_rows` AST was also checked explicitly.

The probes ran actual hybrid emission, GPU coefficient wrappers and backward through Triton's **CPU interpreter** with CUDA hidden. Native numerical methods were loaded unchanged from source with deterministic exhaustive CPU L2 substituted for GPU FAISS. Synthetic grid-varying spectra and small FEM geometry were explicit fixtures. The time parameter was fixed. No fixture/helper/test suite was restored into the repository; temporary verification code stayed under `/tmp`.

The six CPU optimizer updates used Adam and checked its moments. Actual native geodesic tracing, CUDA FAISS streams, GPU atomic ordering, CUDA compilation and full trained-scene behavior were not executed. Their original sources were preserved, but they still require allocated GPU validation. No Slurm/A100 allocation was available; **no GPU smoke test, training or benchmarking job was launched or claimed**. CPU parity is not a GPU correctness or performance claim.

## Compatibility and intentional breaks

Current native/HKS checkpoint writers use original model/optimizer state dictionaries and unchanged native class paths/state keys. No trained checkpoint was loaded and rewritten by this cleanup. These workflow checkpoints remain supported; original native commands/configurations remain.

Old renamed-module imports, arbitrary whole-object HKS pickles naming those modules, legacy forward-trainer commands and standalone experimental modes are intentionally unsupported now. Current canonical imports, `hktex-splatting-train`, `hktex-splatting-compare`, `python -m hktex_splatting` and `compare_backends --backend native|hybrid|original|2dgs` remain. Old object/code environments may require the complete archived source version, not simply one restored alias.

`research_path` no longer redirects an old trainer to a missing entry point; requests for retired historical implementations raise a clear recovery error. Raw relocation identities and all earlier cleanup/recovery records remain unchanged. Historical result readers preserve old selector/cutoff/compiler metadata rather than pretending those runs used today's interface. New runs hash actual current source; old results/manifests were not relabelled or forced to match new source hashes.

## All modified existing files and Git summary

| File | Task-scoped line changes |
| --- | --- |
| `hktex_splatting/README.md` | +26 / −46 |
| `hktex_splatting/cli.py` | +4 / −27 |
| `hktex_splatting/modules/__init__.py` | +3 / −3 |
| `hktex_splatting/modules/spectral_heat_field.py` | +19 / −73 |
| `hktex_splatting/utils/paths.py` | +4 / −3 |
| `hktex_splatting/utils/training_profile.py` | +3 / −44 |
| `hktex_splatting/utils/compiler_runtime.py` | +4 / −69 |
| `hktex_splatting/knn_heat/packed_support.py` | +0 / −63 |
| `hktex_splatting/rendering/splat_autograd.py` | +2 / −92 |
| `hktex_splatting/rendering/gpu_dispatch.py` | +3 / −122 |
| `hktex_splatting/rendering/prepared_geometry.py` | +1 / −44 |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | +60 / −328 |
| `hktex_splatting/rendering/resident.py` | +6 / −37 |
| `hktex_splatting/rendering/triangle_splat_reference.py` | +3 / −39 |
| `hktex_splatting/rendering/triangle_heat_map.py` | +3 / −55 |
| `hktex_splatting/triton/resident_gpu.py` | +3 / −65 |
| `hktex_splatting/triton/heat_kernel_splatting.py` | +3 / −164 |
| `splatting_script/README.md` | +50 / −165 |
| `splatting_script/experiments/__init__.py` | +3 / −3 |
| `splatting_script/experiments/compare_backends.py` | +4 / −11 |
| `splatting_script/benchmarks/pipeline_workflow.py` | +2 / −2 |
| `splatting_script/benchmarks/renderer_comparison.py` | +63 / −88 |
| `docs/HKTEX_RESEARCH_TRAINING.md` | +12 / −6 |
| `docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md` | +119 / −727 |
| `docs/BENCHMARK_PROFILER.md` | +9 / −2 |
| `README.md` | +88 / −130 |

The [scope inventory](hks_pipeline_only/scope_changes.json) records **26 deletions and 26 modifications**, with 497 added and 3574 removed lines relative to the exact starting working tree. Added files are this report, audit/plan/verification records under `docs/hks_pipeline_only/`, and the sealed recovery payload/manifest/seal. No new renderer, dependency or standalone testing infrastructure was introduced.

[Initial status](hks_pipeline_only/git_status_before.txt), [ending status](hks_pipeline_only/git_status_after.txt) and [ordinary Git diff stat](hks_pipeline_only/git_diff_stat.txt) retain context. The checkout already had substantial staged/unstaged changes and untracked canonical sources. Ordinary Git diff includes that prior work and omits some untracked-file removals; the task-scoped inventory and sealed preimages are authoritative. Original index bytes remain unchanged; nothing was staged, committed or pushed.

## Recovery and remaining risks

Archive: `archive/three_backends/hks_pipeline_only_20261116/`. All 89 preimages were verified before removal. The retained recovery tool validates seals, paths, membership, regular files and hashes and refuses overwrites:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.maintenance.recovery archive/three_backends/hks_pipeline_only_20261116
"$HK_PY" -B -m splatting_script.maintenance.recovery archive/three_backends/hks_pipeline_only_20261116 --list
# Restore a selected missing source file as data; this alone does not re-enable a feature.
"$HK_PY" -B -m splatting_script.maintenance.recovery archive/three_backends/hks_pipeline_only_20261116 \
  --restore hktex_splatting/rendering/footprint.py
```

A compatible historical environment may need edited-file preimages as well as deleted files. Restore the complete version in an isolated copy and review newer user edits; do not reset/stash this dirty checkout or overwrite current work. A supplemental starting snapshot is `/tmp/hks-pipeline-only-wz5utsyw/before/`; temporary files may not persist. The archive tarball is covered by the existing ignore rule, so preserve/copy the payload alongside its manifest/seal for durable recovery; metadata alone cannot restore source.

Remaining limitations:

- Real A100 native/hybrid correctness, the compiled fused-selector regime, backward atomics and on-mesh optimizer execution remain allocation-dependent checks. Existing runtime assertions/fallbacks/tolerances remain.
- Benchmark full-spectrum validation remains capacity-bounded and can reject large workloads, as before; internal controls were not weakened to make cleanup pass.
- Optional features and old object/module interfaces listed above are deliberately discontinued. Users of those experiments should use the archive/source version rather than assuming command compatibility.
- Original author model/benchmark/tuning commands and provenance/recovery utilities remain with native HKTex and historical research evidence. Further narrowing those native workflows requires a separate compatibility/research scope decision; no unrelated original files were deleted speculatively.
- Cache/source identities and setup memory/work change when unused code and geometry bounds disappear. No training-speed, memory-reduction or convergence improvement is claimed without measurement.
