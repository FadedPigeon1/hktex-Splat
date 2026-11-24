# HKS file rename and documentation report

Date: November 15, 2026 (America/Chicago). This is a naming/documentation refactor of Hybrid Heat-Kernel Splatting and its retained research controls. No algorithm, kernel primitive, equation, optimizer or benchmark calculation was replaced. No commit, push, reset, stash or Git-index modification was performed.

## Scope and audit

The four requested guides were read before source edits: the main README, architecture guide, final cleanup report and test cleanup report. The [complete rename plan](hks_file_rename/PLAN.md) and [dependency/compatibility plan](hks_file_rename/plan.json) were recorded before moving implementations.

All **63 original Python files** in `hktex_splatting/` and `splatting_script/` were audited. The machine-readable [before inventory](hks_file_rename/audit_before.json) and [after inventory](hks_file_rename/audit_after.json) list imports (including nested imports), every class/function/method, source lines, docstrings and calls made within definitions. The module table below classifies their actual responsibilities. Existing GPU safety guards and selector fallbacks were reviewed as executable code, not inferred from README descriptions.

Fifteen implementation bodies moved to descriptive filenames. The fifteen old files remain small compatibility aliases; there are now **78 Python files** across the two packages. Aliases contain no duplicated renderer or kernel implementation. All 63 original modules and all aliases have short purpose/pipeline/API docstrings. Selected previously undocumented container/autograd classes and public planning functions also gained concise docstrings. Clearly named modules retain their paths; no module was split, class/function renamed, or deleted test suite restored.

## Every implementation rename

| Previous path (retained alias) | New implementation path | Why the new name is clearer |
| --- | --- | --- |
| `hktex_splatting/rendering/direct_tiles.py` | [hktex_splatting/rendering/heat_kernel_splatting.py](../hktex_splatting/rendering/heat_kernel_splatting.py) | Names the main spectral renderer instead of a low-level tile detail. |
| `hktex_splatting/triton/direct_tiles.py` | [hktex_splatting/triton/heat_kernel_splatting.py](../hktex_splatting/triton/heat_kernel_splatting.py) | Identifies the GPU heat-kernel emitter alongside its Python renderer. |
| `hktex_splatting/modules/live_field.py` | [hktex_splatting/modules/spectral_heat_field.py](../hktex_splatting/modules/spectral_heat_field.py) | Makes the learned spectral heat field explicit. |
| `hktex_splatting/modules/coefficients.py` | [hktex_splatting/modules/spectral_heat_coefficients.py](../hktex_splatting/modules/spectral_heat_coefficients.py) | Identifies heat/Gram spectral coefficients rather than generic coefficients. |
| `hktex_splatting/triton/coefficient_gpu.py` | [hktex_splatting/triton/spectral_heat_coefficients.py](../hktex_splatting/triton/spectral_heat_coefficients.py) | Names the same spectral coefficient stage in the GPU-kernel namespace. |
| `hktex_splatting/knn_heat/resident_support.py` | [hktex_splatting/knn_heat/triangle_kernel_support.py](../hktex_splatting/knn_heat/triangle_kernel_support.py) | Makes triangle/kernel selection visible without implying resident-only use. |
| `hktex_splatting/rendering/core.py` | [hktex_splatting/rendering/triangle_splat_reference.py](../hktex_splatting/rendering/triangle_splat_reference.py) | Names the triangle-splat CPU reference and planner instead of an unspecified core. |
| `hktex_splatting/rendering/differentiable.py` | [hktex_splatting/rendering/splat_autograd.py](../hktex_splatting/rendering/splat_autograd.py) | Identifies the shared autograd contract and inherited backward. |
| `hktex_splatting/triton/gpu.py` | [hktex_splatting/triton/triangle_splat_forward.py](../hktex_splatting/triton/triangle_splat_forward.py) | Identifies triangle-splat forward emission, ranking and RGB reconstruction. |
| `hktex_splatting/triton/backward.py` | [hktex_splatting/triton/heat_kernel_backward.py](../hktex_splatting/triton/heat_kernel_backward.py) | Identifies heat-kernel contribution/RGB adjoints. |
| `hktex_splatting/triton/runtime_counts.py` | [hktex_splatting/triton/runtime_splat_kernels.py](../hktex_splatting/triton/runtime_splat_kernels.py) | Shows that the module contains rendering/adjoint kernels with runtime bounds, not a counter utility. |
| `hktex_splatting/rendering/kernel_centric.py` | [hktex_splatting/rendering/triangle_heat_map.py](../hktex_splatting/rendering/triangle_heat_map.py) | Names its triangle coefficient map, support oracle and barycentric shading. |
| `hktex_splatting/rendering/renderer.py` | [hktex_splatting/rendering/triangle_splat_renderer.py](../hktex_splatting/rendering/triangle_splat_renderer.py) | Identifies the retained generic triangle-splat renderer, distinct from the hybrid entry. |
| `hktex_splatting/rendering/kernel_centric_raster.py` | [hktex_splatting/rendering/mesh_visibility_raster.py](../hktex_splatting/rendering/mesh_visibility_raster.py) | Names fixed-mesh visibility rasterization instead of suggesting heat evaluation. |
| `splatting_script/benchmarks/direct_tiles.py` | [splatting_script/benchmarks/renderer_comparison.py](../splatting_script/benchmarks/renderer_comparison.py) | Names the matched renderer training/comparison harness rather than a tile implementation. |

Original `hktex/` files were not renamed or edited. Stable CLI/configuration paths such as `cli.py`, `compare_backends.py`, `train_forward.py` and `converse_training.yaml` remain. `resident.py` and `triton/resident_gpu.py` retain established control names; their updated docstrings explicitly identify shared hybrid dependencies.

## Pipeline and documentation

The [updated architecture guide](HEAT_KERNEL_SPLATTING_ARCHITECTURE.md) contains a current call/stage map, a diagram and a file-reference table with new/previous paths, principal APIs and pipeline roles. It distinguishes native pointwise HKTex from triangle-supported hybrid HKS.

The actual order is fixed mesh visibility → spectral candidate selection → ordered kernel–triangle support → three-corner heat/Gram coefficients → tile queue → GPU interpolation/attenuation/filtering and contribution emission → strongest-30 selection → HKTex weighted RGB. Interpolation and weighting happen inside the GPU emitter after scheduling, not in a newly added standalone pass. RGB reconstruction uses the denominator floor, learned mean and clipping; it is not 3DGS alpha compositing.

Current guides updated:

- `README.md`: principal files/navigation, current implementation links and compatibility explanation; corrected its diffusion-time description to match the already-configured fixed model time. An explicit time override remains differentiable; no training behavior changed.
- `hktex_splatting/README.md` and `splatting_script/README.md`: canonical implementation paths and commands.
- `docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md`: current stage map/diagram, rename table and canonical source references; detailed equations/historical measurements retained.
- `docs/BENCHMARK_PROFILER.md` and `docs/HKTEX_RESEARCH_TRAINING.md`: canonical harness references.

Previous cleanup reports, raw inventories, saved benchmark results, source hashes and recovery archives remain unchanged. Historical filenames in those records still resolve through aliases where applicable. They were not silently rewritten to claim earlier runs used the renamed files.

## Imports, dynamic paths and provenance

Every live import and lazy public export inside the two packages now targets canonical implementation modules. Import-local names such as `gpu`, `coefficient_gpu`, `runtime_counts` and `direct_tiles` remain aliases where that preserves the executable code and profiler hooks. Changed long import statements were formatted for readability. No root/preprocessing script imports a moved module through a stale implementation path.

`_legacy_paths.MIGRATIONS` retains every original identity key; eleven relocation **values** now point to canonical implementation files. The recorded original cleanup manifests themselves remain untouched. `compiler_runtime.source_paths`, `paths.implementation_sources` and research provenance already discover source files recursively; checks confirm they cover the canonical implementations and required aliases. No hashing or benchmark matching rule was weakened.

`pyproject.toml` required no edit: the train/compare targets remain stable and package discovery automatically includes the new modules and aliases. Expanded discovery still finds **13 packages**, and both console targets resolve to callables. No dependency was added/removed and no editable installation was changed.

## Compatibility and checkpoints

- **Old imports/private symbols:** each old path imports the canonical module and registers that same object in `sys.modules`. Old/new imports therefore share classes, functions, private kernels and mutable globals. This avoids independent wrapper copies or lost profiler patches.
- **Profiler patching:** all fifteen alias pairs were checked for exact module/symbol identity and patch visibility. Profile stage labels, saved report names and timing calculations are unchanged.
- **CLI commands:** hybrid/native plus `original`, `2dgs`, `splat`, resident and legacy trainer controls are preserved. The old low-level benchmark module remains executable; new and old commands work under both lowercase `splatting_script` and the existing `Splatting_script` namespace. Those four low-level previews produced identical parsed output; old/new direct script previews also matched in isolated Python.
- **Native checkpoints:** current writers/loaders use original HKTex model/optimizer state dictionaries and kernel-count metadata; original model class paths, state keys and tensor shapes were untouched. Research snapshots still have their existing archival/no-resume policy.
- **Historical HKS pickle globals:** a trusted fixture saved before moving modules contained 52 top-level class/function references from the moved non-Triton modules. All resolved to the exact canonical objects after the refactor. Historical `FaceMap`/`Dispatch` instances and a CPU model/Adam state-dictionary fixture also loaded successfully. No existing user checkpoint was deserialized or rewritten for this check.
- **Forward compatibility limit:** newly pickled HKS objects naturally record canonical module names and may require this refactored checkout. Old checkpoint imports remain supported; this is not a promise that a pre-refactor checkout can import new filenames. Arbitrary pickled JIT/runtime objects were not exhaustively validated.

## Verification results

| Check | Result | Evidence |
| --- | --- | --- |
| All original executable module ASTs | 63 preserved after removing docstrings and undoing only renamed import/path spellings | [AST fingerprints](hks_file_rename/executable_ast_preservation.json) |
| Decorated Triton GPU definitions | All 28 complete decorated definitions are byte-identical, including JIT options and arithmetic | [kernel source fingerprints](hks_file_rename/triton_kernel_preservation.json) |
| Original HKTex and prior cleanup/recovery records | Byte-identical to the starting working tree, including existing user edits | [verification](hks_file_rename/verification.json) |
| Python syntax/docstrings/local imports | 78 files checked; no missing local import targets | [after audit](hks_file_rename/audit_after.json) |
| Runtime imports | 77 module paths loaded (`__main__` checked through command previews) | [verification log](hks_file_rename/verification.log) |
| Compatibility/private-symbol/profiler patch identity | All 15 old/new pairs passed | [verification](hks_file_rename/verification.json) |
| CPU numerical comparison | 40 arrays exactly identical: independent CPU RGB oracles, shared splat forward, six input gradients, Adam updates/first moments, specialized and runtime count variants | [before](hks_file_rename/probe_before.json), [after](hks_file_rename/probe_after.json) |
| Historical serialization | 52 globals and two container instances resolved; state-dictionary fixture loaded | [after receipt](hks_file_rename/probe_after.json) |
| Before/after training/benchmark/profiler previews | 14 preserved control outputs; only module-docstring `scope` text excluded from comparison | [command checks](hks_file_rename/command_checks.json) |
| Canonical/historical low-level benchmark commands | Four lowercase/historical-namespace combinations passed with identical output | [command checks](hks_file_rename/command_checks.json) |
| Workflow/scaling previews | Seven default jobs and 75 scaling jobs planned; no GPU work launched | [command checks](hks_file_rename/command_checks.json) |
| Public hybrid API/backward | Default `selection='hybrid'` and inherited `_Splat.backward` unchanged | [verification](hks_file_rename/verification.json) |
| Packaging/source provenance | 13 packages; stable entry points; canonical files and aliases included in source discovery | [verification](hks_file_rename/verification.json) |
| Required production raster utility | 42 CPU parity cases passed | [raster log](hks_file_rename/raster.log) |
| Converse preprocessing scaffold | Read-only required-input check passed | [scaffold log](hks_file_rename/scaffold.log) |
| Original eager HKTex registry import | Reached unchanged Mitsuba CUDA setup; unavailable without a CUDA-capable device | [original import log](hks_file_rename/original_import.log) |
| Documentation links/references | Checked by the final maintained-guide audit | [documentation checks](hks_file_rename/documentation_checks.json) |

The numerical probe ran actual retained shared splat kernels through Triton's **CPU interpreter**, with CUDA hidden. It is not GPU execution or a complete real-scene hybrid/FAISS training run. Temporary probe/verification helpers and synthetic fixtures stayed under `/tmp`; no standalone testing infrastructure was recreated in the repository. The retired 208-test suite was not restored or claimed as a fresh result for this task.

No allocated Slurm/A100 GPU was available, so no GPU smoke, training or performance job ran. Native eager import failed at unchanged `mi.set_variant("cuda_ad_rgb")` with `no CUDA-capable device is detected`. Native CLI routing/CPU imports and original-source preservation passed; native/hybrid real GPU execution remains an allocation-dependent validation step.

## Complete module responsibility table

Paths below are current implementations, not the compatibility proxies. Full definition/import/call inventories are linked above. Some shared modules span multiple stages; these are not new algorithms or separate passes.

| Current file | Pipeline role | Principal API | Responsibility |
| --- | --- | --- | --- |
| `hktex_splatting/__init__.py` | Rendering and training API; importing this namespace starts no GPU work | `DirectTileRenderer, TileSurface, LiveHeatField, __getattr__` | Expose the public HKS renderer and field types through lazy imports. |
| `hktex_splatting/__main__.py` | Rendering/training entry point; dry unless execution is requested | `main` | Run the hybrid-first package command through cli.main. |
| `hktex_splatting/cli.py` | Rendering and training command routing | `main` | Route normal training to hybrid and preserve explicit historical trainer flags. |
| `hktex_splatting/knn_heat/__init__.py` | Kernel candidate selection and triangle support construction | `packed_support, triangle_kernel_support` | Group spectral triangle candidate and support selection modules. |
| `hktex_splatting/knn_heat/packed_support.py` | Kernel candidate selection; geometry is fixed but source support is refreshed | `PreparedSupport` | Prepare visible triangle embedding bounds and fixed-size candidate staging. |
| `hktex_splatting/knn_heat/triangle_kernel_support.py` | Kernel candidate selection; triangle support construction | `ResidentSupportGeometry, FusedResidentSupportGeometry, DeviceSupport` | Fresh triangle candidate selection and device CSR, including fused selector fallback. |
| `hktex_splatting/modules/__init__.py` | Spectral kernel representation and differentiable corner heat evaluation | `spectral_heat_field, spectral_heat_coefficients, coefficient_topology` | Group learned source-state and spectral coefficient adapters. |
| `hktex_splatting/modules/coefficient_topology.py` | Triangle support construction during rendering and training | `PreparedCoefficientTopology` | Own integer triangle/source topology shared by dispatch and coefficients. |
| `hktex_splatting/modules/spectral_heat_coefficients.py` | Corner heat evaluation; GPU backward propagation | `_Coefficients, fused_coefficients` | Autograd wrapper for fused spectral heat and Gram coefficients. |
| `hktex_splatting/modules/spectral_heat_field.py` | Spectral kernel representation; corner heat evaluation | `LiveHeatField, SourceState` | Learned source state and differentiable three-corner spectral coefficients. |
| `hktex_splatting/rendering/__init__.py` | Rendering, tile scheduling and shared autograd | `heat_kernel_splatting, splat_autograd, resident` | Group hybrid rendering, fixed visibility and retained reference controls. |
| `hktex_splatting/rendering/cameras.py` | Shared rendering/training/benchmark geometry inputs | `camera, cameras` | Provide the recorded fixed training and held-out camera definitions. |
| `hktex_splatting/rendering/triangle_splat_reference.py` | Tile scheduling; contribution weighting and color blending | `Dispatch, kernel_face_order, dispatch, weights` | CPU triangle/tile dispatch planner and independent weighted RGB oracle. |
| `hktex_splatting/rendering/splat_autograd.py` | GPU backward propagation; training | `PreparedScene, _Splat, splat, reference` | Shared splat autograd contract and backward inherited by hybrid. |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | Tile scheduling; barycentric interpolation; contribution weighting | `DirectTileRenderer, TileSurface, triangle_tile_work, _HybridSplat` | Hybrid renderer orchestration, kernel/triangle/tile queue and eager spectral controls. |
| `hktex_splatting/rendering/dynamic_dispatch.py` | Tile scheduling for retained rendering and training controls | `stable_kernel_face_order, compact_tasks` | Build stable integer task order and compact changing triangle/tile work. |
| `hktex_splatting/rendering/fast_raster.py` | CPU mesh visibility before spectral rendering; preserves pixel ties | `_fast_kernel, StaticRaster` | Rasterize fixed triangles with the preserved float64 clipping and depth rules. |
| `hktex_splatting/rendering/footprint.py` | Optional rendering/reference pruning; default hybrid cutoff is zero | `projected_bounds, prune_dispatch` | Bound projected intrinsic heat level sets per supported triangle. |
| `hktex_splatting/rendering/frame.py` | Shared payload/timing utilities for the retained forward rendering control | `SurfaceFrame, StageTimer` | Carry a surface frame and collect diagnostic stage timing. |
| `hktex_splatting/rendering/gpu_dispatch.py` | GPU tile scheduling with stable bounded fallback during rendering/training | `DeviceDispatchGeometry, DeviceDispatch, StepCoefficientTopology, build_scene` | Upload fixed face/tile layout and expand current device dispatch tasks. |
| `hktex_splatting/rendering/triangle_heat_map.py` | Triangle support construction; barycentric interpolation | `face_candidates, polynomial_distance, FaceMap, shade_numpy` | Sparse triangle coefficient container, candidate oracle and CPU shading. |
| `hktex_splatting/rendering/mesh_visibility_raster.py` | Shared utilities; mesh visibility | `BVHRaster` | Fixed-mesh BVH visibility and camera-frustum traversal. |
| `hktex_splatting/rendering/native_reference.py` | Native pointwise rendering reference; uses original prepare/diffuse/forward calls | `render, validate_knn_settings, main` | Evaluate the original trained HKTex field at visible pixel centers. |
| `hktex_splatting/rendering/prepared_geometry.py` | Shared visibility and tile layout for rendering and training | `PreparedGeometry` | Prepare immutable camera/pixel geometry while refreshing learned support. |
| `hktex_splatting/rendering/reference_mesh.py` | Rendering/benchmark target images; no learned kernel evaluation | `texture_sample, render, main` | Sample the original mesh UV texture at triangle-renderer pixel centers. |
| `hktex_splatting/rendering/triangle_splat_renderer.py` | Rendering reference; Triton splatting | `ForwardRenderer` | Retained generic triangle contribution forward renderer. |
| `hktex_splatting/rendering/resident.py` | Shared support/geometry for hybrid plus the separate resident rendering control | `ResidentDispatchGeometry, ResidentTopology, bin_tasks, build_scene` | Build the retained resident triangle renderer topology and work ranges. |
| `hktex_splatting/trainers/__init__.py` | Training and optimization; normal commands route through hktex_splatting.cli | `forward` | Group retained image-space training implementations. |
| `hktex_splatting/trainers/forward.py` | Historical image-space training entry; dry unless --execute is supplied | `parser, main` | Train the retained forward/resident control with original HKTex optimizers. |
| `hktex_splatting/triton/__init__.py` | Rendering and backward propagation; importing the namespace launches no kernel | `heat_kernel_splatting, spectral_heat_coefficients, resident_gpu` | Group GPU spectral, support, splatting and adjoint kernels. |
| `hktex_splatting/triton/heat_kernel_backward.py` | GPU backward propagation | `reconstruct_backward, emit_backward` | Selected contribution and RGB adjoints with compile-time pair counts. |
| `hktex_splatting/triton/spectral_heat_coefficients.py` | Corner heat evaluation; GPU backward propagation | `coefficients_forward, coefficients_backward` | Triton corner heat/Gram contractions and source-state adjoints. |
| `hktex_splatting/triton/heat_kernel_splatting.py` | Triton splatting; tile scheduling | `hybrid_forward, _hybrid_emit, group_triangle_work, splat_responses` | GPU hybrid emission and grouped kernel/tile work, with full-spectral reference emission. |
| `hktex_splatting/triton/triangle_splat_forward.py` | Triton splatting; contribution weighting and color blending | `emit, merge, reconstruct` | Generic triangle-record emission, segmented top-k and RGB reconstruction. |
| `hktex_splatting/triton/resident_gpu.py` | Shared GPU candidate/support construction plus resident control emission | `support_distances, select_support_rows, pack_support, render_ranges` | Select/scan/pack triangle support and render retained resident work ranges. |
| `hktex_splatting/triton/runtime_splat_kernels.py` | Contribution weighting and color blending; GPU backward propagation | `reconstruct, reconstruct_backward, emit_backward` | RGB reconstruction and adjoints with runtime pair bounds. |
| `hktex_splatting/utils/__init__.py` | Shared rendering, training and benchmark support | `paths, reference_assets, experiment_io, support_debug` | Group repository paths, assets, diagnostics and experiment utilities. |
| `hktex_splatting/utils/_legacy_paths.py` | Shared provenance/recovery metadata; recorded strings are not live imports | `MIGRATIONS` | Map historical source identities to current implementation targets; preserve original keys. |
| `hktex_splatting/utils/compiler_runtime.py` | Training/benchmark provenance; standard-library imports only | `source_paths, configure_compiler_cache, runtime_record, matched_policy` | Record compiler policy, source hashes and isolated/shared cache identities. |
| `hktex_splatting/utils/experiment_io.py` | Shared training/benchmark I/O and existing GPU allocation guards | `input_identities, configure_cache, gpu_guard, protected_snapshot` | Manage portable experiment inputs, preservation checks and unique outputs. |
| `hktex_splatting/utils/metrics.py` | Benchmark image-quality evaluation and reporting | `metrics, validate_alignment, evaluate, main` | Evaluate foreground RGB fidelity with missing pixels counted as black. |
| `hktex_splatting/utils/paths.py` | Shared rendering/training paths and research provenance; archives stay data | `implementation_sources, research_path, recorded_source_path, recorded_identity` | Resolve current source locations and verified historical source evidence. |
| `hktex_splatting/utils/reference_assets.py` | Shared preprocessing/rendering/benchmark I/O; import does not initialize CUDA | `load_cpu_mesh, raster_surface, camera, input_identity` | Load fixed mesh/camera inputs and save aligned reference-render artifacts. |
| `hktex_splatting/utils/support_debug.py` | Opt-in rendering/training diagnostics; runtime safety checks remain active | `support_errors, SupportDebugger` | Check live support rows/CSR and save failure evidence without repairing it. |
| `hktex_splatting/utils/training_profile.py` | Shared training/benchmark profiling; nested intervals are accounted explicitly | `timed_phase, summarize_trace` | Measure synchronized phases and correlate CUDA trace activity. |
| `splatting_script/__init__.py` | Training, benchmarking and shared research utilities | `experiments, benchmarks, maintenance` | Group reproducible HKS experiment, benchmark and recovery commands. |
| `splatting_script/benchmarks/__init__.py` | Benchmarking/profiling and controlled training; no work starts at import | `renderer_comparison, pipeline_workflow, pipeline_profile, pipeline_report` | Group matched renderer comparisons, profiling and report readers. |
| `splatting_script/benchmarks/compare_training.py` | Benchmark reporting and workflow/scaling command routing; standard library only | `compare, native_split_psnr, main` | Compare completed native/hybrid training at measured image quality. |
| `splatting_script/benchmarks/renderer_comparison.py` | Training and optimization; benchmarking and profiling | `parser, main, _execute, JITCacheDiagnostics` | Matched native/hybrid training, embedded correctness gates and timing. |
| `splatting_script/benchmarks/legacy/__init__.py` | Training command compatibility; renderer implementations remain shared | `train_forward` | Keep explicit command paths for historical training implementations. |
| `splatting_script/benchmarks/legacy/train_forward.py` | Historical training entry; delegates to hktex_splatting.trainers.forward | `main, parser` | Expose the preserved forward/resident trainer under the legacy command area. |
| `splatting_script/benchmarks/pipeline_profile.py` | Diagnostic benchmarking/profiling with an observer-on/off RGB/VJP gate | `PipelineProfiler, validate_observer, tensor_storages` | Observe existing renderer, optimizer and GPU calls without replacing their math. |
| `splatting_script/benchmarks/pipeline_report.py` | Benchmark reporting; no rendering or GPU job is started by readers | `collect, report, markdown, plots, main` | Build failure-aware timing, memory, quality and scaling reports from saved runs. |
| `splatting_script/benchmarks/pipeline_trace.py` | Benchmark profiling analysis using the standard library | `analyze_trace, classify, duration, distribution` | Analyze CPU/CUPTI traces using correlated intervals and explicit overlap rules. |
| `splatting_script/benchmarks/pipeline_workflow.py` | Benchmark/training workflow; previews are GPU-free and execution uses an allocation | `plan, execute, classify_failure, main` | Plan and orchestrate matched initialization, pilots, speed runs and profiles. |
| `splatting_script/benchmarks/profile_training_report.py` | Benchmark/profiling report review | `review, main` | Read synchronized training profiles without importing Torch or probing GPUs. |
| `splatting_script/benchmarks/training_research.py` | Training/optimization bookkeeping and benchmark reproducibility | `run_training, provenance, aggregate_quality, compare_runs` | Record matched training updates, quality, snapshots and source provenance. |
| `splatting_script/configs/__init__.py` | Training/benchmark configuration resources; no numerical implementation | `converse_training.yaml, authors_source.json` | Package the recorded training recipe and original source declarations. |
| `splatting_script/experiments/__init__.py` | Rendering/training command entry points | `compare_backends, train_forward` | Group main native/hybrid commands and historical trainer compatibility. |
| `splatting_script/experiments/compare_backends.py` | Rendering/training entry; hybrid is default and preview is GPU-free | `BACKENDS, main` | Map hybrid/native and historical backend names to the matched comparison harness. |
| `splatting_script/experiments/train_forward.py` | Training command compatibility; no duplicate trainer implementation | `main, parser` | Preserve the historical trainer command by delegating to the legacy entry. |
| `splatting_script/maintenance/__init__.py` | Shared repository recovery; no renderer or GPU dependency | `recovery` | Group recovery tools for preserved research source archives. |
| `splatting_script/maintenance/recovery.py` | Shared recovery utility; archived implementations are never imported | `records, verify, safe_path, main` | Verify, list and selectively restore sealed research-source archives. |

## Task-scoped files and Git review

- **15 added implementation paths** and **63 modified existing Python files** (including the 15 old-path aliases); six maintained guides modified.
- Added this report and `docs/hks_file_rename/` plan, source inventories, role table, AST/preservation receipts, CPU/CLI/link-check receipts and Git status/diff evidence. These are documentation/audit data, not test runners or fixtures.
- No original HKTex, model/checkpoint, mesh, spectral asset, recorded result or previous cleanup/recovery artifact was deleted or rewritten. No files were deleted by this refactor.
- `pyproject.toml`, the root historical namespace module and existing config resources remain unchanged; their behavior was checked instead of forcing unnecessary edits.

The [task-scoped change inventory](hks_file_rename/scope_changes.json) lists all 69 pre-existing modified files and 15 added implementation paths. [Initial status](hks_file_rename/git_status_before.txt), [ending status](hks_file_rename/git_status_after.txt) and [ordinary Git diff stat](hks_file_rename/git_diff_stat.txt) preserve repository context. The repository already had substantial staged/unstaged and untracked work. Ordinary Git diff includes that earlier work and may omit new/untracked canonical paths; it is not the refactor's complete file inventory. The original index bytes are unchanged. No automatic staging, commit or push occurred.

Supplemental pre-refactor working copies remain at `/tmp/hktex-rename-xdn8esf2/before/`, with original hashes and index/status receipts beside them. These temporary copies may not persist; preserve them separately if exact pre-refactor doc/source bytes are needed. Do not reset or stash the dirty repository to undo a rename. Use the explicit mapping and task-scoped diff; review any newer user edits before restoring a working copy. Prior cleanup recovery archives remain at their recorded locations.

## Remaining risks and follow-up

1. Actual GPU compilation, fused-selector dispatch/fallback execution, hybrid emission/backward atomics and full on-mesh optimization still need the retained A100 benchmark correctness gates. CPU interpreter parity and AST preservation do not establish those GPU results.
2. Filenames/docstrings alter source/provenance and compiler cache identities. New runs must record new identities and keep existing matched-control checks; no old benchmark result was relabelled. A future GPU run may need recompilation. No performance improvement is claimed.
3. Existing pickle import paths are preserved, but downstream code that depends on exact `__file__`, `__module__`, traceback filenames or the text of an old source file will see the canonical location (or the compatibility proxy on disk). Use the canonical table for source inspection.
4. Keep the small aliases while external research scripts/checkpoints may depend on them. They share module objects and add no alternative numerical implementation. Future removal would require a separate compatibility review.
