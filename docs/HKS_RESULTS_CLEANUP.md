# HKS-only results cleanup

## Proposed deletions (written before removal)

Hash-verified current source preimages, including uncommitted work: `archive/hks_results_cleanup_20261123/preimages.tar.gz`. Imports, CLI/configuration references, dynamic hooks and historical docs were checked. Git index is preserved.

| File | Dependency evidence and reason |
|---|---|
| `hktex_splatting/training/adaptive_density.py` | Retired custom HKS policy cluster: only entry is a no-op installer for supported original policies; CLI already rejects hks policies. Remove installer hook first. Incoming imports: none. |
| `hktex_splatting/training/custom_training.py` | Retired custom HKS policy cluster: only entry is a no-op installer for supported original policies; CLI already rejects hks policies. Remove installer hook first. Incoming imports: hktex_splatting.training.uv_training. |
| `hktex_splatting/training/kernel_optimizer.py` | Retired custom HKS policy cluster: only entry is a no-op installer for supported original policies; CLI already rejects hks policies. Remove installer hook first. Incoming imports: hktex_splatting.training.custom_training. |
| `hktex_splatting/training/occupancy_evaluation.py` | Unused occupancy router: no callers/imports; current CLI rejects nonzero occupancy. Incoming imports: none. |
| `hktex_splatting/training/density_controllers/__init__.py` | Retired custom HKS policy cluster: only entry is a no-op installer for supported original policies; CLI already rejects hks policies. Remove installer hook first. Incoming imports: hktex_splatting.training.adaptive_density, hktex_splatting.training.custom_training. |
| `hktex_splatting/training/density_controllers/residual_cost.py` | Retired custom HKS policy cluster: only entry is a no-op installer for supported original policies; CLI already rejects hks policies. Remove installer hook first. Incoming imports: hktex_splatting.training.density_controllers. |
| `hktex_splatting/training/density_controllers/transactions.py` | Retired custom HKS policy cluster: only entry is a no-op installer for supported original policies; CLI already rejects hks policies. Remove installer hook first. Incoming imports: hktex_splatting.training.density_controllers.residual_cost. |
| `hktex_splatting/knn_heat/persistent_association.py` | Unused certified-cache implementation: no constructor/imports; current CLI requires none. Incoming imports: none. |

The six custom-policy modules contain unique historical research. Their exact source is archived before live removal. Custom whole-object pickles may require the archived modules. Supported state-dictionary checkpoints and all historical results remain.

Preserve native HKTex, graph and spectral HKS, triangle support/coefficients, tile scheduling/splatting, custom backward, original density/optimizers and bridges, validation gates, preprocessing, benchmark/profiling/recovery tools, meshes/eigensystems/checkpoints/results and previous recovery records. No active footprint module exists to delete.

## Additional function-level review (before changes)

All77 HKS/script Python modules were parsed and their definitions/imports inventoried in `archive/hks_results_cleanup_20261123/file_audit.json`.

- Remove unused `FixedTriangleBounds` and its exclusive optional cache wiring from `packed_support.py`: no constructor/import/CLI callers; retained triangle bounds still use the same float64 arithmetic.
- Remove unused `group_triangle_work` wrapper and `_group_triangle_work` GPU kernel: no dispatch calls; the profiler only lists the old kernel name. Keep `group_sorted_triangle_work`, scan/pack and all active tile kernels; update the observer and current file tables.
- Remove uncalled `diagnostic_render_acceptance` and its unused re-export/import. Strict GPU/eager/native gates are retained; no tolerances change.
- Simplify custom-policy hooks in HKS trainer, inactive candidate-cost reporting, and dormant association-cache hooks. Original-policy outputs and historical result schema remain.
- Simplify the bounded full-spectrum correctness oracle's always-eager coefficient helper; its only two call sites already disable fusion. Main GPU coefficients and backward are untouched.
- Retain all other modules: runtime APIs, CLI commands, CPU/GPU geometry references, profiler/quality readers, preprocessing, source provenance or recovery dependencies. Original HKTex's registered imports are outside this cleanup.

Unique retired implementations remain in the hash-verified source archive. Historical documentation will be marked retired and linked to recovery rather than erased.

## Completed cleanup

Reviewed **77** HKS/workflow Python modules; **69** remain. Deleted the8 reviewed
retired modules, simplified8 surviving source modules, and removed29 class/function
names. Source diff relative to the actual starting working tree: **+31/-593
lines (net-562)**. No source files moved. No new Python modules or test
runners were created. No dependency was removed: the remaining packages still
serve the native baseline, preprocessing, CPU raster checks and HKS workflows.

Original HKTex was outside the mutation scope. All **89 native files**, original
scripts/configs/root loaders/README and pyproject match their starting hashes:
**154 protected non-HKS files unchanged**. The pre-existing Git modification to
`hktex/modules/tracer.py` was not made or discarded by this cleanup.

### Refactors performed

- HKS trainer validates original policy choices, removes its no-op experimental
  installer/imports, and simplifies inactive density handling. The original
  RecordedUvTrainer class and original controller/optimizer code remain unchanged.
- Fused reference selection always refreshes candidates; obsolete persistent-cache
  lookup/capture/stat hooks and the extra private wrapper were removed. Existing
  fused/tiled selectors, source-count guards and all fallbacks remain.
- PreparedSupport computes the same direct triangle bounds without an unused
  injectable all-mesh bound cache. Graph fixed-hop lookup is a DIFFERENT active
  mechanism and remains, including bounded GPU retries and tensor fallback.
- The full-spectrum correctness oracle now explicitly uses its existing eager
  coefficient path, removing an always-false fusion branch. Main GPU coefficients
  and backward are untouched.
- Removed the old never-dispatched tile grouping kernel/wrapper and its observer
  entry. Active sorted-key scan/pack scheduling and profiling remain.
- Removed unused custom-density exposure metadata, unused diagnostic acceptance
  helper/re-export and json import. Strict native/eager/GPU gates are unchanged.
- Retired custom policy/cache choices no longer appear as supported modes. Inert
  compatibility flags are hidden from help but their safe defaults remain accepted
  and recorded, keeping old valid command previews and result schemas compatible.

### Verification actually performed

| Check | Result |
|---|---|
| CPU Torch/interpreter geometry and support probes | **17 arrays bit exact** before/after: triangle bounds, candidate rows, graph/spectral CSR and spectral candidates |
| Trainer callback control flow | **132 scenarios** preserve output arrays/shapes, trace requests and controller-call order (lightweight tensor-shaped stubs, not CUDA training) |
| Trainer attachment |4 supported original-policy cases match; custom density/optimizer requests reject |
| Command/workflow previews |6 before/after JSON previews identical: graph/spectral adaptive training, native/HKS rendering, profiling and paired workflow; no Torch/Triton import |
| Retired interfaces |6 rejected before GPU imports; obsolete compatibility flags hidden from help |
| GPU code preservation |All **47 retained JIT definitions** unchanged; only the unused grouping kernel retired |
| Numerical/source/backward preservation |7 source/coefficient/backward files byte identical; main graph/spectral support class identical; retained selection and eager-oracle paths formally match after folding removed dead branches |
| Validation preservation |Strict RGB/VJP/native replay/alignment gate function bodies identical |
| Python syntax/local imports |All69 remaining HKS/script modules parse; no broken local module imports |
| Native/source safety |89 native /154 protected non-HKS files unchanged; Git index unchanged |
| CUDA/GPU training |**Not run for this cleanup**; no GPU job/allocation requested |

The CPU probes cover selection/geometry and use the Triton interpreter for its
reference selector; they do not establish fresh CUDA rendering/gradient parity.
The earlier5,000-update performance study remains evidence for its recorded source
version, not a newly executed cleanup benchmark. No performance gain is claimed.
A failed verification harness initially omitted a tensor's detach interface; it
was corrected and all132 scenarios then passed. A preview harness initially gave
the legacy renderer an unsupported output flag; corrected before/after previews
passed. An over-broad intermediate cache-hook edit was immediately restored from
the preimage and replaced with exact AST-bounded edits before final verification.

### Compatibility and recovery

Supported original-policy state-dictionary checkpoints, graph/spectral domains,
CLI names, safety checks, benchmark/profile readers and recovery tools remain.
Direct imports of retired custom-policy/occupancy/cache modules, FixedTriangleBounds,
its fixed_bounds injection, the private fresh-selector wrapper and the unused old
grouping helper are intentionally retired. Historical whole-object pickles naming
those symbols need the archived code. Do not reuse an old initial-state manifest
across changed source hashes; use a fresh workflow/output directory as usual.

The archive contains exact current-working-tree preimages, including uncommitted
work. Inspect an individual file without overwriting the repository:

```bash
tar -xOf archive/hks_results_cleanup_20261123/preimages.tar.gz hktex_splatting/training/custom_training.py
```

Recovery checksum and individual source hashes are in plan.json/before_manifest.json
next to the archive. All previous recovery records, meshes, eigensystems, cameras,
checkpoints and experimental results were retained. No commit, push, reset, stash,
staging or Git-history modification occurred. Several HKS sources were already
untracked, so ordinary Git diff omits some removals; the task-scoped `task.diff`,
`changes.json` and verified archive provide the precise before/after record.

### Every changed pre-existing file

| File | Action | Added / removed lines |
|---|---|---:|
| `docs/CUSTOM_HKS_TRAINING.md` | modified | +7 / -0 |
| `docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md` | modified | +1 / -1 |
| `docs/HKS_WORK_LOG.md` | modified | +5 / -0 |
| `docs/SPARSE_ASSOCIATION.md` | modified | +5 / -0 |
| `hktex_splatting/README.md` | modified | +3 / -3 |
| `hktex_splatting/knn_heat/packed_support.py` | modified | +6 / -46 |
| `hktex_splatting/knn_heat/persistent_association.py` | deleted | +0 / -118 |
| `hktex_splatting/knn_heat/triangle_kernel_support.py` | modified | +2 / -30 |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | modified | +4 / -11 |
| `hktex_splatting/training/adaptive_density.py` | deleted | +0 / -7 |
| `hktex_splatting/training/custom_training.py` | deleted | +0 / -30 |
| `hktex_splatting/training/density_controllers/__init__.py` | deleted | +0 / -4 |
| `hktex_splatting/training/density_controllers/residual_cost.py` | deleted | +0 / -99 |
| `hktex_splatting/training/density_controllers/transactions.py` | deleted | +0 / -60 |
| `hktex_splatting/training/kernel_optimizer.py` | deleted | +0 / -55 |
| `hktex_splatting/training/occupancy_evaluation.py` | deleted | +0 / -76 |
| `hktex_splatting/training/training_checks.py` | modified | +0 / -10 |
| `hktex_splatting/training/uv_training.py` | modified | +7 / -13 |
| `hktex_splatting/triton/heat_kernel_splatting.py` | modified | +0 / -22 |
| `splatting_script/benchmarks/pipeline_profile.py` | modified | +1 / -1 |
| `splatting_script/experiments/train_surface.py` | modified | +11 / -11 |

`docs/HKS_RESULTS_CLEANUP.md` is the new plan/report. HKS README/current architecture
and the historical custom-policy/sparse-association guides were updated; historical
measurements were retained with explicit retirement notices. Full definition,
import and responsibility inventory: `archive/hks_results_cleanup_20261123/file_audit.json`.

### File-by-file HKS review

All77 initial modules are listed below. Keep means an active implementation,
independent reference/gate, public API/CLI, preprocessing, benchmark/profile/quality
or recovery role was verified. Large working numerical/benchmark modules were not
rewritten solely because of size. Their shared dependencies must not be discarded.

| Initial file | Decision | Definitions reviewed | Responsibility / action |
|---|---|---:|---|
| `hktex_splatting/prepare_assets.py` | keep | 6 | Prepare recorded mesh inputs and explicitly generate the original eigensystem. |
| `hktex_splatting/__init__.py` | keep | 1 | Expose the public HKS renderer and field types through lazy imports. |
| `hktex_splatting/__main__.py` | keep | 0 | Run the Heat-Kernel Splatting package command through cli.main. |
| `hktex_splatting/cli.py` | keep | 1 | Route Heat-Kernel Splatting and original HKTex rendering and training commands. |
| `hktex_splatting/modules/__init__.py` | keep | 0 | Group learned source state and differentiable corner spectral coefficients. |
| `hktex_splatting/modules/spectral_heat_coefficients.py` | keep | 12 | Autograd wrappers for live spectral sources and three-corner heat/Gram coefficients. |
| `hktex_splatting/modules/spectral_heat_field.py` | keep | 8 | Learned source state and differentiable three-corner spectral coefficients. |
| `hktex_splatting/utils/support_debug.py` | keep | 4 | Check live support rows/CSR and save failure evidence without repairing it. |
| `hktex_splatting/utils/__init__.py` | keep | 0 | Group repository paths, assets, diagnostics and experiment utilities. |
| `hktex_splatting/utils/metrics.py` | keep | 7 | Evaluate foreground RGB fidelity with missing pixels counted as black. |
| `hktex_splatting/utils/training_profile.py` | keep | 3 | Summarize correlated CUDA trace activity for retained benchmark diagnostics. |
| `hktex_splatting/utils/reference_assets.py` | keep | 18 | Load fixed mesh/camera inputs and save aligned reference-render artifacts. |
| `hktex_splatting/utils/experiment_io.py` | keep | 6 | Manage portable experiment inputs, preservation checks and unique outputs. |
| `hktex_splatting/utils/compiler_runtime.py` | keep | 1 | Compare compiler policies recorded in historical benchmark manifests. |
| `hktex_splatting/knn_heat/persistent_association.py` | remove | 6 | Unsupported experiment; archived before removal. |
| `hktex_splatting/knn_heat/__init__.py` | keep | 0 | Group spectral triangle candidate and support selection modules. |
| `hktex_splatting/knn_heat/triangle_kernel_support.py` | simplify | 29 | Removed unused persistent-association hooks and wrapper; all current graph/spectral discovery and legacy numeric selection retained. |
| `hktex_splatting/knn_heat/packed_support.py` | simplify | 8 | Removed unused FixedTriangleBounds/cache injection; same direct triangle-bound arithmetic. |
| `hktex_splatting/training/uv_training.py` | simplify | 19 | Require original policies, remove no-op installer and dead custom-density branches; preserve original RecordedUvTrainer and density bridges. |
| `hktex_splatting/training/training_checks.py` | simplify | 16 | Remove unused diagnostic acceptance helper/import; all strict gate bodies unchanged. |
| `hktex_splatting/training/__init__.py` | keep | 0 | Adapters for the original HKTex surface-sample training lifecycle. |
| `hktex_splatting/training/surface_profile.py` | keep | 9 | Bounded, synchronized stage timing for original/HKS surface training. |
| `hktex_splatting/training/kernel_optimizer.py` | remove | 3 | Unsupported experiment; archived before removal. |
| `hktex_splatting/training/target_texture.py` | keep | 7 | Fixed target-image buffers and exact nearest-texel camera supervision. |
| `hktex_splatting/training/adaptive_density.py` | remove | 0 | Unsupported experiment; archived before removal. |
| `hktex_splatting/training/custom_training.py` | remove | 1 | Unsupported experiment; archived before removal. |
| `hktex_splatting/training/sample_tile_geometry.py` | keep | 7 | Build flat GPU sample tiles for HKS training without camera-grid packing. |
| `hktex_splatting/training/camera_supervision.py` | keep | 7 | Rasterize fresh fixed-mesh camera frames for the adaptive HKTex/HKS loop. |
| `hktex_splatting/training/surface_adapter.py` | keep | 17 | Pack original mesh samples for the existing HKS renderer and density callbacks. |
| `hktex_splatting/training/occupancy_evaluation.py` | remove | 4 | Unsupported experiment; archived before removal. |
| `hktex_splatting/training/density_controllers/__init__.py` | remove | 0 | Unsupported experiment; archived before removal. |
| `hktex_splatting/training/density_controllers/transactions.py` | remove | 2 | Unsupported experiment; archived before removal. |
| `hktex_splatting/training/density_controllers/residual_cost.py` | remove | 7 | Unsupported experiment; archived before removal. |
| `hktex_splatting/rendering/splat_autograd.py` | keep | 11 | Shared splat autograd contract and backward used by Heat-Kernel Splatting. |
| `hktex_splatting/rendering/gpu_dispatch.py` | keep | 11 | Own fixed device triangle/tile geometry and benchmark control task records. |
| `hktex_splatting/rendering/cameras.py` | keep | 2 | Provide the recorded fixed training and held-out camera definitions. |
| `hktex_splatting/rendering/prepared_geometry.py` | keep | 7 | Prepare immutable camera/pixel geometry while refreshing learned support. |
| `hktex_splatting/rendering/triangle_tile_layout.py` | keep | 6 | Build fixed triangle/tile layouts and ordered kernel/triangle incidences. |
| `hktex_splatting/rendering/__init__.py` | keep | 0 | Group HKS rendering, fixed visibility and retained reference controls. |
| `hktex_splatting/rendering/fast_raster.py` | keep | 5 | Rasterize fixed triangles with the preserved float64 clipping and depth rules. |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | simplify | 27 | Simplify always-eager full-spectrum oracle and remove unused custom candidate-cost metadata; main numerical forward/backward unchanged. |
| `hktex_splatting/rendering/native_reference.py` | keep | 4 | Evaluate the original trained HKTex field at visible pixel centers. |
| `hktex_splatting/rendering/mesh_visibility_raster.py` | keep | 4 | Fixed-mesh BVH visibility and camera-frustum traversal. |
| `hktex_splatting/rendering/camera_raster.py` | keep | 6 | Shared float64 camera, clipping and reference mesh visibility calculations. |
| `hktex_splatting/rendering/resident.py` | keep | 9 | Share fixed dispatch geometry and live triangle coefficient topology with Heat-Kernel Splatting. |
| `hktex_splatting/rendering/reference_mesh.py` | keep | 7 | Sample the original mesh UV texture at triangle-renderer pixel centers. |
| `hktex_splatting/rendering/device_camera.py` | keep | 10 | Fresh GPU visibility snapshots and triangle/tile dispatch geometry for HKS. |
| `hktex_splatting/rendering/cuda_mesh_raster.py` | keep | 5 | Fresh CUDA triangle rasterization through optional nvdiffrast. |
| `hktex_splatting/triton/resident_gpu.py` | keep | 17 | Select, scan and pack fresh ordered triangle/kernel support on the GPU. |
| `hktex_splatting/triton/heat_kernel_splatting.py` | simplify | 11 | Remove never-dispatched old grouping wrapper/kernel; keep all active GPU kernels identical. |
| `hktex_splatting/triton/__init__.py` | keep | 0 | Group GPU spectral, support, splatting and adjoint kernels. |
| `hktex_splatting/triton/spectral_support_distances.py` | keep | 7 | Exact spectral support distances, conservative pruning and reduction oracles. |
| `hktex_splatting/triton/splat_contributions.py` | keep | 3 | Shared contribution emission, strongest-weight selection and reference RGB. |
| `hktex_splatting/triton/splat_reconstruction.py` | keep | 5 | Reconstruct HKS RGB and differentiate selected contributions. |
| `hktex_splatting/triton/spectral_heat_coefficients.py` | keep | 8 | Triton live-source preparation, corner heat/Gram contractions and their adjoints. |
| `hktex_splatting/triton/heat_kernel_backward.py` | keep | 2 | Selected contribution and RGB adjoints with compile-time pair counts. |
| `splatting_script/__init__.py` | keep | 0 | Group reproducible HKS experiment, benchmark and recovery commands. |
| `splatting_script/experiments/__init__.py` | keep | 0 | Expose the original HKTex / Heat-Kernel Splatting training and benchmark command. |
| `splatting_script/experiments/compare_backends.py` | keep | 1 | Compare HKTex (Original, native) and HKTex Splat (Ours, HKS). |
| `splatting_script/experiments/train_surface.py` | simplify | 12 | Retire unsupported policy choices, hide inert compatibility flags, preserve original-policy report schema and supported commands. |
| `splatting_script/configs/__init__.py` | keep | 0 | Package the recorded training recipe and original source declarations. |
| `splatting_script/maintenance/__init__.py` | keep | 0 | Group recovery tools for preserved research source archives. |
| `splatting_script/maintenance/recovery.py` | keep | 5 | Verify, list and selectively restore sealed research-source archives. |
| `splatting_script/benchmarks/pipeline_workflow.py` | keep | 9 | Benchmark HKTex (Original) versus HKTex Splat (Ours). |
| `splatting_script/benchmarks/spectral_distances.py` | keep | 3 | Compare exact HKS distance layouts without launching training. |
| `splatting_script/benchmarks/__init__.py` | keep | 0 | Group matched renderer comparisons, profiling and report readers. |
| `splatting_script/benchmarks/compare_rendering.py` | keep | 8 | Compare original Mitsuba rendering and fresh HKS rasterization as different workloads. |
| `splatting_script/benchmarks/profile_training_report.py` | keep | 2 | Read synchronized training profiles without importing Torch or probing GPUs. |
| `splatting_script/benchmarks/training_research.py` | keep | 27 | Record matched training updates, quality, snapshots and source provenance. |
| `splatting_script/benchmarks/camera_scaling.py` | keep | 2 | Run paired adaptive camera training at matched resolutions and report scaling. |
| `splatting_script/benchmarks/renderer_comparison.py` | keep | 40 | Compare HKTex (Original) and HKTex Splat (Ours), with correctness gates. |
| `splatting_script/benchmarks/compare_training.py` | keep | 3 | Compare HKTex (Original) and HKTex Splat (Ours) at measured image quality. |
| `splatting_script/benchmarks/pipeline_trace.py` | keep | 7 | Analyze CPU/CUPTI traces using correlated intervals and explicit overlap rules. |
| `splatting_script/benchmarks/pipeline_report.py` | keep | 16 | Build failure-aware timing, memory, quality and scaling reports from saved runs. |
| `splatting_script/benchmarks/pipeline_profile.py` | simplify | 33 | Remove observer mapping for deleted inactive kernel; keep active instrumentation. |
| `splatting_script/benchmarks/legacy/__init__.py` | keep | 0 | Optional historical diagnostics, excluded from normal rendering and training. |
| `splatting_script/benchmarks/legacy/geodesic_capture.py` | keep | 9 | Capture failed geodesic inputs without changing the original tracer's results. |

### Remaining review recommendations

Keep future structural refactors separate from numerical changes. The large
`triangle_kernel_support.py` still includes actual benchmark reference/fused/tiled
selection dependencies; `renderer_comparison.py` contains active correctness and
profiling workflows. Their size alone does not make them obsolete. Native-reference,
regional-quality and recovery CLIs are legitimate standalone entry points even
when the latest training pair does not import them. Graph and spectral support
remain distinct supported methods; default rendering and training behavior did
not change. No remaining file was deleted on an uncertain-use assumption.
