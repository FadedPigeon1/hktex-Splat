# Dependency-backed pipeline-only deletion plan

Date: November 16, 2026. Recorded before deleting files or changing runtime code.

The preserved public workflows are original/native HKTex and hybrid/2dgs HKS. Internal resident and full-spectrum eager controls remain solely because the existing correctness gates require them. Shared support/coefficients/scheduling/backward are not obsolete.

## Proposed file removals

| File | Reason / dependency disposition |
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

## Intentional optional-feature removals

- Positive footprint_cutoff/projected-conic pruning and bounds/nearest-selection options.
- Standalone resident/splat backends and alternative direct selection modes; native/original and hybrid/2dgs remain.
- Historical forward trainer and its commands.
- Tile-KNN/direct-native duplicate path and full-spectral Triton emitter; full-spectrum eager oracle retained only for benchmark correctness.
- Packed/eager legacy support builders, unused coefficient topology and duplicate CPU shading oracles.
- Cooperative resident rendering prototype; baseline resident control remains internal to correctness diagnostics.
- Renamed-module aliases; intentional old Python import/whole-object-pickle compatibility break, native model/optimizer state dictionaries preserved.

## Shared code preserved

- `hktex/`: Preserve every original module, registered implementation, optimizer and original author command dependency.
- `hktex_splatting/knn_heat/triangle_kernel_support.py`: Shared spectral support, fused selector and exact >4096 fallback; preserve implementation bytes.
- `hktex_splatting/knn_heat/packed_support.py`: Keep immutable embedding bounds, matches and candidate_rows. Remove only its unused packed CPU select adapter.
- `hktex_splatting/rendering/resident.py`: Hybrid uses ResidentDispatchGeometry/ResidentTopology; build_scene baseline control remains required by benchmark RGB/VJP/optimizer gates.
- `hktex_splatting/rendering/gpu_dispatch.py`: Keep DeviceDispatchGeometry/DeviceDispatch required by hybrid and internal control; retire exclusive legacy builder/topology.
- `hktex_splatting/rendering/triangle_splat_reference.py`: Keep Dispatch/order/visible face-tile planner required by PreparedGeometry; retire unused CPU shade/reconstruct functions.
- `hktex_splatting/rendering/triangle_heat_map.py`: Keep FaceMap required by fixed geometry; retire duplicate unused candidate/shade routines.
- `hktex_splatting/rendering/splat_autograd.py`: Keep shared backward, baseline reference forward and safety guards; retire unused construction/oracle/cooperative APIs.
- `hktex_splatting/rendering/fast_raster.py`: Required pixel/depth raster implementation, imported by active BVH visibility.
- `scripts/visualize_kernel_faces.py`: Production camera basis/geometry helper despite historical name; preserve.
- `scripts/test_numba_raster.py`: Production native reference rasterizer; preserve.
- `scripts/`: Original HKTex author preprocessing/training/benchmark commands remain distinct native workflows. None imports optional HKS footprint code; preserve original baseline behavior.
- `splatting_script/benchmarks/`: Preserve native/hybrid timing, memory, PSNR, profiling, report readers and existing embedded correctness tolerances.
- `Splatting_script.py`: Tiny established namespace command compatibility; contains no legacy renderer.
- `docs/*cleanup* and archive/`: Historical recovery/evidence protected; do not rewrite old reports or scientific results.

Unique legacy research implementations are identified here, not silently discarded. Their exact sources and edited-file preimages will be sealed in a recovery archive before removal. No models, meshes, eigensystems, results or previous cleanup/recovery records will be deleted. Existing staged/unstaged work remains the baseline; no reset, stash, commit or push.

All import edges, definitions and call names for 115 live HKS/script Python files are in [plan.json](plan.json). Non-imported native author scripts were treated as command entry points, not presumed unused.

Additional dependency-confirmed retirement before deletion: `hktex_splatting/rendering/dynamic_dispatch.py`; its exclusive legacy callers are removed above. Active GPU key sorting and selector fallback are preserved. Total: 26 files.

Exclusive compiler-cache setup/runtime helpers are also retired from `utils/compiler_runtime.py`: only `matched_policy` is called by retained benchmark readers; its implementation is preserved. Preimages were already sealed.
