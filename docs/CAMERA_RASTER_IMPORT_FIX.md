# Missing camera/raster helper dependency

Both preflights in `hktex_vs_hks_5000_smoke_retry` failed during import with
`ModuleNotFoundError: No module named 'scripts.visualize_kernel_faces'`.
The native and hybrid logs contain this exception before GPU/model loading
completed and before either replay check ran. The user's A100 allocation
subsequently expired; cancellation was a separate event.

Two deleted script filenames still supplied production geometry functions:

- `scripts/visualize_kernel_faces.py`: camera basis, polygon clipping and NumPy
  reference visibility rasterization.
- `scripts/test_numba_raster.py`: the Numba reference raster kernel, argument
  preparation and raster wrapper used by `reference_assets.raster_surface`.

The six required functions were moved unchanged from the retained pre-deletion
source snapshot into `hktex_splatting/rendering/camera_raster.py`. Imports in
`fast_raster.py`, `mesh_visibility_raster.py` and `utils/reference_assets.py`
now use this production module. Both reference raster backends remain available;
the active optimized raster kernel is unchanged. The deleted script files,
visualization CLI, standalone tests and test runner were not restored.

The README's obsolete script-based check is replaced by a CPU visibility import
check. No spectral, splatting, backward, optimizer, selector or tolerance code
is changed. Original HKTex and existing benchmark artifacts are untouched.

Verification compares the six function ASTs with their original snapshots,
executes old/new NumPy and Numba raster functions on identical float32/float64
inputs, exercises both reference-asset backends and imports/runs static/BVH
visibility. Exact results are in
[camera_import_verification.json](hktex_vs_hks_benchmark/camera_import_verification.json).
Temporary verification helpers stay outside the repository; no GPU job or
allocation is launched by the agent.

This fixes the missing-import failure. The earlier 5,000-source native
optimizer/VJP replay mismatch remains unresolved and still requires its
diagnostic run. After obtaining a new A100 allocation, rerun the short smoke
workflow using an existing initialization and a new output directory. A
completed import is not evidence that replay or GPU correctness has passed.
