# Original Mitsuba rendering versus triangle-rasterized HKS

This is a rendering-only experiment, separate from surface/camera training.
`splatting_script.benchmarks.compare_rendering` loads **one shared final.pt** in
two independent workers. It never trains or changes the checkpoint. Original
`hktex/` files are unchanged.

The native worker calls the original `HeatKernelsRendererKNN`, including its
principled heat-kernel texture, scene construction, configured integrator,
ground plane, sampling, denoising, KNN preparation/reset and cache flush. The
original renderer's camera dictionary is overridden to use the recorded shared
views and an explicit vertical field of view. Other saved scene settings remain
active; this is the original renderer with controlled cameras, not a replay of
the original author's entire experiment or training recipe.

The HKS worker freshly rasterizes the object each frame with nvdiffrast and
calls the existing triangle support/corner evaluation/tile splatting pipeline.
It produces unlit RGB on black, with one center sample per pixel, no ground
plane lighting/shadows, no path bounces, and no denoising.

**These are different rendering workloads.** The report shows raw time and
images, not a validated equivalent-rendering speedup. No cross-method PSNR is
reported because lighting and sampling differ. The unresolved HKS/eager RGB
discrepancy at resolution 128 is not waived or fixed by this separate experiment;
this command checks finite images and model/asset identity, not that strict gate.

## Run

Use your existing GPU allocation. The same GCC/CUDA modules used to install
nvdiffrast must be available. The checkpoint must remain with its original
`manifest.json` and `resolved_config.json`. Both methods use this same checkpoint,
even if it was trained by only one method; this isolates renderer input state.

```bash
cd /path/to/hktex
HK_PY=/path/to/your/hktex/environment/bin/python
CHECKPOINT=experiments/optimized_camera_64_20261121_104112/native_train/final.pt

# Preview only: no CUDA imports, rendering or allocation.
"$HK_PY" -B -m splatting_script.benchmarks.compare_rendering \
  --checkpoint "$CHECKPOINT" --resolution 64 \
  --views side --frames 1 --warmup 0

# Small runtime smoke check: one first frame and one measured frame per method.
"$HK_PY" -B -m splatting_script.benchmarks.compare_rendering \
  --checkpoint "$CHECKPOINT" --resolution 64 \
  --views side --frames 1 --warmup 0 --execute \
  --output "experiments/original_render_smoke_$(date +%Y%m%d_%H%M%S)"
```

On Grace, the existing environment is typically
`/scratch/user/$USER/.conda/envs/hktex/bin/python`; load
`GCC/12.3.0 CUDA/12.9.0` before executing. Other systems should use their own
CUDA/PyTorch environment and installation paths.

If loading GCC makes FAISS/cuVS fail with `CXXABI_1.3.15 not found`, put the
environment's newer C++ runtime ahead of the compiler module's libraries:

```bash
HK_ENV=$("$HK_PY" -c 'import sys; print(sys.prefix)')
export LD_LIBRARY_PATH="$HK_ENV/lib:${LD_LIBRARY_PATH:-}"
"$HK_PY" -c 'import faiss; import nvdiffrast.torch; print("Imports OK")'
```

Apply this after loading modules and before launching workers. FAISS need not
be reinstalled for this library-order error. The native smoke attempt
`original_render_smoke_20261121_110757` stopped at this import before rendering.

After the smoke check succeeds, render the recorded view sequence:

```bash
"$HK_PY" -B -m splatting_script.benchmarks.compare_rendering \
  --checkpoint "$CHECKPOINT" --resolution 256 \
  --frames 12 --warmup 4 --execute \
  --output "experiments/original_render_full_$(date +%Y%m%d_%H%M%S)"
```

This default preserves native sample count from the saved config and enables
OptiX denoising. `--samples-per-pixel N` changes only native sampling;
`--no-denoise` explicitly disables native denoising. Both are recorded. These
options never make HKS equivalent to the native lit renderer.

Each native full frame rebuilds the original scene and acceleration structures
as part of the original rendering lifecycle; HKS rebuilds visibility and dynamic
support. Samples include synchronization of both CUDA/Dr.Jit and Torch. Image
export and finite checks are outside frame timing. The first frame is recorded
separately; warmup frames are excluded from medians. Medians aggregate a cycling
camera sequence, not individual per-view distributions. Cold process/model setup
and complete worker times remain in receipts. Peak Torch memory is **not total
VRAM**: Dr.Jit, Mitsuba and OptiX allocations are outside that allocator.

Outputs: `comparison.md`, workflow/worker JSON receipts, native/HKS logs, raw
floating-point images and clamped display PNGs. PNGs have no added tone mapping
or sRGB conversion, so use the raw images for subsequent color analysis. Files
are created in a fresh directory and existing research results are preserved.

## Validation status

Python syntax, CPU-only command previews, protocol/report checks and a synthetic
camera-ray helper check are available without requesting a GPU. Actual original
Mitsuba/nvdiffrast execution, OptiX compatibility, full-frame memory feasibility
and real camera alignment remain pending GPU smoke validation. No GPU allocation
or benchmark was launched while implementing this command.
