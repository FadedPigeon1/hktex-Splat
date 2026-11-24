# Device target-texture sampling for camera training

The camera smoke recorded 123.3 seconds in CPU target preparation over 601
frames, versus 38.8 seconds rasterizing triangles. `--target-sampling gpu`
addresses the repeated target-image conversion/readback path, not heat-kernel
mathematics. It applies equally to native and HKS workers. The CPU backend remains
the default and a comparison reference. It is camera-only; original surface
sampling, HKTex files and existing renderer/backward code are unchanged.

## Implementation

`training/target_texture.py` uploads fixed target RGB bytes and a 256-entry
normalization table once. UV interpolation keeps the original NumPy float64
`einsum` order. Device operations then use the original repeated UV wrapping,
float64 coordinate arithmetic, round-to-even nearest texel and encoded RGB/255
colors. The normalization table is generated with the same float64 division and
float32 storage as the CPU reference, avoiding a different device division rule.
The module supports material assignments, different texture dimensions, and
grayscale/palette/RGBA inputs via the original RGBA conversion at setup.

Training/held-out datasets share one device target sampler. There is no per-frame
RGBA conversion or full image upload. Visible UVs/indices are small per-frame
transfers; UV interpolation and triangle rasterization are still CPU operations.
Input texture buffers are fixed assets; kernel locations, anisotropy, heat,
contributions, learned colors, gradients and density updates remain live. Changing
target pixels intentionally requires recreating the sampler. This does not
reinstate the deleted surface-sampling cache.

The default upload limit is 512 MiB (`--max-target-texture-mib`). Setup fails
explicitly on excess allocation or missing/invalid material assignments. Fixed
buffer mutation/replacement and invalid face IDs are rejected. Target texture
bytes and startup lookup-validation results are recorded separately from learned
model state. GPU memory peaks now include the live target image allocation.

## Exactness gate and timing scope

Every sampler runs a strict setup comparison against Trimesh for 4096 deterministic
UV probes plus wrap endpoints and half-pixel rounding cases, for every material.
The local probe generator does not consume training RNG. Any mismatch stops
execution; `--allow-render-variation` does not bypass this target check. That flag
still labels the separate unresolved HKS renderer disagreement as diagnostic.

Camera target-preparation counters now measure CPU work and device enqueue, not
completed GPU sampling. Synchronized training/process wall times include actual
GPU completion. Comparing those counters alone would overstate any speed gain.
Fresh output and initialization are required because source/backend controls
remain strict. No end-to-end performance improvement is claimed before timing.

## GPU smoke

Use the previous camera command with `--target-sampling gpu`. Keep all other
settings matched, including density/optimizer policies and the explicit diagnostic
render allowance if continuing that investigation. For example:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --supervision camera --resolution 64 --target-sampling gpu \
  --backend both --kernels 1000 --iterations 601 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct --selector-merge row-radix \
  --density-policy original --optimizer-policy original --allow-render-variation \
  --no-training-plots --verify-samples 1024 --execute \
  --output "experiments/raster_gpu_targets_$(date +%Y%m%d_%H%M%S)"
```

The worker records `target_texture_validation`, device bytes, target backend and
timing scope. Check the exact target gate, full process/training time, quality,
fixed geometry and optimizer health. The render allowance means this is not a
validated numerical-equivalence/speedup claim. A CPU-target control is available
with `--target-sampling cpu`; it requires fresh output too.

Changed files: new `training/target_texture.py`, `training/camera_supervision.py`,
`splatting_script/experiments/train_surface.py`, this note, and the camera-training
guide. No original HKTex code, results or meshes were deleted. Recovery copies
and temporary CPU checks are in `/tmp/hks-gpu-target-bp9kty6i`.

## Completed checks

The device-operation implementation executed on CPU tensors matched the original
Trimesh/reference camera targets exactly for RGB, RGBA, grayscale and palette
images, including 1x1 textures and mixed materials of different dimensions.
Setup probes covered negative/out-of-range UVs, endpoints and rounding ties.
Full face/barycentric samples matched reference RGB exactly; allocation limits,
invalid IDs and buffer mutation rejection passed. Paired worker previews,
report labels, syntax and original HKTex/Git index preservation passed.

No GPU driver was accessible here, so no GPU lookup, training or performance
measurement was claimed. The setup device-vs-CPU check will execute on the
allocated GPU before the camera training loop begins. Original renderer/kernel
math and the unresolved diagnostic render gate were not changed. Whole-tree
diff checking still reports pre-existing trailing whitespace in original
`hktex/modules/tracer.py` at lines 285 and 290; that file was left untouched.
