# HKS distance optimization and cleanup — 2026-11-21

## Training-profile follow-up

`experiments/training_bottleneck_20261121_112201` measured inclusive HKS support
selection at 4.11 ms/update (1.91 ms distance, 1.72 ms cutoff selection), corner
coefficients at 2.41 ms, and splatting at 0.42 ms. These are synchronized profile
scopes, not additive uninstrumented GPU kernel timings.

Two opt-in implementations now target those costs:

- `--selector-merge row-topk`: on the existing bounded 4097–8192-source row path,
  one native Torch top-k finds the exact Kth/cap **values**. A Triton pass applies
  the unchanged radius/guard bound and repairs ties by ascending source ID.
  Native top-k IDs are ignored, so their boundary-tie policy cannot change CSR
  ordering. NaNs retain the existing non-admitted behavior. Existing small-row,
  large-count and safety fallbacks remain. Explicit value/index scratch is
  budgeted; Torch's internal selection workspace is additional.
- `--corner-evaluation parallel`: evaluates the three independent corner
  contractions and six Gram entries as parallel tensor rows, instead of a
  sequential corner loop and separate scalar reductions. It still evaluates
  all three corners, uses the same heat/metric formulas, and calls the existing
  coefficient backward kernel. It does not deduplicate vertex/kernel pairs or
  change the renderer to pointwise HKTex.

Defaults remain the previous paths. Both options are recorded in workflow,
control and result receipts; fixed-view renderer reuse keys also include the
corner layout. Original HKTex is unaffected. The autograd wrapper accepts the
layout internally and returns no gradient for that non-tensor choice.

CPU interpreter checks passed exact cutoff packing versus row radix for 51,
257, 5000 and 8192 sources with ties and nonfinite entries. Serial/parallel
corner heat, Gram values and five source-state VJPs matched exactly for
mode/dimension pairs (3,2), (32,64), and (65,33). This does not prove matching
CUDA rounding, CUDA backward behavior or faster training; strict existing GPU
gates remain required. No GPU request/job was made for this follow-up.

The real CPU candidate selector also matched at 5000 sources and retained the
tiled fallback at 8193 sources. All-infinity/all-NaN cutoff cases matched the
radix reference. Offline SM80 compilation passed for serial/parallel corner
forward and top-k cutoff packing. Parallel corner shared storage was 128 bytes
versus 512 bytes for serial in this compilation; this is not a timing result.

Modified files: `triton/resident_gpu.py`, `knn_heat/triangle_kernel_support.py`,
`rendering/device_camera.py`, `triton/spectral_heat_coefficients.py`,
`modules/spectral_heat_coefficients.py`, `modules/spectral_heat_field.py`,
`training/surface_adapter.py`, and `splatting_script/experiments/train_surface.py`.
No implementation, asset or result file was deleted in this follow-up.

For a controlled manual profile retry, use the previous 40-update command with
`--selector-merge row-topk --corner-evaluation parallel`, keeping resolution,
checkpoint initialization policy, pruning and all other settings the same.
Test either flag independently to attribute any improvement. Use fresh output
directories; do not reuse old initial snapshots across source changes. Confirm
any training benefit in a separate unprofiled full-budget run.

This change targets redundant work in spectral support distances and camera
geometry preparation. No GPU allocations, training or timing jobs were run after
the instruction to make no GPU requests. **Runtime improvement is not measured.**

## Evidence and scope

The completed 64-square camera experiment in
`experiments/gpu_full_20261121_022432/64/comparison.md` remains the baseline:
native training 79.37 seconds, HKS training 124.90 seconds for 5000 updates.
An earlier surface profile attributed 14.92 ms/update to spectral distances
inside 18.94 ms support selection. That is evidence for investigating distance
work, not proof of the current camera workload's stage ranking. The camera
benchmark did not record a stage profile.

The 128-square run's strict eager-reference RGB discrepancy is still unresolved.
No tolerance, top-30 rule or correctness gate was relaxed by this optimization.

## Implementation changes

| File | Change |
|---|---|
| `hktex_splatting/triton/spectral_support_distances.py` | New focused distance module, extracted from `resident_gpu.py`. Adds a compact float64 reduction with the same native warp addition order; retains the former gather reduction as a benchmark oracle. Contains conservative prefix pruning and its seed bound. |
| `hktex_splatting/triton/resident_gpu.py` | Keeps support ranking, scans and packing. Imports/re-exports distance kernels at their historical names. Reduced from 488 to 373 lines. |
| `hktex_splatting/rendering/device_camera.py` | Pixel vertex indices and spectral embeddings are now computed only when a diagnostic/reference path requests them. Normal corner-based HKS does not consume these arrays. Mutation guards cover lazily created arrays. |
| `hktex_splatting/rendering/heat_kernel_splatting.py` | Allocates the output image from the fixed embedding tensor, with the same dtype/device, avoiding a read that would force the unused pixel embedding gather. |
| `hktex_splatting/training/training_checks.py` | New focused module containing existing initialization receipts and RGB/VJP verification helpers. Moved bodies are unchanged. |
| `hktex_splatting/training/uv_training.py` | Retains trainer classes and re-exports historical helper imports. Reduced from 469 to 238 lines. Class/checkpoint module paths remain unchanged. |
| `hktex_splatting/rendering/prepared_geometry.py` | Replaces an empty heat-map object with the two empty CSR arrays actually consumed by layout construction. |
| `splatting_script/benchmarks/spectral_distances.py` | Compares old versus compact reductions at identical source/block layouts, checks exact distances/support, and supports CPU interpretation or offline SM80 compilation without GPU execution. |
| `hktex_splatting/README.md`, `docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md`, `docs/GPU_CAMERA_PIPELINE.md` | Updates current file responsibilities and links this report. Historical cleanup reports retain their recovery paths. |

No original `hktex/` source was changed. Kernel parameters, heat equations,
three-corner evaluation, interpolation, weighting, selected support ordering,
top-30 reconstruction, custom backward and optimizer implementations are retained.
Fresh camera rasterization remains fresh. Learned values/support are not cached.
The default pruning policy remains `none`; prefix pruning stays an explicit
experiment because its training speed benefit has not been established.

### Exact reduction

The previous distance kernel propagated five shuffle/gather additions to every
lane, although only lane zero's sum survived. It then padded warp totals to 32
lanes and repeated the same process. The new tree keeps only the dependencies
needed for that result: offsets 16, 8, 4, 2, 1 inside each warp, then the same
order over warp totals. Subtraction, float64 products, RN square root, and
disabled FP fusion remain unchanged. It uses no norm/dot-product identity,
approximate nearest neighbors, lower precision or modified support bound.

At 64 dimensions, each pair needs 63 additions in this explicit compact tree
instead of the old gather implementation's 480 lane additions at the Triton
source level. This operation count is not a GPU instruction or speed ratio.

Offline SM80 compilation produced these resource changes:

| Kernel | Old binary bytes | Compact binary bytes | Old shared bytes | Compact shared bytes |
|---|---:|---:|---:|---:|
| face4/source16, unpruned | 177888 | 30944 | 4096 | 2048 |
| face4/source16, prefix-pruned | 356192 | 38240 | 4096 | 2048 |
| source8 | 32608 | 39392 | 2048 | 2048 |

Binary size is not execution time; source8 actually grew. The old and new
implementations remain separately benchmarkable before adopting a performance
claim. Existing layouts and guards are preserved.

## Deletion review and evidence

The deletion was identified to the user before execution:

| Deleted file | Why it was unnecessary | Dependency evidence |
|---|---|---|
| `hktex_splatting/rendering/triangle_heat_map.py` | Only contained `FaceMap`, an old sparse heat-map payload now used solely as an empty layout placeholder. Its heat/metric/upper fields were unused by this caller. | Repository Python/config/package search found one importer: `PreparedGeometry`. `dispatch` consumes only `offsets` and `kernels`. Identical pre/post layout/device arrays were verified at three image shapes. Both saved native/HKS checkpoints from the baseline have no `FaceMap` or deleted-module references. |

No other implementation file was verified obsolete enough to delete safely.
Standalone benchmarks, full-spectrum numerical controls, native rendering,
preprocessing, profiling, selector fallbacks and safety assertions remain.
Historical metadata such as `footprint_cutoff=0` remains for old report schemas;
it does not activate projected-footprint rendering. No assets, checkpoints or
research results were deleted or edited.

## Verification

Passed without GPU execution:

- Bit-exact legacy/compact source and blocked distances: 39 strided/tail/magnitude
  cases across 13 dimensions from 1 to 256.
- CPU interpreter benchmark comparisons: 54 layout/dimension cases with selected
  support IDs and conservative pruning across six dimensions; plus dedicated pruning
  cases at 51, 257 and 5000 sources, large-radius rows and nonfinite seed guards.
- Real renderer CPU fixtures at 1×1, 9×17 and 16×16: RGB, selected IDs, weights
  and six VJPs agree with the existing camera geometry using synthetic spectral
  coefficients. Empty views, buffer mutation and subset placement checks pass.
- A real normal HKS fixture leaves the pixel embedding/vertex gather uncreated;
  full tracing still produces identical RGB and checks derived-array mutations.
- Support ordering and fallback comparisons at 12, 5000 and 8209 sources,
  including uncapped legacy and above-row-capacity tiled selection.
- Exact pre/post fixed camera layouts and device arrays at three image shapes.
- Identical AST bodies for every moved verification helper and trainer class.
- All HKS/script Python syntax, helper global dependencies, camera scaling,
  baseline benchmark and profiler command previews.
- Offline SM80 compilation of all distance layouts, both old references, and
  prefix pruning. This does not establish CUDA runtime correctness.
- Original HKTex source bytes and the pre-existing Git index are unchanged.

A direct original/HKS trainer import on this CPU session stops in the original
Mitsuba CUDA initialization because no CUDA device is available. With only those
original CUDA import boundaries isolated, real HKS module imports, historical
helper exports and checkpoint class paths pass. This is not a full native/HKS
CUDA execution check. No GPU test or full training speed comparison is claimed.

Temporary checks and raw CPU/offline reports were written under
`/tmp/hks-measured-optimization-4mp7i39l`. That directory became unavailable
during the final review; the completed results are recorded above. Existing
standalone testing infrastructure was not recreated in the repository.

## Git and recovery

The working tree already contained extensive staged, unstaged and untracked
cleanup work. No reset, stash, staging, commit, push or Git history rewrite was
performed. Task-start copies of Python sources, the package file, selected
documentation and the index listing were temporarily kept in
`/tmp/hks-measured-optimization-4mp7i39l`. Byte/AST comparisons completed before
that directory became unavailable. These temporary copies must not be treated
as durable recovery or as currently available artifacts.

The removed source was untracked at task start, so `git restore` alone cannot
recover it. Its complete source is preserved in the appendix below, outside
the importable implementation. Previous cleanup recovery records were
preserved. Whole-tree whitespace checks still report pre-existing
whitespace in original `hktex/modules/tracer.py`; that file remains untouched.

Remaining work requires separate GPU validation: exact old/new distance/support
checks on CUDA, full live spectral-data RGB/gradient checks, pruning on/off
timings, and paired unprofiled training at matched resolution and quality.
The large `renderer_comparison.py` benchmark was retained because its existing
validation/profiling paths are active; splitting it is a separate, higher-risk
refactor with no demonstrated update-speed benefit.

## Deleted-file recovery

Restore the following content to `hktex_splatting/rendering/triangle_heat_map.py`
if the retired payload is needed when investigating an older working tree:

```python
"""Store sparse triangle coefficient arrays for fixed-layout preparation.

FaceMap is shared geometry data, not a standalone kernel-centric renderer.
Main API: FaceMap."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FaceMap:
    """Store ordered per-triangle source IDs, three heat values and six distance coefficients."""
    offsets: np.ndarray
    kernels: np.ndarray
    heat: np.ndarray
    metric: np.ndarray
    upper: np.ndarray

    def validate(self, source_count):
        n = len(self.kernels)
        if (self.offsets.ndim != 1 or self.offsets.dtype.kind not in "iu" or self.offsets[0] != 0 or
            self.offsets[-1] != n or (np.diff(self.offsets) < 0).any() or
            self.heat.shape != (n,3) or self.metric.shape != (n,6) or self.upper.shape != (n,) or
            self.kernels.dtype.kind not in "iu" or (n and (self.kernels.min()<0 or self.kernels.max()>=source_count))):
            raise ValueError("Invalid sparse face map")
        if not all(np.isfinite(a).all() for a in (self.heat,self.metric,self.upper)) or (self.upper<0).any():
            raise ValueError("Nonfinite/negative primitive bounds")
        return self

    @property
    def bytes(self):
        return sum(a.nbytes for a in (self.offsets,self.kernels,self.heat,self.metric,self.upper))
```
