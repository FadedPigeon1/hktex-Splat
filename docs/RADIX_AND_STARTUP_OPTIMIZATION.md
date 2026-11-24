# Research-guided exact selection and startup cleanup

The target is faster complete HKS surface training, not an isolated kernel claim.
The last unprofiled HKS worker took 358.50 seconds, with a 217.74-second training
interval. Roughly 140.76 seconds were outside that interval. A fivefold reduction
of that worker would require about 71.70 seconds total: optimizing selection alone
cannot achieve it while leaving the measured setup/evaluation costs unchanged.
No fivefold improvement, or any new speed improvement, is claimed by this pass.

## Literature reviewed

| Primary source | Relevant technique and decision |
|---|---|
| [RadiK, ICS 2024](https://arxiv.org/html/2501.14336v1) | Radix selection avoids sorting all candidates. Implemented a small exact byte-radix selector for each existing source tile, not a reproduction of the authors' full framework. |
| [AIR top-K / GridSelect, SC 2023](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap294.html) | Keep selection iterations on the GPU, avoiding host-controlled refinement/readbacks. The new radix loop stays inside one Triton program. |
| [RTop-K, ICLR 2025](https://arxiv.org/abs/2409.00822) | Pivot selection and early stopping were assessed. Approximate termination is excluded to preserve exact candidate membership. |
| [Qrita, 2026](https://arxiv.org/html/2602.01518v1) | Deterministic tie handling and reduced sorting are relevant. Statistical truncation specialized to logits is not applied to arbitrary spectral-distance distributions. |

Published speedups describe the papers' workloads and implementations. They are
not predictions for this repository. No paper source code was copied or new
runtime library installed. This implementation combines general exact radix
selection with the existing HKS merge contract; it is not labelled a full RadiK,
AIR, RTop-K or Qrita implementation.

## Implemented

- `hktex_splatting/triton/resident_gpu.py`: opt-in `merge_support_tile_radix`.
  It finds the exact KEEP-th float64 bit key using most-significant-byte
  histograms, compacts entries with stable source-ID ties, sorts only KEEP local
  entries and merges with the retained list. Early exit is exact only when all
  surviving keys are identical. Private scratch has CTA synchronization before
  readback. All sources, cap rules, guard arithmetic and padding remain intact.
- `hktex_splatting/knn_heat/triangle_kernel_support.py`: exposes `radix` alongside
  the existing `reference` and `compact` implementations. Private scratch is
  reused only within one selection call; original fallbacks remain intact.
- `hktex_splatting/training/cached_sampling.py`: optional fixed CPU face-area
  probability cache. Original uniform sampling recomputed areas for every batch
  while constructing the million-point importance pool. This cache preserves
  multinomial/rand calls, texture interpolation and RNG progression. The original
  native sampler remains unchanged. Actual model/RNG/pool/evaluation receipts
  must still match exactly before training; mismatches are never silently reset.
- `hktex_splatting/training/uv_training.py`: optional suppression of original
  diagnostic histogram/loss plots for BOTH recording adapters. Defaults preserve
  plots; original files under `hktex/` are not modified.
- Renderer, adapter and training CLI forward and record the new policies. These
  remain explicit options, not a silent benchmark/default change.

## Cleanup list and dependency evidence

Before moving code, a repository search found exactly one runtime import of
`hktex_splatting/training/trace_diagnostics.py`, in the explicit diagnostic branch
of `train_surface.py`. No renderer, optimizer, preprocessing module or benchmark
imported it otherwise. It contains unique failure-investigation work, so it was
preserved byte-for-byte at
`splatting_script/benchmarks/legacy/geodesic_capture.py`. The old training-package
file was removed and the sole import updated. A package initializer ensures the
archive is included by the existing non-namespace package discovery.

The historical `--trace-diagnostics` flag remains explicit compatibility access
to that archive. Normal training does not import it. This move itself cannot
speed up a run where capture was already disabled. Runtime assertions, baseline
and HKS correctness gates, stage timing, quality evaluation, checkpoints, past
cleanup recovery records and every research result/asset were preserved.

## Validation and pending work

CPU interpreter checks compare every retained key bit and source ID after each
tile against the original full sort and an independent NumPy lexicographic
oracle. Cases cover ties, NaNs, infinities, partial tiles, KEEP 1 through 512 and
source counts below/above 4096. Offline SM80 compilation checks the new kernel.
Real CUDA execution and a performance win remain pending.

CPU checks execute the original sampler/controller source bodies while isolating
GPU package initialization: original/cached batches and importance pools match
exactly in coordinates, colors, face IDs, barycentrics, errors, indices and RNG,
including separate UV topology. Cache mutation is rejected. Plot delegation,
CLI forwarding, archived imports/package discovery and protected source/index
checks are covered. Temporary test fixtures remain outside the repository.

On an allocated GPU, first verify the complete changed path without optimizer
updates, using a fresh directory:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 5000 --iterations 1001 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter aggressive --selector-merge radix \
  --no-training-plots --verify-samples 1024 \
  --verify-only --execute \
  --output "experiments/radix_startup_verify_$(date +%Y%m%d_%H%M%S)"
```

Require actual selector `tiled_radix_merge`, no fallback, exact candidate rows and
all existing RGB/VJP/initial-state checks to pass. Then run a fresh full 5000-update
comparison starting from 1000 kernels, with the same options and no profiling or
trace capture. Keep plotting policy matched between native and HKS. Report cached
sampler startup savings separately from render/update savings. Repeated runs and
quality checks are necessary; this experiment is still surface-sample training.

Mixed precision, reduced spectral dimensions/support caps, approximate stopping,
changed sampling distributions and density schedules were deliberately excluded.
CUDA graphs require a separate redesign of host-dependent dynamic allocation and
kernel-count transitions; they were not blindly enabled. Broader shader/source
fusion is deferred until measured selector and sampler results identify the next
cost. None of these exclusions can honestly be turned into a promised 5x gain.

Historical note: the optional sampler cache described above was subsequently removed. Current commands use the original sampler. Reproducing old cached-sampler results requires their recorded source version; results were retained.
