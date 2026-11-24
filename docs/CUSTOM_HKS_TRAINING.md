> Retired HKS experiment (2026-11-23). Current comparisons use the original
> optimizers and density controllers. The custom policy modules below were
> removed from the active HKS package after dependency review; exact sources are
> in `archive/hks_results_cleanup_20261123/preimages.tar.gz`.
> This document preserves historical methods/results, not supported commands.
> See [the cleanup report](HKS_RESULTS_CLEANUP.md) and [current run guide](../hktex_splatting/README.md).

# Experimental HKS density and optimizer policies

`train_surface` now accepts independently selectable HKS-only flags:
`--density-policy hks` and `--optimizer-policy hks`. Both default to `original`.
Native HKTex always retains its original implementation and configured policies.
The custom density policy requires `--surface-adapter direct`.

## Density allocation

`training/density_controllers/residual_cost.py` accumulates detached absolute RGB residuals,
normalized selected influence, and candidate kernel/triangle exposure. At steps
400, 600, ... before 3500, it ranks per-kernel residual divided by support cost.
It can add at most 10% per event, up to 5000 kernels. A child inherits its parent
appearance/filter parameters and is placed at that parent's highest observed
weighted-error sample using the sample's mesh face and barycentrics.

Conservative pruning requires repeated support exposure, positive but tiny
selected influence, and a small observed residual. A never-selected kernel is
not pruned solely because it missed top-30. At most 2% are removed per event;
the minimum is the smaller of the initial count and 1000. These thresholds and
budgets are experimental defaults, not tuned or validated convergence choices.
Support cost counts triangle pairs; it is a proxy, not measured per-kernel GPU
time. Top-30 statistics are selection-biased and do not describe every candidate's
response. All statistics are reset at density events.

Density replacement is one row-mapping transaction. It retains survivor gradients
and moments, zeros child gradients/moments, updates kernel face IDs and optimizer
references, and regenerates location barycentric metadata. Global Adam step
counters remain unchanged. Fixed nested mesh/spectral buffers are never resized.
Original geodesic motion handles subsequent location updates.

The dedicated package is organized as follows:

| File | Purpose |
|---|---|
| `training/density_controllers/residual_cost.py` | HKS residual/support-cost statistics, conservative pruning and child allocation. |
| `training/density_controllers/transactions.py` | Parameter replacement with gradient, optimizer moment and mesh-face bindings preserved. |
| `training/density_controllers/__init__.py` | Public policy import. |
| `training/adaptive_density.py` | Small historical import shim; no duplicate implementation. |

The move preserves all policy and transaction method bodies. After the custom
policy's stop iteration, the HKS adapter now skips contribution-statistic
construction/observation; callbacks already return without density decisions at
those steps. Rendering, gradients, optimization and the density event schedule
are unchanged. Unused post-stop density statistics therefore stop accumulating.
Original HKTex controllers/defaults remain untouched.

Enable it explicitly with `--density-policy hks --surface-adapter direct`.
Keep `--optimizer-policy original` when isolating density-policy effects. A
40-step profile does not exercise density events: use a longer smoke run that
crosses step 400, then compare full training time and held-out quality. The
5000-kernel cap still applies, so a run initialized with 5000 kernels cannot grow
above that cap. This custom policy already existed as an experiment; moving it
into a folder does not establish a speed or quality improvement. GPU validation
of the new package/stop-path integration remains pending; no GPU job was run.
CPU checks passed for package/historical imports, unchanged moved method ASTs,
parameter/gradient/optimizer-moment/face-ID remapping, nested fixed-buffer
preservation and the stopped-policy forward/callback path. Syntax and CLI
previews passed. These checks do not establish CUDA correctness or speed.

## Optimizer

`training/kernel_optimizer.py` implements `HksAdam`, batching appearance updates
across parameter groups with Torch foreach operations. Separate learning rates,
standard Adam moments/bias correction and StepLR schedules remain. It supports
real dense tensors, shared betas and zero weight decay; unsupported recipes fail
explicitly. Floating point operation order may differ from native Adam.
This is a custom execution implementation of Adam, not a new learning equation.
The location optimizer/tracer is original HKTex GeodesicOpt, retaining momentum
transport and mesh constraints; it is deliberately not replaced with Euclidean
Adam. No external dependencies were added.

`training/custom_training.py` installs policies before any updates. CLI flags
are forwarded to workers, recorded in controls and reported as HKS experiments.
Density configuration/events are included in result JSON. Existing archival
checkpoint format records model, optimizer and density state; resume remains
unsupported. Full-model initial receipts and renderer RGB/VJP gates remain strict.

## Validation and running

First use an allocated A100 and the project's Python environment:

```bash
HK_PY=${HK_PY:-python}
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 1000 --iterations 601 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter direct --selector-merge row-radix \
  --density-policy hks --optimizer-policy hks \
  --no-training-plots --verify-samples 1024 --execute \
  --output "experiments/custom_hks_smoke_$(date +%Y%m%d_%H%M%S)"
```

The workflow runs initial renderer verification, then 601 updates to exercise
density events at 400 and 600. Inspect `hybrid_train/result.json` for density
events, parameter/optimizer health and effective policies. `--verify-only` checks
rendering/VJPs but does not exercise a density event or optimizer update.
Only after smoke validation, use 5000 updates with fresh output to assess time
and held-out quality. Separate density-only, optimizer-only and combined runs
from the original-policy direct-adapter control. Custom density changes kernel
counts and learning trajectories; total time to matched quality is the relevant
comparison, not an isolated renderer speedup claim.

CPU checks and GPU availability are recorded after implementation. No GPU jobs
are launched automatically. Prior results and staged/unstaged work are retained;
source recovery copies are in `/tmp/hks-custom-training-jmld2lb6` on this machine.

Completed CPU checks: twelve Adam steps with heterogeneous learning rates and
missing gradients matched native Adam parameters/moments; density replacement
retained survivor gradients/moments, zeroed child state, refreshed face/barycentric
bindings and allowed subsequent updates. Residual/sentinel accumulation, growth,
conservative pruning, never-selected protection, scheduler rebinding and learning
rate changes passed on synthetic tensors. Syntax and CLI previews passed for
native/HKS verification and training. Original HKTex sources, all existing Triton
kernel files and Git staging match the pre-change copies. No GPU driver was
accessible, so real-mesh density transitions and A100 forward/backward/update
validation remain pending. No speed or convergence improvement is claimed.

The first standalone CPU optimizer check encountered an installed Torch Dynamo
import-order error (`NP_SUPPORTED_MODULES`). Explicitly initializing
`torch._numpy` and `torch._dynamo` before the test resolved it; these CPU results
do not replace testing the actual worker environment. No installed package was
modified. Whole-tree `git diff --check` still reports pre-existing whitespace
in original `hktex/modules/tracer.py`, lines 285 and 290; it was left unchanged.

## Training-folder cleanup

Proposed deletion was reviewed before execution: `training/cached_sampling.py`
had exactly one optional runtime caller in `train_surface.py`; neither renderer,
preprocessing nor optimizer required it. That caller, the
`--cache-surface-sampling` parser option and worker flag forwarding were removed
together. Historical JSON cache fields remain readable and new runs record
false. Results and the old source recovery copy were preserved. Historical cached
runs require their recorded source. No generated training `.pyc` files were found.

Retained training modules and active callers:

| Module | Why required |
|---|---|
| `uv_training.py` | Original-loop adapters, receipts, gradient gates and worker training |
| `surface_adapter.py` | Rendering bridge and original density compatibility |
| `sample_tile_geometry.py` | Direct adapter sample grouping and dispatch |
| `surface_profile.py` | Explicit stage-profiling workflow |
| `adaptive_density.py` | New custom density policy |
| `kernel_optimizer.py` | New batched appearance optimizer |
| `custom_training.py` | Installs independent custom policies before training |
| `__init__.py` | Training package namespace |

Modified production files: `training/uv_training.py`, `training/surface_adapter.py`,
`rendering/heat_kernel_splatting.py`, `splatting_script/experiments/train_surface.py`.
Added production files are the three custom-policy modules above. Updated docs:
this note, `FULL_SURFACE_TRAINING.md`, `RADIX_AND_STARTUP_OPTIMIZATION.md` and
`ROW_SELECTOR_AND_TEXTURE_CACHE.md`. Only the verified sampler-cache source was
deleted. No meshes, checkpoints, results, original HKTex files or Git history
were removed or rewritten. No commit or push was performed.

### Native attachment regression fix

The first GPU smoke attempt (`custom_hks_smoke_20261120_214229`) failed in native
setup before any update: a duplicate `RecordedUvTrainer.attach_run` incorrectly
called a nonexistent original parent hook. That duplicate was removed; the
shared/native initializer now matches its pre-custom-policy implementation.
Only `HksUvTrainer.attach_run` installs the requested custom policies, once after
shared setup. No original HKTex source changed. The custom-policy report caption
was also moved before the job table to preserve valid Markdown formatting.

A temporary CPU regression check reproduced the old exception and executed the
actual adapter class bodies with isolated base/installer boundaries. All four
policy combinations initialized the native recorder without calling the custom
installer, and initialized HKS before calling the installer exactly once. The
check also rejects duplicate methods in every training class. Syntax, paired CLI
previews and report table structure passed. This is not a GPU worker rerun;
retry with fresh output and initialization. Recovery copies/checks are under
`/tmp/hks-attach-fix-jab0wmbj`; the failed run remains unchanged.
