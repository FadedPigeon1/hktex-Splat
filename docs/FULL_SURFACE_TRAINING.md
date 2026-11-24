# Original HKTex lifecycle with HKS surface-sample training

This experimental workflow uses **the original UV surface samples for both
methods**. The original files under `hktex/` and `optimisation.py` are not edited.
The existing camera-image benchmark remains a separate command and protocol.

## What is reused

Both adapters inherit `hktex.trainers.base.BaseTrainer.optimise` itself. Its
data-dependent initialization, original UV/importance sampler, loss, backward,
density callbacks, optimizer/zero-gradient ordering, schedulers, parameter
constraints and diagnostic plots remain in that function. Adam, GeodesicOpt and
the original pruning/densification implementations are reused.

`RecordedUvTrainer` delegates native preparation/forward to the original code
and records receipts. `HksUvTrainer` replaces only source/batch/forward
preparation with the existing HKS triangle selection, three-corner coefficients,
barycentric interpolation, tile splatting, top-30 RGB and custom backward.
Kernel movement still uses the original optimizer/tracer.

Both recording adapters keep `pool_indices` on CPU for the original importance
sampler's CPU error pool. These IDs are feedback metadata, not model inputs;
surface samples, supervised colors, error values and sampler update equations
are unchanged. The original trainer/sampler source files are not patched.

HKS packs each batch's original face IDs and barycentrics into a padded sample
grid, preserving every sample once and in order. These are **work tiles, not
camera pixels**. No projection, visibility or camera rendering is implied.
Support/topology is reconstructed per batch, so pruning, cloning and splitting
cannot leave a cached source-count-dependent support table behind.

The original density callbacks require rectangular valid top-k indices. HKS
padding is excluded by grouping points with the same actual number of selected
contributions, then calling the original accumulation methods on those groups.
Valid zero-weight slots remain hits, matching the original controller's rule.
Grouping can change floating-point accumulation order across groups. Controller
equations and optimizer-state migration are not reimplemented. Unknown density
controller types are rejected rather than skipped.

## Run

Install the environment and prepare the mesh/archive using the
[HKS guide](../hktex_splatting/README.md). Run from the repository root. A usable
CUDA GPU is required for execution; Slurm is optional. Tools never allocate a
GPU or run training during previews.

```bash
HK_PY=${HK_PY:-python}

# Preview: no Torch import, output files, preprocessing or GPU work.
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 1000 --iterations 1000 --seed 0

# New surface-layout health/RGB/VJP gates only: no optimizer updates.
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 1000 --iterations 4 --seed 0 \
  --verify-only --execute --output experiments/surface_verify_new

# Short training smoke. It does not reach this config's density schedules.
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 1000 --iterations 4 --seed 0 \
  --execute --output experiments/surface_smoke_new

# Full comparison, including configured pruning/densification schedules.
# Omit --iterations to use this config's original 5,000-update budget.
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 1000 --seed 0 \
  --execute --output experiments/surface_training_new
```

Use a new output directory for every invocation. On the existing Grace setup,
`HK_PY=/scratch/user/$USER/.conda/envs/hktex/bin/python` selects the user's
environment; other machines should use their own activated environment.
If native libraries require it, set
`LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`.

The default config is `splatting_script/configs/converse_training.yaml`. Omitting
kernel/iteration/seed options preserves its values. It starts density operations
at its recorded schedules and caps densification at 5,000 kernels. Starting with
5,000 kernels leaves no growth room until pruning frees capacity. Neither
budgets nor controller limits are increased automatically. Configured importance
sampling stays enabled. `data.num_workers=0` is required to verify feedback and
sampler state; unsupported trainer/data recipes are rejected. Spectral
precomputation must already exist and is never triggered by this command.

Steps are zero-based: this config first densifies at step 400 and first checks
pruning at step 1,000. A 1,000-update run ends at step 999 and does not exercise
pruning. Full GPU validation must include density events and coincident controller
callbacks, not only the four-update smoke or initial 32-sample gate.

## Controlled boundary-stop experiment

The retained diagnostic capture exposed a shared digeo boundary-corner failure
with the original `avoid_holes=true` policy. The separate overlay
[`converse_boundary_stop.yaml`](../splatting_script/configs/converse_boundary_stop.yaml)
sets only `trainer.tracer.avoid_holes: false`. This stops moves at mesh boundaries
instead of attempting to circumvent holes. Original HKTex code and the default
training config remain unchanged. This is an explicit experimental boundary
policy, not a change to heat equations or a validated tracer repair.

Run both methods with the same overlay, first for 320 updates. This reaches the
previously observed failure window but precedes the first density event:

```bash
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 1000 --iterations 320 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model \
  --trace-diagnostics --execute \
  --output experiments/surface_boundary_stop_320_new
```

Omit `--execute` to preview. Use a fresh output directory and do not supply an
initial snapshot from the original-policy run: source/configuration controls
must match. The original loader merges the base config and overlay before CLI
overrides. Both verification workers and both training workers receive the same
inputs. Manifests record the overlay hash and effective tracer policy; the
comparison table records `avoid_holes` separately for each backend.

This command explicitly shares the first original kernel placement, as described
below. Start with `--verify-only` added and a fresh output directory when repeating
the failed initialization checks; omit it only after both gates pass.

GPU validation of this overlay is pending. Inspect trace failures and finite
state before attempting longer runs with density changes. Diagnostic capture
adds synchronization/copies, so these timings are not performance evidence.
Omitting `--config-overlay` keeps the existing default policy. See the
[captured failure investigation](GEODESIC_TRACE_AUDIT.md) for evidence and limits.

## Verification and retained records

Workers run in separate processes. Both verification workers must pass before
either training worker starts. The first verification worker saves a common
`initial.pt`; every later worker checks its own original initialization, RNG,
importance pool and held-out surface batch against that snapshot **before any
optimizer update**. Mismatch is a failure, not a silent reset. Old camera-benchmark
snapshots are not accepted as full-surface snapshots.

If that exact identity check fails, `initial_mismatch.json` records which model,
RNG, sampler or evaluation components differed, with bounded per-field hashes
and numeric differences. `initial_actual.pt` preserves the rejected worker's
state alongside the original expected snapshot in the first worker's directory.
Training still stops: no tolerance is relaxed and no state is silently replaced.

### Explicit shared initial placement

`--shared-initial-model` is an optional comparison policy for independently
initialized positions that differ despite matching RNG and data. The original
initializer samples mesh points and applies greedy farthest-point selection.
It runs unchanged in each worker; the first verification worker saves its
original fully initialized model. Before any kernel preparation or optimizer
update, later workers explicitly copy that reference's positions and triangle
IDs into their existing tensors. Parameter objects and optimizer face-ID aliases
remain intact. This does not change the original initializer or training code.

Only `_kernel_locations` and `_kernel_face_ids` may differ under this policy.
All other model fields, RNG state, importance pool and held-out batch must
already match exactly. Shapes/dtypes must match; existing gradients, optimizer
state or cached barycentrics prevent adoption. The complete identity check runs
again afterward. With this flag omitted, the previous strict independent-replay
behavior is unchanged.

Each worker writes `initialization.json` with the policy, reference path, model
hashes before/after and copied fields. If a local placement is replaced, its
positions/face IDs are retained in `initial_kernel_proposal.pt`. Manifests and
the comparison report label this explicit policy. This is a common-initial-state
experiment, not evidence that the original GPU initializer replays identically.
Use fresh workflow initialization when changing this setting or source code.

Before the original loop starts, later workers also check source/config/asset,
software and GPU controls against the common snapshot's companion manifest.
Keep an explicitly supplied `--initial-state` beside its original `manifest.json`.
Only a different explicit update budget is allowed when reusing that snapshot;
the two current arms must still agree on their budget. Other changes require a
fresh initialization workflow and are never silently accepted.

Native verification checks finite RGB and parameter VJPs through original
pointwise evaluation. HKS compares RGB and all parameter VJPs with the retained
eager coefficient/splat control, using the existing tolerances (`atol=3e-5`,
`rtol=3e-4`). By default this is a bounded 32-sample gate, not proof for every
training configuration. That control shares the fused selector; continue to use
the independent selector/renderer GPU gate in the main HKS guide.

Each training worker saves:

- `manifest.json`, `resolved_config.json`, `initial_identity.json` and
  `sample_validation.json`: actual source hashes, software/hardware, asset hashes,
  precision, configured controllers/optimizers and initial-state receipts.
- `trajectory.jsonl`: zero-based step, loss, kernel counts and sampled-batch
  fingerprints (including importance-pool indices), plus parameter-replacement
  flags. This includes every density count change and net-zero replacements.
- `final.pt`: final parameters/buffers, kernel count, optimizer and scheduler
  states, RNG, importance pool/errors and density-controller state.
- `result.json`: timing scopes, allocator memory, held-out deployed/common-native
  quality, final model/checkpoint hashes and source/asset/optimizer-binding
  preservation checks.

The workflow retains `comparison.md`, `workflow.json`, logs and job receipts.
These archives are intentional: a single statistics file cannot verify evolving
sampler and optimizer state. Checkpoints are verification archives; **resume is
not implemented**. Existing results and original assets are never removed.

## Interpretation and limits

### Experimental surface-adapter optimization and stage profiling

For full adaptive training with **fresh triangle rasterization and actual camera
pixels**, use [`--supervision camera`](RASTERIZED_CAMERA_TRAINING.md). Surface
sampling and camera rendering are separate supervision modes and their results
must not be compared as identical workloads.

Separate [custom HKS training policies](CUSTOM_HKS_TRAINING.md) are available via
`--density-policy hks` and `--optimizer-policy hks`. Both are opt-in; native HKTex
remains unchanged. These experiments change training allocation/execution and
must be distinguished from comparisons isolating the renderer.

The opt-in [direct GPU surface adapter](DIRECT_SURFACE_ADAPTER.md) accepts
`--surface-adapter direct`. It bypasses image-grid packing and constructs flat
sample work tiles, while preserving the original sampler and training lifecycle.
The reference adapter remains available for forward/backward gates.

The [whole-row selector pass](ROW_SELECTOR_AND_TEXTURE_CACHE.md) adds
`--selector-merge row-radix` to reduce selection/merge overhead. GPU validation is
required. The proposed target-texture cache was withdrawn: the current focus is
faster HKS computation with the original sampler for both arms. Startup timings
are reported separately from the training interval.

An additional [research-guided radix and startup pass](RADIX_AND_STARTUP_OPTIMIZATION.md)
provides `--selector-merge radix` and
`--no-training-plots` as explicit options. Historical ray-capture implementation
now lives under `splatting_script/benchmarks/legacy/`; the trace flag still works
but normal training does not import that module.

An additional opt-in `--surface-adapter aggressive` caches fixed triangle bounds
and batches existing CUDA selector dispatches. See the
[aggressive optimization implementation and validation guide](HKS_AGGRESSIVE_OPTIMIZATION.md).

`--surface-adapter reference` remains the default, retaining the previous sample
layout and full training tape. The opt-in `--surface-adapter optimized`:

- Stably groups samples by triangle in work tiles, then restores RGB and selected
  contribution rows to the original sample order before loss/controller calls.
- Requests only selected IDs and weights from the unchanged splatting calculation,
  avoiding full response/debug tables and their diagnostic GPU-to-CPU count read.
- Computes valid-width groups once per batch for both original controllers. One
  small width histogram replaces up to 30 per-width GPU-to-CPU Boolean checks.
  Width order, zero-weight hits, sentinel exclusion and controller calls remain
  unchanged. Cache entries are cleared per batch and checked for tensor mutation.

The optimized GPU verification gate compares RGB and all parameter VJPs against
the ungrouped eager reference. It also compares selected IDs exactly and weights/
RGB within existing tolerances against the ungrouped Triton full-tape path.
Original HKTex code, selectors/fallbacks, heat equations, GPU kernels and optimizer
implementations are not edited. Grouping may alter floating-point reduction order;
GPU validation remains required. No speed improvement is claimed.

CPU validation passed full/compact/grouped rendering on 1/8/65 samples, including
selected IDs/weights and VJPs using synthetic differentiable coefficients. The
actual surface-forward assembly restored original sample order correctly. Actual
original density-controller method bodies produced exactly matching hit,
contribution and error accumulators across start/stop schedules, ragged support
and zero-weight slots. Cache mutation/clear guards, bounded profiler restoration
after exceptions, CLI previews and source/index preservation checks also passed.
These checks used eager CPU rendering and Triton's CPU interpreter for support
packing, not CUDA kernels or real-spectrum GPU evaluation. No test suite was
added to the repository. The implementation session had no usable NVIDIA driver.

Start with verification only on an allocated GPU (fresh output/initialization):

```bash
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 1000 --iterations 1001 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter optimized \
  --verify-only --execute --output experiments/surface_optimized_verify
```

After both gates pass, use a short diagnostic run to measure actual stages and
sample/triangle reuse around the first density event:

```bash
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend both --kernels 1000 --iterations 1001 --seed 0 \
  --config-overlay splatting_script/configs/converse_boundary_stop.yaml \
  --shared-initial-model --surface-adapter optimized \
  --profile-start 400 --profile-steps 4 \
  --execute --output experiments/surface_optimized_profile
```

Timing wrappers are installed temporarily around native/HKS preparation, forward,
loss, backward, density callbacks and optimizer steps. HKS includes geometry,
support, coefficient, tile-queue and splat sub-stages. Profiling drains CUDA before
and after each measured stage; inclusive nested times overlap and must not be
added. Sampling/schedulers are covered by whole-run timing but not separate stage
rows. Only requested steps after the initial receipt are measured. The existing
result JSON and comparison Markdown contain the summaries; no trace archive is
created. Profiling and geodesic diagnostics must be disabled for speed runs.

Full-batch geometry/support caching is deliberately absent: samples change and
kernel movement/density changes invalidate learned support. Fixed triangle-bound
caching and a GPU-native sample-layout builder require further profiling and
correctness work. The native sampler's CPU feedback metadata stays on CPU.

This is a full **UV surface-training lifecycle**, not the full lit Mitsuba
inverse-rendering experiment. Original plot calls remain, using the noninteractive
Agg backend. Precision follows `optimisation.py` (`high`/TF32), rather than the
camera benchmark's `highest` policy. Initialization, gates, receipt capture and
plots contribute overhead; reported timing scopes identify what is included.
HKS density tape capture and CPU sample-grid preparation are also included.
The post-training Mitsuba previews/video export performed by the
`optimisation.py` wrapper are not part of this comparison.

Once importance sampling uses learned errors, actual sampled batches can differ.
Pruning/splitting decisions and final kernel counts can differ as the learned
fields diverge. The report checks common initial controls and records whether
the subsequent batch trajectories stayed identical; it does not assert that
adaptive training is the fixed-workload camera benchmark. No validated speedup
is automatically claimed for this experimental integration.

Original CUDA reductions can vary numerically between runs. These records make
the executed setup/state inspectable; they do not promise bitwise repeatability.
The detailed camera-pipeline profiler remains available in the separate matched
camera benchmark. This adapter reports full-loop timing, not a new per-kernel
profiling system.

Original controller behavior, including parameter/optimizer replacement at density
events, is inherited as-is. The adapter does not silently repair upstream behavior
or alter gradient-transfer rules. A failure is recorded with its traceback and
blocks later jobs. Current working-tree source hashes include any pre-existing
local edits; they do not certify byte-identical upstream author code.

One original-code interaction needs manual/GPU review: the original loop passes
one `kernel_info` snapshot to all post-backward controllers. In the retained
configuration, pruning runs before densification when both fire on the same
step. If pruning changes the kernel ordering, later splitting may receive stale
per-kernel metadata. This is a source-based risk, not a measured GPU failure in
this implementation task. The adapter preserves that original behavior; it does
not silently change callback ordering or repair the original metadata logic.

GPU verification of this new integration is still required. No full training job
was launched while implementing it.

Subsequent user runs passed the 32-sample GPU gates and four-update smoke tests,
then completed 1,001 updates with growth to 5,000 kernels on an A100. The longer
runs produced geodesic movement warnings in both backends; these are not a clean
movement validation. See the [trace investigation and capture command](GEODESIC_TRACE_AUDIT.md).
Optional `--trace-diagnostics` records bounded raw failure cases without changing
the original tracer/optimizer behavior. Those instrumented timings are diagnostic.

## Implementation verification

| Check | Result and scope |
|---|---|
| Packing | Passed CPU cases from 1 to 4,096 samples: order, barycentrics, padding, no dropped/duplicated samples and invalid-input rejection. |
| Density accumulation | Original controller method bodies passed exact dense parity and ragged/padding/zero-weight/error-normalization cases on CPU. |
| Density state migration | Original clone/prune/split methods preserved optimizer references, state shapes, face-ID aliases and zero new moments with a CPU tracer fixture. |
| Original loop parity | Native/HKS recording adapters inherited the actual original `optimise` function; a six-update CPU fixture matched bare-loop parameters, optimizer moments, schedulers and sampler feedback exactly. Its HKS forward was a fixture, not real spectral rendering. |
| Packed renderer | Existing renderer forward/backward passed 1/8/65-sample permutation/padding cases with synthetic coefficients, eager Torch splatting and Triton's CPU interpreter for support scans/packing. This was not GPU or real-spectrum validation. |
| Initial-state gates | Actual receipt code passed matching snapshots, RNG restoration, mismatch rejection and old-format rejection on CPU; the GPU gate was stubbed for this check. |
| Workflow gates | CPU worker stubs exercised success, child failure and mismatched-control failure; failed verification blocked training. |
| Syntax/import surface | Python syntax, preview imports, full/verify-only plans, existing native/HKS previews and package/console discovery passed. Production CUDA imports remain GPU-dependent. |
| Boundary-stop overlay | CPU checks of the original typed config loader confirmed only the boundary policy changes; preview checks confirmed identical overlay forwarding to all four workers and unchanged no-overlay defaults. GPU execution is pending. |
| Shared placement | CPU checks loaded the actual rejected/reference GPU snapshots and reproduced exact matching model state after in-place adoption. Parameter/face-ID aliases, RNG, toy-loss VJPs, strict default behavior, non-placement mismatch rejection, receipts and all-worker flag forwarding passed. Optimizer bindings used a fixture; GPU execution is pending. |
| Preservation | Every original `hktex/` file and `optimisation.py` matched its task-start bytes. Existing Git index/staging was preserved. |

Verification scripts and fixtures were temporary; no standalone test suite was
added to the repository. GPU smoke, real spectral RGB/VJP gates and adaptive
density events still need execution on an allocated GPU.

During the shared-placement check, full CPU Adam construction hit an installed
Torch/Dynamo import error (`NP_SUPPORTED_MODULES` missing from
`torch._dynamo.utils`). No environment packages were changed. Binding checks
therefore used an optimizer fixture; they are not a claim of an executed original
optimizer update. Actual optimizer integration remains part of the GPU rerun.

## File map

| File | Responsibility |
|---|---|
| `hktex_splatting/training/surface_adapter.py` | Order-preserving sample packing, existing renderer calls and valid density-contribution groups. |
| `hktex_splatting/training/uv_training.py` | Inherited-loop adapters, initial receipts and bounded sample RGB/VJP gates. |
| `splatting_script/experiments/train_surface.py` | Preview, isolated workers, provenance, checkpoints and comparison report. |
| `splatting_script/configs/converse_boundary_stop.yaml` | Optional one-setting overlay for the controlled boundary-stop experiment; applied to both methods. |
| `hktex/trainers/base.py` | Original `optimise` loop, unchanged. |
| `hktex/trainers/uv_texture.py` | Original pointwise UV preparation, initialization and rendering, unchanged. |
