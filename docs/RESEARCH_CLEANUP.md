# Research repository cleanup — 2026-11-14

The default remains `DirectTileRenderer(selection='hybrid')`: triangle-supported,
tile-based spectral Heat-Kernel Splatting. Native HKTex, the direct spectral
`splat` reference, resident comparisons, training, backward operations and the
benchmark/profiler workflow remain available.

## Proposed removals and dependency evidence

This list was produced **before deletion** in the conversation and in
[plan.json](research_cleanup/plan.json). Inspection covered 251 current Python
modules across the repository, AST import edges, symbol references, dynamic
import strings, CLI/configuration references, and current documentation.
Archived sources and generated results were treated as evidence/data, not as
active dependencies. An unimported module was not assumed unused if it exposed
a standalone entry point.

| Deleted source | Reason and evidence | Shared functionality preserved |
| --- | --- | --- |
| `hktex_splatting/modules/optimized_kernel_splat.py` | `OptimizedKernelRenderer`, `SparseFaceCache`, `interpolate_corners`, and `native_reduce` had no callers outside that obsolete experiment. The only active incoming import was `StageTimer` from the forward control renderer; the unused cached kernel-centric renderer also imported it. | `StageTimer` moved verbatim to `rendering/frame.py`; the retained forward renderer imports it there. |
| `hktex_splatting/modules/surface_kernel_splat.py` | `SurfaceKernelSplatRenderer`, its accumulation/selection helpers, and its parity driver had no callers outside the two obsolete experimental modules. `valid_triangles` was also used by the unused offline `build_map`. | `SurfaceFrame` moved verbatim to `rendering/frame.py`. The forward control and its tests still use it. |
| `hktex_splatting/triton/kernel_centric_gpu.py` | Its `shade` wrapper was called only by `KernelCentricRenderer.frame`; that class had no incoming symbol references or active entry point. Its pixel shader is separate from the hybrid emission and backward kernels. | All Triton operations used by native/hybrid/resident/splat and their validation remain. |

The unused `KernelCentricRenderer`, offline `build_map`, map save/load helpers,
`distance_coefficients`, and `select_support` were removed from
`rendering/kernel_centric.py`. `FaceMap`, `face_candidates`,
`polynomial_distance`, and `shade_numpy` remain at their existing import path.
The retained definitions and the two relocated classes were checked for exact
AST equality against the pre-cleanup source. No shader was substituted.

The user subsequently narrowed the sharing scope and explicitly requested old
visualization removal. A second pre-deletion
[plan](research_cleanup/visualization_plan.json) records these seven additional
deletions, with hashes and Git recovery commands:

- `scripts/visualise_all_kernels.py`
- `scripts/visualise_bh_weights.py`
- `scripts/visualise_eigenvectors.py`
- `scripts/visualise_hktex_steps.py`
- `scripts/visualise_kernel_split.py`
- `scripts/visualise_local_frames.py`
- `scripts/visualise_single_kernel.py`

No preserved implementation, test, configuration or required command imports
or calls them. Their distinct eigenvector/biharmonic/intermediate-stage figure
work was explicitly flagged before deletion. Existing figure outputs and
research data remain; the source is recoverable from Git's index. In contrast,
`scripts/visualize_kernel_faces.py` is a real transitive camera/raster dependency
of the current renderers and essential tests, so it remains.

The three removed module entries were removed from the legacy relocation
table. That table also contains historical destinations retired in earlier
cleanups; those historical strings are not executable dependencies. Historical
manifests and result records were not rewritten.

## Why footprint code remains

| Code or policy | Current use |
| --- | --- |
| `rendering/footprint.py:projected_bounds` | `DirectTileRenderer._hybrid` calls it when `footprint_cutoff > 0`, bounds are not full, and the exhaustive oracle is off. The default cutoff is still zero. |
| `rendering/footprint.py:prune_dispatch` | The legacy forward trainer uses it after projection. CPU tests exercise it against actual spectral contributions. |
| `rendering/direct_tiles.py:bin_footprints` | Used by the retained native-selection tile control and its validation; it is not dead projected-conic code. |
| `rendering/direct_tiles.py:kernel_footprints` | Builds the direct spectral `splat` reference's kernel/tile queue. |
| Triangle/tile support and `triangle_tile_work` | Schedules the main hybrid renderer. Removing this as “footprint code” would remove the main pipeline. |

Concrete callers include `rendering/direct_tiles.py`, `trainers/forward.py`,
`validation/test_direct_tiles_cpu.py`, and
`validation/forward_splatting/test_training.py`. The original pointwise HKTex
baseline in `hktex/` does not import the splatting footprint module. The
benchmark forwards and records the cutoff; research speed comparisons retain
the existing zero-cutoff requirement. No footprint flags or tests were removed.

## Retained for comparison or review

- All original `hktex/` implementations and root training/interactive tools
  remain. The original benchmark/precomputation tools remain distinct from the
  deleted unreferenced plotting scripts. Lack of an incoming import alone does
  not establish that a standalone baseline command is obsolete.
- `rendering/native_reference.py` remains a standalone saved-model native
  evaluation command and a documented reference.
- The [abandoned renderer plan](research_cleanup/abandoned_renderers_plan.json)
  records four further verified deletions: `scripts/triangle_splat_renderer.py`,
  `scripts/surface_splatting.py`, `scripts/compare_triangle_splat_backends.py`,
  and `scripts/test_triangle_splat_backends.py`. Their distinct cached/selected
  face renderer and Gaussian surface-splat work was flagged before deletion.
  The last file tested only those retired renderers; no hybrid gradient,
  support-ordering, training, or GPU validation assertion was removed.
- The two CPU timing/raster helpers required by the NumPy/Numba parity tests
  moved verbatim into the existing `scripts/test_numba_raster.py`. The three
  camera/clipping/reference-raster functions remain verbatim at
  `scripts/visualize_kernel_faces.py`; its unused footprint-export CLI and
  diagnostic selection/color functions were removed in place.
- Resident dispatch, forward controls, CPU numerical oracles, and the legacy
  trainer remain because they are used by comparisons and correctness tests.
- All meshes, spectral assets, trained checkpoints, research results, original
  identities, and recovery archives remain. The metadata-only
  [storage inventory](research_cleanup/storage_inventory.json) identifies the
  repository's top-level storage categories; it is not a deletion plan.
- `docs/three_backends_cleanup/` and the existing consolidation inventories are
  historical audit records. Their source hashes describe those earlier states.

No standalone research script or correctness test was deleted simply to reduce
the number of files. No algorithm, cap, cutoff default, support assertion,
4096-source fallback, optimizer or gradient behavior was changed.

## Git recovery and review

The repository already contained staged edits, unstaged edits, untracked source,
and deletions. Cleanup stayed on `cleanup/hybrid-main-renderer` and preserved
that existing work. Before editing, a separate snapshot was recorded at
`/tmp/hktex-research-cleanup-1xehgi_b`. The seven cleanup input sources that
were still untracked were added to Git's index before mutation. The current
`triangle_splat_renderer.py`, including its pre-existing unstaged edits, was
also retained in the index before deletion. Other pre-existing staged content
was preserved. All 14 deleted files are recoverable without a commit:

```bash
# Inspect this cleanup's deletions against their exact retained input bytes.
git diff --stat -- hktex_splatting Splatting_script/validation/run_retained_cpu.py
git diff --name-only --diff-filter=D -- hktex_splatting scripts

# Restore one deleted source from the index, if desired.
git restore --worktree -- hktex_splatting/modules/optimized_kernel_splat.py
```

The index intentionally retains pre-cleanup source for recovery. No commit,
push, reset, stash, branch-history rewrite, or broad staging operation occurred.
Review staged additions before making a later commit; this repository had many
pre-existing staged changes. The before/after manifest in
`research_cleanup/validation.json` distinguishes this cleanup from earlier work.
The old triangle CLI had no separately staged edits; its previously unstaged
source is now the explicit recovery preimage in the index.

## Validation

The full canonical CPU runner now includes the previously omitted
`test_training_research_cpu` and `test_cleanup_recovery` suites, in addition to
the numerical, selector/fallback, forward/backward, profiler, CLI, and parity
suites. Tests run in isolated subprocesses with CUDA hidden, one CPU thread,
bytecode writes disabled, and the existing Triton interpreter fixtures.

```bash
HK_PY=/scratch/user/"$USER"/.conda/envs/hktex/bin/python
"$HK_PY" -B -m Splatting_script.validation.run_cpu
CUDA_VISIBLE_DEVICES='' "$HK_PY" -B scripts/test_numba_raster.py --synthetic-only
```

Execution results, source-preservation checks, syntax/import/reference checks,
and the scoped Git diff are recorded in
[validation.json](research_cleanup/validation.json). Historical benchmark
measurements remain historical; this cleanup makes no new performance claim.

Completed CPU validation: **207 tests across all 20 canonical suites passed**;
the extracted/reference/Numba raster comparison passed **42 synthetic cases**.
The 12 old triangle-backend integration tests passed at the intermediate
checkpoint and were then retired with their exclusive renderer; they are not
counted among the remaining 207 tests. The benchmark workflow preview passed
without importing Torch/Triton or starting GPU work. All six edited
documentation pages' local links resolve. The full CPU logs and summary are in
`experiments/converse_heat_kernel_splatting/results/research_cleanup_cpu_20261114/`.

The Python-only scoped Git diff is **21 files changed, 203 insertions and
5,028 deletions**, a net reduction of **4,825 lines**. Documentation and audit
records are additional changes. Older staged/unstaged edits and deletions that
were present before this task are excluded from this source summary.

No allocated Slurm GPU was available. GPU numerical parity, CUPTI timing and
real CUDA backward execution require the existing manual validation commands
inside an allocation. `scripts/test_renderers.py` is a manual Mitsuba/CUDA demo
using external checkpoints, rather than an automated CPU correctness suite; it
was retained and was not launched.
