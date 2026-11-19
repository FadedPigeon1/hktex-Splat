# Shared HKS module names

Date: November 17, 2026 (America/Chicago).

## Rename plan and dependency review

These are shared stages of the existing Heat-Kernel Splatting implementation.
The refactor moves three complete modules without splitting implementations,
renaming functions/classes, or changing mathematical operations.

| Previous path | New path | Responsibility |
| --- | --- | --- |
| `hktex_splatting/rendering/triangle_splat_reference.py` | `hktex_splatting/rendering/triangle_tile_layout.py` | Fixed visible triangle/tile patches, ordered incidences and bounded dispatch records. Required by normal HKS geometry preparation. |
| `hktex_splatting/triton/triangle_splat_forward.py` | `hktex_splatting/triton/splat_contributions.py` | Shared top-30 contribution ranking plus generic emission and RGB operations retained for correctness comparisons. |
| `hktex_splatting/triton/runtime_splat_kernels.py` | `hktex_splatting/triton/splat_reconstruction.py` | Main HKS RGB reconstruction and selected contribution backward kernels with runtime pair bounds. |

Dependencies checked before moving:

- `rendering/prepared_geometry.py`, `gpu_dispatch.py` and `splat_autograd.py`
  import the fixed-layout types or planner.
- `triton/heat_kernel_splatting.py` imports the shared contribution merge and
  runtime reconstruction kernels.
- `rendering/splat_autograd.py` imports both GPU modules for the shared backward
  and internal correctness reference.
- `splatting_script/benchmarks/renderer_comparison.py` and `pipeline_profile.py`
  import/patch these same GPU modules. Their references must move together.
- Package discovery and benchmark source hashing already enumerate the actual
  package directories; there is no hardcoded registration of these filenames.
- Current initialization/training/debug snapshots store tensors and state
  dictionaries, not pickled instances of the moved layout class.

The main Python and Triton `heat_kernel_splatting.py` modules keep their names:
the former orchestrates rendering and the latter emits GPU tile contributions.
Original `hktex/`, support selection, fallback limits, optimizers, function
signatures and profiler stage labels are outside this refactor's scope.

## Compatibility

Use the new canonical module paths. No old-name proxy files are introduced.
Command names, backend identifiers and model/optimizer state-dictionary keys
remain unchanged. External imports and arbitrary object pickles that explicitly
name a retired module require updating or the matching historical source.

Historical cleanup reports, inventories and measurements retain their original
paths as recovery evidence. Current README and architecture references are
updated. Source provenance hashes and Triton kernel module identities change,
so comparisons should use a fresh matched workflow and may compile new kernels.

## Verification

| Check | Result |
| --- | --- |
| Python syntax and executable AST comparison | 176 modules matched the starting source after normalizing only the renamed imports and docstrings. |
| Original HKTex and author scripts | Byte-identical to the starting working tree. |
| Complete decorated Triton definitions | All 21 definitions byte-identical, including specialization settings and equations. |
| Runtime imports | 51 canonical HKS/benchmark modules imported successfully with CUDA hidden. |
| Renderer/backward interfaces | `DirectTileRenderer` still defaults to `selection='hybrid'`; `_HybridSplat` still inherits the same `_Splat.backward`. |
| CLI previews | Eight native/HKS/alias, correctness, profiler, standard-workflow and scaling-workflow previews passed without GPU work. |
| Package discovery | Both package families and existing console entry points remain available. |
| Deterministic numerical comparison | 32 before/after arrays matched exactly: fixed triangle/tile layout, record batching, contribution emission, top-30 selection, RGB and contribution/color/heat/filter adjoints. |
| Current documentation | 44 local links/anchors checked; Markdown fences balanced. |
| Old module references | None remain in active HKS/benchmark Python or the maintained README/architecture guides. Historical records intentionally retain old paths. |
| Git index | Unchanged; nothing staged, committed, reset, stashed or pushed. |

Numerical kernel checks used **Triton's CPU interpreter**, not a GPU. The probe
used small deterministic CPU arrays and stayed outside the repository; no retired
test infrastructure was recreated. This establishes rename parity for those
inputs, not CUDA execution, real-scene training or performance. A fresh A100
correctness/smoke run remains necessary for GPU validation.

## Files changed

The three complete-module moves appear in the first table. Six other Python
files changed only their import paths:

- `hktex_splatting/rendering/prepared_geometry.py`
- `hktex_splatting/rendering/gpu_dispatch.py`
- `hktex_splatting/rendering/splat_autograd.py`
- `hktex_splatting/triton/heat_kernel_splatting.py`
- `splatting_script/benchmarks/renderer_comparison.py`
- `splatting_script/benchmarks/pipeline_profile.py`

The moved modules also received purpose docstrings. Current file references were
updated in `hktex_splatting/README.md` and
`docs/HEAT_KERNEL_SPLATTING_ARCHITECTURE.md`. This report records the plan,
compatibility limits and results. `pyproject.toml` needed no changes because its
package discovery already includes the new filenames.

No implementation body, asset, checkpoint, historical measurement or recovery
record was removed. No compatibility stubs were added.

## Recovery and remaining risks

The starting worktree already contained staged, unstaged and untracked work;
those changes were preserved. Before-images and verification receipts for this
operation are in `/tmp/hks-shared-names-hly6rawh/`. This is temporary storage,
not a permanent recovery archive. Several moved sources were untracked at the
start, so a checkout of the existing HEAD alone cannot recover those versions.
Existing historical archive/recovery records remain unchanged.

External scripts must adopt the three new import paths. Historical whole-object
pickles and external tools matching old fully qualified Triton kernel names may
need their original source or an update. State dictionaries and profiler stage
labels are unchanged. No GPU or performance improvement is claimed.
