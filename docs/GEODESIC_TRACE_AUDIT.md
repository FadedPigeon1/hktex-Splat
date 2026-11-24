# Geodesic warning investigation

Inspected the retained `experiments/surface_density_check_new` runs on November
20, 2026. Hardware/software recorded by those runs: NVIDIA A100-PCIE-40GB,
digeo 0.0.4, original `high`/TF32 precision policy.

## Confirmed observations

| Observation | Original HKTex | HKS |
|---|---:|---:|
| Completed updates | 1,001 | 1,001 |
| Initial → final kernel count | 1,000 → 5,000 | 1,000 → 5,000 |
| Warning calls in training-worker log | 714 | 763 |
| Failed-sample events in those calls | 1,011 | 2,431 |
| Maximum failed rays in one warning | 4 | 6 |
| Invalid final kernel face IDs | 0 | 0 |
| Final kernels outside their face by barycentric tolerance `1e-4` | 0 | 0 |
| Maximum final point-to-assigned-triangle plane residual | 8.65e-8 | 7.90e-8 |
| Stored barycentric reconstruction residual, maximum | 2.17e-7 | 5.53e-7 |

The preceding four-update smoke runs had no geodesic warnings. Source/asset
checks, final parameter finiteness and optimizer-binding checks passed in the
1,001-update runs. These observations establish completion and finite state,
not successful movement of every kernel.

The fixed checkpoint mesh has 37,691 vertices, 75,000 triangles, 392 boundary
edges, no edges with more than two incident triangles, and no zero/near-zero-area
triangles under the diagnostic scale-aware double-precision threshold. Boundaries
are present; they are not by themselves proof of malformed geometry.

All initial face IDs and barycentrics were valid. Final geometry and stored
barycentric metadata also remained consistent within float32 tolerance. This
rules out persistent gross off-surface corruption in those saved states; it does
not rule out transient bad inputs between checkpoints.

The warning indices are **local trace-batch indices**, not persistent kernel IDs.
Density operations can reorder parameters; split calls trace a separate batch of
children. A frequently printed index cannot safely be matched to that row in the
final checkpoint. Logs do not associate each warning with an exact training step
or save its starting ray, so they cannot establish the original failing input.

## Initial source review

Both methods use the original `GeodesicOpt` → `GPUGeodesicTracer.trace` →
`digeo.ops.trace_geodesics` path. The warning is emitted when the returned face ID
is negative, before the original fallback restores the starting position/face.
See [the original fallback](../hktex/modules/tracer.py) and
[the original optimizer](../hktex/modules/geodesic_opt.py).

Source review identified two issues to investigate, not confirmed explanations
of the recorded GPU failures:

- The fallback restores failed coordinates/face IDs but still transports momentum
  through the rotation supplied by the failed trace. A CPU stub test demonstrates
  that this path can change momentum while leaving position unchanged. The real
  failed-ray rotations were not retained, so their effects are not established.
- Installed digeo's CUDA hole-detection branch subtracts one from an edge index
  whose convention elsewhere is already `0..2`; index zero can consequently
  access column `-1`. The installed sources also contain unchecked vertex/hole
  corner cases. Relevant package files are `digeo/ops/cuda/geodesic_utils_cuda.cu`
  and `digeo/mesh.py`. The actual failing path must be replayed before attributing
  these warnings to a specific source issue or changing the installed library.

CPU replay of 84 small momentum-derived proposals from selected **final-state**
rows returned valid endpoints, with hole avoidance both enabled and disabled.
Those proposals were not the historical failed inputs, and CPU/GPU execution is
different. This did not reproduce or resolve the original failure.

No mesh, spectral archive, optimizer, learning rate, original HKTex file or
installed digeo source/binary was changed during this investigation.

## Capture the actual failed inputs

The optional `--trace-diagnostics` mode observes raw digeo outputs **before**
HKTex repairs them. It records exact starting faces/UVs, actual coordinates and
directions, raw negative endpoint faces, transport rotations, momentum before/
after fallback, the training step, current Torch CUDA stream and in-place versus
out-of-place call type. It also retains the exact tracer mesh and call options.
Only the first eight failing calls (up to 16 rays each) are saved; all failed
events are counted. Files are `geodesic_diagnostics.json` and, if a case occurs,
`geodesic_failures.pt` inside the worker's run directory.

Start with the native baseline to isolate the shared movement path:

```bash
"$HK_PY" -B -m splatting_script.experiments.train_surface \
  --backend native --kernels 1000 --iterations 320 --seed 0 \
  --trace-diagnostics --execute \
  --output experiments/geodesic_native_diag_new
```

Use a new output directory. This starts fresh verification and training workers;
the previous snapshots/results remain untouched. It does not change optimizers,
trace options, hole handling, learning rates or failed-ray fallback behavior.
If no failing case is captured, do not interpret that as a fix; a later diagnostic
window may be needed.

Capture adds CUDA synchronization and copies. It can change scheduling and its
timings are **diagnostic only**. The normal path leaves capture disabled. A CPU
test using the actual original trace-method body verified identical returned
positions/faces/momentum, raw-failure capture, bounded storage, correct counters
and restoration of wrapped functions, including exception handling. The subsequent
user-run GPU capture is described below; the implementation agent ran no GPU jobs.

New comparison reports display whole-worker warning counts even without capture.
Existing reports can be reviewed using retained logs; historical result files
were not overwritten. Counters include repeated optimizer/density/setup events,
not unique kernels, and are not a per-update or failed-movement percentage.

## Exact captured failure: reproduced

The subsequent native diagnostic run is retained locally at
`experiments/geodesic_native_diag_new`. It completed 320 optimizer calls with
320,000 attempted rays and 407 negative-end-face events (0.1271875%). The first
failure was at zero-based step 9, before any density event, on CUDA stream 0.
It was an in-place optimizer call, not splitting or pruning.

The first saved ray starts on face 14,498 with UV coordinates approximately
`[0.0043127434, 0.8892034888]`. Its start reconstruction residual is 7.45e-9,
and its barycentrics are valid. The captured GPU output face is `-1`.

Safe Python replay of that exact ray with the saved mesh and hole avoidance
enabled follows boundary faces 14,498 → 14,497 → 14,792. At vertex 7,657,
the incident-face list contains only triangle 14,792. The installed digeo
`compute_parallel_transport_vertex_hole` helper excludes the current triangle
from both candidate searches. There are no other candidates, so its selected
face remains `-1`, which it returns without a validity guard. The replay was
stopped at that return, before further invalid indexing.

The native C++ CPU replay with hole avoidance enabled exits with SIGSEGV on the
same captured input. Core-file output was disabled. Python and C++ CPU replays
with hole avoidance disabled both return a valid boundary endpoint on face
14,498, at UV approximately `[0, 0.8902995586]`. Replays used a bounded step
limit; they did not change the installed library, original mesh or training.
Intermediate GPU hops were not captured, so the listed path is the Python
reproducer, not a claimed GPU instruction trace.

This establishes an unhandled single-incident-face boundary-corner case in the
shared tracer. It is independent of HKS splatting and precedes density changes.
The earlier edge-index concern is a separate source-review finding; it is not
needed to explain this reproduced corner failure. Other warning events may
still require their own captured cases.

The momentum concern is also confirmed for the captured GPU event: its transport
matrix has determinant zero, numerical rank one and orthogonality error about
1.414. It is not a valid 3D rotation. The original fallback restores the old
position exactly, but applies that matrix to momentum, changing the stored
vector despite the rejected movement. This is a consequence of the failed
trace, separate from the initial boundary-corner trigger.

## Next controlled experiment

The optional overlay
[`converse_boundary_stop.yaml`](../splatting_script/configs/converse_boundary_stop.yaml)
now sets `trainer.tracer.avoid_holes: false` for **both** backends. Original files
and the default configuration remain unchanged. The workflow uses the original
config loader's overlay support and records the overlay hash and effective
policy. CPU checks confirmed that this is the only configuration change and
that all verification/training workers receive the same overlay.

Follow the [320-update comparison command](FULL_SURFACE_TRAINING.md#controlled-boundary-stop-experiment)
with fresh initialization and a new output directory. This tests the valid
boundary-stop behavior observed in CPU replay. It changes the boundary policy;
it is not a silent fix or evidence of full-training correctness. GPU execution
of the short training workload remains pending. Revalidate it before any longer
run or performance claim.

The user's first overlay attempt, `experiments/surface_boundary_stop_320_new`,
passed native verification but stopped HKS at the exact initial-state identity
check, before its RGB/VJP gate or any training update. Both manifests recorded
identical controls and `avoid_holes=false`. The rejected HKS state was not saved
by that version, so the differing component cannot be recovered from those logs.
This attempt neither validates nor disproves the boundary-stop policy. New
failures retain `initial_mismatch.json` and `initial_actual.pt` for diagnosis;
the exact match requirement remains unchanged.

The retry, `experiments/surface_boundary_stop_verify_retry`, retained both initial
states. RNG, sampler, evaluation and every model field except kernel coordinates
and face IDs matched exactly. Comparing coordinate/face-ID pairs found 998
identical kernels and two different placements among 1,000. The sorted selection
arrays consequently differed at 488 rows; this is not 488 independently moved
kernels. The maximum distance from a native point to its nearest HKS point was
about 0.049 in mesh coordinates. No optimizer step had run.

The original model constructor uses random surface sampling followed by greedy
farthest-point selection and returns the selected points in mask order. That
explains how a small selection difference can shift many array rows. A fresh GPU
replay of the sampler/FPS intermediate tensors has not established the exact
numerical cause of the two selection differences. Hole avoidance is not called
by this placement routine.

The optional `--shared-initial-model` policy now uses the first original kernel
placement in all workers, explicitly and with provenance. Only coordinate/face-ID
differences are eligible; all other state remains strictly checked. The original
initializer and default independent-replay behavior are unchanged. See
[shared placement details](FULL_SURFACE_TRAINING.md#explicit-shared-initial-placement).
GPU verification with this explicit policy remains pending.

No GPU training or GPU replay was launched by the implementation agent during
this investigation. The GPU evidence above comes from the user's retained
diagnostic capture; replay and inspection were CPU-only. All original HKTex and
rendering/spectral/GPU implementation files remained unchanged.
