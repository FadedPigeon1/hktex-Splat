"""End-to-end validation for triangle_splat_renderer --compare-raster-backends.

On a Grace GPU node, in the HKTex environment, from the repository root:
    python scripts/triangle_splat_renderer.py \\
      --experiment outputs/uv-texture-fitting/cloudrunner_indexed@20261006-225053 \\
      --kernel-face-cache outputs/triangle_splat/cloudrunner_kernel_face_cache_cutoff_0.005.npz \\
      --cutoff 0.005 --image-width 128 --image-height 128 --fov-y 45 \\
      --camera-position 0 3 0 --camera-look-at 0 0 0 \\
      --visibility-mode raster --raster-backend numba --splat-raster-mode reuse \\
      --compare-raster-backends --benchmark-runs 5 \\
      --raster-backend-report outputs/triangle_splat/backend_side_128.json

Repeat with --camera-position 2.5 -1 1 and report backend_angled_128.json.
Use --validate-raster-reuse for an additional selected-only NumPy check.
The underlying heat-kernel evaluator and its mathematics are unchanged.
Two renders per backend report repeatability and exact input equality before
strict checks. Failed diagnostics are saved to --raster-backend-report, or to
outputs/triangle_splat/raster_backend_diagnostics.json when no path is supplied.
Strict RGB validation is the default. --allow-rgb-mismatch completes benchmarking
with RGB/full correctness marked FAILED while retaining fatal raster/support checks.
"""

import argparse
from contextlib import redirect_stdout
import io
import json
import platform
import statistics
import time

import numpy as np


class RGBParityError(AssertionError):
    """An unchanged RGB tolerance check failed after all surface checks passed."""


def compare_buffers(reference, candidate):
    """Report exact mismatches, permitting only float64 roundoff in float buffers."""
    depth, faces, barys = reference
    cdepth, cfaces, cbarys = candidate
    covered = faces >= 0
    ccovered = cfaces >= 0

    def mismatch_count(a, b):
        return int(np.count_nonzero(~((a == b) | (np.isnan(a) & np.isnan(b)))))

    metrics = {
        "numpy_visible_pixels": int(covered.sum()),
        "numba_visible_pixels": int(ccovered.sum()),
        "coverage_mismatches": int(np.count_nonzero(covered != ccovered)),
        "face_id_mismatches": int(np.count_nonzero(faces != cfaces)),
        "depth_value_mismatches": mismatch_count(depth, cdepth),
        "barycentric_value_mismatches": mismatch_count(barys, cbarys),
        "max_depth_absolute_error": float(np.max(np.abs(depth[covered] - cdepth[covered]))) if covered.any() else 0.0,
        "max_barycentric_absolute_error": float(np.max(np.abs(barys[covered] - cbarys[covered]))) if covered.any() else 0.0,
    }
    np.testing.assert_array_equal(cfaces, faces)
    np.testing.assert_array_equal(np.isinf(cdepth), np.isinf(depth))
    np.testing.assert_array_equal(np.isnan(cbarys), np.isnan(barys))
    np.testing.assert_allclose(cdepth, depth, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cbarys, barys, rtol=1e-10, atol=1e-12, equal_nan=True)
    return metrics


def compare_frames(reference, candidate, strict_rgb=True):
    """Check visibility, filtered support, selected surfaces, and unquantized RGB."""
    np.testing.assert_array_equal(candidate["kernel_ids"], reference["kernel_ids"])
    if len(candidate["visible_sets"]) != len(reference["visible_sets"]):
        raise AssertionError("Backend kernel-to-face support counts differ")
    for original, alternate in zip(reference["visible_sets"], candidate["visible_sets"]):
        np.testing.assert_array_equal(alternate, original)
    metrics = {"full_mesh_visibility": compare_buffers(reference["visibility"], candidate["visibility"])}
    _, depth, faces, barys = reference["buffers"]
    _, cdepth, cfaces, cbarys = candidate["buffers"]
    metrics["splat_surfaces"] = compare_buffers((depth, faces, barys), (cdepth, cfaces, cbarys))
    np.testing.assert_array_equal(candidate["buffers"][0], reference["buffers"][0])
    if reference["reused"] != candidate["reused"]:
        raise AssertionError("Backend raster-buffer reuse decisions differ")
    metrics["raster_buffers_reused"] = reference["reused"]
    metrics["visible_kernels"] = len(reference["kernel_ids"])
    rgb, crgb = reference["colors"], candidate["colors"]
    np.testing.assert_equal(crgb.shape, rgb.shape)
    errors = np.abs(crgb.astype(np.float64) - rgb.astype(np.float64))
    covered = (faces >= 0) | (cfaces >= 0)
    visible_errors = errors[covered]
    metrics["rendered_rgb"] = {
        "exact_channel_mismatches": int(np.count_nonzero(rgb != crgb)),
        "covered_pixel_mae": float(visible_errors.mean()) if covered.any() else 0.0,
        "covered_pixel_rmse": float(np.sqrt(np.mean(visible_errors ** 2))) if covered.any() else 0.0,
        "covered_pixel_max_absolute_error": float(visible_errors.max()) if covered.any() else 0.0,
        "full_image_max_absolute_error": float(errors.max()),
    }
    try:
        np.testing.assert_allclose(crgb, rgb, rtol=0.0, atol=1e-6)
    except AssertionError as error:
        metrics["rendered_rgb"]["parity"] = "FAILED"
        metrics["rendered_rgb"]["assertion_failure"] = str(error)
        if strict_rgb:
            raise RGBParityError(str(error)) from error
    else:
        metrics["rendered_rgb"]["parity"] = "PASSED"
    return metrics


def diagnose_frames(reference, candidate):
    """Collect exact input equality and RGB differences before strict assertions."""
    def exact_buffers(a, b):
        return {name: bool(np.array_equal(x, y, equal_nan=True))
                for name, x, y in zip(("depth", "face_ids", "barycentrics"), a, b)}

    rgb, crgb = reference["colors"], candidate["colors"]
    errors = np.abs(crgb.astype(np.float64) - rgb.astype(np.float64))
    differing = np.flatnonzero(errors.ravel() > 0.0)
    largest = differing[np.argsort(-errors.ravel()[differing], kind="stable")[:10]]
    locations = []
    for index in largest:
        y, x, channel = np.unravel_index(index, errors.shape)
        locations.append({"x": int(x), "y": int(y), "channel": int(channel),
                          "absolute_error": float(errors[y, x, channel]),
                          "reference_rgb": float(rgb[y, x, channel]),
                          "candidate_rgb": float(crgb[y, x, channel]),
                          "reference_face_id": int(reference["buffers"][2][y, x]),
                          "candidate_face_id": int(candidate["buffers"][2][y, x])})
    support_equal = (np.array_equal(reference["kernel_ids"], candidate["kernel_ids"])
                     and len(reference["visible_sets"]) == len(candidate["visible_sets"])
                     and all(np.array_equal(a, b) for a, b in
                             zip(reference["visible_sets"], candidate["visible_sets"])))
    return {
        "exact_full_mesh_visibility": exact_buffers(reference["visibility"], candidate["visibility"]),
        "exact_splat_surfaces": exact_buffers(reference["buffers"][1:], candidate["buffers"][1:]),
        "selected_kernel_ids_exactly_equal": bool(np.array_equal(reference["kernel_ids"], candidate["kernel_ids"])),
        "kernel_to_face_support_exactly_equal": bool(support_equal),
        "reuse_decisions_equal": reference["reused"] == candidate["reused"],
        "differing_rgb_channels": int(np.count_nonzero(rgb != crgb)),
        "rgb_channels_above_1e_6": int(np.count_nonzero(errors > 1e-6)),
        "max_rgb_absolute_error": float(errors.max()),
        "largest_rgb_differences": locations,
        "coordinate_convention": "zero-based x=column, y=row; channel 0=R, 1=G, 2=B",
        "exact_equality_convention": "elementwise equality; matching NaN sentinels are equal",
    }


def compare_backends(mesh, affected_sets, args, numba_args, raster_geometry,
                     trainer, gating_lookup, timing, synchronize, force_separate, startup):
    """Compare the shared full-frame path with prepared KNN/gating, outside profiling."""
    from triangle_splat_renderer import OUTPUT_DIR, render_splat_frame
    import numba
    import torch
    from test_numba_raster import _raster_kernel

    numpy_args = argparse.Namespace(**vars(args))
    numpy_args.raster_backend = "numpy"
    backend_args = {"numpy": numpy_args, "numba": numba_args}
    allow_rgb_mismatch = getattr(args, "allow_rgb_mismatch", False)
    report = {
        "validation_policy": "benchmark_allow_rgb_mismatch" if allow_rgb_mismatch else "strict",
        "environment": {"hostname": platform.node(), "machine": platform.machine(),
                        "python": platform.python_version(), "numpy": np.__version__,
                        "numba": numba.__version__, "torch": torch.__version__},
        "experiment": str(args.experiment), "mesh_faces": len(raster_geometry[1]),
        "kernel_face_cache": str(args.kernel_face_cache) if args.kernel_face_cache else "default",
        "camera": {"position": args.camera_position, "look_at": args.camera_look_at,
                   "width": args.image_width, "height": args.image_height, "fov_y": args.fov_y},
        "cutoff": args.cutoff, "batch_size": args.batch_size,
        "splat_raster_mode": args.splat_raster_mode, "force_separate": force_separate,
        "startup": startup, "validation_warmup_s": {},
        "float_tolerance": {"rtol": 1e-10, "atol": 1e-12}, "rgb_absolute_tolerance": 1e-6,
        "benchmark_runs": args.benchmark_runs,
        "timing_scope": "CUDA-synchronized full frame: full-mesh visibility, kernel/face filtering, safe buffer reuse or original selected-only raster, unchanged heat-kernel evaluation and RGB transfer; excludes loading, footprint cache preparation, JIT, KNN/gating preparation, warmup, validation, reference ray rendering and PNG encoding",
    }
    signatures_before = tuple(_raster_kernel.signatures)

    def frame(backend, evaluation_timing=timing):
        # NumPy emits classification diagnostics; keep console I/O outside timing.
        with redirect_stdout(io.StringIO()):
            result = render_splat_frame(
                mesh, affected_sets, backend_args[backend], raster_geometry, trainer,
                gating_lookup, evaluation_timing, force_separate=force_separate,
            )
        if backend_args[backend].raster_backend != backend:
            raise RuntimeError("Backend comparison encountered a NumPy fallback: "
                               + getattr(backend_args[backend], "_raster_backend_fallback", "unknown"))
        return result

    def save_report():
        print("End-to-end NumPy / Numba comparison (same camera, mesh, KNN and gating state):")
        print(f"Raster/support parity: {report['raster_support_parity']}; "
              f"RGB parity: {report['rgb_parity']}; full correctness: {report['full_correctness']}")
        print(json.dumps(report, indent=2))
        if args.raster_backend_report is not None:
            args.raster_backend_report.parent.mkdir(parents=True, exist_ok=True)
            args.raster_backend_report.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Backend comparison report: {args.raster_backend_report}")

    warm = {"numpy": [], "numba": []}
    traces = {"numpy": [], "numba": []}
    trace_enabled = getattr(args, "trace_color_evaluation", False)
    if trace_enabled:
        from diagnose_color_evaluation import ColorEvaluationTrace, compare_traces, state_changes
        report["color_trace"] = {
            "state_audit": {},
            "scope": "actual intermediates at call/stage boundaries; persistent model/cache/gating content hashes; large spectral constants audited by pointer/version only; FAISS internal index storage is not inspected",
            "observation_caveat": "device clones and boundary state reads perturb allocation/scheduling; no per-intermediate CPU barrier; lack of divergence under observation does not establish repeatability",
            "faiss_distance_note": "FAISS search distances are observed but discarded by query_points; recomputed_distances feed heat weighting",
            "gpu_settings": {"allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "matmul_precision": torch.get_float32_matmul_precision(),
                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                             "torch_cuda_stream": int(torch.cuda.current_stream().cuda_stream)},
        }
    report["diagnostic_render_order"] = ["numpy_1", "numba_1", "numpy_2", "numba_2"]
    previous_state = None
    for repeat in range(2):
        for backend in ("numpy", "numba"):
            synchronize()
            observer = ColorEvaluationTrace(trainer, gating_lookup) if trace_enabled else timing
            if trace_enabled:
                before = observer.state()
            start = time.perf_counter()
            warm[backend].append(frame(backend, observer))
            synchronize()
            report["validation_warmup_s"][backend] = (
                report["validation_warmup_s"].get(backend, 0.0) + time.perf_counter() - start)
            if trace_enabled:
                traces[backend].append(observer.finish())
                after = observer.state()
                report["color_trace"]["state_audit"][f"{backend}_{repeat + 1}"] = {
                    "changed_during_render": state_changes(before, after),
                    "changed_since_previous_render": state_changes(previous_state, before) if previous_state is not None else [],
                    "before": before, "after": after,
                }
                previous_state = after
    pairs = {"numpy_vs_numpy": (warm["numpy"][0], warm["numpy"][1]),
             "numba_vs_numba": (warm["numba"][0], warm["numba"][1]),
             "numpy_vs_numba": (warm["numpy"][0], warm["numba"][0]),
             "numpy_vs_numba_second": (warm["numpy"][1], warm["numba"][1])}
    report["repeat_diagnostics"] = {name: diagnose_frames(a, b) for name, (a, b) in pairs.items()}
    if trace_enabled:
        report["color_trace"]["comparisons"] = {
            "numpy_vs_numpy": compare_traces(*traces["numpy"]),
            "numba_vs_numba": compare_traces(*traces["numba"]),
            "numpy_vs_numba": compare_traces(traces["numpy"][0], traces["numba"][0]),
        }
        # Instrumented repeats are excluded from performance measurements.
        report["validation_warmup_timing_note"] = "color tracing enabled; observer overhead included"
        del traces
    report["strict_assertion_failures"] = {}
    report["comparison_status"] = {}
    first_failure = None
    for name, (a, b) in pairs.items():
        try:
            metrics = compare_frames(a, b, strict_rgb=not allow_rgb_mismatch)
            rgb_status = metrics["rendered_rgb"]["parity"]
            report["comparison_status"][name] = {"raster_support": "PASSED", "rgb": rgb_status,
                                                 "full_correctness": rgb_status}
            if rgb_status == "FAILED":
                report["strict_assertion_failures"][name] = metrics["rendered_rgb"]["assertion_failure"]
            if name == "numpy_vs_numba":
                report["correctness"] = metrics
        except RGBParityError as error:
            report["comparison_status"][name] = {"raster_support": "PASSED", "rgb": "FAILED",
                                                 "full_correctness": "FAILED"}
            report["strict_assertion_failures"][name] = str(error)
            if first_failure is None:
                first_failure = error
        except AssertionError as error:
            report["comparison_status"][name] = {"raster_support": "FAILED", "rgb": "NOT_CHECKED",
                                                 "full_correctness": "FAILED"}
            report["strict_assertion_failures"][name] = str(error)
            if first_failure is None:
                first_failure = error
    statuses = list(report["comparison_status"].values())
    report["raster_support_parity"] = "PASSED" if all(s["raster_support"] == "PASSED" for s in statuses) else "FAILED"
    report["rgb_parity"] = ("FAILED" if any(s["rgb"] == "FAILED" for s in statuses)
                            else "PASSED" if all(s["rgb"] == "PASSED" for s in statuses) else "NOT_CHECKED")
    report["full_correctness"] = "PASSED" if all(s["full_correctness"] == "PASSED" for s in statuses) else "FAILED"
    if report["strict_assertion_failures"] and args.raster_backend_report is None:
        args.raster_backend_report = OUTPUT_DIR / "raster_backend_diagnostics.json"
    if first_failure is not None:
        save_report()  # Persist all repeat/input diagnostics before propagating failure.
        raise first_failure
    del pairs
    del warm
    samples = {"numpy": [], "numba": []}
    for run in range(args.benchmark_runs):
        # Alternate order to reduce systematic drift between the two paths.
        for backend in (("numpy", "numba") if run % 2 == 0 else ("numba", "numpy")):
            synchronize()
            start = time.perf_counter()
            result = frame(backend)
            synchronize()
            samples[backend].append(time.perf_counter() - start)
            del result
    if tuple(_raster_kernel.signatures) != signatures_before:
        raise RuntimeError("Unexpected Numba compilation during warmed frame measurements")
    report["warmed_full_frame"] = {
        backend: {"samples_s": values, "median_s": statistics.median(values),
                  "min_s": min(values), "max_s": max(values)}
        for backend, values in samples.items()
    }
    report["measured_numpy_over_numba_frame_ratio"] = (
        report["warmed_full_frame"]["numpy"]["median_s"]
        / report["warmed_full_frame"]["numba"]["median_s"])
    save_report()
    return report
