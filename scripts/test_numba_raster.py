"""Standalone CPU visibility rasterizer and parity/benchmark harness.

Run on Grace, from the repository root in the HKTex environment (Numba required):
    python scripts/test_numba_raster.py \\
        --experiment outputs/uv-texture-fitting/cloudrunner_indexed@20261006-225053 \\
        --image-width 128 --image-height 128 \\
        --camera-position 0 3 0 --camera-look-at 0 0 0 --fov-y 45 \\
        --benchmark-runs 5 --warmup-runs 1 \\
        --report outputs/triangle_splat/numba_cloudrunner_128.json

CPU-only synthetic checks (no Torch, Mitsuba, GLB, or checkpoint loading):
    python scripts/test_numba_raster.py --synthetic-only

The mesh benchmark reconstructs trainer.mesh exactly as triangle_splat_renderer
and loads the trained checkpoint. It never rasterizes an independently loaded
GLB. All rasterization is serial CPU float64, without fastmath or disk JIT cache.
Compilation is explicitly timed before execution; steady-state timings include
camera setup, allocation, rasterization, and output creation, but exclude mesh
loading, comparison, visibility-mask extraction, and JIT compilation. This is a
visibility benchmark, not an end-to-end splat/color rendering benchmark.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from numba import njit, typeof

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "outputs/uv-texture-fitting/cloudrunner_indexed@20261006-225053"
# Support both direct execution and importing from the repository root.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from visualize_kernel_faces import perspective_camera_basis, rasterize_visible_triangles


@njit(cache=False, fastmath=False)
def _raster_kernel(vertices, faces, face_ids, origin, basis, width, height,
                   tan_x, tan_y, normals, offsets):
    depth = np.full((height, width), np.inf)
    winners = np.full((height, width), -1, dtype=np.int64)
    barys = np.full((height, width, 3), np.nan)
    count = 0
    # A convex triangle clipped by five planes has at most 3 + 5 vertices.
    polygon = np.empty((8, 6), dtype=np.float64)
    scratch = np.empty((8, 6), dtype=np.float64)
    for face_id in face_ids:
        relative = np.empty((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                relative[i, j] = vertices[faces[face_id, i], j] - origin[j]
        camera = relative @ basis
        for i in range(3):
            for j in range(3):
                polygon[i, j] = camera[i, j]
                polygon[i, j + 3] = 1.0 if i == j else 0.0
        size = 3
        for plane in range(5):
            new_size = 0
            previous = size - 1
            # Match the reference's three-element dot products and edge order.
            previous_distance = np.dot(polygon[previous, :3], normals[plane]) - offsets[plane]
            for current in range(size):
                distance = np.dot(polygon[current, :3], normals[plane]) - offsets[plane]
                if (distance >= 0.0) != (previous_distance >= 0.0):
                    fraction = previous_distance / (previous_distance - distance)
                    for j in range(6):
                        scratch[new_size, j] = (polygon[previous, j]
                            + fraction * (polygon[current, j] - polygon[previous, j]))
                    new_size += 1
                if distance >= 0.0:
                    for j in range(6):
                        scratch[new_size, j] = polygon[current, j]
                    new_size += 1
                previous = current
                previous_distance = distance
            polygon, scratch = scratch, polygon
            size = new_size
            if size < 3:
                break
        if size < 3:
            continue
        covers_pixels = False
        for fan in range(1, size - 1):
            triangle = np.empty((3, 6), dtype=np.float64)
            for i, index in enumerate((0, fan, fan + 1)):
                for j in range(6):
                    triangle[i, j] = polygon[index, j]
            screen = np.empty((3, 2), dtype=np.float64)
            for i in range(3):
                z = triangle[i, 2]
                screen[i, 0] = (triangle[i, 0] / (z * tan_x) + 1.0) * width / 2.0
                screen[i, 1] = (1.0 - triangle[i, 1] / (z * tan_y)) * height / 2.0
            xmin = max(int(math.ceil(min(screen[0, 0], screen[1, 0], screen[2, 0]) - 0.5)), 0)
            xmax = min(int(math.floor(max(screen[0, 0], screen[1, 0], screen[2, 0]) - 0.5)), width - 1)
            ymin = max(int(math.ceil(min(screen[0, 1], screen[1, 1], screen[2, 1]) - 0.5)), 0)
            ymax = min(int(math.floor(max(screen[0, 1], screen[1, 1], screen[2, 1]) - 0.5)), height - 1)
            if xmin > xmax or ymin > ymax:
                continue
            ax, ay = screen[0]
            bx, by = screen[1]
            cx, cy = screen[2]
            denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denominator) <= 1e-12:
                continue
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    px, py = x + 0.5, y + 0.5
                    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
                    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
                    w2 = 1.0 - w0 - w1
                    if w0 < -1e-10 or w1 < -1e-10 or w2 < -1e-10:
                        continue
                    covers_pixels = True  # Coverage count includes occluded faces.
                    reciprocal_z = w0 / triangle[0, 2] + w1 / triangle[1, 2] + w2 / triangle[2, 2]
                    if reciprocal_z <= 0.0:
                        continue
                    z = 1.0 / reciprocal_z
                    if z < depth[y, x]:  # Strict comparison: earlier face wins ties.
                        depth[y, x] = z
                        winners[y, x] = face_id
                        p0 = (w0 / triangle[0, 2]) / reciprocal_z
                        p1 = (w1 / triangle[1, 2]) / reciprocal_z
                        p2 = (w2 / triangle[2, 2]) / reciprocal_z
                        for j in range(3):
                            barys[y, x, j] = (p0 * triangle[0, j + 3]
                                + p1 * triangle[1, j + 3] + p2 * triangle[2, j + 3])
        count += int(covers_pixels)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if np.isfinite(depth[y, x]):
                image[y, x, 1] = 255
    return image, depth, count, winners, barys


def _kernel_args(vertices, faces, face_ids, camera_position, camera_look_at,
                 image_width, image_height, fov_y):
    """Prepare the same camera/frustum as the original, without altering face order."""
    # The validated code computes mesh scale in the input dtype (usually
    # float32), then casts only the triangles for camera arithmetic.
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    face_ids = np.ascontiguousarray(face_ids, dtype=np.int64)
    origin, basis = perspective_camera_basis(camera_position, camera_look_at)
    tan_y = math.tan(math.radians(fov_y) / 2.0)
    tan_x = tan_y * image_width / image_height
    near = max(scale * 1e-7, 1e-12)
    normals = np.array([[0., 0., 1.], [1., 0., tan_x], [-1., 0., tan_x],
                        [0., 1., tan_y], [0., -1., tan_y]])
    offsets = np.array([near, 0., 0., 0., 0.])
    return (vertices, faces, face_ids, origin, basis, image_width, image_height,
            tan_x, tan_y, normals, offsets)


def rasterize_numba(vertices, faces, visible_face_ids, camera_position,
                    camera_look_at, image_width, image_height, fov_y):
    """Drop-in five-output counterpart to rasterize_visible_triangles."""
    return _raster_kernel(*_kernel_args(
        vertices, faces, visible_face_ids, camera_position, camera_look_at,
        image_width, image_height, fov_y))


def compare_results(reference, candidate, label):
    """Require exact discrete results and tight float64 parity, including sentinels."""
    image, depth, count, faces, barys = reference
    cimage, cdepth, ccount, cfaces, cbarys = candidate
    covered = faces >= 0
    coverage_mismatches = int(np.count_nonzero(covered != (cfaces >= 0)))
    face_mismatches = int(np.count_nonzero(faces != cfaces))
    assert coverage_mismatches == 0, f"{label}: {coverage_mismatches} coverage mismatches"
    assert face_mismatches == 0, f"{label}: {face_mismatches} face ID mismatches"
    assert count == ccount, f"{label}: covered-face count {count} != {ccount}"
    np.testing.assert_array_equal(image, cimage, err_msg=label)
    np.testing.assert_array_equal(np.isinf(depth), np.isinf(cdepth), err_msg=label)
    np.testing.assert_array_equal(np.isnan(barys), np.isnan(cbarys), err_msg=label)
    np.testing.assert_allclose(cdepth, depth, rtol=1e-10, atol=1e-12, err_msg=label)
    np.testing.assert_allclose(cbarys, barys, rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=label)
    if covered.any():
        np.testing.assert_allclose(cbarys[covered].sum(axis=1), 1., rtol=0., atol=1e-10)
    return {
        "coverage_mismatches": coverage_mismatches, "face_id_mismatches": face_mismatches,
        "covered_pixels": int(covered.sum()), "faces_covering_pixels": int(count),
        "visible_faces": int(len(np.unique(faces[covered]))),
        "max_depth_absolute_error": float(np.max(np.abs(cdepth[covered] - depth[covered]))) if covered.any() else 0.,
        "max_barycentric_absolute_error": float(np.max(np.abs(cbarys[covered] - barys[covered]))) if covered.any() else 0.,
    }


def run_synthetic_tests():
    """Exercise clipping, winding, ties, empty selection, perspective, and cameras."""
    from triangle_splat_renderer import profiled_visibility_raster, RasterVisibilityTiming

    camera = ([0., 0., 0.], [0., 0., 1.], 32, 32, 90.)
    # Coordinates below are camera coordinates. The reference camera flips X.
    triangle = np.array([[-.8, -.8, 1.], [.8, -.8, 1.], [0., .8, 1.]])
    cases = []

    def add(name, triangles, ids=None, view=camera):
        vertices = np.asarray(triangles, dtype=np.float64).reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
        selected = np.arange(len(faces), dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        cases.append((name, (vertices, faces, selected, *view)))

    add("interior", [triangle])
    add("reverse winding", [triangle[::-1]])
    add("empty face selection", [triangle], [])
    add("behind camera", [triangle * [1, 1, -1]])
    add("degenerate line", [[[-.5, 0, 1], [0, 0, 1], [.5, 0, 1]]])
    add("degenerate point", [[[0, 0, 1]] * 3])
    add("subpixel", [triangle * [.00001, .00001, 1]])
    add("near plane crossing", [[[-.1, -.1, -1], [.8, -.8, 1], [0, .8, 1]]])
    add("near plane tiny depth", [[[-1e-8, -1e-8, 1e-8], [.8, -.8, 1], [0, .8, 1]]])
    add("entirely before near plane", [triangle * [1., 1., 1e-10]])
    for axis in (0, 1):
        for sign in (-1, 1):
            clipped = triangle.copy()
            clipped[0, axis] = sign * 4
            add(f"side clip {axis} {sign}", [clipped])
            outside = triangle.copy()
            outside[:, axis] += sign * 5
            add(f"outside plane {axis} {sign}", [outside])
    add("all side planes / fan", [[[-4, -4, 1], [4, -4, 1], [0, 4, 1]]])
    add("perspective varying depth", [triangle * np.array([[1], [2], [4]])])
    add("occlusion", [triangle * 2, triangle])
    add("equal-depth ordered tie", [triangle, triangle], [1, 0])
    add("duplicate selection", [triangle], [0, 0])
    add("shared edge square", [[[-1, -1, 1], [1, -1, 1], [1, 1, 1]],
                               [[-1, -1, 1], [1, 1, 1], [-1, 1, 1]]])
    # Put triangle edges exactly on pixel centers to catch half-pixel shifts.
    center_triangle = [[-.875, -.875, 1], [.875, -.875, 1], [-.875, .875, 1]]
    add("pixel-center boundaries", [center_triangle], view=(*camera[:2], 8, 8, 90.))
    add("non-square aspect", [triangle], view=(*camera[:2], 47, 19, 45.))
    add("pole camera target", [triangle], view=([0., 3., 0.], [0., 0., 0.], 128, 128, 45.))
    add("arbitrary camera", [triangle], view=([2.5, -1., 1.], [0., 0., 0.], 31, 23, 60.))
    # The trainer supplies float32 vertices and may supply non-contiguous views.
    add("float32 trained-style input", [triangle, triangle * 2])
    name, inputs = cases[-1]
    cases[-1] = (name, (inputs[0].astype(np.float32), *inputs[1:]))
    add("non-contiguous vertices", [triangle])
    name, inputs = cases[-1]
    strided = np.zeros((len(inputs[0]), 6))
    strided[:, ::2] = inputs[0]
    cases[-1] = (name, (strided[:, ::2], *inputs[1:]))
    rng = np.random.default_rng(91273)
    for index in range(12):
        triangles = rng.uniform(-3., 3., (40, 3, 3))
        triangles[:, :, 2] += 1.
        add(f"random clipped triangles {index}", triangles, rng.permutation(40))
    for name, args in cases:
        candidate = rasterize_numba(*args)
        reference = rasterize_visible_triangles(*args)
        compare_results(reference, candidate, name)
        with redirect_stdout(io.StringIO()):
            current = profiled_visibility_raster(*args, RasterVisibilityTiming(enabled=False))
        compare_results(current, candidate, name + " / current visibility")
        if name == "equal-depth ordered tie":
            assert set(candidate[3][candidate[3] >= 0]) == {1}
            assert candidate[2] == 2  # Count includes the fully occluded face.
        if name == "occlusion":
            assert set(candidate[3][candidate[3] >= 0]) == {1}
            np.testing.assert_allclose(candidate[1][candidate[3] >= 0], 1., atol=1e-12)
        # Independent check: returned original-face barycentrics reconstruct
        # points lying on the pixel-center camera rays at the reported depth.
        covered = candidate[3] >= 0
        if covered.any():
            vertices, faces, _, position, look_at, width, height, fov = args
            tri = vertices[faces[candidate[3][covered]]]
            points = np.einsum("ij,ijk->ik", candidate[4][covered], tri)
            origin, basis = perspective_camera_basis(position, look_at)
            coords = (points - origin) @ basis
            ty = math.tan(math.radians(fov) / 2.)
            tx = ty * width / height
            ys, xs = np.nonzero(covered)
            np.testing.assert_allclose(coords[:, 2], candidate[1][covered], rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose((coords[:, 0] / (coords[:, 2] * tx) + 1.) * width / 2., xs + .5, atol=1e-7, rtol=0.)
            np.testing.assert_allclose((1. - coords[:, 1] / (coords[:, 2] * ty)) * height / 2., ys + .5, atol=1e-7, rtol=0.)
    print(f"Synthetic parity against both NumPy baselines and surface reconstruction: {len(cases)} cases passed")
    return len(cases)


def test_synthetic_parity():
    """Pytest entry point; the CLI additionally isolates compilation timing."""
    run_synthetic_tests()


def load_trained_geometry(experiment, gpu):
    """Use the validated experiment loader and trainer mesh, preserving face IDs."""
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    from interactive_density_knn import load_datamodule_and_trainer_only

    config = experiment / "configs/parsed.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    cfg, _, trainer = load_datamodule_and_trainer_only(
        argparse.Namespace(config=str(config)), extras=[])
    checkpoint = experiment / "ckpts" / cfg.optim.save_model_name
    trainer.model.load_torch(str(checkpoint))
    trainer.model.eval()
    # Copy CPU arrays before releasing the trainer. No mesh processing/reindexing.
    vertices = trainer.mesh.verts.detach().cpu().numpy().copy()
    faces = trainer.mesh.faces.detach().cpu().numpy().copy()
    digest = hashlib.sha256()
    for array in (vertices, faces):
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes(order="C"))
    return vertices, faces, {
        "source": "trained experiment trainer.mesh", "experiment": str(experiment),
        "config": str(config), "checkpoint": str(checkpoint),
        "geometry_sha256": digest.hexdigest(), "vertices": len(vertices), "faces": len(faces),
    }


def time_calls(call, warmups, runs):
    for _ in range(warmups):
        call()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
        del result
    return {"samples_s": samples, "median_s": float(np.median(samples)),
            "min_s": min(samples), "max_s": max(samples)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--camera-position", type=float, nargs=3, default=[0., 3., 0.])
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--fov-y", type=float, default=45.)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--gpu", default="0", help="GPU used only for existing trainer initialization")
    parser.add_argument("--report", type=Path, help="Write correctness, environment, and raw timings as JSON")
    args = parser.parse_args()
    if min(args.image_width, args.image_height, args.benchmark_runs) < 1 or args.warmup_runs < 1:
        parser.error("Image dimensions, benchmark runs, and warmup runs must be positive")
    if not math.isfinite(args.fov_y) or not 0. < args.fov_y < 180.:
        parser.error("FOV must be finite and between 0 and 180 degrees")
    if not all(math.isfinite(x) for x in args.camera_position + args.camera_look_at):
        parser.error("Camera coordinates must be finite")
    if args.camera_position == args.camera_look_at:
        parser.error("Camera position and target must differ")
    args.experiment = args.experiment.resolve()
    if args.report is not None:
        args.report = args.report.resolve()
    return args


def main():
    args = parse_args()
    import numba
    report = {"environment": {"hostname": platform.node(), "machine": platform.machine(),
              "platform": platform.platform(), "python": platform.python_version(),
              "numpy": np.__version__, "numba": numba.__version__,
              "cpu_count": os.cpu_count(), "raster_threads": 1, "fastmath": False,
              "jit_disk_cache": False},
              "float_tolerance": {"rtol": 1e-10, "atol": 1e-12}}
    # Explicit compile() performs no raster execution. All wrapper calls have
    # this same signature regardless of array sizes, original dtypes, or camera.
    example = _kernel_args(np.zeros((3, 3)), np.array([[0, 1, 2]]), np.array([0]),
                           [0., 3., 0.], [0., 0., 0.], 128, 128, 45.)
    print("Compiling serial float64 raster kernel...", flush=True)
    start = time.perf_counter()
    _raster_kernel.compile(tuple(typeof(value) for value in example))
    report["jit_compilation_s"] = time.perf_counter() - start
    print(f"JIT compilation only (no raster execution, cache disabled): {report['jit_compilation_s']:.6f} s", flush=True)
    report["synthetic_cases_passed"] = run_synthetic_tests()
    if not args.synthetic_only:
        vertices, faces, provenance = load_trained_geometry(args.experiment, args.gpu)
        report["geometry"] = provenance
        report["camera"] = {"position": args.camera_position, "look_at": args.camera_look_at,
                            "width": args.image_width, "height": args.image_height, "fov_y": args.fov_y}
        inputs = (vertices, faces, np.arange(len(faces), dtype=np.int64),
                  args.camera_position, args.camera_look_at,
                  args.image_width, args.image_height, args.fov_y)
        print(f"Trained geometry: {len(vertices)} vertices, {len(faces)} faces; {provenance['checkpoint']}", flush=True)
        candidate = rasterize_numba(*inputs)
        original = rasterize_visible_triangles(*inputs)
        report["correctness_original"] = compare_results(original, candidate, "trained mesh / original")
        # Also check against the renderer's current visibility implementation,
        # with its timer scopes disabled. Importing it does not run main().
        from triangle_splat_renderer import profiled_visibility_raster, RasterVisibilityTiming

        def current_visibility():
            with redirect_stdout(io.StringIO()):
                return profiled_visibility_raster(*inputs, RasterVisibilityTiming(enabled=False))

        report["correctness_current_visibility"] = compare_results(
            current_visibility(), candidate, "trained mesh / current visibility")
        print("Trained mesh parity passed against original and current visibility:")
        print(json.dumps(report["correctness_original"], indent=2), flush=True)
        del candidate, original
        report["timing_scope"] = "CPU raster call including setup, allocations, outputs; excludes loading, JIT, comparisons, mask extraction, GPU work and color evaluation"
        report["benchmark_runs"] = args.benchmark_runs
        report["warmup_runs"] = args.warmup_runs
        report["timings"] = {}
        calls = {"original_numpy": lambda: rasterize_visible_triangles(*inputs),
                 "current_visibility_numpy": current_visibility,
                 "numba": lambda: rasterize_numba(*inputs)}
        for name, call in calls.items():
            report["timings"][name] = time_calls(call, args.warmup_runs, args.benchmark_runs)
            print(f"{name} warmed CPU median: {report['timings'][name]['median_s']:.6f} s "
                  f"({args.benchmark_runs} runs; samples {report['timings'][name]['samples_s']})", flush=True)
        assert len(_raster_kernel.signatures) == 1, "Unexpected compilation during benchmark"
        measured = report["timings"]["numba"]["median_s"]
        report["measured_median_ratios"] = {
            name: report["timings"][name]["median_s"] / measured
            for name in ("original_numpy", "current_visibility_numpy")}
        print("Measured baseline / Numba median ratios (visibility only): "
              + json.dumps(report["measured_median_ratios"]))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
