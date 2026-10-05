"""Correctness-first HKTex triangle-splat renderer.

Reuses the validated diagnostic helpers. Only the union of camera-visible
affected triangles produces splat pixels; no full-mesh silhouette/base-color
fill or rendering fallback is used. Full-mesh ray queries are limited to the
validated visibility selection and the separate reference renderer. Raster
visibility uses a full-mesh depth-only pass; splat colors use selected triangles.

Run in the HKTex GPU environment:
    python scripts/triangle_splat_renderer.py
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "outputs/uv-texture-fitting/test_connected@20260930-220225"
OUTPUT_DIR = REPO_ROOT / "outputs/triangle_splat"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--camera-position", type=float, nargs=3,
                        default=[2.5, -1.0, 1.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-look-at", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--fov-y", type=float, default=45.0)
    parser.add_argument("--visibility-mode", choices=("raster", "samples"), default="raster",
                        help="Visibility from pixel-center z-buffer winners (default) or seven surface samples")
    parser.add_argument("--compare-visibility", action="store_true",
                        help="Also run the alternate full-mesh visibility test and report differences; adds validation cost")
    parser.add_argument("--splat-raster-mode", choices=("reuse", "separate"), default="reuse",
                        help="Reuse safe raster-visibility surface buffers, or force the original separate splat raster")
    parser.add_argument("--validate-raster-reuse", action="store_true",
                        help="Compare reused surface buffers with a separate selected-triangle raster; adds validation cost")
    parser.add_argument("--benchmark-runs", type=int, default=1,
                        help="Steady-state evaluation repetitions on the same raster buffers and prepared KNN cache")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES if unset")
    parser.add_argument("--precompute-cache", action="store_true",
                        help="Save camera-independent affected faces for every kernel, then exit")
    parser.add_argument("--kernel-face-cache", type=Path,
                        help="Explicit footprint cache; otherwise use the default cutoff cache if present")
    args = parser.parse_args()
    if not math.isfinite(args.cutoff) or args.cutoff < 0:
        parser.error("--cutoff must be finite and nonnegative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.benchmark_runs < 1:
        parser.error("--benchmark-runs must be positive")
    if args.image_width < 1 or args.image_height < 1:
        parser.error("Image dimensions must be positive")
    if not math.isfinite(args.fov_y) or not 0.0 < args.fov_y < 180.0:
        parser.error("--fov-y must be finite and between 0 and 180 degrees")
    if not all(math.isfinite(value) for value in args.camera_position + args.camera_look_at):
        parser.error("Camera coordinates must be finite")
    if args.camera_position == args.camera_look_at:
        parser.error("Camera position and look-at target must differ")
    args.experiment = args.experiment.resolve()
    if args.kernel_face_cache is not None:
        args.kernel_face_cache = args.kernel_face_cache.resolve()
        if args.kernel_face_cache.suffix.lower() != ".npz":
            parser.error("--kernel-face-cache must end in .npz")
    return args


def default_cache_path(cutoff):
    return OUTPUT_DIR / f"kernel_face_cache_cutoff_{cutoff:g}.npz"


def cache_identity(trainer, checkpoint, config_path, vertices, faces):
    """Identify exact checkpoint, experiment config, and evaluated mesh geometry."""
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    geometry_digest = hashlib.sha256()
    for array in (vertices, faces):
        geometry_digest.update(str((array.shape, array.dtype.str)).encode())
        geometry_digest.update(array.tobytes(order="C"))
    return {
        "format_version": 1,
        "footprint_mode": "vertices+centroid",
        "total_kernels": int(trainer.model.N_sources),
        "total_faces": len(faces),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "mesh_geometry_sha256": geometry_digest.hexdigest(),
        "parsed_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def precompute_kernel_faces(trainer, density_model, args):
    """Reuse validated footprints in bounded groups, with no camera filtering."""
    import numpy as np
    from visualize_kernel_faces import multi_kernel_affected_faces

    affected_sets = []
    total_kernels = trainer.model.N_sources
    face_count = trainer.mesh.N_faces
    group_size = max(1, min(64, (64 * 1024 * 1024) // max(4 * face_count, 1)))
    for start in range(0, total_kernels, group_size):
        stop = min(start + group_size, total_kernels)
        mappings, _ = multi_kernel_affected_faces(
            trainer, density_model, list(range(start, stop)), args.cutoff, args.batch_size
        )
        affected_sets.extend(np.unique(faces).astype(np.int64) for faces in mappings)
        print(f"Precomputed kernel footprints: {stop} / {total_kernels}")
    return affected_sets


def save_kernel_face_cache(path, affected_sets, cutoff, identity):
    """Store ragged kernel-face mappings as offsets and IDs, without pickling."""
    import numpy as np

    offsets = np.zeros(len(affected_sets) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(faces) for faces in affected_sets], dtype=np.int64)
    face_ids = (np.concatenate(affected_sets).astype(np.int64) if affected_sets
                else np.empty(0, dtype=np.int64))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream, kernel_ids=np.arange(len(affected_sets), dtype=np.int64),
            face_offsets=offsets, affected_face_ids=face_ids,
            cutoff=np.float64(cutoff), metadata_json=np.array(json.dumps(identity, sort_keys=True)),
        )
    print(f"Saved camera-independent kernel face cache: {path}")


def load_kernel_face_cache(path, cutoff, identity):
    """Reject stale/incompatible caches rather than silently changing footprints."""
    import numpy as np

    with np.load(path, allow_pickle=False) as cache:
        stored_cutoff = float(cache["cutoff"].item())
        if stored_cutoff != cutoff:
            raise ValueError(f"Cache cutoff mismatch: {path} uses {stored_cutoff}, requested {cutoff}")
        metadata = json.loads(str(cache["metadata_json"].item()))
        for key in ("format_version", "footprint_mode", "total_kernels", "total_faces",
                    "checkpoint_sha256", "mesh_geometry_sha256", "parsed_config_sha256"):
            if metadata.get(key) != identity[key]:
                raise ValueError(f"Incompatible kernel-face cache ({key} differs): {path}; regenerate it")
        kernel_ids = cache["kernel_ids"]
        offsets = cache["face_offsets"]
        face_ids = cache["affected_face_ids"]
        if not (np.issubdtype(kernel_ids.dtype, np.integer)
                and np.issubdtype(offsets.dtype, np.integer)
                and np.issubdtype(face_ids.dtype, np.integer)):
            raise ValueError(f"Invalid kernel-face cache index types: {path}")
        if not np.array_equal(kernel_ids, np.arange(identity["total_kernels"])):
            raise ValueError(f"Cache must contain every model kernel in ID order: {path}")
        if (offsets.shape != (len(kernel_ids) + 1,) or face_ids.ndim != 1
                or offsets[0] != 0 or offsets[-1] != len(face_ids)
                or np.any(np.diff(offsets) < 0)):
            raise ValueError(f"Invalid kernel-face cache offsets: {path}")
        if np.any(face_ids < 0) or np.any(face_ids >= identity["total_faces"]):
            raise ValueError(f"Cache contains invalid mesh face IDs: {path}")
        affected_sets = [np.unique(face_ids[offsets[i]:offsets[i + 1]])
                         for i in range(len(kernel_ids))]
    print(f"Loaded camera-independent kernel face cache: {path}")
    return affected_sets


class RasterVisibilityTiming:
    """CPU wall times for the existing NumPy rasterizer; no nested double count."""

    def __init__(self):
        self.times = {}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - start

    def report(self, total):
        print("Raster visibility sub-step timings (CPU wall time):")
        for name, elapsed in self.times.items():
            print(f"  {name}: {elapsed:.6f} s")
        subtotal = sum(self.times.values())
        overhead = total - subtotal
        print("  CPU/GPU raster transfers: 0.000000 s (input/output buffers are NumPy; no transfers)")
        print(f"  Measured operation sub-step sum: {subtotal:.6f} s")
        print(f"  Python loops / batching / timer overhead (remainder): {overhead:.6f} s")
        print(f"  Accounted sum including remainder: {subtotal + overhead:.6f} s")
        print(f"  Existing raster visibility time: {total:.6f} s")
        print(f"  Operation sum close to total (within 5% or 5 ms): {abs(overhead) <= max(0.005, 0.05 * total)}")
        print("  Repeated CPU timing scopes add overhead; no raster algorithm/quality changes.")


def profiled_visibility_raster(vertices, faces, visible_face_ids, camera_position,
                               camera_look_at, image_width, image_height, fov_y, timing):
    """Original validated rasterizer operations, with CPU-only timing boundaries."""
    import numpy as np
    from visualize_kernel_faces import perspective_camera_basis, clip_camera_polygon

    with timing.stage("Camera setup / buffer initialization"):
        origin, basis = perspective_camera_basis(camera_position, camera_look_at)
        tan_y = math.tan(math.radians(fov_y) / 2.0)
        tan_x = tan_y * image_width / image_height
        scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
        near = max(scale * 1e-7, 1e-12)
        planes = (
            (np.array([0.0, 0.0, 1.0]), near),
            (np.array([1.0, 0.0, tan_x]), 0.0),
            (np.array([-1.0, 0.0, tan_x]), 0.0),
            (np.array([0.0, 1.0, tan_y]), 0.0),
            (np.array([0.0, -1.0, tan_y]), 0.0),
        )
        depth_buffer = np.full((image_height, image_width), np.inf)
        face_buffer = np.full((image_height, image_width), -1, dtype=np.int64)
        barycentric_buffer = np.full((image_height, image_width, 3), np.nan)
        rasterized_count = 0
    with timing.stage("Reusable pixel-center coordinate setup"):
        # Pixel centers and their arithmetic match the original arange + 0.5.
        # Slice these 1-D arrays instead of allocating two dense meshgrids for
        # every triangle. Broadcasting still yields full H-by-W barycentrics.
        pixel_x = (np.arange(image_width) + 0.5)[None, :]
        pixel_y = (np.arange(image_height) + 0.5)[:, None]
    with timing.stage("Vectorized trivial frustum classification"):
        plane_normals = np.stack([normal for normal, _ in planes])
        plane_offsets = np.array([offset for _, offset in planes])
        # Classification only: raster coordinates below still use the original
        # per-triangle transform, avoiding changes in floating-point evaluation.
        offsets_for_classification = vertices[faces[visible_face_ids]].astype(np.float64) - origin
        camera_for_classification = offsets_for_classification @ basis
        distances = camera_for_classification @ plane_normals.T - plane_offsets
        # Only take fast paths away from numerical plane boundaries. Include
        # transform and plane-dot magnitudes in a conservative roundoff guard;
        # uncertain/edge cases retain the exact original polygon clipping path.
        transform_magnitude = np.abs(offsets_for_classification) @ np.abs(basis)
        roundoff_guard = 64.0 * np.finfo(np.float64).eps * (
            transform_magnitude @ np.abs(plane_normals.T) + np.abs(plane_offsets)
        )
        trivially_accepted = np.all(distances > roundoff_guard, axis=(1, 2))
        trivially_rejected = np.any(np.all(distances < -roundoff_guard, axis=1), axis=1)
        requires_clipping = ~(trivially_accepted | trivially_rejected)
        initial_barycentrics = np.eye(3)
    print(f"Trivially accepted triangles: {int(trivially_accepted.sum())}")
    print(f"Trivially rejected triangles: {int(trivially_rejected.sum())}")
    print(f"Triangles requiring actual clipping: {int(requires_clipping.sum())}")
    for face_index, face_id in enumerate(visible_face_ids):
        if trivially_rejected[face_index]:
            continue
        with timing.stage("Vertex transformation"):
            camera_vertices = (vertices[faces[face_id]].astype(np.float64) - origin) @ basis
        with timing.stage("Triangle setup / frustum clipping"):
            polygon = np.column_stack((camera_vertices, initial_barycentrics))
            if requires_clipping[face_index]:
                for normal, offset in planes:
                    polygon = clip_camera_polygon(polygon, normal, offset)
                    if len(polygon) < 3:
                        break
        if len(polygon) < 3:
            continue
        face_covers_pixels = False
        for index in range(1, len(polygon) - 1):
            with timing.stage("Triangle fan setup / perspective projection"):
                triangle = polygon[[0, index, index + 1]]
                z = triangle[:, 2]
                screen = np.column_stack((
                    (triangle[:, 0] / (z * tan_x) + 1.0) * image_width / 2.0,
                    (1.0 - triangle[:, 1] / (z * tan_y)) * image_height / 2.0,
                ))
            with timing.stage("Screen bounding-box computation"):
                lower = np.maximum(np.ceil(screen.min(axis=0) - 0.5), [0, 0]).astype(int)
                upper = np.minimum(np.floor(screen.max(axis=0) - 0.5),
                                   [image_width - 1, image_height - 1]).astype(int)
                if np.any(lower > upper):
                    continue
            with timing.stage("Triangle barycentric setup"):
                a, b, c = screen
                denominator = ((b[1] - c[1]) * (a[0] - c[0])
                               + (c[0] - b[0]) * (a[1] - c[1]))
                if abs(denominator) <= 1e-12:
                    continue
            with timing.stage("Pixel coverage / screen barycentrics"):
                x = pixel_x[:, lower[0]:upper[0] + 1]
                y = pixel_y[lower[1]:upper[1] + 1, :]
                x_offset = x - c[0]
                y_offset = y - c[1]
                # Retain subtraction, multiplication, addition, and division
                # order; do not replace the existing edge/inclusion convention.
                w0 = ((b[1] - c[1]) * x_offset + (c[0] - b[0]) * y_offset) / denominator
                w1 = ((c[1] - a[1]) * x_offset + (a[0] - c[0]) * y_offset) / denominator
                w2 = 1.0 - w0 - w1
                inside = (w0 >= -1e-10) & (w1 >= -1e-10) & (w2 >= -1e-10)
                face_covers_pixels |= bool(inside.any())
            with timing.stage("Depth computation / z-buffer comparison"):
                reciprocal_z = w0 / z[0] + w1 / z[1] + w2 / z[2]
                depth = np.full_like(reciprocal_z, np.inf)
                np.divide(1.0, reciprocal_z, out=depth, where=inside & (reciprocal_z > 0.0))
                region = depth_buffer[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
                wins = inside & (depth < region)
            with timing.stage("Depth buffer writes"):
                region[wins] = depth[wins]
            with timing.stage("Face/barycentric buffer view setup"):
                face_region = face_buffer[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
                bary_region = barycentric_buffer[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
            with timing.stage("Winning face-ID writes"):
                face_region[wins] = face_id
            with timing.stage("Perspective-correct barycentrics / buffer writes"):
                if wins.any():
                    perspective_weights = np.column_stack((
                        w0[wins] / z[0], w1[wins] / z[1], w2[wins] / z[2],
                    )) / reciprocal_z[wins, None]
                    bary_region[wins] = perspective_weights @ triangle[:, 3:]
        rasterized_count += int(face_covers_pixels)
    with timing.stage("Final covered-pixel mask image"):
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        image[np.isfinite(depth_buffer)] = [0, 255, 0]
    return image, depth_buffer, rasterized_count, face_buffer, barycentric_buffer


def raster_visible_face_mask(mesh, args, retain_buffers=False, raster_geometry=None, profiling=None):
    """Full-mesh geometry/depth pass only; never supplies the splat color image."""
    import numpy as np
    import torch

    timing = profiling if profiling is not None else RasterVisibilityTiming()
    profile_start = time.perf_counter()
    with timing.stage("CUDA boundary synchronization"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    vertices, faces = (mesh.vertices, mesh.faces) if raster_geometry is None else raster_geometry
    _, depth, _, winning_faces, barycentrics = profiled_visibility_raster(
        vertices, faces, np.arange(len(faces)),
        args.camera_position, args.camera_look_at,
        args.image_width, args.image_height, args.fov_y, timing,
    )
    with timing.stage("Final visible-face extraction"):
        visible_face_mask = np.zeros(len(mesh.faces), dtype=bool)
        visible_face_mask[np.unique(winning_faces[winning_faces >= 0])] = True
    with timing.stage("CUDA boundary synchronization"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if profiling is None:
        timing.report(time.perf_counter() - profile_start)
    if retain_buffers:
        return visible_face_mask, (depth, winning_faces, barycentrics)
    return visible_face_mask


def sample_visible_face_mask(mesh, face_ids, args):
    """Preserve the validated seven-sample ray visibility semantics/tolerances."""
    import numpy as np
    from visualize_kernel_faces import visible_affected_faces

    camera_visible_faces = visible_affected_faces(
        mesh, face_ids, args.camera_position, args.camera_look_at, args.batch_size,
    )
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[camera_visible_faces] = True
    return mask


def filter_cached_kernel_visibility(mesh, affected_sets, args, raster_geometry=None):
    """Select kernels by a shared face mask from the requested visibility mode."""
    import numpy as np

    total_memberships = sum(len(affected) for affected in affected_sets)
    affected_union = (np.unique(np.concatenate(affected_sets)) if affected_sets
                      else np.empty(0, dtype=np.int64))
    determination_start = time.perf_counter()
    raster_mask = sample_mask = None
    raster_buffers = None
    raster_visibility_time = 0.0
    if args.visibility_mode == "raster":
        raster_profiling = RasterVisibilityTiming()
        raster_start = time.perf_counter()
        raster_mask, raster_buffers = raster_visible_face_mask(
            mesh, args, retain_buffers=True, raster_geometry=raster_geometry,
            profiling=raster_profiling,
        )
        raster_visibility_time = time.perf_counter() - raster_start
        raster_profiling.report(raster_visibility_time)
        face_visibility = raster_mask
    else:
        # Full-mesh samples are needed only for a full-mesh mode comparison.
        sample_faces = np.arange(len(mesh.faces)) if args.compare_visibility else affected_union
        sample_mask = sample_visible_face_mask(mesh, sample_faces, args)
        face_visibility = sample_mask
    determination_time = time.perf_counter() - determination_start
    kernel_ids, visible_sets = [], []
    for kernel_id, affected in enumerate(affected_sets):
        # Cache loading/precomputation already sorts and deduplicates each set.
        visible = affected[face_visibility[affected]]
        if len(visible):
            kernel_ids.append(kernel_id)
            visible_sets.append(visible)
    print(f"Visible kernels: {len(kernel_ids)} / {len(affected_sets)}")
    print(f"Visibility mode: {args.visibility_mode}")
    print(f"Mesh triangle count: {len(mesh.faces)}")
    print(f"Total affected-face memberships: {total_memberships}")
    print(f"Unique affected faces considered for visibility: {len(affected_union)}")
    print(f"Unique visible affected faces: {int(face_visibility[affected_union].sum())}")
    print(f"Visibility determination time ({args.visibility_mode}): {determination_time:.6f} s")
    if args.compare_visibility:
        comparison_start = time.perf_counter()
        if raster_mask is None:
            raster_mask = raster_visible_face_mask(mesh, args, raster_geometry=raster_geometry)
        if sample_mask is None:
            sample_mask = sample_visible_face_mask(mesh, np.arange(len(mesh.faces)), args)
        print(f"Faces visible in both: {int(np.count_nonzero(raster_mask & sample_mask))}")
        print(f"Faces only in raster visibility: {int(np.count_nonzero(raster_mask & ~sample_mask))}")
        print(f"Faces only in 7-sample visibility: {int(np.count_nonzero(sample_mask & ~raster_mask))}")
        print(f"Alternate visibility validation time: {time.perf_counter() - comparison_start:.6f} s")
        print("Visibility comparison is diagnostic only; masks are not merged or forced to match.")
    else:
        print("Visibility comparison skipped; use --compare-visibility to compare full-mesh face masks.")
    if raster_mask is not None:
        print(f"Raster-visible unique face count: {int(raster_mask.sum())}")
    else:
        print("Raster-visible unique face count: N/A (use raster mode or --compare-visibility)")
    return kernel_ids, visible_sets, raster_buffers, raster_visibility_time


def reuse_selected_raster_buffers(raster_buffers, visible_union, mesh_face_count):
    """Reuse only when full-mesh and selected-only z-buffer winners are equivalent.

    If any full-mesh winner is not selected, removing that face could expose a
    different selected surface. Preserve the original rendering semantics by
    falling back to a selected-only raster in that case, rather than masking out
    a pixel which the original splat pass might cover.
    """
    import numpy as np

    depth, face_ids, barycentrics = raster_buffers
    selected_faces = np.zeros(mesh_face_count, dtype=bool)
    selected_faces[visible_union] = True
    covered = face_ids >= 0
    if not np.all(selected_faces[face_ids[covered]]):
        return None
    # Every covered pixel is supported by the selected union. The full pass uses
    # the same geometry, triangle order, clipping, depth rule and barycentrics.
    # Uncovered face/depth/barycentric entries already carry the proper sentinels.
    mask_image = np.zeros((*face_ids.shape, 3), dtype=np.uint8)
    mask_image[covered] = [0, 255, 0]
    return mask_image, depth, face_ids, barycentrics


class KernelEvaluationTiming:
    """CUDA-synchronized exclusive wall times; nested stages are not double-counted."""

    def __init__(self, synchronize, enabled=True):
        self.synchronize = synchronize
        self.enabled = enabled
        self.times = {}
        self.stack = []
        self.last = None

    def boundary(self):
        self.synchronize()
        now = time.perf_counter()
        if self.stack and self.last is not None:
            name = self.stack[-1]
            self.times[name] = self.times.get(name, 0.0) + now - self.last
        self.last = now

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        self.boundary()
        self.times.setdefault(name, 0.0)
        self.stack.append(name)
        try:
            yield
        finally:
            self.boundary()
            self.stack.pop()

    @contextmanager
    def library_hooks(self, trainer):
        """Observe original library calls/scopes temporarily; preserve their results."""
        import torch

        scopes = {
            "query_points": "Outer KNN orchestration overhead",
            "diffuse_heat": "Heat-kernel weight/evaluation computation",
            "kernel_filter": "Trained threshold/sharpness filtering",
            "inner_knn_reduce": "Library candidate colors + inner top-k/gather/normalization",
        }
        original_record = torch.profiler.record_function
        hooks = []

        @contextmanager
        def timed_record(name, *args, **kwargs):
            if name in scopes:
                with self.stage(scopes[name]), original_record(name, *args, **kwargs):
                    yield
            else:
                with original_record(name, *args, **kwargs):
                    yield

        def wrap(owner, attribute, label):
            original = getattr(owner, attribute)
            had_instance_value = attribute in vars(owner)

            def timed_call(*args, **kwargs):
                with self.stage(label):
                    return original(*args, **kwargs)

            hooks.append((owner, attribute, original, had_instance_value))
            setattr(owner, attribute, timed_call)

        try:
            eig = trainer.eigalbo_interp
            wrap(eig.faiss_index, "search", "Outer KNN lookup")
            wrap(eig.faiss_index, "knn_distances", "Outer KNN candidate distances")
            wrap(eig.knn_gather, "gather_queries", "Candidate spectral parameter gathering")
            if eig.use_weighting:
                wrap(eig.heat_weighting, "compute", "Candidate distance weighting")
            torch.profiler.record_function = timed_record
            yield
        finally:
            torch.profiler.record_function = original_record
            for owner, attribute, original, had_instance_value in reversed(hooks):
                if had_instance_value:
                    setattr(owner, attribute, original)
                else:
                    delattr(owner, attribute)

    def report(self, total):
        print("Kernel evaluation/blending sub-step timings (exclusive, CUDA-synchronized):")
        for name, elapsed in self.times.items():
            print(f"  {name}: {elapsed:.6f} s")
        subtotal = sum(self.times.values())
        difference = total - subtotal
        close = abs(difference) <= max(0.005, 0.05 * total)
        print(f"  Sub-step sum: {subtotal:.6f} s")
        print(f"  Existing kernel evaluation/blending total: {total:.6f} s")
        print(f"  Unattributed setup/timer overhead: {difference:.6f} s")
        print(f"  Sub-step sum close to total (within 5% or 5 ms): {close}")
        print("  Library inner reduction is grouped; its nested operations are not counted twice.")
        print("  Synchronization/profiling overhead is included; these are instrumented timings.")


def prepare_selected_face_gating(trainer, face_buffer, kernel_ids, visible_sets):
    """Precompute exact selected-kernel patch membership for each covered face.

    The existing per-pixel patch mask is covered & isin(winning_face, visible_set).
    Thus membership is constant across all pixels on the same winning face. Use
    compact covered-face rows and global kernel columns; unselected columns stay
    false. This table is built/transferred once, then reused by warmup/benchmarks.
    """
    import numpy as np
    import torch

    covered_faces = np.unique(face_buffer[face_buffer >= 0])
    face_to_row = np.full(trainer.mesh.N_faces, -1, dtype=np.int64)
    face_to_row[covered_faces] = np.arange(len(covered_faces))
    support = np.zeros((len(covered_faces), trainer.model.N_sources), dtype=bool)
    for kernel_id, visible_faces in zip(kernel_ids, visible_sets):
        support[np.isin(covered_faces, visible_faces), kernel_id] = True
    device = trainer.mesh.faces.device
    face_rows_gpu = torch.as_tensor(face_to_row, device=device)
    support_gpu = torch.as_tensor(support, device=device)
    print(f"Selected kernel/patch lookup shape (covered faces, all model kernels): {support.shape}")
    print(f"Selected kernel/patch lookup storage: {support.nbytes + face_to_row.nbytes} bytes")
    print("Gating uses precomputed face membership; all unselected global kernel columns are false.")
    return face_rows_gpu, support_gpu


def profiled_selected_kernel_colors(trainer, face_buffer, barycentric_buffer,
                                    kernel_ids, batch_size, patch_masks, timing,
                                    manage_knn_cache=True, gating_lookup=None,
                                    covered_mask=None):
    """Same operations/order as the validated evaluator, with timing boundaries."""
    import numpy as np
    import torch

    with timing.stage("Raster pixel/mask preparation"):
        covered = face_buffer >= 0 if covered_mask is None else covered_mask
        face_ids = face_buffer[covered]
        barys = barycentric_buffer[covered]
        covered_indices = np.flatnonzero(covered)
        colors = np.zeros((*face_buffer.shape, trainer.model.out_dim), dtype=np.float32)
        masks = None if patch_masks is None else [mask[covered] for mask in patch_masks]
    with torch.no_grad(), timing.library_hooks(trainer):
        try:
            if manage_knn_cache:
                with timing.stage("KNN graph/cache preparation"):
                    trainer.prepare_knn(save_barycentric=False)
            for start in range(0, len(face_ids), batch_size):
                stop = min(start + batch_size, len(face_ids))
                with timing.stage("Surface points / barycentric positions"):
                    batch_faces = torch.as_tensor(face_ids[start:stop], device=trainer.mesh.faces.device)
                    batch_barys = torch.as_tensor(barys[start:stop], device=trainer.mesh.verts.device,
                                                 dtype=trainer.mesh.verts.dtype)
                    points = trainer.mesh.barycentric_to_cartesian(
                        batch_barys, trainer.mesh.get_face_vertices(batch_faces)
                    )
                with timing.stage("Point preparation overhead"):
                    points_info = trainer.model.prepare_points(
                        mesh=trainer.mesh, eigalbo_interp=trainer.eigalbo_interp,
                        face_ids=batch_faces, barys=None, pts=points,
                    )
                with timing.stage("Library diffusion/blend overhead and original clamp"):
                    _, _, topk_ids, topk_weights = trainer.model.diffuse_heat_kernels(
                        eigalbo_interp=trainer.eigalbo_interp, pts_info=points_info
                    )
                with timing.stage("Selected kernel/patch gating"):
                    ids = topk_ids.squeeze(-1).transpose(0, 1)
                    weights = topk_weights.squeeze(-1).transpose(0, 1)
                    if gating_lookup is not None:
                        face_rows, support = gating_lookup
                        rows = face_rows[batch_faces]
                        selected = support[rows[:, None], ids]
                    else:
                        # Preserve generic evaluator semantics for callers with
                        # arbitrary per-pixel masks or without a face lookup.
                        selected = torch.zeros_like(ids, dtype=torch.bool)
                        for index, kernel_id in enumerate(kernel_ids):
                            matches = ids == kernel_id
                            if masks is not None:
                                enabled = torch.as_tensor(masks[index][start:stop], device=ids.device)
                                matches &= enabled[:, None]
                            selected |= matches
                with timing.stage("Selected candidate colors / weighted accumulation"):
                    contribution_colors = weights.unsqueeze(-1) * trainer.model.kernel_colours[ids]
                    contribution_colors = torch.where(selected.unsqueeze(-1), contribution_colors, 0.0)
                    # Preserve the original expression's evaluation order:
                    # numerator sum, denominator sum/clamp, then division.
                    numerator = contribution_colors.sum(dim=1)
                with timing.stage("Normalization denominator computation"):
                    denominator = torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0)
                with timing.stage("Selected color normalization/blending"):
                    batch_colors = numerator / denominator
                with timing.stage("Final mean-color addition / clamp / postprocessing"):
                    batch_colors = (trainer.model._mean_colour + batch_colors).clamp(0.0, 1.0)
                    batch_colors = trainer.model(batch_colors)
                with timing.stage("RGB transfer / output-buffer write"):
                    colors.reshape(-1, trainer.model.out_dim)[covered_indices[start:stop]] = batch_colors.cpu().numpy()
        finally:
            if manage_knn_cache:
                with timing.stage("KNN cache reset"):
                    trainer.reset_knn()
    with timing.stage("Output validation"):
        if not np.isfinite(colors).all():
            raise ValueError("Nonfinite multi-kernel colors encountered")
    return colors


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import numpy as np
    import torch
    import trimesh

    # This loader selects the same Mitsuba variant and reconstructs the exact
    # experiment trainer/model as the diagnostic script.
    from interactive_density_knn import load_datamodule_and_trainer_only
    from hktex.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
    from hktex.modules.eigen_albo_knn import EigenAlboInterpolationKNN
    from hktex.modules.heat_kernel_density_knn import HeatKernelDensityKNN
    from hktex.utils import config_to_primitive
    from visualize_kernel_faces import (
        rasterize_visible_triangles,
        evaluate_selected_kernel_colors,
        reference_mesh_surface,
        save_raster_png,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    cfg, _, trainer = load_datamodule_and_trainer_only(
        argparse.Namespace(config=str(args.experiment / "configs/parsed.yaml")), extras=[]
    )
    if not isinstance(trainer.model, HeatKernelTextureKNN):
        raise TypeError("The experiment must construct HeatKernelTextureKNN")
    if not isinstance(trainer.eigalbo_interp, EigenAlboInterpolationKNN):
        raise TypeError("The experiment must construct EigenAlboInterpolationKNN")
    checkpoint = args.experiment / "ckpts" / cfg.optim.save_model_name
    trainer.model.load_torch(str(checkpoint))
    trainer.model.eval()
    if (trainer.model.cfg.knn_outer_k, trainer.model.cfg.knn_inner_k) != (50, 30):
        raise ValueError("Expected trained outer KNN=50 and inner top-k=30")
    if trainer.model.cfg.power_diffused_diracs != 1:
        raise ValueError("Footprint filter parity requires power_diffused_diracs=1")
    if trainer.model.out_dim != 3:
        raise ValueError("PNG rendering requires three output color channels")
    vertices = trainer.mesh.verts.detach().cpu().numpy()
    faces = trainer.mesh.faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Synchronize around timings so asynchronous CUDA work is included.
    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize(trainer.mesh.verts.device)

    def make_density_model():
        model = HeatKernelDensityKNN(config_to_primitive(trainer.model.cfg), trainer.mesh)
        model.load_torch(str(checkpoint))
        model.eval()
        return model

    cache_path = args.kernel_face_cache or default_cache_path(args.cutoff)
    config_path = args.experiment / "configs/parsed.yaml"
    if args.precompute_cache:
        density_model = make_density_model()
        synchronize()
        precompute_start = time.perf_counter()
        affected_sets = precompute_kernel_faces(trainer, density_model, args)
        identity = cache_identity(trainer, checkpoint, config_path, vertices, faces)
        save_kernel_face_cache(cache_path, affected_sets, args.cutoff, identity)
        synchronize()
        print(f"Footprint precomputation/cache save: {time.perf_counter() - precompute_start:.6f} s")
        print("Precompute mode does not perform visibility filtering or render any images.")
        return

    use_cache = cache_path.exists()
    if args.kernel_face_cache is not None and not use_cache:
        raise FileNotFoundError(f"Requested kernel face cache not found: {cache_path}; run --precompute-cache first")
    density_model = None
    if not use_cache:
        print(f"SLOW PATH: no default footprint cache at {cache_path}; recomputing kernel footprints.")
        print("Run --precompute-cache once to remove footprint discovery from subsequent renders.")
        density_model = make_density_model()

    synchronize()
    splat_start = time.perf_counter()
    selection_start = splat_start
    cache_load_time = slow_selection_time = 0.0
    if use_cache:
        cache_load_start = time.perf_counter()
        identity = cache_identity(trainer, checkpoint, config_path, vertices, faces)
        affected_sets = load_kernel_face_cache(cache_path, args.cutoff, identity)
        cache_load_time = time.perf_counter() - cache_load_start
    else:
        slow_start = time.perf_counter()
        affected_sets = precompute_kernel_faces(trainer, density_model, args)
        synchronize()
        slow_selection_time = time.perf_counter() - slow_start
    visibility_start = time.perf_counter()
    kernel_ids, visible_sets, visibility_buffers, raster_visibility_time = filter_cached_kernel_visibility(
        mesh, affected_sets, args, raster_geometry=(vertices, faces)
    )
    visibility_time = time.perf_counter() - visibility_start
    visible_union = (np.unique(np.concatenate(visible_sets)) if visible_sets
                     else np.empty(0, dtype=np.int64))
    synchronize()
    selection_time = time.perf_counter() - selection_start

    reuse_start = time.perf_counter()
    reused = None
    if args.splat_raster_mode == "reuse" and visibility_buffers is not None:
        reused = reuse_selected_raster_buffers(visibility_buffers, visible_union, len(faces))
    reuse_time = time.perf_counter() - reuse_start
    raster_time = reuse_validation_time = 0.0
    selected_buffers = None

    def separate_splat_raster():
        # This remains the original selected-only pass and never uses full-mesh
        # pixels to fill holes or contribute unsupported surfaces.
        mask, depth, _, face_ids, barys = rasterize_visible_triangles(
            vertices, faces, visible_union, args.camera_position, args.camera_look_at,
            args.image_width, args.image_height, args.fov_y,
        )
        return mask, depth, face_ids, barys

    if reused is not None and args.validate_raster_reuse:
        validation_start = time.perf_counter()
        separate = separate_splat_raster()
        covered = reused[2] >= 0
        matches = (
            np.array_equal(reused[0], separate[0])
            and np.array_equal(reused[2], separate[2])
            and np.array_equal(reused[1][covered], separate[1][covered])
            and np.array_equal(reused[3][covered], separate[3][covered])
        )
        print(f"Raster reuse matches separate mask/face/depth/barycentric buffers exactly: {matches}")
        reuse_validation_time = time.perf_counter() - validation_start
        if not matches:
            print("Raster reuse validation differed; using the separate selected-triangle buffers.")
            reused = None
            selected_buffers = separate
    if reused is not None:
        selected_buffers = reused
        print("Splat surface data reused from raster visibility: all winners are selected-supported triangles.")
    elif selected_buffers is None:
        raster_start = time.perf_counter()
        selected_buffers = separate_splat_raster()
        raster_time = time.perf_counter() - raster_start
        if args.splat_raster_mode == "reuse":
            print("Raster reuse unavailable/unsafe; using the original separate selected-triangle raster.")
        else:
            print("Separate selected-triangle raster forced for validation.")
    triangle_mask, _, splat_faces, splat_barys = selected_buffers
    pre_evaluation_time = time.perf_counter() - splat_start

    def evaluate_splat(timing):
        # The GPU face/kernel table already encodes every per-kernel patch mask.
        # Reuse frame coverage instead of rebuilding unused full-image masks.
        with timing.stage("Patch-mask construction"):
            covered = frame_covered
        colors = profiled_selected_kernel_colors(
            trainer, splat_faces, splat_barys, kernel_ids, args.batch_size,
            None, timing, manage_knn_cache=False, gating_lookup=gating_lookup,
            covered_mask=covered,
        )
        return colors, covered

    # Startup is outside every steady-state measurement. Warm every covered
    # batch through the identical evaluator/hooks, with sub-step timing disabled.
    # This exercises compilation for both full batches and the final short batch.
    synchronize()
    startup_start = time.perf_counter()
    benchmark_times = []
    try:
        with torch.no_grad():
            trainer.prepare_knn(save_barycentric=False)
        synchronize()
        knn_preparation_time = time.perf_counter() - startup_start
        gating_start = time.perf_counter()
        frame_covered = splat_faces >= 0
        gating_lookup = prepare_selected_face_gating(trainer, splat_faces, kernel_ids, visible_sets)
        synchronize()
        gating_preparation_time = time.perf_counter() - gating_start
        warmup_start = time.perf_counter()
        warmup_timing = KernelEvaluationTiming(synchronize, enabled=False)
        warmup_colors, _ = evaluate_splat(warmup_timing)
        synchronize()
        warmup_time = time.perf_counter() - warmup_start
        startup_time = time.perf_counter() - startup_start
        del warmup_colors

        for run in range(args.benchmark_runs):
            synchronize()
            evaluation_start = time.perf_counter()
            run_timing = KernelEvaluationTiming(synchronize)
            run_colors, run_covered = evaluate_splat(run_timing)
            synchronize()
            elapsed = time.perf_counter() - evaluation_start
            benchmark_times.append(elapsed)
            if run == 0:
                # Images/metrics and the detailed breakdown describe the first
                # measured run. Repetitions never rebuild or reset the KNN cache.
                splat_colors, splat_covered = run_colors, run_covered
                evaluation_timing = run_timing
                evaluation_time = elapsed
            else:
                del run_colors
    finally:
        trainer.reset_knn()
    # Preserve selection/raster costs, but exclude startup, extra benchmark runs,
    # and cache teardown from the steady-state single-render total.
    total_splat_time = pre_evaluation_time + evaluation_time

    # The reference is independent and cannot change the splat render buffers.
    # It retains the same selected color terms and full-model top-k denominator,
    # but has no triangle-patch gate and evaluates every first-hit mesh pixel.
    reference_start = time.perf_counter()
    reference_faces, reference_barys, _ = reference_mesh_surface(
        mesh, args.camera_position, args.camera_look_at,
        args.image_width, args.image_height, args.fov_y, args.batch_size,
    )
    reference_colors = evaluate_selected_kernel_colors(
        trainer, reference_faces, reference_barys, kernel_ids, args.batch_size
    )
    synchronize()
    reference_time = time.perf_counter() - reference_start

    reference_visible = reference_faces >= 0
    differences = np.abs(splat_colors - reference_colors)
    difference_image = np.zeros_like(differences)
    difference_image[reference_visible] = differences[reference_visible]
    for name, colors in (
        ("splat_render", splat_colors), ("reference", reference_colors),
        ("diff", difference_image),
    ):
        path = OUTPUT_DIR / f"{name}.png"
        save_raster_png(path, np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8))
        print(f"Exported: {path}")
    save_raster_png(OUTPUT_DIR / "triangle_mask.png", triangle_mask)
    print(f"Exported: {OUTPUT_DIR / 'triangle_mask.png'}")

    covered_count = int(splat_covered.sum())
    reference_count = int(reference_visible.sum())
    print(f"Checkpoint: {checkpoint}")
    print("Footprint mode: vertices+centroid")
    print("Splat pixels come exclusively from the selected triangle union; uncovered pixels stay black.")
    print("Blend: validated selected color terms, full-model outer KNN=50 / inner top-k=30,")
    print("       original normalization denominator and clamped/postprocessed mean color.")
    print(f"Total model kernels: {trainer.model.N_sources}")
    print(f"Number of camera-relevant kernels: {len(kernel_ids)}")
    print(f"Total unique visible splat triangles: {len(visible_union)}")
    print(f"Splat-covered pixel count: {covered_count}")
    print(f"Reference mesh-visible pixel count: {reference_count}")
    if reference_count:
        coverage = int((splat_covered & reference_visible).sum()) / reference_count
        errors = differences[reference_visible]
        print(f"Coverage percentage: {100.0 * coverage:.6f}%")
        print(f"MAE: {errors.mean():.9g}")
        print(f"RMSE: {np.sqrt(np.mean(errors ** 2)):.9g}")
        print(f"Max absolute error: {errors.max():.9g}")
    else:
        print("Coverage / MAE / RMSE / max absolute error: N/A (no reference mesh-visible pixels)")
    print("Metrics use unquantized RGB over reference mesh-visible pixels/channels; background is excluded.")
    print("Timings exclude checkpoint/model loading and PNG encoding; include CUDA synchronization.")
    print(f"Kernel/triangle selection: {selection_time:.6f} s")
    print(f"Cache load (including identity verification): {cache_load_time:.6f} s")
    print(f"Visibility filtering: {visibility_time:.6f} s")
    print(f"Raster visibility pass: {raster_visibility_time:.6f} s")
    print(f"Raster buffer reuse/filtering overhead: {reuse_time:.6f} s")
    print(f"Separate splat rasterization avoided: {reused is not None}")
    print(f"Remaining triangle splat rasterization: {raster_time:.6f} s")
    if args.validate_raster_reuse:
        print(f"Separate raster reuse validation: {reuse_validation_time:.6f} s")
        print("Total splat time includes requested raster-reuse validation overhead.")
    if args.compare_visibility:
        print("Visibility filtering and total splat timings include optional alternate-mode validation.")
    if not use_cache:
        print(f"Slow footprint recomputation: {slow_selection_time:.6f} s")
    print(f"Triangle splat rasterization: {raster_time:.6f} s")
    print(f"One-time KNN graph/cache preparation: {knn_preparation_time:.6f} s")
    print(f"One-time selected kernel/patch GPU lookup preparation: {gating_preparation_time:.6f} s")
    print("Patch coverage is prepared once; per-kernel support comes from the GPU face/kernel lookup.")
    print(f"Untimed evaluation warmup cost: {warmup_time:.6f} s")
    print(f"One-time KNN preparation + warmup cost: {startup_time:.6f} s")
    print("Startup total includes selected kernel/patch GPU lookup preparation.")
    print(f"Kernel evaluation/blending: {evaluation_time:.6f} s")
    print(f"Steady-state kernel evaluation/blending total (first run): {evaluation_time:.6f} s")
    evaluation_timing.report(evaluation_time)
    print(f"Steady-state benchmark runs: {args.benchmark_runs}")
    print(f"Steady-state evaluation times: {[round(value, 6) for value in benchmark_times]} s")
    print(f"Steady-state evaluation min: {min(benchmark_times):.6f} s")
    print(f"Steady-state evaluation median: {statistics.median(benchmark_times):.6f} s")
    print("Benchmarks reuse raster buffers and the warmed KNN cache; sub-step synchronization remains enabled.")
    print("Total splat time includes one steady-state evaluation, selection, and rasterization; excludes startup and extra runs.")
    print(f"Total splat render time: {total_splat_time:.6f} s")
    print(f"Total steady-state splat time: {total_splat_time:.6f} s")
    print(f"Total render time (splat path, excluding reference): {total_splat_time:.6f} s")
    print(f"Reference render time: {reference_time:.6f} s")


if __name__ == "__main__":
    main()
