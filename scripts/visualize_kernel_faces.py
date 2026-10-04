"""Export a kernel's filtered outer-KNN footprint using vertices+centroid.

Example:
    python scripts/visualize_kernel_faces.py --kernel-id 34 --cutoff 0.01

Influence is measured before inner top-k selection and color blending. Kernels
absent from a sample's trained outer-KNN candidate set have influence zero.
Face selection uses the maximum influence at its three vertices and centroid.
"""

import argparse
import math
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "outputs/uv-texture-fitting/test_connected@20260930-220225"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    kernel_mode = parser.add_mutually_exclusive_group(required=True)
    kernel_mode.add_argument("--kernel-id", type=int)
    kernel_mode.add_argument("--kernel-ids", type=int, nargs="+", help="Render and compare a set of kernels")
    kernel_mode.add_argument(
        "--all-visible-kernels", action="store_true",
        help="Find and render all kernels with at least one camera-visible affected face",
    )
    parser.add_argument(
        "--debug-missing-coverage", action="store_true",
        help="In multi-kernel mode, report uncovered mesh pixels and highlight them on the reference",
    )
    parser.add_argument("--cutoff", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, help="Output .ply or .glb path; both formats are written with this stem")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES if unset")
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--fov-y", type=float, default=45.0, help="Vertical field of view in degrees")
    parser.add_argument(
        "--reference-comparison", action="store_true",
        help="Ray trace the full mesh and compare raw kernel influence with the rasterized patch",
    )
    parser.add_argument(
        "--camera-position", type=float, nargs=3, default=[0.0, 0.0, 3.7],
        metavar=("X", "Y", "Z"), help="Camera position in evaluated mesh coordinates",
    )
    parser.add_argument(
        "--camera-look-at", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"), help="Camera target (forward hemisphere; no FOV limit)",
    )
    args = parser.parse_args()
    requested_ids = [] if args.all_visible_kernels else (
        args.kernel_ids if args.kernel_ids is not None else [args.kernel_id]
    )
    if any(kernel_id < 0 for kernel_id in requested_ids):
        parser.error("Kernel IDs must be nonnegative")
    if len(set(requested_ids)) != len(requested_ids):
        parser.error("--kernel-ids must not contain duplicates")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.image_width < 1 or args.image_height < 1:
        parser.error("Image dimensions must be positive")
    if not math.isfinite(args.fov_y) or not 0.0 < args.fov_y < 180.0:
        parser.error("--fov-y must be finite and between 0 and 180 degrees")
    if not math.isfinite(args.cutoff) or args.cutoff < 0:
        parser.error("--cutoff must be finite and nonnegative")
    if not all(math.isfinite(value) for value in args.camera_position + args.camera_look_at):
        parser.error("Camera coordinates must be finite")
    if args.camera_position == args.camera_look_at:
        parser.error("Camera position and look-at target must differ")
    args.experiment = args.experiment.resolve()
    if args.output is None:
        stem = (f"kernel_{args.kernel_id}_faces" if args.kernel_id is not None
                else "multi_kernel_faces")
        args.output = REPO_ROOT / "outputs/kernel_face_debug" / f"{stem}.ply"
    args.output = args.output.resolve()
    if args.output.suffix.lower() not in {".ply", ".glb"}:
        parser.error("--output must end in .ply or .glb")
    return args


def visible_affected_faces(mesh, affected, camera_position, camera_look_at, batch_size):
    """Return affected IDs with any of seven samples visible to the camera.

    Test vertices, edge midpoints, and centroid against two-sided opaque geometry.
    Behind-camera samples and failed/no-hit rays do not establish visibility.
    """
    import numpy as np
    from trimesh.ray.ray_triangle import RayMeshIntersector

    camera = np.asarray(camera_position, dtype=np.float64)
    forward = np.asarray(camera_look_at, dtype=np.float64) - camera
    forward /= np.linalg.norm(forward)
    intersector = RayMeshIntersector(mesh)
    visible = []
    print("Visibility mode: vertices+edge-midpoints+centroid")
    sample_barys = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ])
    mesh_scale = max(float(np.linalg.norm(mesh.extents)), np.finfo(float).eps)
    for start in range(0, len(affected), batch_size):
        face_ids = affected[start:start + batch_size]
        triangles = mesh.vertices[mesh.faces[face_ids]]
        samples = np.einsum("sj,fjk->fsk", sample_barys, triangles).reshape(-1, 3)
        sample_face_ids = np.repeat(face_ids, len(sample_barys))
        # Keep at most --batch-size rays in each full-mesh intersection call.
        for sample_start in range(0, len(samples), batch_size):
            sample_stop = min(sample_start + batch_size, len(samples))
            visible.extend(visible_sample_faces(
                intersector, camera, forward, samples[sample_start:sample_stop],
                sample_face_ids[sample_start:sample_stop], mesh_scale,
            ))
    return np.asarray(sorted(set(visible)), dtype=np.int64)


def visible_sample_faces(intersector, camera, forward, samples, face_ids, mesh_scale):
    """Return face IDs for samples reached by the nearest opaque mesh hit."""
    import numpy as np

    offsets = samples - camera
    distances = np.linalg.norm(offsets, axis=1)
    eligible = (distances > 0.0) & (offsets @ forward > 0.0)
    if not eligible.any():
        return []
    target_ids = face_ids[eligible]
    target_distances = distances[eligible]
    directions = offsets[eligible] / target_distances[:, None]
    origins = np.broadcast_to(camera, directions.shape).copy()
    try:
        locations, ray_ids, _ = intersector.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Trimesh occlusion testing requires its ray dependencies (including rtree). "
            "Install them in your HKTex environment."
        ) from error
    hit_distances = np.linalg.norm(locations - camera, axis=1)
    tolerance = 1e-6 * mesh_scale + 1e-7 * target_distances[ray_ids]
    # At a shared vertex/edge, Trimesh may return an adjacent face as the first
    # hit. The sample is still visible if that hit reaches the sample distance;
    # requiring face identity would incorrectly reject such boundary samples.
    is_visible = np.abs(hit_distances - target_distances[ray_ids]) <= tolerance
    return target_ids[ray_ids[is_visible]].tolist()


def perspective_camera_basis(camera_position, camera_look_at):
    """Return camera origin and a world-to-camera basis with positive forward Z."""
    import numpy as np

    origin = np.asarray(camera_position, dtype=np.float64)
    forward = np.asarray(camera_look_at, dtype=np.float64) - origin
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(forward @ up_hint) > 0.99:
        up_hint = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return origin, np.stack((right, up, forward), axis=1)


def clip_camera_polygon(polygon, plane_normal, plane_offset=0.0):
    """Clip camera XYZ and optional attributes, interpolating entire vertex rows."""
    import numpy as np

    clipped = []
    if not len(polygon):
        return np.empty((0, polygon.shape[1]), dtype=np.float64)
    previous = polygon[-1]
    previous_distance = previous[:3] @ plane_normal - plane_offset
    for current in polygon:
        distance = current[:3] @ plane_normal - plane_offset
        if (distance >= 0.0) != (previous_distance >= 0.0):
            fraction = previous_distance / (previous_distance - distance)
            clipped.append(previous + fraction * (current - previous))
        if distance >= 0.0:
            clipped.append(current)
        previous, previous_distance = current, distance
    return np.asarray(clipped, dtype=np.float64).reshape(-1, polygon.shape[1])


def rasterize_visible_triangles(vertices, faces, visible_face_ids, camera_position,
                                camera_look_at, image_width, image_height, fov_y):
    """Fill projected interiors at pixel centers using barycentrics and a Z buffer.

    Only the already-selected visible faces participate. Depth is interpolated
    perspectively (reciprocal Z), not linearly in screen space. Return RGB image,
    depth buffer, count of original faces covering at least one pixel, winning
    original mesh face IDs, and perspective-correct original-face barycentrics.
    """
    import numpy as np

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
    for face_id in visible_face_ids:
        camera_vertices = (vertices[faces[face_id]].astype(np.float64) - origin) @ basis
        polygon = np.column_stack((camera_vertices, np.eye(3)))
        for normal, offset in planes:
            polygon = clip_camera_polygon(polygon, normal, offset)
            if len(polygon) < 3:
                break
        if len(polygon) < 3:
            continue
        face_covers_pixels = False
        # Clipping can turn a triangle into a polygon; fill a triangle fan.
        for index in range(1, len(polygon) - 1):
            triangle = polygon[[0, index, index + 1]]
            z = triangle[:, 2]
            screen = np.column_stack((
                (triangle[:, 0] / (z * tan_x) + 1.0) * image_width / 2.0,
                (1.0 - triangle[:, 1] / (z * tan_y)) * image_height / 2.0,
            ))
            lower = np.maximum(np.ceil(screen.min(axis=0) - 0.5), [0, 0]).astype(int)
            upper = np.minimum(np.floor(screen.max(axis=0) - 0.5),
                               [image_width - 1, image_height - 1]).astype(int)
            if np.any(lower > upper):
                continue
            a, b, c = screen
            denominator = ((b[1] - c[1]) * (a[0] - c[0])
                           + (c[0] - b[0]) * (a[1] - c[1]))
            if abs(denominator) <= 1e-12:
                continue
            x, y = np.meshgrid(np.arange(lower[0], upper[0] + 1) + 0.5,
                               np.arange(lower[1], upper[1] + 1) + 0.5)
            w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
            w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-10) & (w1 >= -1e-10) & (w2 >= -1e-10)
            face_covers_pixels |= bool(inside.any())
            reciprocal_z = w0 / z[0] + w1 / z[1] + w2 / z[2]
            depth = np.full_like(reciprocal_z, np.inf)
            np.divide(1.0, reciprocal_z, out=depth, where=inside & (reciprocal_z > 0.0))
            region = depth_buffer[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
            wins = inside & (depth < region)
            region[wins] = depth[wins]
            face_region = face_buffer[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
            bary_region = barycentric_buffer[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
            face_region[wins] = face_id
            if wins.any():
                # Clipped vertex attributes express barycentrics in the original
                # face. Perspective-correct interpolation preserves that mapping.
                perspective_weights = np.column_stack((
                    w0[wins] / z[0], w1[wins] / z[1], w2[wins] / z[2],
                )) / reciprocal_z[wins, None]
                bary_region[wins] = perspective_weights @ triangle[:, 3:]
        rasterized_count += int(face_covers_pixels)
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[np.isfinite(depth_buffer)] = [0, 255, 0]
    return image, depth_buffer, rasterized_count, face_buffer, barycentric_buffer


def evaluate_raster_kernel(trainer, density_model, face_buffer, barycentric_buffer,
                           kernel_id, batch_size):
    """Evaluate covered surface points with the trained outer-KNN filter path."""
    import numpy as np
    import torch

    covered = face_buffer >= 0
    face_ids = face_buffer[covered]
    barycentrics = barycentric_buffer[covered]
    influence = np.zeros(len(face_ids), dtype=np.float32)
    if not len(face_ids):
        return covered, influence
    with torch.no_grad():
        try:
            trainer.prepare_knn(save_barycentric=False)
            for start in range(0, len(face_ids), batch_size):
                stop = min(start + batch_size, len(face_ids))
                batch_faces = torch.as_tensor(
                    face_ids[start:stop], device=trainer.mesh.faces.device,
                    dtype=torch.long,
                )
                batch_barys = torch.as_tensor(
                    barycentrics[start:stop], device=trainer.mesh.verts.device,
                    dtype=trainer.mesh.verts.dtype,
                )
                surface_points = trainer.mesh.barycentric_to_cartesian(
                    batch_barys, trainer.mesh.get_face_vertices(batch_faces)
                )
                points_info = trainer.model.prepare_points(
                    mesh=trainer.mesh, eigalbo_interp=trainer.eigalbo_interp,
                    face_ids=batch_faces, barys=None, pts=surface_points,
                )
                filtered, global_indices = density_model.filtered_kernel_weights(
                    points_info, trainer.eigalbo_interp
                )
                matches = global_indices == kernel_id
                influence[start:stop] = torch.where(matches, filtered, 0.0).sum(dim=1).cpu().numpy()
        finally:
            trainer.reset_knn()
    if not np.isfinite(influence).all():
        raise ValueError("Nonfinite rasterized pixel influence encountered")
    return covered, influence


def reference_mesh_surface(mesh, camera_position, camera_look_at,
                           image_width, image_height, fov_y, batch_size):
    """Ray trace pixel centers against the full mesh, retaining the first hit."""
    import numpy as np
    from trimesh.ray.ray_triangle import RayMeshIntersector
    from trimesh.triangles import points_to_barycentric

    origin, basis = perspective_camera_basis(camera_position, camera_look_at)
    tan_y = math.tan(math.radians(fov_y) / 2.0)
    tan_x = tan_y * image_width / image_height
    pixel_count = image_width * image_height
    face_ids = np.full(pixel_count, -1, dtype=np.int64)
    barycentrics = np.full((pixel_count, 3), np.nan)
    depths = np.full(pixel_count, np.inf)
    intersector = RayMeshIntersector(mesh)
    for start in range(0, pixel_count, batch_size):
        stop = min(start + batch_size, pixel_count)
        pixels = np.arange(start, stop)
        x = ((pixels % image_width + 0.5) * 2.0 / image_width - 1.0) * tan_x
        y = (1.0 - (pixels // image_width + 0.5) * 2.0 / image_height) * tan_y
        directions = np.column_stack((x, y, np.ones(len(pixels)))) @ basis.T
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origins = np.broadcast_to(origin, directions.shape).copy()
        try:
            locations, ray_ids, hit_faces = intersector.intersects_location(
                ray_origins=origins, ray_directions=directions, multiple_hits=False,
            )
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Reference comparison requires Trimesh ray dependencies (including rtree)."
            ) from error
        if not len(ray_ids):
            continue
        destination = start + ray_ids
        face_ids[destination] = hit_faces
        barycentrics[destination] = points_to_barycentric(
            mesh.vertices[mesh.faces[hit_faces]], locations
        )
        depths[destination] = (locations - origin) @ basis[:, 2]
    return (face_ids.reshape(image_height, image_width),
            barycentrics.reshape(image_height, image_width, 3),
            depths.reshape(image_height, image_width))


def compare_reference_kernel(trainer, density_model, mesh, args, raster_faces,
                             raster_depth, covered, pixel_influence):
    """Compare unnormalized influence only where camera rays see the mesh."""
    import numpy as np

    reference_faces, reference_barys, reference_depth = reference_mesh_surface(
        mesh, args.camera_position, args.camera_look_at,
        args.image_width, args.image_height, args.fov_y, args.batch_size,
    )
    mesh_visible, reference_values = evaluate_raster_kernel(
        trainer, density_model, reference_faces, reference_barys,
        args.kernel_id, args.batch_size,
    )
    patch_values = np.zeros(reference_faces.shape, dtype=np.float64)
    patch_values[covered] = pixel_influence
    errors = np.abs(patch_values[mesh_visible] - reference_values)
    reference_image = np.zeros(reference_faces.shape, dtype=np.uint8)
    difference_image = np.zeros(reference_faces.shape, dtype=np.uint8)
    print(f"Reference mesh-visible pixels: {len(reference_values)}")
    print("Comparison uses raw filtered influence; background pixels are excluded.")
    if len(reference_values):
        reference_max = float(reference_values.max())
        error_max = float(errors.max())
        if reference_max > 0.0:
            reference_image[mesh_visible] = np.rint(
                np.clip(reference_values / reference_max, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        if error_max > 0.0:
            difference_image[mesh_visible] = np.rint(errors / error_max * 255.0).astype(np.uint8)
        print(f"Mean absolute error: {errors.mean():.9g}")
        print(f"Max absolute error: {error_max:.9g}")
        print(f"RMSE: {np.sqrt(np.mean(errors ** 2)):.9g}")
        # Do not count a selected face hidden behind the actual first-hit face.
        # The face-ID check also avoids accepting nearby but different surfaces.
        mesh_scale = max(float(np.linalg.norm(mesh.extents)), np.finfo(float).eps)
        patch_hit = raster_faces[mesh_visible] == reference_faces[mesh_visible]
        patch_hit &= np.isclose(
            raster_depth[mesh_visible], reference_depth[mesh_visible],
            rtol=1e-6, atol=1e-6 * mesh_scale,
        )
        total_energy = float(reference_values.sum(dtype=np.float64))
        if total_energy > 0.0:
            captured = float(reference_values[patch_hit].sum(dtype=np.float64))
            print(f"Reference kernel energy captured by patch: {100.0 * captured / total_energy:.6f}%")
        else:
            print("Reference kernel energy captured by patch: N/A (zero reference energy)")
    else:
        print("Mean absolute error: N/A (no mesh-visible pixels)")
        print("Max absolute error: N/A (no mesh-visible pixels)")
        print("RMSE: N/A (no mesh-visible pixels)")
        print("Reference kernel energy captured by patch: N/A (no mesh-visible pixels)")
    output_dir = REPO_ROOT / "outputs/kernel_face_debug"
    reference_path = output_dir / f"kernel_{args.kernel_id}_reference.png"
    difference_path = output_dir / f"kernel_{args.kernel_id}_diff.png"
    save_raster_png(reference_path, reference_image)
    save_raster_png(difference_path, difference_image)
    print(f"Exported reference PNG (zero to reference maximum): {reference_path}")
    print(f"Exported difference PNG (zero to maximum absolute error): {difference_path}")


def save_raster_png(path, image):
    """Write an RGB or grayscale PNG without an image-library dependency."""
    import struct
    import zlib

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    height, width = image.shape[:2]
    color_type = 0 if len(image.shape) == 2 else 2
    rows = b"".join(b"\x00" + row.tobytes() for row in image)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def multi_kernel_affected_faces(trainer, density_model, kernel_ids, cutoff, batch_size):
    """Use the existing vertices+centroid max rule for each requested kernel."""
    import numpy as np
    import torch

    mesh = trainer.mesh
    influences = np.zeros((len(kernel_ids), mesh.N_faces), dtype=np.float32)
    sample_barys = torch.tensor(
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1./3., 1./3., 1./3.]],
        device=mesh.verts.device, dtype=mesh.verts.dtype,
    )
    with torch.no_grad():
        try:
            trainer.prepare_knn(save_barycentric=False)
            for start in range(0, mesh.N_faces, batch_size):
                stop = min(start + batch_size, mesh.N_faces)
                face_ids = torch.arange(start, stop, device=mesh.faces.device)
                maxima = torch.full((len(kernel_ids), stop - start), -torch.inf,
                                    device=mesh.verts.device, dtype=mesh.verts.dtype)
                for sample in sample_barys:
                    points_info = trainer.model.prepare_points(
                        mesh=mesh, eigalbo_interp=trainer.eigalbo_interp,
                        face_ids=face_ids, barys=sample.expand(stop - start, -1).contiguous(),
                        pts=None,
                    )
                    filtered, global_indices = density_model.filtered_kernel_weights(
                        points_info, trainer.eigalbo_interp
                    )
                    for index, kernel_id in enumerate(kernel_ids):
                        values = torch.where(global_indices == kernel_id, filtered, 0.0).sum(dim=1)
                        maxima[index] = torch.maximum(maxima[index], values)
                influences[:, start:stop] = maxima.cpu().numpy()
        finally:
            trainer.reset_knn()
    if not np.isfinite(influences).all():
        raise ValueError("Nonfinite multi-kernel face influence encountered")
    return [np.flatnonzero(values > cutoff) for values in influences], influences


def evaluate_selected_kernel_colors(trainer, face_buffer, barycentric_buffer,
                                    kernel_ids, batch_size, patch_masks=None):
    """Keep full-model top-k and denominator, retaining selected color terms.

    This is the model's blend with nonselected kernel colors zeroed. Optional
    per-kernel masks also zero color terms outside that kernel's triangle patch.
    Weights, top-k competition, mean color, clamping, and postprocessing remain
    those of HeatKernelTextureKNN.diffuse_heat_kernels/forward.
    """
    import numpy as np
    import torch

    covered = face_buffer >= 0
    face_ids = face_buffer[covered]
    barys = barycentric_buffer[covered]
    covered_indices = np.flatnonzero(covered)
    colors = np.zeros((*face_buffer.shape, trainer.model.out_dim), dtype=np.float32)
    masks = None if patch_masks is None else [mask[covered] for mask in patch_masks]
    with torch.no_grad():
        try:
            trainer.prepare_knn(save_barycentric=False)
            for start in range(0, len(face_ids), batch_size):
                stop = min(start + batch_size, len(face_ids))
                batch_faces = torch.as_tensor(face_ids[start:stop], device=trainer.mesh.faces.device)
                batch_barys = torch.as_tensor(barys[start:stop], device=trainer.mesh.verts.device,
                                             dtype=trainer.mesh.verts.dtype)
                points = trainer.mesh.barycentric_to_cartesian(
                    batch_barys, trainer.mesh.get_face_vertices(batch_faces)
                )
                points_info = trainer.model.prepare_points(
                    mesh=trainer.mesh, eigalbo_interp=trainer.eigalbo_interp,
                    face_ids=batch_faces, barys=None, pts=points,
                )
                # The library performs the original outer-KNN heat/filter and
                # inner-top-k selection; returned weights are unnormalized.
                _, _, topk_ids, topk_weights = trainer.model.diffuse_heat_kernels(
                    eigalbo_interp=trainer.eigalbo_interp, pts_info=points_info
                )
                ids = topk_ids.squeeze(-1).transpose(0, 1)
                weights = topk_weights.squeeze(-1).transpose(0, 1)
                selected = torch.zeros_like(ids, dtype=torch.bool)
                for index, kernel_id in enumerate(kernel_ids):
                    matches = ids == kernel_id
                    if masks is not None:
                        enabled = torch.as_tensor(masks[index][start:stop], device=ids.device)
                        matches &= enabled[:, None]
                    selected |= matches
                contribution_colors = weights.unsqueeze(-1) * trainer.model.kernel_colours[ids]
                contribution_colors = torch.where(selected.unsqueeze(-1), contribution_colors, 0.0)
                # Exact equation from HeatKernelTextureKNN: all top-k weights
                # participate in normalization, even when their colors are zero.
                batch_colors = contribution_colors.sum(dim=1) / torch.clamp(
                    weights.sum(dim=1, keepdim=True), min=1.0
                )
                batch_colors = (trainer.model._mean_colour + batch_colors).clamp(0.0, 1.0)
                batch_colors = trainer.model(batch_colors)
                colors.reshape(-1, trainer.model.out_dim)[covered_indices[start:stop]] = batch_colors.cpu().numpy()
        finally:
            trainer.reset_knn()
    if not np.isfinite(colors).all():
        raise ValueError("Nonfinite multi-kernel colors encountered")
    return colors


def find_visible_kernels(trainer, density_model, mesh, args):
    """Scan all trained sources, reusing footprint and camera visibility tests."""
    import numpy as np

    total_kernels = trainer.model.N_sources
    # Visibility is a property of a face and this camera, independent of kernel.
    # Use the same full-mesh seven-sample occlusion test once for all face IDs.
    camera_visible_faces = visible_affected_faces(
        mesh, np.arange(len(mesh.faces)), args.camera_position,
        args.camera_look_at, args.batch_size,
    )
    visible_kernel_ids, affected_sets, visible_sets, maximum_influences = [], [], [], []
    source_faces = trainer.model._kernel_face_ids.detach().cpu().numpy()
    # Bound the dense kernel-by-face influence array used by the existing helper.
    # Limit each group to roughly 64 MiB of host influence storage, up to 64 IDs.
    group_size = max(1, min(64, (64 * 1024 * 1024) // max(4 * len(mesh.faces), 1)))
    for start in range(0, total_kernels, group_size):
        kernel_ids = list(range(start, min(start + group_size, total_kernels)))
        local_faces, influences = multi_kernel_affected_faces(
            trainer, density_model, kernel_ids, args.cutoff, args.batch_size
        )
        for index, (kernel_id, affected) in enumerate(zip(kernel_ids, local_faces)):
            # Both arrays contain original global face IDs. Intersection preserves
            # exactly the per-kernel visibility test while eliminating duplicates.
            visible = np.intersect1d(affected, camera_visible_faces)
            print(f"Kernel {kernel_id}: affected={len(affected)}, visible={len(visible)}, "
                  f"source_face={int(source_faces[kernel_id])}")
            if not len(visible):
                continue
            visible_kernel_ids.append(kernel_id)
            affected_sets.append(affected)
            visible_sets.append(visible)
            maximum_influences.append(float(influences[index].max()))
    print(f"Visible kernels: {len(visible_kernel_ids)} / {total_kernels}")
    print(f"Total model kernels: {total_kernels}")
    print(f"Number of camera-relevant kernels: {len(visible_kernel_ids)}")
    print(f"visible_kernel_ids: {visible_kernel_ids}")
    return visible_kernel_ids, affected_sets, visible_sets, maximum_influences


def debug_missing_coverage(trainer, args, reference_faces, patch_covered,
                           reference_colors, kernel_ids, affected_sets, visible_union):
    """Report missing coverage without changing any selection or render buffers."""
    import numpy as np

    reference_visible = reference_faces >= 0
    missing_mask = reference_visible & ~patch_covered
    rows, columns = np.nonzero(missing_mask)
    print(f"Missing coverage pixel count: {len(rows)}")
    print("Missing pixel coordinates (x, y; origin at top-left): "
          f"{list(zip(columns.tolist(), rows.tolist()))}")
    selected_ids = set(kernel_ids)
    affected_union = set(int(face) for faces in affected_sets for face in faces)
    visible_face_union = set(int(face) for face in visible_union)
    source_face_ids = trainer.model._kernel_face_ids.detach().cpu().numpy()
    print("_kernel_face_ids associates kernels with their source face, not their full affected footprint.")
    for y, x in zip(rows, columns):
        face_id = int(reference_faces[y, x])
        source_kernels = np.flatnonzero(source_face_ids == face_id).tolist()
        footprint_kernels = [kernel_id for kernel_id, faces in zip(kernel_ids, affected_sets)
                             if face_id in faces]
        print(f"Missing pixel (x={int(x)}, y={int(y)}): reference face ID={face_id}")
        print(f"  Source-face kernels (_kernel_face_ids): {source_kernels}")
        for kernel_id in source_kernels:
            print(f"  Kernel {kernel_id}: selected={kernel_id in selected_ids}; "
                  f"selected by --all-visible-kernels={bool(args.all_visible_kernels and kernel_id in selected_ids)}")
        print(f"  In union of selected affected faces: {face_id in affected_union}")
        print(f"  Selected kernels whose affected footprint contains face: {footprint_kernels}")
        print(f"  In union of selected visible faces: {face_id in visible_face_union}")
    debug_image = np.rint(np.clip(reference_colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    debug_image[missing_mask] = [255, 0, 255]
    output_path = REPO_ROOT / "outputs/kernel_face_debug/missing_coverage.png"
    save_raster_png(output_path, debug_image)
    print(f"Exported missing coverage debug (magenta pixels): {output_path}")


def report_uncovered_reference_pixels(reference_faces, reference_barys, reference_colors,
                                      patch_covered, kernel_ids, affected_sets, visible_union):
    """Diagnose gaps using existing buffers; leave all rendering data untouched."""
    import numpy as np

    reference_visible = reference_faces >= 0
    missing_mask = reference_visible & ~patch_covered
    rows, columns = np.nonzero(missing_mask)
    affected_union = set(int(face) for faces in affected_sets for face in faces)
    visible_faces = set(int(face) for face in visible_union)
    missing_faces = set(int(face) for face in reference_faces[missing_mask])
    face_kernels = {face: [] for face in missing_faces}
    for kernel_id, affected in zip(kernel_ids, affected_sets):
        for face in missing_faces.intersection(int(face) for face in affected):
            face_kernels[face].append(kernel_id)
    counts = dict.fromkeys((
        "face_not_selected", "face_selected_but_not_visible",
        "face_visible_but_not_rasterized", "other",
    ), 0)
    print(f"Uncovered reference mesh-visible pixels: {len(rows)}")
    print("Uncovered pixel coordinates use (x, y), with origin at the top-left.")
    for y, x in zip(rows, columns):
        face_id = int(reference_faces[y, x])
        barys = reference_barys[y, x]
        rgb = reference_colors[y, x]
        if not (np.isfinite(barys).all() and np.isfinite(rgb).all()):
            category = "other"
        elif face_id not in affected_union:
            category = "face_not_selected"
        elif face_id not in visible_faces:
            category = "face_selected_but_not_visible"
        else:
            category = "face_visible_but_not_rasterized"
        counts[category] += 1
        print(f"Uncovered pixel (x={int(x)}, y={int(y)}): reference face ID={face_id}")
        print(f"  Reference barycentric coordinates: {barys.tolist()}")
        print(f"  Reference RGB: {rgb.tolist()}")
        print(f"  Face in total unique visible affected faces: {face_id in visible_faces}")
        print(f"  Selected kernels with face in affected-face set: {face_kernels[face_id]}")
        print(f"  Category: {category}")
    print("Uncovered pixel category counts:")
    for category, count in counts.items():
        print(f"  {category}: {count}")
    # A subdued reference silhouette ensures only uncovered pixels are bright.
    # This is a fresh buffer; the reference render and its PNG remain unchanged.
    image = np.zeros((*reference_faces.shape, 3), dtype=np.uint8)
    image[reference_visible] = [55, 55, 55]
    image[missing_mask] = [255, 0, 255]
    path = REPO_ROOT / "outputs/kernel_face_debug/multi_kernel_uncovered.png"
    save_raster_png(path, image)
    print(f"Exported uncovered reference silhouette (magenta gaps): {path}")


def run_multi_kernel(trainer, density_model, args):
    """Render each kernel patch and compare their blend to full-mesh first hits."""
    import numpy as np
    import torch
    import trimesh

    if (trainer.model.cfg.knn_outer_k, trainer.model.cfg.knn_inner_k) != (50, 30):
        raise ValueError("Multi-kernel comparison expects the trained outer KNN=50 and inner top-k=30")
    if trainer.model.out_dim != 3:
        raise ValueError("Multi-kernel PNG rendering requires three output color channels")
    vertices = trainer.mesh.verts.detach().cpu().numpy()
    faces = trainer.mesh.faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if args.all_visible_kernels:
        kernel_ids, affected_sets, precomputed_visible, maximum_influences = find_visible_kernels(
            trainer, density_model, mesh, args
        )
    else:
        kernel_ids = args.kernel_ids
        affected_sets, influences = multi_kernel_affected_faces(
            trainer, density_model, kernel_ids, args.cutoff, args.batch_size
        )
        precomputed_visible = None
        maximum_influences = [float(values.max()) for values in influences]
    visible_sets, patch_faces, patch_depths = [], [], []
    for index, (kernel_id, affected) in enumerate(zip(kernel_ids, affected_sets)):
        visible = (precomputed_visible[index] if precomputed_visible is not None
                   else visible_affected_faces(
                       mesh, affected, args.camera_position, args.camera_look_at, args.batch_size
                   ))
        _, depth, _, face_ids, barys = rasterize_visible_triangles(
            vertices, faces, visible, args.camera_position, args.camera_look_at,
            args.image_width, args.image_height, args.fov_y,
        )
        covered, values = evaluate_raster_kernel(
            trainer, density_model, face_ids, barys, kernel_id, args.batch_size
        )
        visible_sets.append(visible)
        patch_faces.append(face_ids)
        patch_depths.append(depth)
        print(f"Kernel {kernel_id}: affected={len(affected)}, visible={len(visible)}, "
              f"pixels={int(covered.sum())}, maximum face sample influence={maximum_influences[index]:.9g}")
        if len(values):
            print(f"  Pixel influence min/max/mean: {values.min():.9g} / {values.max():.9g} / {values.mean():.9g}")
    visible_union = (np.unique(np.concatenate(visible_sets)) if visible_sets
                     else np.empty(0, dtype=np.int64))
    patch_image, union_depth, _, union_faces, union_barys = rasterize_visible_triangles(
        vertices, faces, visible_union, args.camera_position, args.camera_look_at,
        args.image_width, args.image_height, args.fov_y,
    )
    union_covered = union_faces >= 0
    # A kernel may contribute only where its own patch covers the union's
    # winning surface. Several masks may be true at the same pixel.
    patch_masks = []
    scale = max(float(np.linalg.norm(mesh.extents)), np.finfo(float).eps)
    for face_ids, depth in zip(patch_faces, patch_depths):
        patch_masks.append(union_covered & (face_ids == union_faces)
                           & np.isclose(depth, union_depth, rtol=1e-6, atol=scale * 1e-6))
    patch_colors = evaluate_selected_kernel_colors(
        trainer, union_faces, union_barys, kernel_ids, args.batch_size, patch_masks
    )
    reference_faces, reference_barys, _ = reference_mesh_surface(
        mesh, args.camera_position, args.camera_look_at,
        args.image_width, args.image_height, args.fov_y, args.batch_size,
    )
    reference_colors = evaluate_selected_kernel_colors(
        trainer, reference_faces, reference_barys, kernel_ids, args.batch_size
    )
    mesh_visible = reference_faces >= 0
    outside_patch = mesh_visible & ~union_covered
    if args.all_visible_kernels:
        report_uncovered_reference_pixels(
            reference_faces, reference_barys, reference_colors, union_covered,
            kernel_ids, affected_sets, visible_union,
        )
    if args.debug_missing_coverage:
        debug_missing_coverage(
            trainer, args, reference_faces, union_covered, reference_colors,
            kernel_ids, affected_sets, visible_union,
        )
    # Zero kernel color contribution still receives the trained mean, followed
    # by the same clamp and model postprocessing as the normal HKTex blend.
    # Fill only uncovered mesh pixels; actual background remains black.
    with torch.no_grad():
        mean_color = trainer.model._mean_colour.detach().cpu().numpy().reshape(-1)
        base_color = trainer.model(
            trainer.model._mean_colour.clamp(0.0, 1.0)
        ).detach().cpu().numpy().reshape(-1)
    patch_colors[outside_patch] = base_color
    print(f"Model _mean_colour RGB: {mean_color.tolist()}")
    print(f"HKTex zero-contribution base RGB (clamped/postprocessed): {base_color.tolist()}")
    if outside_patch.any():
        outside_reference_mean = reference_colors[outside_patch].mean(axis=0, dtype=np.float64)
        mean_matches = np.allclose(mean_color, outside_reference_mean, rtol=1e-5, atol=1e-6)
        base_matches = np.allclose(base_color, outside_reference_mean, rtol=1e-5, atol=1e-6)
        print(f"Outside-patch reference mean RGB: {outside_reference_mean.tolist()}")
        print(f"Model mean colour matches outside-patch reference mean RGB: {bool(mean_matches)}")
        print(f"Postprocessed base matches outside-patch reference mean RGB: {bool(base_matches)}")
        print(f"Outside-patch reference deviation from base (max absolute): "
              f"{np.abs(reference_colors[outside_patch] - base_color).max():.9g}")
    else:
        print("Model mean colour matches outside-patch reference mean RGB: N/A (empty region)")
    differences = np.abs(patch_colors - reference_colors)
    masked_diff = np.zeros_like(differences)
    masked_diff[mesh_visible] = differences[mesh_visible]
    output_dir = REPO_ROOT / "outputs/kernel_face_debug"
    for name, image in (
        ("multi_kernel_patch", patch_image),
        ("multi_kernel_render", np.rint(np.clip(patch_colors, 0, 1) * 255).astype(np.uint8)),
        ("multi_kernel_reference", np.rint(np.clip(reference_colors, 0, 1) * 255).astype(np.uint8)),
        ("multi_kernel_diff", np.rint(np.clip(masked_diff, 0, 1) * 255).astype(np.uint8)),
    ):
        path = output_dir / f"{name}.png"
        save_raster_png(path, image)
        print(f"Exported: {path}")
    print("Footprint mode: vertices+centroid")
    print(f"Number of kernels: {len(kernel_ids)}")
    print(f"Affected face memberships (sum across kernels): {sum(len(ids) for ids in affected_sets)}")
    print(f"Total affected faces summed across selected kernels: {sum(len(ids) for ids in affected_sets)}")
    unique_affected_count = len(np.unique(np.concatenate(affected_sets))) if affected_sets else 0
    print(f"Total unique affected faces: {unique_affected_count}")
    print(f"Total unique visible faces: {len(visible_union)}")
    print(f"Total unique visible affected faces: {len(visible_union)}")
    print(f"Rasterized pixel count: {int(union_covered.sum())}")
    print(f"Reference mesh-visible pixels: {int(mesh_visible.sum())}")
    if mesh_visible.any():
        mesh_visible_patch_pixels = int((union_covered & mesh_visible).sum())
        coverage = mesh_visible_patch_pixels / int(mesh_visible.sum())
        print(f"Mesh-visible pixels covered by selected patch union: {mesh_visible_patch_pixels}")
        print(f"Visible pixel coverage: {100.0 * coverage:.6f}%")
    else:
        print("Visible pixel coverage: N/A (no reference mesh-visible pixels)")
    print("Blend uses full-model outer KNN=50, inner top-k=30, and denominator; only selected color terms remain.")
    print("Metrics use unquantized RGB values over mesh-visible pixels/channels; background is excluded.")
    if mesh_visible.any():
        errors = differences[mesh_visible]
        print(f"MAE: {errors.mean():.9g}")
        print(f"RMSE: {np.sqrt(np.mean(errors ** 2)):.9g}")
        print(f"Max absolute error: {errors.max():.9g}")
    else:
        print("MAE / RMSE / max absolute error: N/A (no mesh-visible pixels)")

    # Diagnostic only: partition the existing comparison mask by patch coverage.
    # Use unquantized RGB, with equal weight for every pixel and color channel.
    for region_name, region_mask in (
        ("inside_patch", mesh_visible & union_covered),
        ("outside_patch", mesh_visible & ~union_covered),
    ):
        pixel_count = int(region_mask.sum())
        print(f"{region_name} pixel count: {pixel_count}")
        if pixel_count:
            region_errors = differences[region_mask]
            print(f"{region_name} MAE: {region_errors.mean():.9g}")
            print(f"{region_name} RMSE: {np.sqrt(np.mean(region_errors ** 2)):.9g}")
            print(f"{region_name} max absolute error: {region_errors.max():.9g}")
            print(f"{region_name} multi-kernel render mean RGB: {patch_colors[region_mask].mean(axis=0).tolist()}")
            print(f"{region_name} reference mean RGB: {reference_colors[region_mask].mean(axis=0).tolist()}")
        else:
            print(f"{region_name} MAE / RMSE / max absolute error: N/A (empty region)")
            print(f"{region_name} multi-kernel render mean RGB: N/A (empty region)")
            print(f"{region_name} reference mean RGB: N/A (empty region)")


def main():
    args = parse_args()
    # Experiment paths are relative to the repository, as in the training entrypoint.
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    import numpy as np
    import torch
    import trimesh

    # This module selects Mitsuba's cuda_ad_rgb variant before importing HKTex.
    from interactive_density_knn import load_datamodule_and_trainer_only
    from hktex.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
    from hktex.modules.eigen_albo_knn import EigenAlboInterpolationKNN
    from hktex.modules.heat_kernel_density_knn import HeatKernelDensityKNN
    from hktex.utils import config_to_primitive

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    config_path = args.experiment / "configs/parsed.yaml"
    cfg, _, trainer = load_datamodule_and_trainer_only(
        argparse.Namespace(config=str(config_path)), extras=[]
    )
    if not isinstance(trainer.model, HeatKernelTextureKNN):
        raise TypeError("The experiment must construct HeatKernelTextureKNN")
    if not isinstance(trainer.eigalbo_interp, EigenAlboInterpolationKNN):
        raise TypeError("The experiment must construct EigenAlboInterpolationKNN")

    checkpoint = args.experiment / "ckpts" / cfg.optim.save_model_name
    trainer.model.load_torch(str(checkpoint))
    trainer.model.eval()
    requested_ids = [] if args.all_visible_kernels else (
        args.kernel_ids if args.kernel_ids is not None else [args.kernel_id]
    )
    if any(kernel_id >= trainer.model.N_sources for kernel_id in requested_ids):
        raise ValueError(f"Kernel ID must be below {trainer.model.N_sources}")

    density_model = HeatKernelDensityKNN(
        config_to_primitive(trainer.model.cfg), trainer.mesh
    )
    density_model.load_torch(str(checkpoint))
    density_model.eval()
    # The existing density helper omits this power operation; this experiment
    # uses the default 1. Reject other experiments rather than silently diverge.
    if trainer.model.cfg.power_diffused_diracs != 1:
        raise ValueError("filtered_kernel_weights requires power_diffused_diracs=1 for parity")

    if args.kernel_ids is not None or args.all_visible_kernels:
        run_multi_kernel(trainer, density_model, args)
        return

    mesh = trainer.mesh
    source_face = int(trainer.model.kernel_face_ids[args.kernel_id].item())
    influence = np.zeros(mesh.N_faces, dtype=np.float32)
    sample_barycentrics = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
        device=mesh.verts.device, dtype=mesh.verts.dtype,
    )
    with torch.no_grad():
        try:
            trainer.prepare_knn(save_barycentric=False)
            for start in range(0, mesh.N_faces, args.batch_size):
                stop = min(start + args.batch_size, mesh.N_faces)
                face_ids = torch.arange(start, stop, device=mesh.faces.device)
                sample_influences = []
                # Each call stays within --batch-size points. The four samples
                # share face IDs but each runs its own trained outer-KNN query.
                for sample_bary in sample_barycentrics:
                    barys = sample_bary.expand(stop - start, -1).contiguous()
                    points_info = trainer.model.prepare_points(
                        mesh=mesh, eigalbo_interp=trainer.eigalbo_interp,
                        face_ids=face_ids, barys=barys, pts=None,
                    )
                    filtered, global_indices = density_model.filtered_kernel_weights(
                        points_info, trainer.eigalbo_interp
                    )
                    matches = global_indices == args.kernel_id
                    batch_influence = torch.where(matches, filtered, 0.0).sum(dim=1)
                    sample_influences.append(batch_influence)
                    del points_info, filtered, global_indices, matches, batch_influence
                influence[start:stop] = torch.stack(sample_influences).amax(dim=0).cpu().numpy()
                del sample_influences
        finally:
            trainer.reset_knn()

    if not np.isfinite(influence).all():
        raise ValueError("Nonfinite face sample influence encountered")
    affected = np.flatnonzero(influence > args.cutoff)
    print(f"Checkpoint: {checkpoint}")
    print(f"Kernel ID: {args.kernel_id}")
    print(f"Source face ID: {source_face}")
    print(f"Outer KNN: {trainer.model.cfg.knn_outer_k}; inner top-k: {trainer.model.cfg.knn_inner_k}")
    print("Footprint mode: vertices+centroid")
    print("Face selection influence: maximum across vertex 0, vertex 1, vertex 2, and centroid")
    print(f"Affected face count (influence > {args.cutoff}): {len(affected)} / {mesh.N_faces}")
    print(f"Min/max face sample maxima: {influence.min():.9g} / {influence.max():.9g}")
    print(f"Maximum sample influence used for face selection: {influence.max():.9g}")
    print(f"Affected face IDs: {affected.tolist()}")
    # Report centroids for spatial inspection, independently of selection samples.
    vertices = mesh.verts.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    source_centroid = vertices[faces[source_face]].mean(axis=0)
    print(f"Source face {source_face} centroid: {source_centroid.tolist()}")
    print("Affected face centroids:")
    for face_id in affected:
        centroid = vertices[faces[face_id]].mean(axis=0)
        print(f"  Face {int(face_id)}: {centroid.tolist()}")

    # Export the evaluated mesh with its exact face order. Split shared vertices
    # so GLB's vertex-color representation preserves distinct triangle colors.
    result = trimesh.Trimesh(
        vertices=vertices,
        faces=faces, process=False,
    )
    # Use original shared-edge adjacency before splitting vertices for export.
    # Only selected faces may connect components; isolated faces count too.
    neighbors = {int(face_id): [] for face_id in affected}
    for left, right in result.face_adjacency:
        left, right = int(left), int(right)
        if left in neighbors and right in neighbors:
            neighbors[left].append(right)
            neighbors[right].append(left)
    visited = set()
    components = []
    for face_id in neighbors:
        if face_id in visited:
            continue
        visited.add(face_id)
        pending = [face_id]
        component = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        components.append(sorted(component))
    print(f"Affected face connected component count: {len(components)}")
    for component_id, component in enumerate(components, start=1):
        print(f"  Component {component_id} ({len(component)} faces): {component}")
    visible = visible_affected_faces(
        result, affected, args.camera_position, args.camera_look_at, args.batch_size
    )
    occluded = np.setdiff1d(affected, visible)
    print(f"Camera position: {args.camera_position}; look-at: {args.camera_look_at}")
    print("Visibility tests triangle vertices, edge midpoints, and centroid against the full opaque mesh, two-sided.")
    print("Behind-camera faces are included in the occluded group; no FOV limit is applied.")
    print(f"Affected face count: {len(affected)}")
    print(f"Visible affected face count: {len(visible)}")
    print(f"Occluded affected face count: {len(occluded)}")
    print(f"Visible affected face IDs: {visible.tolist()}")
    print(f"Occluded affected face IDs: {occluded.tolist()}")
    visible_face_ids = visible
    raster, raster_depth, rasterized_count, raster_faces, raster_barys = rasterize_visible_triangles(
        vertices, faces, visible_face_ids, args.camera_position, args.camera_look_at,
        args.image_width, args.image_height, args.fov_y,
    )
    raster_path = REPO_ROOT / "outputs/kernel_face_debug" / f"kernel_{args.kernel_id}_raster.png"
    save_raster_png(raster_path, raster)
    print(f"Visible triangles rasterized (covering pixel centers): {rasterized_count}")
    print(f"Raster pixels covered: {int(np.isfinite(raster_depth).sum())}")
    print(f"Exported raster PNG: {raster_path}")
    covered, pixel_influence = evaluate_raster_kernel(
        trainer, density_model, raster_faces, raster_barys, args.kernel_id, args.batch_size
    )
    heat_image = np.zeros((args.image_height, args.image_width), dtype=np.uint8)
    if len(pixel_influence):
        minimum = float(pixel_influence.min())
        maximum = float(pixel_influence.max())
        mean = float(pixel_influence.mean())
        if maximum > 0.0:
            heat_image[covered] = np.rint(
                np.clip(pixel_influence / maximum, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        print(f"Rasterized pixel influence min/max/mean: {minimum:.9g} / {maximum:.9g} / {mean:.9g}")
    else:
        print("Rasterized pixel influence min/max/mean: N/A (no covered pixels)")
    heat_path = REPO_ROOT / "outputs/kernel_face_debug" / f"kernel_{args.kernel_id}_heat.png"
    save_raster_png(heat_path, heat_image)
    print(f"Exported heat PNG: {heat_path}")
    if args.reference_comparison:
        compare_reference_kernel(
            trainer, density_model, result, args, raster_faces, raster_depth,
            covered, pixel_influence,
        )
    result.unmerge_vertices()
    colors = np.tile(np.array([45, 45, 45, 90], dtype=np.uint8), (mesh.N_faces, 1))
    colors[affected] = [255, 80, 0, 255]
    colors[source_face] = [0, 110, 255, 255]
    result.visual = trimesh.visual.ColorVisuals(mesh=result, face_colors=colors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ply_path = args.output.with_suffix(".ply")
    glb_path = args.output.with_suffix(".glb")
    result.export(str(ply_path), file_type="ply")

    # Use three explicit glTF PBR materials rather than relying on a viewer's
    # vertex-color shader. Groups partition the original faces without overlap;
    # the source face remains blue even when it is selected as affected.
    scene = trimesh.Scene()
    for name, rgba in (
        ("unaffected", [45, 45, 45, 90]),
        ("affected", [255, 80, 0, 255]),
        ("source", [0, 110, 255, 255]),
    ):
        group_ids = np.flatnonzero(np.all(colors == rgba, axis=1))
        if not len(group_ids):
            continue
        part = result.submesh([group_ids], append=True, repair=False)
        part.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name=name, baseColorFactor=np.array(rgba, dtype=np.uint8),
                metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True,
                alphaMode="BLEND" if rgba[3] < 255 else "OPAQUE",
            )
        )
        scene.add_geometry(part, geom_name=name, node_name=name)
    scene.export(str(glb_path), file_type="glb")
    print(f"Exported PLY: {ply_path}")
    print(f"Exported GLB: {glb_path}")
    print("Unaffected faces use partial alpha; PLY transparency depends on the viewer.")


if __name__ == "__main__":
    main()
