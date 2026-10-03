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
    parser.add_argument("--kernel-id", type=int, required=True)
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
    if args.kernel_id < 0:
        parser.error("--kernel-id must be nonnegative")
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
        args.output = REPO_ROOT / "outputs/kernel_face_debug" / f"kernel_{args.kernel_id}_faces.ply"
    args.output = args.output.resolve()
    if args.output.suffix.lower() not in {".ply", ".glb"}:
        parser.error("--output must end in .ply or .glb")
    return args


def visible_affected_faces(mesh, affected, camera_position, camera_look_at, batch_size):
    """Return affected IDs whose centroids are the nearest full-mesh ray hit.

    This tests two-sided geometry, not face normals. Partially exposed triangles
    with hidden centroids are occluded by this diagnostic. Behind-camera faces
    and failed/no-hit rays are conservatively classified as occluded.
    """
    import numpy as np
    from trimesh.ray.ray_triangle import RayMeshIntersector

    camera = np.asarray(camera_position, dtype=np.float64)
    forward = np.asarray(camera_look_at, dtype=np.float64) - camera
    forward /= np.linalg.norm(forward)
    intersector = RayMeshIntersector(mesh)
    visible = []
    mesh_scale = max(float(np.linalg.norm(mesh.extents)), np.finfo(float).eps)
    for start in range(0, len(affected), batch_size):
        face_ids = affected[start:start + batch_size]
        centroids = mesh.vertices[mesh.faces[face_ids]].mean(axis=1)
        offsets = centroids - camera
        distances = np.linalg.norm(offsets, axis=1)
        eligible = (distances > 0.0) & (offsets @ forward > 0.0)
        if not eligible.any():
            continue
        target_ids = face_ids[eligible]
        target_distances = distances[eligible]
        directions = offsets[eligible] / target_distances[:, None]
        origins = np.broadcast_to(camera, directions.shape).copy()
        try:
            locations, ray_ids, hit_faces = intersector.intersects_location(
                ray_origins=origins, ray_directions=directions, multiple_hits=False,
            )
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Trimesh occlusion testing requires its ray dependencies (including rtree). "
                "Install them in your HKTex environment."
            ) from error
        hit_distances = np.linalg.norm(locations - camera, axis=1)
        tolerance = 1e-6 * mesh_scale + 1e-7 * target_distances[ray_ids]
        is_visible = (hit_faces == target_ids[ray_ids]) & (
            np.abs(hit_distances - target_distances[ray_ids]) <= tolerance
        )
        visible.extend(target_ids[ray_ids[is_visible]].tolist())
    return np.asarray(sorted(set(visible)), dtype=np.int64)


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
    if args.kernel_id >= trainer.model.N_sources:
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
    print("Visibility tests triangle centroids against the full opaque mesh, two-sided.")
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
