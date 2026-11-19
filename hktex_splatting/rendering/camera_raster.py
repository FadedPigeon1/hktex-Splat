"""Shared float64 camera, clipping and reference mesh visibility calculations.

Used by native/HKS geometry preparation and reference rendering. Camera basis,
NumPy rasterize_visible_triangles and rasterize_numba retain the original
helper equations; this module contains no visualization or test runner.
"""
import math
import numpy as np
from numba import njit

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

