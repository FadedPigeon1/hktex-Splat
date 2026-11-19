"""Rasterize fixed triangles with the preserved float64 clipping and depth rules.

CPU mesh visibility before spectral rendering; preserves pixel ties.
Main API: _fast_kernel, StaticRaster."""
import math
import numpy as np
from numba import njit
from hktex_splatting.rendering.camera_raster import perspective_camera_basis


@njit(cache=False, fastmath=False)
def _fast_kernel(vertices, faces, face_ids, origin, basis, width, height,
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
            # Homogeneous planes need no polygon copies. Mixed cases retain
            # every original clipping operation in its original order.
            inside = 0
            for vertex in range(size):
                if np.dot(polygon[vertex, :3], normals[plane]) - offsets[plane] >= 0.0:
                    inside += 1
            if inside == size:
                continue
            if inside == 0:
                size = 0
                break
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


class StaticRaster:
    def __init__(self, vertices, faces):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.face_ids = np.arange(len(faces), dtype=np.int64)
        self.near = max(max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)*1e-7, 1e-12)

    def all_buffers(self, camera):
        # Same camera arithmetic as _kernel_args; static geometry reductions and
        # casts are already prepared, while the raster is always fresh.
        origin, basis = perspective_camera_basis(camera["position"], camera["target"])
        width, height = camera["width"], camera["height"]
        tan_y = math.tan(math.radians(camera["fov_y_degrees"])/2.0)
        tan_x = tan_y*width/height
        normals = np.array([[0.,0.,1.], [1.,0.,tan_x], [-1.,0.,tan_x], [0.,1.,tan_y], [0.,-1.,tan_y]])
        offsets = np.array([self.near,0.,0.,0.,0.])
        return _fast_kernel(self.vertices, self.faces, self.face_ids, origin, basis,
                            width, height, tan_x, tan_y, normals, offsets)

    def __call__(self, vertices, faces, camera, backend="numba"):
        if backend != "numba":
            raise ValueError("Optimized rasterizer requires numba")
        _, depth, _, ids, bary = self.all_buffers(camera)
        return ids, bary, depth
