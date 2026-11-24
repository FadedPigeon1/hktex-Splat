"""CPU-only Converse inspection; never trains, computes eigenpairs, or overwrites assets.

Run with the existing hktex environment (numpy, scipy, trimesh, Pillow).
The actual load_mesh function is compiled unchanged from its AST to avoid the
package initializer, which selects Mitsuba's CUDA variant even during import.
Only its hktex.info logging dependency is replaced with Python logging.
"""

import ast
import hashlib
import io
import json
import logging
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh
from PIL import Image, ImageDraw
import PIL


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data/converse/prepared"
RAW = ROOT / "data/converse/raw/converse__free.glb"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new_or_identical(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite different existing file: {path}")
        return "existing_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return "created"


def save_png(path, image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    write_new_or_identical(path, buffer.getvalue())


def source_function(relative_path, name, namespace, class_name=None):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def component_sizes(faces, vertex_count):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
                       shape=(vertex_count, vertex_count)).tocsr()
    count, labels = connected_components(graph, directed=False)
    return {"count": int(count),
            "vertices_descending": sorted(np.bincount(labels).tolist(), reverse=True)}


def preview(mesh, texture):
    """Small orthographic, unlit, two-sided, nearest-texel CPU z-buffer preview.

    Uses every original triangle and the full-resolution source texture. Display
    sampling is intentionally simple; these images are inspection aids, not
    camera-matched experiment renders or measurements of texture sharpness.
    """
    vertices = mesh.original_vertices
    faces = mesh.original_faces
    uv = mesh.original_uv
    pixels = np.asarray(texture)
    height, width = pixels.shape[:2]
    size = 640
    canvas = Image.new("RGB", (size * 3, size + 45), "white")
    coverage = []
    for view, angle in enumerate([75, -75, 170]):
        a = np.deg2rad(angle)
        forward = np.array([np.cos(a), 0.24, np.sin(a)])
        forward /= np.linalg.norm(forward)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        xyz = vertices @ np.stack([right, up, forward], axis=1)
        xy = xyz[:, :2]
        factor = (size - 50) / np.ptp(xy, axis=0).max()
        xy = (xy - (xy.min(axis=0) + xy.max(axis=0)) / 2) * factor
        xy[:, 1] *= -1
        xy += size / 2
        image = np.full((size, size, 3), 245, dtype=np.uint8)
        depth = np.full((size, size), -np.inf)
        for ids in faces:
            tri = xy[ids]
            lo = np.maximum(np.ceil(tri.min(axis=0)).astype(int), 0)
            hi = np.minimum(np.floor(tri.max(axis=0)).astype(int), size - 1)
            if np.any(lo > hi):
                continue
            x, y = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            p, q, r = tri
            den = (q[1] - r[1]) * (p[0] - r[0]) + (r[0] - q[0]) * (p[1] - r[1])
            if abs(den) < 1e-12:
                continue
            b0 = ((q[1] - r[1]) * (x - r[0]) + (r[0] - q[0]) * (y - r[1])) / den
            b1 = ((r[1] - p[1]) * (x - r[0]) + (p[0] - r[0]) * (y - r[1])) / den
            bary = np.stack([b0, b1, 1 - b0 - b1], axis=-1)
            z = bary @ xyz[ids, 2]
            mask = (bary.min(axis=-1) >= -1e-8) & (z > depth[y, x])
            if not mask.any():
                continue
            xx, yy = x[mask], y[mask]
            tuv = bary[mask] @ uv[ids]
            tx = np.rint(tuv[:, 0] * (width - 1)).astype(int) % width
            ty = np.rint((1 - tuv[:, 1]) * (height - 1)).astype(int) % height
            image[yy, xx] = pixels[ty, tx, :3]
            depth[yy, xx] = z[mask]
        visible = np.isfinite(depth)
        coverage.append({"view": view, "azimuth_degrees": angle,
                         "foreground_pixels": int(visible.sum()),
                         "exact_black_foreground_pixels": int(np.all(image[visible] == 0, axis=1).sum())})
        canvas.paste(Image.fromarray(image), (view * size, 45))
    ImageDraw.Draw(canvas).text((12, 10),
        "Original UV texture | CPU unlit inspection views | not an experiment render", fill="black")
    return canvas, coverage


def main():
    original = RAW.read_bytes()
    magic, version, length = struct.unpack_from("<III", original)
    assert magic == 0x46546C67 and version == 2 and length == len(original)
    chunks = {}
    offset = 12
    while offset < length:
        size, kind = struct.unpack_from("<II", original, offset)
        chunks[kind] = original[offset + 8:offset + 8 + size]
        offset += 8 + size
    gltf = json.loads(chunks[0x4E4F534A])
    binary = chunks[0x004E4942]
    assert len(gltf["meshes"]) == len(gltf["materials"]) == len(gltf["images"]) == 1
    assert len(gltf["meshes"][0]["primitives"]) == 1
    primitive = gltf["meshes"][0]["primitives"][0]
    assert primitive.get("mode", 4) == 4
    image_view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    start = image_view.get("byteOffset", 0)
    embedded = binary[start:start + image_view["byteLength"]]
    tex = Image.open(io.BytesIO(embedded))
    tex.load()
    prepared = PREPARED / RAW.name
    texture_path = PREPARED / "textures/base_color.jpg"
    actions = {str(p.relative_to(ROOT)): write_new_or_identical(p, data)
               for p, data in [(prepared, original), (texture_path, embedded)]}
    print("Original and prepared bytes verified", flush=True)

    load_mesh = source_function("hktex/utils/mesh.py", "load_mesh",
        {"trimesh": trimesh, "hktex": SimpleNamespace(info=logging.info)})
    get_texture_image = source_function("hktex/data/uv_texture_sampler.py", "get_texture_image", {},
                                       class_name="UvTextureSamplerDataset")
    scene = trimesh.load(str(RAW), process=False)
    node = scene.graph.nodes_geometry[0]
    transform, geometry_name = scene.graph[node]
    geometry = scene.geometry[geometry_name]
    assert geometry.visual.uv.shape == (len(geometry.vertices), 2)
    loaded = {}
    for path in [RAW, prepared]:
        for merge_tex in [False, True]:
            mesh = load_mesh(str(path), merge_tex=merge_tex, bake_vert_colors=False)
            loaded[(path, merge_tex)] = mesh
            print(path.relative_to(ROOT), "merge_tex", merge_tex,
                  "vertices", len(mesh.vertices), "faces", len(mesh.faces), flush=True)
    mesh = loaded[(prepared, True)]
    reference = loaded[(RAW, True)]
    assert np.array_equal(mesh.faces, reference.faces)
    assert np.array_equal(mesh.vertices, reference.vertices)
    assert np.array_equal(mesh.original_uv, reference.original_uv)
    assert np.array_equal(mesh.original_faces, geometry.faces)
    assert np.array_equal(mesh.original_uv, geometry.visual.uv)
    assert len(mesh.faces) == 75000 and len(geometry.vertices) == 39030
    assert np.allclose(mesh.vertices[mesh.faces], mesh.original_vertices[mesh.original_faces], atol=1e-7)
    texture = get_texture_image(SimpleNamespace(mesh=mesh))
    assert np.array_equal(np.asarray(texture), np.asarray(tex))
    assert np.isfinite(mesh.vertices).all() and np.isfinite(mesh.original_uv).all()
    colors = trimesh.visual.uv_to_color(mesh.original_uv[mesh.original_faces].mean(axis=1), texture)
    assert colors.shape == (len(mesh.faces), 4)

    world = geometry.copy()
    world.apply_transform(transform)
    translation = -world.centroid
    scale = 2.0 / world.extents.max()
    normalization = np.eye(4)
    normalization[:3, :3] *= scale
    normalization[:3, 3] = translation * scale
    assert np.allclose(mesh.original_vertices,
        trimesh.transform_points(geometry.vertices, normalization @ transform))
    welded = component_sizes(mesh.faces, len(mesh.vertices))
    indexed = component_sizes(geometry.faces, len(geometry.vertices))
    areas = world.area_faces
    nonmerged = loaded[(prepared, False)]
    print("Components (indexed / HKTex welded)", indexed["count"], welded["count"], flush=True)
    print("Generating small CPU preview", flush=True)
    reference_preview, coverage = preview(mesh, texture)
    save_png(PREPARED / "reference_preview.png", reference_preview)
    overview = tex.copy()
    overview.thumbnail((1400, 1400))
    save_png(PREPARED / "texture_overview.png", overview)
    # Native-resolution crop containing the round Converse patch, selected from
    # inspection of the texture overview. Does not alter the training texture.
    save_png(PREPARED / "branding_crop.png", tex.crop((4400, 2100, 6100, 3400)))

    report = {
        "original_path": str(RAW.relative_to(ROOT)), "prepared_path": str(prepared.relative_to(ROOT)),
        "asset_actions": actions, "original_sha256": sha(original),
        "prepared_sha256": sha(prepared.read_bytes()), "file_bytes": length,
        "asset_metadata": gltf["asset"], "scene_count": len(gltf["scenes"]),
        "node_count": len(gltf["nodes"]), "mesh_object_count": len(scene.graph.nodes_geometry),
        "mesh_count": len(gltf["meshes"]), "primitive_count": 1, "primitive_mode": 4,
        "source_vertices": len(geometry.vertices), "triangles": len(geometry.faces),
        "uv_sets": [k for k in primitive["attributes"] if k.startswith("TEXCOORD")],
        "uv_min_trimesh": geometry.visual.uv.min(axis=0).tolist(),
        "uv_max_trimesh": geometry.visual.uv.max(axis=0).tolist(),
        "materials": gltf["materials"], "textures": gltf["textures"], "samplers": gltf["samplers"],
        "texture": {"path": str(texture_path.relative_to(ROOT)), "width": tex.width,
                    "height": tex.height, "format": tex.format, "mode": tex.mode,
                    "mime_type": gltf["images"][0]["mimeType"], "encoded_bytes": len(embedded),
                    "sha256": sha(embedded), "byte_identical_to_embedded": True},
        "scene_transform": transform.tolist(), "scene_transform_scale_singular_values":
            np.linalg.svd(transform[:3, :3], compute_uv=False).tolist(),
        "world_bounds": world.bounds.tolist(), "world_centroid": world.centroid.tolist(),
        "hktex_translation_before_scale": translation.tolist(), "hktex_scale": float(scale),
        "local_to_evaluated_matrix": (normalization @ transform).tolist(),
        "components_source_indexed": indexed, "components_after_hktex_merge_tex_true": welded,
        "hktex_vertices_merge_tex_false": len(nonmerged.vertices),
        "hktex_vertices_merge_tex_true": len(mesh.vertices),
        "hktex_faces": len(mesh.faces), "stored_original_uv_vertices": len(mesh.original_uv),
        "degenerate_world_triangles_area_le_1e_16": int((areas <= 1e-16).sum()),
        "watertight_after_merge": bool(mesh.is_watertight),
        "face_centroid_samples": {"count": len(colors),
            "exact_black_rgb_count": int(np.all(colors[:, :3] == 0, axis=1).sum()),
            "rgb_min": colors[:, :3].min(axis=0).tolist(),
            "rgb_max": colors[:, :3].max(axis=0).tolist()},
        "preview": {"path": "data/converse/prepared/reference_preview.png",
                    "width": reference_preview.width, "height": reference_preview.height,
                    "method": "CPU orthographic unlit two-sided nearest texel z-buffer; all source faces",
                    "views": coverage, "experiment_reference": False},
        "validation": {"loader": "Unmodified load_mesh function compiled from repository AST; logging-only hktex.info substitute",
            "full_hktex_import": "Not attempted: package selects cuda_ad_rgb during import",
            "sampler_texture_accessor": "Actual get_texture_image method compiled from repository AST",
            "loader_raw_and_prepared_merge_tex_false_and_true": "passed",
            "face_corner_positions_and_original_uv": "passed", "decoded_texture_pixels": "identical",
            "full_gpu_datamodule_training_and_mitsuba_render": "not performed"},
        "environment": {"python": sys.executable, "numpy": np.__version__,
                        "scipy": scipy.__version__, "trimesh": trimesh.__version__, "Pillow": PIL.__version__},
    }
    write_new_or_identical(PREPARED / "inspection.json", (json.dumps(report, indent=2) + "\n").encode())
    assert RAW.read_bytes() == original
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
