"""Sample the original mesh UV texture at triangle-renderer pixel centers.

Rendering/benchmark target images; no learned kernel evaluation.
Main API: texture_sample, render, main."""
from __future__ import annotations

import time
from hktex_splatting.utils.reference_assets import (
    MESH,
    camera,
    check_repetitions,
    common_parser,
    geometry_digest,
    glb_json,
    load_cpu_mesh,
    raster_surface,
    save_npy,
    save_png,
    save_surface,
    sha256,
    stats,
    write_json,
)


def texture_sample(mesh, ids, barys, filtering="nearest"):
    import numpy as np
    import trimesh
    covered = ids >= 0
    face_ids = ids[covered]
    # UV corners come from the pre-weld geometry; model faces retain face order.
    uv = np.einsum("ij,ijk->ik", barys[covered],
                   mesh.original_uv[mesh.original_faces[face_ids]])
    material = mesh.visual.material
    materials = getattr(material, "materials", [material])
    assignments = getattr(mesh.visual, "face_materials", None)
    if len(materials) > 1 and assignments is None:
        raise ValueError("Multiple materials without face assignments; refusing approximate texture mapping")
    assignments = np.zeros(len(mesh.faces), dtype=int) if assignments is None else assignments
    rgb = np.zeros((*ids.shape, 3), np.float32)
    values = np.empty((len(face_ids), 3), np.float32)
    textures = []
    for index, mat in enumerate(materials):
        texture = getattr(mat, "baseColorTexture", None)
        if texture is None:
            texture = getattr(mat, "image", None)
        if texture is None:
            raise ValueError(f"Material {index} has no diffuse texture")
        textures.append(list(texture.size))
        selected = assignments[face_ids] == index
        coords = uv[selected]
        if filtering == "nearest":
            # Identical to the UV training target: encoded RGB/255, no gamma operation.
            values[selected] = trimesh.visual.uv_to_color(coords, texture)[:, :3] / 255.0
        else:
            # Explicit bilinear in encoded sRGB, same V orientation/repeat as Trimesh.
            pixels = np.asarray(texture.convert("RGB"))
            height, width = pixels.shape[:2]
            x = (coords[:, 0] * (width - 1)) % width
            y = ((1 - coords[:, 1]) * (height - 1)) % height
            x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
            dx, dy = (x - x0)[:, None], (y - y0)[:, None]
            x1, y1 = (x0 + 1) % width, (y0 + 1) % height
            values[selected] = ((1-dx)*(1-dy)*pixels[y0, x0] + dx*(1-dy)*pixels[y0, x1]
                                + (1-dx)*dy*pixels[y1, x0] + dx*dy*pixels[y1, x1]) / 255
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite texture values")
    rgb[covered] = values
    return rgb, textures


def render(args):
    import numpy as np
    check_repetitions(args)
    started = time.perf_counter()
    c = camera(args.view, args.resolution, args.cameras)
    args.output.mkdir(parents=True, exist_ok=False)
    load_start = time.perf_counter()
    gltf = glb_json()
    # Fail explicitly on unsupported material behavior rather than label an
    # approximate shader/material treatment as original ground truth.
    for material in gltf.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        if pbr.get("baseColorFactor", [1,1,1,1]) != [1,1,1,1] or material.get("alphaMode", "OPAQUE") != "OPAQUE":
            raise ValueError("This experiment requires opaque, unit-factor diffuse materials")
        binding = pbr.get("baseColorTexture", {})
        if binding.get("texCoord", 0) != 0 or binding.get("extensions"):
            raise ValueError("Unsupported texture coordinate set/transform; refusing approximate UV sampling")
    mesh = load_cpu_mesh()
    # Mesh.from_trimesh stores float32 verts and int32 faces; match this quantization.
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    corner_error = float(np.max(np.abs(mesh.original_vertices[mesh.original_faces] - mesh.vertices[mesh.faces])))
    if corner_error > 1e-6:
        raise ValueError(f"Welding changed UV corner geometry by {corner_error}")
    load_s = time.perf_counter() - load_start
    def frame():
        t = time.perf_counter()
        ids, barys, depth = raster_surface(vertices, faces, c, args.raster_backend)
        raster_s = time.perf_counter() - t
        t = time.perf_counter()
        rgb, textures = texture_sample(mesh, ids, barys, args.filtering)
        return rgb, ids, barys, depth, textures, raster_s, time.perf_counter()-t
    warm_start = time.perf_counter()
    for _ in range(args.warmup_runs):
        result = frame()
    warm_s = time.perf_counter() - warm_start
    startup_s = time.perf_counter() - started
    samples, raster_samples, sample_samples = [], [], []
    for _ in range(args.benchmark_runs):
        t = time.perf_counter()
        result = frame()
        samples.append(time.perf_counter() - t)
        raster_samples.append(result[-2]); sample_samples.append(result[-1])
    rgb, ids, barys, depth, textures, _, _ = result
    save_npy(args.output / "rgb.npy", rgb)
    save_png(args.output / "rgb.png", rgb)
    save_surface(args.output, ids, barys, depth)
    report = {"method": "original_uv", "appearance": "unlit, double sided, black background",
              "color_space": "srgb_encoded", "filtering": args.filtering,
              "filtering_note": "nearest matches actual HKTex training target; bilinear is optional and changes that target; no mipmaps",
              "camera": c, "texture_sizes": textures, "mesh_sha256": sha256(MESH),
              "source_materials": gltf.get("materials", []),
              "geometry_sha256": geometry_digest(vertices, faces), "triangles": len(faces),
              "max_original_corner_position_error": corner_error,
              "timings": {"startup_s": startup_s, "mesh_texture_load_s": load_s,
                          "warmup_s": warm_s, "frame": stats(samples),
                          "raster_visibility": stats(raster_samples), "texture_sampling": stats(sample_samples)},
              "timing_device": "CPU", "raster_backend": args.raster_backend}
    write_json(args.output / "metadata.json", report)
    return report


def main():
    p = common_parser(__doc__)
    p.add_argument("--filtering", choices=("nearest", "bilinear"), default="nearest")
    render(p.parse_args())


if __name__ == "__main__":
    main()
