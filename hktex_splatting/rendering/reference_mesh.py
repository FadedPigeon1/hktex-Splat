"""Sample the original mesh UV texture at triangle-renderer pixel centers.

Rendering/benchmark target images; no learned kernel evaluation.
Main API: texture_sample, render, shade_material, shade_batch, main."""
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


def shade_material(albedo, normals, view_directions, *, mode='none',
                   direction=(.3, .8, .6), ambient=.1, intensity=3.,
                   roughness=.35, specular=.04, backend='auto'):
    """Shade sRGB albedo into linear RGB radiance, independently of heat kernels.

    A world-space distant light uses Lambert diffuse and optional isotropic GGX,
    separable Smith masking and Schlick Fresnel. Ambient is a diffuse fill, not
    traced indirect light. Roughness/F0 are fixed material controls, not learned.
    CUDA auto uses a first-order albedo kernel for contiguous fixed geometry;
    Torch retains the reference graph and geometry/higher-order derivatives.
    No shadows, tone mapping or display clipping; ``none`` is an exact identity.
    Directions point from surface to light/viewer. Normals are outward, one-sided.
    """
    if mode == 'none':
        return albedo
    import math
    import torch
    import torch.nn.functional as F
    if mode not in ('diffuse', 'ggx'):
        raise ValueError('Unknown lighting mode')
    if (not all(math.isfinite(x) for x in (*direction, ambient, intensity, roughness, specular))
            or len(direction) != 3 or sum(x*x for x in direction) <= 0
            or ambient < 0 or intensity < 0 or not .05 <= roughness <= 1
            or not 0 <= specular <= 1):
        raise ValueError('Invalid light or material settings')
    if albedo.ndim != 2 or albedo.shape[-1] != 3:
        raise ValueError('Expected [visible samples, 3] albedo')
    if normals.shape != albedo.shape or view_directions.shape != albedo.shape:
        raise ValueError('Lighting geometry must match albedo samples')
    if backend not in ('auto', 'torch', 'triton'):
        raise ValueError('Unknown material-lighting backend')
    normals, view_directions = normals.to(albedo), view_directions.to(albedo)
    fused = (albedo.is_cuda and albedo.dtype == torch.float32 and
             not normals.requires_grad and not view_directions.requires_grad and
             normals.is_contiguous() and view_directions.is_contiguous())
    if backend == 'triton' and not fused:
        raise ValueError('Fused lighting requires CUDA float32 albedo and contiguous fixed normal/view inputs')
    if fused and backend != 'torch':
        from hktex_splatting.rendering.splat_autograd import material_lighting
        return material_lighting(albedo, normals, view_directions, mode, direction,
                                 ambient, intensity, roughness, specular)
    n = F.normalize(normals.to(albedo), dim=-1, eps=1e-6)
    v = F.normalize(view_directions.to(albedo), dim=-1, eps=1e-6)
    l = F.normalize(albedo.new_tensor(direction), dim=-1, eps=1e-6)
    nl = (n*l).sum(-1, keepdim=True).clamp(0, 1)
    nv = (n*v).sum(-1, keepdim=True).clamp(0, 1)
    # Both branches stay finite even for the inactive low-sRGB branch.
    linear = torch.where(albedo <= .04045, albedo/12.92,
                         ((albedo+.055).clamp_min(0)/1.055).pow(2.4))
    diffuse = linear/math.pi
    highlight = 0.
    if mode == 'ggx':
        h = F.normalize(l+v, dim=-1, eps=1e-6)
        nh = (n*h).sum(-1, keepdim=True).clamp(0, 1)
        vh = (v*h).sum(-1, keepdim=True).clamp(0, 1)
        alpha2 = roughness**4  # perceptual roughness -> alpha = roughness squared
        distribution = alpha2/(math.pi*(nh.square()*(alpha2-1)+1).square())
        def masking(cosine):
            return 2*cosine/(cosine+(alpha2+(1-alpha2)*cosine.square()).sqrt())
        fresnel = specular+(1-specular)*(1-vh).pow(5)
        highlight = distribution*masking(nl)*masking(nv)*fresnel/(4*nv.clamp_min(1e-6))
        diffuse = diffuse*(1-fresnel)
    direct = diffuse*nl+highlight
    return ambient*linear+intensity*torch.where((nl > 0)&(nv > 0), direct, 0.)


def shade_batch(albedo, data, settings, *, backend=None):
    """Apply optional shared camera lighting after material RGB reconstruction."""
    if not settings or settings.get('mode', 'none') == 'none':
        return albedo
    if backend is not None:
        settings = dict(settings, backend=backend)
    return shade_material(albedo, data['lighting_normal'], data['lighting_view'], **settings)


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
