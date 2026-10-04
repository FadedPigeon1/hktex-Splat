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
import hashlib
import json
import math
import os
from pathlib import Path
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


def raster_visible_face_mask(mesh, args):
    """Full-mesh geometry/depth pass only; never supplies the splat color image."""
    import numpy as np
    from visualize_kernel_faces import rasterize_visible_triangles

    _, _, _, winning_faces, _ = rasterize_visible_triangles(
        mesh.vertices, mesh.faces, np.arange(len(mesh.faces)),
        args.camera_position, args.camera_look_at,
        args.image_width, args.image_height, args.fov_y,
    )
    visible_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    visible_face_mask[np.unique(winning_faces[winning_faces >= 0])] = True
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


def filter_cached_kernel_visibility(mesh, affected_sets, args):
    """Select kernels by a shared face mask from the requested visibility mode."""
    import numpy as np

    total_memberships = sum(len(affected) for affected in affected_sets)
    affected_union = (np.unique(np.concatenate(affected_sets)) if affected_sets
                      else np.empty(0, dtype=np.int64))
    determination_start = time.perf_counter()
    raster_mask = sample_mask = None
    if args.visibility_mode == "raster":
        raster_mask = raster_visible_face_mask(mesh, args)
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
            raster_mask = raster_visible_face_mask(mesh, args)
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
    return kernel_ids, visible_sets


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
    kernel_ids, visible_sets = filter_cached_kernel_visibility(mesh, affected_sets, args)
    visibility_time = time.perf_counter() - visibility_start
    visible_union = (np.unique(np.concatenate(visible_sets)) if visible_sets
                     else np.empty(0, dtype=np.int64))
    synchronize()
    selection_time = time.perf_counter() - selection_start

    raster_start = time.perf_counter()
    # Rasterize only these primitive IDs, once per unique geometric triangle.
    triangle_mask, _, _, splat_faces, splat_barys = rasterize_visible_triangles(
        vertices, faces, visible_union, args.camera_position, args.camera_look_at,
        args.image_width, args.image_height, args.fov_y,
    )
    raster_time = time.perf_counter() - raster_start
    evaluation_start = time.perf_counter()
    splat_covered = splat_faces >= 0
    # Every kernel whose visible patch contains the winning triangle can
    # contribute at its pixels. Shared geometry is not alpha-composited.
    patch_masks = [splat_covered & np.isin(splat_faces, visible_faces)
                   for visible_faces in visible_sets]
    splat_colors = evaluate_selected_kernel_colors(
        trainer, splat_faces, splat_barys, kernel_ids, args.batch_size, patch_masks
    )
    synchronize()
    evaluation_time = time.perf_counter() - evaluation_start
    total_splat_time = time.perf_counter() - splat_start

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
    if args.compare_visibility:
        print("Visibility filtering and total splat timings include optional alternate-mode validation.")
    if not use_cache:
        print(f"Slow footprint recomputation: {slow_selection_time:.6f} s")
    print(f"Triangle splat rasterization: {raster_time:.6f} s")
    print(f"Kernel evaluation/blending: {evaluation_time:.6f} s")
    print(f"Total splat render time: {total_splat_time:.6f} s")
    print(f"Total render time (splat path, excluding reference): {total_splat_time:.6f} s")
    print(f"Reference render time: {reference_time:.6f} s")


if __name__ == "__main__":
    main()
