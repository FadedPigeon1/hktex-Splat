"""Evaluate the original trained HKTex field at visible pixel centers.

Native pointwise rendering reference; uses original prepare/diffuse/forward calls.
Main API: render, validate_knn_settings, main."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from hktex_splatting.utils.reference_assets import (
    CACHE,
    EXPERIMENT,
    ROOT,
    camera,
    check_repetitions,
    common_parser,
    geometry_digest,
    raster_surface,
    require_gpu_allocation,
    runtime_cache_environment,
    save_npy,
    save_png,
    save_surface,
    sha256,
    stats,
    write_json,
)


def validate_knn_settings(experiment, outer_k, inner_k):
    # Preserve the original baseline's contract; explicit other experiments may
    # use their own saved KNN recipe (including the deferred HR configuration).
    if Path(experiment).resolve() == EXPERIMENT.resolve() and (outer_k, inner_k) != (50, 30):
        raise ValueError("Unexpected trained KNN settings")


def render(args):
    started = time.perf_counter()
    check_repetitions(args)
    experiment = getattr(args, "experiment", EXPERIMENT)
    support_cache = getattr(args, "kernel_face_cache", CACHE)
    os.chdir(ROOT)
    runtime_cache_environment(args.output)
    require_gpu_allocation()
    import numpy as np
    import torch
    from interactive_density_knn import load_datamodule_and_trainer_only
    from hktex.modules.heat_kernel_texture_knn import HeatKernelTextureKNN
    from hktex.modules.eigen_albo_knn import EigenAlboInterpolationKNN
    import_s = time.perf_counter() - started
    c = camera(args.view, args.resolution, args.cameras)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    sync = torch.cuda.synchronize
    sync(); t = time.perf_counter()
    cfg, _, trainer = load_datamodule_and_trainer_only(
        argparse.Namespace(config=str(experiment / "configs/parsed.yaml")),
        extras=["trainer.eigen_albo.error_if_not_precomputed=true",
                "optim.iters=0", "optim.save_model=false", "optim.save_logs=false"])
    sync(); reconstruction_s = time.perf_counter() - t
    if not isinstance(trainer.model, HeatKernelTextureKNN) or not isinstance(trainer.eigalbo_interp, EigenAlboInterpolationKNN):
        raise TypeError("Checkpoint configuration did not construct original KNN HKTex")
    checkpoint = experiment / "ckpts/model.pt"
    if cfg.optim.save_model_name != checkpoint.name:
        raise ValueError("Saved configuration points at a different checkpoint")
    original_vertices = trainer.mesh.verts.detach().cpu().numpy().copy()
    original_faces = trainer.mesh.faces.detach().cpu().numpy().copy()
    original_digest = geometry_digest(original_vertices, original_faces)
    sync(); t = time.perf_counter()
    trainer.model.load_torch(str(checkpoint)); trainer.model.eval()
    sync(); checkpoint_s = time.perf_counter() - t
    outer_k, inner_k = trainer.model.cfg.knn_outer_k, trainer.model.cfg.knn_inner_k
    validate_knn_settings(experiment, outer_k, inner_k)
    vertices = trainer.mesh.verts.detach().cpu().numpy()
    faces = trainer.mesh.faces.detach().cpu().numpy()
    digest = geometry_digest(vertices, faces)
    if not np.array_equal(original_faces, faces):
        raise ValueError("Checkpoint changed source mesh topology/face order")
    geometry_delta = float(np.max(np.abs(vertices-original_vertices)))
    if geometry_delta > .001:
        raise ValueError(f"Checkpoint/source geometry deviation exceeds 0.001 normalized units: {geometry_delta}")
    with np.load(support_cache, allow_pickle=False) as cache:
        identity = json.loads(str(cache["metadata_json"].item()))
    if (identity["mesh_geometry_sha256"] != digest or identity["checkpoint_sha256"] != sha256(checkpoint)
            or identity["parsed_config_sha256"] != sha256(experiment / "configs/parsed.yaml")):
        raise ValueError("Model/geometry differs from the existing splat cache")
    sync(); t = time.perf_counter()
    with torch.no_grad():
        trainer.prepare_knn(save_barycentric=False)
    sync(); preparation_s = time.perf_counter() - t

    @torch.no_grad()
    def frame():
        t = time.perf_counter()
        ids, barys, depth = raster_surface(vertices, faces, c, args.raster_backend)
        raster_s = time.perf_counter() - t
        sync(); t = time.perf_counter()
        covered = ids >= 0
        selected_ids, selected_barys = ids[covered], barys[covered]
        output = np.zeros((*ids.shape, 3), np.float32)
        values = np.empty((len(selected_ids), 3), np.float32)
        for start in range(0, len(selected_ids), args.batch_size):
            stop = min(start + args.batch_size, len(selected_ids))
            fi = torch.as_tensor(selected_ids[start:stop], device=trainer.mesh.faces.device, dtype=torch.int32)
            bc = torch.as_tensor(selected_barys[start:stop], device=trainer.mesh.verts.device, dtype=trainer.mesh.verts.dtype)
            pts = trainer.mesh.barycentric_to_cartesian(bc, trainer.mesh.get_face_vertices(fi))
            # Original inference calls, including original KNN, weights, blend,
            # normalization, mean color, clamp and optional output network.
            info = trainer.model.prepare_points(mesh=trainer.mesh,
                eigalbo_interp=trainer.eigalbo_interp, face_ids=fi, barys=None, pts=pts)
            colors, _, _, _ = trainer.model.diffuse_heat_kernels(
                eigalbo_interp=trainer.eigalbo_interp, pts_info=info)
            values[start:stop] = trainer.model(colors).detach().cpu().numpy()
        output[covered] = values
        sync(); evaluation_s = time.perf_counter() - t
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite full HKTex output")
        return output, ids, barys, depth, raster_s, evaluation_s

    try:
        warm_t = time.perf_counter()
        for _ in range(args.warmup_runs):
            result = frame()
        warm_s = time.perf_counter() - warm_t
        startup_s = time.perf_counter() - started
        samples, raster_samples, evaluation_samples = [], [], []
        for _ in range(args.benchmark_runs):
            sync(); t = time.perf_counter()
            result = frame()
            sync(); samples.append(time.perf_counter() - t)
            raster_samples.append(result[-2]); evaluation_samples.append(result[-1])
    finally:
        trainer.reset_knn()
    rgb, ids, barys, depth, _, _ = result
    save_npy(args.output / "rgb.npy", rgb)
    save_png(args.output / "rgb.png", rgb)
    save_surface(args.output, ids, barys, depth)
    report = {"method": "hktex_full", "inference": __doc__, "full_original_texture_inference": True,
              "lit_mitsuba_path_render": False, "camera_dependent_kernel_pruning": False,
              "triangle_support_restriction": False, "available_model_kernels": int(trainer.model.N_sources),
              "knn_outer_k": outer_k, "knn_inner_k": inner_k,
              "power_diffused_diracs": trainer.model.cfg.power_diffused_diracs,
              "color_space": "srgb_encoded",
              "appearance": "unlit model RGB; training was encoded source RGB/255; no extra gamma",
              "camera": c, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "geometry_sha256": digest, "gpu_model": torch.cuda.get_device_name(),
              "original_normalized_geometry_sha256": original_digest,
              "checkpoint_geometry_max_coordinate_delta": geometry_delta,
              "geometry_note": "Original checkpoint load_state_dict also loads embedded mesh buffers; compressed float16 buffers may differ slightly from normalized source GLB. Original UV uses the real normalized source geometry. Silhouette/depth alignment must be checked.",
              "timings": {"startup_s": startup_s, "imports_cuda_s": import_s,
                          "model_reconstruction_s": reconstruction_s, "checkpoint_load_s": checkpoint_s,
                          "knn_preparation_s": preparation_s, "warmup_s": warm_s,
                          "frame": stats(samples), "raster_visibility": stats(raster_samples),
                          "kernel_evaluation": stats(evaluation_samples)},
              "raster_backend": args.raster_backend,
              "timing_scope": "CPU full-mesh raster plus GPU full inference, transfers and RGB assembly; synchronized; excludes setup and file encoding"}
    write_json(args.output / "metadata.json", report)
    return report


def main():
    p = common_parser(__doc__)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--experiment", type=Path, default=EXPERIMENT,
                   help="Saved experiment to render; defaults to the original Converse baseline")
    p.add_argument("--kernel-face-cache", type=Path, default=CACHE,
                   help="Matching cache used for checkpoint/geometry identity validation only")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    render(args)


if __name__ == "__main__":
    main()
