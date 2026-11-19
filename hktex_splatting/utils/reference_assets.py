"""Load fixed mesh/camera inputs and save aligned reference-render artifacts.

Shared preprocessing/rendering/benchmark I/O; import does not initialize CUDA.
Main API: load_cpu_mesh, raster_surface, camera, camera_basis."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[2] / "experiments/converse_fidelity"
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True


MESH = ROOT / "data/converse/prepared/converse__free.glb"
EXPERIMENT = ROOT / "outputs/uv-texture-fitting/converse_fidelity"
CACHE = ROOT / "outputs/triangle_splat/converse_cutoff_0.005.npz"


def read_json(path):
    return json.loads(Path(path).read_text())


def glb_json(path=MESH):
    import struct
    with Path(path).open("rb") as stream:
        magic, version, total = struct.unpack("<4sII", stream.read(12))
        size, kind = struct.unpack("<II", stream.read(8))
        if magic != b"glTF" or version != 2 or kind != 0x4e4f534a:
            raise ValueError("Expected GLB 2.0 with a JSON first chunk")
        return json.loads(stream.read(size))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_npy(path, array):
    import numpy as np
    with Path(path).open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def save_png(path, array):
    import numpy as np
    from PIL import Image
    array = np.asarray(array)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif array.dtype != np.uint8:
        array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    with Path(path).open("xb") as stream:
        Image.fromarray(array).save(stream, format="PNG")


def stats(samples):
    median = statistics.median(samples)
    return {"samples_s": samples, "median_s": median,
            "std_s": statistics.pstdev(samples), "min_s": min(samples),
            "max_s": max(samples), "fps_from_median": 1 / median if median else None}


def camera(view, resolution=512, path=HERE / "cameras.json"):
    import numpy as np
    data = read_json(path)
    c = dict(data["views"][view])
    c.update(width=resolution, height=resolution, view=view,
             projection="perspective", pixel_sampling="center", background="black")
    if resolution not in (512, 1024):
        raise ValueError("Comparison resolution must be 512 or 1024")
    if not 0 < c["fov_y_degrees"] < 180:
        raise ValueError("Invalid FOV")
    origin, basis = camera_basis(c)
    if not np.isfinite(basis).all() or not np.allclose(basis.T @ basis, np.eye(3)):
        raise ValueError("Invalid camera basis")
    c["up"] = basis[:, 1].tolist()
    return c


def camera_basis(c):
    from hktex_splatting.rendering.camera_raster import perspective_camera_basis
    return perspective_camera_basis(c["position"], c["target"])


def common_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--view", default="top")
    p.add_argument("--resolution", type=int, choices=(512, 1024), default=512)
    p.add_argument("--cameras", type=Path, default=HERE / "cameras.json")
    p.add_argument("--output", type=Path, required=True, help="New, nonexistent method directory")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--benchmark-runs", type=int, default=5)
    p.add_argument("--raster-backend", choices=("numpy", "numba"), default="numba")
    return p


def load_cpu_mesh(path=MESH):
    """Execute the unmodified loader function without importing GPU-only hktex.__init__.

    Only package logging is supplied locally. No rendering/model code is patched.
    This is the same isolation mechanism used by the existing CPU inspector.
    """
    import trimesh
    source = ROOT / "hktex/utils/mesh.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_mesh")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"trimesh": trimesh, "hktex": SimpleNamespace(info=lambda *a: None)}
    exec(compile(module, str(source), "exec"), namespace)
    mesh = namespace["load_mesh"](str(path), merge_tex=True, bake_vert_colors=False)
    if not hasattr(mesh, "original_uv"):
        raise ValueError("Loader did not preserve original UV corners")
    return mesh


def geometry_digest(vertices, faces):
    import numpy as np
    h = hashlib.sha256()
    for array in (vertices, faces):
        array = np.asarray(array)
        h.update(str((array.shape, array.dtype.str)).encode())
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def raster_surface(vertices, faces, c, backend="numba"):
    import numpy as np
    if backend == "numba":
        from hktex_splatting.rendering.camera_raster import rasterize_numba as rasterize
    else:
        from hktex_splatting.rendering.camera_raster import rasterize_visible_triangles as rasterize
    _, depth, _, ids, barys = rasterize(
        vertices, faces, np.arange(len(faces)), c["position"], c["target"],
        c["width"], c["height"], c["fov_y_degrees"])
    return ids, barys, depth


def save_surface(out, ids, barys, depth):
    save_npy(out / "face_ids.npy", ids)
    save_npy(out / "barycentrics.npy", barys)
    save_npy(out / "depth.npy", depth)
    save_png(out / "mask.png", ids >= 0)


def check_repetitions(args):
    if args.warmup_runs < 1 or args.benchmark_runs < 1:
        raise ValueError("At least one warmup and one measured frame are required")


def srgb_to_linear(rgb):
    import numpy as np
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def require_gpu_allocation():
    import socket
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().lower().startswith("login"):
        raise RuntimeError("Run on a Grace GPU node inside your existing Slurm allocation; no allocation is launched here")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated environment has no available CUDA GPU")


def runtime_cache_environment(output):
    """Keep incidental library/JIT files with the experiment, preserve CUDA selection."""
    cache = Path(output).resolve() / "runtime_cache"
    for key, name in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton"),
                      ("TORCH_EXTENSIONS_DIR", "extensions"), ("MPLCONFIGDIR", "matplotlib")):
        os.environ.setdefault(key, str(cache / name))
