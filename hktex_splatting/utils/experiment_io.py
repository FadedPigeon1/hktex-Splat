"""Manage portable experiment inputs, preservation checks and unique outputs.

Shared training/benchmark I/O and existing GPU allocation guards.
Main API: configure_cache, gpu_guard, unique_output, write_json."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import uuid

HERE = Path(__file__).resolve().parents[2] / "experiments/converse_heat_kernel_splatting"
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "experiments/converse_renderer_comparison"
MESH = ROOT / "data/converse/prepared/converse__free.glb"
SPECTRAL = MESH.with_name("converse__free_eigen_albo_principal_curvatures.pt")
CAMERAS = ROOT / "experiments/converse_fidelity/cameras.json"
VIEWS = ("top", "side", "three_quarter", "logo_closeup")


def install_import_paths():
    for directory in (ROOT, ROOT / "scripts", ROOT / "experiments/converse_fidelity", PREVIOUS):
        if str(directory) not in sys.path:
            sys.path.append(str(directory))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, update=False):
    """Exclusive by default; updates allowed only for this invocation's manifest."""
    path = Path(path)
    data = (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()
    if not update:
        with path.open("xb") as stream:
            stream.write(data)
        return
    fd, temporary = tempfile.mkstemp(prefix=".manifest_", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    os.replace(temporary, path)


def unique_output(prefix):
    results = Path(os.environ.get("HKTEX_RUN_ROOT", HERE / "results"))
    results.mkdir(parents=True, exist_ok=True)
    path = results / (prefix+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_")+uuid.uuid4().hex[:10])
    path.mkdir(exist_ok=False)
    return path


def configure_cache(run_name, root=Path("/tmp/hktex_heat_kernel_splatting")):
    root = Path(os.environ.get("HKTEX_CACHE_ROOT", root))
    root = Path(root).resolve()
    if root.is_relative_to(ROOT) or ROOT.is_relative_to(root):
        raise ValueError("Disposable caches must be outside the repository")
    base = root / run_name
    base.mkdir(parents=True, exist_ok=False)
    mapping = {"TORCHINDUCTOR_CACHE_DIR":"inductor", "TRITON_CACHE_DIR":"triton",
        "TORCH_EXTENSIONS_DIR":"extensions", "NUMBA_CACHE_DIR":"numba", "MPLCONFIGDIR":"mpl",
        "CUDA_CACHE_PATH":"cuda", "XDG_CACHE_HOME":"xdg", "TMPDIR":"tmp", "TMP":"tmp", "TEMP":"tmp"}
    for key, folder in mapping.items():
        (base / folder).mkdir(exist_ok=True)
        os.environ[key] = str(base / folder)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    tempfile.tempdir = None
    return base


def gpu_guard():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().lower().startswith("login"):
        raise RuntimeError("Manual execution requires an existing interactive A100 Slurm allocation")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True)
    if not result.stdout.strip() or any("A100" not in s for s in result.stdout.splitlines()):
        raise RuntimeError("This validation protocol requires an A100; no GPU is requested automatically")
    return result.stdout.strip()
