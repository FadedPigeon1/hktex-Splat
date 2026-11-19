"""Prepare exact Converse inputs and explicitly generate the original eigensystem.

Asset/preprocessing CLI. Previews and --check are CPU-only; spectral computation
requires both --precompute and --execute and uses unchanged original HKTex code.
"""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from hktex_splatting.utils.experiment_io import ROOT, MESH, SPECTRAL, CAMERAS, VIEWS

MESH_SHA256 = "bbbd2d5a292f49c98aafa227196f7abec42349bfdf47755b39865cfb50d845cd"
CONFIG = ROOT / "splatting_script/configs/converse_training.yaml"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def check_inputs():
    """Check required paths and exact mesh identity without loading GPU libraries."""
    problems = [f"Missing input: {p}" for p in (MESH, SPECTRAL, CAMERAS, CONFIG) if not p.is_file()]
    if MESH.is_file() and digest(MESH) != MESH_SHA256:
        problems.append("Mesh checksum differs from the recorded Converse experiment")
    if CAMERAS.is_file():
        try:
            data = json.loads(CAMERAS.read_text())
            if not set(VIEWS) <= data.get("views", {}).keys():
                problems.append("Camera JSON is missing required recorded views")
        except (ValueError, AttributeError):
            problems.append("Camera JSON is invalid")
    return problems


def copy_mesh(source):
    """Copy exact bytes without replacing an existing asset or source file."""
    if digest(source) != MESH_SHA256:
        raise ValueError("This is not the recorded Converse GLB. Obtain the matching source; "
                         "do not rename an arbitrary mesh into the benchmark input.")
    if MESH.exists():
        if digest(MESH) != MESH_SHA256:
            raise FileExistsError(f"Refusing to replace a different mesh: {MESH}")
        return
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mesh-", dir=MESH.parent) as directory:
        temporary = Path(directory) / MESH.name
        shutil.copyfile(source, temporary)
        os.link(temporary, MESH)  # Atomic creation; fail rather than overwrite.


def precompute():
    """Generate the configured original spectral grid without training a model."""
    if SPECTRAL.exists():
        if not SPECTRAL.is_file():
            raise FileExistsError(f"Spectral archive path is not a file: {SPECTRAL}")
        print(f"Existing archive preserved: {SPECTRAL}; run the GPU correctness gate before use")
        return
    from hktex_splatting.utils.experiment_io import gpu_guard
    gpu_guard()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build and usable NVIDIA GPU are required")
    from omegaconf import OmegaConf
    from hktex.modules.mesh import Mesh
    from hktex.modules.eigen_albo import EigenAlboInterpolation
    from hktex.utils import load_mesh
    config = OmegaConf.load(CONFIG)
    source = OmegaConf.to_container(config.trainer.eigen_albo, resolve=True)
    options = {f.name: source[f.name] for f in fields(EigenAlboInterpolation.Config) if f.name in source}
    mesh = Mesh.from_trimesh(load_mesh(str(MESH), show=False,
        merge_tex=config.data.merge_tex, bake_vert_colors=config.data.bake_vert_colours_if_textured),
        device="cuda:0")
    # The original writer saves beside cfg.mesh_path. Compute in a private
    # directory so interruption cannot leave a partial final archive.
    with tempfile.TemporaryDirectory(prefix=".spectral-", dir=MESH.parent) as directory:
        temporary_mesh = Path(directory) / MESH.name
        shutil.copyfile(MESH, temporary_mesh)
        options.update(mesh_path=str(temporary_mesh), use_precomputed=True, error_if_not_precomputed=False)
        EigenAlboInterpolation(options, mesh)
        archive = temporary_mesh.with_name(SPECTRAL.name)
        if not archive.is_file():
            raise RuntimeError("Original preprocessing did not create the expected spectral archive")
        os.link(archive, SPECTRAL)
    print(f"Spectral archive created: {SPECTRAL}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, help="Downloaded exact Converse GLB; original file is preserved")
    parser.add_argument("--check", action="store_true", help="CPU-only input/path/checksum checks")
    parser.add_argument("--precompute", action="store_true", help="Explicitly generate the spectral archive; potentially expensive")
    parser.add_argument("--execute", action="store_true", help="Apply preparation; otherwise print a preview")
    args = parser.parse_args(argv)
    if args.check:
        if args.execute or args.precompute or args.mesh:
            parser.error("--check is a separate read-only operation")
        problems = check_inputs()
        for problem in problems:
            print(problem)
        if not problems:
            print("Inputs present and mesh checksum matches; spectral compatibility still requires the GPU gate")
        return int(bool(problems))
    source = args.mesh.expanduser().resolve() if args.mesh else MESH
    if not source.is_file():
        parser.error(f"Mesh file does not exist: {source}")
    if not args.execute:
        print(json.dumps(dict(mesh_source=str(source), mesh_target=str(MESH),
            spectral_target=str(SPECTRAL), expected_mesh_sha256=MESH_SHA256,
            precompute=args.precompute, gpu_started=False), indent=2))
        return 0
    copy_mesh(source)
    print(f"Prepared mesh: {MESH}")
    if args.precompute:
        precompute()
    else:
        print("No spectral computation performed. Use --precompute --execute to generate the archive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
