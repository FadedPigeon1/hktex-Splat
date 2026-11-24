"""Prepare recorded mesh inputs and explicitly generate the original eigensystem.

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


def asset_paths(name):
    """Keep each recorded mesh, camera file and spectral archive independent."""
    if name == 'converse':
        return dict(mesh=MESH, spectral=SPECTRAL, cameras=CAMERAS, config=CONFIG,
                    expected_sha256=MESH_SHA256)
    assets = {
        'piggy': ('piggy_bank', '3a3007dfdaddf9c3879a0fc9f3fb8296daec7ab06218c76e5f086c46eb41b091'),
        'teapot': ('teapot', '50031dfc6e66e288dbabef834dcad078257801d8267f271f22d16f029977da6d'),
        'basketball': ('basketball', 'fb899ce585729bd3fb6fd4221099c281b38a5fc9e18dae021c7e9b453c1464cc'),
    }
    if name not in assets:
        raise ValueError(f'Unknown recorded mesh asset: {name}')
    stem, checksum = assets[name]
    mesh = ROOT / f'data/{name}/prepared/{stem}.glb'
    return dict(mesh=mesh, spectral=mesh.with_name(stem+'_eigen_albo_principal_curvatures.pt'),
                cameras=ROOT / f'data/{name}/cameras.json',
                config=ROOT / f'splatting_script/configs/{name}_training.yaml',
                expected_sha256=checksum)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def check_inputs(*, mesh=MESH, spectral=SPECTRAL, cameras=CAMERAS, config=CONFIG,
                 expected_sha256=MESH_SHA256):
    """Check required paths and exact mesh identity without loading GPU libraries."""
    problems = [f"Missing input: {p}" for p in (mesh, spectral, cameras, config) if not p.is_file()]
    if mesh.is_file() and digest(mesh) != expected_sha256:
        problems.append("Mesh checksum differs from the recorded asset")
    if cameras.is_file():
        try:
            data = json.loads(cameras.read_text())
            if not set(VIEWS) <= data.get("views", {}).keys():
                problems.append("Camera JSON is missing required recorded views")
        except (ValueError, AttributeError):
            problems.append("Camera JSON is invalid")
    return problems


def copy_mesh(source, *, mesh=MESH, expected_sha256=MESH_SHA256):
    """Copy exact bytes without replacing an existing asset or source file."""
    if digest(source) != expected_sha256:
        raise ValueError("This is not the recorded GLB. Obtain the matching source; "
                         "do not rename an arbitrary mesh into the benchmark input.")
    if mesh.exists():
        if digest(mesh) != expected_sha256:
            raise FileExistsError(f"Refusing to replace a different mesh: {mesh}")
        return
    mesh.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mesh-", dir=mesh.parent) as directory:
        temporary = Path(directory) / mesh.name
        shutil.copyfile(source, temporary)
        os.link(temporary, mesh)  # Atomic creation; fail rather than overwrite.


def precompute(*, mesh_path=MESH, spectral=SPECTRAL, config_path=CONFIG):
    """Generate the configured original spectral grid without training a model."""
    if spectral.exists():
        if not spectral.is_file():
            raise FileExistsError(f"Spectral archive path is not a file: {spectral}")
        print(f"Existing archive preserved: {spectral}; run the GPU correctness gate before use")
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
    config = OmegaConf.load(config_path)
    source = OmegaConf.to_container(config.trainer.eigen_albo, resolve=True)
    options = {f.name: source[f.name] for f in fields(EigenAlboInterpolation.Config) if f.name in source}
    mesh = Mesh.from_trimesh(load_mesh(str(mesh_path), show=False,
        merge_tex=config.data.merge_tex, bake_vert_colors=config.data.bake_vert_colours_if_textured),
        device="cuda:0")
    # The original writer saves beside cfg.mesh_path. Compute in a private
    # directory so interruption cannot leave a partial final archive.
    with tempfile.TemporaryDirectory(prefix=".spectral-", dir=mesh_path.parent) as directory:
        temporary_mesh = Path(directory) / mesh_path.name
        shutil.copyfile(mesh_path, temporary_mesh)
        options.update(mesh_path=str(temporary_mesh), use_precomputed=True, error_if_not_precomputed=False)
        EigenAlboInterpolation(options, mesh)
        archive = temporary_mesh.with_name(spectral.name)
        if not archive.is_file():
            raise RuntimeError("Original preprocessing did not create the expected spectral archive")
        os.link(archive, spectral)
    print(f"Spectral archive created: {spectral}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', choices=('converse','piggy','teapot','basketball'), default='converse')
    parser.add_argument("--mesh", type=Path, help="Downloaded matching GLB; original file is preserved")
    parser.add_argument("--check", action="store_true", help="CPU-only input/path/checksum checks")
    parser.add_argument("--precompute", action="store_true", help="Explicitly generate the spectral archive; potentially expensive")
    parser.add_argument("--execute", action="store_true", help="Apply preparation; otherwise print a preview")
    args = parser.parse_args(argv)
    paths = asset_paths(args.asset)
    if args.check:
        if args.execute or args.precompute or args.mesh:
            parser.error("--check is a separate read-only operation")
        problems = check_inputs(**paths)
        for problem in problems:
            print(problem)
        if not problems:
            print("Inputs present and mesh checksum matches; spectral compatibility still requires the GPU gate")
        return int(bool(problems))
    source = args.mesh.expanduser().resolve() if args.mesh else paths['mesh']
    if not source.is_file():
        parser.error(f"Mesh file does not exist: {source}")
    if not args.execute:
        print(json.dumps(dict(mesh_source=str(source), mesh_target=str(paths['mesh']),
            spectral_target=str(paths['spectral']), expected_mesh_sha256=paths['expected_sha256'],
            precompute=args.precompute, gpu_started=False), indent=2))
        return 0
    copy_mesh(source, mesh=paths['mesh'], expected_sha256=paths['expected_sha256'])
    print(f"Prepared mesh: {paths['mesh']}")
    if args.precompute:
        precompute(mesh_path=paths['mesh'], spectral=paths['spectral'], config_path=paths['config'])
    else:
        print("No spectral computation performed. Use --precompute --execute to generate the archive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
