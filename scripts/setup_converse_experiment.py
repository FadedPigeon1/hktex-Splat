"""Create or check the Converse experiment layout without processing assets."""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (
    "data/converse",
    "data/converse/raw",
    "data/converse/prepared",
    "data/converse/prepared/textures",
    "experiments/converse_fidelity",
)
PLACEHOLDERS = (
    "data/converse/raw/.gitkeep",
    "data/converse/prepared/.gitkeep",
    "data/converse/prepared/textures/.gitkeep",
)
DOCUMENTS = (
    "data/converse/README.md",
    "experiments/converse_fidelity/cameras.json",
    "experiments/converse_fidelity/rois.json",
    "splatting_script/configs/converse_training.yaml",
)


def setup_layout(repo_root):
    """Restore directories/placeholders; never replace an existing file."""
    for relative_path in DIRECTORIES:
        (repo_root / relative_path).mkdir(parents=True, exist_ok=True)
    for relative_path in PLACEHOLDERS:
        try:
            with (repo_root / relative_path).open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            pass


def verify_layout(repo_root):
    """Report missing or incorrectly typed scaffold entries without writes."""
    problems = []
    for relative_path in DIRECTORIES:
        if not (repo_root / relative_path).is_dir():
            problems.append(f"Missing directory or not a directory: {relative_path}")
    for relative_path in PLACEHOLDERS + DOCUMENTS:
        if not (repo_root / relative_path).is_file():
            problems.append(f"Missing file or not a file: {relative_path}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify the layout without writing files"
    )
    args = parser.parse_args()
    print(f"Upload the Converse ZIP or complete extracted assets to: {REPO_ROOT / 'data/converse/raw'}")
    try:
        if not args.check:
            setup_layout(REPO_ROOT)
    except OSError as error:
        print(f"Layout setup failed: {error}")
        return 1
    problems = verify_layout(REPO_ROOT)
    if problems:
        for problem in problems:
            print(problem)
        print("Restore missing source/configuration files from the repository if needed.")
        return 1
    print("Layout verified; existing files preserved. No asset processing was performed by this check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
