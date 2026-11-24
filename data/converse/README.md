# Converse experiment assets

The current comparison reproduces a specific textured mesh. Download
[Converse — FREE by Yaroslav Karas (@yakaras)](https://sketchfab.com/3d-models/converse-free-bc791439d26f4dd9974819c168e12385)
from its original Sketchfab model page and obtain the GLB version. The source
page labels it Creative Commons Attribution; the recorded GLB identifies
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain the model's
original author/source/license attribution when sharing derived research assets.

Mesh files and generated spectral data are not bundled with the repository.
Provider exports may change; the exact benchmark file has SHA256:

```text
bbbd2d5a292f49c98aafa227196f7abec42349bfdf47755b39865cfb50d845cd
```

If a new download has a different checksum, obtain the matching original GLB
from the provider or experiment maintainer before claiming exact reproduction.
The preparation command rejects a different file instead of silently changing
geometry or textures. Other meshes need their own configuration and cameras;
renaming an arbitrary mesh to the Converse filename is insufficient.

## Prepare the mesh

With the environment installed and the repository root as your working directory:

```bash
# Inspect the plan: no copies, preprocessing or GPU work.
python -B -m hktex_splatting.prepare_assets --mesh /path/to/converse__free.glb

# Copy the verified bytes to the configured input location.
python -B -m hktex_splatting.prepare_assets --mesh /path/to/converse__free.glb --execute
```

The downloaded source remains untouched. You may store it in `data/converse/raw/`
or elsewhere. The prepared mesh is `data/converse/prepared/converse__free.glb`.
Its texture is embedded, so extracting `textures/base_color.jpg` is unnecessary
for training/rendering. No conversion, resampling, normalization rewrite or
triangle decimation is performed during copying. Existing different files are
never overwritten.

## Generate the spectral archive once

This step uses a CUDA GPU, can take substantial time, and performs no training:

```bash
python -B -m hktex_splatting.prepare_assets --precompute --execute
```

It loads the mesh using the original HKTex loader and generates the configured
Laplace–Beltrami spectral grid with the original `EigenAlboInterpolation` code.
The settings come from `splatting_script/configs/converse_training.yaml` (256
eigenvectors, principal-curvature frames, recorded anisotropy and angle grids).
The final archive is:

```text
data/converse/prepared/converse__free_eigen_albo_principal_curvatures.pt
```

Computation occurs in a private temporary directory beside the prepared mesh;
the final archive is published only after the original writer succeeds. Existing
archives are preserved. A forced process kill can leave that temporary directory;
inspect it before manually removing it. No archive is automatically regenerated
by the renderer. The preprocessing command only runs when explicitly given both
`--precompute` and `--execute`.

## Check and validate

```bash
python -B -m hktex_splatting.prepare_assets --check
python -B -m splatting_script.experiments.compare_backends \
  --backend hybrid --kernels 5000 --resolution 32 --iterations 1 \
  --support-selector fused --validate-only --execute
```

The CPU check verifies paths, required camera views and mesh checksum; it does
not establish numerical compatibility of a supplied spectral archive. The GPU
gate checks the actual rendering/gradient path. Regenerated eigensystems may
vary numerically across solver/software environments; use the recorded archive
and matching environment when exact historical numerical reproduction is needed.

All asset files are ignored by Git. The guide, raw/prepared placeholders and
`experiments/converse_fidelity/cameras.json` remain source-controlled.
For the main workflow, see the [HKS run guide](../../hktex_splatting/README.md)
and [training protocol](../../docs/HKTEX_RESEARCH_TRAINING.md).
