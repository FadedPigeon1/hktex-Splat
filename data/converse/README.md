# Converse assets

Upload the Converse shoe ZIP or its complete extracted directory to:

```text
data/converse/raw/
```

Keep the archive and extracted assets here with their original names and directory
relationships. The uploaded original is now `raw/converse__free.glb`. HKTex
supports it directly; `prepared/converse__free.glb` is a byte-identical copy.
`prepared/textures/base_color.jpg` contains its embedded 8192 x 8192 JPEG bytes,
extracted unchanged for inspection. The prepared GLB still uses its embedded
texture, so it requires no external material or image files.

The current model is already triangulated and needs no format conversion or
decimation. Preserve the originals in `raw/`. `prepared/textures/` is available for external diffuse maps,
but use the paths required by the prepared mesh's material references; embedded
textures can remain embedded. Do not relocate maps without updating and checking
those references. Preserve UV seams, material assignments, and texture resolution.

Assets and generated precomputations here are ignored by Git. Only this README
and the directory placeholders are intended to be tracked.

From the repository root, `python scripts/setup_converse_experiment.py --check`
checks the current layout without modifying assets. See the
[main README](../../README.md#data-and-preprocessing) and
[training protocol](../../docs/HKTEX_RESEARCH_TRAINING.md) for the current
configuration and comparisons. `scripts/inspect_converse.py` retains the
inspection/preparation workflow.
