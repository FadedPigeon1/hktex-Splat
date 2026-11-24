"""Evaluate foreground RGB fidelity with missing pixels counted as black.

Benchmark image-quality evaluation and reporting.
Main API: metrics, validate_alignment, evaluate, main."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, uniform_filter
from hktex_splatting.utils.reference_assets import read_json, save_png, srgb_to_linear, write_json

PAIRS = (("uv_hktex", "original_uv", "hktex_full"),
         ("hktex_splat", "hktex_full", "triangle_splat"),
         ("uv_splat", "original_uv", "triangle_splat"))


def ssim_foreground(a, b, mask, window=7):
    """Wang-style uniform-window RGB SSIM; sample covariance, range=1.

    Only centers whose entire 7x7 window is in reference foreground/ROI enter
    the average. A missing candidate pixel never removes a reference window.
    Returns None when no valid window exists. This excludes background context.
    """
    valid = minimum_filter(mask.astype(np.uint8), size=window, mode="constant", cval=0).astype(bool)
    if not valid.any():
        return None
    size = (window, window, 1)
    ma, mb = uniform_filter(a, size=size), uniform_filter(b, size=size)
    factor = window * window / (window * window - 1)
    va = np.maximum(0, (uniform_filter(a*a, size=size)-ma*ma)*factor)
    vb = np.maximum(0, (uniform_filter(b*b, size=size)-mb*mb)*factor)
    cov = (uniform_filter(a*b, size=size)-ma*mb)*factor
    field = ((2*ma*mb+.01**2)*(2*cov+.03**2) /
             ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))
    return float(field[valid].mean())


def metrics(a, b, foreground, candidate_mask):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    foreground, candidate_mask = np.asarray(foreground, bool), np.asarray(candidate_mask, bool)
    if a.shape != b.shape or a.shape != (*foreground.shape, 3) or candidate_mask.shape != foreground.shape:
        raise ValueError("RGB and mask shapes must match")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1:
        raise ValueError("Metrics require finite RGB in [0,1]")
    count = int(foreground.sum())
    if not count:
        raise ValueError("Empty reference foreground/ROI")
    # Coverage is independent of RGB: a genuine black texel is still covered.
    b = b.copy(); b[~candidate_mask] = 0
    error = a[foreground] - b[foreground]
    mse = float(np.mean(error**2)); shared = foreground & candidate_mask
    shared_mse = float(np.mean((a[shared]-b[shared])**2)) if shared.any() else None
    return {"reference_foreground_pixels": count,
            "covered_reference_pixels": int(shared.sum()),
            "missing_pixels": int((foreground & ~candidate_mask).sum()),
            "extra_pixels": int((candidate_mask & ~foreground).sum()),
            "foreground_coverage_percent": 100*float(shared.sum())/count,
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(mse)),
            "psnr_db": "inf" if mse == 0 else float(-10*np.log10(mse)),
            "ssim": ssim_foreground(a, b, foreground),
            "ssim_valid_window_centers": int(minimum_filter(foreground.astype(np.uint8), size=7, mode="constant", cval=0).sum()),
            "common_covered_mae": float(np.abs(a[shared]-b[shared]).mean()) if shared.any() else None,
            "common_covered_rmse": float(np.sqrt(shared_mse)) if shared_mse is not None else None,
            "common_covered_psnr_db": ("inf" if shared_mse == 0 else float(-10*np.log10(shared_mse))) if shared_mse is not None else None,
            "common_covered_ssim": ssim_foreground(a, b, shared) if shared.any() else None}


def load_rgb(view, method):
    # These are direct RGB exports, never difference maps or display screenshots.
    return np.asarray(Image.open(view / f"{method}.png").convert("RGB"), dtype=np.float64) / 255


def roi_mask(shape, box):
    height, width = shape
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"Invalid normalized ROI: {box}")
    mask = np.zeros(shape, bool)
    mask[int(y0*height):int(np.ceil(y1*height)), int(x0*width):int(np.ceil(x1*width))] = True
    return mask


def validate_alignment(view):
    """Compare full geometry and surface buffers; splat holes are not alignment failures."""
    uv, hk, splat = [read_json(view / "methods" / name / "metadata.json")
                     for name in ("original_uv", "hktex_full", "triangle_splat")]
    if not (uv["camera"] == hk["camera"] == splat["camera"]):
        raise ValueError("Camera metadata mismatch")
    if uv["geometry_sha256"] != hk["original_normalized_geometry_sha256"]:
        raise ValueError("UV geometry differs from HKTex's normalized source GLB")
    if hk["checkpoint_geometry_max_coordinate_delta"] > .001:
        raise ValueError("Checkpoint/source geometry deviation is too large")
    udir, hdir = [view / "methods" / n for n in ("original_uv", "hktex_full")]
    uf, hf = [np.load(d / "face_ids.npy") for d in (udir, hdir)]
    ud, hd = [np.load(d / "depth.npy") for d in (udir, hdir)]
    visible, trained_visible = uf >= 0, hf >= 0
    union, shared = visible | trained_visible, visible & trained_visible
    iou = float(shared.sum()/union.sum()) if union.any() else 0
    same_face = shared & (uf == hf)
    same_face_fraction = float(same_face.sum()/shared.sum()) if shared.any() else 0
    depth_error = np.abs(ud[same_face]-hd[same_face])
    depth_p999 = float(np.quantile(depth_error, .999)) if len(depth_error) else None
    if iou < .995 or same_face_fraction < .95 or depth_p999 is None or depth_p999 > .005:
        raise ValueError(f"UV/HKTex alignment failed: IoU={iou}, same-face={same_face_fraction}, depth-p99.9={depth_p999}")
    return {"passed": True, "camera_identical": True, "normalized_source_geometry_identical": True,
            "checkpoint_geometry_max_coordinate_delta": hk["checkpoint_geometry_max_coordinate_delta"],
            "silhouette_iou": iou, "silhouette_disagreement_pixels": int((visible ^ trained_visible).sum()),
            "shared_pixel_face_agreement_fraction": same_face_fraction,
            "matched_face_depth_absolute_error_p999": depth_p999,
            "thresholds": {"minimum_silhouette_iou": .995, "minimum_shared_face_agreement": .95, "maximum_depth_error_p999": .005},
            "foreground_pixels": int(visible.sum()),
            "splat_note": "Same CLI camera and cache geometry identity; missing splat pixels are reported as coverage errors"}


def evaluate(run, rois_path=None, difference_scale=5.0, alignment_reviewed=False):
    run = Path(run)
    if not alignment_reviewed:
        raise ValueError("Inspect the actual RGB renders and alignment overlays first, then use --alignment-reviewed")
    if rois_path is None:
        rois_path = run / "rois.json"
    if difference_scale <= 0 or not np.isfinite(difference_scale):
        raise ValueError("Difference scale must be finite and positive")
    if (run / "metrics.json").exists() or (run / "metrics.csv").exists():
        raise FileExistsError("Metrics already exist; previous measurements will not be replaced")
    rois = read_json(rois_path)["views"]
    rows, alignment = [], {}
    views = sorted(p for p in (run / "views").iterdir() if p.is_dir())
    if not views:
        raise ValueError("No completed views")
    for view in views:
        alignment[view.name] = validate_alignment(view)
        # Visual contact sheet must have been generated from these files first.
        if not (view / "alignment_overlay.png").is_file():
            raise ValueError("Generate the silhouette alignment overlay before fidelity metrics")
        images = {n: load_rgb(view, n) for n in ("original_uv", "hktex_full", "triangle_splat")}
        masks = {n: np.asarray(Image.open(view / (n + "_mask.png"))) > 0 for n in images}
        for label, reference, candidate in PAIRS:
            a, b = images[reference], images[candidate]
            foreground = masks[reference]
            regions = {"foreground": np.ones(foreground.shape, bool)}
            regions.update({name: roi_mask(foreground.shape, box) for name, box in rois.get(view.name, {}).items()})
            for region_name, region in regions.items():
                region_foreground = foreground & region
                if not region_foreground.any():
                    raise ValueError(f"Empty ROI {view.name}/{region_name}")
                for space in ("srgb_encoded", "linear_rgb"):
                    aa, bb = (a, b) if space == "srgb_encoded" else (srgb_to_linear(a), srgb_to_linear(b))
                    m = metrics(aa, bb, region_foreground, masks[candidate] & region)
                    rows.append({"view": view.name, "comparison": label, "region": region_name,
                                 "color_space": space, "input_precision": "8-bit direct RGB exports for all three methods", **m})
            diff = np.abs(a-b)
            diff[~foreground] = 0
            save_png(view / f"diff_{label}.png", np.clip(diff*difference_scale, 0, 1))
    write_json(run / "metrics.json", {"metrics": rows, "alignment": alignment,
               "difference_map": {"operation": "clip(abs(encoded_RGB_A - encoded_RGB_B) * scale, 0, 1)", "scale": difference_scale},
               "alignment_visual_review_asserted": True,
               "policy": "Primary errors on all reference foreground including missing candidate pixels as black; extra pixels separate. SSIM 7x7 uniform window, only windows entirely in reference/ROI. All methods equally quantized; float UV/HKTex arrays retained."})
    with (run / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--rois", type=Path, help="Defaults to the run's saved ROI definitions")
    p.add_argument("--difference-scale", type=float, default=5)
    p.add_argument("--alignment-reviewed", action="store_true", help="Assert that you inspected this run's direct RGB renders and alignment overlays")
    args = p.parse_args()
    evaluate(args.run, args.rois, args.difference_scale, args.alignment_reviewed)


if __name__ == "__main__":
    main()
