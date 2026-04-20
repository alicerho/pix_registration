#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import re
import nd2
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage.io import imsave
from scipy.ndimage import uniform_filter


# ------------------------
# CONFIG
# ------------------------

DATASET = Path("../../data/Dataset_300Fovs")
UNMIXED = DATASET / "unmixed"
RAW = DATASET / "RAW"

# NEW output root so nothing gets overwritten
OUT_ROOT = Path("batch_affine_results_global_apply")

# folder containing the averaged/global affine matrices
GLOBAL_AFFINE_DIR = Path("batch_affine_results/global_affine_from_inliers")

MODE = "green"   # "blue", "red", "yellow", "green"
CROP_SIZE = 512

CONFIG = {
    "blue": {
        "spectral_dir": UNMIXED,
        "pattern": "unmixed_EYrainbow_slide-*_field-*_spectral-blue.nd2",
        "camera_channel": "DAPI",
        "output_names": ["px", "vo"],
        "n_channels": 2,
    },
    "red": {
        "spectral_dir": UNMIXED,
        "pattern": "unmixed_EYrainbow_slide-*_field-*_spectral-red.nd2",
        "camera_channel": "TRITC",
        "output_names": ["mt", "ld"],
        "n_channels": 2,
    },
    "yellow": {
        "spectral_dir": RAW,
        "pattern": "EYrainbow_slide-*_field-*_spectral-yellow.nd2",
        "camera_channel": "YFP",
        "output_names": ["go"],
        "n_channels": 1,
    },
    "green": {
        "spectral_dir": RAW,
        "pattern": "EYrainbow_slide-*_field-*_spectral-green.nd2",
        "camera_channel": "FITC",
        "output_names": ["er"],
        "n_channels": 1,
    },
}

cfg = CONFIG[MODE]

# Thresholding for score / figures
CAM_THRESH_PCT = 95
SPEC_THRESH_PCT = 95

PROJECT_CAMERA_MAX = True
PROJECT_SPEC_MAX = True


# ------------------------
# HELPERS
# ------------------------

def load_nd2(path: Path) -> np.ndarray:
    with nd2.ND2File(str(path)) as f:
        arr = f.asarray()
    return np.squeeze(arr).astype(np.float32)


def project_to_2d(arr: np.ndarray, use_max=True) -> np.ndarray:
    while arr.ndim > 2:
        arr = arr.max(axis=0) if use_max else arr.mean(axis=0)
    return arr


def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def norm(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) - img.min()
    mx = x.max()
    return x / mx if mx > 0 else x


def save_tif(path: Path, img: np.ndarray):
    x = norm(img)
    imsave(str(path), (x * 65535).astype(np.uint16))


def make_binary_like(img: np.ndarray, pct: float) -> np.ndarray:
    thr = np.percentile(img, pct)
    out = np.zeros_like(img, dtype=np.float32)
    out[img >= thr] = 1000.0
    return out


def overlay_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*a.shape, 3), dtype=np.float32)
    rgb[..., 0] = norm(a)
    rgb[..., 1] = norm(b)
    return rgb


def parse_slide_field(name: str):
    m = re.search(r"slide-(\d+)_field-(\d+)", name)
    if not m:
        raise ValueError(f"Could not parse slide/field from {name}")
    return m.group(1), m.group(2)


def load_global_affine(mode: str) -> tuple[np.ndarray, Path]:
    camera_map = {
        "blue": "DAPI",
        "green": "FITC",
        "yellow": "YFP",
        "red": "TRITC",
    }
    cam = camera_map[mode]
    path = GLOBAL_AFFINE_DIR / f"{mode}_to_{cam}_global_affine.txt"
    M = np.loadtxt(path)
    if M.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix in {path}, got {M.shape}")
    return M, path


def apply_affine_matrix(img: np.ndarray, M: np.ndarray, output_shape) -> np.ndarray:
    tform = AffineTransform(matrix=M)
    return warp(
        img,
        inverse_map=tform.inverse,
        output_shape=output_shape,
        preserve_range=True,
        mode="constant",
        cval=0.0,
        order=1,
    ).astype(np.float32)


# ------------------------
# PER-FILE PROCESSING
# ------------------------

def process_file(spec_path: Path):
    slide, field = parse_slide_field(spec_path.name)
    prefix = f"registered_EYrainbow_slide-{slide}_field-{field}"

    out_dir = OUT_ROOT / MODE / f"slide-{slide}_field-{field}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Applying GLOBAL affine for {MODE}: slide {slide}, field {field} ===")

    # load camera
    cam_path = RAW / f"EYrainbow_slide-{slide}_field-{field}_camera-{cfg['camera_channel']}.nd2"
    cam = load_nd2(cam_path)
    cam = project_to_2d(cam, use_max=PROJECT_CAMERA_MAX)

    # load spectral
    arr = load_nd2(spec_path)

    # choose channels
    if cfg["n_channels"] == 2:
        if arr.shape[0] == 2:
            ch1, ch2 = arr[0], arr[1]
        else:
            ch1, ch2 = arr[..., 0], arr[..., 1]
        channels = [(cfg["output_names"][0], ch1), (cfg["output_names"][1], ch2)]
        score_img = ch1
    else:
        arr = project_to_2d(arr, use_max=PROJECT_SPEC_MAX)
        channels = [(cfg["output_names"][0], arr)]
        score_img = arr

    print("camera shape before crop:", cam.shape)
    print("spectral scoring image shape:", score_img.shape)

    if cam.shape != score_img.shape:
        cam = center_crop(cam, CROP_SIZE)
        print("camera shape after crop:", cam.shape)

    if cam.shape != score_img.shape:
        raise ValueError(f"Shapes do not match: camera={cam.shape}, spectral={score_img.shape}")

    output_shape = cam.shape

    cam_bin = make_binary_like(cam, CAM_THRESH_PCT)
    spec_bin = make_binary_like(score_img, SPEC_THRESH_PCT)

    # load global affine
    M, M_path = load_global_affine(MODE)
    print(f"Using global affine: {M_path}")
    print(M)

    # save text report
    with open(out_dir / "applied_global_affine.txt", "w") as f:
        f.write(f"camera:   {cam_path}\n")
        f.write(f"spectral: {spec_path}\n")
        f.write(f"mode:     {MODE}\n")
        f.write(f"matrix:   {M_path}\n\n")
        f.write("Applied global affine matrix:\n")
        for row in M:
            f.write(" ".join(f"{v:+.8f}" for v in row) + "\n")

    # apply to all channels + save tif + QC
    for name, img in channels:
        img_w = apply_affine_matrix(img, M, output_shape)

        # ------------------------
        # scatterplot using ALL pixels
        # ------------------------
        cam_flat = cam.ravel()
        raw_flat = img.ravel()
        warp_flat = img_w.ravel()

        n_points = 50000
        if len(cam_flat) > n_points:
            idx_all = np.random.choice(len(cam_flat), n_points, replace=False)
            cam_all_plot = cam_flat[idx_all]
            raw_all_plot = raw_flat[idx_all]
            warp_all_plot = warp_flat[idx_all]
        else:
            cam_all_plot = cam_flat
            raw_all_plot = raw_flat
            warp_all_plot = warp_flat

        corr_before_all = np.corrcoef(cam_flat, raw_flat)[0, 1]
        corr_after_all = np.corrcoef(cam_flat, warp_flat)[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(cam_all_plot, raw_all_plot, s=1, alpha=0.2)
        axes[0].set_title(f"Before (all pixels)\ncorr = {corr_before_all:.3f}")
        axes[0].set_xlabel(f"Camera {cfg['camera_channel']} intensity")
        axes[0].set_ylabel(f"Spectral {name} intensity")

        axes[1].scatter(cam_all_plot, warp_all_plot, s=1, alpha=0.2)
        axes[1].set_title(f"After (all pixels)\ncorr = {corr_after_all:.3f}")
        axes[1].set_xlabel(f"Camera {cfg['camera_channel']} intensity")
        axes[1].set_ylabel(f"Warped spectral {name} intensity")

        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_scatter_before_after_{name}_all_pixels.png", dpi=200)
        plt.close()

        print(f"  Correlation before (all pixels): {corr_before_all:.4f}")
        print(f"  Correlation after  (all pixels): {corr_after_all:.4f}")

        # ------------------------
        # scatterplot using brightest 5% of camera pixels
        # ------------------------
        sig_thresh = np.percentile(cam, 95)
        sig_mask = cam >= sig_thresh

        cam_sig = cam[sig_mask]
        raw_sig = img[sig_mask]
        warp_sig = img_w[sig_mask]

        if len(cam_sig) > n_points:
            idx_sig = np.random.choice(len(cam_sig), n_points, replace=False)
            cam_sig_plot = cam_sig[idx_sig]
            raw_sig_plot = raw_sig[idx_sig]
            warp_sig_plot = warp_sig[idx_sig]
        else:
            cam_sig_plot = cam_sig
            raw_sig_plot = raw_sig
            warp_sig_plot = warp_sig

        corr_before_sig = np.corrcoef(cam_sig, raw_sig)[0, 1]
        corr_after_sig = np.corrcoef(cam_sig, warp_sig)[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(cam_sig_plot, raw_sig_plot, s=1, alpha=0.2)
        axes[0].set_title(f"Before (brightest 5% camera pixels)\ncorr = {corr_before_sig:.3f}")
        axes[0].set_xlabel(f"Camera {cfg['camera_channel']} intensity")
        axes[0].set_ylabel(f"Spectral {name} intensity")

        axes[1].scatter(cam_sig_plot, warp_sig_plot, s=1, alpha=0.2)
        axes[1].set_title(f"After (brightest 5% camera pixels)\ncorr = {corr_after_sig:.3f}")
        axes[1].set_xlabel(f"Camera {cfg['camera_channel']} intensity")
        axes[1].set_ylabel(f"Warped spectral {name} intensity")

        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_scatter_before_after_{name}_top5pct.png", dpi=200)
        plt.close()

        print(f"  Correlation before (brightest 5% camera pixels): {corr_before_sig:.4f}")
        print(f"  Correlation after  (brightest 5% camera pixels): {corr_after_sig:.4f}")

        # ------------------------
        # ADDITIONAL: intersection of top 5% camera + spectral
        # ------------------------
        cam_thresh = np.percentile(cam, 95)
        raw_thresh = np.percentile(img, 95)
        warp_thresh = np.percentile(img_w, 95)

        before_mask = (cam >= cam_thresh) & (img >= raw_thresh)
        after_mask = (cam >= cam_thresh) & (img_w >= warp_thresh)

        cam_before = cam[before_mask]
        raw_before = img[before_mask]

        cam_after = cam[after_mask]
        warp_after = img_w[after_mask]

        if len(cam_before) > n_points:
            idx_before = np.random.choice(len(cam_before), n_points, replace=False)
            cam_before_plot = cam_before[idx_before]
            raw_before_plot = raw_before[idx_before]
        else:
            cam_before_plot = cam_before
            raw_before_plot = raw_before

        if len(cam_after) > n_points:
            idx_after = np.random.choice(len(cam_after), n_points, replace=False)
            cam_after_plot = cam_after[idx_after]
            warp_after_plot = warp_after[idx_after]
        else:
            cam_after_plot = cam_after
            warp_after_plot = warp_after

        corr_before_inter = np.corrcoef(cam_before, raw_before)[0, 1]
        corr_after_inter = np.corrcoef(cam_after, warp_after)[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(cam_before_plot, raw_before_plot, s=1, alpha=0.2)
        axes[0].set_title(
            f"Before (top 5% intersection)\ncorr = {corr_before_inter:.3f}"
        )

        axes[1].scatter(cam_after_plot, warp_after_plot, s=1, alpha=0.2)
        axes[1].set_title(
            f"After (top 5% intersection)\ncorr = {corr_after_inter:.3f}"
        )

        for ax in axes:
            ax.set_xlabel(f"Camera {cfg['camera_channel']} intensity")
            ax.set_ylabel(f"Spectral {name} intensity")

        plt.tight_layout()
        plt.savefig(
            out_dir / f"{prefix}_scatter_before_after_{name}_top5pct_intersection.png",
            dpi=200
        )
        plt.close()

        print(f"  Correlation before (intersection): {corr_before_inter:.4f}")
        print(f"  Correlation after  (intersection): {corr_after_inter:.4f}")

        # ------------------------
        # ADDITIONAL: local-average scatterplot
        # ------------------------
        win = 5

        cam_local = uniform_filter(cam.astype(np.float32), size=win)
        raw_local = uniform_filter(img.astype(np.float32), size=win)
        warp_local = uniform_filter(img_w.astype(np.float32), size=win)

        cam_local_flat = cam_local.ravel()
        raw_local_flat = raw_local.ravel()
        warp_local_flat = warp_local.ravel()

        if len(cam_local_flat) > n_points:
            idx_local = np.random.choice(len(cam_local_flat), n_points, replace=False)
            cam_local_plot = cam_local_flat[idx_local]
            raw_local_plot = raw_local_flat[idx_local]
            warp_local_plot = warp_local_flat[idx_local]
        else:
            cam_local_plot = cam_local_flat
            raw_local_plot = raw_local_flat
            warp_local_plot = warp_local_flat

        corr_before_local = np.corrcoef(cam_local_flat, raw_local_flat)[0, 1]
        corr_after_local = np.corrcoef(cam_local_flat, warp_local_flat)[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(cam_local_plot, raw_local_plot, s=1, alpha=0.2)
        axes[0].set_title(f"Before (local mean {win}x{win})\ncorr = {corr_before_local:.3f}")
        axes[0].set_xlabel(f"Camera {cfg['camera_channel']} local mean")
        axes[0].set_ylabel(f"Spectral {name} local mean")

        axes[1].scatter(cam_local_plot, warp_local_plot, s=1, alpha=0.2)
        axes[1].set_title(f"After (local mean {win}x{win})\ncorr = {corr_after_local:.3f}")
        axes[1].set_xlabel(f"Camera {cfg['camera_channel']} local mean")
        axes[1].set_ylabel(f"Warped spectral {name} local mean")

        plt.tight_layout()
        plt.savefig(
            out_dir / f"{prefix}_scatter_before_after_{name}_localmean.png",
            dpi=200
        )
        plt.close()

        print(f"  Correlation before (local mean {win}x{win}): {corr_before_local:.4f}")
        print(f"  Correlation after  (local mean {win}x{win}): {corr_after_local:.4f}")

        # ------------------------
        # ADDITIONAL: top 95% of spectral pixels
        # ------------------------
        raw_thresh = np.percentile(img, 5)
        warp_thresh = np.percentile(img_w, 5)

        before_mask = img >= raw_thresh
        after_mask = img_w >= warp_thresh

        cam_before = cam[before_mask]
        raw_before = img[before_mask]

        cam_after = cam[after_mask]
        warp_after = img_w[after_mask]

        if len(cam_before) > n_points:
            idx_before = np.random.choice(len(cam_before), n_points, replace=False)
            cam_before_plot = cam_before[idx_before]
            raw_before_plot = raw_before[idx_before]
        else:
            cam_before_plot = cam_before
            raw_before_plot = raw_before

        if len(cam_after) > n_points:
            idx_after = np.random.choice(len(cam_after), n_points, replace=False)
            cam_after_plot = cam_after[idx_after]
            warp_after_plot = warp_after[idx_after]
        else:
            cam_after_plot = cam_after
            warp_after_plot = warp_after

        corr_before_spec = np.corrcoef(cam_before, raw_before)[0, 1]
        corr_after_spec = np.corrcoef(cam_after, warp_after)[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(cam_before_plot, raw_before_plot, s=1, alpha=0.2)
        axes[0].set_title(
            f"Before (top 95% spectral)\ncorr = {corr_before_spec:.3f}"
        )

        axes[1].scatter(cam_after_plot, warp_after_plot, s=1, alpha=0.2)
        axes[1].set_title(
            f"After (top 95% spectral)\ncorr = {corr_after_spec:.3f}"
        )

        for ax in axes:
            ax.set_xlabel(f"Camera {cfg['camera_channel']} intensity")
            ax.set_ylabel(f"Spectral {name} intensity")

        plt.tight_layout()
        plt.savefig(
            out_dir / f"{prefix}_scatter_before_after_{name}_top95pct_spectral.png",
            dpi=200
        )
        plt.close()

        print(f"  Correlation before (top 95% spectral): {corr_before_spec:.4f}")
        print(f"  Correlation after  (top 95% spectral): {corr_after_spec:.4f}")

        # ------------------------
        # ADDITIONAL: local averages + top 50% of spectral intensity
        # ------------------------
        win = 5

        cam_local = uniform_filter(cam.astype(np.float32), size=win)
        raw_local = uniform_filter(img.astype(np.float32), size=win)
        warp_local = uniform_filter(img_w.astype(np.float32), size=win)

        raw_thresh = np.percentile(raw_local, 50)
        warp_thresh = np.percentile(warp_local, 50)

        before_mask = raw_local >= raw_thresh
        after_mask = warp_local >= warp_thresh

        cam_before = cam_local[before_mask]
        raw_before = raw_local[before_mask]

        cam_after = cam_local[after_mask]
        warp_after = warp_local[after_mask]

        if len(cam_before) > n_points:
            idx_before = np.random.choice(len(cam_before), n_points, replace=False)
            cam_before_plot = cam_before[idx_before]
            raw_before_plot = raw_before[idx_before]
        else:
            cam_before_plot = cam_before
            raw_before_plot = raw_before

        if len(cam_after) > n_points:
            idx_after = np.random.choice(len(cam_after), n_points, replace=False)
            cam_after_plot = cam_after[idx_after]
            warp_after_plot = warp_after[idx_after]
        else:
            cam_after_plot = cam_after
            warp_after_plot = warp_after

        corr_before_local_spec50 = np.corrcoef(cam_before, raw_before)[0, 1]
        corr_after_local_spec50 = np.corrcoef(cam_after, warp_after)[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(cam_before_plot, raw_before_plot, s=1, alpha=0.2)
        axes[0].set_title(
            f"Before (local mean, top 50% spectral)\ncorr = {corr_before_local_spec50:.3f}"
        )

        axes[1].scatter(cam_after_plot, warp_after_plot, s=1, alpha=0.2)
        axes[1].set_title(
            f"After (local mean, top 50% spectral)\ncorr = {corr_after_local_spec50:.3f}"
        )

        for ax in axes:
            ax.set_xlabel(f"Camera {cfg['camera_channel']} local mean")
            ax.set_ylabel(f"Spectral {name} local mean")

        plt.tight_layout()
        plt.savefig(
            out_dir / f"{prefix}_scatter_before_after_{name}_localmean_top50pct_spectral.png",
            dpi=200
        )
        plt.close()

        print(f"  Correlation before (local mean, top 50% spectral): {corr_before_local_spec50:.4f}")
        print(f"  Correlation after  (local mean, top 50% spectral): {corr_after_local_spec50:.4f}")

        # ------------------------
        # ADDITIONAL: local averages + top 30% of spectral intensity
        # ------------------------
        win = 5

        cam_local = uniform_filter(cam.astype(np.float32), size=win)
        raw_local = uniform_filter(img.astype(np.float32), size=win)
        warp_local = uniform_filter(img_w.astype(np.float32), size=win)

        raw_thresh = np.percentile(raw_local, 70)
        warp_thresh = np.percentile(warp_local, 70)

        before_mask = raw_local >= raw_thresh
        after_mask = warp_local >= warp_thresh

        cam_before = cam_local[before_mask]
        raw_before = raw_local[before_mask]

        cam_after = cam_local[after_mask]
        warp_after = warp_local[after_mask]

        if len(cam_before) > n_points:
            idx_before = np.random.choice(len(cam_before), n_points, replace=False)
            cam_before_plot = cam_before[idx_before]
            raw_before_plot = raw_before[idx_before]
        else:
            cam_before_plot = cam_before
            raw_before_plot = raw_before

        if len(cam_after) > n_points:
            idx_after = np.random.choice(len(cam_after), n_points, replace=False)
            cam_after_plot = cam_after[idx_after]
            warp_after_plot = warp_after[idx_after]
        else:
            cam_after_plot = cam_after
            warp_after_plot = warp_after

        corr_before_local_spec30 = np.corrcoef(cam_before, raw_before)[0, 1]
        corr_after_local_spec30 = np.corrcoef(cam_after, warp_after)[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(cam_before_plot, raw_before_plot, s=1, alpha=0.2)
        axes[0].set_title(
            f"Before (local mean, top 30% spectral)\ncorr = {corr_before_local_spec30:.3f}"
        )

        axes[1].scatter(cam_after_plot, warp_after_plot, s=1, alpha=0.2)
        axes[1].set_title(
            f"After (local mean, top 30% spectral)\ncorr = {corr_after_local_spec30:.3f}"
        )

        for ax in axes:
            ax.set_xlabel(f"Camera {cfg['camera_channel']} local mean")
            ax.set_ylabel(f"Spectral {name} local mean")

        plt.tight_layout()
        plt.savefig(
            out_dir / f"{prefix}_scatter_before_after_{name}_localmean_top30pct_spectral.png",
            dpi=200
        )
        plt.close()

        print(f"  Correlation before (local mean, top 30% spectral): {corr_before_local_spec30:.4f}")
        print(f"  Correlation after  (local mean, top 30% spectral): {corr_after_local_spec30:.4f}")

        # ------------------------
        # ADDITIONAL: local averages + top 40% of spectral intensity
        # ------------------------
        win = 5

        cam_local = uniform_filter(cam.astype(np.float32), size=win)
        raw_local = uniform_filter(img.astype(np.float32), size=win)
        warp_local = uniform_filter(img_w.astype(np.float32), size=win)

        raw_thresh = np.percentile(raw_local, 60)
        warp_thresh = np.percentile(warp_local, 60)

        before_mask = raw_local >= raw_thresh
        after_mask = warp_local >= warp_thresh

        cam_before = cam_local[before_mask]
        raw_before = raw_local[before_mask]

        cam_after = cam_local[after_mask]
        warp_after = warp_local[after_mask]

        if len(cam_before) > n_points:
            idx_before = np.random.choice(len(cam_before), n_points, replace=False)
            cam_before_plot = cam_before[idx_before]
            raw_before_plot = raw_before[idx_before]
        else:
            cam_before_plot = cam_before
            raw_before_plot = raw_before

        if len(cam_after) > n_points:
            idx_after = np.random.choice(len(cam_after), n_points, replace=False)
            cam_after_plot = cam_after[idx_after]
            warp_after_plot = warp_after[idx_after]
        else:
            cam_after_plot = cam_after
            warp_after_plot = warp_after

        corr_before_local_spec40 = np.corrcoef(cam_before, raw_before)[0, 1]
        corr_after_local_spec40 = np.corrcoef(cam_after, warp_after)[0, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(cam_before_plot, raw_before_plot, s=1, alpha=0.2)
        axes[0].set_title(
            f"Before (local mean, top 40% spectral)\ncorr = {corr_before_local_spec40:.3f}"
        )

        axes[1].scatter(cam_after_plot, warp_after_plot, s=1, alpha=0.2)
        axes[1].set_title(
            f"After (local mean, top 40% spectral)\ncorr = {corr_after_local_spec40:.3f}"
        )

        for ax in axes:
            ax.set_xlabel(f"Camera {cfg['camera_channel']} local mean")
            ax.set_ylabel(f"Spectral {name} local mean")

        plt.tight_layout()
        plt.savefig(
            out_dir / f"{prefix}_scatter_before_after_{name}_localmean_top40pct_spectral.png",
            dpi=200
        )
        plt.close()

        print(f"  Correlation before (local mean, top 40% spectral): {corr_before_local_spec40:.4f}")
        print(f"  Correlation after  (local mean, top 40% spectral): {corr_after_local_spec40:.4f}")

        save_tif(out_dir / f"{prefix}_spectral-{name}.tif", img_w)

        # QC PNGs
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(norm(cam), cmap="gray")
        axes[0].set_title(f"Camera {cfg['camera_channel']}")
        axes[1].imshow(norm(img), cmap="gray")
        axes[1].set_title(f"Raw {name}")
        axes[2].imshow(norm(img_w), cmap="gray")
        axes[2].set_title(f"Warped {name}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_side_by_side_{name}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_rgb(cam, img_w))
        plt.title(f"Overlay: {cfg['camera_channel']} vs {name}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_overlay_{name}.png", dpi=200)
        plt.close()

    # summary figure based on scoring image
    best_warp = apply_affine_matrix(score_img, M, output_shape)
    best_warp_bin = apply_affine_matrix(spec_bin, M, output_shape)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(norm(cam), cmap="gray")
    axes[0, 0].set_title("Camera raw")
    axes[0, 1].imshow(norm(score_img), cmap="gray")
    axes[0, 1].set_title("Spectral raw")
    axes[0, 2].imshow(norm(best_warp), cmap="gray")
    axes[0, 2].set_title("Warped (global affine)")
    axes[1, 0].imshow(overlay_rgb(cam, score_img))
    axes[1, 0].set_title("Overlay before")
    axes[1, 1].imshow(overlay_rgb(cam, best_warp))
    axes[1, 1].set_title("Overlay after")
    axes[1, 2].imshow(overlay_rgb(cam_bin, best_warp_bin))
    axes[1, 2].set_title("Thresholded overlap")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "registration_debug.png", dpi=200)
    plt.close()

    residual = norm(cam) - norm(best_warp)
    plt.figure(figsize=(6, 5))
    plt.imshow(residual, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    plt.colorbar(label="camera - warped spectral")
    plt.title("Residual map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_map.png", dpi=200)
    plt.close()

    return {
        "slide": slide,
        "field": field,
        "mode": MODE,
        "camera": str(cam_path),
        "spectral": str(spec_path),
        "global_affine_file": str(M_path),
    }


# ------------------------
# MAIN
# ------------------------

def main():
    mode_out = OUT_ROOT / MODE
    mode_out.mkdir(parents=True, exist_ok=True)

    files = sorted(cfg["spectral_dir"].glob(cfg["pattern"]))
    print(f"MODE={MODE}")
    print(f"Found {len(files)} files")

    summaries = []
    for spec_path in files:
        try:
            row = process_file(spec_path)
            summaries.append(row)
        except Exception as e:
            print(f"ERROR on {spec_path.name}: {e}")

    if summaries:
        import csv
        csv_path = mode_out / f"{MODE}_global_apply_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nSaved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()