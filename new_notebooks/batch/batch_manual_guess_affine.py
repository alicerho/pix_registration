#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import re
import nd2
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage.io import imsave
from scipy.optimize import minimize, differential_evolution


# ------------------------
# CONFIG
# ------------------------

DATASET = Path("../../data/Dataset_300Fovs")
UNMIXED = DATASET / "unmixed"
RAW = DATASET / "RAW"
OUT_ROOT = Path("batch_affine_results")

MODE = "blue"   # "blue", "red", "yellow", "green"
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
        "spectral_dir": UNMIXED / "red",
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

# --- Initial guess ---
# [dx, dy, rotation_rad, scale_x, scale_y, shear_rad]
INIT_PARAMS = [-20.0, 10.0, 0.0, 1.0, 1.0, 0.0]

# --- Global search ---
DE_BOUNDS = [
    (-60,   20),    # dx
    (-20,   40),    # dy
    (-0.05,  0.05), # rotation
    ( 0.97,  1.03), # scale_x
    ( 0.97,  1.03), # scale_y
    (-0.03,  0.03), # shear
]
DE_MAXITER = 300
DE_POPSIZE = 12
DE_SEED = 42

# --- Local polish ---
NM_XTOL = 1e-4
NM_FTOL = 1.0
NM_MAXITER = 2000

# Thresholding for score
CAM_THRESH_PCT = 95
SPEC_THRESH_PCT = 95

PROJECT_CAMERA_MAX = True
PROJECT_SPEC_MAX = True

_CENTER = None


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

def build_affine_matrix(dx, dy, rotation, scale_x, scale_y, shear):
    cx, cy = _CENTER

    T_to_origin = np.array([[1, 0, -cx],
                            [0, 1, -cy],
                            [0, 0,   1]], dtype=np.float64)

    T_back = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0,  1]], dtype=np.float64)

    T_shift = np.array([[1, 0, dx],
                        [0, 1, dy],
                        [0, 0,  1]], dtype=np.float64)

    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[ c, -s, 0],
                  [ s,  c, 0],
                  [ 0,  0, 1]], dtype=np.float64)

    S = np.array([[scale_x, 0,       0],
                  [0,       scale_y, 0],
                  [0,       0,       1]], dtype=np.float64)

    Sh = np.array([[1, np.tan(shear), 0],
                   [0, 1,             0],
                   [0, 0,             1]], dtype=np.float64)

    return T_back @ T_shift @ R @ S @ Sh @ T_to_origin

def apply_affine(img: np.ndarray, params, output_shape) -> np.ndarray:
    dx, dy, rotation, scale_x, scale_y, shear = params
    M = build_affine_matrix(dx, dy, rotation, scale_x, scale_y, shear)
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

def score_dot_product(fixed: np.ndarray, moving: np.ndarray) -> float:
    return float(np.sum(fixed * moving))

def neg_score(params, cam_bin, spec_bin, output_shape):
    warped = apply_affine(spec_bin, params, output_shape)
    return -score_dot_product(cam_bin, warped)

def print_params(label, params, score=None):
    dx, dy, rot, sx, sy, sh = params
    extra = f"  score={score:.0f}" if score is not None else ""
    print(
        f"  [{label}] dx={dx:+.3f} dy={dy:+.3f} "
        f"rot={np.degrees(rot):+.4f}deg "
        f"sx={sx:.5f} sy={sy:.5f} "
        f"sh={np.degrees(sh):+.4f}deg"
        f"{extra}"
    )

def parse_slide_field(name: str):
    m = re.search(r"slide-(\d+)_field-(\d+)", name)
    if not m:
        raise ValueError(f"Could not parse slide/field from {name}")
    return m.group(1), m.group(2)


# ------------------------
# PER-FILE PROCESSING
# ------------------------

def process_file(spec_path: Path):
    global _CENTER

    slide, field = parse_slide_field(spec_path.name)
    prefix = f"registered_EYrainbow_slide-{slide}_field-{field}"

    out_dir = OUT_ROOT / MODE / f"slide-{slide}_field-{field}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {MODE}: slide {slide}, field {field} ===")

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

    _CENTER = (cam.shape[1] / 2.0, cam.shape[0] / 2.0)

    cam_bin = make_binary_like(cam, CAM_THRESH_PCT)
    spec_bin = make_binary_like(score_img, SPEC_THRESH_PCT)
    output_shape = cam.shape

    objective = lambda p: neg_score(p, cam_bin, spec_bin, output_shape)

    # Stage 1: DE
    de_result = differential_evolution(
        objective,
        bounds=DE_BOUNDS,
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        seed=DE_SEED,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        init="sobol",
        workers=1,
        updating="deferred",
        disp=False,
    )

    de_params = de_result.x
    de_score = -de_result.fun
    print_params("DE best", de_params, de_score)

    # Stage 2: Nelder-Mead
    nm_result = minimize(
        objective,
        x0=de_params,
        method="Nelder-Mead",
        options={
            "xatol": NM_XTOL,
            "fatol": NM_FTOL,
            "maxiter": NM_MAXITER,
            "adaptive": True,
        },
    )

    best_params = nm_result.x
    best_score = -nm_result.fun
    print_params("NM best", best_params, best_score)

    # save text report
    dx, dy, rot, sx, sy, sh = best_params
    M = build_affine_matrix(dx, dy, rot, sx, sy, sh)

    with open(out_dir / "best_affine.txt", "w") as f:
        f.write(f"camera:   {cam_path}\n")
        f.write(f"spectral: {spec_path}\n\n")
        f.write(f"dx:       {dx:.4f}\n")
        f.write(f"dy:       {dy:.4f}\n")
        f.write(f"rotation: {np.degrees(rot):.6f} deg\n")
        f.write(f"scale_x:  {sx:.6f}\n")
        f.write(f"scale_y:  {sy:.6f}\n")
        f.write(f"shear:    {np.degrees(sh):.6f} deg\n")
        f.write(f"score:    {best_score:.0f}\n\n")
        f.write("Affine matrix:\n")
        for row in M:
            f.write(" ".join(f"{v:+.8f}" for v in row) + "\n")

    # apply to all channels + save tif + QC
    for name, img in channels:
        img_w = apply_affine(img, best_params, output_shape)
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
    best_warp = apply_affine(score_img, best_params, output_shape)
    best_warp_bin = apply_affine(spec_bin, best_params, output_shape)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(norm(cam), cmap="gray")
    axes[0, 0].set_title("Camera raw")
    axes[0, 1].imshow(norm(score_img), cmap="gray")
    axes[0, 1].set_title("Spectral raw")
    axes[0, 2].imshow(norm(best_warp), cmap="gray")
    axes[0, 2].set_title(
        f"Warped\n"
        f"dx={dx:.1f} dy={dy:.1f} "
        f"rot={np.degrees(rot):.3f}deg\n"
        f"sx={sx:.4f} sy={sy:.4f} sh={np.degrees(sh):.3f}deg"
    )
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
        "dx": dx,
        "dy": dy,
        "rotation_deg": float(np.degrees(rot)),
        "scale_x": float(sx),
        "scale_y": float(sy),
        "shear_deg": float(np.degrees(sh)),
        "score": float(best_score),
        "camera": str(cam_path),
        "spectral": str(spec_path),
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
        csv_path = mode_out / f"{MODE}_batch_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nSaved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()