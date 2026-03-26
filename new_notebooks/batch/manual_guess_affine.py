#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import nd2
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from scipy.optimize import minimize, differential_evolution


# ------------------------
# CONFIG: EDIT THESE
# ------------------------

CAM_PATH  = Path("../../data/Dataset_300Fovs/RAW/EYrainbow_slide-19_field-2_camera-FITC.nd2")
SPEC_PATH = Path("../../data/Dataset_300Fovs/RAW/EYrainbow_slide-19_field-2_spectral-green.nd2")

OUT_DIR   = Path("manual_affine_debug")
CROP_SIZE = 512

# --- Initial guess (from previous translation-only result if you have it) ---
# [dx, dy, rotation_rad, scale_x, scale_y, shear_rad]
INIT_PARAMS = [
    -20.0,   # dx        (pixels)
     10.0,   # dy        (pixels)
      0.0,   # rotation  (radians)  ~0.017 rad per degree
      1.0,   # scale_x
      1.0,   # scale_y
      0.0,   # shear     (radians)
]

# --- Stage 1: Differential evolution (global search) ---
# Bounds: [min, max] for each of the 6 parameters above
DE_BOUNDS = [
    (-60,   20),    # dx
    (-20,   40),    # dy
    (-0.05,  0.05), # rotation  (~±3 degrees)
    ( 0.97,  1.03), # scale_x
    ( 0.97,  1.03), # scale_y
    (-0.03,  0.03), # shear
]
DE_MAXITER  = 300
DE_POPSIZE  = 12    # population = popsize * len(params) = 72 individuals
DE_SEED     = 42

# --- Stage 2: Nelder-Mead polish ---
NM_XTOL    = 1e-4
NM_FTOL    = 1.0
NM_MAXITER = 2000

# Thresholding percentiles for scoring masks
CAM_THRESH_PCT  = 95
SPEC_THRESH_PCT = 95

PROJECT_CAMERA_MAX = True
PROJECT_SPEC_MAX   = True


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
    """
    Build a full 3x3 affine matrix composed as:
        T(back) @ T(shift) @ R @ S @ Sh @ T(to_origin)

    All non-translation transforms are applied around the image center,
    so rotation and scale don't produce a spurious translation offset.
    """
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

    Sh = np.array([[1,            np.tan(shear), 0],
                   [0,            1,             0],
                   [0,            0,             1]], dtype=np.float64)

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
    score_str = f"  score={score:.0f}" if score is not None else ""
    print(f"  [{label}]  dx={dx:+.3f}  dy={dy:+.3f}  "
          f"rot={np.degrees(rot):+.4f}deg  "
          f"sx={sx:.5f}  sy={sy:.5f}  sh={np.degrees(sh):+.4f}deg"
          + score_str)


# ------------------------
# MAIN
# ------------------------

_CENTER = None  # set after image is loaded


def main():
    global _CENTER
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    cam  = load_nd2(CAM_PATH)
    spec = load_nd2(SPEC_PATH)

    cam  = project_to_2d(cam,  use_max=PROJECT_CAMERA_MAX)
    spec = project_to_2d(spec, use_max=PROJECT_SPEC_MAX)

    print("camera shape before crop:", cam.shape)
    print("spectral shape:", spec.shape)

    if cam.shape != spec.shape:
        cam = center_crop(cam, CROP_SIZE)
        print("camera shape after crop:", cam.shape)

    if cam.shape != spec.shape:
        raise ValueError(f"Shapes do not match: camera={cam.shape}, spectral={spec.shape}")

    # Image center — used to decouple rotation/scale from translation
    _CENTER = (cam.shape[1] / 2.0, cam.shape[0] / 2.0)  # (cx, cy)
    print(f"Image center: {_CENTER}")

    output_shape = cam.shape
    cam_bin  = make_binary_like(cam,  CAM_THRESH_PCT)
    spec_bin = make_binary_like(spec, SPEC_THRESH_PCT)

    objective = lambda p: neg_score(p, cam_bin, spec_bin, output_shape)

    # -------------------------------------------------------
    # Stage 1: Differential evolution — global search
    # -------------------------------------------------------
    print("\n=== Stage 1: Differential evolution (global search) ===")
    print(f"  population={DE_POPSIZE * len(INIT_PARAMS)}  max_iter={DE_MAXITER}")

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
        disp=True,
    )

    de_params = de_result.x
    de_score  = -de_result.fun
    print_params("DE best", de_params, de_score)

    # -------------------------------------------------------
    # Stage 2: Nelder-Mead — sub-pixel / sub-milliradian polish
    # -------------------------------------------------------
    print("\n=== Stage 2: Nelder-Mead polish ===")

    nm_result = minimize(
        objective,
        x0=de_params,
        method="Nelder-Mead",
        options={
            "xatol":    NM_XTOL,
            "fatol":    NM_FTOL,
            "maxiter":  NM_MAXITER,
            "adaptive": True,   # scales simplex to 6-D better than default
        },
    )

    best_params = nm_result.x
    best_score  = -nm_result.fun
    print_params("NM best", best_params, best_score)
    print(f"  converged={nm_result.success}  iters={nm_result.nit}  msg={nm_result.message}")

    # -------------------------------------------------------
    # Final warp
    # -------------------------------------------------------
    best_warp     = apply_affine(spec,     best_params, output_shape)
    best_warp_bin = apply_affine(spec_bin, best_params, output_shape)

    # -------------------------------------------------------
    # Save text results
    # -------------------------------------------------------
    dx, dy, rot, sx, sy, sh = best_params
    M = build_affine_matrix(dx, dy, rot, sx, sy, sh)

    with open(OUT_DIR / "best_affine.txt", "w") as f:
        f.write(f"camera:   {CAM_PATH}\n")
        f.write(f"spectral: {SPEC_PATH}\n\n")
        f.write("=== Final affine parameters ===\n")
        f.write(f"dx:        {dx:.4f}  pixels\n")
        f.write(f"dy:        {dy:.4f}  pixels\n")
        f.write(f"rotation:  {np.degrees(rot):.5f}  degrees\n")
        f.write(f"scale_x:   {sx:.6f}\n")
        f.write(f"scale_y:   {sy:.6f}\n")
        f.write(f"shear:     {np.degrees(sh):.5f}  degrees\n")
        f.write(f"score:     {best_score:.0f}\n\n")
        f.write("=== Affine matrix (3x3, forward) ===\n")
        for row in M:
            f.write("  " + "  ".join(f"{v:+.8f}" for v in row) + "\n")
        f.write(f"\n=== DE stage ===\n")
        f.write(f"converged: {de_result.success}  iters: {de_result.nit}\n")
        f.write(f"score:     {de_score:.0f}\n\n")
        f.write(f"=== Nelder-Mead stage ===\n")
        f.write(f"converged: {nm_result.success}  iters: {nm_result.nit}\n")

    # -------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(norm(cam), cmap="gray")
    axes[0, 0].set_title("Camera raw")

    axes[0, 1].imshow(norm(spec), cmap="gray")
    axes[0, 1].set_title("Spectral raw")

    axes[0, 2].imshow(norm(best_warp), cmap="gray")
    axes[0, 2].set_title(
        f"Spectral warped\n"
        f"dx={dx:.1f} dy={dy:.1f} rot={np.degrees(rot):.3f}deg\n"
        f"sx={sx:.4f} sy={sy:.4f} sh={np.degrees(sh):.3f}deg"
    )

    axes[1, 0].imshow(overlay_rgb(cam, spec))
    axes[1, 0].set_title("Overlay before")

    axes[1, 1].imshow(overlay_rgb(cam, best_warp))
    axes[1, 1].set_title("Overlay after")

    axes[1, 2].imshow(overlay_rgb(cam_bin, best_warp_bin))
    axes[1, 2].set_title("Thresholded overlap")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "registration_debug.png", dpi=200)
    plt.close()

    # Residual error map — should look like uniform noise, not a spatial gradient
    residual = norm(cam) - norm(best_warp)
    plt.figure(figsize=(6, 5))
    plt.imshow(residual, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    plt.colorbar(label="camera - warped spectral")
    plt.title("Residual map\n(uniform noise = good; spatial gradient = incomplete correction)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "residual_map.png", dpi=200)
    plt.close()

    print(f"\nSaved outputs to: {OUT_DIR}")
    print(f"\nFinal parameters:")
    print_params("result", best_params, best_score)


if __name__ == "__main__":
    main()