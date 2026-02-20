#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import ants
import nd2
from skimage.transform import warp, AffineTransform
from skimage.io import imsave

# --------------------
# CONFIG (match your reg_eval / dataset layout)
# --------------------
DATA_DIR = "../../data/2025-11-05_Registration2D"  # ND2 folder (same as your other scripts)
ROOT_OUT = "outputs_2d"                            # where registration outputs live

FOV_LIST = [1, 2, 3, 4, 5, 6]
PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]
CROP_SIZE = 512

# If ND2 has Z or extra dims, how to project:
CAMERA_PROJECT = "max"  # "max" or "mean"
SPECTRAL_PROJECT = "sum"  # "sum" (your README said blue/red are summed) or "max"/"mean"

# --------------------
# Helpers
# --------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def center_crop(img2d, crop_size):
    h, w = img2d.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img2d[y0:y0 + crop_size, x0:x0 + crop_size]

def load_nd2_as_float(path):
    with nd2.ND2File(path) as f:
        arr = f.asarray()
    arr = np.squeeze(arr)
    return arr.astype(np.float32)

def project_to_2d(arr, mode="max", axis=0):
    """Project arr to 2D if it has extra dimensions."""
    if arr.ndim == 2:
        return arr
    if mode == "max":
        return np.max(arr, axis=axis)
    if mode == "mean":
        return np.mean(arr, axis=axis)
    if mode == "sum":
        return np.sum(arr, axis=axis)
    raise ValueError(f"Unknown projection mode: {mode}")

def load_camera_raw(fov, cam_ch):
    path = os.path.join(DATA_DIR, f"FOV-{fov}_camera-{cam_ch}.nd2")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = load_nd2_as_float(path)

    # If has Z or channel dims, collapse them
    # Common shapes you’ve seen: (2044,2048) or (z,2044,2048)
    if arr.ndim == 3:
        arr = project_to_2d(arr, mode=CAMERA_PROJECT, axis=0)
    elif arr.ndim > 3:
        # last resort: collapse everything except last 2 dims
        while arr.ndim > 2:
            arr = project_to_2d(arr, mode=CAMERA_PROJECT, axis=0)

    return arr

def load_spectral_raw(fov, spec_ch):
    path = os.path.join(DATA_DIR, f"FOV-{fov}_spectral-{spec_ch}.nd2")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = load_nd2_as_float(path)

    # Common shape you saw: (5,512,512) -> sum over axis 0
    if arr.ndim == 3:
        arr = project_to_2d(arr, mode=SPECTRAL_PROJECT, axis=0)
    elif arr.ndim > 3:
        while arr.ndim > 2:
            arr = project_to_2d(arr, mode=SPECTRAL_PROJECT, axis=0)

    return arr

def norm01(img, p_lo=1, p_hi=99):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p_lo, p_hi])
    out = (img - lo) / (hi - lo + 1e-8)
    return np.clip(out, 0, 1)

def save_tif(path, img):
    # Save float as 16-bit for reasonable TIFF viewing
    img = img.astype(np.float32)
    x = img - np.min(img)
    if np.max(x) > 0:
        x = x / np.max(x)
    x16 = (x * 65535).astype(np.uint16)
    imsave(path, x16)

def find_affine_3x3(pair_dir):
    """
    Match your existing outputs: prefer the mask_affine_mutualNN_ransac output,
    then fallback to pair root.
    """
    p1 = os.path.join(pair_dir, "mask_affine_mutualNN_ransac", "affine_matrix_3x3.txt")
    p2 = os.path.join(pair_dir, "affine_matrix_3x3.txt")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

def load_affine_3x3(path):
    # Some people accidentally save with brackets/commas; this handles most cases.
    try:
        A = np.loadtxt(path)
        A = np.array(A, dtype=float)
    except Exception:
        # fallback: try cleaning lines
        with open(path, "r") as f:
            lines = [ln.strip().replace(",", " ") for ln in f if ln.strip()]
        A = np.array([[float(x) for x in ln.split()] for ln in lines], dtype=float)
    if A.shape != (3, 3):
        raise ValueError(f"Affine matrix not 3x3: got {A.shape} from {path}")
    return A

def overlay_rgb(fixed, other):
    """fixed red, other green."""
    fixed_n = norm01(fixed)
    other_n = norm01(other)
    rgb = np.zeros((fixed.shape[0], fixed.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = fixed_n
    rgb[..., 1] = other_n
    return rgb

# --------------------
# Core: apply saved affine to raw spectral
# --------------------
def apply_one(fov, spec_ch, cam_ch):
    pair_dir = os.path.join(ROOT_OUT, f"FOV-{fov}", f"{spec_ch}_to_{cam_ch}")
    aff_path = find_affine_3x3(pair_dir)
    if aff_path is None:
        print(f"  -> No affine found for FOV-{fov} {spec_ch}->{cam_ch} (skipping)")
        return None

    out_dir = ensure_dir(os.path.join(pair_dir, "applied_affine_output"))

    # Load raw images
    cam_full = load_camera_raw(fov, cam_ch)
    cam_crop = center_crop(cam_full, CROP_SIZE)

    spec_raw = load_spectral_raw(fov, spec_ch)

    # If spectral isn't already 512x512, resample to 512x512 (rare, but safe)
    if spec_raw.shape != (CROP_SIZE, CROP_SIZE):
        spec_ants = ants.from_numpy(spec_raw.astype(np.float32))
        spec_ants_rs = ants.resample_image(
            spec_ants, resample_params=(CROP_SIZE, CROP_SIZE),
            use_voxels=True, interp_type=1
        )
        spec_raw = spec_ants_rs.numpy().astype(np.float32)

    # Load affine and warp
    A_h = load_affine_3x3(aff_path)

    # skimage AffineTransform expects params that map (x,y)->(x',y')
    # and warp() uses inverse_map (output->input), so we pass tform.inverse
    tform = AffineTransform(matrix=A_h)

    spec_warp = warp(
        spec_raw.astype(np.float32),
        inverse_map=tform.inverse,
        output_shape=cam_crop.shape,
        order=1,
        mode="constant",
        cval=0.0,
        preserve_range=True
    ).astype(np.float32)

    # Save TIFs
    save_tif(os.path.join(out_dir, "camera_crop.tif"), cam_crop)
    save_tif(os.path.join(out_dir, "spectral_raw.tif"), spec_raw)
    save_tif(os.path.join(out_dir, "spectral_warped_to_camera.tif"), spec_warp)

    # Save PNGs: side-by-side + overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(norm01(cam_crop), cmap="gray")
    axes[0].set_title(f"Camera (cropped) {cam_ch}")
    axes[1].imshow(norm01(spec_raw), cmap="gray")
    axes[1].set_title(f"Spectral raw {spec_ch}")
    axes[2].imshow(norm01(spec_warp), cmap="gray")
    axes[2].set_title("Spectral warped to camera")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "side_by_side.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(overlay_rgb(cam_crop, spec_warp))
    ax.set_title("Overlay (red=camera, green=warped spectral)")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlay_fixed_vs_warped.png"), dpi=200)
    plt.close(fig)

    # Quick numeric sanity check (not “the evaluation”, just a sanity metric)
    cam_n = norm01(cam_crop)
    warp_n = norm01(spec_warp)
    corr = np.corrcoef(cam_n.ravel(), warp_n.ravel())[0, 1]

    # Write per-pair row csv
    row = {
        "FOV": fov,
        "spectral": spec_ch,
        "camera": cam_ch,
        "affine_path": aff_path,
        "out_dir": out_dir,
        "corr_norm_intensity": float(corr),
    }
    with open(os.path.join(out_dir, "summary_row.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    return row

def main():
    rows = []
    for fov in FOV_LIST:
        for spec_ch, cam_ch in PAIRS:
            print(f"\n=== Apply saved affine to RAW: FOV-{fov} {spec_ch}->{cam_ch} ===")
            try:
                r = apply_one(fov, spec_ch, cam_ch)
                if r is not None:
                    rows.append(r)
            except Exception as e:
                print(f"  -> ERROR: {e}")
                import traceback
                traceback.print_exc()

    if rows:
        out_csv = os.path.join(ROOT_OUT, "applied_affine_output_summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Wrote global summary: {out_csv}")
        print(f"✓ Completed {len(rows)} runs")
    else:
        print("\n✗ No outputs created (missing affine files / missing ND2s).")

if __name__ == "__main__":
    main()