#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, measure


# --------------------
# CONFIG (edit if needed)
# --------------------
ROOT_OUT = "outputs_2d"          # same as your pipeline
MASK_DIR = "../../data/Masks"    # Simon's .tif masks

FOV_LIST = [1, 2, 3, 4, 5, 6]
PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]

CROP_SIZE = 512

# Mutual NN settings for residual arrows (AFTER affine)
MUTUAL_MAX_DIST_PX = 30.0

# Arrow display knobs
ARROW_SCALE = 1.0          # 1.0 means dx,dy are in pixel units
ARROW_WIDTH = 0.003
MAX_ARROWS = 800           # downsample for visibility if too many


# --------------------
# I/O helpers
# --------------------
def find_mask_path(fov, kind, ch):
    """
    kind: "camera" or "spectral"
    ch: camera channel (DAPI/FITC/YFP/TRITC) or spectral (blue/green/yellow/red)
    """
    base = f"FOV-{fov}_{kind}-{ch}.tif"
    p = os.path.join(MASK_DIR, base)
    if os.path.exists(p):
        return p

    # Special-case: sometimes spectral-red has suffixes in masks (red_2)
    if kind == "spectral" and ch == "red":
        for alt in [
            f"FOV-{fov}_spectral-red_1.tif",
            f"FOV-{fov}_spectral-red_2.tif",
            f"FOV-{fov}_spectral-red-2.tif",
        ]:
            p2 = os.path.join(MASK_DIR, alt)
            if os.path.exists(p2):
                return p2

    return None


def load_label_mask(path):
    m = io.imread(path)
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"Mask not 2D after squeeze: {path} shape={m.shape}")
    return m.astype(np.int32)


def center_crop(img, crop_size):
    h, w = img.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


# --------------------
# Mask -> points
# --------------------
def centroids_from_label_mask(label_img):
    """
    label_img: integer label image (0 background, 1..N objects)
    returns: Nx2 array in (y,x)
    """
    props = measure.regionprops(label_img)
    cents = []
    for p in props:
        if p.label == 0:
            continue
        y, x = p.centroid
        cents.append((y, x))
    if len(cents) == 0:
        return np.zeros((0, 2), dtype=float)
    return np.array(cents, dtype=float)


# --------------------
# Affine application (skimage convention)
# --------------------
def apply_affine_3x3_to_points_yx(points_yx, A_h):
    """
    points_yx: Nx2 (y,x)
    A_h: 3x3 mapping in (x,y) coords: [x'; y'; 1] = A_h @ [x; y; 1]
    returns Nx2 (y,x)
    """
    if len(points_yx) == 0:
        return points_yx

    xy = points_yx[:, ::-1]  # (x,y)
    ones = np.ones((xy.shape[0], 1), dtype=float)
    homo = np.hstack([xy, ones])              # Nx3
    out = (homo @ A_h.T)[:, :2]               # Nx2 in (x',y')
    return out[:, ::-1]                       # back to (y',x')


# --------------------
# Mutual NN (after affine)
# --------------------
def mutual_nn_pairs_yx(src_yx, dst_yx, max_dist_px):
    """
    Mutual NN pairing between src and dst.
    Inputs/outputs in (y,x).
    Returns: src_matched_yx, dst_matched_yx, dists
    """
    from scipy.spatial import cKDTree

    if len(src_yx) == 0 or len(dst_yx) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([])

    tree_dst = cKDTree(dst_yx)
    d_s2d, idx_s2d = tree_dst.query(src_yx, k=1)

    tree_src = cKDTree(src_yx)
    d_d2s, idx_d2s = tree_src.query(dst_yx, k=1)

    src_keep, dst_keep, d_keep = [], [], []
    for si, di in enumerate(idx_s2d):
        if idx_d2s[di] == si and d_s2d[si] <= max_dist_px:
            src_keep.append(src_yx[si])
            dst_keep.append(dst_yx[di])
            d_keep.append(d_s2d[si])

    if len(src_keep) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([])

    return np.array(src_keep, float), np.array(dst_keep, float), np.array(d_keep, float)


# --------------------
# Plot residual arrows
# --------------------
def plot_residual_quiver(background_bin, src_yx, dst_yx, out_png, title):
    """
    background_bin: 2D bool/0-1 for backdrop
    src_yx: matched transformed spectral points (y,x)
    dst_yx: matched camera points (y,x)
    arrows from src -> dst
    """
    if len(src_yx) == 0:
        # still write something so you know it ran
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(background_bin.astype(float), cmap="gray")
        ax.set_title(title + "\n(no mutual pairs found)")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    dx = dst_yx[:, 1] - src_yx[:, 1]
    dy = dst_yx[:, 0] - src_yx[:, 0]

    # optional downsample if too many arrows
    n = len(dx)
    if n > MAX_ARROWS:
        idx = np.random.choice(n, size=MAX_ARROWS, replace=False)
        src_yx = src_yx[idx]
        dx = dx[idx]
        dy = dy[idx]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(background_bin.astype(float), cmap="gray")

    X = src_yx[:, 1]
    Y = src_yx[:, 0]
    VISUAL_GAIN = 10.0  # or 20.0

    ax.quiver(
        X, Y, dx * VISUAL_GAIN, dy * VISUAL_GAIN,
        angles="xy",
        scale_units="xy",
        scale=1,             # keep in pixel units after gain
        width=0.003,         # thinner looks cleaner
        headwidth=4,
        headlength=5,
        headaxislength=4,
        color="cyan",
        alpha=0.9
    )

    ax.set_title(title + f"\n(mutual pairs: {len(X)})")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def summarize_residuals(src_yx, dst_yx):
    if len(src_yx) == 0:
        return {
            "num_pairs": 0,
            "mean_residual_px": np.nan,
            "median_residual_px": np.nan,
            "p90_residual_px": np.nan,
            "max_residual_px": np.nan,
            "mean_dx": np.nan,
            "mean_dy": np.nan,
        }

    dx = dst_yx[:, 1] - src_yx[:, 1]
    dy = dst_yx[:, 0] - src_yx[:, 0]
    r = np.sqrt(dx**2 + dy**2)

    return {
        "num_pairs": int(len(r)),
        "mean_residual_px": float(np.mean(r)),
        "median_residual_px": float(np.median(r)),
        "p90_residual_px": float(np.percentile(r, 90)),
        "max_residual_px": float(np.max(r)),
        "mean_dx": float(np.mean(dx)),
        "mean_dy": float(np.mean(dy)),
    }


# --------------------
# One pair runner
# --------------------
def residuals_one(fov, spec_ch, cam_ch):
    pair_dir = os.path.join(ROOT_OUT, f"FOV-{fov}", f"{spec_ch}_to_{cam_ch}")
    if not os.path.isdir(pair_dir):
        return None

    # Prefer 3x3 if you have it (your mask-affine script writes it)
    affine_3x3_path = os.path.join(pair_dir, "mask_affine_mutualNN_ransac", "affine_matrix_3x3.txt")
    if not os.path.exists(affine_3x3_path):
        # fallback: maybe you stored it directly in the pair folder
        affine_3x3_path = os.path.join(pair_dir, "affine_matrix_3x3.txt")
    if not os.path.exists(affine_3x3_path):
        print(f"  -> Missing affine 3x3 for FOV-{fov} {spec_ch}->{cam_ch}")
        return None

    cam_mask_path = find_mask_path(fov, "camera", cam_ch)
    spec_mask_path = find_mask_path(fov, "spectral", spec_ch)
    if cam_mask_path is None or spec_mask_path is None:
        print(f"  -> Missing masks for FOV-{fov} {spec_ch}->{cam_ch}")
        return None

    # Load masks
    cam_full = load_label_mask(cam_mask_path)     # big
    spec = load_label_mask(spec_mask_path)        # 512-ish

    cam = center_crop(cam_full, CROP_SIZE)        # match your registration space

    cam_pts = centroids_from_label_mask(cam)      # (y,x)
    spec_pts = centroids_from_label_mask(spec)    # (y,x)

    if len(cam_pts) < 3 or len(spec_pts) < 3:
        print(f"  -> Too few centroids: cam={len(cam_pts)} spec={len(spec_pts)}")
        return None

    A_h = np.loadtxt(affine_3x3_path)

    # Transform spectral points into camera crop space
    spec_aff = apply_affine_3x3_to_points_yx(spec_pts, A_h)

    # Mutual NN pairing AFTER affine
    src_m, dst_m, d = mutual_nn_pairs_yx(spec_aff, cam_pts, MUTUAL_MAX_DIST_PX)

    # Output folder (keep next to the pair)
    out_dir = os.path.join(pair_dir, "residual_vectors_after_affine")
    os.makedirs(out_dir, exist_ok=True)

    # Plot arrows over camera binary as background
    bg = (cam > 0)
    out_png = os.path.join(out_dir, "residual_quiver.png")
    title = f"FOV-{fov} {spec_ch}->{cam_ch} residuals after affine"
    plot_residual_quiver(bg, src_m, dst_m, out_png, title)

    stats = summarize_residuals(src_m, dst_m)
    stats.update({
        "FOV": fov,
        "spectral": spec_ch,
        "camera": cam_ch,
        "mutual_max_dist_px": MUTUAL_MAX_DIST_PX,
        "out_dir": out_dir
    })

    # Write per-pair CSV
    per_csv = os.path.join(out_dir, "residual_stats.csv")
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats.keys()))
        w.writeheader()
        w.writerow(stats)

    return stats


def main():
    rows = []
    for fov in FOV_LIST:
        for spec_ch, cam_ch in PAIRS:
            print(f"\n=== Residual vectors: FOV-{fov} {spec_ch}->{cam_ch} ===")
            r = residuals_one(fov, spec_ch, cam_ch)
            if r is not None:
                rows.append(r)

    if rows:
        out_csv = os.path.join(ROOT_OUT, "residual_vectors_after_affine_summary.csv")
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Wrote summary CSV: {out_csv}")
    else:
        print("\nNo residual plots created (missing affine/masks/output dirs).")


if __name__ == "__main__":
    main()