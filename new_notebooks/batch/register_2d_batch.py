import os
import numpy as np
import pandas as pd
import ants
import matplotlib.pyplot as plt
from skimage import io, measure
from scipy.spatial import cKDTree
import csv

# --------------------
# CONFIG
# --------------------
ROOT_OUT = "outputs_2d"        # same ROOT_OUT you used in register_2d_batch.py
MASK_DIR = "../../data/Masks"  # where Simon's .tif masks are

FOV_LIST = [1, 2, 3, 4, 5, 6]

# Simon-defined pairs
PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]

# ---- SUPER SIMPLE EVAL SETTINGS ----
INLIER_THR_PX = 10        # "close enough" threshold for bead-to-bead match
PASS_INLIER_PCT = 80.0    # PASS if >= this % of points are inliers
PASS_MEDIAN_PX = 5.0      # and median error <= this many pixels


# --------------------
# HELPERS
# --------------------
def find_mask_path(fov, kind, ch):
    """
    kind: "camera" or "spectral"
    ch: camera channel (DAPI/FITC/YFP/TRITC) or spectral (blue/green/yellow/red)

    Returns a path to a .tif mask in MASK_DIR.
    Handles special red variants if present (e.g. spectral-red_2).
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
    """
    Loads integer-labeled mask (0 background; 1..N beads).
    Returns numpy int array.
    """
    m = io.imread(path)
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"Mask not 2D after squeeze: {path} shape={m.shape}")
    return m.astype(np.int32)


def centroids_from_labels(label_img):
    """
    Extract centroids from integer label image.
    Returns Nx2 array in (y, x) order.
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


def nn_distances(src_yx, dst_yx):
    """
    For each point in src, compute nearest-neighbor distance to dst.
    Inputs Nx2 in (y,x).
    Returns distances array length N.
    """
    if len(src_yx) == 0 or len(dst_yx) == 0:
        return np.array([])
    tree = cKDTree(dst_yx)
    dists, _ = tree.query(src_yx, k=1)
    return dists


def apply_affine_to_points(points_yx, A2x2, t_xy):
    """
    Apply affine to points.
    points_yx: Nx2 in (y,x)
    A2x2 is linear part in (x,y) coordinates.
    t_xy translation in (x,y).
    Returns Nx2 in (y,x).
    """
    if len(points_yx) == 0:
        return points_yx

    # (y,x) -> (x,y)
    xy = points_yx[:, ::-1]
    xy2 = (xy @ A2x2.T) + t_xy
    # (x,y) -> (y,x)
    return xy2[:, ::-1]


def apply_ants_transforms_to_points(points_yx, transform_list):
    """
    Apply ANTs transforms (moving->fixed) to points.
    points_yx: Nx2 (y,x)
    transform_list: list of transform filenames in correct order
    Returns Nx2 (y,x)
    """
    if len(points_yx) == 0:
        return points_yx

    # ANTs expects columns named x,y (in that order)
    df = pd.DataFrame({"x": points_yx[:, 1], "y": points_yx[:, 0]})

    df_warp = ants.apply_transforms_to_points(
        dim=2,
        points=df,
        transformlist=transform_list,
        whichtoinvert=[False] * len(transform_list)
    )

    out = np.stack([df_warp["y"].to_numpy(), df_warp["x"].to_numpy()], axis=1)
    return out


def safe_median(d):
    return float(np.median(d)) if d is not None and len(d) > 0 else np.nan


def inlier_pct(d, thr):
    return float(np.mean(d <= thr) * 100.0) if d is not None and len(d) > 0 else np.nan


def make_dot_overlay_png(fixed_img, cam_pts_yx, pts_before_yx, pts_aff_yx, pts_final_yx,
                         out_path, title, final_label):
    """
    Creates ONE PNG with 3 panels:
      (1) camera pts vs spectral BEFORE
      (2) camera pts vs spectral AFTER affine
      (3) camera pts vs spectral AFTER final (SyN if present else affine)
    """
    fixed = np.squeeze(fixed_img)
    if fixed.ndim != 2:
        fixed = fixed.squeeze()

    def norm01(img):
        lo, hi = np.percentile(img, [1, 99])
        x = (img - lo) / (hi - lo + 1e-8)
        return np.clip(x, 0, 1)

    bg = norm01(fixed)

    # dot size heuristic
    s = max(10, int(0.03 * fixed.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        ("BEFORE", pts_before_yx),
        ("AFTER Affine", pts_aff_yx),
        (final_label, pts_final_yx),
    ]

    for ax, (lab, spec_pts) in zip(axes, panels):
        ax.imshow(bg, cmap="gray")
        if len(cam_pts_yx):
            ax.scatter(cam_pts_yx[:, 1], cam_pts_yx[:, 0],
                       s=s, c="lime", alpha=0.55, edgecolors="none")
        if len(spec_pts):
            ax.scatter(spec_pts[:, 1], spec_pts[:, 0],
                       s=s, c="red", alpha=0.45, edgecolors="none")
        ax.set_title(f"{title}\n{lab}")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------
# MAIN EVAL FOR ONE PAIR
# --------------------
def evaluate_one(fov, spec_ch, cam_ch):
    out_dir = os.path.join(ROOT_OUT, f"FOV-{fov}", f"{spec_ch}_to_{cam_ch}")
    if not os.path.isdir(out_dir):
        print(f"  -> Missing output dir, skipping: {out_dir}")
        return None

    fixed_path = os.path.join(out_dir, "fixed_camera.nii.gz")
    if not os.path.exists(fixed_path):
        print(f"  -> Missing fixed image, skipping: {fixed_path}")
        return None

    affine_txt = os.path.join(out_dir, "affine_matrix_2x2.txt")
    trans_txt  = os.path.join(out_dir, "affine_translation.txt")
    if not (os.path.exists(affine_txt) and os.path.exists(trans_txt)):
        print(f"  -> Missing affine matrix files in {out_dir}, skipping.")
        return None

    # Option A: use syn_warp.nii.gz if it exists
    syn_warp_path = os.path.join(out_dir, "syn_warp.nii.gz")

    cam_mask_path  = find_mask_path(fov, "camera", cam_ch)
    spec_mask_path = find_mask_path(fov, "spectral", spec_ch)
    if cam_mask_path is None or spec_mask_path is None:
        print(f"  -> Missing masks for FOV-{fov} {spec_ch}->{cam_ch}, skipping.")
        return None

    # load fixed image (for plotting background)
    fixed_img = ants.image_read(fixed_path).numpy()

    # load label masks + centroids
    cam_label  = load_label_mask(cam_mask_path)    # ~2048x2048
    spec_label = load_label_mask(spec_mask_path)   # ~512x512

    cam_centroids_full = centroids_from_labels(cam_label)   # (y,x) in full camera space
    spec_centroids     = centroids_from_labels(spec_label)  # (y,x) in spectral space

    # shift camera centroids into 512x512 center crop coords (registration space)
    H, W = cam_label.shape
    crop = 512
    y0 = H // 2 - crop // 2
    x0 = W // 2 - crop // 2

    cam_centroids_crop = cam_centroids_full.copy()
    cam_centroids_crop[:, 0] -= y0
    cam_centroids_crop[:, 1] -= x0

    keep = (
        (cam_centroids_crop[:, 0] >= 0) & (cam_centroids_crop[:, 0] < crop) &
        (cam_centroids_crop[:, 1] >= 0) & (cam_centroids_crop[:, 1] < crop)
    )
    cam_centroids_crop = cam_centroids_crop[keep]

    # Load affine params (A in x/y, t in x/y)
    A2x2 = np.loadtxt(affine_txt)
    t_xy = np.loadtxt(trans_txt)
    t_xy = np.array(t_xy).reshape(2,)

    # BEFORE: spec points vs camera points (both in 512-space)
    d_before = nn_distances(spec_centroids, cam_centroids_crop)

    # AFTER affine: apply affine to spec points
    spec_aff = apply_affine_to_points(spec_centroids, A2x2, t_xy)
    d_aff = nn_distances(spec_aff, cam_centroids_crop)

    # AFTER SyN: IMPORTANT — in your registration, SyN refines the *affine-warped* moving image.
    # So we apply SyN warp ON TOP OF the affine-transformed points (spec_aff), not raw spec_centroids.
    spec_syn = None
    d_syn = np.array([])
    if os.path.exists(syn_warp_path):
        try:
            spec_syn = apply_ants_transforms_to_points(spec_aff, [syn_warp_path])
            d_syn = nn_distances(spec_syn, cam_centroids_crop)
        except Exception as e:
            print(f"  -> Failed applying syn_warp.nii.gz for points: {e}")
            spec_syn = None
            d_syn = np.array([])

    # choose final: SyN if available, else affine
    if d_syn is not None and len(d_syn) > 0:
        d_final = d_syn
        final_method = "syn"
        spec_final = spec_syn
        final_label = "AFTER Affine+SyN"
    else:
        d_final = d_aff
        final_method = "affine"
        spec_final = spec_aff
        final_label = "AFTER Affine (no SyN)"

    # simple metrics
    final_inlier = inlier_pct(d_final, INLIER_THR_PX)
    final_median = safe_median(d_final)

    passed = (
        np.isfinite(final_inlier) and np.isfinite(final_median) and
        (final_inlier >= PASS_INLIER_PCT) and
        (final_median <= PASS_MEDIAN_PX)
    )

    # overlay PNG
    overlay_path = os.path.join(out_dir, "mask_centroid_overlays.png")
    title = f"FOV-{fov} {spec_ch}->{cam_ch}"
    make_dot_overlay_png(
        fixed_img=fixed_img,
        cam_pts_yx=cam_centroids_crop,
        pts_before_yx=spec_centroids,
        pts_aff_yx=spec_aff,
        pts_final_yx=spec_final,
        out_path=overlay_path,
        title=title,
        final_label=final_label
    )

    return {
        "FOV": fov,
        "pair": f"{spec_ch}->{cam_ch}",
        "final_method": final_method,
        f"final_inlier_pct_{INLIER_THR_PX}px": final_inlier,
        "median_error_px_final": final_median,
        "PASS": "YES" if passed else "NO",
    }


def main():
    rows = []

    for fov in FOV_LIST:
        for spec_ch, cam_ch in PAIRS:
            print(f"\n=== MASK EVAL (OPTION A): FOV-{fov} {spec_ch}->{cam_ch} ===")
            res = evaluate_one(fov, spec_ch, cam_ch)
            if res is not None:
                rows.append(res)

    if rows:
        out_csv = os.path.join(ROOT_OUT, "evaluate_2d_masks_simple.csv")
        keys = [
            "FOV",
            "pair",
            "final_method",
            f"final_inlier_pct_{INLIER_THR_PX}px",
            "median_error_px_final",
            "PASS",
        ]

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

        print(f"\nWrote SIMPLE mask-based evaluation CSV to: {out_csv}")
    else:
        print("\nNo mask-based evaluations ran (missing output dirs or masks).")


if __name__ == "__main__":
    main()