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
ROOT_OUT = "outputs_2d"
MASK_DIR = "../../data/Masks"

FOV_LIST = [1, 2, 3, 4, 5, 6]

PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]

# ---- SIMPLE EVAL SETTINGS ----
INLIER_THR_PX = 10
PASS_INLIER_PCT = 80.0
PASS_MEDIAN_PX = 5.0


# --------------------
# HELPERS
# --------------------
def find_mask_path(fov, kind, ch):
    base = f"FOV-{fov}_{kind}-{ch}.tif"
    p = os.path.join(MASK_DIR, base)
    if os.path.exists(p):
        return p

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


def centroids_from_labels(label_img):
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
    if len(src_yx) == 0 or len(dst_yx) == 0:
        return np.array([])
    tree = cKDTree(dst_yx)
    dists, _ = tree.query(src_yx, k=1)
    return dists


def apply_affine_to_points(points_yx, A2x2, t_xy):
    if len(points_yx) == 0:
        return points_yx
    xy = points_yx[:, ::-1]           # (x,y)
    xy2 = (xy @ A2x2.T) + t_xy
    return xy2[:, ::-1]               # back to (y,x)


def apply_ants_transforms_to_points(points_yx, transform_list):
    if len(points_yx) == 0:
        return points_yx

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


def pass_fail(final_inlier, final_median):
    ok = (
        np.isfinite(final_inlier) and np.isfinite(final_median) and
        (final_inlier >= PASS_INLIER_PCT) and
        (final_median <= PASS_MEDIAN_PX)
    )
    return "YES" if ok else "NO"


def make_dot_overlay_png(fixed_img, cam_pts_yx, pts_before_yx, pts_aff_yx, pts_syn_yx,
                         out_path, title):
    fixed = np.squeeze(fixed_img)
    if fixed.ndim != 2:
        fixed = fixed.squeeze()

    def norm01(img):
        lo, hi = np.percentile(img, [1, 99])
        x = (img - lo) / (hi - lo + 1e-8)
        return np.clip(x, 0, 1)

    bg = norm01(fixed)
    s = max(10, int(0.03 * fixed.shape[0]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        ("BEFORE", pts_before_yx),
        ("AFTER Affine", pts_aff_yx),
        ("AFTER Affine+SyN", pts_syn_yx),
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
# EVAL FOR ONE PAIR (RETURNS 2 ROWS)
# --------------------
def evaluate_one(fov, spec_ch, cam_ch):
    out_dir = os.path.join(ROOT_OUT, f"FOV-{fov}", f"{spec_ch}_to_{cam_ch}")
    if not os.path.isdir(out_dir):
        print(f"  -> Missing output dir, skipping: {out_dir}")
        return []

    fixed_path = os.path.join(out_dir, "fixed_camera.nii.gz")
    affine_txt = os.path.join(out_dir, "affine_matrix_2x2.txt")
    trans_txt  = os.path.join(out_dir, "affine_translation.txt")

    if not os.path.exists(fixed_path):
        print(f"  -> Missing fixed image, skipping: {fixed_path}")
        return []
    if not (os.path.exists(affine_txt) and os.path.exists(trans_txt)):
        print(f"  -> Missing affine matrix files in {out_dir}, skipping.")
        return []

    syn_warp_path = os.path.join(out_dir, "syn_warp.nii.gz")

    cam_mask_path  = find_mask_path(fov, "camera", cam_ch)
    spec_mask_path = find_mask_path(fov, "spectral", spec_ch)
    if cam_mask_path is None or spec_mask_path is None:
        print(f"  -> Missing masks for FOV-{fov} {spec_ch}->{cam_ch}, skipping.")
        return []

    fixed_img = ants.image_read(fixed_path).numpy()

    cam_label  = load_label_mask(cam_mask_path)
    spec_label = load_label_mask(spec_mask_path)

    cam_centroids_full = centroids_from_labels(cam_label)
    spec_centroids     = centroids_from_labels(spec_label)

    # camera crop shift (512 center crop)
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

    # affine params
    A2x2 = np.loadtxt(affine_txt)
    t_xy = np.array(np.loadtxt(trans_txt)).reshape(2,)

    # BEFORE + AFFINE
    spec_aff = apply_affine_to_points(spec_centroids, A2x2, t_xy)

    d_aff = nn_distances(spec_aff, cam_centroids_crop)
    aff_inlier = inlier_pct(d_aff, INLIER_THR_PX)
    aff_median = safe_median(d_aff)

    # SyN ON TOP OF affine points (matches your registration pipeline)
    spec_syn = None
    d_syn = np.array([])
    syn_error = ""

    if os.path.exists(syn_warp_path):
        try:
            spec_syn = apply_ants_transforms_to_points(spec_aff, [syn_warp_path])
            d_syn = nn_distances(spec_syn, cam_centroids_crop)
        except Exception as e:
            syn_error = str(e)
            spec_syn = None
            d_syn = np.array([])
    else:
        syn_error = "syn_warp.nii.gz not found"

    syn_inlier = inlier_pct(d_syn, INLIER_THR_PX)
    syn_median = safe_median(d_syn)

    # Save overlay PNG (use syn points if available; else show affine in panel 3)
    overlay_path = os.path.join(out_dir, "mask_centroid_overlays.png")
    title = f"FOV-{fov} {spec_ch}->{cam_ch}"
    make_dot_overlay_png(
        fixed_img=fixed_img,
        cam_pts_yx=cam_centroids_crop,
        pts_before_yx=spec_centroids,
        pts_aff_yx=spec_aff,
        pts_syn_yx=(spec_syn if spec_syn is not None else spec_aff),
        out_path=overlay_path,
        title=title
    )

    base = {
        "FOV": fov,
        "pair": f"{spec_ch}->{cam_ch}",
        "num_cam_centroids_in_crop": int(len(cam_centroids_crop)),
        "num_spec_centroids": int(len(spec_centroids)),
    }

    row_aff = {
        **base,
        "method": "affine",
        f"inlier_pct_{INLIER_THR_PX}px": aff_inlier,
        "median_error_px": aff_median,
        "PASS": pass_fail(aff_inlier, aff_median),
        "syn_status": "",  # blank for affine
    }

    row_syn = {
        **base,
        "method": "syn",
        f"inlier_pct_{INLIER_THR_PX}px": syn_inlier,
        "median_error_px": syn_median,
        "PASS": pass_fail(syn_inlier, syn_median),
        "syn_status": "OK" if (d_syn is not None and len(d_syn) > 0) else f"FAILED: {syn_error}",
    }

    return [row_aff, row_syn]


def main():
    rows = []

    for fov in FOV_LIST:
        for spec_ch, cam_ch in PAIRS:
            print(f"\n=== MASK EVAL (AFFINE + SYN ROWS): FOV-{fov} {spec_ch}->{cam_ch} ===")
            rows.extend(evaluate_one(fov, spec_ch, cam_ch))

    if rows:
        out_csv = os.path.join(ROOT_OUT, "evaluate_2d_masks_affine_and_syn.csv")
        keys = [
            "FOV",
            "pair",
            "method",
            f"inlier_pct_{INLIER_THR_PX}px",
            "median_error_px",
            "PASS",
            "syn_status",
            "num_cam_centroids_in_crop",
            "num_spec_centroids",
        ]

        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

        print(f"\nWrote CSV with separate affine/syn rows to: {out_csv}")
    else:
        print("\nNo evaluations ran (missing output dirs or masks).")


if __name__ == "__main__":
    main()