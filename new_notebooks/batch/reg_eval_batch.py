import os
import numpy as np
import matplotlib.pyplot as plt
import csv

from skimage import io, measure
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac
from scipy.spatial import cKDTree


# --------------------
# CONFIG
# --------------------
ROOT_OUT = "outputs_2d"          # base output folder
MASK_DIR = "../../data/Masks"    # where Simon's .tif masks are

FOV_LIST = [1, 2, 3, 4, 5, 6]

PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]

CROP_SIZE = 512

# ---- Matching / RANSAC knobs ----
MUTUAL_MAX_DIST_PX = 80.0     # only consider mutual NN pairs closer than this (before RANSAC)
RANSAC_RESID_THR_PX = 6.0     # inlier threshold for RANSAC (px)
RANSAC_MIN_SAMPLES = 3
RANSAC_MAX_TRIALS = 5000

# ---- Simple “did it work?” metrics ----
INLIER_THR_PX = 10.0          # a match is "good" if within 10px after affine
PASS_INLIER_PCT = 80.0        # pass if >= 80% of moving points are within 10px
PASS_MEDIAN_PX = 5.0          # and median NN error <= 5px


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
# Mask → points
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
# Matching + RANSAC affine
# --------------------
def mutual_nn_pairs(moving_yx, fixed_yx, max_dist_px):
    """
    Return mutual NN pairs between moving and fixed points.
    Inputs are Nx2 in (y,x). Returns (src_xy, dst_xy, pair_dists)
      src_xy: moving points in (x,y)
      dst_xy: fixed points in (x,y)
    """
    if len(moving_yx) == 0 or len(fixed_yx) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([])

    # KD trees in (y,x) space is fine as long as consistent
    tree_fixed = cKDTree(fixed_yx)
    d_m2f, idx_m2f = tree_fixed.query(moving_yx, k=1)

    tree_moving = cKDTree(moving_yx)
    d_f2m, idx_f2m = tree_moving.query(fixed_yx, k=1)

    mutual_src = []
    mutual_dst = []
    mutual_d = []

    for mi, fi in enumerate(idx_m2f):
        # Check mutual: moving mi's nearest fixed is fi, and fixed fi's nearest moving is mi
        if idx_f2m[fi] == mi:
            # Also gate by distance
            if d_m2f[mi] <= max_dist_px:
                m_yx = moving_yx[mi]
                f_yx = fixed_yx[fi]
                mutual_src.append((m_yx[1], m_yx[0]))  # to (x,y)
                mutual_dst.append((f_yx[1], f_yx[0]))  # to (x,y)
                mutual_d.append(d_m2f[mi])

    if len(mutual_src) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.array([])

    return np.array(mutual_src, dtype=float), np.array(mutual_dst, dtype=float), np.array(mutual_d, dtype=float)


def fit_affine_ransac(src_xy, dst_xy):
    """
    Fit affine transform mapping src_xy -> dst_xy with RANSAC.
    Returns (model, inliers_bool). model is skimage.transform.AffineTransform or None.
    """
    if len(src_xy) < RANSAC_MIN_SAMPLES or len(dst_xy) < RANSAC_MIN_SAMPLES:
        return None, None

    try:
        model_robust, inliers = ransac(
            (src_xy, dst_xy),
            AffineTransform,
            min_samples=RANSAC_MIN_SAMPLES,
            residual_threshold=RANSAC_RESID_THR_PX,
            max_trials=RANSAC_MAX_TRIALS
        )
        return model_robust, inliers
    except Exception:
        return None, None


# --------------------
# Evaluation metrics
# --------------------
def nn_distances(src_yx, dst_yx):
    """
    For each point in src, compute nearest-neighbor distance to dst.
    Inputs Nx2 in (y,x). Returns distances length N.
    """
    if len(src_yx) == 0 or len(dst_yx) == 0:
        return np.array([])
    tree = cKDTree(dst_yx)
    d, _ = tree.query(src_yx, k=1)
    return d


def inlier_pct(dists, thr):
    if dists is None or len(dists) == 0:
        return np.nan
    return float(np.mean(dists <= thr) * 100.0)


def safe_median(dists):
    if dists is None or len(dists) == 0:
        return np.nan
    return float(np.median(dists))


# --------------------
# Visualization
# --------------------
def overlay_rgb(fixed_bin, moving_bin):
    fixed = (fixed_bin > 0).astype(np.float32)
    moving = (moving_bin > 0).astype(np.float32)
    rgb = np.zeros((fixed.shape[0], fixed.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = fixed   # red
    rgb[..., 1] = moving  # green
    return rgb


def save_before_after_overlay_png(fixed_bin, moving_bin_before, moving_bin_after, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(overlay_rgb(fixed_bin, moving_bin_before))
    axes[0].set_title("BEFORE (red=fixed, green=moving)")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb(fixed_bin, moving_bin_after))
    axes[1].set_title("AFTER affine (red=fixed, green=warped moving)")
    axes[1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------
# CORE: mask-centroid mutualNN + RANSAC affine
# --------------------
def register_one_mask_affine(fov, spec_ch, cam_ch):
    """
    Uses masks to extract object centroids, builds mutual-NN correspondences,
    fits affine with RANSAC, warps the moving mask, and writes:
      - overlay_before_after.png
      - affine_matrix_3x3.txt (skimage convention in (x,y))
      - affine_matrix_2x2.txt, affine_translation.txt
      - metrics_simple.csv (per pair)
    Returns a dict row for summary CSV.
    """
    # Put outputs inside each pair folder so it’s easy to find next to the old bead outputs
    pair_dir = os.path.join(ROOT_OUT, f"FOV-{fov}", f"{spec_ch}_to_{cam_ch}")
    out_dir = os.path.join(pair_dir, "mask_affine_mutualNN_ransac")
    os.makedirs(out_dir, exist_ok=True)

    cam_mask_path = find_mask_path(fov, "camera", cam_ch)
    spec_mask_path = find_mask_path(fov, "spectral", spec_ch)
    if cam_mask_path is None or spec_mask_path is None:
        print(f"  -> Missing masks for FOV-{fov} {spec_ch}->{cam_ch}, skipping.")
        return None

    cam_label_full = load_label_mask(cam_mask_path)   # e.g. 2048x2048
    spec_label = load_label_mask(spec_mask_path)      # e.g. 512x512

    # Match registration space: camera is center-cropped to 512x512
    cam_label = center_crop(cam_label_full, CROP_SIZE)

    fixed_bin = (cam_label > 0)
    moving_bin = (spec_label > 0)

    # Centroids in (y,x) coords of the 512x512 space
    fixed_cyxc = centroids_from_label_mask(cam_label)    # (y,x)
    moving_cyxc = centroids_from_label_mask(spec_label)  # (y,x)

    if len(fixed_cyxc) < 3 or len(moving_cyxc) < 3:
        status = "TOO_FEW_OBJECTS"
        return {
            "FOV": fov,
            "pair": f"{spec_ch}->{cam_ch}",
            "status": status,
            "num_fixed_objects": int(len(fixed_cyxc)),
            "num_moving_objects": int(len(moving_cyxc)),
            "num_mutual_pairs": 0,
            "num_ransac_inliers": 0,
            f"inlier_pct_{INLIER_THR_PX:.0f}px": np.nan,
            "median_error_px": np.nan,
            "PASS": "NO",
            "out_dir": out_dir
        }

    # Build mutual NN correspondences (inlier candidates)
    src_xy, dst_xy, pair_d = mutual_nn_pairs(moving_cyxc, fixed_cyxc, MUTUAL_MAX_DIST_PX)

    if len(src_xy) < 3:
        status = "NOT_ENOUGH_MUTUAL_PAIRS"
        return {
            "FOV": fov,
            "pair": f"{spec_ch}->{cam_ch}",
            "status": status,
            "num_fixed_objects": int(len(fixed_cyxc)),
            "num_moving_objects": int(len(moving_cyxc)),
            "num_mutual_pairs": int(len(src_xy)),
            "num_ransac_inliers": 0,
            f"inlier_pct_{INLIER_THR_PX:.0f}px": np.nan,
            "median_error_px": np.nan,
            "PASS": "NO",
            "out_dir": out_dir
        }

    # Fit affine via RANSAC
    model, inliers = fit_affine_ransac(src_xy, dst_xy)

    if model is None or inliers is None or np.sum(inliers) < 3:
        status = "RANSAC_FAILED"
        return {
            "FOV": fov,
            "pair": f"{spec_ch}->{cam_ch}",
            "status": status,
            "num_fixed_objects": int(len(fixed_cyxc)),
            "num_moving_objects": int(len(moving_cyxc)),
            "num_mutual_pairs": int(len(src_xy)),
            "num_ransac_inliers": int(np.sum(inliers)) if inliers is not None else 0,
            f"inlier_pct_{INLIER_THR_PX:.0f}px": np.nan,
            "median_error_px": np.nan,
            "PASS": "NO",
            "out_dir": out_dir
        }

    # Save affine params like your older script did
    # skimage AffineTransform.params is 3x3 mapping (x,y)->(x',y')
    A_h = model.params.copy()
    A2x2 = A_h[:2, :2]
    t_xy = A_h[:2, 2]

    np.savetxt(os.path.join(out_dir, "affine_matrix_3x3.txt"), A_h)
    np.savetxt(os.path.join(out_dir, "affine_matrix_2x2.txt"), A2x2)
    np.savetxt(os.path.join(out_dir, "affine_translation.txt"), t_xy)

    # Warp the moving binary mask into fixed space (nearest-neighbor)
    # warp expects inverse mapping; model.inverse maps output->input coords
    warped_moving = warp(
        moving_bin.astype(np.float32),
        inverse_map=model.inverse,
        output_shape=fixed_bin.shape,
        order=0,
        mode="constant",
        cval=0.0,
        preserve_range=True
    ) > 0.5

    # Evaluate: NN distances from warped moving centroids to fixed centroids
    # Important: evaluate on ALL moving objects (not just mutual pairs)
    # Transform ALL moving centroids using affine model
    moving_xy_all = np.stack([moving_cyxc[:, 1], moving_cyxc[:, 0]], axis=1)  # (x,y)
    moved_xy = model(moving_xy_all)  # (x',y')
    moved_yx = np.stack([moved_xy[:, 1], moved_xy[:, 0]], axis=1)  # back to (y,x)

    d_after = nn_distances(moved_yx, fixed_cyxc)
    inlier = inlier_pct(d_after, INLIER_THR_PX)
    med = safe_median(d_after)

    passed = (
        np.isfinite(inlier) and np.isfinite(med) and
        (inlier >= PASS_INLIER_PCT) and
        (med <= PASS_MEDIAN_PX)
    )

    # Save overlay PNG (binary, crisp by design)
    overlay_path = os.path.join(out_dir, "overlay_before_after.png")
    save_before_after_overlay_png(
        fixed_bin=fixed_bin,
        moving_bin_before=moving_bin,
        moving_bin_after=warped_moving,
        out_path=overlay_path,
        title=f"FOV-{fov} {spec_ch}->{cam_ch} (mask-centroids mutualNN+RANSAC affine)"
    )

    # Per-pair simple CSV
    per_pair_csv = os.path.join(out_dir, "metrics_simple.csv")
    with open(per_pair_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "FOV", "pair", "status",
            "num_fixed_objects", "num_moving_objects",
            "num_mutual_pairs", "num_ransac_inliers",
            f"inlier_pct_{INLIER_THR_PX:.0f}px", "median_error_px",
            "PASS"
        ])
        w.writerow([
            fov, f"{spec_ch}->{cam_ch}", "OK",
            int(len(fixed_cyxc)), int(len(moving_cyxc)),
            int(len(src_xy)), int(np.sum(inliers)),
            inlier, med,
            "YES" if passed else "NO"
        ])

    return {
        "FOV": fov,
        "pair": f"{spec_ch}->{cam_ch}",
        "status": "OK",
        "num_fixed_objects": int(len(fixed_cyxc)),
        "num_moving_objects": int(len(moving_cyxc)),
        "num_mutual_pairs": int(len(src_xy)),
        "num_ransac_inliers": int(np.sum(inliers)),
        f"inlier_pct_{INLIER_THR_PX:.0f}px": inlier,
        "median_error_px": med,
        "PASS": "YES" if passed else "NO",
        "out_dir": out_dir
    }


def main():
    os.makedirs(ROOT_OUT, exist_ok=True)
    rows = []

    for fov in FOV_LIST:
        for spec_ch, cam_ch in PAIRS:
            print(f"\n=== MASK AFFINE (mutualNN+RANSAC): FOV-{fov} {spec_ch}->{cam_ch} ===")
            try:
                r = register_one_mask_affine(fov, spec_ch, cam_ch)
                if r is not None:
                    rows.append(r)
            except Exception as e:
                print(f"  -> ERROR: {e}")
                import traceback
                traceback.print_exc()

    if rows:
        summary_path = os.path.join(ROOT_OUT, "register_2d_mask_affine_mutualNN_ransac_summary.csv")
        keys = [
            "FOV", "pair", "status",
            "num_fixed_objects", "num_moving_objects",
            "num_mutual_pairs", "num_ransac_inliers",
            f"inlier_pct_{INLIER_THR_PX:.0f}px", "median_error_px",
            "PASS", "out_dir"
        ]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in keys} for r in rows])

        print(f"\n✓ Wrote summary CSV: {summary_path}")
        print(f"✓ Completed {len(rows)} runs")
    else:
        print("\n✗ No runs completed successfully")


if __name__ == "__main__":
    main()