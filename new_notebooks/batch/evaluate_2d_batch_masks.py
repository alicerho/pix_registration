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
ROOT_OUT = "outputs_2d"  # <-- same ROOT_OUT you used in register_2d_batch.py
MASK_DIR = "../../data/Masks"  # <-- where Simon's .tif masks are

FOV_LIST = [1, 2, 3, 4, 5, 6]

# Simon-defined pairs
PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]

# thresholds for “good” matches (in pixels)
THRESHOLDS = [5, 10, 20, 30]


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
        # try a few common alternates
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


def centroids_from_labels(label_img, use_weighted=False, intensity_img=None):
    """
    Extract centroids from integer label image.
    Returns Nx2 array in (y, x) order.
    """
    if use_weighted and intensity_img is None:
        raise ValueError("use_weighted=True requires intensity_img")

    props = measure.regionprops(label_img, intensity_image=intensity_img)

    cents = []
    for p in props:
        if p.label == 0:
            continue
        if use_weighted:
            y, x = p.weighted_centroid
        else:
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

    # convert to (x,y)
    xy = points_yx[:, ::-1]

    # x'y' = A * xy + t
    xy2 = (xy @ A2x2.T) + t_xy

    # back to (y,x)
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

    # IMPORTANT: ANTs expects columns named x,y (in that order)
    df = pd.DataFrame({
        "x": points_yx[:, 1],
        "y": points_yx[:, 0],
    })

    # NOTE: your ANTsPy build does NOT accept 'dimension'
    df_warp = ants.apply_transforms_to_points(
        dim=2,
        points=df,
        transformlist=transform_list,
        whichtoinvert=[False] * len(transform_list)
    )

    out = np.stack([df_warp["y"].to_numpy(), df_warp["x"].to_numpy()], axis=1)
    return out


def make_dot_overlay_png(fixed_img, cam_pts_yx, pts_before_yx, pts_aff_yx, pts_syn_yx, out_path, title):
    """
    Creates ONE PNG with 3 panels:
      (1) camera pts vs spectral BEFORE
      (2) camera pts vs spectral AFTER affine
      (3) camera pts vs spectral AFTER syn
    Dots sized to look bead-like.
    """
    fixed = np.squeeze(fixed_img)
    if fixed.ndim != 2:
        fixed = fixed.squeeze()

    def norm01(img):
        lo, hi = np.percentile(img, [1, 99])
        x = (img - lo) / (hi - lo + 1e-8)
        return np.clip(x, 0, 1)

    bg = norm01(fixed)

    # heuristic dot size: scale with image size so dots are bead-ish, not tiny
    # tweak if you want bigger/smaller
    s = max(10, int(0.03 * fixed.shape[0]))  # ~15 for 512, ~60 for 2048

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        ("BEFORE", pts_before_yx),
        ("AFTER Affine", pts_aff_yx),
        ("AFTER Affine+SyN", pts_syn_yx),
    ]

    for ax, (lab, spec_pts) in zip(axes, panels):
        ax.imshow(bg, cmap="gray")
        if len(cam_pts_yx):
            ax.scatter(cam_pts_yx[:, 1], cam_pts_yx[:, 0], s=s, c="lime", alpha=0.55, edgecolors="none", label="camera")
        if len(spec_pts):
            ax.scatter(spec_pts[:, 1], spec_pts[:, 0], s=s, c="red", alpha=0.45, edgecolors="none", label="spectral")
        ax.set_title(f"{title}\n{lab}")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------
# MAIN EVAL FOR ONE PAIR
# --------------------
def evaluate_one(fov, spec_ch, cam_ch):
    # folder structure you requested:
    # ROOT_OUT/FOV-x/spec_to_cam/
    out_dir = os.path.join(ROOT_OUT, f"FOV-{fov}", f"{spec_ch}_to_{cam_ch}")
    if not os.path.isdir(out_dir):
        print(f"  -> Missing output dir, skipping: {out_dir}")
        return None

    # required outputs from registration step
    fixed_path = os.path.join(out_dir, "fixed_camera.nii.gz")
    if not os.path.exists(fixed_path):
        print(f"  -> Missing fixed image, skipping: {fixed_path}")
        return None

    # transform files we expect to exist (see IMPORTANT note below)
    affine_txt = os.path.join(out_dir, "affine_matrix_2x2.txt")
    trans_txt = os.path.join(out_dir, "affine_translation.txt")
    syn_transform_list_txt = os.path.join(out_dir, "syn_fwdtransforms.txt")

    if not (os.path.exists(affine_txt) and os.path.exists(trans_txt)):
        print(f"  -> Missing affine matrix files in {out_dir}, skipping.")
        return None

    # masks
    cam_mask_path = find_mask_path(fov, "camera", cam_ch)
    spec_mask_path = find_mask_path(fov, "spectral", spec_ch)

    if cam_mask_path is None or spec_mask_path is None:
        print(f"  -> Missing masks for FOV-{fov} {spec_ch}->{cam_ch}, skipping.")
        return None

    # load fixed image (for plotting background)
    fixed_img = ants.image_read(fixed_path).numpy()

    # load label masks and centroids (in their native coordinate spaces)
    cam_label = load_label_mask(cam_mask_path)     # ~2048x2048
    spec_label = load_label_mask(spec_mask_path)   # ~512x512

    cam_centroids_full = centroids_from_labels(cam_label)   # (y,x) in full camera space
    spec_centroids = centroids_from_labels(spec_label)      # (y,x) in spectral space

    # Your registration is done on a 512x512 center crop of camera.
    # So we must shift camera centroids into crop coordinates and drop those outside crop.
    H, W = cam_label.shape
    crop = 512
    y0 = H // 2 - crop // 2
    x0 = W // 2 - crop // 2

    cam_centroids_crop = cam_centroids_full.copy()
    cam_centroids_crop[:, 0] -= y0
    cam_centroids_crop[:, 1] -= x0

    # keep only centroids within the crop
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
    # (spec is already 512-space)
    d_before = nn_distances(spec_centroids, cam_centroids_crop)

    # AFTER affine: apply affine to spec points
    spec_aff = apply_affine_to_points(spec_centroids, A2x2, t_xy)
    d_aff = nn_distances(spec_aff, cam_centroids_crop)

    # AFTER SyN: apply ANTs transforms to spec points if available
    spec_syn = None
    d_syn = np.array([])
    if os.path.exists(syn_transform_list_txt):
        with open(syn_transform_list_txt, "r") as f:
            tlist = [line.strip() for line in f if line.strip()]
        # apply all forward transforms (moving->fixed) to points in one shot
        spec_syn = apply_ants_transforms_to_points(spec_centroids, tlist)
        d_syn = nn_distances(spec_syn, cam_centroids_crop)

    # aggregate metrics
    def summarize(d):
        if d is None or len(d) == 0:
            return {
                "mean": np.nan,
                "median": np.nan,
                "n": 0,
                **{f"pct_le_{thr}px": np.nan for thr in THRESHOLDS},
            }
        out = {
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "n": int(len(d)),
        }
        for thr in THRESHOLDS:
            out[f"pct_le_{thr}px"] = float(np.mean(d <= thr) * 100.0)
        return out

    s_before = summarize(d_before)
    s_aff = summarize(d_aff)
    s_syn = summarize(d_syn)

    # plots (dots overlays using mask centroids, not peak detection)
    overlay_path = os.path.join(out_dir, "mask_centroid_overlays.png")
    title = f"FOV-{fov} {spec_ch}->{cam_ch}"

    # for syn plot, if missing, just reuse affine so file still makes sense
    spec_syn_for_plot = spec_syn if spec_syn is not None else spec_aff
    make_dot_overlay_png(
        fixed_img=fixed_img,
        cam_pts_yx=cam_centroids_crop,
        pts_before_yx=spec_centroids,
        pts_aff_yx=spec_aff,
        pts_syn_yx=spec_syn_for_plot,
        out_path=overlay_path,
        title=title
    )

    return {
        "FOV": fov,
        "spectral": spec_ch,
        "camera": cam_ch,

        "num_cam_centroids_in_crop": int(len(cam_centroids_crop)),
        "num_spec_centroids": int(len(spec_centroids)),

        "before_mean_nn_px": s_before["mean"],
        "before_median_nn_px": s_before["median"],
        "before_n": s_before["n"],
        **{f"before_{k}": v for k, v in s_before.items() if k.startswith("pct_")},

        "affine_mean_nn_px": s_aff["mean"],
        "affine_median_nn_px": s_aff["median"],
        "affine_n": s_aff["n"],
        **{f"affine_{k}": v for k, v in s_aff.items() if k.startswith("pct_")},

        "syn_mean_nn_px": s_syn["mean"],
        "syn_median_nn_px": s_syn["median"],
        "syn_n": s_syn["n"],
        **{f"syn_{k}": v for k, v in s_syn.items() if k.startswith("pct_")},
    }


def main():
    rows = []

    for fov in FOV_LIST:
        for spec_ch, cam_ch in PAIRS:
            print(f"\n=== MASK EVAL: FOV-{fov} {spec_ch}->{cam_ch} ===")
            res = evaluate_one(fov, spec_ch, cam_ch)
            if res is not None:
                rows.append(res)

    if rows:
        out_csv = os.path.join(ROOT_OUT, "evaluate_2d_masks_summary.csv")
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"\nWrote mask-based evaluation CSV to: {out_csv}")
    else:
        print("\nNo mask-based evaluations ran (missing output dirs or masks).")


if __name__ == "__main__":
    main()