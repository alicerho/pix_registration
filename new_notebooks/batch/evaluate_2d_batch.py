import os
import numpy as np
import ants
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import csv

ROOT_OUT = "outputs_2d_w_masks"

FOV_LIST = [1, 2, 3, 4, 5, 6]
PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]


def detect_beads(img, threshold_rel=0.3, min_distance=5):
    img = np.squeeze(img)
    if img.max() == img.min():
        return np.zeros((0, 2), dtype=int)
    img_norm = (img - img.min()) / (img.max() - img.min())
    coords = peak_local_max(
        img_norm,
        min_distance=min_distance,
        threshold_rel=threshold_rel
    )
    return coords


def match_beads(coords_fixed, coords_moving):
    if len(coords_fixed) == 0 or len(coords_moving) == 0:
        return np.array([]), np.array([])
    tree = cKDTree(coords_fixed)
    dists, idxs = tree.query(coords_moving)
    return idxs, dists


def bead_overlay_figure(fixed, moving, aff, syn, out_path, title_prefix=""):
    fixed = np.squeeze(fixed)
    moving = np.squeeze(moving)
    aff = np.squeeze(aff)
    syn = np.squeeze(syn)

    beads_fixed = detect_beads(fixed)
    beads_mov = detect_beads(moving)
    beads_aff = detect_beads(aff)
    beads_syn = detect_beads(syn)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # BEFORE
    ax = axes[0]
    ax.imshow(fixed, cmap="gray")
    if len(beads_fixed):
        ax.scatter(beads_fixed[:, 1], beads_fixed[:, 0],
                   s=20, edgecolor="none", c="lime", alpha=0.9, label="fixed")
    if len(beads_mov):
        ax.scatter(beads_mov[:, 1], beads_mov[:, 0],
                   s=20, edgecolor="none", c="red", alpha=0.6, label="moving")
    ax.set_title(f"{title_prefix} beads: BEFORE")
    ax.axis("off")

    # AFTER Affine
    ax = axes[1]
    ax.imshow(fixed, cmap="gray")
    if len(beads_fixed):
        ax.scatter(beads_fixed[:, 1], beads_fixed[:, 0],
                   s=20, edgecolor="none", c="lime", alpha=0.9)
    if len(beads_aff):
        ax.scatter(beads_aff[:, 1], beads_aff[:, 0],
                   s=20, edgecolor="none", c="red", alpha=0.6)
    ax.set_title(f"{title_prefix} beads: AFTER Affine")
    ax.axis("off")

    # AFTER Syn
    ax = axes[2]
    ax.imshow(fixed, cmap="gray")
    if len(beads_fixed):
        ax.scatter(beads_fixed[:, 1], beads_fixed[:, 0],
                   s=20, edgecolor="none", c="lime", alpha=0.9)
    if len(beads_syn):
        ax.scatter(beads_syn[:, 1], beads_syn[:, 0],
                   s=20, edgecolor="none", c="red", alpha=0.6)
    ax.set_title(f"{title_prefix} beads: AFTER Affine+SyN")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_one(fov, spec_ch, cam_ch):
    fov_dir = os.path.join(ROOT_OUT, f"FOV-{fov}")
    pair_dir = f"{spec_ch}_to_{cam_ch}"
    out_dir = os.path.join(fov_dir, pair_dir)

    if not os.path.isdir(out_dir):
        print(f"  -> No output dir {out_dir}, skipping.")
        return None

    print(f"\n=== Evaluating FOV-{fov}: spectral-{spec_ch} -> camera-{cam_ch} ===")

    fixed = ants.image_read(os.path.join(out_dir, "fixed_camera.nii.gz")).numpy()
    moving = ants.image_read(os.path.join(out_dir, "moving_spectral_resampled.nii.gz")).numpy()
    aff = ants.image_read(os.path.join(out_dir, "warped_affine.nii.gz")).numpy()
    syn = ants.image_read(os.path.join(out_dir, "warped_affine_syn.nii.gz")).numpy()

    fixed = np.squeeze(fixed)
    moving = np.squeeze(moving)
    aff = np.squeeze(aff)
    syn = np.squeeze(syn)

    beads_fixed = detect_beads(fixed)
    beads_aff = detect_beads(aff)
    beads_syn = detect_beads(syn)

    _, dist_aff = match_beads(beads_fixed, beads_aff)
    _, dist_syn = match_beads(beads_fixed, beads_syn)

    mean_aff = float(np.mean(dist_aff)) if len(dist_aff) else np.nan
    mean_syn = float(np.mean(dist_syn)) if len(dist_syn) else np.nan

    print("Mean bead displacement after Affine:", mean_aff)
    print("Mean bead displacement after Syn:", mean_syn)

    overlay_path = os.path.join(out_dir, "bead_overlays.png")
    title_prefix = f"FOV-{fov} {spec_ch}->{cam_ch}"
    bead_overlay_figure(fixed, moving, aff, syn, overlay_path, title_prefix)
    print("Saved bead overlay figure to:", overlay_path)

    return {
        "FOV": fov,
        "spectral": spec_ch,
        "camera": cam_ch,
        "mean_bead_disp_affine": mean_aff,
        "mean_bead_disp_syn": mean_syn,
        "num_beads_fixed": len(beads_fixed),
        "num_beads_affine": len(beads_aff),
        "num_beads_syn": len(beads_syn),
    }


def main():
    os.makedirs(ROOT_OUT, exist_ok=True)

    rows = []
    for fov in FOV_LIST:
        for spec_ch, cam_ch in PAIRS:
            res = evaluate_one(fov, spec_ch, cam_ch)
            if res is not None:
                rows.append(res)

    if rows:
        out_csv = os.path.join(ROOT_OUT, "evaluate_2d_batch_summary.csv")
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nWrote evaluation summary to", out_csv)
    else:
        print("\nNo evaluations run (no output dirs?).")


if __name__ == "__main__":
    main()