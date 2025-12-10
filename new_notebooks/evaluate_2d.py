import numpy as np
import ants
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

OUT_DIR = "outputs_FOV6_blue_to_DAPI"  # adjust if needed


def detect_beads(img, threshold_rel=0.3, min_distance=5):
    """Detect bright bead-like peaks."""
    img = np.squeeze(img)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    coords = peak_local_max(
        img_norm,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )
    # coords is (N, 2) as (row, col)
    return coords


def match_beads(coords_fixed, coords_moving):
    """For each moving bead, find nearest fixed bead."""
    tree = cKDTree(coords_fixed)
    dists, idxs = tree.query(coords_moving)
    return idxs, dists


def make_overlay(ax, fixed, other, title):
    """Overlay two images: fixed in red, other in green."""
    fixed = np.squeeze(fixed)
    other = np.squeeze(other)

    f = (fixed - np.percentile(fixed, 1)) / (
        np.percentile(fixed, 99) - np.percentile(fixed, 1) + 1e-8
    )
    m = (other - np.percentile(other, 1)) / (
        np.percentile(other, 99) - np.percentile(other, 1) + 1e-8
    )
    f = np.clip(f, 0, 1)
    m = np.clip(m, 0, 1)

    rgb = np.zeros(fixed.shape + (3,), dtype=float)
    rgb[..., 0] = f        # red = camera (fixed)
    rgb[..., 1] = m        # green = spectral (moving/warped)
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")


def make_bead_dot_panel(ax, shape, beads_fixed, beads_other, title, marker_size=40):
    """
    Make a black background and draw solid dots:
      - red = fixed beads
      - green = other beads (moving/warped)
    """
    h, w = shape
    ax.imshow(np.zeros((h, w)), cmap="gray", vmin=0, vmax=1)  # black background

    if beads_fixed is not None and len(beads_fixed) > 0:
        ax.scatter(
            beads_fixed[:, 1],  # x = col
            beads_fixed[:, 0],  # y = row
            s=marker_size,
            c="red",
            label="fixed",
            alpha=0.9,
        )
    if beads_other is not None and len(beads_other) > 0:
        ax.scatter(
            beads_other[:, 1],
            beads_other[:, 0],
            s=marker_size,
            c="lime",
            label="other",
            alpha=0.9,
        )
    ax.set_title(title)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8)


def main():
    # --------- load images ----------
    fixed = ants.image_read(os.path.join(OUT_DIR, "fixed_camera.nii.gz")).numpy()
    moving = ants.image_read(os.path.join(OUT_DIR, "moving_spectral_resampled.nii.gz")).numpy()
    aff = ants.image_read(os.path.join(OUT_DIR, "warped_affine.nii.gz")).numpy()
    syn = ants.image_read(os.path.join(OUT_DIR, "warped_affine_syn.nii.gz")).numpy()

    fixed = np.squeeze(fixed)
    moving = np.squeeze(moving)
    aff = np.squeeze(aff)
    syn = np.squeeze(syn)

    # --------- bead detection ----------
    beads_fixed = detect_beads(fixed)
    beads_moving = detect_beads(moving)
    beads_aff = detect_beads(aff)
    beads_syn = detect_beads(syn)

    # --------- bead matching & metrics (Affine + Syn) ----------
    idx_aff, dist_aff = match_beads(beads_fixed, beads_aff)
    idx_syn, dist_syn = match_beads(beads_fixed, beads_syn)

    print("Mean bead displacement after Affine:", float(np.mean(dist_aff)))
    print("Mean bead displacement after Syn:", float(np.mean(dist_syn)))

    # --------- FIGURE 1: three overlays (before, affine, syn) ----------
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

    make_overlay(
        axes1[0],
        fixed,
        moving,
        "Overlay BEFORE registration\n(fixed camera vs spectral resampled)",
    )
    make_overlay(
        axes1[1],
        fixed,
        aff,
        "Overlay AFTER Affine",
    )
    make_overlay(
        axes1[2],
        fixed,
        syn,
        "Overlay AFTER Affine + SyN",
    )

    fig1.tight_layout()
    fig1_path = os.path.join(OUT_DIR, "overlays_registration.png")
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)
    print("Saved registration overlays to:", fig1_path)

    # --------- FIGURE 2: raw camera/spectral + overlay (no beads) ----------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    # camera alone
    axes2[0].imshow(fixed, cmap="gray")
    axes2[0].set_title("Camera (fixed)")
    axes2[0].axis("off")

    # spectral alone
    axes2[1].imshow(moving, cmap="gray")
    axes2[1].set_title("Spectral (resampled moving)")
    axes2[1].axis("off")

    # overlay camera + spectral
    make_overlay(
        axes2[2],
        fixed,
        moving,
        "Camera + Spectral overlay\n(before registration)",
    )

    fig2.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "raw_camera_spectral.png")
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)
    print("Saved raw camera/spectral figure to:", fig2_path)

    # --------- FIGURE 3: bead-dot overlays (black background) ----------
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: BEFORE registration (fixed vs moving)
    make_bead_dot_panel(
        axes3[0],
        fixed.shape,
        beads_fixed,
        beads_moving,
        "Beads BEFORE registration\n(fixed vs moving)",
        marker_size=40,
    )

    # Panel 2: AFTER Affine
    make_bead_dot_panel(
        axes3[1],
        fixed.shape,
        beads_fixed,
        beads_aff,
        "Beads AFTER Affine",
        marker_size=40,
    )

    # Panel 3: AFTER Affine + SyN
    make_bead_dot_panel(
        axes3[2],
        fixed.shape,
        beads_fixed,
        beads_syn,
        "Beads AFTER Affine+SyN",
        marker_size=40,
    )

    fig3.tight_layout()
    fig3_path = os.path.join(OUT_DIR, "bead_overlays.png")
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)
    print("Saved bead-dot overlays to:", fig3_path)


if __name__ == "__main__":
    main()