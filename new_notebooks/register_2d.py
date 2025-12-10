import os
import csv
import numpy as np
import ants
import nd2
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# --------------------
# CONFIG
# --------------------
DATA_DIR = "../data/2025-11-05_Registration2D"  # folder with FOV-6_*.nd2
FOV = 6

PAIR = {
    "spectral_channel": "blue",   # blue/green/yellow/red
    "camera_channel":   "DAPI",   # DAPI/FITC/YFP/TRITC
}

CROP_SIZE = 512  # center crop size for camera


# --------------------
# HELPERS
# --------------------
def load_camera_image(fov, camera_name):
    path = os.path.join(DATA_DIR, f"FOV-{fov}_camera-{camera_name}.nd2")
    with nd2.ND2File(path) as f:
        arr = f.asarray()
    arr = np.squeeze(arr)  # remove singleton dims
    if arr.ndim == 3:
        # if there is a Z or C dimension, just max project it
        arr = np.max(arr, axis=0)
    return arr.astype("float32")


def load_spectral_image(fov, spectral_name):
    path = os.path.join(DATA_DIR, f"FOV-{fov}_spectral-{spectral_name}.nd2")
    with nd2.ND2File(path) as f:
        arr = f.asarray()
    arr = np.squeeze(arr)
    # README: blue/red are (5,512,512) and should be summed over axis 0
    if arr.ndim == 3:
        arr = np.sum(arr, axis=0)
    return arr.astype("float32")


def center_crop(img, crop_size):
    h, w = img.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img[y0:y0+crop_size, x0:x0+crop_size]


def corr(a, b):
    return pearsonr(a.ravel(), b.ravel())[0]


def norm01(img, p_lo=1, p_hi=99):
    """Normalize image to 0–1 using percentiles for contrast."""
    lo, hi = np.percentile(img, [p_lo, p_hi])
    out = (img - lo) / (hi - lo + 1e-8)
    return np.clip(out, 0, 1)


def make_visualizations(cam_crop, moving_resampled, warped_aff, warped_syn, out_dir,
                        spec_ch, cam_ch):
    os.makedirs(out_dir, exist_ok=True)

    # Normalize all arrays for display
    fixed_n = norm01(cam_crop)
    mov_n = norm01(moving_resampled)
    aff_n = norm01(warped_aff)
    syn_n = norm01(warped_syn)

    # ----- 1) Fixed vs moving vs warped -----
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(fixed_n, cmap="gray")
    axes[0].set_title(f"Fixed: camera-{cam_ch}")
    axes[1].imshow(mov_n, cmap="gray")
    axes[1].set_title(f"Moving: spectral-{spec_ch}\n(resampled)")
    axes[2].imshow(aff_n, cmap="gray")
    axes[2].set_title("After Affine")
    axes[3].imshow(syn_n, cmap="gray")
    axes[3].set_title("After Affine + SyN")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "images_side_by_side.png"), dpi=200)
    plt.close(fig)

    # ----- 2) Overlays (R = fixed, G = moving/warped) -----
    def make_rgb(fixed, other):
        return np.dstack([fixed, other, np.zeros_like(fixed)])

    rgb_before = make_rgb(fixed_n, mov_n)
    rgb_aff = make_rgb(fixed_n, aff_n)
    rgb_syn = make_rgb(fixed_n, syn_n)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_before)
    axes[0].set_title("Overlay BEFORE (fixed red, moving green)")
    axes[1].imshow(rgb_aff)
    axes[1].set_title("Overlay AFTER Affine")
    axes[2].imshow(rgb_syn)
    axes[2].set_title("Overlay AFTER Affine+SyN")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlays.png"), dpi=200)
    plt.close(fig)


# --------------------
# MAIN
# --------------------
def main():
    spec_ch = PAIR["spectral_channel"]
    cam_ch = PAIR["camera_channel"]

    # define output directory early so everyone can use it
    out_dir = f"outputs_FOV{FOV}_{spec_ch}_to_{cam_ch}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== FOV-{FOV}: spectral-{spec_ch} -> camera-{cam_ch} ===")

    # 1) Load images
    cam = load_camera_image(FOV, cam_ch)          # (2044, 2048)
    spec = load_spectral_image(FOV, spec_ch)      # (512, 512)

    # 2) Crop camera center to 512x512
    cam_crop = center_crop(cam, CROP_SIZE)        # (512, 512)

    # 3) Convert to ANTs and match sizes
    fixed = ants.from_numpy(cam_crop)
    moving = ants.from_numpy(spec)

    if moving.shape != fixed.shape:
        moving = ants.resample_image(
            moving,
            resample_params=fixed.shape,
            use_voxels=True,
            interp_type=1
        )

    moving_np_resampled = moving.numpy()

    # Metrics BEFORE registration
    c_before = corr(cam_crop, moving_np_resampled)
    mae_before = np.mean(np.abs(cam_crop - moving_np_resampled))
    print(f"Before registration: corr={c_before:.4f}, MAE={mae_before:.4f}")

    # 4) Affine registration
    print("\n--- Affine registration ---")
    reg_aff = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine",
        aff_metric="mattes",
        aff_sampling=64,
        aff_iterations=(2000, 1000, 500),
        aff_smoothing_sigmas=(2, 1, 0),
        aff_shrink_factors=(4, 2, 1),
        verbose=False
    )

    warped_aff_img = reg_aff["warpedmovout"]
    warped_aff = warped_aff_img.numpy()
    c_aff = corr(cam_crop, warped_aff)
    mae_aff = np.mean(np.abs(cam_crop - warped_aff))
    print(f"After Affine:      corr={c_aff:.4f}, MAE={mae_aff:.4f}")
    print("Affine transform file:", reg_aff["fwdtransforms"][0])

    # ---- extract and save affine matrix ----
    affine_path = reg_aff["fwdtransforms"][0]
    affine_tx = ants.read_transform(affine_path)
    params = np.array(affine_tx.parameters)   # [a11, a12, a21, a22, tx, ty]
    A = params[:4].reshape(2, 2)
    t = params[4:]

    print("Affine 2x2 matrix:\n", A)
    print("Affine translation (tx, ty):", t)

    np.savetxt(os.path.join(out_dir, "affine_matrix_2x2.txt"), A)
    np.savetxt(os.path.join(out_dir, "affine_translation.txt"), t)

    # optional: 3×3 homogeneous matrix
    A_h = np.eye(3)
    A_h[:2, :2] = A
    A_h[:2, 2] = t
    np.savetxt(os.path.join(out_dir, "affine_matrix_3x3.txt"), A_h)

    # 5) Nonlinear (SyN) starting from affine result
    print("\n--- Affine + SyN (nonlinear) ---")
    reg_syn = ants.registration(
        fixed=fixed,
        moving=warped_aff_img,  # start from Affine result
        type_of_transform="SyN",
        syn_metric="mattes",
        syn_sampling=64,
        reg_iterations=(100, 50, 25),
        flow_sigma=3,
        total_sigma=0,
        verbose=False
    )

    warped_syn_img = reg_syn["warpedmovout"]
    warped_syn = warped_syn_img.numpy()
    c_syn = corr(cam_crop, warped_syn)
    mae_syn = np.mean(np.abs(cam_crop - warped_syn))
    print(f"After Affine+SyN:  corr={c_syn:.4f}, MAE={mae_syn:.4f}")
    print("SyN forward transform:", reg_syn["fwdtransforms"])

    # 6) Save images
    ants.image_write(ants.from_numpy(cam_crop),
                     os.path.join(out_dir, "fixed_camera.nii.gz"))
    ants.image_write(ants.from_numpy(moving_np_resampled),
                     os.path.join(out_dir, "moving_spectral_resampled.nii.gz"))
    ants.image_write(warped_aff_img,
                     os.path.join(out_dir, "warped_affine.nii.gz"))
    ants.image_write(warped_syn_img,
                     os.path.join(out_dir, "warped_affine_syn.nii.gz"))

    # visualizations
    make_visualizations(cam_crop, moving_np_resampled,
                        warped_aff, warped_syn, out_dir, spec_ch, cam_ch)

    # 7) Save metrics to CSV
    metrics_path = os.path.join(out_dir, "metrics.csv")
    header = [
        "FOV", "spectral", "camera",
        "corr_before", "mae_before",
        "corr_affine", "mae_affine",
        "corr_affine_syn", "mae_affine_syn"
    ]
    row = [
        FOV, spec_ch, cam_ch,
        c_before, mae_before,
        c_aff, mae_aff,
        c_syn, mae_syn
    ]
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print("\nSaved warped images, overlays, matrices, and metrics to", out_dir)


if __name__ == "__main__":
    main()