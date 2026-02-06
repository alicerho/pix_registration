import os
import numpy as np
import ants
import matplotlib.pyplot as plt
import csv

from skimage import io
from scipy.ndimage import distance_transform_edt


# --------------------
# CONFIG
# --------------------
ROOT_OUT = "outputs_2d_w_masks"          # same ROOT_OUT you used before
MASK_DIR = "../../data/Masks"    # Simon masks

FOV_LIST = [1, 2, 3, 4, 5, 6]

PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]

CROP_SIZE = 512

# ANTs affine settings (distance transform images)
AFF_METRIC = "meansquares"
AFF_SAMPLING = 32
AFF_ITERATIONS = (200, 100, 50)


# --------------------
# HELPERS
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

    # Special-case: sometimes spectral-red has suffixes
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


def binary_from_label(label_img):
    return (label_img > 0)


def distance_map(binary_mask):
    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.float32)
    d = distance_transform_edt(binary_mask).astype(np.float32)
    # normalize to [0,1] so fixed/moving scales match
    if d.max() > 0:
        d = d / d.max()
    return d


def dice(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.sum((a == 1) & (b == 1))
    denom = np.sum(a == 1) + np.sum(b == 1)
    if denom == 0:
        return np.nan
    return float(2.0 * inter / denom)


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
# CORE REGISTRATION (MASK-BASED)
# --------------------
def register_one_mask_affine(fov, spec_ch, cam_ch):
    out_dir = os.path.join(ROOT_OUT, f"FOV-{fov}", f"{spec_ch}_to_{cam_ch}", "maskDT_affine")
    os.makedirs(out_dir, exist_ok=True)

    cam_mask_path = find_mask_path(fov, "camera", cam_ch)
    spec_mask_path = find_mask_path(fov, "spectral", spec_ch)
    if cam_mask_path is None or spec_mask_path is None:
        print(f"  -> Missing masks for FOV-{fov} {spec_ch}->{cam_ch}, skipping.")
        return None

    cam_label_full = load_label_mask(cam_mask_path)     # likely 2048x2048
    spec_label = load_label_mask(spec_mask_path)        # likely 512x512

    # Put camera into the same 512x512 center crop space as registration
    cam_label_crop = center_crop(cam_label_full, CROP_SIZE)

    # Binary masks
    fixed_bin = binary_from_label(cam_label_crop)  # camera (fixed)
    moving_bin = binary_from_label(spec_label)     # spectral (moving)

    if fixed_bin.sum() == 0 or moving_bin.sum() == 0:
        print("  -> Empty foreground in one of the masks; skipping.")
        return {
            "FOV": fov,
            "pair": f"{spec_ch}->{cam_ch}",
            "status": "EMPTY_MASK",
            "dice_before": np.nan,
            "dice_after": np.nan,
            "dice_delta": np.nan,
            "out_dir": out_dir
        }

    # Distance maps for registration
    fixed_dt = distance_map(fixed_bin)
    moving_dt = distance_map(moving_bin)

    fixed_ants = ants.from_numpy(fixed_dt.astype(np.float32))
    moving_ants = ants.from_numpy(moving_dt.astype(np.float32))

    # Affine registration on distance maps
    reg = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform="Affine",
        aff_metric=AFF_METRIC,
        aff_sampling=AFF_SAMPLING,
        reg_iterations=AFF_ITERATIONS,
        verbose=False
    )

    # Warp the moving BINARY mask with nearest-neighbor interpolation
    warped_bin_img = ants.apply_transforms(
        fixed=ants.from_numpy(fixed_dt.astype(np.float32)),      # reference grid
        moving=ants.from_numpy(moving_bin.astype(np.float32)),   # moving binary
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor"
    )
    warped_bin = (warped_bin_img.numpy() > 0.5)

    # Dice before/after
    dice_before = dice(fixed_bin, moving_bin)
    dice_after = dice(fixed_bin, warped_bin)

    # Save outputs
    ants.image_write(ants.from_numpy(fixed_bin.astype(np.uint8)), os.path.join(out_dir, "fixed_binary.nii.gz"))
    ants.image_write(ants.from_numpy(moving_bin.astype(np.uint8)), os.path.join(out_dir, "moving_binary.nii.gz"))
    ants.image_write(ants.from_numpy(warped_bin.astype(np.uint8)), os.path.join(out_dir, "warped_binary_affine.nii.gz"))

    ants.image_write(ants.from_numpy(fixed_dt.astype(np.float32)), os.path.join(out_dir, "fixed_dt.nii.gz"))
    ants.image_write(ants.from_numpy(moving_dt.astype(np.float32)), os.path.join(out_dir, "moving_dt.nii.gz"))
    ants.image_write(reg["warpedmovout"], os.path.join(out_dir, "warped_dt_affine.nii.gz"))

    # Save overlay PNG
    overlay_path = os.path.join(out_dir, "overlay_before_after.png")
    save_before_after_overlay_png(
        fixed_bin=fixed_bin,
        moving_bin_before=moving_bin,
        moving_bin_after=warped_bin,
        out_path=overlay_path,
        title=f"FOV-{fov} {spec_ch}->{cam_ch} (maskDT affine)"
    )

    # Save per-pair CSV
    per_pair_csv = os.path.join(out_dir, "metrics_simple.csv")
    with open(per_pair_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FOV", "pair", "status", "dice_before", "dice_after", "dice_delta"])
        w.writerow([fov, f"{spec_ch}->{cam_ch}", "OK", dice_before, dice_after, dice_after - dice_before])

    return {
        "FOV": fov,
        "pair": f"{spec_ch}->{cam_ch}",
        "status": "OK",
        "dice_before": dice_before,
        "dice_after": dice_after,
        "dice_delta": dice_after - dice_before,
        "out_dir": out_dir
    }


def main():
    os.makedirs(ROOT_OUT, exist_ok=True)
    rows = []

    for fov in FOV_LIST:
        for spec_ch, cam_ch in PAIRS:
            print(f"\n=== MASK-DT AFFINE: FOV-{fov} {spec_ch}->{cam_ch} ===")
            try:
                r = register_one_mask_affine(fov, spec_ch, cam_ch)
                if r is not None:
                    rows.append(r)
            except Exception as e:
                print(f"  -> ERROR: {e}")
                import traceback
                traceback.print_exc()

    if rows:
        summary_path = os.path.join(ROOT_OUT, "register_2d_maskDT_affine_summary.csv")
        keys = ["FOV", "pair", "status", "dice_before", "dice_after", "dice_delta", "out_dir"]
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