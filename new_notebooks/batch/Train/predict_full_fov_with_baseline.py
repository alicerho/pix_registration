#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import nd2
import matplotlib.pyplot as plt

from skimage.transform import AffineTransform, warp


# ------------------------
# CONFIG
# ------------------------

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DATASET = PROJECT_ROOT / "../../data/Dataset_300Fovs"
RAW = DATASET / "RAW"
UNMIXED = DATASET / "unmixed"

AFFINE_DIR = PROJECT_ROOT / "batch_affine_results" / "all_300FOV_affine_matrices"

MODEL_PATH = ROOT / "models" / "baseline_random_forest.pkl"
PRED_CSV = ROOT / "results" / "baseline_test_predictions.csv"

OUT_DIR = ROOT / "figures" / "full_fov_baseline_prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_SIZE = 512

# Set to None to use first held-out FOV from baseline_test_predictions.csv
SLIDE = None
FIELD = None


PAIRS = {
    "green": {
        "camera": "FITC",
        "spectral_file_type": "raw",
        "spectral_color": "green",
        "spectral_channels": ["er"],
    },
    "yellow": {
        "camera": "YFP",
        "spectral_file_type": "raw",
        "spectral_color": "yellow",
        "spectral_channels": ["go"],
    },
    "blue": {
        "camera": "DAPI",
        "spectral_file_type": "unmixed",
        "spectral_color": "blue",
        "spectral_channels": ["px", "vo"],
    },
    "red": {
        "camera": "TRITC",
        "spectral_file_type": "unmixed",
        "spectral_color": "red",
        "spectral_channels": ["mt", "ld"],
    },
}

INPUT_COLS = ["DAPI", "FITC", "TRITC", "YFP"]
OUTPUT_COLS = ["er", "go", "px", "vo", "mt", "ld"]


# ------------------------
# HELPERS
# ------------------------

def load_nd2(path: Path) -> np.ndarray:
    with nd2.ND2File(str(path)) as f:
        arr = f.asarray()
    return np.squeeze(arr).astype(np.float32)


def center_crop(img, crop_size):
    h, w = img.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def normalize_01(img):
    img = img.astype(np.float32)
    img = img - np.nanmin(img)
    mx = np.nanmax(img)
    if mx > 0:
        img = img / mx
    return img


def load_camera(slide, field, channel):
    path = RAW / f"EYrainbow_slide-{slide}_field-{field}_camera-{channel}.nd2"
    img = load_nd2(path)

    while img.ndim > 2:
        img = img.max(axis=0)

    if img.shape != (CROP_SIZE, CROP_SIZE):
        img = center_crop(img, CROP_SIZE)

    return normalize_01(img)


def load_affine_matrix(slide, field, mode):
    path = AFFINE_DIR / f"EYrainbow_slide-{slide}_field-{field}_{mode}.txt"
    M = np.loadtxt(path)

    if M.shape != (3, 3):
        raise ValueError(f"Expected 3x3 affine matrix, got {M.shape}: {path}")

    return M


def apply_affine_matrix(img, M, output_shape=(512, 512)):
    tform = AffineTransform(matrix=M)

    warped = warp(
        img,
        inverse_map=tform.inverse,
        output_shape=output_shape,
        preserve_range=True,
        mode="constant",
        cval=0.0,
        order=1,
    )

    return warped.astype(np.float32)


def load_spectral(slide, field, mode):
    info = PAIRS[mode]

    if info["spectral_file_type"] == "raw":
        path = RAW / f"EYrainbow_slide-{slide}_field-{field}_spectral-{info['spectral_color']}.nd2"
    else:
        path = UNMIXED / f"unmixed_EYrainbow_slide-{slide}_field-{field}_spectral-{info['spectral_color']}.nd2"

    arr = load_nd2(path)
    out = {}

    if mode in ["green", "yellow"]:
        while arr.ndim > 2:
            arr = arr.max(axis=0)
        out[info["spectral_channels"][0]] = arr
    else:
        if arr.ndim != 3 or arr.shape[0] != 2:
            raise ValueError(f"Expected 2-channel spectral file for {mode}, got {arr.shape}: {path}")
        out[info["spectral_channels"][0]] = arr[0]
        out[info["spectral_channels"][1]] = arr[1]

    return out


def robust_norm(img):
    lo = np.nanpercentile(img, 1)
    hi = np.nanpercentile(img, 99)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    x = (img - lo) / (hi - lo)
    return np.clip(x, 0, 1)


def overlay_actual_pred(actual, pred):
    rgb = np.zeros((*actual.shape, 3), dtype=np.float32)
    rgb[..., 0] = robust_norm(actual)  # actual = red
    rgb[..., 1] = robust_norm(pred)    # pred = green
    return rgb


# ------------------------
# MAIN
# ------------------------

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    if SLIDE is None or FIELD is None:
        pred_df = pd.read_csv(PRED_CSV)
        first = pred_df[["slide", "field"]].drop_duplicates().iloc[0]
        slide = int(first["slide"])
        field = int(first["field"])
    else:
        slide = int(SLIDE)
        field = int(FIELD)

    print(f"Using FOV: slide-{slide}_field-{field}")

    # ------------------------
    # Load full camera input images
    # ------------------------

    camera = {
        "DAPI": load_camera(slide, field, "DAPI"),
        "FITC": load_camera(slide, field, "FITC"),
        "TRITC": load_camera(slide, field, "TRITC"),
        "YFP": load_camera(slide, field, "YFP"),
    }

    X_img = np.stack([camera[ch] for ch in INPUT_COLS], axis=-1)
    X_flat = X_img.reshape(-1, 4)

    print("Predicting all 512x512 pixels...")
    Y_pred_flat = model.predict(X_flat)

    pred_imgs = {}
    for i, ch in enumerate(OUTPUT_COLS):
        pred_imgs[ch] = Y_pred_flat[:, i].reshape(CROP_SIZE, CROP_SIZE).astype(np.float32)

    # ------------------------
    # Load actual registered spectral images
    # ------------------------

    actual_imgs = {}

    for mode in PAIRS:
        M = load_affine_matrix(slide, field, mode)
        spectral_channels = load_spectral(slide, field, mode)

        for ch_name, img in spectral_channels.items():
            warped = apply_affine_matrix(img, M, output_shape=(CROP_SIZE, CROP_SIZE))
            actual_imgs[ch_name] = normalize_01(warped)

    # ------------------------
    # Save figures
    # ------------------------

    for ch in OUTPUT_COLS:
        actual = actual_imgs[ch]
        pred = pred_imgs[ch]
        residual = actual - pred

        r_abs = np.nanpercentile(np.abs(residual), 99)
        if r_abs <= 0:
            r_abs = 1e-6

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(robust_norm(actual), cmap="gray")
        axes[0].set_title(f"Actual {ch}")

        axes[1].imshow(robust_norm(pred), cmap="gray")
        axes[1].set_title(f"Predicted {ch}")

        axes[2].imshow(overlay_actual_pred(actual, pred))
        axes[2].set_title("Overlay\nactual=red, pred=green")

        im = axes[3].imshow(
            residual,
            cmap="RdBu_r",
            vmin=-r_abs,
            vmax=r_abs,
        )
        axes[3].set_title("Residual\nactual - pred")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")

        plt.suptitle(f"Full-FOV baseline prediction | slide-{slide}_field-{field} | {ch}")
        plt.tight_layout()

        out_path = OUT_DIR / f"slide-{slide}_field-{field}_{ch}_full_fov_prediction.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")

    print("\nDone.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()