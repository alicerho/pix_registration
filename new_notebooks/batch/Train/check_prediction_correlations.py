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

OUT_DIR = ROOT / "results" / "full_fov_prediction_correlation_checks"
FIG_DIR = ROOT / "figures" / "full_fov_prediction_correlation_checks"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CROP_SIZE = 512

# Set to None to process all held-out FOVs from baseline_test_predictions.csv.
# Or set N_FOVS = 3 for a faster test.
N_FOVS = None

RANDOM_STATE = 42

CAMERA_CHANNELS = ["DAPI", "FITC", "TRITC", "YFP"]
SPECTRAL_CHANNELS = ["er", "go", "px", "vo", "mt", "ld"]


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


def pearson_corr(a, b):
    a = a.ravel()
    b = b.ravel()

    valid = np.isfinite(a) & np.isfinite(b)

    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def r2_score_np(true, pred):
    true = true.ravel()
    pred = pred.ravel()

    valid = np.isfinite(true) & np.isfinite(pred)

    true = true[valid]
    pred = pred[valid]

    if len(true) < 2:
        return np.nan

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1 - ss_res / ss_tot)


def mae_np(true, pred):
    true = true.ravel()
    pred = pred.ravel()

    valid = np.isfinite(true) & np.isfinite(pred)

    if not np.any(valid):
        return np.nan

    return float(np.mean(np.abs(true[valid] - pred[valid])))


def predict_full_fov(model, camera_imgs):
    X_img = np.stack(
        [camera_imgs[ch] for ch in CAMERA_CHANNELS],
        axis=-1,
    )

    X_flat = X_img.reshape(-1, 4)

    Y_pred_flat = model.predict(X_flat)

    pred_imgs = {}

    for i, ch in enumerate(SPECTRAL_CHANNELS):
        pred_imgs[ch] = Y_pred_flat[:, i].reshape(CROP_SIZE, CROP_SIZE).astype(np.float32)

    return pred_imgs


def load_actual_registered_spectral(slide, field):
    actual_imgs = {}

    for mode in PAIRS:
        M = load_affine_matrix(slide, field, mode)
        spectral_channels = load_spectral(slide, field, mode)

        for ch_name, img in spectral_channels.items():
            warped = apply_affine_matrix(
                img,
                M,
                output_shape=(CROP_SIZE, CROP_SIZE),
            )

            actual_imgs[ch_name] = normalize_01(warped)

    return actual_imgs


def save_correlation_heatmap(matrix_df, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        matrix_df.values,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
    )

    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_yticks(np.arange(len(matrix_df.index)))

    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix_df.index)

    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            val = matrix_df.values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ------------------------
# MAIN
# ------------------------

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Loading held-out FOV list from: {PRED_CSV}")
    pred_df = pd.read_csv(PRED_CSV)

    fovs = (
        pred_df[["slide", "field"]]
        .drop_duplicates()
        .sort_values(["slide", "field"])
        .reset_index(drop=True)
    )

    if N_FOVS is not None:
        fovs = fovs.sample(
            n=min(N_FOVS, len(fovs)),
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    print(f"Processing {len(fovs)} held-out FOVs")

    rows = []

    for row in fovs.itertuples(index=False):
        slide = int(row.slide)
        field = int(row.field)

        print(f"\n=== slide-{slide}_field-{field} ===")

        try:
            camera_imgs = {
                ch: load_camera(slide, field, ch)
                for ch in CAMERA_CHANNELS
            }

            actual_imgs = load_actual_registered_spectral(slide, field)

            pred_imgs = predict_full_fov(model, camera_imgs)

            # ------------------------
            # Metrics per predicted spectral channel
            # ------------------------

            for spec_ch in SPECTRAL_CHANNELS:
                actual = actual_imgs[spec_ch]
                pred = pred_imgs[spec_ch]

                row_out = {
                    "slide": slide,
                    "field": field,
                    "predicted_channel": spec_ch,
                    "corr_pred_vs_actual_same_channel": pearson_corr(pred, actual),
                    "r2_pred_vs_actual_same_channel": r2_score_np(actual, pred),
                    "mae_pred_vs_actual_same_channel": mae_np(actual, pred),
                }

                # Compare predicted channel to every camera input
                for cam_ch in CAMERA_CHANNELS:
                    row_out[f"corr_pred_vs_camera_{cam_ch}"] = pearson_corr(
                        pred,
                        camera_imgs[cam_ch],
                    )

                # Compare predicted channel to every actual spectral channel
                for actual_ch in SPECTRAL_CHANNELS:
                    row_out[f"corr_pred_vs_actual_{actual_ch}"] = pearson_corr(
                        pred,
                        actual_imgs[actual_ch],
                    )

                # Compare predicted channel to every predicted spectral channel
                for pred_ch2 in SPECTRAL_CHANNELS:
                    row_out[f"corr_pred_vs_pred_{pred_ch2}"] = pearson_corr(
                        pred,
                        pred_imgs[pred_ch2],
                    )

                rows.append(row_out)

        except Exception as e:
            print(f"ERROR on slide-{slide}_field-{field}: {e}")

    results = pd.DataFrame(rows)

    out_csv = OUT_DIR / "full_fov_prediction_correlation_checks.csv"
    results.to_csv(out_csv, index=False)

    print(f"\nSaved metrics: {out_csv}")

    # ------------------------
    # Summary tables
    # ------------------------

    summary = (
        results
        .groupby("predicted_channel")
        .mean(numeric_only=True)
        .reset_index()
    )

    summary_csv = OUT_DIR / "full_fov_prediction_correlation_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"Saved summary: {summary_csv}")

    # ------------------------
    # Heatmap 1:
    # predicted channel vs actual spectral channels
    # ------------------------

    actual_corr_cols = [
        f"corr_pred_vs_actual_{ch}"
        for ch in SPECTRAL_CHANNELS
    ]

    actual_matrix = summary.set_index("predicted_channel")[actual_corr_cols]
    actual_matrix.columns = SPECTRAL_CHANNELS

    save_correlation_heatmap(
        actual_matrix,
        "Mean corr(predicted spectral, actual spectral)",
        FIG_DIR / "mean_corr_pred_vs_actual_spectral.png",
    )

    # ------------------------
    # Heatmap 2:
    # predicted channel vs camera inputs
    # ------------------------

    camera_corr_cols = [
        f"corr_pred_vs_camera_{ch}"
        for ch in CAMERA_CHANNELS
    ]

    camera_matrix = summary.set_index("predicted_channel")[camera_corr_cols]
    camera_matrix.columns = CAMERA_CHANNELS

    save_correlation_heatmap(
        camera_matrix,
        "Mean corr(predicted spectral, camera inputs)",
        FIG_DIR / "mean_corr_pred_vs_camera_inputs.png",
    )

    # ------------------------
    # Heatmap 3:
    # predicted channels vs predicted channels
    # ------------------------

    pred_corr_cols = [
        f"corr_pred_vs_pred_{ch}"
        for ch in SPECTRAL_CHANNELS
    ]

    pred_matrix = summary.set_index("predicted_channel")[pred_corr_cols]
    pred_matrix.columns = SPECTRAL_CHANNELS

    save_correlation_heatmap(
        pred_matrix,
        "Mean corr(predicted spectral, predicted spectral)",
        FIG_DIR / "mean_corr_pred_vs_predicted_spectral.png",
    )

    print(f"Saved figures to: {FIG_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()