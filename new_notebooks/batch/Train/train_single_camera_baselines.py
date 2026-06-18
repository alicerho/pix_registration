#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import nd2
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from skimage.transform import AffineTransform, warp


# ------------------------
# CONFIG
# ------------------------

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

CSV_PATH = PROJECT_ROOT / "correlated_pixel_database" / "top_correlated_pixels.csv"

DATASET = PROJECT_ROOT / "../../data/Dataset_300Fovs"
RAW = DATASET / "RAW"
UNMIXED = DATASET / "unmixed"
AFFINE_DIR = PROJECT_ROOT / "batch_affine_results" / "all_300FOV_affine_matrices"

OUT_DIR = ROOT / "results" / "single_camera_baselines"
MODEL_DIR = ROOT / "models" / "single_camera_baselines"
FIG_DIR = ROOT / "figures" / "single_camera_baselines"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 200_000
RANDOM_STATE = 42
TEST_SIZE = 0.2
CROP_SIZE = 512


# ------------------------
# MODEL DEFINITIONS
# ------------------------

PAIR_MODELS = {
    "FITC_to_er": {
        "inputs": ["camera_FITC"],
        "camera_channels": ["FITC"],
        "outputs": ["spectral_er"],
        "spectral_channels": ["er"],
    },
    "YFP_to_go": {
        "inputs": ["camera_YFP"],
        "camera_channels": ["YFP"],
        "outputs": ["spectral_go"],
        "spectral_channels": ["go"],
    },
    "DAPI_to_px_vo": {
        "inputs": ["camera_DAPI"],
        "camera_channels": ["DAPI"],
        "outputs": ["spectral_px", "spectral_vo"],
        "spectral_channels": ["px", "vo"],
    },
    "TRITC_to_mt_ld": {
        "inputs": ["camera_TRITC"],
        "camera_channels": ["TRITC"],
        "outputs": ["spectral_mt", "spectral_ld"],
        "spectral_channels": ["mt", "ld"],
    },
    "all_camera_to_all_spectral": {
        "inputs": ["camera_DAPI", "camera_FITC", "camera_TRITC", "camera_YFP"],
        "camera_channels": ["DAPI", "FITC", "TRITC", "YFP"],
        "outputs": [
            "spectral_er",
            "spectral_go",
            "spectral_px",
            "spectral_vo",
            "spectral_mt",
            "spectral_ld",
        ],
        "spectral_channels": ["er", "go", "px", "vo", "mt", "ld"],
    },
}


PAIRS = {
    "green": {
        "spectral_file_type": "raw",
        "spectral_color": "green",
        "spectral_channels": ["er"],
    },
    "yellow": {
        "spectral_file_type": "raw",
        "spectral_color": "yellow",
        "spectral_channels": ["go"],
    },
    "blue": {
        "spectral_file_type": "unmixed",
        "spectral_color": "blue",
        "spectral_channels": ["px", "vo"],
    },
    "red": {
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


def normalize_01(img):
    img = img.astype(np.float32)
    img = img - np.nanmin(img)
    mx = np.nanmax(img)
    if mx > 0:
        img = img / mx
    return img


def robust_norm(img):
    lo = np.nanpercentile(img, 1)
    hi = np.nanpercentile(img, 99)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    x = (img - lo) / (hi - lo)
    return np.clip(x, 0, 1)


def center_crop(img, crop_size):
    h, w = img.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


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


def apply_affine(img, M):
    tform = AffineTransform(matrix=M)

    warped = warp(
        img,
        inverse_map=tform.inverse,
        output_shape=(CROP_SIZE, CROP_SIZE),
        preserve_range=True,
        mode="constant",
        cval=0.0,
        order=1,
    )

    return warped.astype(np.float32)


def load_registered_spectral(slide, field):
    out = {}

    for mode, info in PAIRS.items():
        if info["spectral_file_type"] == "raw":
            path = RAW / f"EYrainbow_slide-{slide}_field-{field}_spectral-{info['spectral_color']}.nd2"
        else:
            path = UNMIXED / f"unmixed_EYrainbow_slide-{slide}_field-{field}_spectral-{info['spectral_color']}.nd2"

        arr = load_nd2(path)
        M = load_affine_matrix(slide, field, mode)

        if mode in ["green", "yellow"]:
            while arr.ndim > 2:
                arr = arr.max(axis=0)

            out[info["spectral_channels"][0]] = normalize_01(apply_affine(arr, M))

        else:
            out[info["spectral_channels"][0]] = normalize_01(apply_affine(arr[0], M))
            out[info["spectral_channels"][1]] = normalize_01(apply_affine(arr[1], M))

    return out


def pearson_safe(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return pearsonr(a, b)[0]


def overlay_actual_pred(actual, pred):
    rgb = np.zeros((*actual.shape, 3), dtype=np.float32)
    rgb[..., 0] = robust_norm(actual)  # actual = red
    rgb[..., 1] = robust_norm(pred)    # predicted = green
    return rgb


def predict_full_fov(model, model_spec, slide, field):
    camera_imgs = {
        ch: load_camera(slide, field, ch)
        for ch in model_spec["camera_channels"]
    }

    X_img = np.stack(
        [camera_imgs[ch] for ch in model_spec["camera_channels"]],
        axis=-1,
    )

    X_flat = X_img.reshape(-1, len(model_spec["camera_channels"]))

    Y_pred = model.predict(X_flat)

    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    pred_imgs = {}

    for i, spec_ch in enumerate(model_spec["spectral_channels"]):
        pred_imgs[spec_ch] = Y_pred[:, i].reshape(CROP_SIZE, CROP_SIZE).astype(np.float32)

    return pred_imgs


def save_prediction_figure(actual, pred, slide, field, model_name, spec_ch):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(robust_norm(actual), cmap="gray")
    axes[0].set_title(f"Actual {spec_ch}")

    axes[1].imshow(robust_norm(pred), cmap="gray")
    axes[1].set_title(f"Predicted {spec_ch}")

    axes[2].imshow(overlay_actual_pred(actual, pred))
    axes[2].set_title("Overlay\nactual=red, pred=green")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"{model_name} | slide-{slide}_field-{field} | {spec_ch}")
    plt.tight_layout()

    out_path = FIG_DIR / f"{model_name}_slide-{slide}_field-{field}_{spec_ch}_actual_pred_overlay.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure: {out_path}")


# ------------------------
# MAIN
# ------------------------

def main():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print(f"Original rows: {len(df):,}")

    all_needed = ["slide", "field"]

    for spec in PAIR_MODELS.values():
        all_needed.extend(spec["inputs"])
        all_needed.extend(spec["outputs"])

    all_needed = sorted(set(all_needed))

    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=all_needed).copy()

    print(f"Rows after dropping NaNs: {len(df):,}")

    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).copy()
        print(f"Subsampled rows: {len(df):,}")

    # ------------------------
    # FOV-level split
    # ------------------------

    fovs = (
        df[["slide", "field"]]
        .drop_duplicates()
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    n_test = max(1, int(len(fovs) * TEST_SIZE))

    test_fovs = fovs.iloc[:n_test]
    train_fovs = fovs.iloc[n_test:]

    test_keys = set(zip(test_fovs["slide"], test_fovs["field"]))

    is_test = np.array([
        (s, f) in test_keys
        for s, f in zip(df["slide"], df["field"])
    ])

    train_df = df[~is_test].copy()
    test_df = df[is_test].copy()

    print(f"Train FOVs: {len(train_fovs)}")
    print(f"Test FOVs : {len(test_fovs)}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows : {len(test_df):,}")

    # Use specific FOV for visualization
    vis_slide = 21
    vis_field = 2

    print(f"\nVisualization FOV: slide-{vis_slide}_field-{vis_field}")

    actual_imgs = load_registered_spectral(vis_slide, vis_field)

    all_results = []

    # ------------------------
    # Train + visualize models
    # ------------------------

    for model_name, spec in PAIR_MODELS.items():
        input_cols = spec["inputs"]
        output_cols = spec["outputs"]

        print(f"\n=== Training {model_name} ===")
        print("Inputs :", input_cols)
        print("Outputs:", output_cols)

        X_train = train_df[input_cols].values
        Y_train = train_df[output_cols].values

        X_test = test_df[input_cols].values
        Y_test = test_df[output_cols].values

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)

        model_path = MODEL_DIR / f"{model_name}.pkl"
        joblib.dump(model, model_path)

        for i, out_col in enumerate(output_cols):
            true = Y_test[:, i]
            pred = Y_pred[:, i]

            r2 = r2_score(true, pred)
            mae = mean_absolute_error(true, pred)
            pr = pearson_safe(true, pred)

            all_results.append({
                "model": model_name,
                "input_cols": ",".join(input_cols),
                "output_col": out_col,
                "r2": r2,
                "mae": mae,
                "pearson_r": pr,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_fovs": len(train_fovs),
                "test_fovs": len(test_fovs),
                "model_path": str(model_path),
            })

            print(
                f"{out_col:>12} | "
                f"R2={r2: .4f} | "
                f"MAE={mae: .4f} | "
                f"Pearson r={pr: .4f}"
            )

        # ------------------------
        # Full-FOV visualization
        # ------------------------

        print(f"Predicting full visualization FOV for {model_name}...")

        pred_imgs = predict_full_fov(
            model,
            spec,
            vis_slide,
            vis_field,
        )

        for spec_ch in spec["spectral_channels"]:
            save_prediction_figure(
                actual=actual_imgs[spec_ch],
                pred=pred_imgs[spec_ch],
                slide=vis_slide,
                field=vis_field,
                model_name=model_name,
                spec_ch=spec_ch,
            )

    results_df = pd.DataFrame(all_results)

    out_csv = OUT_DIR / "single_camera_baseline_results.csv"
    results_df.to_csv(out_csv, index=False)

    print(f"\nSaved results: {out_csv}")
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()