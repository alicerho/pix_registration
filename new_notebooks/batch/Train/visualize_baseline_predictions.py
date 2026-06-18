#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------
# CONFIG
# ------------------------

ROOT = Path(__file__).resolve().parent

PRED_CSV = ROOT / "results" / "baseline_test_predictions.csv"

OUT_DIR = ROOT / "figures" / "baseline_prediction_qc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_SIZE = 512

# Set both to None to automatically use the first test FOV
SLIDE = None
FIELD = None

OUTPUT_COLS = [
    "spectral_er",
    "spectral_go",
    "spectral_px",
    "spectral_vo",
    "spectral_mt",
    "spectral_ld",
]


# ------------------------
# HELPERS
# ------------------------

def make_image(df: pd.DataFrame, value_col: str) -> np.ndarray:
    """
    Reconstructs a sparse 512x512 image from selected pixel rows.
    Missing pixels are NaN.
    """
    img = np.full((CROP_SIZE, CROP_SIZE), np.nan, dtype=np.float32)

    y = df["y"].astype(int).values
    x = df["x"].astype(int).values

    img[y, x] = df[value_col].values.astype(np.float32)

    return img


def norm_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image for display while treating NaNs as blank.
    """
    arr = img.copy().astype(np.float32)
    valid = np.isfinite(arr)

    if not np.any(valid):
        return np.zeros_like(arr)

    mn = np.nanmin(arr)
    mx = np.nanmax(arr)

    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)

    arr[~valid] = 0

    return arr


def residual_for_display(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Computes residual only where both actual and predicted are finite.
    """
    residual = np.full_like(actual, np.nan, dtype=np.float32)

    valid = np.isfinite(actual) & np.isfinite(pred)

    residual[valid] = actual[valid] - pred[valid]

    return residual


def overlay_actual_pred(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Red = actual, green = predicted, yellow = overlap.
    """
    rgb = np.zeros((*actual.shape, 3), dtype=np.float32)

    rgb[..., 0] = norm_for_display(actual)
    rgb[..., 1] = norm_for_display(pred)

    return rgb


# ------------------------
# MAIN
# ------------------------

def main():
    print(f"Loading predictions table: {PRED_CSV}")

    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Could not find prediction CSV: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV)

    required_cols = ["slide", "field", "y", "x", "pair_used"]

    for col in OUTPUT_COLS:
        required_cols.append(f"true_{col}")
        required_cols.append(f"pred_{col}")

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns from prediction CSV: {missing}")

    if SLIDE is None or FIELD is None:
        first = df[["slide", "field"]].drop_duplicates().iloc[0]
        slide = int(first["slide"])
        field = int(first["field"])
    else:
        slide = int(SLIDE)
        field = int(FIELD)

    print(f"Using FOV: slide-{slide}_field-{field}")

    sub = df[
        (df["slide"] == slide)
        & (df["field"] == field)
    ].copy()

    if sub.empty:
        raise ValueError(f"No rows found for slide-{slide}_field-{field}")

    print(f"Rows for this FOV: {len(sub):,}")

    for col in OUTPUT_COLS:
        actual_col = f"true_{col}"
        pred_col = f"pred_{col}"

        actual_img = make_image(sub, actual_col)
        pred_img = make_image(sub, pred_col)
        residual_img = residual_for_display(actual_img, pred_img)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(norm_for_display(actual_img), cmap="gray")
        axes[0].set_title(f"Actual {col}")

        axes[1].imshow(norm_for_display(pred_img), cmap="gray")
        axes[1].set_title(f"Predicted {col}")

        axes[2].imshow(overlay_actual_pred(actual_img, pred_img))
        axes[2].set_title("Overlay\nactual=red, pred=green")

        im = axes[3].imshow(
            residual_img,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        axes[3].set_title("Residual\nactual - pred")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")

        plt.suptitle(f"slide-{slide}_field-{field} | {col}")
        plt.tight_layout()

        out_path = OUT_DIR / f"slide-{slide}_field-{field}_{col}_actual_predicted.png"

        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")

    print("\nDone.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()