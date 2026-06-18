#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


# ------------------------
# CONFIG
# ------------------------

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

CSV_PATH = PROJECT_ROOT / "correlated_pixel_database" / "top_correlated_pixels.csv"

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_OUT = MODEL_DIR / "baseline_random_forest.pkl"
RESULTS_OUT = RESULTS_DIR / "baseline_results.txt"
PREDICTIONS_OUT = RESULTS_DIR / "baseline_test_predictions.csv"

# Set to None to use all rows.
# For first test, 200_000 is safer/faster.
MAX_ROWS = 200_000

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ------------------------
# COLUMNS
# ------------------------

INPUT_COLS = [
    "camera_DAPI",
    "camera_FITC",
    "camera_TRITC",
    "camera_YFP",
]

OUTPUT_COLS = [
    "spectral_er",
    "spectral_go",
    "spectral_px",
    "spectral_vo",
    "spectral_mt",
    "spectral_ld",
]


# ------------------------
# MAIN
# ------------------------

def main():
    print(f"Loading data from: {CSV_PATH}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"Original rows: {len(df):,}")

    needed_cols = ["slide", "field", "y", "x", "pair_used"] + INPUT_COLS + OUTPUT_COLS
    missing = [c for c in needed_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=needed_cols).copy()

    print(f"Rows after dropping NaNs: {len(df):,}")

    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        df = df.sample(
            n=MAX_ROWS,
            random_state=RANDOM_STATE,
        ).copy()

        print(f"Subsampled rows: {len(df):,}")

    # ------------------------
    # FOV-level train/test split
    # ------------------------

    fovs = (
        df[["slide", "field"]]
        .drop_duplicates()
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    n_test = max(1, int(len(fovs) * TEST_SIZE))

    test_fovs = fovs.iloc[:n_test].copy()
    train_fovs = fovs.iloc[n_test:].copy()

    test_keys = set(zip(test_fovs["slide"], test_fovs["field"]))

    fov_keys = list(zip(df["slide"], df["field"]))
    is_test = np.array([key in test_keys for key in fov_keys])

    train_df = df[~is_test].copy()
    test_df = df[is_test].copy()

    print(f"Train FOVs: {len(train_fovs)}")
    print(f"Test FOVs : {len(test_fovs)}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows : {len(test_df):,}")

    X_train = train_df[INPUT_COLS].values
    Y_train = train_df[OUTPUT_COLS].values

    X_test = test_df[INPUT_COLS].values
    Y_test = test_df[OUTPUT_COLS].values

    # ------------------------
    # Train model
    # ------------------------

    print("\nTraining Random Forest baseline...")

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    model.fit(X_train, Y_train)

    print("Predicting test set...")

    Y_pred = model.predict(X_test)

    # ------------------------
    # Evaluate
    # ------------------------

    lines = []

    lines.append("Baseline Random Forest Model")
    lines.append("=" * 40)
    lines.append(f"CSV: {CSV_PATH}")
    lines.append(f"Rows used: {len(df):,}")
    lines.append(f"Train FOVs: {len(train_fovs)}")
    lines.append(f"Test FOVs: {len(test_fovs)}")
    lines.append(f"Train rows: {len(train_df):,}")
    lines.append(f"Test rows: {len(test_df):,}")
    lines.append("")
    lines.append("Inputs:")
    lines.extend([f"  - {c}" for c in INPUT_COLS])
    lines.append("")
    lines.append("Outputs:")
    lines.extend([f"  - {c}" for c in OUTPUT_COLS])
    lines.append("")
    lines.append("Per-channel metrics:")
    lines.append("")

    print("\nPer-channel metrics:")

    for i, col in enumerate(OUTPUT_COLS):
        true = Y_test[:, i]
        pred = Y_pred[:, i]

        r2 = r2_score(true, pred)
        mae = mean_absolute_error(true, pred)

        if np.std(true) > 0 and np.std(pred) > 0:
            pearson_r = pearsonr(true, pred)[0]
        else:
            pearson_r = np.nan

        line = (
            f"{col:>12} | "
            f"R2 = {r2: .4f} | "
            f"MAE = {mae: .4f} | "
            f"Pearson r = {pearson_r: .4f}"
        )

        print(line)
        lines.append(line)

    overall_r2 = r2_score(
        Y_test,
        Y_pred,
        multioutput="uniform_average",
    )

    overall_mae = mean_absolute_error(
        Y_test,
        Y_pred,
    )

    lines.append("")
    lines.append(f"Overall R2:  {overall_r2:.4f}")
    lines.append(f"Overall MAE: {overall_mae:.4f}")

    print(f"\nOverall R2:  {overall_r2:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")

    # ------------------------
    # Save outputs
    # ------------------------

    joblib.dump(model, MODEL_OUT)

    with open(RESULTS_OUT, "w") as f:
        f.write("\n".join(lines))

    pred_df = test_df[
        ["slide", "field", "y", "x", "pair_used"]
    ].copy()

    for col in OUTPUT_COLS:
        pred_df[f"true_{col}"] = test_df[col].values

    for i, col in enumerate(OUTPUT_COLS):
        pred_df[f"pred_{col}"] = Y_pred[:, i]

    pred_df.to_csv(PREDICTIONS_OUT, index=False)

    print(f"\nSaved model to: {MODEL_OUT}")
    print(f"Saved results to: {RESULTS_OUT}")
    print(f"Saved predictions to: {PREDICTIONS_OUT}")


if __name__ == "__main__":
    main()