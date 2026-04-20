#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------
# CONFIG
# ------------------------

CSV_PATH = Path("batch_affine_results/affine_parameters_summary.csv")
OUT_DIR = Path("batch_affine_results/global_affine_from_inliers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = [
    "dx",
    "dy",
    "rotation_deg",
    "scale_x",
    "scale_y",
    "shear_deg",
]

MODES = ["blue", "green", "yellow", "red"]

# registration was done in 512x512 space
CROP_SIZE = 512
CENTER = (CROP_SIZE / 2.0, CROP_SIZE / 2.0)

IQR_K = 1.5  # standard Tukey rule

MODE_TO_CAMERA = {
    "blue": "DAPI",
    "green": "FITC",
    "yellow": "YFP",
    "red": "TRITC",
}

# ------------------------
# HELPERS
# ------------------------

def build_affine_matrix(dx, dy, rotation_deg, scale_x, scale_y, shear_deg, center):
    cx, cy = center

    rotation = np.radians(rotation_deg)
    shear = np.radians(shear_deg)

    T_to_origin = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0,   1],
    ], dtype=np.float64)

    T_back = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0,  1],
    ], dtype=np.float64)

    T_shift = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0,  1],
    ], dtype=np.float64)

    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ], dtype=np.float64)

    S = np.array([
        [scale_x, 0,       0],
        [0,       scale_y, 0],
        [0,       0,       1],
    ], dtype=np.float64)

    Sh = np.array([
        [1, np.tan(shear), 0],
        [0, 1,             0],
        [0, 0,             1],
    ], dtype=np.float64)

    return T_back @ T_shift @ R @ S @ Sh @ T_to_origin

def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

# ------------------------
# MAIN
# ------------------------

def main():
    df = pd.read_csv(CSV_PATH)

    if df.empty:
        print("CSV is empty.")
        return

    summary_rows = []

    for mode in MODES:
        sub = df[df["mode"] == mode].copy()

        if sub.empty:
            print(f"No rows for mode={mode}")
            continue

        # mark inliers/outliers parameter by parameter
        sub["is_inlier"] = True

        for param in PARAMS:
            lo, hi = iqr_bounds(sub[param], k=IQR_K)
            param_inlier_col = f"{param}_inlier"
            sub[param_inlier_col] = (sub[param] >= lo) & (sub[param] <= hi)
            sub["is_inlier"] &= sub[param_inlier_col]

        inliers = sub[sub["is_inlier"]].copy()
        outliers = sub[~sub["is_inlier"]].copy()

        # save per-mode classification CSV
        class_csv = OUT_DIR / f"{mode}_inlier_outlier_table.csv"
        sub.to_csv(class_csv, index=False)

        print(f"{mode}: total={len(sub)}, inliers={len(inliers)}, outliers={len(outliers)}")
        print(f"Saved: {class_csv}")

        if inliers.empty:
            print(f"Skipping averaged matrix for {mode}: no inliers.")
            continue

        # average inlier parameters
        mean_dx = inliers["dx"].mean()
        mean_dy = inliers["dy"].mean()
        mean_rotation = inliers["rotation_deg"].mean()
        mean_scale_x = inliers["scale_x"].mean()
        mean_scale_y = inliers["scale_y"].mean()
        mean_shear = inliers["shear_deg"].mean()

        M = build_affine_matrix(
            dx=mean_dx,
            dy=mean_dy,
            rotation_deg=mean_rotation,
            scale_x=mean_scale_x,
            scale_y=mean_scale_y,
            shear_deg=mean_shear,
            center=CENTER,
        )

        # save averaged matrix
        cam = MODE_TO_CAMERA[mode]
        matrix_path = OUT_DIR / f"{mode}_to_{cam}_global_affine.txt"
        np.savetxt(matrix_path, M, fmt="%.10f")

        # save summary text
        txt_path = OUT_DIR / f"{mode}_to_{cam}_global_affine_summary.txt"
        with open(txt_path, "w") as f:
            f.write(f"mode: {mode}\n")
            f.write(f"camera: {cam}\n")
            f.write(f"n_total: {len(sub)}\n")
            f.write(f"n_inliers: {len(inliers)}\n")
            f.write(f"n_outliers: {len(outliers)}\n\n")

            f.write("Averaged inlier parameters:\n")
            f.write(f"dx: {mean_dx:.6f}\n")
            f.write(f"dy: {mean_dy:.6f}\n")
            f.write(f"rotation_deg: {mean_rotation:.6f}\n")
            f.write(f"scale_x: {mean_scale_x:.6f}\n")
            f.write(f"scale_y: {mean_scale_y:.6f}\n")
            f.write(f"shear_deg: {mean_shear:.6f}\n\n")

            f.write("Affine matrix:\n")
            for row in M:
                f.write(" ".join(f"{v:+.10f}" for v in row) + "\n")

        print(f"Saved: {matrix_path}")
        print(f"Saved: {txt_path}")

        summary_rows.append({
            "mode": mode,
            "camera": cam,
            "n_total": len(sub),
            "n_inliers": len(inliers),
            "n_outliers": len(outliers),
            "dx": mean_dx,
            "dy": mean_dy,
            "rotation_deg": mean_rotation,
            "scale_x": mean_scale_x,
            "scale_y": mean_scale_y,
            "shear_deg": mean_shear,
            "matrix_file": str(matrix_path),
        })

    if summary_rows:
        summary_csv = OUT_DIR / "global_affine_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"Saved: {summary_csv}")

if __name__ == "__main__":
    main()