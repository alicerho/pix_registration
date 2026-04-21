#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------
# CONFIG
# ------------------------

CSV_PATH = Path("batch_affine_results/affine_parameters_summary.csv")
GLOBAL_SUMMARY_PATH = Path("batch_affine_results/global_affine_from_inliers/global_affine_summary.csv")
OUT_DIR = Path("batch_affine_results/global_affine_from_inliers/parameter_distribution_plots_combined")
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
IQR_K = 1.5


# ------------------------
# HELPERS
# ------------------------

def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper


def mark_inliers(sub_df: pd.DataFrame) -> pd.DataFrame:
    sub = sub_df.copy()
    sub["is_inlier"] = True

    for param in PARAMS:
        lo, hi = iqr_bounds(sub[param], k=IQR_K)
        sub[f"{param}_inlier"] = (sub[param] >= lo) & (sub[param] <= hi)
        sub["is_inlier"] &= sub[f"{param}_inlier"]

    return sub


# ------------------------
# MAIN
# ------------------------

def main():
    df = pd.read_csv(CSV_PATH)
    global_df = pd.read_csv(GLOBAL_SUMMARY_PATH)

    if df.empty:
        print("Input CSV is empty.")
        return

    if global_df.empty:
        print("Global summary CSV is empty.")
        return

    # recompute inlier flags exactly as in averaging script
    marked = []
    for mode in MODES:
        sub = df[df["mode"] == mode].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["slide", "field"]).reset_index(drop=True)
        sub = mark_inliers(sub)
        marked.append(sub)

    if not marked:
        print("No mode data found.")
        return

    df_marked = pd.concat(marked, ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, param in zip(axes, PARAMS):
        for i, mode in enumerate(MODES, start=1):
            sub = df_marked[df_marked["mode"] == mode].copy()
            if sub.empty:
                continue

            values = sub[param].values
            inlier_mask = sub["is_inlier"].values
            outlier_mask = ~inlier_mask
            inlier_values = values[inlier_mask]

            global_row = global_df[global_df["mode"] == mode]
            if global_row.empty:
                continue
            avg_val = float(global_row.iloc[0][param])

            # boxplot of inliers only
            if len(inlier_values) > 0:
                ax.boxplot(
                    inlier_values,
                    positions=[i],
                    widths=0.5,
                    showfliers=False,
                    patch_artist=False,
                    medianprops=dict(color="orange", linewidth=1.5),
                )

            # all points with jitter
            rng = np.random.default_rng(1000 + i)
            x_jitter = rng.normal(i, 0.045, size=len(values))

            # inlier points
            if np.any(inlier_mask):
                ax.scatter(
                    x_jitter[inlier_mask],
                    values[inlier_mask],
                    s=28,
                    alpha=0.8,
                )

            # outlier points
            if np.any(outlier_mask):
                ax.scatter(
                    x_jitter[outlier_mask],
                    values[outlier_mask],
                    s=42,
                    alpha=0.9,
                    marker="x",
                )

            # inlier mean: dashed line segment + diamond
            ax.hlines(
                avg_val,
                i - 0.32,
                i + 0.32,
                linestyles="--",
                linewidth=1.5,
            )
            ax.scatter(
                [i + 0.34],
                [avg_val],
                marker="D",
                s=70,
            )

        ax.set_xticks(range(1, len(MODES) + 1))
        ax.set_xticklabels(MODES)
        ax.set_title(param)
        ax.set_ylabel(param)

    # single legend for whole figure
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=6, label='inlier point'),
        Line2D([0], [0], marker='x', linestyle='None', markersize=7, label='outlier point'),
        Line2D([0], [0], color='orange', linewidth=1.5, label='inlier median'),
        Line2D([0], [0], color='C0', linestyle='--', linewidth=1.5, label='inlier mean'),
        Line2D([0], [0], marker='D', linestyle='None', markersize=7, label='mean marker'),
    ]
    fig.legend(handles=legend_handles, loc="upper right")

    plt.suptitle("Affine parameter distributions across modalities", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    out_path = OUT_DIR / "all_parameters_all_modes_combined.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()