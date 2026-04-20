#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# CONFIG
# ------------------------

CSV_PATH = Path("batch_affine_results/affine_parameters_summary.csv")
OUT_DIR = Path("batch_affine_results/parameter_plots")
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

# ------------------------
# MAIN
# ------------------------

def main():
    df = pd.read_csv(CSV_PATH)

    if df.empty:
        print("CSV is empty.")
        return

    # global y-limits per parameter across all modes
    param_limits = {}
    for param in PARAMS:
        vals = df[param].dropna()
        vmin, vmax = vals.min(), vals.max()
        pad = 0.05 * (vmax - vmin + 1e-8)
        param_limits[param] = (vmin - pad, vmax + pad)

    for param in PARAMS:
        # ------------------------
        # boxplots: one parameter, all modes side by side
        # ------------------------
        fig, axes = plt.subplots(1, len(MODES), figsize=(16, 4), sharey=True)

        for ax, mode in zip(axes, MODES):
            sub = df[df["mode"] == mode]
            data = sub[param].dropna()

            ax.boxplot(data, vert=True)

            # overlay jittered points
            x_jitter = np.random.normal(1, 0.04, size=len(data))
            ax.scatter(x_jitter, data, alpha=0.5, s=12)

            ax.set_title(mode)
            ax.set_ylim(param_limits[param])
            ax.set_xticks([])

            if ax is axes[0]:
                ax.set_ylabel(param)

        plt.suptitle(f"{param} across modes (boxplot + points)")
        plt.tight_layout()

        box_path = OUT_DIR / f"{param}_boxplots_all_modes.png"
        plt.savefig(box_path, dpi=200)
        plt.close()

        print(f"Saved: {box_path}")

        # ------------------------
        # scatterplots: one parameter, all modes side by side
        # ------------------------
        fig, axes = plt.subplots(1, len(MODES), figsize=(16, 4), sharey=True)

        for ax, mode in zip(axes, MODES):
            sub = df[df["mode"] == mode].reset_index(drop=True)

            ax.scatter(range(len(sub)), sub[param], s=18, alpha=0.7)
            ax.set_title(mode)
            ax.set_ylim(param_limits[param])
            ax.set_xlabel("FOV index")

            if ax is axes[0]:
                ax.set_ylabel(param)

        plt.suptitle(f"{param} across modes (scatter)")
        plt.tight_layout()

        scatter_path = OUT_DIR / f"{param}_scatter_all_modes.png"
        plt.savefig(scatter_path, dpi=200)
        plt.close()

        print(f"Saved: {scatter_path}")

if __name__ == "__main__":
    main()