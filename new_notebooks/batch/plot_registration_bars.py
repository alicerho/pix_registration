import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------
# CONFIG
# --------------------
CSV_PATH = "outputs_2d/evaluate_2d_masks_summary.csv"
OUT_PNG = "registration_accuracy_barplot.png"

# --------------------
# LOAD DATA
# --------------------
df = pd.read_csv(CSV_PATH)

# Drop rows where evaluation failed
df = df.dropna(subset=[
    "before_mean_nn_px",
    "affine_mean_nn_px",
    "syn_mean_nn_px"
])

# --------------------
# AGGREGATE
# --------------------
means = {
    "Before": df["before_mean_nn_px"].mean(),
    "Affine": df["affine_mean_nn_px"].mean(),
    "Affine + SyN": df["syn_mean_nn_px"].mean(),
}

stds = {
    "Before": df["before_mean_nn_px"].std(),
    "Affine": df["affine_mean_nn_px"].std(),
    "Affine + SyN": df["syn_mean_nn_px"].std(),
}

labels = list(means.keys())
y = [means[l] for l in labels]
yerr = [stds[l] for l in labels]

# --------------------
# PLOT
# --------------------
plt.figure(figsize=(6, 5))
bars = plt.bar(
    labels,
    y,
    yerr=yerr,
    capsize=6,
    color=["gray", "steelblue", "darkorange"],
    alpha=0.85
)

plt.ylabel("Mean bead NN distance (pixels)")
plt.title("Registration accuracy using bead centroids")

# Annotate bars
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        f"{h:.1f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.show()

print("Saved bar plot to:", OUT_PNG)