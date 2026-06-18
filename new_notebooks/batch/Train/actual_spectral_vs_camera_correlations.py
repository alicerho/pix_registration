#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
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

AFFINE_DIR = (
    PROJECT_ROOT
    / "batch_affine_results"
    / "all_300FOV_affine_matrices"
)

TEST_CSV = ROOT / "results" / "baseline_test_predictions.csv"

OUT_DIR = ROOT / "results" / "camera_vs_spectral"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_SIZE = 512

# ------------------------
# CHANNELS
# ------------------------

CAMERAS = ["DAPI", "FITC", "TRITC", "YFP"]

SPECTRAL = [
    "er",
    "go",
    "px",
    "vo",
    "mt",
    "ld",
]

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

def load_nd2(path):
    with nd2.ND2File(str(path)) as f:
        arr = f.asarray()

    return np.squeeze(arr).astype(np.float32)


def normalize_01(img):

    img = img.astype(np.float32)

    img -= np.nanmin(img)

    mx = np.nanmax(img)

    if mx > 0:
        img /= mx

    return img


def center_crop(img, crop_size):

    h, w = img.shape

    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2

    return img[y0:y0+crop_size, x0:x0+crop_size]


def load_camera(slide, field, channel):

    path = (
        RAW
        / f"EYrainbow_slide-{slide}_field-{field}_camera-{channel}.nd2"
    )

    img = load_nd2(path)

    while img.ndim > 2:
        img = img.max(axis=0)

    if img.shape != (512,512):
        img = center_crop(img, 512)

    return normalize_01(img)


def load_affine_matrix(slide, field, mode):

    path = (
        AFFINE_DIR
        / f"EYrainbow_slide-{slide}_field-{field}_{mode}.txt"
    )

    return np.loadtxt(path)


def apply_affine(img, M):

    tform = AffineTransform(matrix=M)

    warped = warp(
        img,
        inverse_map=tform.inverse,
        output_shape=(512,512),
        preserve_range=True,
        mode="constant",
        cval=0,
        order=1,
    )

    return warped.astype(np.float32)


def load_registered_spectral(slide, field):

    out = {}

    for mode, info in PAIRS.items():

        if info["spectral_file_type"] == "raw":

            path = (
                RAW
                / f"EYrainbow_slide-{slide}_field-{field}_spectral-{info['spectral_color']}.nd2"
            )

        else:

            path = (
                UNMIXED
                / f"unmixed_EYrainbow_slide-{slide}_field-{field}_spectral-{info['spectral_color']}.nd2"
            )

        arr = load_nd2(path)

        M = load_affine_matrix(slide, field, mode)

        if mode in ["green", "yellow"]:

            while arr.ndim > 2:
                arr = arr.max(axis=0)

            warped = apply_affine(arr, M)

            out[info["spectral_channels"][0]] = normalize_01(warped)

        else:

            out[info["spectral_channels"][0]] = normalize_01(
                apply_affine(arr[0], M)
            )

            out[info["spectral_channels"][1]] = normalize_01(
                apply_affine(arr[1], M)
            )

    return out


def corr(a,b):

    a = a.ravel()
    b = b.ravel()

    mask = np.isfinite(a) & np.isfinite(b)

    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return np.nan

    return np.corrcoef(a,b)[0,1]


# ------------------------
# MAIN
# ------------------------

def main():

    test_df = pd.read_csv(TEST_CSV)

    fovs = (
        test_df[["slide","field"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Using {len(fovs)} held-out FOVs")

    corr_sum = np.zeros((6,4))
    count = 0

    for row in fovs.itertuples(index=False):

        slide = int(row.slide)
        field = int(row.field)

        print(f"slide-{slide}_field-{field}")

        cameras = {
            c: load_camera(slide, field, c)
            for c in CAMERAS
        }

        spectral = load_registered_spectral(
            slide,
            field,
        )

        for i, spec in enumerate(SPECTRAL):
            for j, cam in enumerate(CAMERAS):

                corr_sum[i,j] += corr(
                    spectral[spec],
                    cameras[cam],
                )

        count += 1

    corr_mean = corr_sum / count

    df = pd.DataFrame(
        corr_mean,
        index=SPECTRAL,
        columns=CAMERAS,
    )

    print("\nMean correlations:")
    print(df)

    df.to_csv(
        OUT_DIR / "spectral_vs_camera_correlations.csv"
    )

    plt.figure(figsize=(6,5))

    plt.imshow(
        corr_mean,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    plt.xticks(
        np.arange(len(CAMERAS)),
        CAMERAS,
        rotation=45,
    )

    plt.yticks(
        np.arange(len(SPECTRAL)),
        SPECTRAL,
    )

    for i in range(len(SPECTRAL)):
        for j in range(len(CAMERAS)):

            plt.text(
                j,
                i,
                f"{corr_mean[i,j]:.2f}",
                ha="center",
                va="center",
            )

    plt.title(
        "Actual Spectral vs Camera Correlation"
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "spectral_vs_camera_heatmap.png",
        dpi=200,
    )

    plt.close()

    print("\nSaved heatmap.")

if __name__ == "__main__":
    main()