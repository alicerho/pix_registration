
#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import nd2
import matplotlib.pyplot as plt

from skimage.transform import AffineTransform, warp
from sklearn.linear_model import LinearRegression
from scipy.ndimage import uniform_filter


# ------------------------
# CONFIG
# ------------------------

DATASET = Path("../../data/Dataset_300Fovs")

RAW = DATASET / "RAW"
UNMIXED = DATASET / "unmixed"

PRESENCE_CSV = Path("fov_file_presence_table.csv")

AFFINE_DIR = Path("batch_affine_results/all_300FOV_affine_matrices")

OUT_DIR = Path("correlated_pixel_database")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "top_correlated_pixels.csv"

CROP_SIZE = 512

# keep closest fraction of pixels to regression line
TOP_FRACTION = 0.05

# optional regression fitting subsample
MAX_FIT_PIXELS = 50000

# plotting subsample
MAX_PLOT_PIXELS = 50000


# ------------------------
# MODALITY CONFIG
# ------------------------

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


def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:

    h, w = img.shape

    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2

    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def normalize_01(img: np.ndarray) -> np.ndarray:

    img = img.astype(np.float32)

    img = img - np.nanmin(img)

    mx = np.nanmax(img)

    if mx > 0:
        img = img / mx

    return img


def load_camera(slide: int, field: int, channel: str) -> np.ndarray:

    path = RAW / f"EYrainbow_slide-{slide}_field-{field}_camera-{channel}.nd2"

    img = load_nd2(path)

    while img.ndim > 2:
        img = img.max(axis=0)

    if img.shape != (CROP_SIZE, CROP_SIZE):
        img = center_crop(img, CROP_SIZE)

    return img


def load_spectral(slide: int, field: int, mode: str) -> dict:

    info = PAIRS[mode]

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

    out = {}

    # green/yellow single-channel
    if mode in ["green", "yellow"]:

        while arr.ndim > 2:
            arr = arr.max(axis=0)

        out[info["spectral_channels"][0]] = arr

    # blue/red two-channel
    else:

        if arr.ndim != 3 or arr.shape[0] != 2:
            raise ValueError(
                f"Expected 2-channel spectral file for {mode}, "
                f"got shape {arr.shape} at {path}"
            )

        out[info["spectral_channels"][0]] = arr[0]
        out[info["spectral_channels"][1]] = arr[1]

    return out


def load_affine_matrix(slide: int, field: int, mode: str) -> np.ndarray:

    path = AFFINE_DIR / f"EYrainbow_slide-{slide}_field-{field}_{mode}.txt"

    M = np.loadtxt(path)

    if M.shape != (3, 3):
        raise ValueError(f"Expected 3x3 affine matrix, got {M.shape}: {path}")

    return M


def apply_affine_matrix(
    img: np.ndarray,
    M: np.ndarray,
    output_shape=(512, 512),
) -> np.ndarray:

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


def fit_regression_and_select(camera_img, spectral_img, top_fraction=0.05):
    """
    Fits spectral ~ camera, but only considers the top 30% brightest spectral pixels.
    Then selects pixels closest to the regression line.
    """

    win = 5

    camera_local = uniform_filter(camera_img.astype(np.float32), size=win)
    spectral_local = uniform_filter(spectral_img.astype(np.float32), size=win)

    cam = normalize_01(camera_local).ravel()
    spec = normalize_01(spectral_local).ravel()

    valid = np.isfinite(cam) & np.isfinite(spec)

    cam_all = cam[valid]
    spec_all = spec[valid]

    # keep only top 30% brightest spectral pixels
    spec_thresh = np.percentile(spec_all, 70)
    signal_mask = spec_all >= spec_thresh

    cam_signal = cam_all[signal_mask]
    spec_signal = spec_all[signal_mask]

    cam_valid = cam_signal.reshape(-1, 1)
    spec_valid = spec_signal

    # optional subsample for fitting
    n = len(spec_valid)

    if n > MAX_FIT_PIXELS:
        idx_fit = np.random.choice(n, MAX_FIT_PIXELS, replace=False)
        X_fit = cam_valid[idx_fit]
        y_fit = spec_valid[idx_fit]
    else:
        X_fit = cam_valid
        y_fit = spec_valid

    model = LinearRegression()
    model.fit(X_fit, y_fit)

    pred = model.predict(cam_valid)
    residual = spec_valid - pred
    abs_residual = np.abs(residual)

    # from the top 30% spectral pixels, keep closest TOP_FRACTION to regression line
    cutoff = np.percentile(abs_residual, top_fraction * 100)
    selected_signal = abs_residual <= cutoff

    # map back to full image
    full_selected = np.zeros_like(cam, dtype=bool)
    full_residual = np.full_like(cam, np.nan, dtype=np.float32)

    valid_indices = np.where(valid)[0]
    signal_indices = valid_indices[signal_mask]

    full_selected[signal_indices[selected_signal]] = True
    full_residual[signal_indices] = abs_residual.astype(np.float32)

    return (
        full_selected.reshape(camera_img.shape),
        full_residual.reshape(camera_img.shape),
        float(model.coef_[0]),
        float(model.intercept_),
    )


# ------------------------
# MAIN
# ------------------------

def main():

    presence = pd.read_csv(PRESENCE_CSV)

    complete = presence[
        presence["has_all_files"] == True
    ].copy()

    print(f"Using {len(complete)} complete FOVs")

    all_rows = []

    for row in complete.itertuples(index=False):

        slide = int(row.slide)
        field = int(row.field)

        print(f"\n=== Processing slide-{slide}_field-{field} ===")

        try:

            # ------------------------
            # load camera channels
            # ------------------------

            camera_imgs = {
                "DAPI": load_camera(slide, field, "DAPI"),
                "FITC": load_camera(slide, field, "FITC"),
                "TRITC": load_camera(slide, field, "TRITC"),
                "YFP": load_camera(slide, field, "YFP"),
            }

            camera_norm = {
                k: normalize_01(v)
                for k, v in camera_imgs.items()
            }

            # ------------------------
            # load + warp spectral channels
            # ------------------------

            spectral_norm = {}

            for mode in PAIRS.keys():

                M = load_affine_matrix(
                    slide,
                    field,
                    mode,
                )

                spectral_channels = load_spectral(
                    slide,
                    field,
                    mode,
                )

                for ch_name, img in spectral_channels.items():

                    warped = apply_affine_matrix(
                        img,
                        M,
                        output_shape=(CROP_SIZE, CROP_SIZE),
                    )

                    spectral_norm[ch_name] = normalize_01(warped)

            # ------------------------
            # regression selection
            # ------------------------

            for mode, info in PAIRS.items():

                cam_ch = info["camera"]

                for spec_ch in info["spectral_channels"]:

                    (
                        selected_mask,
                        residual_map,
                        slope,
                        intercept,
                    ) = fit_regression_and_select(
                        camera_norm[cam_ch],
                        spectral_norm[spec_ch],
                        top_fraction=TOP_FRACTION,
                    )

                    yy, xx = np.where(selected_mask)

                    # ------------------------
                    # REGRESSION QC PLOT
                    # ------------------------

                    cam_flat = camera_norm[cam_ch].ravel()

                    spec_flat = spectral_norm[spec_ch].ravel()

                    selected_flat = selected_mask.ravel()

                    n_plot = min(
                        MAX_PLOT_PIXELS,
                        len(cam_flat),
                    )

                    idx = np.random.choice(
                        len(cam_flat),
                        n_plot,
                        replace=False,
                    )

                    cam_plot = cam_flat[idx]
                    spec_plot = spec_flat[idx]
                    selected_plot = selected_flat[idx]

                    x_line = np.linspace(
                        cam_plot.min(),
                        cam_plot.max(),
                        200,
                    )

                    y_line = slope * x_line + intercept

                    plt.figure(figsize=(7, 7))

                    # all pixels
                    plt.scatter(
                        cam_plot[~selected_plot],
                        spec_plot[~selected_plot],
                        s=1,
                        alpha=0.1,
                        label="all pixels",
                    )

                    # selected pixels
                    plt.scatter(
                        cam_plot[selected_plot],
                        spec_plot[selected_plot],
                        s=2,
                        alpha=0.4,
                        label="selected pixels",
                    )

                    plt.plot(
                        x_line,
                        y_line,
                        linewidth=2,
                        label="regression",
                    )

                    plt.xlabel(cam_ch)
                    plt.ylabel(spec_ch)

                    plt.title(
                        f"slide-{slide}_field-{field}\n"
                        f"{cam_ch} -> {spec_ch}"
                    )

                    plt.legend()

                    plt.tight_layout()

                    plt.savefig(
                        OUT_DIR
                        / f"slide-{slide}_field-{field}_{cam_ch}_to_{spec_ch}_regression.png",
                        dpi=200,
                    )

                    plt.close()

                    # ------------------------
                    # SPATIAL QC PLOT
                    # ------------------------

                    plt.figure(figsize=(6, 6))

                    plt.imshow(
                        camera_norm[cam_ch],
                        cmap="gray",
                    )

                    overlay = np.zeros(
                        (*selected_mask.shape, 4)
                    )

                    overlay[..., 0] = 1.0

                    overlay[..., 3] = (
                        selected_mask.astype(float) * 0.7
                    )

                    plt.imshow(overlay)

                    plt.title(
                        f"Selected pixels\n"
                        f"{cam_ch} -> {spec_ch}"
                    )

                    plt.axis("off")

                    plt.tight_layout()

                    plt.savefig(
                        OUT_DIR
                        / f"slide-{slide}_field-{field}_{cam_ch}_to_{spec_ch}_selected_pixels.png",
                        dpi=200,
                    )

                    plt.close()

                    print(
                        f"  {cam_ch} -> {spec_ch}: "
                        f"selected {len(yy)} pixels "
                        f"(slope={slope:.4f}, "
                        f"intercept={intercept:.4f})"
                    )

                    # ------------------------
                    # SAVE DATABASE ROWS
                    # ------------------------

                    for y, x in zip(yy, xx):

                        all_rows.append({

                            "slide": slide,
                            "field": field,

                            "y": int(y),
                            "x": int(x),

                            "pair_used": f"{cam_ch}_to_{spec_ch}",

                            "residual_abs": float(
                                residual_map[y, x]
                            ),

                            "camera_DAPI": float(
                                camera_norm["DAPI"][y, x]
                            ),

                            "camera_FITC": float(
                                camera_norm["FITC"][y, x]
                            ),

                            "camera_TRITC": float(
                                camera_norm["TRITC"][y, x]
                            ),

                            "camera_YFP": float(
                                camera_norm["YFP"][y, x]
                            ),

                            "spectral_er": float(
                                spectral_norm["er"][y, x]
                            ),

                            "spectral_go": float(
                                spectral_norm["go"][y, x]
                            ),

                            "spectral_px": float(
                                spectral_norm["px"][y, x]
                            ),

                            "spectral_vo": float(
                                spectral_norm["vo"][y, x]
                            ),

                            "spectral_mt": float(
                                spectral_norm["mt"][y, x]
                            ),

                            "spectral_ld": float(
                                spectral_norm["ld"][y, x]
                            ),
                        })

        except Exception as e:

            print(
                f"ERROR on slide-{slide}_field-{field}: {e}"
            )

    # ------------------------
    # SAVE DATABASE
    # ------------------------

    df = pd.DataFrame(all_rows)

    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved database: {OUT_CSV}")

    print(f"Total selected pixels: {len(df)}")


if __name__ == "__main__":
    main()