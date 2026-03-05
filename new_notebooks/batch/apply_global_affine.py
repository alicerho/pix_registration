#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import re
import nd2
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage.io import imsave


# ------------------------
# CONFIG
# ------------------------

DATASET = Path("../../data/Dataset_300Fovs")
UNMIXED = DATASET / "unmixed"
RAW = DATASET / "RAW"
OUT = DATASET / "registered"

GLOBAL_AFFINE = Path("blue_to_DAPI_global_affine.txt")

CROP_SIZE = 512


# ------------------------
# helpers
# ------------------------

def load_nd2(path):
    with nd2.ND2File(str(path)) as f:
        arr = f.asarray()
    return np.squeeze(arr).astype(np.float32)


def center_crop(img, crop_size):
    h, w = img.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img[y0:y0+crop_size, x0:x0+crop_size]


def norm(img):
    img = img.astype(float)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


def save_tif(path, img):
    img = norm(img)
    imsave(str(path), (img*65535).astype(np.uint16))


def overlay_rgb(a, b):
    a = norm(a)
    b = norm(b)
    rgb = np.zeros((*a.shape,3))
    rgb[...,0] = a
    rgb[...,1] = b
    return rgb


def checkerboard(a,b,tile=32):
    h,w = a.shape
    yy,xx = np.indices((h,w))
    mask = ((yy//tile + xx//tile) % 2)==0
    out = np.where(mask,a,b)
    return norm(out)


# ------------------------
# main
# ------------------------

def main():

    OUT.mkdir(exist_ok=True)

    # Load affine matrix
    A = np.loadtxt(GLOBAL_AFFINE)
    
    print("Transform matrix:")
    print(A)
    print("\nRegistering in 512×512 space (cropping camera to center)")

    pattern = "unmixed_EYrainbow_slide-*_field-*_spectral-blue.nd2"
    files = sorted(UNMIXED.glob(pattern))

    for f in files:

        m = re.search(r"slide-(\d+)_field-(\d+)", f.name)
        slide, field = m.group(1), m.group(2)

        print(f"\nProcessing slide {slide} field {field}")

        arr = load_nd2(f)

        if arr.shape[0] == 2:
            px, vo = arr[0], arr[1]
        else:
            px, vo = arr[...,0], arr[...,1]

        cam_path = RAW / f"EYrainbow_slide-{slide}_field-{field}_camera-DAPI.nd2"
        cam = load_nd2(cam_path)

        if cam.ndim == 3:
            cam = cam.max(axis=0)

        print(f"  Spectral: {px.shape}")
        print(f"  Camera (full): {cam.shape}")
        
        # CROP CAMERA TO 512×512 CENTER
        cam_cropped = center_crop(cam, CROP_SIZE)
        print(f"  Camera (cropped): {cam_cropped.shape}")
        
        # Use transform in the 512×512 space where it was computed
        tform = AffineTransform(matrix=A)

        # Warp to 512×512 space
        px_w = warp(
            px,
            inverse_map=tform.inverse,
            output_shape=cam_cropped.shape,
            preserve_range=True
        )

        vo_w = warp(
            vo,
            inverse_map=tform.inverse,
            output_shape=cam_cropped.shape,
            preserve_range=True
        )

        print(f"  Warped: {px_w.shape}, nonzero={np.sum(px_w > 0)}")

        prefix = f"registered_EYrainbow_slide-{slide}_field-{field}"

        save_tif(OUT / f"{prefix}_spectral-px.tif", px_w)
        save_tif(OUT / f"{prefix}_spectral-vo.tif", vo_w)

        # ---------- visualizations ----------
        
        # BEFORE/AFTER overlay comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # BEFORE: Camera vs raw spectral (misaligned)
        axes[0].imshow(overlay_rgb(cam_cropped, px))
        axes[0].set_title("BEFORE Registration\nRed=Camera, Green=Spectral (misaligned)")
        axes[0].axis("off")
        
        # AFTER: Camera vs warped spectral (aligned)
        axes[1].imshow(overlay_rgb(cam_cropped, px_w))
        axes[1].set_title("AFTER Registration\nRed=Camera, Green=Spectral (aligned)")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(OUT / f"{prefix}_overlay_before_after.png", bbox_inches="tight", dpi=200)
        plt.close()

        # side by side
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(norm(cam_cropped), cmap="gray")
        ax[0].set_title("Camera (512×512 center crop)")
        ax[1].imshow(norm(px), cmap="gray")
        ax[1].set_title("Spectral raw (512×512)")
        ax[2].imshow(norm(px_w), cmap="gray")
        ax[2].set_title("Spectral warped (512×512)")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(OUT / f"{prefix}_side_by_side.png", dpi=200)
        plt.close()

        # checkerboard
        plt.imshow(checkerboard(cam_cropped, px_w), cmap="gray")
        plt.axis("off")
        plt.title("Checkerboard: Camera vs Warped Spectral (both 512×512)")
        plt.savefig(OUT / f"{prefix}_checkerboard.png", dpi=200)
        plt.close()

        print(f"  Saved registered images and visualizations (512×512)")


if __name__ == "__main__":
    main()