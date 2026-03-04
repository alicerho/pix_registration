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


def center_crop(img):
    h, w = img.shape
    y0 = h // 2 - CROP_SIZE // 2
    x0 = w // 2 - CROP_SIZE // 2
    return img[y0:y0+CROP_SIZE, x0:x0+CROP_SIZE]


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

    A = np.loadtxt(GLOBAL_AFFINE)
    tform = AffineTransform(matrix=A)

    pattern = "unmixed_EYrainbow_slide-*_field-*_spectral-blue.nd2"
    files = sorted(UNMIXED.glob(pattern))

    for f in files:

        m = re.search(r"slide-(\d+)_field-(\d+)", f.name)
        slide, field = m.group(1), m.group(2)

        print(f"Processing slide {slide} field {field}")

        arr = load_nd2(f)

        # channel order: (2,H,W) OR (H,W,2)
        if arr.shape[0] == 2:
            px, vo = arr[0], arr[1]
        else:
            px, vo = arr[...,0], arr[...,1]

        # load camera
        cam_path = RAW / f"EYrainbow_slide-{slide}_field-{field}_spectral-blue.nd2"
        cam = load_nd2(cam_path)

        if cam.ndim == 3:
            cam = cam.max(axis=0)

        print("spectral:", px.shape)
        print("camera shape:", cam.shape)
        # cam = center_crop(cam)

        # warp px
        px_w = warp(
            px,
            inverse_map=tform.inverse,
            output_shape=cam.shape,
            preserve_range=True
        )

        # warp vo
        vo_w = warp(
            vo,
            inverse_map=tform.inverse,
            output_shape=cam.shape,
            preserve_range=True
        )

        prefix = f"registered_EYrainbow_slide-{slide}_field-{field}"

        save_tif(OUT / f"{prefix}_spectral-px.tif", px_w)
        save_tif(OUT / f"{prefix}_spectral-vo.tif", vo_w)

        # ---------- visualizations ----------

        # overlay
        plt.imshow(overlay_rgb(cam,px_w))
        plt.axis("off")
        plt.savefig(OUT / f"{prefix}_overlay.png",bbox_inches="tight",dpi=200)
        plt.close()

        # side by side
        fig,ax = plt.subplots(1,3,figsize=(12,4))
        ax[0].imshow(norm(cam),cmap="gray"); ax[0].set_title("camera")
        ax[1].imshow(norm(px),cmap="gray"); ax[1].set_title("spectral raw")
        ax[2].imshow(norm(px_w),cmap="gray"); ax[2].set_title("spectral warped")
        for a in ax: a.axis("off")
        plt.tight_layout()
        plt.savefig(OUT / f"{prefix}_side_by_side.png",dpi=200)
        plt.close()

        # checkerboard
        plt.imshow(checkerboard(cam,px_w),cmap="gray")
        plt.axis("off")
        plt.savefig(OUT / f"{prefix}_checkerboard.png",dpi=200)
        plt.close()


if __name__ == "__main__":
    main()