# #!/usr/bin/env python3

# from pathlib import Path
# import numpy as np
# import re
# import nd2
# import matplotlib.pyplot as plt
# from skimage.transform import AffineTransform, warp
# from skimage.io import imsave


# # ------------------------
# # CONFIG
# # ------------------------

# DATASET = Path("../../data/Dataset_300Fovs")
# UNMIXED = DATASET / "unmixed"
# RAW = DATASET / "RAW"
# OUT = DATASET / "registered"

# GLOBAL_AFFINE = Path("blue_to_DAPI_global_affine.txt")

# CROP_SIZE = 512


# # ------------------------
# # helpers
# # ------------------------

# def load_nd2(path):
#     with nd2.ND2File(str(path)) as f:
#         arr = f.asarray()
#     return np.squeeze(arr).astype(np.float32)


# def center_crop(img):
#     h, w = img.shape
#     y0 = h // 2 - CROP_SIZE // 2
#     x0 = w // 2 - CROP_SIZE // 2
#     return img[y0:y0+CROP_SIZE, x0:x0+CROP_SIZE]


# def norm(img):
#     img = img.astype(float)
#     img -= img.min()
#     if img.max() > 0:
#         img /= img.max()
#     return img


# def save_tif(path, img):
#     img = norm(img)
#     imsave(str(path), (img*65535).astype(np.uint16))


# def overlay_rgb(a, b):
#     a = norm(a)
#     b = norm(b)
#     rgb = np.zeros((*a.shape,3))
#     rgb[...,0] = a
#     rgb[...,1] = b
#     return rgb


# def checkerboard(a,b,tile=32):
#     h,w = a.shape
#     yy,xx = np.indices((h,w))
#     mask = ((yy//tile + xx//tile) % 2)==0
#     out = np.where(mask,a,b)
#     return norm(out)


# # ------------------------
# # main
# # ------------------------

# def main():

#     OUT.mkdir(exist_ok=True)

#     A = np.loadtxt(GLOBAL_AFFINE)
#     tform = AffineTransform(matrix=A)

#     pattern = "unmixed_EYrainbow_slide-*_field-*_spectral-blue.nd2"
#     files = sorted(UNMIXED.glob(pattern))

#     for f in files:

#         m = re.search(r"slide-(\d+)_field-(\d+)", f.name)
#         slide, field = m.group(1), m.group(2)

#         print(f"Processing slide {slide} field {field}")

#         arr = load_nd2(f)

#         # channel order: (2,H,W) OR (H,W,2)
#         if arr.shape[0] == 2:
#             px, vo = arr[0], arr[1]
#         else:
#             px, vo = arr[...,0], arr[...,1]

#         # load camera
#         cam_path = RAW / f"EYrainbow_slide-{slide}_field-{field}_camera-DAPI.nd2"
#         cam = load_nd2(cam_path)

#         if cam.ndim == 3:
#             cam = cam.max(axis=0)

#         print("spectral:", px.shape)
#         print("camera shape:", cam.shape)
#         center_cam = (cam.shape[1]/2, cam.shape[0]/2)   # (x,y)
#         center_spec = (px.shape[1]/2, px.shape[0]/2)
#         print("center_cam:", center_cam)
#         print("center_spec:", center_spec)
#         print("center offset (cam - spec):", (center_cam[0]-center_spec[0], center_cam[1]-center_spec[1]))
#         # cam = center_crop(cam)

#         # warp px
#         px_w = warp(
#             px,
#             inverse_map=tform.inverse,
#             output_shape=cam.shape,
#             preserve_range=True
#         )

#         # warp vo
#         vo_w = warp(
#             vo,
#             inverse_map=tform.inverse,
#             output_shape=cam.shape,
#             preserve_range=True
#         )

#         prefix = f"registered_EYrainbow_slide-{slide}_field-{field}"

#         save_tif(OUT / f"{prefix}_spectral-px.tif", px_w)
#         save_tif(OUT / f"{prefix}_spectral-vo.tif", vo_w)

#         # ---------- visualizations ----------

#         # overlay
#         plt.imshow(overlay_rgb(cam,px_w))
#         plt.axis("off")
#         plt.savefig(OUT / f"{prefix}_overlay.png",bbox_inches="tight",dpi=200)
#         plt.close()

#         # side by side
#         fig,ax = plt.subplots(1,3,figsize=(12,4))
#         ax[0].imshow(norm(cam),cmap="gray"); ax[0].set_title("camera")
#         ax[1].imshow(norm(px),cmap="gray"); ax[1].set_title("spectral raw")
#         ax[2].imshow(norm(px_w),cmap="gray"); ax[2].set_title("spectral warped")
#         for a in ax: a.axis("off")
#         plt.tight_layout()
#         plt.savefig(OUT / f"{prefix}_side_by_side.png",dpi=200)
#         plt.close()

#         # checkerboard
#         plt.imshow(checkerboard(cam,px_w),cmap="gray")
#         plt.axis("off")
#         plt.savefig(OUT / f"{prefix}_checkerboard.png",dpi=200)
#         plt.close()


# if __name__ == "__main__":
#     main()


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


def convert_xy_to_rc_transform(A_xy):
    """
    Convert affine transform from (x,y) Cartesian coordinates 
    to (row,col) array indexing coordinates.
    
    In (x,y): x is horizontal, y is vertical (up)
    In (row,col): row is vertical (down), col is horizontal
    
    So: col ↔ x, row ↔ y (but y points down in array coords)
    
    Args:
        A_xy: 3x3 affine matrix in (x,y) format
        
    Returns:
        A_rc: 3x3 affine matrix in (row,col) format
    """
    A_rc = np.eye(3)
    
    # Swap axes: (x,y) -> (row,col) means (x,y) -> (col,row)... wait, that's not quite right
    # Actually: row = y, col = x
    # So we need to swap the transform components appropriately
    
    # A_xy maps (x,y) -> (x',y')
    # We need A_rc to map (row,col) -> (row',col')
    # where row=y, col=x
    
    # A_xy is:
    # [a11  a12  tx]     [x]     [x']
    # [a21  a22  ty]  *  [y]  =  [y']
    # [0    0    1 ]     [1]     [1 ]
    
    # We want A_rc:
    # [b11  b12  tr]     [row]     [row']
    # [b21  b22  tc]  *  [col]  =  [col']
    # [0    0    1 ]     [1  ]     [1   ]
    
    # Since row'=y' and col'=x':
    # col' = a11*x + a12*y + tx = a11*col + a12*row + tx
    # row' = a21*x + a22*y + ty = a21*col + a22*row + ty
    
    # So:
    A_rc[0, 0] = A_xy[1, 1]  # row-row component (y-y)
    A_rc[0, 1] = A_xy[1, 0]  # row-col component (y-x)
    A_rc[0, 2] = A_xy[1, 2]  # row translation (ty)
    A_rc[1, 0] = A_xy[0, 1]  # col-row component (x-y)
    A_rc[1, 1] = A_xy[0, 0]  # col-col component (x-x)
    A_rc[1, 2] = A_xy[0, 2]  # col translation (tx)
    
    return A_rc


# ------------------------
# main
# ------------------------

def main():

    OUT.mkdir(exist_ok=True)

    # Load affine matrix (in x,y coordinates)
    A_xy = np.loadtxt(GLOBAL_AFFINE)
    
    print("Original affine matrix (x,y format):")
    print(A_xy)
    
    # Convert to row,col coordinates for skimage.transform.warp
    A_rc = convert_xy_to_rc_transform(A_xy)
    
    print("\nConverted affine matrix (row,col format):")
    print(A_rc)
    
    tform = AffineTransform(matrix=A_rc)

    pattern = "unmixed_EYrainbow_slide-*_field-*_spectral-blue.nd2"
    files = sorted(UNMIXED.glob(pattern))

    for f in files:

        m = re.search(r"slide-(\d+)_field-(\d+)", f.name)
        slide, field = m.group(1), m.group(2)

        print(f"\nProcessing slide {slide} field {field}")

        arr = load_nd2(f)

        # channel order: (2,H,W) OR (H,W,2)
        if arr.shape[0] == 2:
            px, vo = arr[0], arr[1]
        else:
            px, vo = arr[...,0], arr[...,1]

        # load camera
        cam_path = RAW / f"EYrainbow_slide-{slide}_field-{field}_camera-DAPI.nd2"
        cam = load_nd2(cam_path)

        if cam.ndim == 3:
            cam = cam.max(axis=0)

        print(f"  Spectral shape: {px.shape}")
        print(f"  Camera shape: {cam.shape}")
        
        # Optional: crop camera to 512x512 if needed
        # cam = center_crop(cam)

        # Warp spectral images to camera coordinates
        # Using inverse because warp() does backward mapping (pulls pixels)

        print(f"got rid of inverse")
        px_w = warp(
            px,
            inverse_map=tform,
            output_shape=cam.shape,
            preserve_range=True
        )

        vo_w = warp(
            vo,
            inverse_map=tform,
            output_shape=cam.shape,
            preserve_range=True
        )

        prefix = f"registered_EYrainbow_slide-{slide}_field-{field}"

        save_tif(OUT / f"{prefix}_spectral-px.tif", px_w)
        save_tif(OUT / f"{prefix}_spectral-vo.tif", vo_w)

        # ---------- visualizations ----------

        # overlay
        plt.imshow(overlay_rgb(cam, px_w))
        plt.axis("off")
        plt.title(f"Overlay: Red=Camera, Green=Spectral")
        plt.savefig(OUT / f"{prefix}_overlay.png", bbox_inches="tight", dpi=200)
        plt.close()

        # side by side
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(norm(cam), cmap="gray")
        ax[0].set_title("Camera (fixed)")
        ax[1].imshow(norm(px), cmap="gray")
        ax[1].set_title("Spectral raw")
        ax[2].imshow(norm(px_w), cmap="gray")
        ax[2].set_title("Spectral warped")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(OUT / f"{prefix}_side_by_side.png", dpi=200)
        plt.close()

        # checkerboard
        plt.imshow(checkerboard(cam, px_w), cmap="gray")
        plt.axis("off")
        plt.title("Checkerboard: Camera vs Warped Spectral")
        plt.savefig(OUT / f"{prefix}_checkerboard.png", dpi=200)
        plt.close()

        print(f"  Saved registered images and visualizations")


if __name__ == "__main__":
    main()