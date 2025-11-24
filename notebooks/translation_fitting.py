# import numpy as np
# from nd2reader import ND2Reader
# from skimage import io, img_as_ubyte
# import matplotlib.pyplot as plt
# from skimage.color import gray2rgb
# import tifffile as tiff
# import csv
# import os

# # ---------------------------------------------
# # Settings
# # ---------------------------------------------
# FOV = 23
# Z_START = 10
# Z_END = 20       # inclusive range for max projection
# CROP_SIZE = 512
# VIS_DIR = f"fov{FOV}_qc_outputs"


# # ---------------------------------------------
# # Helper: center crop
# # ---------------------------------------------
# def center_crop(img, size=512):
#     h, w = img.shape
#     start_y = (h - size) // 2
#     start_x = (w - size) // 2
#     return img[start_y:start_y+size, start_x:start_x+size]


# # ---------------------------------------------
# # Helper: scan between two 2D images
# # ---------------------------------------------
# def scan_dual(dual, range_y, range_x):
#     front = dual[0]
#     back = dual[1]
#     size_y, size_x = front.shape

#     coords = np.array([[(y, x) for x in range(*range_x)]
#                        for y in range(*range_y)])
#     diff = np.zeros((range_y[1] - range_y[0],
#                      range_x[1] - range_x[0]))

#     pad_top = -min(0, range_y[0])
#     pad_left = -min(0, range_x[0])
#     pad_bottom = max(0, range_y[1])
#     pad_right = max(0, range_x[1])

#     back_padded = np.pad(
#         back,
#         ((pad_top, pad_bottom), (pad_left, pad_right)),
#         mode='reflect'
#     )

#     for iy, y in enumerate(range(*range_y)):
#         for ix, x in enumerate(range(*range_x)):
#             window = back_padded[
#                 pad_top + y : pad_top + y + size_y,
#                 pad_left + x : pad_left + x + size_x
#             ]
#             diff[iy, ix] = np.sum((window - front)**2)

#     result = coords[np.unravel_index(np.argmin(diff), diff.shape)]
#     return result


# # ---------------------------------------------
# # Visualization helper (overlay)
# # ---------------------------------------------
# def save_overlay_png(cam_img, spec_img, outpath, title):
#     cam_norm = cam_img / cam_img.max() if cam_img.max() != 0 else cam_img
#     spec_norm = spec_img / spec_img.max() if spec_img.max() != 0 else spec_img

#     cam_rgb = np.stack((np.zeros_like(cam_norm),
#                         cam_norm,
#                         np.zeros_like(cam_norm)), axis=-1)
#     spec_rgb = np.stack((spec_norm,
#                          np.zeros_like(spec_norm),
#                          spec_norm), axis=-1)

#     overlay = cam_rgb + spec_rgb
#     overlay = np.clip(overlay, 0, 1)

#     plt.figure(figsize=(6, 6))
#     plt.imshow(overlay)
#     plt.title(title)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(outpath, dpi=300)
#     plt.close()


# # ---------------------------------------------
# # Ensure output directory
# # ---------------------------------------------
# os.makedirs(VIS_DIR, exist_ok=True)


# # ---------------------------------------------
# # Load CAMERA ND2 (PATCHED)
# # ---------------------------------------------
# cam_path = f"/Users/shinyeongrho/Library/CloudStorage/Box-Box/2025-10-02_Registration3D/fov-{FOV}_camera.nd2"

# with ND2Reader(cam_path) as cam:
#     cam.iter_axes = 'z'
#     cam.bundle_axes = 'cyx'
#     cam_data = np.array([frame for frame in cam])   # <-- PATCHED


# # max projection
# cam_proj = cam_data[Z_START:Z_END].max(axis=0)  # (5,1024,1024)

# # extract channels
# cam_blue   = cam_proj[0]
# cam_green  = cam_proj[2]
# cam_yellow = cam_proj[3]
# cam_red    = cam_proj[4]

# # crop to 512×512
# cam_blue_c   = center_crop(cam_blue)
# cam_green_c  = center_crop(cam_green)
# cam_yellow_c = center_crop(cam_yellow)
# cam_red_c    = center_crop(cam_red)

# camera_channels = [cam_blue_c, cam_green_c, cam_yellow_c, cam_red_c]
# color_names = ["Blue", "Green", "Yellow", "Red"]


# # ---------------------------------------------
# # Load SPECTRAL ND2 (PATCHED)
# # ---------------------------------------------
# spec_path = f"/Users/shinyeongrho/Library/CloudStorage/Box-Box/2025-10-02_Registration3D/fov-{FOV}_spectral.nd2"

# with ND2Reader(spec_path) as spec:
#     spec.iter_axes = 'z'
#     spec.bundle_axes = 'cyx'
#     spec_data = np.array([frame for frame in spec])   # <-- PATCHED

# spec_proj = spec_data[Z_START:Z_END].max(axis=0)  # (4,512,512)


# # ---------------------------------------------
# # Camera → Spectral shifts
# # ---------------------------------------------
# print("\n=== Camera → Spectral Translations ===")

# results_cam2spec = np.zeros((4, 2), dtype=int)

# for i in range(4):
#     shift = scan_dual(
#         np.stack((camera_channels[i], spec_proj[i]), axis=0),
#         (-40, 40),
#         (-40, 40)
#     )
#     results_cam2spec[i] = shift
#     print(f"{color_names[i]}: {shift}")

#     dy, dx = shift
#     shifted_spec = np.roll(np.roll(spec_proj[i], dy, axis=0), dx, axis=1)

#     save_overlay_png(
#         camera_channels[i],
#         shifted_spec,
#         f"{VIS_DIR}/overlay_{color_names[i].lower()}.png",
#         f"{color_names[i]} Overlay (Camera vs Shifted Spectral)"
#     )


# # ---------------------------------------------
# # Spectral ↔ Spectral 4×4 shifts
# # ---------------------------------------------
# print("\n=== Spectral ↔ Spectral Channels ===")

# results_spec2spec = np.zeros((4, 4, 2), dtype=int)

# for i in range(4):
#     for j in range(4):
#         shift = scan_dual(
#             np.stack((spec_proj[i], spec_proj[j]), axis=0),
#             (-20, 20),
#             (-20, 20)
#         )
#         results_spec2spec[i, j] = shift
#         print(f"S{i} → S{j}: {shift}")


# # ---------------------------------------------
# # Save multipage TIFFs for Fiji
# # ---------------------------------------------
# tiff.imwrite(
#     f"{VIS_DIR}/camera_crop_channels.tif",
#     np.stack(camera_channels, axis=0).astype('uint16')
# )

# tiff.imwrite(
#     f"{VIS_DIR}/spectral_proj_channels.tif",
#     spec_proj.astype('uint16')
# )


# # ---------------------------------------------
# # Save translation CSV
# # ---------------------------------------------
# csvfile = f"{VIS_DIR}/translations_fov{FOV}.csv"
# with open(csvfile, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Type", "From", "To", "dy", "dx"])

#     # cam→spec
#     for i in range(4):
#         writer.writerow(["cam2spec", color_names[i], color_names[i],
#                          results_cam2spec[i][0], results_cam2spec[i][1]])

#     # spec→spec
#     for i in range(4):
#         for j in range(4):
#             dy, dx = results_spec2spec[i, j]
#             writer.writerow(["spec2spec", i, j, dy, dx])

# print(f"\nAll visualizations + CSV saved in: {VIS_DIR}\n")

import numpy as np
from nd2reader import ND2Reader
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
import tifffile as tiff
import csv
import os

# ---------------------------------------------
# Settings
# ---------------------------------------------
FOV = 23
Z_START = 10
Z_END = 20       # inclusive range for max projection
CROP_SIZE = 512
VIS_DIR = f"fov{FOV}_qc_outputs"


# ---------------------------------------------
# Helper: center crop
# ---------------------------------------------
def center_crop(img, size=512):
    h, w = img.shape
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    return img[start_y:start_y+size, start_x:start_x+size]


# ---------------------------------------------
# Helper: scan between two 2D images
# ---------------------------------------------
def scan_dual(dual, range_y, range_x):
    front = dual[0]
    back = dual[1]
    size_y, size_x = front.shape

    coords = np.array([[(y, x) for x in range(*range_x)]
                       for y in range(*range_y)])
    diff = np.zeros((range_y[1] - range_y[0],
                     range_x[1] - range_x[0]))

    pad_top = -min(0, range_y[0])
    pad_left = -min(0, range_x[0])
    pad_bottom = max(0, range_y[1])
    pad_right = max(0, range_x[1])

    back_padded = np.pad(
        back,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='reflect'
    )

    for iy, y in enumerate(range(*range_y)):
        for ix, x in enumerate(range(*range_x)):
            window = back_padded[
                pad_top + y : pad_top + y + size_y,
                pad_left + x : pad_left + x + size_x
            ]
            diff[iy, ix] = np.sum((window - front)**2)

    result = coords[np.unravel_index(np.argmin(diff), diff.shape)]
    return result


# ---------------------------------------------
# Visualization: before/after overlays
# ---------------------------------------------
def save_overlay_png(cam_img, spec_img, outpath, title):
    cam_norm = cam_img / cam_img.max() if cam_img.max() != 0 else cam_img
    spec_norm = spec_img / spec_img.max() if spec_img.max() != 0 else spec_img

    cam_rgb = np.stack((np.zeros_like(cam_norm),
                        cam_norm,
                        np.zeros_like(cam_norm)), axis=-1)

    spec_rgb = np.stack((spec_norm,
                         np.zeros_like(spec_norm),
                         spec_norm), axis=-1)

    overlay = np.clip(cam_rgb + spec_rgb, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ---------------------------------------------
# Ensure output directory
# ---------------------------------------------
os.makedirs(VIS_DIR, exist_ok=True)


# ---------------------------------------------
# Load camera ND2 (PATCHED)
# ---------------------------------------------
cam_path = f"/Users/shinyeongrho/Library/CloudStorage/Box-Box/2025-10-02_Registration3D/fov-{FOV}_camera.nd2"

with ND2Reader(cam_path) as cam:
    cam.iter_axes = 'z'
    cam.bundle_axes = 'cyx'
    cam_data = np.array([frame for frame in cam])   # PATCHED

cam_proj = cam_data[Z_START:Z_END].max(axis=0)  # (5,1024,1024)

cam_blue   = cam_proj[0]
cam_green  = cam_proj[2]
cam_yellow = cam_proj[3]
cam_red    = cam_proj[4]

cam_blue_c   = center_crop(cam_blue)
cam_green_c  = center_crop(cam_green)
cam_yellow_c = center_crop(cam_yellow)
cam_red_c    = center_crop(cam_red)

camera_channels = [cam_blue_c, cam_green_c, cam_yellow_c, cam_red_c]
color_names = ["Blue", "Green", "Yellow", "Red"]


# ---------------------------------------------
# Load spectral ND2 (PATCHED)
# ---------------------------------------------
spec_path = f"/Users/shinyeongrho/Library/CloudStorage/Box-Box/2025-10-02_Registration3D/fov-{FOV}_spectral.nd2"

with ND2Reader(spec_path) as spec:
    spec.iter_axes = 'z'
    spec.bundle_axes = 'cyx'
    spec_data = np.array([frame for frame in spec])   # PATCHED

spec_proj = spec_data[Z_START:Z_END].max(axis=0)  # (4,512,512)


# ---------------------------------------------
# Camera → Spectral shifts (WITH BEFORE/AFTER OVERLAYS)
# ---------------------------------------------
print("\n=== Camera → Spectral Translations ===")

results_cam2spec = np.zeros((4, 2), dtype=int)

for i in range(4):
    cam_img = camera_channels[i]
    spec_img = spec_proj[i]

    # ---- BEFORE OVERLAY (unaligned) ----
    save_overlay_png(
        cam_img,
        spec_img,
        f"{VIS_DIR}/before_overlay_{color_names[i].lower()}.png",
        f"{color_names[i]} BEFORE (Unaligned)"
    )

    # ---- Compute shift ----
    shift = scan_dual(
        np.stack((cam_img, spec_img), axis=0),
        (-40, 40),
        (-40, 40)
    )
    results_cam2spec[i] = shift
    print(f"{color_names[i]} shift: {shift}")

    dy, dx = shift
    shifted_spec = np.roll(np.roll(spec_img, dy, axis=0), dx, axis=1)

    # ---- AFTER OVERLAY ----
    save_overlay_png(
        cam_img,
        shifted_spec,
        f"{VIS_DIR}/after_overlay_{color_names[i].lower()}.png",
        f"{color_names[i]} AFTER (Aligned)"
    )


# ---------------------------------------------
# Spectral ↔ Spectral 4×4 shifts
# ---------------------------------------------
print("\n=== Spectral ↔ Spectral Channels ===")

results_spec2spec = np.zeros((4, 4, 2), dtype=int)

for i in range(4):
    for j in range(4):
        shift = scan_dual(
            np.stack((spec_proj[i], spec_proj[j]), axis=0),
            (-20, 20),
            (-20, 20)
        )
        results_spec2spec[i, j] = shift
        print(f"S{i} → S{j}: {shift}")


# ---------------------------------------------
# Save TIFFs for Fiji
# ---------------------------------------------
tiff.imwrite(
    f"{VIS_DIR}/camera_crop_channels.tif",
    np.stack(camera_channels, axis=0).astype('uint16')
)

tiff.imwrite(
    f"{VIS_DIR}/spectral_proj_channels.tif",
    spec_proj.astype('uint16')
)


# ---------------------------------------------
# Save translation CSV
# ---------------------------------------------
csvfile = f"{VIS_DIR}/translations_fov{FOV}.csv"
with open(csvfile, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Type", "From", "To", "dy", "dx"])

    # cam→spec
    for i in range(4):
        dy, dx = results_cam2spec[i]
        writer.writerow(["cam2spec", color_names[i], color_names[i], dy, dx])

    # spec→spec
    for i in range(4):
        for j in range(4):
            dy, dx = results_spec2spec[i, j]
            writer.writerow(["spec2spec", i, j, dy, dx])

print(f"\nAll visualization files + CSV saved in: {VIS_DIR}\n")