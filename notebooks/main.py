import numpy as np
from pathlib import Path 
from skimage import io,util,transform
from nd2reader import ND2Reader

def crop_center(path_i,path_o):
    """Crop the 512*512 pixels in the center of camera images"""
    with ND2Reader(str(path_i)) as pim_camera:
        # print(pim_camera.sizes)
        pim_camera.bundle_axes = "zyx"
        pim_camera.iter_axes = 't'
        img_camera = pim_camera[0]
    # print(img_camera.shape)
    _,y_dim,x_dim = img_camera.shape
    io.imsave(
        str(path_o),
        util.img_as_uint(
            img_camera[
                :,y_dim//2-256:y_dim//2+256,x_dim//2-256:x_dim//2+256
            ].astype(int)
        )
    )  
    return None

for path_camera in Path("data/2022-11-08_beads_equal_pixel_size").glob("camera*.nd2"):
    crop_center(
        path_camera,
        f"intermediate/center-crop_{path_camera.stem}.tif"
    )
# 2022-1-09: It turns out that the quality of `2022-11-08_beads_equal_pixel_size` is not good enough. 

crop_center(
    "data/camera-TRITC_field-0_size-2048_zoom-1.nd2",
    "intermediate/camera-TRITC_field-0_size-2048_zoom-1.tif"
)
# Almost sure there is a ~4 deg difference between spectra -90 and camera imgs


