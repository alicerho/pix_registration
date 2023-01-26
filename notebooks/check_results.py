import numpy as np
from skimage import util,io,transform
from nd2reader import ND2Reader

prefix = "EYrainbow-WTstar_CSM-complete_field-7"

for cam,spec,ch,transfile in (
    ("DAPI", "blue",  0,    "spectral-512_camera-1024_DAPI_weighted_Affine"),
    ("CFP",  "blue",  1,    "spectral-512_camera-1024_CFP_weighted_Affine"),
    ("FITC", "green", None, "spectral-512_camera-1024_green_weighted_Affine"),
    ("YFP",  "yellow",None, "spectral-512_camera-1024_yellow_weighted_Affine"),
    ("TRITC","red",   0,    "spectral-512_camera-1024_red_weighted_Affine"),
    ("TRITC","red",   1,    "spectral-512_camera-1024_red_weighted_Affine")
):
    
    if ch is None:
        idx = 0
        iter_axes = 't'
        suffix = 'spectral'
    else:
        idx = ch
        iter_axes = 'c'
        suffix = 'unmixed'
    with ND2Reader(f"data/2022-12-12_CellsEqualPixels_blue/{prefix}_camera-{cam}.nd2") as file_camera:
        file_camera.bundle_axes = 'yx'
        file_camera.iter_axes = 't'
        img_camera = file_camera[0].astype(int)
        img_camera = (img_camera-img_camera.min())/(img_camera.max()-img_camera.min())
    with ND2Reader(f"data/2022-12-12_CellsEqualPixels_blue/{prefix}_{suffix}-{spec}.nd2") as file_spectral:
        file_spectral.bundle_axes = 'yx'
        file_spectral.iter_axes = iter_axes
        img_spectral = file_spectral[idx].astype(int)
        img_spectral = (img_spectral-img_spectral.min())/(img_spectral.max()-img_spectral.min())
    np_transf  = np.loadtxt(f"intermediate/final/{transfile}.txt")
    transf = transform.AffineTransform(matrix=np_transf)
    img_transf = transform.warp(img_spectral,transf.inverse,output_shape=(1024,1024))
    img_transf = (img_transf - img_transf.min())/(img_transf.max() - img_transf.min())
    img_compare = np.stack([img_camera,img_transf])
    io.imsave(f"intermediate/compare/{prefix}_camera-{cam}_spectral-{spec}.tif",util.img_as_float32(img_compare))
