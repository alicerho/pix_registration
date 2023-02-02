# This script tests if the problem of affine transform results from the 
# different traditions of coordinates (row,column) vs. (x,y)

# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import util,io,transform,filters

# %%
df_coords = pd.read_csv("intermediate/coordinations.csv")
# %%
transf_matrices = {}
transf_matrices["green-FITC"] = np.loadtxt("intermediate/transforms/spectral2camera_green_weighted_AffineTransform.txt")
# %%
transforms = {}
transforms["green-FITC"] = transform.AffineTransform(transf_matrices["green-FITC"])
# %%
coords_camera = {}
coords_camera["FITC"] = df_coords.loc[(df_coords["color"].eq("green") & df_coords["camera_mode"].eq("weighted")),["camera_y","camera_x"]].to_numpy().astype(int)
coords_spectral = {}
coords_spectral["green"] = df_coords.loc[(df_coords["color"].eq("green") & df_coords["camera_mode"].eq("weighted")),["spectral_y","spectral_x"]].to_numpy().astype(int)
# %%
img_camera = np.zeros((2044,2048))
for coord in coords_camera["FITC"]:
    img_camera[tuple(coord)] = 1.
img_camera = filters.gaussian(img_camera)
io.imsave("intermediate/coords_traditions/FITC.tif",util.img_as_float32(img_camera))
# %%
img_spectral = np.zeros((512,512),dtype=int)
for coord in coords_spectral["green"]:
    img_spectral[tuple(coord)] = 1.
img_spectral = filters.gaussian(img_spectral)
io.imsave("intermediate/coords_traditions/green.tif",util.img_as_float32(img_spectral))

# %%
img_compare = np.zeros((3,2044,2048))
predicted_coords = np.transpose(transforms["green-FITC"].params[:2,:2] @ np.transpose(coords_camera["FITC"]) + transforms["green-FITC"].params[:2,-1].reshape((-1,1)))
for coord in predicted_coords:
    img_compare[tuple([0,*coord.astype(int)])] = 1.
img_compare[0] = filters.gaussian(img_compare[0])
img_compare[1] = img_camera
img_compare[2] = transform.warp(img_spectral,inverse_map=transforms["green-FITC"].inverse,output_shape=(2044,2048))
io.imsave("intermediate/coords_traditions/compare_FITC_green.tif",util.img_as_float32(img_compare))
# %%
