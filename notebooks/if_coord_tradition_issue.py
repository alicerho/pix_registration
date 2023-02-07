# This script tests if the problem of affine transform results from the 
# different traditions of coordinates (row,column) vs. (x,y)

# Use green channel as an initial test.
# It turns out the old fitted transform is already a little confusing.
# Although the backtest did work, there were a lot of implicit wild 
# implicit coordinate transforms.

# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import util,io,transform,filters

# %%
df_coords = pd.read_csv("intermediate/coordinations.csv")

# %%
coords_camera = df_coords.loc[
        (
            df_coords["color"].eq("green")
          & df_coords["camera_mode"].eq("weighted")
        ),
        ["camera_y","camera_x"]
    ].to_numpy().astype(int)
coords_spectral = df_coords.loc[
        (
            df_coords["color"].eq("green")
          & df_coords["camera_mode"].eq("weighted")
        ),
        ["spectral_y","spectral_x"]
    ].to_numpy().astype(int)

# %%
coords_camera_xy = df_coords.loc[
        (
            df_coords["color"].eq("green")
          & df_coords["camera_mode"].eq("weighted")
        ),
        ["camera_x","camera_y"]
    ].to_numpy().astype(int)
coords_spectral_xy = df_coords.loc[
        (
            df_coords["color"].eq("green")
          & df_coords["camera_mode"].eq("weighted")
        ),
        ["spectral_x","spectral_y"]
    ].to_numpy().astype(int)


# %%
img_camera = np.zeros((2044,2048))
for coord in coords_camera:
    img_camera[tuple(coord)] = 1.
img_camera = filters.gaussian(img_camera)
# %%
io.imsave("intermediate/coords_traditions/FITC.tif",util.img_as_float32(img_camera))

# %%
img_spectral = np.zeros((512,512),dtype=int)
for coord in coords_spectral:
    img_spectral[tuple(coord)] = 1.
img_spectral = filters.gaussian(img_spectral)
# %%
io.imsave("intermediate/coords_traditions/green.tif",util.img_as_float32(img_spectral))

# %%
# https://github.com/scikit-image/scikit-image/issues/3856
transf = transform.AffineTransform()
transf.estimate(src=coords_spectral,dst=coords_camera)
transf_reverse = transform.AffineTransform()
transf_reverse.estimate(src=coords_camera,dst=coords_spectral)

# %%
transf_xy = transform.AffineTransform()
transf_xy.estimate(src=coords_spectral_xy,dst=coords_camera_xy)
img_transf_xy = transform.warp(
    img_spectral,
    transf_xy.inverse,
    output_shape=(2044,2048)
)
img_transf_xy = (img_transf_xy-np.unique(img_transf_xy)[1])/(np.unique(img_transf_xy)[-1]-np.unique(img_transf_xy)[1])
img_transf_xy[img_transf_xy<0] = 0.
# %%
io.imsave(
    "intermediate/coords_traditions/transf_xy.tif",
    util.img_as_float32(img_transf_xy)
)

# %%
from sklearn.linear_model import LinearRegression
np_fit_src = coords_spectral
np_fit_dst = coords_camera

fitter = LinearRegression()
fitter.fit(np_fit_src,np_fit_dst)
# print(fitter.coef_)
# print(fitter.intercept_)
# print(np.linalg.det(fitter.coef_))

matrix_affine = np.zeros((3,3))
matrix_affine[:2,:2] = fitter.coef_
matrix_affine[:2,2]  = fitter.intercept_
matrix_affine[2,2]   = 1
matrix_inv = np.linalg.inv(matrix_affine)

# %%
img_compare = np.zeros((5,2044,2048))

coords_camera_from_spectral = np.transpose(
        transf.params[:-1,-1].reshape((-1,1)) + 
        transf.params[:2,:2] @ np.transpose(coords_spectral)
    )
for coord in coords_camera_from_spectral:
    img_compare[tuple([0,*coord.astype(int)])] = 1.
img_compare[0] = filters.gaussian(img_compare[0])

img_compare[1] = img_camera

img_compare[2] = transform.warp(img_spectral,transf.inverse,output_shape=(2044,2048))
img_compare[2] = (img_compare[2]-np.unique(img_compare[2])[1])/(np.unique(img_compare[2])[-1]-np.unique(img_compare[2])[1])
img_compare[2,img_compare[2]<0] = 0.

img_compare[3] = transform.warp(img_spectral,transf_reverse,output_shape=(2044,2048))
img_compare[3] = (img_compare[3]-np.unique(img_compare[3])[1])/(np.unique(img_compare[3])[-1]-np.unique(img_compare[3])[1])
img_compare[3,img_compare[3]<0] = 0.

img_compare[4] = transform.warp(img_spectral,matrix_inv,output_shape=(2044,2048))
img_compare[4] = (img_compare[4]-np.unique(img_compare[4])[1])/(np.unique(img_compare[4])[-1]-np.unique(img_compare[4])[1])
img_compare[4,img_compare[4]<0] = 0.
# %%
io.imsave("intermediate/coords_traditions/compare.tif",util.img_as_float32(img_compare))

# %%
io.imsave("intermediate/coords_traditions/warp.tif",util.img_as_float32(img_compare[2]))


# We can conclude that the `transform.warp` is the problem! 
# It turns out that the coordinates to estimate the transform obey the 
# convension of (x,y) instead of (row, column).