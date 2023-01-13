# The old mapping between camera and spectral detectors assumed that the 
# size camera images are 2028*2044.
# This one assumes 1024*1024

import numpy as np
import pandas as pd
from pathlib import Path
from skimage import transform

df_heads = pd.read_csv("intermediate/coordinations.csv")
df_heads = df_heads[df_heads["camera_mode"].eq("weighted")]

df_blues = pd.read_csv("intermediate/coordinations_blue.csv")
df_blues = df_blues[df_blues["camera_mode"].eq("weighted")]

df_coord = pd.concat([df_heads,df_blues],ignore_index=True)

for old in ["green","yellow","red"]:
    df_coord.loc[df_coord["color"].eq(old),"camera_y"] = df_coord.loc[df_coord["color"].eq(old),"camera_y"] - 510
    df_coord.loc[df_coord["color"].eq(old),"camera_x"] = df_coord.loc[df_coord["color"].eq(old),"camera_x"] - 512

for color in ["DAPI","CFP","green","yellow","red"]:
    yx_camera   = df_coord[["camera_y",  "camera_x"]].to_numpy()
    yx_spectral = df_coord[["spectral_y","spectral_x"]].to_numpy()
    
    transf_camera2spectral = transform.AffineTransform()
    success_camera2spectral = transf_camera2spectral.estimate(yx_camera,yx_spectral)
    if success_camera2spectral:
        np.savetxt(f"intermediate/final/camera-1024_spectral-512_{color}_weighted_Affine.txt",transf_camera2spectral.params)
    
    transf_spectral2camera = transform.AffineTransform()
    success_spectral2camera = transf_spectral2camera.estimate(yx_spectral,yx_camera)
    if success_spectral2camera:
        np.savetxt(f"intermediate/final/spectral-512_camera-1024_{color}_weighted_Affine.txt",transf_spectral2camera.params)
