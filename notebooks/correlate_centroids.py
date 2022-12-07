# I first use ilastik to segment both spectral and camera images, 
# then find the centroids inside each components 
# weighted by the intensity in the original image.
# Hopefully the rotation is within the plane, 
# 

import numpy as np
import pandas as pd
from pathlib import Path
from nd2reader import ND2Reader
from skimage import util,io,measure


# format images from nd2 to tif

def nd2tif(path_in,path_out):
    with ND2Reader(str(path_in)) as file_nd2:
        if 'c' in file_nd2.sizes.keys():
            file_nd2.bundle_axes = 'czyx'
            need_sum = True
        else:
            file_nd2.bundle_axes = 'zyx'
            need_sum = False
        file_nd2.iter_axies  = 't'
        img_nd2 = file_nd2[0].astype(int)
        if need_sum:
            img_nd2 = np.sum(img_nd2,axis=0)
    io.imsave(str(path_out),util.img_as_uint(img_nd2))
    return None

for path_nd2 in Path("data/2022-11-08_beads_equal_pixel_size").glob("camera*.nd2"):
    nd2tif(str(path_nd2),f"tiff/{path_nd2.stem}.tif")
for path_nd2 in Path("data/2022-11-08_beads_equal_pixel_size").glob("spectra*.nd2"):
    nd2tif(str(path_nd2),f"tiff/{path_nd2.stem}.tif")


# turn 1-2 binary images to label images

def binary2label(path_in,path_out):
    img_binary = io.imread(str(path_in))
    img_binary = (img_binary>1)
    img_label  = measure.label(img_binary)
    io.imsave(str(path_out),util.img_as_uint(img_label))
    return None

for path_binary in Path("intermediate/segment").glob("*.tiff"):
    binary2label(
        str(path_binary),
        f"intermediate/label/{path_binary.stem.rpartition('_')[0]}_label.tif"
    )


# mapping labels between camera and spectral images (done manually from Fiji)


# turn mapping between labels to mapping between coordiantes 
# (max or weighted average)
dict_images = {
    "green":  {
        "camera": {
            "intensity": "intermediate/tiff/camera-DAPI_zStack_Beads-100nm_field-0.tif",
            "label":     "intermediate/label/camera-FITC_zStack_Beads-100nm_field-0_label.tif"
        },
        "spectral": {
            "intensity": "intermediate/tiff/spectra-green_zStack_Beads-100nm_field-0.tif",
            "label":     "intermediate/label/spectra-green_zStack_Beads-100nm_field-0_label.tif"
        }
    },
    "yellow": {
        "camera": {
            "intensity": "intermediate/tiff/camera-YFP_zStack_Beads-100nm_field-0.tif",
            "label":     "intermediate/label/camera-YFP_zStack_Beads-100nm_field-0_label.tif"
        },
        "spectral": {
            "intensity": "intermediate/tiff/spectra-yellow_zStack_Beads-100nm_field-0.tif",
            "label":     "intermediate/label/spectra-yellow_zStack_Beads-100nm_field-0_label.tif"
        }
    },
    "red":    {
        "camera": {
            "intensity": "intermediate/tiff/camera-TRITC_zStack_Beads-100nm_field-0.tif",
            "label":     "intermediate/label/camera-TRITC_zStack_Beads-100nm_field-0_label.tif"
        },
        "spectral": {
            "intensity": "intermediate/tiff/spectra-red_zStack_Beads-100nm_field-0.tif",
            "label":     "intermediate/label/spectra-red_zStack_Beads-100nm_field-0_label.tif"
        }
    }
}

mappings  = pd.read_csv("intermediate/mapping.csv")
list_coords = []
for color in ["green","yellow","red"]:
    # read images
    camera_intensity = io.imread(dict_images[color]["camera"]["intensity"])
    camera_label     = io.imread(dict_images[color]["camera"]["label"])
    spectral_intensity = io.imread(dict_images[color]["spectral"]["intensity"])
    spectral_label     = io.imread(dict_images[color]["spectral"]["label"])
    # measure 
    regionprops_camera   = measure.regionprops(label_image=camera_label,  intensity_image=camera_intensity)
    regionprops_spectral = measure.regionprops(label_image=spectral_label,intensity_image=spectral_intensity)
    # map to coordinates
    mapping = mappings[mappings["channel"].eq(color)]
    for pair in mapping.iterrows():
        label_camera,label_spectral = pair[1]["camera"],pair[1]["spectral"]

        centroid_camera_masked = regionprops_camera[label_camera-1].centroid
        centroid_camera_weight = regionprops_camera[label_camera-1].weighted_centroid
        centroid_camera_maxima = (np.array(regionprops_camera[label_camera-1].bbox[:3])
                                 +np.array(tuple(map(np.mean,
                                       np.where(
                                           regionprops_camera[label_camera-1].intensity_image
                                         ==regionprops_camera[label_camera-1].max_intensity)
                                 ))))

        centroid_spectral_masked = regionprops_camera[label_spectral-1].centroid
        centroid_spectral_weight = regionprops_camera[label_spectral-1].weighted_centroid
        centroid_spectral_maxima = (np.array(regionprops_camera[label_spectral-1].bbox[:3])
                                   +np.array(tuple(map(np.mean,
                                       np.where(
                                           regionprops_spectral[label_spectral-1].intensity_image
                                         ==regionprops_spectral[label_spectral-1].max_intensity)
                                   ))))
        list_coords.append(pd.DataFrame({
            "color": [color]*3,
            "camera_label": [label_camera]*3,
            "spectral_label": [label_spectral]*3,
            "camera_mode":   ["masked","weighted","maxima"],
            "spectral_mode": ["masked","weighted","maxima"],
            "camera_z": [centroid_camera_masked[0],centroid_camera_weight[0],centroid_camera_maxima[0]],
            "camera_y": [centroid_camera_masked[1],centroid_camera_weight[1],centroid_camera_maxima[1]],
            "camera_x": [centroid_camera_masked[2],centroid_camera_weight[2],centroid_camera_maxima[2]],
            "spectral_z": [centroid_spectral_masked[0],centroid_spectral_weight[0],centroid_spectral_maxima[0]],
            "spectral_y": [centroid_spectral_masked[1],centroid_spectral_weight[1],centroid_spectral_maxima[1]],
            "spectral_x": [centroid_spectral_masked[2],centroid_spectral_weight[2],centroid_spectral_maxima[2]]
        }))
df_coords = pd.concat(list_coords,ignore_index=True)
df_coords.to_csv("intermediate/coordinations.csv",index=False)