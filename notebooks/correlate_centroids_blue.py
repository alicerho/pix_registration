import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from nd2reader import ND2Reader
from skimage import util,io,measure,transform


# format images from nd2 to tif

def nd2tif(path_in,path_out):
    with ND2Reader(str(path_in)) as file_nd2:
        if 'c' in file_nd2.sizes.keys():
            file_nd2.bundle_axes = 'cyx'
            need_sum = True
        else:
            file_nd2.bundle_axes = 'yx'
            need_sum = False
        file_nd2.iter_axies  = 't'
        img_nd2 = file_nd2[0].astype(int)
        if need_sum:
            img_nd2 = np.sum(img_nd2,axis=0)
    io.imsave(str(path_out),util.img_as_uint(img_nd2))
    return None

list_files = [
    "EYrainbow-Whi5Up_CSM-complete_field-0_camera-CFP.nd2",
    "EYrainbow-Whi5Up_CSM-complete_field-0_camera-DAPI.nd2",
    "EYrainbow-Whi5Up_CSM-complete_field-0_spectral-blue.nd2",
    "EYrainbow-Whi5Up_CSM-complete_field-2_camera-CFP.nd2",
    "EYrainbow-Whi5Up_CSM-complete_field-2_camera-DAPI.nd2",
    "EYrainbow-Whi5Up_CSM-complete_field-2_spectral-blue.nd2"
]

for file in list_files:
    path_nd2 = Path("data/2022-12-12_CellsEqualPixels_blue")/file
    nd2tif(str(path_nd2),f"intermediate/tiff/{path_nd2.stem}.tif")

# use ilastik to segment both spectral and camera images

# turn 1-2 binary images to label images

def binary2label(path_in,path_out):
    img_binary = io.imread(str(path_in))
    img_binary = (img_binary>1)
    img_label  = measure.label(img_binary)
    io.imsave(str(path_out),util.img_as_uint(img_label))
    return None

for path_binary in Path("intermediate/segment").glob("EY*.tiff"):
    binary2label(
        str(path_binary),
        f"intermediate/label/{path_binary.stem.rpartition('_')[0]}_label.tif"
    )


# mapping labels between camera and spectral images (done manually from Fiji)


# turn mapping between labels to mapping between coordiantes 
# (max or weighted average)
dict_images = {
    "DAPI": {
        "camera": {
            "intensity": [
                "intermediate/tiff/EYrainbow-Whi5Up_CSM-complete_field-0_camera-DAPI.tif",
                "intermediate/tiff/EYrainbow-Whi5Up_CSM-complete_field-2_camera-DAPI.tif"
            ],
            "label":     [
                "intermediate/label/EYrainbow-Whi5Up_CSM-complete_field-0_camera-DAPI_label.tif",
                "intermediate/label/EYrainbow-Whi5Up_CSM-complete_field-2_camera-DAPI_label.tif"
            ]
        },
        "spectral": {
            "intensity": [
                "intermediate/tiff/EYrainbow-Whi5Up_CSM-complete_field-0_spectral-blue.tif",
                "intermediate/tiff/EYrainbow-Whi5Up_CSM-complete_field-2_spectral-blue.tif"
            ],
            "label":     [
                "intermediate/label/EYrainbow-Whi5Up_CSM-complete_field-0_spectral-blue_label.tif",
                "intermediate/label/EYrainbow-Whi5Up_CSM-complete_field-2_spectral-blue_label.tif"
            ]
        }
    },
    "CFP":    {
        "camera": {
            "intensity": [
                "intermediate/tiff/EYrainbow-Whi5Up_CSM-complete_field-0_camera-CFP.tif",
                "intermediate/tiff/EYrainbow-Whi5Up_CSM-complete_field-2_camera-CFP.tif"
            ],
            "label":     [
                "intermediate/label/EYrainbow-Whi5Up_CSM-complete_field-0_camera-CFP_label.tif",
                "intermediate/label/EYrainbow-Whi5Up_CSM-complete_field-2_camera-CFP_label.tif"
            ]
        },
        "spectral": {
            "intensity": [
                "intermediate/tiff/EYrainbow-Whi5Up_CSM-complete_field-0_spectral-blue.tif",
                "intermediate/tiff/EYrainbow-Whi5Up_CSM-complete_field-2_spectral-blue.tif"
            ],
            "label":     [
                "intermediate/label/EYrainbow-Whi5Up_CSM-complete_field-0_spectral-blue_label.tif",
                "intermediate/label/EYrainbow-Whi5Up_CSM-complete_field-2_spectral-blue_label.tif"
            ]
        }
    }
}


mappings  = pd.read_csv("intermediate/mapping_blue.csv")
list_coords = []
for color in ["DAPI","CFP"]:
    for f,field in enumerate([0,2]):
        # read images
        camera_intensity = io.imread(dict_images[color]["camera"]["intensity"][f])
        camera_label     = io.imread(dict_images[color]["camera"]["label"][f])
        spectral_intensity = io.imread(dict_images[color]["spectral"]["intensity"][f])
        spectral_label     = io.imread(dict_images[color]["spectral"]["label"][f])
        # measure 
        regionprops_camera   = measure.regionprops(label_image=camera_label,  intensity_image=camera_intensity)
        regionprops_spectral = measure.regionprops(label_image=spectral_label,intensity_image=spectral_intensity)
        # map to coordinates
        mapping = mappings[mappings["field"].eq(field)]
        for pair in mapping.iterrows():
            label_camera,label_spectral = pair[1][color],pair[1]["spectral"]

            centroid_camera_masked = regionprops_camera[label_camera-1].centroid
            centroid_camera_weight = regionprops_camera[label_camera-1].weighted_centroid
            centroid_camera_maxima = (np.array(regionprops_camera[label_camera-1].bbox[:2])
                                     +np.array(tuple(map(np.mean,
                                           np.where(
                                               regionprops_camera[label_camera-1].intensity_image
                                             ==regionprops_camera[label_camera-1].max_intensity)
                                     ))))

            centroid_spectral_masked = regionprops_spectral[label_spectral-1].centroid
            centroid_spectral_weight = regionprops_spectral[label_spectral-1].weighted_centroid
            centroid_spectral_maxima = (np.array(regionprops_spectral[label_spectral-1].bbox[:2])
                                       +np.array(tuple(map(np.mean,
                                           np.where(
                                               regionprops_spectral[label_spectral-1].intensity_image
                                             ==regionprops_spectral[label_spectral-1].max_intensity)
                                       ))))
            list_coords.append(pd.DataFrame({
                "color": [color]*3,
                "field": [field]*3,
                "camera_label": [label_camera]*3,
                "spectral_label": [label_spectral]*3,
                "camera_mode":   ["masked","weighted","maxima"],
                "spectral_mode": ["masked","weighted","maxima"],
                "camera_y": [centroid_camera_masked[0],centroid_camera_weight[0],centroid_camera_maxima[0]],
                "camera_x": [centroid_camera_masked[1],centroid_camera_weight[1],centroid_camera_maxima[1]],
                "spectral_y": [centroid_spectral_masked[0],centroid_spectral_weight[0],centroid_spectral_maxima[0]],
                "spectral_x": [centroid_spectral_masked[1],centroid_spectral_weight[1],centroid_spectral_maxima[1]]
            }))
df_coords = pd.concat(list_coords,ignore_index=True)
df_coords.to_csv("intermediate/coordinations_blue.csv",index=False)


# visualize the coordinates

df_coords = pd.read_csv("intermediate/coordinations_blue.csv")
df_coords.sort_values(["camera_mode","color","field","spectral_label","camera_label"],inplace=True)

df_coords_camera = df_coords.loc[:,["color","field","camera_label","camera_mode","camera_y","camera_x"]]
df_coords_camera["detector"] = "camera"
df_coords_camera.rename(
    columns={
        "camera_label": "label",
        "camera_mode":  "mode",
        "camera_y":     "y",
        "camera_x":     "x"
    },
    inplace=True
)


df_coords_spectral = df_coords.loc[:,["color","field","spectral_label","spectral_mode","spectral_y","spectral_x"]]
df_coords_spectral["detector"] = "spectral"
df_coords_spectral.rename(
    columns={
        "spectral_label": "label",
        "spectral_mode":  "mode",
        "spectral_y":     "y",
        "spectral_x":     "x"
    },
    inplace=True
)

df_coords_spectral['y'] = df_coords_spectral['y'] + 256
df_coords_spectral['x'] = df_coords_spectral['x'] + 256

df_coords_new = pd.concat([df_coords_camera,df_coords_spectral],ignore_index=True)

for mode in ["masked","weighted","maxima"]:
    fig = px.scatter(
        data_frame=df_coords_new[df_coords_new["mode"].eq(mode)],
        x="x",y="y",color="color",symbol="detector",size="field",
        title=f"{mode}; all colors"
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    ) # keep the aspect ratio of both axes.
    fig.write_html(f"intermediate/visual/2d_{mode}_bluechannels.html")
    for color in ["DAPI","CFP"]:
        fig = px.scatter(
            data_frame=df_coords_new[
                df_coords_new["color"].eq(color) & 
                df_coords_new["mode"].eq(mode)
            ],
            x="x",y="y",color="detector",
            title=f"{mode}; {color}"
        )
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        ) # keep the aspect ratio of both axes.
        fig.write_html(f"intermediate/visual/2d_{mode}_{color}.html")
# It looks 2d images are similar across different modes,
# and the colors make a difference, and the difference is consistent.
# So we needn't consider 3d transformations.


# fit the transformation
for mode in ["masked","weighted","maxima"]:
    for color in ["DAPI","CFP"]:
        df_subset = df_coords.loc[df_coords["color"].eq(color) & df_coords["camera_mode"].eq(mode)]
        yx_camera   = df_subset[["camera_y",  "camera_x"]].to_numpy()
        yx_spectral = df_subset[["spectral_y","spectral_x"]].to_numpy()
        for transf_type in ["EuclideanTransform","SimilarityTransform","AffineTransform"]:
            transf = getattr(transform,transf_type)()
            success = transf.estimate(yx_camera,yx_spectral)
            if success:
                np.savetxt(f"intermediate/transforms/camera2spectral_{color}_{mode}_{transf_type}.txt",transf.params)
            success = transf.estimate(yx_spectral,yx_camera)
            if success:
                np.savetxt(f"intermediate/transforms/spectral2camera_{color}_{mode}_{transf_type}.txt",transf.params)
        for f,field in enumerate([0,2]):
            df_subset = df_coords.loc[df_coords["color"].eq(color) & df_coords["camera_mode"].eq(mode) & df_coords["field"].eq(f)]
            yx_camera   = df_subset[["camera_y",  "camera_x"]].to_numpy()
            yx_spectral = df_subset[["spectral_y","spectral_x"]].to_numpy()
            for transf_type in ["EuclideanTransform","SimilarityTransform","AffineTransform"]:
                transf = getattr(transform,transf_type)()
                success = transf.estimate(yx_camera,yx_spectral)
                if success:
                    np.savetxt(f"intermediate/transforms/camera2spectral_{color}_{mode}_{transf_type}_field-{field}.txt",transf.params)
                success = transf.estimate(yx_spectral,yx_camera)
                if success:
                    np.savetxt(f"intermediate/transforms/spectral2camera_{color}_{mode}_{transf_type}_field-{field}.txt",transf.params)

# TODO
# backtest the transform parameters
list_coords_predict = [df_coords_new]
for mode in ["masked","weighted","maxima"]:
    for color in ["green","yellow","red"]:
        df_subset = df_coords.loc[df_coords["color"].eq(color) & df_coords["camera_mode"].eq(mode)]
        yx_camera   = df_subset[["camera_y",  "camera_x"]].to_numpy()
        yx_spectral = df_subset[["spectral_y","spectral_x"]].to_numpy()
        for transf_type in ["EuclideanTransform","SimilarityTransform","AffineTransform"]:
            matrix_spectral2camera = np.loadtxt(f"intermediate/transforms/spectral2camera_{color}_{mode}_{transf_type}.txt")
            a_spectral2camera = matrix_spectral2camera[:2 ,:2]
            b_spectral2camera = matrix_spectral2camera[:-1,-1].reshape((2,-1))
            predict_camera = np.transpose(b_spectral2camera + a_spectral2camera @ yx_spectral.transpose())
            list_coords_predict.append(pd.DataFrame({
                "color": color,
                "label": df_coords_new.loc[
                            df_coords_new["color"].eq(color) & 
                            df_coords_new["mode"].eq(mode) & 
                            df_coords_new["detector"].eq("camera"),
                            "label"
                         ],
                "mode": mode,
                "z": 0.,
                'y': predict_camera[:,0],
                'x': predict_camera[:,1],
                "detector": f"predicted_camera_{transf_type}"
            }))

            matrix_camera2spectral = np.loadtxt(f"intermediate/transforms/camera2spectral_{color}_{mode}_{transf_type}.txt")
            a_camera2spectral = matrix_camera2spectral[:2 ,:2]
            b_camera2spectral = matrix_camera2spectral[:-1,-1].reshape((2,-1))
            predict_spectral = np.transpose(b_camera2spectral + a_camera2spectral @ yx_camera.transpose())
            list_coords_predict.append(pd.DataFrame({
                "color": color,
                "label": df_coords_new.loc[
                            df_coords_new["color"].eq(color) & 
                            df_coords_new["mode"].eq(mode) & 
                            df_coords_new["detector"].eq("spectral"),
                            "label"
                         ],
                "mode": mode,
                "z": 0.,
                'y': predict_spectral[:,0] + 766,
                'x': predict_spectral[:,1] + 768,
                "detector": f"predicted_spectral_{transf_type}"
            }))
df_coords_prediction = pd.concat(list_coords_predict,ignore_index=True)

for mode in ["masked","weighted","maxima"]:
    for color in ["green","yellow","red"]:
        fig = px.scatter(
            data_frame=df_coords_prediction[
                df_coords_prediction["color"].eq(color) & 
                df_coords_prediction["mode"].eq(mode)
            ],
            x="x",y="y",color="detector",
            title=f"{mode}; {color}"
        )
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        ) # keep the aspect ratio of both axes.
        fig.write_html(f"intermediate/visual/test_{mode}_{color}.html")
