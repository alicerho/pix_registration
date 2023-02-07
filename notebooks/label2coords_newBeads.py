import pandas as pd
from skimage import io,measure

df_label = pd.read_csv("intermediate/mapping_new_beads.csv")

list_fov      = []
list_by       = []
list_detector = []
list_color    = []
list_idx      = []
list_label    = []
list_y        = []
list_x        = []
for fov in [2,3]:
    for by in ['detector','color']:
        for spectral in ['blue','green','yellow','red']:
            labels = (df_label.loc[
                (
                    df_label['FOV'].eq(fov)
                  & df_label["by"].eq(by)
                ),
                f"spectral-{spectral}"
            ]).dropna()
            if len(labels)==0:
                continue
            img_tiff  = io.imread(f"intermediate/tiff/FOV-{fov}_PFS_by-{by}_spectral-{spectral}_px-512.tif")
            img_label = io.imread(f"intermediate/label/label_FOV-{fov}_PFS_by-{by}_spectral-{spectral}_px-512.tif")
            properties = measure.regionprops(
                                label_image=img_label,
                                intensity_image=img_tiff
                         )
            for idx,label in enumerate(labels):
                prop = properties[int(label)-1]
                y,x  = prop.weighted_centroid
                list_fov.append(fov)
                list_by.append(by)
                list_detector.append("spectral")
                list_color.append(spectral)
                list_idx.append(idx)
                list_label.append(int(label))
                list_y.append(y)
                list_x.append(x)
        for camera in ['DAPI','CFP','FITC','YFP','TRITC']:
            labels = (df_label.loc[
                (
                    df_label['FOV'].eq(fov)
                  & df_label["by"].eq(by)
                ),
                f"camera-{camera}"
            ]).dropna()
            if len(labels)==0:
                continue
            img_tiff  = io.imread(f"intermediate/tiff/FOV-{fov}_PFS_by-{by}_camera-{camera}_px-1024.tif")
            img_label = io.imread(f"intermediate/label/label_FOV-{fov}_PFS_by-{by}_camera-{camera}_px-1024.tif")
            properties = measure.regionprops(
                                label_image=img_label,
                                intensity_image=img_tiff
                         )
            for idx,label in enumerate(labels):
                prop = properties[int(label)-1]
                y,x  = prop.weighted_centroid
                list_fov.append(fov)
                list_by.append(by)
                list_detector.append("camera")
                list_color.append(camera)
                list_idx.append(idx)
                list_label.append(int(label))
                list_y.append(y)
                list_x.append(x)
df_coords = pd.DataFrame({
    "FOV":      list_fov,
    "by":       list_by,
    "detector": list_detector,
    "color":    list_color,
    "index":    list_idx,
    "label":    list_label,
    "y":        list_y,
    "x":        list_x
})
df_coords.to_csv("intermediate/coordinations_new_beads.csv",index=False)
