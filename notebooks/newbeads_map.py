# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import util,io,measure

# %%
# label segmented images
stems = [
    "FOV-2_PFS_by-color",
    "FOV-2_PFS_by-detector",
    "FOV-3_PFS_by-detector"
]

colors = {
    "camera":   ["DAPI","FITC","YFP","TRITC"],
    "spectral": ["blue","green","yellow","red"]
}

# %%
for fov in stems:
    for detector in colors.keys():
        for color in colors[detector]:
            if fov=="FOV-2_PFS_by-color" and detector=="camera" and color=="YFP":
                continue
            path_segment = list(Path("intermediate/segment").glob(f"Simple Segmentation_{fov}_{detector}-{color}*.tiff"))[0]
            img_segment = io.imread(str(path_segment))
            img_segment = (img_segment>1)
            img_label = measure.label(img_segment)
            io.imsave(
                f"intermediate/label/label_{path_segment.stem.partition('_')[2]}.tif",
                util.img_as_uint(img_label)
            )

# %%
# on ilastik, 
# correlate coordinates between channels of cameras and spectral detectors.

# %%
df_coords = pd.read_csv("intermediate/coordinations_new_beads.csv")

# %%
# assign average coords to CFP camera channel.
# %% already done in transform_new_beads.py
for stem in stems:
    list_y = []
    list_x = []
    fov = int(stem.partition("FOV-")[2].partition("_"))
    by  = stem.partition("by-")[2]
    for color in colors["camera"]:
        list_y.append(
            df_coords.loc[
                (
                    df_coords["FOV"].eq(fov)
                  & df_coords["by"].eq(by)
                  & df_coords["detector"].eq("camera")
                  & df_coords["color"].eq(color)
                ),
                "y"
            ].to_numpy()
        )
        list_x.append(
            df_coords.loc[
                (
                    df_coords["FOV"].eq(fov)
                  & df_coords["by"].eq(by)
                  & df_coords["detector"].eq("camera")
                  & df_coords["color"].eq(color)
                ),
                "x"
            ].to_numpy()
        )
