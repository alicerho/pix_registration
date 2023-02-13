import numpy as np
import pandas as pd
from pathlib import Path
from skimage import transform

df_coords = pd.read_csv("intermediate/coordinations_new_beads.csv")
df_coords = df_coords.astype({"by":"str","detector":"str","color":"str"})


# fill the camera coordinates that didn't show beads
list_df = [df_coords]
for fov,by in ((2,"detector"),(2,"color"),(3,"detector")):
    stack = []
    for color in ["DAPI","FITC","YFP","TRITC"]:
        df_camera = df_coords.loc[
            (
                df_coords["FOV"].eq(fov)
              & df_coords["by"].eq(by)
              & df_coords["color"].eq(color)
            ),
            ['x','y']
        ].dropna()
        if len(df_camera) == 0:
            continue 
        stack.append(df_camera.to_numpy())
    ave_camera = np.mean(np.stack(stack,axis=0),axis=0)
    list_df.append(pd.DataFrame({
        "FOV": fov,
        "by": by,
        "detector": "camera",
        "color": "CFP",
        "index": np.arange(ave_camera.shape[0]),
        "y": ave_camera[:,1],
        "x": ave_camera[:,0]
    }))
    if fov==2 and by=="color":
        list_df.append(pd.DataFrame({
            "FOV": fov,
            "by": by,
            "detector": "camera",
            "color": "YFP",
            "index": np.arange(ave_camera.shape[0]),
            "y": ave_camera[:,1],
            "x": ave_camera[:,0]
        }))
df_filled = pd.concat(list_df,ignore_index=True)

maps = (
    ("blue","DAPI"),
    ("blue","CFP"),
    ("blue","FITC"),
    ("green","DAPI"),
    ("yellow","YFP"),
    ("red","TRITC"),
)

for fov,by in ((2,"detector"),(2,"color"),(3,"detector")):
    for c_src,c_dst in maps:
        xy_src = df_filled.loc[
                    (
                        df_filled["FOV"].eq(fov)
                      & df_filled["by"].eq(by)
                      & df_filled["detector"].eq("spectral")
                      & df_filled["color"].eq(c_src)
                    ),
                    ["x","y"]
                ].to_numpy()
        xy_dst = df_filled.loc[
                    (
                        df_filled["FOV"].eq(fov)
                      & df_filled["by"].eq(by)
                      & df_filled["detector"].eq("spectral")
                      & df_filled["color"].eq(c_dst)            
                    ),
                    ["x","y"]
                ].to_numpy()
        transf = transform.AffineTransform()
        success = transf.estimate(xy_src,xy_dst)
        if success:
            np.savetxt(f"intermediate/transforms/new-beads_{c_src}-512_{c_dst}-1024.txt",transf.params)

