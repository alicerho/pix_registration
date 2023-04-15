import numpy as np
import pandas as pd
from pathlib import Path
from skimage import util,io,transform
from PIL import Image
from batch_apply import batch_apply

PATH_DATA = Path(r"D:\Documents\FILES\lab_ARCHIVE")

matrix_pex = np.loadtxt("intermediate/transforms/new-beads_blue-512_CFP-1024.txt")
transf_pex = transform.AffineTransform(matrix_pex)

matrix_vph = np.loadtxt("intermediate/transforms/new-beads_blue-512_CFP-1024.txt")
transf_vph = transform.AffineTransform(matrix_vph)

matrix_sec = np.loadtxt("intermediate/transforms/new-beads_blue-512_CFP-1024.txt")
transf_sec = transform.AffineTransform(matrix_sec)


def register_blue_probability(idx:int):
    stem  = f"FOV-{idx}"
    
    stack_in = np.zeros((3,1024,1024))
    for c,cam in enumerate(['BF','CFP','FITC']):
        name = f"{stem}_camera-{cam}.tif"
        image = io.imread(str(PATH_DATA/"2023-02-07_PFS-blue-tiff"/name))
        stack_in[c] = (image-image.min())/(image.max()-image.min())
    io.imsave(
        str(PATH_DATA/"2023-02-07_PFS-blue-train"/f"in_{stem}.tif"),
        util.img_as_float32(stack_in[:,272:752,272:752])
    )

    stack_out = np.zeros((3,1024,1024))

    pex = io.imread(str(PATH_DATA/"2023-02-07_PFS-blue-ilastik"/f"{stem}_unmixed-peroxisome_Probabilities.tiff"))
    # for spec,transf in zip([1,2],(transf_pex,transf_vph)):
    img_pex = transform.warp(pex,transf_pex.inverse,output_shape=(1024,1024))
    mini = np.unique(img_pex)[1]
    img_pex = (img_pex-mini)/(img_pex.max()-mini)
    img_pex[img_pex<0] = 0
    stack_out[0] = img_pex

    vac = io.imread(str(PATH_DATA/"2023-02-07_PFS-blue-ilastik"/f"{stem}_unmixed-vacuole_Probabilities.tiff"))
    img_vac = transform.warp(vac,transf_vph.inverse,output_shape=(1024,1024))
    mini = np.unique(img_vac)[1]
    img_vac = (img_vac-mini)/(img_vac.max()-mini)
    img_vac[img_vac<0] = 0
    stack_out[1] = img_vac

    green = io.imread(str(PATH_DATA/"2023-02-07_PFS-blue-ilastik"/f"{stem}_spectral-green_Probabilities.tiff"))
    img_er = transform.warp(green[...,1],transf_sec.inverse,output_shape=(1024,1024))
    mini = np.unique(img_er)[1]
    img_er = (img_er-mini)/(img_er.max()-mini)
    img_er[img_er<0] = 0.
    stack_out[-1] = img_er

    io.imsave(
        str(PATH_DATA/"2023-02-07_PFS-blue-train"/f"out_{stem}.tif"),
        util.img_as_float32(stack_out[:,272:752,272:752])
    )
    return None

args = pd.DataFrame({
    "idx": np.arange(2,71)
})
# batch_apply(register_blue_probability,args)


def register_blue_spectral(idx:int):
    stem  = f"FOV-{idx}"

    stack_spectral = np.zeros((3,1024,1024))

    spectral_peroxisome = io.imread(str(PATH_DATA/"2023-02-07_PFS-blue-channels"/f"{stem}_unmixed-peroxisome.tif"))
    register_peroxisome = transform.warp(spectral_peroxisome,transf_pex.inverse,output_shape=(1024,1024))
    register_peroxisome = (register_peroxisome - register_peroxisome.min())/(register_peroxisome.max() - register_peroxisome.min())
    stack_spectral[0] = register_peroxisome

    spectral_vacuole = io.imread(str(PATH_DATA/"2023-02-07_PFS-blue-channels"/f"{stem}_unmixed-vacuole.tif"))
    register_vacuole = transform.warp(spectral_vacuole,transf_vph.inverse,output_shape=(1024,1024))
    register_vacuole = (register_vacuole - register_vacuole.min())/(register_vacuole.max() - register_vacuole.min())
    stack_spectral[1] = register_vacuole

    spectral_ER = io.imread(str(PATH_DATA/"2023-02-07_PFS-blue-tiff"/f"{stem}_spectral-green.tif"))
    register_ER = transform.warp(spectral_ER,transf_sec.inverse,output_shape=(1024,1024))
    register_ER = (register_ER - register_ER.min())/(register_ER.max() - register_ER.min())
    stack_spectral[2] = register_ER

    io.imsave(
        str(PATH_DATA/"2023-02-07_PFS-blue-train"/f"spectral_{stem}.tif"),
        util.img_as_float32(stack_spectral[:,272:752,272:752])
    )

    return None

args = pd.DataFrame({
    "idx": np.arange(2,71)
})
batch_apply(register_blue_spectral,args)
