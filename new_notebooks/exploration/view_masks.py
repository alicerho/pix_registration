import numpy as np
from skimage import io

mask = io.imread("../../data/Masks/FOV-1_spectral-blue.tif")

print(mask.shape)                 # image size
print(mask.dtype)                # usually int32 / uint16
print(np.unique(mask)[:20])      # first 20 unique labels
print("Num objects:", len(np.unique(mask)) - 1)  # minus background (0)