import numpy as np
from skimage import util,io
from nd2reader import ND2Reader


# %%
# find the translation vector between 2044*2048 and 1024*1024 images.
with ND2Reader("data/FOV-2_PFS_by-detector_camera-TRITC_px-2044.nd2") as file2044:
    file2044.iter_axes = 't'
    file2044.bundle_axes = 'yx'
    img2044 = file2044[0].astype(int)

with ND2Reader("data/FOV-2_PFS_by-detector_camera-TRITC_px-1024.nd2") as file1024:
    file1024.iter_axes = 't'
    file1024.bundle_axes = 'yx'
    img1024 = file1024[0].astype(int)

def scan(range_y,range_x):
    coords = np.array([[(y,x) for x in range(*range_x)] for y in range(*range_y)])
    difference = np.zeros((range_y[1]-range_y[0],range_x[1]-range_x[0]))
    for iy,y in enumerate(range(range_y[0],range_y[1])):
        for ix,x in enumerate(range(range_x[0],range_x[1])):
            difference[iy,ix] = np.sum(np.square(img2044[y:y+1024,x:x+1024]-img1024))/(1024*1024)
    result = coords[tuple(np.unravel_index(np.argmin(difference),difference.shape))]
    print(f"The minimum found at coord {result}")
    return result

scan((490,530),(490,530))
# The minimum found at coord [510 512]


# %%
# find the translation vector between colors of spectral detectors.


def scan_dual(dual,range_y,range_x):
    """
    dual is a 2-channel image, axes order 'cyx'. This function
    scans over `range_y` for rows and over `range_x` for columns,
    to find the translation vector from 1st channel to 2nd channel.
    """
    size_y,size_x = dual.shape[-2:]
    
    len_y = range_y[1] - range_y[0]
    len_x = range_x[1] - range_x[0]
    
    pad_top    = -min(0,range_y[0])
    pad_bottom = max(0,range_y[1])
    pad_left   = -min(0,range_x[0])
    pad_right  = max(0,range_x[1])
    
    frontframe = dual[0]
    backframe = np.pad(
        dual[1],
        ((pad_top,pad_bottom),(pad_left,pad_right)),
        mode="reflect"
    )
    coords = np.array([[(y,x) for x in range(*range_x)] for y in range(*range_y)])
    difference = np.zeros((len_y,len_x))
    for iy,y in enumerate(range(*range_y)):
        for ix,x in enumerate(range(*range_x)):
            difference[iy,ix] = np.sum(np.square(
                backframe[
                    pad_top+y
                   :pad_top+y+size_y,
                    pad_left+x
                   :pad_left+x+size_x] 
              - frontframe
            ))
    result = coords[tuple(np.unravel_index(np.argmin(difference),difference.shape))]
    print(f"The minimum found at coord {result}")
    return result
# %%
img_stack = io.imread("intermediate/Stack_FOV-2_PFS_by-color_spectral.tif")
img_stack = np.transpose(img_stack,axes=(2,0,1))

results = np.zeros((4,4,2),dtype=int)
for i in range(4):
    for j in range(4):
        scan_dual(np.stack((img_stack[i],img_stack[j]),axis=0),(-20,20),(-20,20))
# 1st\2nd blue,    green,   yellow,  red
#   blue: [ 0  0], [-3 -3], [-6  3], [ 2 -6]
#  green: [ 3  3], [ 0  0], [-3  5], [ 5 -3]
# yellow: [ 6 -3], [ 3 -5], [ 0  0], [ 8 -9]
#    red: [-2  6], [-5  3], [-8  9], [ 0  0]

# %%
img_stack = io.imread("intermediate/Stack_FOV-2_PFS_by-detector_spectral.tif")
img_stack = np.transpose(img_stack,axes=(2,0,1))

results = np.zeros((4,4,2),dtype=int)
for i in range(4):
    for j in range(4):
        results[i,j] = scan_dual(np.stack((img_stack[i],img_stack[j]),axis=0),(-20,20),(-20,20))
# 1st\2nd blue,    green,   yellow,  red
#   blue: [ 0  0], [-3 -4], [-6  2], [ 1 -5]
#  green: [ 3  4], [ 0  0], [-3  6], [ 4 -1]
# yellow: [ 6 -2], [ 3 -6], [ 0  0], [ 7 -8]
#    red: [-1  5], [-4  1], [-7  8], [ 0  0]

# %%
img_stack = io.imread("intermediate/Stack_FOV-3_PFS_by-detector_spectral.tif")
img_stack = np.transpose(img_stack,axes=(2,0,1))

results = np.zeros((4,4,2),dtype=int)
for i in range(4):
    for j in range(4):
        results[i,j] = scan_dual(np.stack((img_stack[i],img_stack[j]),axis=0),(-20,20),(-20,20))
# 1st\2nd blue,    green,   yellow,  red
#   blue: [ 0  0], [-3 -3], [-6  3], diverged
#  green: [ 3  3], [ 0  0], [-3  6], diverged
# yellow: [ 6 -3], [ 3 -6], [ 0  0], diverged
#    red: [-1  5], [-4  2], [-7  8], [ 0  0]
# %%
