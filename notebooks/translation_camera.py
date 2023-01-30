import numpy as np
from scipy import optimize
from nd2reader import ND2Reader

with ND2Reader("data/FOV-2_PFS_by-detector_camera-TRITC_px-2044.nd2") as file2044:
    file2044.iter_axes = 't'
    file2044.bundle_axes = 'yx'
    img2044 = file2044[0].astype(int)

with ND2Reader("data/FOV-2_PFS_by-detector_camera-TRITC_px-1024.nd2") as file1024:
    file1024.iter_axes = 't'
    file1024.bundle_axes = 'yx'
    img1024 = file1024[0].astype(int)

# def translate(coord):
#     coord = coord.astype(int)
#     cropped = img2044[coord[0]:coord[0]+1024,coord[1]:coord[1]+1024]
#     return np.sum(np.square(cropped-img1024))/(1024*1024)


# difference = np.zeros((1020,1024))
# for y in range(1020):
#     for x in range(1024):
#         difference[y,x] = np.sum(np.square(img2044[y:y+1024,x:x+1024]-img1024))/(1024*1024)

# print(np.argmin(difference))
# 

def scan(range_y,range_x):
    coords = np.array([[(y,x) for x in range(range_x[0],range_x[1])] for y in range(range_y[0],range_y[1])])
    difference = np.zeros((range_y[1]-range_y[0],range_x[1]-range_x[0]))
    for iy,y in enumerate(range(range_y[0],range_y[1])):
        for ix, x in enumerate(range(range_x[0],range_x[1])):
            difference[iy,ix] = np.sum(np.square(img2044[y:y+1024,x:x+1024]-img1024))/(1024*1024)
    result = coords[tuple(np.unravel_index(np.argmin(difference),difference.shape))]
    print(f"The minimum found at coord {result}")
    return result

scan((490,530),(490,530))
# The minimum found at coord [510 512]
