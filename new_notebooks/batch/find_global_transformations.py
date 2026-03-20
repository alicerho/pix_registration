import numpy as np
from pathlib import Path

# good FOVs you identified
GOOD_FOVS = [1,2]

mats = []

for fov in GOOD_FOVS:
    
    path = Path(f"outputs_2d/FOV-{fov}/yellow_to_YFP/mask_affine_mutualNN_ransac/affine_matrix_3x3.txt")
    
    A = np.loadtxt(path)
    mats.append(A)

mats = np.stack(mats)

# compute average transform
avg_affine = np.mean(mats, axis=0)

# enforce correct affine format
avg_affine[2] = [0,0,1]

print(avg_affine)

np.savetxt("yellow_to_YFP_global_affine.txt", avg_affine)