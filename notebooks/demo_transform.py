import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

# Simulate some "bead" coordinates
# Spectral system (512x512)
spectral_coords = np.array([
    [100, 100],
    [400, 100],
    [100, 400],
    [400, 400],
    [250, 250]
])

# Camera system (1024x1024) - should be ~2x larger + rotated + shifted
camera_coords = spectral_coords * 2.0  # Scale by 2
rotation = 4 * np.pi / 180  # 4 degrees
R = np.array([[np.cos(rotation), -np.sin(rotation)],
              [np.sin(rotation),  np.cos(rotation)]])
camera_coords = camera_coords @ R.T  # Rotate
camera_coords += [512, 510]  # Translate

# Fit affine transform (x, y ordering!)
coords_spectral_xy = spectral_coords[:, [1, 0]]  # Swap to (x,y)
coords_camera_xy = camera_coords[:, [1, 0]]

transf = transform.AffineTransform()
success = transf.estimate(coords_spectral_xy, coords_camera_xy)

print(f"Transform successful: {success}")
print(f"Transform matrix:\n{transf.params}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(coords_spectral_xy[:, 0], coords_spectral_xy[:, 1], 
           c='blue', s=100, label='Spectral')
ax.scatter(coords_camera_xy[:, 0], coords_camera_xy[:, 1], 
           c='red', s=100, label='Camera')

# Show predicted positions
predicted = transf(coords_spectral_xy)
ax.scatter(predicted[:, 0], predicted[:, 1], 
           c='green', s=50, marker='x', label='Predicted')

ax.legend()
ax.set_aspect('equal')
ax.set_title('Demo: Affine Transform Registration')
plt.show()