import nd2
import numpy as np

cam_path = "../../data/2025-11-05_Registration2D/FOV-1_camera-DAPI.nd2"
spec_path = "../../data/2025-11-05_Registration2D/FOV-1_spectral-blue.nd2"

with nd2.ND2File(cam_path) as f:
    cam = np.squeeze(f.asarray())

with nd2.ND2File(spec_path) as f:
    spec = np.squeeze(f.asarray())

print("Camera shape:", cam.shape)
print("Spectral shape:", spec.shape)

print("Camera pixels:", cam.size)
print("Spectral pixels:", spec.size)