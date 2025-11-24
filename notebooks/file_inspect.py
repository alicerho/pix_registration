from nd2reader import ND2Reader

with ND2Reader("/Users/shinyeongrho/Library/CloudStorage/Box-Box/2025-10-02_Registration3D/fov-23_camera.nd2") as f:
    print("Sizes:", f.sizes)
    print("Axes:", f._axes)