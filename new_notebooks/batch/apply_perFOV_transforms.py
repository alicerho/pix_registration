#!/usr/bin/env python3

"""
Per-FOV mask-based registration.
For each FOV, uses the pre-computed mask-based transform to register spectral to camera.
"""

from pathlib import Path
import numpy as np
import nd2
from skimage.transform import AffineTransform, warp
from skimage.io import imsave
import matplotlib.pyplot as plt

# ------------------------
# CONFIG
# ------------------------

DATASET = Path("../../data/Dataset_300Fovs")
UNMIXED = DATASET / "unmixed"
RAW = DATASET / "RAW"
REGISTERED = DATASET / "registered_perFOV"

# Where the per-FOV transforms from mask registration are stored
MASK_REGISTRATION_DIR = Path("outputs_2d")

# Output naming based on spectral channel
OUTPUT_NAMES = {
    "blue": ["px"],           # blue: 1st channel only
    "green": ["er"],          # green from RAW
    "yellow": ["go"],         # yellow from RAW  
    "red": ["mt", "ld"]       # red: both channels
}

# Channel mappings
CHANNEL_PAIRS = {
    "blue": "DAPI",
    "green": "FITC", 
    "yellow": "YFP",
    "red": "TRITC"
}

CROP_SIZE = 512

# ------------------------
# HELPERS
# ------------------------

def load_nd2(path):
    with nd2.ND2File(str(path)) as f:
        arr = f.asarray()
    return np.squeeze(arr).astype(np.float32)


def center_crop(img, crop_size):
    h, w = img.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img[y0:y0+crop_size, x0:x0+crop_size]


def get_spectral_path(slide, field, channel):
    """Get the correct path for spectral file based on channel."""
    if channel == "blue":
        return UNMIXED / f"unmixed_EYrainbow_slide-{slide}_field-{field}_spectral-blue.nd2"
    elif channel == "red":
        return UNMIXED / f"unmixed_EYrainbow_slide-{slide}_field-{field}_spectral-red.nd2"
    elif channel == "yellow":
        return RAW / f"EYrainbow_slide-{slide}_field-{field}_spectral-yellow.nd2"
    elif channel == "green":
        return RAW / f"EYrainbow_slide-{slide}_field-{field}_spectral-green.nd2"
    else:
        raise ValueError(f"Unknown channel: {channel}")


def get_transform_for_fov(fov, spectral_ch, camera_ch):
    """
    Load the per-FOV transform from mask registration results.
    Returns 3x3 affine matrix or None if not found.
    """
    # Path to the mask registration outputs
    pair_dir = MASK_REGISTRATION_DIR / f"FOV-{fov}" / f"{spectral_ch}_to_{camera_ch}/mask_affine_mutualNN_ransac"
    transform_path = pair_dir / "affine_matrix_3x3.txt"
    
    if not transform_path.exists():
        return None
    
    try:
        A = np.loadtxt(transform_path)
        return A
    except Exception as e:
        print(f"    Error loading transform: {e}")
        return None


def norm(img):
    img = img.astype(float)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img


def overlay_rgb(a, b):
    a = norm(a)
    b = norm(b)
    rgb = np.zeros((*a.shape, 3))
    rgb[..., 0] = a  # red
    rgb[..., 1] = b  # green
    return rgb


def make_overlay_visualization(cam, spec, spec_warped, out_path, title):
    """Create before/after overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # BEFORE
    axes[0].imshow(overlay_rgb(cam, spec))
    axes[0].set_title('BEFORE: Red=Camera, Green=Spectral')
    axes[0].axis('off')
    
    # AFTER
    axes[1].imshow(overlay_rgb(cam, spec_warped))
    axes[1].set_title('AFTER: Red=Camera, Green=Registered')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ------------------------
# MAIN REGISTRATION
# ------------------------

def register_one_fov(slide, field, spectral_ch):
    """
    Register one FOV using its pre-computed per-FOV transform.
    """
    camera_ch = CHANNEL_PAIRS[spectral_ch]
    
    print(f"\nSlide {slide}, Field {field}: {spectral_ch}→{camera_ch}")
    
    # Field number IS the FOV number
    fov = int(field)
    
    # Load transform
    A = get_transform_for_fov(fov, spectral_ch, camera_ch)
    
    if A is None:
        print(f"  ⚠ No transform found for FOV {fov}, trying global average...")
        # Fallback to global average if available
        global_transform_path = MASK_REGISTRATION_DIR / f"{spectral_ch}_to_{camera_ch}_global_affine.txt"
        if global_transform_path.exists():
            A = np.loadtxt(global_transform_path)
            print(f"  Using global average transform")
        else:
            print(f"  ✗ No transform available, skipping")
            return False
    else:
        print(f"  ✓ Using per-FOV transform from FOV-{fov}")
    
    # Load spectral
    spec_path = get_spectral_path(slide, field, spectral_ch)
    if not spec_path.exists():
        print(f"  ✗ Spectral file not found: {spec_path.name}")
        return False
    
    arr = load_nd2(spec_path)
    
    # Handle different array shapes
    if arr.ndim == 2:
        channels = [arr]
    elif arr.shape[0] == 2:
        channels = [arr[0], arr[1]]
    elif arr.ndim == 3 and arr.shape[2] == 2:
        channels = [arr[..., 0], arr[..., 1]]
    else:
        print(f"  ✗ Unexpected array shape: {arr.shape}")
        return False
    
    # Load camera
    cam_path = RAW / f"EYrainbow_slide-{slide}_field-{field}_camera-{camera_ch}.nd2"
    if not cam_path.exists():
        print(f"  ✗ Camera file not found")
        return False
    
    cam = load_nd2(cam_path)
    if cam.ndim == 3:
        cam = cam.max(axis=0)
    
    # Crop camera to 512×512
    cam_cropped = center_crop(cam, CROP_SIZE)
    
    # Create transform
    tform = AffineTransform(matrix=A)
    
    # Warp all channels
    warped_channels = []
    for ch in channels:
        ch_warped = warp(
            ch,
            inverse_map=tform.inverse,
            output_shape=cam_cropped.shape,
            preserve_range=True
        )
        warped_channels.append(ch_warped)
    
    # Save registered images
    REGISTERED.mkdir(parents=True, exist_ok=True)
    
    output_names = OUTPUT_NAMES[spectral_ch]
    
    for i, name in enumerate(output_names):
        if i < len(warped_channels):
            output_path = REGISTERED / f"registered_EYrainbow_slide-{slide}_field-{field}_spectral-{name}.tif"
            
            # Save as 16-bit TIFF
            img = norm(warped_channels[i])
            imsave(str(output_path), (img * 65535).astype(np.uint16))
            print(f"  Saved: {output_path.name}")
    
    # Save overlay visualization
    viz_dir = REGISTERED / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    overlay_path = viz_dir / f"overlay_slide-{slide}_field-{field}_{spectral_ch}.png"
    make_overlay_visualization(
        cam_cropped, 
        channels[0], 
        warped_channels[0],
        overlay_path,
        f"Slide {slide} Field {field}: {spectral_ch}→{camera_ch}"
    )
    print(f"  Saved overlay: {overlay_path.name}")
    
    return True


def main():
    """
    Process all spectral channels for all available FOVs.
    """
    print("="*80)
    print("PER-FOV MASK-BASED REGISTRATION")
    print("="*80)
    print(f"Transform directory: {MASK_REGISTRATION_DIR}")
    print(f"Output directory: {REGISTERED}")
    
    # Process each spectral channel
    for spectral_ch in ["blue", "green", "yellow", "red"]:
        camera_ch = CHANNEL_PAIRS[spectral_ch]
        
        print(f"\n{'='*80}")
        print(f"Processing {spectral_ch} → {camera_ch}")
        print(f"Output channels: {OUTPUT_NAMES[spectral_ch]}")
        print('='*80)
        
        # Find all files for this channel
        if spectral_ch == "blue":
            pattern = f"unmixed_EYrainbow_slide-*_field-*_spectral-blue.nd2"
            search_dir = UNMIXED
        elif spectral_ch == "red":
            pattern = f"unmixed_EYrainbow_slide-*_field-*_spectral-red.nd2"
            search_dir = UNMIXED
        elif spectral_ch in ["yellow", "green"]:
            pattern = f"EYrainbow_slide-*_field-*_spectral-{spectral_ch}.nd2"
            search_dir = RAW
        else:
            continue
        
        files = sorted(search_dir.glob(pattern))
        print(f"Found {len(files)} files")
        
        success_count = 0
        for f in files:
            import re
            m = re.search(r"slide-(\d+)_field-(\d+)", f.name)
            if m:
                slide, field = m.group(1), m.group(2)
                if register_one_fov(slide, field, spectral_ch):
                    success_count += 1
        
        print(f"\n✓ {spectral_ch}: {success_count}/{len(files)} FOVs registered")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Registered TIFFs: {REGISTERED}/")
    print(f"Overlay visualizations: {REGISTERED}/visualizations/")


if __name__ == "__main__":
    main()