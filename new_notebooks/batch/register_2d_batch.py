import os
import numpy as np
import ants
import nd2
from scipy.stats import pearsonr
from skimage.feature import peak_local_max, blob_log
from skimage.transform import estimate_transform, warp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import csv

# --------------------
# CONFIG
# --------------------
DATA_DIR = "../../data/2025-11-05_Registration2D"
ROOT_OUT = "outputs_2d"

FOV_LIST = [1, 2, 3, 4, 5, 6]

PAIRS = [
    ("blue",   "DAPI"),
    ("green",  "FITC"),
    ("yellow", "YFP"),
    ("red",    "TRITC"),
]

CROP_SIZE = 512


# --------------------
# HELPERS
# --------------------
def load_camera_image(fov, camera_name):
    path = os.path.join(DATA_DIR, f"FOV-{fov}_camera-{camera_name}.nd2")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with nd2.ND2File(path) as f:
        arr = f.asarray()
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = np.max(arr, axis=0)
    return arr.astype("float32")


def load_spectral_image(fov, spectral_name):
    path = os.path.join(DATA_DIR, f"FOV-{fov}_spectral-{spectral_name}.nd2")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with nd2.ND2File(path) as f:
        arr = f.asarray()
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = np.sum(arr, axis=0)
    return arr.astype("float32")


def center_crop(img, crop_size):
    h, w = img.shape
    y0 = h // 2 - crop_size // 2
    x0 = w // 2 - crop_size // 2
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def corr(a, b):
    return pearsonr(a.ravel(), b.ravel())[0]


def norm01(img, p_lo=1, p_hi=99):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    out = (img - lo) / (hi - lo + 1e-8)
    return np.clip(out, 0, 1)


# --------------------
# BEAD DETECTION
# --------------------
def detect_beads_robust(img, min_sigma=2, max_sigma=8, num_sigma=5, 
                        threshold_percentile=99.0, max_beads=200):
    """
    Detect beads as blob-like structures (extended bright regions).
    Uses Laplacian of Gaussian blob detection which finds circular features.
    """
    img = np.squeeze(img)
    if img.max() == img.min():
        return np.zeros((0, 2), dtype=float)
    
    # Normalize to [0, 1]
    img_norm = (img - img.min()) / (img.max() - img.min())
    
    # Calculate adaptive threshold
    threshold_value = np.percentile(img_norm, threshold_percentile)
    
    print(f"    Image range: [{img.min():.1f}, {img.max():.1f}]")
    print(f"    Normalized {threshold_percentile}th percentile: {threshold_value:.3f}")
    
    # Detect blobs (beads are circular, ~5-10 pixels diameter)
    # min_sigma=2 → detects ~4 pixel diameter blobs
    # max_sigma=8 → detects ~16 pixel diameter blobs
    blobs = blob_log(
        img_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold_value,
        exclude_border=15,
        overlap=0.5
    )
    
    if len(blobs) == 0:
        print(f"    Detected 0 beads (threshold may be too high)")
        return np.zeros((0, 2), dtype=float)
    
    # Sort by intensity (brightest first) and limit
    coords = blobs[:, :2]  # (y, x) positions
    radii = blobs[:, 2]
    
    # Get intensity at each blob center
    intensities = []
    for y, x in coords:
        y_int, x_int = int(round(y)), int(round(x))
        if 0 <= y_int < img.shape[0] and 0 <= x_int < img.shape[1]:
            intensities.append(img[y_int, x_int])
        else:
            intensities.append(0)
    
    intensities = np.array(intensities)
    sorted_indices = np.argsort(intensities)[::-1]  # brightest first
    
    # Keep only the brightest beads
    num_keep = min(len(coords), max_beads)
    coords = coords[sorted_indices[:num_keep]]
    radii = radii[sorted_indices[:num_keep]]
    
    print(f"    Detected {len(coords)} beads (radii: {radii.mean():.1f}±{radii.std():.1f} px)")
    
    # Return as (x, y) for consistency
    return np.column_stack([coords[:, 1], coords[:, 0]])


def match_beads_ransac(coords_fixed, coords_moving, max_distance=30, inlier_threshold=5.0):
    """
    Match beads between images using nearest neighbor + RANSAC.
    Returns matched pairs and the affine transformation.
    """
    if len(coords_fixed) < 3 or len(coords_moving) < 3:
        print(f"  Not enough beads: fixed={len(coords_fixed)}, moving={len(coords_moving)}")
        return None, None
    
    # Strategy: Start with relaxed matching to get more candidates for RANSAC
    # RANSAC will filter out the bad matches
    strategies = [
        ("moderate", 40, 5.0),   # Start with moderate distance
        ("relaxed", 60, 6.0),    # Try relaxed if moderate fails
        ("strict", 20, 3.0),     # Fall back to strict only if others fail
    ]
    
    for strategy_name, distance_thresh, inlier_thresh in strategies:
        print(f"  Trying {strategy_name} matching (distance < {distance_thresh} px, inlier < {inlier_thresh} px)...")
        
        # Nearest-neighbor matching
        tree = cKDTree(coords_fixed)
        distances, indices = tree.query(coords_moving, k=1)
        
        # Filter by distance threshold
        valid_mask = distances < distance_thresh
        num_valid = np.sum(valid_mask)
        
        if num_valid < 4:  # Need at least 4 for robust affine estimation
            print(f"    Not enough close matches: {num_valid} < 4")
            continue
        
        src_pts = coords_moving[valid_mask]
        dst_pts = coords_fixed[indices[valid_mask]]
        
        print(f"    Initial matches: {len(src_pts)}")
        
        # Try to estimate affine transform
        try:
            model = estimate_transform('affine', src_pts, dst_pts)
            
            # Apply transform to ALL source points (not just initial matches)
            src_transformed = model(src_pts)
            
            # Find inliers based on residual distance
            residuals = np.sqrt(np.sum((src_transformed - dst_pts)**2, axis=1))
            inliers = residuals < inlier_thresh
            
            num_inliers = np.sum(inliers)
            inlier_ratio = num_inliers / len(src_pts)
            
            if num_inliers > 0:
                mean_residual = np.mean(residuals[inliers])
                max_residual = np.max(residuals[inliers])
                print(f"    Inliers: {num_inliers}/{len(src_pts)} ({100*inlier_ratio:.1f}%)")
                print(f"    Residuals - mean: {mean_residual:.2f}, max: {max_residual:.2f}")
            else:
                print(f"    Inliers: 0/{len(src_pts)} (0.0%)")
            
            # Accept if we have enough inliers
            # Lower threshold: accept 20% inlier ratio OR at least 6 inliers
            min_inliers = 4
            min_ratio = 0.20
            
            if num_inliers >= min_inliers and (inlier_ratio >= min_ratio or num_inliers >= 6):
                src_inliers = src_pts[inliers]
                dst_inliers = dst_pts[inliers]
                
                # Re-estimate transform using only inliers
                model_refined = estimate_transform('affine', src_inliers, dst_inliers)
                
                # Calculate final residuals with refined model
                final_transformed = model_refined(src_inliers)
                final_residuals = np.sqrt(np.sum((final_transformed - dst_inliers)**2, axis=1))
                
                print(f"  ✓ Success with {strategy_name} strategy!")
                print(f"    Refined residuals: mean={np.mean(final_residuals):.2f}, max={np.max(final_residuals):.2f}")
                
                return (src_inliers, dst_inliers), model_refined
            elif num_inliers == 0:
                print(f"    Rejected: no inliers found")
            else:
                print(f"    Rejected: {num_inliers} inliers < {min_inliers} required (ratio {100*inlier_ratio:.1f}%)")
                
        except Exception as e:
            print(f"    Failed: {e}")
            continue
    
    # All strategies failed
    print("  ✗ All matching strategies failed")
    return None, None


# --------------------
# VISUALIZATION
# --------------------
def make_visualizations(cam_crop, spec_crop, warped_affine, warped_syn,
                       beads_fixed, beads_moving, beads_matched,
                       out_dir, spec_ch, cam_ch):
    
    fixed_n = norm01(cam_crop)
    mov_n = norm01(spec_crop)
    aff_n = norm01(warped_affine)
    syn_n = norm01(warped_syn)
    
    # Side by side images
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(fixed_n, cmap="gray")
    axes[0].set_title(f"Fixed: camera-{cam_ch}")
    if beads_fixed is not None and len(beads_fixed) > 0:
        axes[0].scatter(beads_fixed[:, 0], beads_fixed[:, 1], 
                       c='red', s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    axes[1].imshow(mov_n, cmap="gray")
    axes[1].set_title(f"Moving: spectral-{spec_ch}")
    if beads_moving is not None and len(beads_moving) > 0:
        axes[1].scatter(beads_moving[:, 0], beads_moving[:, 1], 
                       c='lime', s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    axes[2].imshow(aff_n, cmap="gray")
    axes[2].set_title("After Affine (bead-based)")
    
    axes[3].imshow(syn_n, cmap="gray")
    axes[3].set_title("After Affine + SyN")
    
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "images_side_by_side.png"), dpi=200)
    plt.close(fig)
    
    # Overlays with bead correspondences
    def rgb(fixed, other):
        return np.dstack([fixed, other, np.zeros_like(fixed)])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Before
    axes[0].imshow(rgb(fixed_n, mov_n))
    axes[0].set_title("BEFORE (red=fixed, green=moving)")
    if beads_fixed is not None and len(beads_fixed) > 0:
        axes[0].scatter(beads_fixed[:, 0], beads_fixed[:, 1], 
                       c='red', s=40, marker='x', linewidths=2)
    if beads_moving is not None and len(beads_moving) > 0:
        axes[0].scatter(beads_moving[:, 0], beads_moving[:, 1], 
                       c='lime', s=40, marker='+', linewidths=2)
    
    # After affine
    axes[1].imshow(rgb(fixed_n, aff_n))
    axes[1].set_title("AFTER Affine")
    if beads_matched is not None:
        src_pts, dst_pts = beads_matched
        for i in range(len(src_pts)):
            axes[1].plot([src_pts[i, 0], dst_pts[i, 0]], 
                        [src_pts[i, 1], dst_pts[i, 1]], 
                        'c-', linewidth=1, alpha=0.5)
        axes[1].scatter(dst_pts[:, 0], dst_pts[:, 1], 
                       c='yellow', s=50, marker='o', edgecolors='white', linewidth=1)
    
    # After SyN
    axes[2].imshow(rgb(fixed_n, syn_n))
    axes[2].set_title("AFTER Affine+SyN")
    
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "overlays.png"), dpi=200)
    plt.close(fig)


# --------------------
# CORE REGISTRATION
# --------------------
def register_one(fov, spec_ch, cam_ch):
    print(f"\n{'='*60}")
    print(f"FOV-{fov}: spectral-{spec_ch} -> camera-{cam_ch}")
    print('='*60)

    spec_path = os.path.join(DATA_DIR, f"FOV-{fov}_spectral-{spec_ch}.nd2")
    cam_path = os.path.join(DATA_DIR, f"FOV-{fov}_camera-{cam_ch}.nd2")
    if not (os.path.exists(spec_path) and os.path.exists(cam_path)):
        print("  -> Files not found, skipping")
        return None

    out_dir = os.path.join(ROOT_OUT, f"FOV-{fov}", f"{spec_ch}_to_{cam_ch}")
    os.makedirs(out_dir, exist_ok=True)

    # Load images
    cam = load_camera_image(fov, cam_ch)
    spec = load_spectral_image(fov, spec_ch)
    cam_crop = center_crop(cam, CROP_SIZE)
    
    print(f"Camera shape: {cam_crop.shape}, range: [{cam_crop.min():.1f}, {cam_crop.max():.1f}]")
    print(f"Spectral shape: {spec.shape}, range: [{spec.min():.1f}, {spec.max():.1f}]")

    # STEP 1: Detect beads in both images
    print("\n--- Detecting beads ---")
    print("  Fixed (camera):")
    beads_fixed = detect_beads_robust(cam_crop, threshold_percentile=96.0, max_beads=200)
    print("  Moving (spectral):")
    beads_moving = detect_beads_robust(spec, threshold_percentile=96.0, max_beads=200)
    
    print(f"\n  Final counts - Fixed: {len(beads_fixed)}, Moving: {len(beads_moving)}")
    
    if len(beads_fixed) < 3 or len(beads_moving) < 3:
        print("ERROR: Not enough beads detected!")
        return None

    # STEP 2: Match beads and find affine transform
    print("\n--- Matching beads with RANSAC ---")
    matched_pairs, affine_model = match_beads_ransac(beads_fixed, beads_moving)
    
    if matched_pairs is None:
        print("ERROR: Bead matching failed!")
        return None
    
    src_pts, dst_pts = matched_pairs
    
    # Extract affine matrix from scikit-image model
    # Model transforms (x,y) -> (x',y')
    A = affine_model.params[:2, :2]
    t = affine_model.params[:2, 2]
    
    print(f"Affine matrix:\n{A}")
    print(f"Translation: {t}")
    
    # Save affine parameters
    np.savetxt(os.path.join(out_dir, "affine_matrix_2x2.txt"), A)
    np.savetxt(os.path.join(out_dir, "affine_translation.txt"), t)
    
    A_h = np.eye(3)
    A_h[:2, :2] = A
    A_h[:2, 2] = t
    np.savetxt(os.path.join(out_dir, "affine_matrix_3x3.txt"), A_h)

    # STEP 3: Apply affine transform to spectral image
    print("\n--- Applying affine transform ---")
    
    # Extract affine matrix from scikit-image model
    A = affine_model.params[:2, :2]
    t = affine_model.params[:2, 2]
    
    print(f"  Affine matrix:\n{A}")
    print(f"  Translation: {t}")
    
    # Test: transform a few matched bead positions to verify
    if len(src_pts) > 0:
        test_pt = src_pts[0:1]
        transformed_pt = affine_model(test_pt)[0]
        expected_pt = dst_pts[0]
        error = np.sqrt(np.sum((transformed_pt - expected_pt)**2))
        print(f"  Verification: bead at {test_pt[0]} -> {transformed_pt} (expected {expected_pt}, error={error:.2f}px)")
    
    # CRITICAL FIX: Use skimage.transform.warp instead of scipy
    # scikit-image warp uses forward mapping correctly
    from skimage.transform import warp
    
    # Invert the transform for backward mapping (warp pulls pixels)
    inverse_model = affine_model.inverse
    
    # Apply transformation
    warped_affine_np = warp(
        spec.astype('float32'),
        inverse_model,
        output_shape=cam_crop.shape,
        order=1,  # bilinear
        mode='constant',
        cval=0,
        preserve_range=True
    ).astype('float32')
    
    # Metrics after affine
    c_before = corr(cam_crop, spec)
    mae_before = float(np.mean(np.abs(cam_crop - spec)))
    c_aff = corr(cam_crop, warped_affine_np)
    mae_aff = float(np.mean(np.abs(cam_crop - warped_affine_np)))
    
    print(f"Before: corr={c_before:.4f}, MAE={mae_before:.1f}")
    print(f"After Affine: corr={c_aff:.4f}, MAE={mae_aff:.1f}")
    print(f"  Improvement: Δcorr={c_aff - c_before:+.4f}, ΔMAE={mae_before - mae_aff:+.1f}")

    # STEP 4: SyN refinement (intensity-based on affine-corrected images)
    print("\n--- SyN refinement ---")
    
    # Create ANTs images for SyN
    fixed_ants = ants.from_numpy(cam_crop.astype('float32'))
    moving_ants = ants.from_numpy(warped_affine_np.astype('float32'))
    
    # Normalize for SyN
    fixed_norm = (cam_crop - cam_crop.min()) / (cam_crop.max() - cam_crop.min() + 1e-8)
    warped_norm = (warped_affine_np - warped_affine_np.min()) / (warped_affine_np.max() - warped_affine_np.min() + 1e-8)
    
    fixed_syn = ants.from_numpy(fixed_norm.astype('float32'))
    moving_syn = ants.from_numpy(warped_norm.astype('float32'))
    
    reg_syn = ants.registration(
        fixed=fixed_syn,
        moving=moving_syn,
        type_of_transform="SyN",
        syn_metric="meansquares",  # Try MSE for normalized images
        syn_sampling=32,
        reg_iterations=(100, 50, 25),
        flow_sigma=3.0,
        total_sigma=0.5,
        verbose=False
    )
    
    warped_syn_np = reg_syn["warpedmovout"].numpy()
    
    c_syn = corr(cam_crop, warped_syn_np)
    mae_syn = float(np.mean(np.abs(cam_crop - warped_syn_np)))
    
    print(f"After SyN: corr={c_syn:.4f}, MAE={mae_syn:.1f}")
    print(f"  SyN improvement: Δcorr={c_syn - c_aff:+.4f}, ΔMAE={mae_aff - mae_syn:+.1f}")

    # Save images
    ants.image_write(fixed_ants, os.path.join(out_dir, "fixed_camera.nii.gz"))
    ants.image_write(moving_ants, os.path.join(out_dir, "moving_spectral_affine_transformed.nii.gz"))
    ants.image_write(reg_syn["warpedmovout"], os.path.join(out_dir, "warped_affine_syn.nii.gz"))

    # Save transform files
    warp_path = reg_syn["fwdtransforms"][0]
    warp_dest = os.path.join(out_dir, "syn_warp.nii.gz")
    with open(warp_path, "rb") as src, open(warp_dest, "wb") as dst:
        dst.write(src.read())

    # Visualizations
    make_visualizations(cam_crop, spec, warped_affine_np, warped_syn_np,
                       beads_fixed, beads_moving, matched_pairs,
                       out_dir, spec_ch, cam_ch)

    # Save metrics
    result = {
        "FOV": fov,
        "spectral": spec_ch,
        "camera": cam_ch,
        "num_beads_fixed": len(beads_fixed),
        "num_beads_moving": len(beads_moving),
        "num_beads_matched": len(src_pts),
        "corr_before": c_before,
        "mae_before": mae_before,
        "corr_affine": c_aff,
        "mae_affine": mae_aff,
        "corr_improvement_affine": c_aff - c_before,
        "corr_syn": c_syn,
        "mae_syn": mae_syn,
        "corr_improvement_syn": c_syn - c_aff,
    }
    
    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(result.keys()))
        w.writerow(list(result.values()))
    
    return result


def main():
    os.makedirs(ROOT_OUT, exist_ok=True)
    rows = []

    for fov in FOV_LIST:
        for spec, cam in PAIRS:
            try:
                r = register_one(fov, spec, cam)
                if r:
                    rows.append(r)
            except Exception as e:
                print(f"\n✗ ERROR in FOV-{fov}, {spec}->{cam}: {e}")
                import traceback
                traceback.print_exc()

    if rows:
        summary_path = os.path.join(ROOT_OUT, "register_2d_batch_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Wrote summary to {summary_path}")
        print(f"✓ Completed {len(rows)} registrations")
    else:
        print("\n✗ No registrations completed successfully")


if __name__ == "__main__":
    main()