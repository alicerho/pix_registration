#!/usr/bin/env python3

from pathlib import Path
import re

ROOT = Path("outputs_2d")
OUT_DIR = Path("prev_affine_matrices")
OUT_DIR.mkdir(exist_ok=True)

for path in ROOT.rglob("affine_matrix_3x3.txt"):
    
    # Example path:
    # outputs_2d/FOV-3/green_to_FITC/mask_affine_mutualNN_ransac/affine_matrix_3x3.txt
    
    parts = path.parts
    
    try:
        fov_part = next(p for p in parts if p.startswith("FOV-"))
        pair_part = next(p for p in parts if "_to_" in p)
    except StopIteration:
        print(f"Skipping malformed path: {path}")
        continue

    # Extract FOV
    fov = fov_part.replace("FOV-", "")

    # Extract mode (spectral channel)
    mode = pair_part.split("_to_")[0]

    # Build output filename
    out_name = f"FOV-{fov}_{mode}.txt"
    out_path = OUT_DIR / out_name

    # Copy matrix contents
    with open(path, "r") as f:
        matrix = f.read()

    with open(out_path, "w") as f:
        f.write(matrix)

    print(f"Saved: {out_path}")

print("Done.")