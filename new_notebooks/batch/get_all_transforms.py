#!/usr/bin/env python3

from pathlib import Path
import re

# ------------------------
# CONFIG
# ------------------------

SOURCE_ROOT = Path("batch_affine_results")
DEST_DIR = SOURCE_ROOT / "all_300FOV_affine_matrices"

DEST_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# FIND + EXTRACT MATRIX
# ------------------------

files = SOURCE_ROOT.glob("*/*/best_affine.txt")

for src in files:

    mode = src.parent.parent.name
    slide_field = src.parent.name

    m = re.match(r"slide-(\d+)_field-(\d+)", slide_field)

    if m is None:
        print(f"Skipping malformed folder: {slide_field}")
        continue

    slide = m.group(1)
    field = m.group(2)

    out_name = f"EYrainbow_slide-{slide}_field-{field}_{mode}.txt"
    dst = DEST_DIR / out_name

    # read original file
    text = src.read_text()

    # extract only the affine matrix block
    matrix_match = re.search(
        r"Affine matrix:\n(.*?\n.*?\n.*?)$",
        text,
        re.DOTALL
    )

    if matrix_match is None:
        print(f"Could not find matrix in {src}")
        continue

    matrix_text = matrix_match.group(1).strip()

    # save ONLY matrix
    dst.write_text(matrix_text + "\n")

    print(f"Saved matrix -> {dst}")

print("\nDone.")