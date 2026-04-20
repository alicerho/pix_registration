#!/usr/bin/env python3

from pathlib import Path
import re

ROOT = Path("batch_affine_results")   # adjust if needed
OUT_DIR = ROOT / "affine_matrices_only"
OUT_DIR.mkdir(exist_ok=True)


def extract_matrix(lines):
    for i, line in enumerate(lines):
        if "Affine matrix" in line:
            return lines[i+1:i+4]
    return None


def extract_slide_field(text):
    m = re.search(r"slide-(\d+)_field-(\d+)", text)
    if m:
        return m.group(1), m.group(2)
    return None, None


def extract_mode(txt_path: Path):
    # batch_affine_results / MODE / slide-X_field-Y / best_affine.txt
    parts = txt_path.parts
    try:
        idx = parts.index(ROOT.name)
        return parts[idx + 1]
    except Exception:
        # fallback: parent of slide-field folder
        return txt_path.parent.parent.name


for txt_path in ROOT.rglob("best_affine.txt"):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    matrix_lines = extract_matrix(lines)
    if matrix_lines is None:
        print(f"Skipping (no matrix found): {txt_path}")
        continue

    camera_line = next((l for l in lines if l.startswith("camera:")), "")
    slide, field = extract_slide_field(camera_line)

    if slide is None:
        print(f"Skipping (no slide/field found): {txt_path}")
        continue

    mode = extract_mode(txt_path)

    out_name = f"EYrainbow_slide-{slide}_field-{field}_{mode}.txt"
    out_path = OUT_DIR / out_name

    with open(out_path, "w") as f:
        for row in matrix_lines:
            f.write(row.strip() + "\n")

    print(f"Saved: {out_path}")

print("Done.")