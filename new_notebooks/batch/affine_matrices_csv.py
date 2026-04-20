#!/usr/bin/env python3

from pathlib import Path
import csv
import re

# ------------------------
# CONFIG
# ------------------------

ROOT = Path("batch_affine_results")
OUT_CSV = ROOT / "affine_parameters_summary.csv"

# ------------------------
# HELPERS
# ------------------------

def parse_best_affine(txt_path: Path):
    """
    Parse one best_affine.txt file of the form:

    camera:   ...
    spectral: ...

    dx:       ...
    dy:       ...
    rotation: ... deg
    scale_x:  ...
    scale_y:  ...
    shear:    ... deg
    score:    ...

    Affine matrix:
    ...
    ...
    ...
    """

    mode = txt_path.parent.parent.name  # batch_affine_results / MODE / slide-field / best_affine.txt

    text = txt_path.read_text()

    # slide/field from camera path or spectral path
    m = re.search(r"slide-(\d+)_field-(\d+)", text)
    if not m:
        raise ValueError(f"Could not find slide/field in {txt_path}")
    slide = int(m.group(1))
    field = int(m.group(2))

    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"Could not match pattern {pattern} in {txt_path}")
        return cast(m.group(1))

    camera = grab(r"camera:\s+(.+)", str)
    spectral = grab(r"spectral:\s+(.+)", str)

    dx = grab(r"dx:\s+([-\d.eE+]+)")
    dy = grab(r"dy:\s+([-\d.eE+]+)")
    rotation_deg = grab(r"rotation:\s+([-\d.eE+]+)\s+deg")
    scale_x = grab(r"scale_x:\s+([-\d.eE+]+)")
    scale_y = grab(r"scale_y:\s+([-\d.eE+]+)")
    shear_deg = grab(r"shear:\s+([-\d.eE+]+)\s+deg")
    score = grab(r"score:\s+([-\d.eE+]+)")

    # matrix rows
    mat_match = re.search(
        r"Affine matrix:\s*\n"
        r"([^\n]+)\n"
        r"([^\n]+)\n"
        r"([^\n]+)",
        text
    )
    if not mat_match:
        raise ValueError(f"Could not find affine matrix in {txt_path}")

    row1 = [float(x) for x in mat_match.group(1).split()]
    row2 = [float(x) for x in mat_match.group(2).split()]
    row3 = [float(x) for x in mat_match.group(3).split()]

    if not (len(row1) == len(row2) == len(row3) == 3):
        raise ValueError(f"Matrix in {txt_path} is not 3x3")

    return {
        "file": str(txt_path),
        "mode": mode,
        "slide": slide,
        "field": field,
        "camera": camera,
        "spectral": spectral,
        "dx": dx,
        "dy": dy,
        "rotation_deg": rotation_deg,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "shear_deg": shear_deg,
        "score": score,
        "m00": row1[0], "m01": row1[1], "m02": row1[2],
        "m10": row2[0], "m11": row2[1], "m12": row2[2],
        "m20": row3[0], "m21": row3[1], "m22": row3[2],
    }

# ------------------------
# MAIN
# ------------------------

def main():
    files = sorted(ROOT.rglob("best_affine.txt"))
    if not files:
        print(f"No best_affine.txt files found under {ROOT}")
        return

    rows = []
    for txt_path in files:
        try:
            row = parse_best_affine(txt_path)
            rows.append(row)
        except Exception as e:
            print(f"Skipping {txt_path}: {e}")

    if not rows:
        print("No valid files parsed.")
        return

    fieldnames = [
        "file",
        "mode",
        "slide",
        "field",
        "camera",
        "spectral",
        "dx",
        "dy",
        "rotation_deg",
        "scale_x",
        "scale_y",
        "shear_deg",
        "score",
        "m00", "m01", "m02",
        "m10", "m11", "m12",
        "m20", "m21", "m22",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Parsed {len(rows)} best_affine.txt files.")

if __name__ == "__main__":
    main()