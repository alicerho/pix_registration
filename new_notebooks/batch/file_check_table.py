#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd

# ------------------------
# CONFIG
# ------------------------

DATASET = Path("../../data/Dataset_300Fovs")
RAW = DATASET / "RAW"
UNMIXED = DATASET / "unmixed"

OUT_CSV = Path("fov_file_presence_table.csv")

FILE_GROUPS = {
    # camera files
    "camera_DAPI": {
        "folder": RAW,
        "pattern": "EYrainbow_slide-*_field-*_camera-DAPI.nd2",
    },
    "camera_FITC": {
        "folder": RAW,
        "pattern": "EYrainbow_slide-*_field-*_camera-FITC.nd2",
    },
    "camera_TRITC": {
        "folder": RAW,
        "pattern": "EYrainbow_slide-*_field-*_camera-TRITC.nd2",
    },
    "camera_YFP": {
        "folder": RAW,
        "pattern": "EYrainbow_slide-*_field-*_camera-YFP.nd2",
    },

    # spectral files
    "spectral_green": {
        "folder": RAW,
        "pattern": "EYrainbow_slide-*_field-*_spectral-green.nd2",
    },
    "spectral_yellow": {
        "folder": RAW,
        "pattern": "EYrainbow_slide-*_field-*_spectral-yellow.nd2",
    },
    "spectral_blue_unmixed": {
        "folder": UNMIXED,
        "pattern": "unmixed_EYrainbow_slide-*_field-*_spectral-blue.nd2",
    },
    "spectral_red_unmixed": {
        "folder": UNMIXED,
        "pattern": "unmixed_EYrainbow_slide-*_field-*_spectral-red.nd2",
    },
}


# ------------------------
# HELPERS
# ------------------------

def parse_slide_field(filename):
    m = re.search(r"slide-(\d+)_field-(\d+)", filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# ------------------------
# MAIN
# ------------------------

def main():
    records = {}

    for group_name, info in FILE_GROUPS.items():
        folder = info["folder"]
        pattern = info["pattern"]

        files = sorted(folder.glob(pattern))
        print(f"{group_name}: found {len(files)} files")

        for f in files:
            parsed = parse_slide_field(f.name)
            if parsed is None:
                print(f"Could not parse: {f.name}")
                continue

            slide, field = parsed
            key = (slide, field)

            if key not in records:
                records[key] = {
                    "slide": slide,
                    "field": field,
                }

            records[key][group_name] = True
            records[key][f"{group_name}_path"] = str(f)

    df = pd.DataFrame(records.values())

    # fill missing booleans as False
    for group_name in FILE_GROUPS.keys():
        if group_name not in df.columns:
            df[group_name] = False
        df[group_name] = df[group_name].fillna(False)

    # add completeness columns
    df["has_all_camera"] = (
        df["camera_DAPI"]
        & df["camera_FITC"]
        & df["camera_TRITC"]
        & df["camera_YFP"]
    )

    df["has_all_spectral"] = (
        df["spectral_green"]
        & df["spectral_yellow"]
        & df["spectral_blue_unmixed"]
        & df["spectral_red_unmixed"]
    )

    df["has_all_files"] = df["has_all_camera"] & df["has_all_spectral"]

    df = df.sort_values(["slide", "field"])

    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved table to: {OUT_CSV}")
    print(f"Total unique slide-field combos: {len(df)}")
    print(f"Complete camera FOVs: {df['has_all_camera'].sum()}")
    print(f"Complete spectral FOVs: {df['has_all_spectral'].sum()}")
    print(f"Complete all-file FOVs: {df['has_all_files'].sum()}")


if __name__ == "__main__":
    main()