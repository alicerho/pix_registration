from pathlib import Path
import nd2

FOLDERS = [
    Path("../../data/Dataset_300Fovs/RAW"),
    Path("../../data/Dataset_300Fovs/unmixed"),
]

for folder in FOLDERS:
    print(f"\n=== Inspecting {folder} ===\n")
    for fpath in sorted(folder.glob("*.nd2")):
        try:
            with nd2.ND2File(str(fpath)) as f:
                arr = f.asarray()
                print(f"{fpath.name}")
                print(f"  sizes: {f.sizes}")
                print(f"  shape: {arr.shape}")
                print()
        except Exception as e:
            print(f"ERROR on {fpath.name}: {e}")