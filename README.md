# Pixel Registration Between Spectral Detector and Camera

## Active RF Pipeline

The active Random Forest/modeling pipeline uses the 300-FOV yeast rainbow dataset,
not the older bead datasets described below. The bead data and 2022 registration
notes are legacy/earlier registration work.

The active goal is to predict six registered spectral/organelle channels from four
widefield camera channels:

- Inputs: `DAPI`, `FITC`, `TRITC`, `YFP`
- Outputs: `ER`, `GO`, `PX`, `VO`, `MT`, `LD`

### 1. Raw Camera Inputs

Camera images are used both for registration and as model inputs:

```text
data/Dataset_300Fovs/RAW/EYrainbow_slide-<slide>_field-<field>_camera-DAPI.nd2
data/Dataset_300Fovs/RAW/EYrainbow_slide-<slide>_field-<field>_camera-FITC.nd2
data/Dataset_300Fovs/RAW/EYrainbow_slide-<slide>_field-<field>_camera-TRITC.nd2
data/Dataset_300Fovs/RAW/EYrainbow_slide-<slide>_field-<field>_camera-YFP.nd2
```

### 2. Raw/Unmixed Spectral Targets

Spectral images are registered to their matching camera channels and used as model
targets:

```text
data/Dataset_300Fovs/RAW/EYrainbow_slide-<slide>_field-<field>_spectral-green.nd2
  -> ER, registered to FITC

data/Dataset_300Fovs/RAW/EYrainbow_slide-<slide>_field-<field>_spectral-yellow.nd2
  -> GO, registered to YFP

data/Dataset_300Fovs/unmixed/unmixed_EYrainbow_slide-<slide>_field-<field>_spectral-blue.nd2
  -> PX/VO, registered to DAPI

data/Dataset_300Fovs/unmixed/unmixed_EYrainbow_slide-<slide>_field-<field>_spectral-red.nd2
  -> MT/LD, registered to TRITC
```

### 3. Affine Matrix Generation

Run from `new_notebooks/batch`:

```bash
cd new_notebooks/batch
```

Estimate per-FOV affine transforms between each spectral color and its matching
camera channel:

```bash
python batch_manual_guess_affine.py
```

Extract the final 3x3 affine matrices:

```bash
python get_all_transforms.py
```

Matrix output:

```text
new_notebooks/batch/batch_affine_results/all_300FOV_affine_matrices/
```

### 4. Dataset Generation

Run from `new_notebooks/batch`:

```bash
cd new_notebooks/batch
```

Check which FOVs have all required camera and spectral files:

```bash
python file_check_table.py
```

Output:

```text
new_notebooks/batch/fov_file_presence_table.csv
```

Build the correlated-pixel dataset. This script loads camera images, loads spectral
images, applies the corresponding affine matrices, and writes the selected pixel
table:

```bash
python build_correlated_pixel_database.py
```

Output:

```text
new_notebooks/batch/correlated_pixel_database/top_correlated_pixels.csv
```

### 5. Training And Evaluation

Run from `new_notebooks/batch`:

```bash
cd new_notebooks/batch
```

Train the baseline all-camera Random Forest model:

```bash
python Train/train_baseline_model.py
```

Train the single-camera baseline models:

```bash
python Train/train_single_camera_baselines.py
```

Generate full-FOV prediction figures:

```bash
python Train/predict_full_fov_with_baseline.py
```

Generate held-out pixel prediction QC figures:

```bash
python Train/visualize_baseline_predictions.py
```

Check full-FOV prediction correlations:

```bash
python Train/check_prediction_correlations.py
```

Check actual registered spectral-vs-camera correlations:

```bash
python Train/actual_spectral_vs_camera_correlations.py
```

Primary outputs:

```text
new_notebooks/batch/Train/models/
new_notebooks/batch/Train/results/
new_notebooks/batch/Train/figures/
```

## Raw Data 

- `2022-11-08_beads_equal_pixel_size`
    - Only 1 field of view.
    - `field-0_focus_TRITC.nd2` is a single z-slice
    - Camera images are `71*2048*2044` z-stack. Beads did not show up in DAPI and CFP. 
    - Spectral images are `71*512*512` z-stack. 
    - The change of z center had not been noticed. But z stack should handle it.
    - **not good enough**, see `notebooks/comfirm_rotation.py`
- `2022-11-10_beadsEqualPixel`
    - Acquisition **not** very successful, the beads moved.
    - Camera images are `71*2048*2044`.
    - In field-0, the red spectra image with equal pixel size. Well taken. But there is only one red spectral image that has the same pixel size. 
    - In field-1, I only took red channel spectra and camera, with single z. In the zoomed spectra image the beads are too few and close to each other. **Not usable**.
    - In field-2, all image are single-sliced. 
- `2022-11-14_forAnnualReport`
    - Images of cells rather than beads. 
    - "longInterval" in file names meant the camera images were taken after all spectral images were taken.
    - Camera images were blury in FITC, YFP and TRITC channels. 
    - **Not usable** for training or registration.

## Pipeline (reviewed Jan 24, 2024)

### Working one: New Beads

Scripts with `newbead` prefixes.

1. (Done elsewhere) Taking images of the same beads with different optical systems.
2. `./ilastik/*.ilp`: Segment the objects (cells, beads) with `ilastik`
3. `notebooks/newbeads_label2coords.py` -> `./intermediate/coordination_new_beads.csv`: the coordinates of the centroids of the "new beads". `by` column is how the imaging was grouped by.
4. `notebooks/newbeads_map.py` -> `./intermediate/mapping_new_beads.csv`: the correspondance between the centroids of beads in images of different detectors.
5. `notebooks/newbeads_map.py` -> `./intermediate/transforms/*.txt`: find the transforms, (until Jan 23, 2024) only from 

### History

Scripts without `newbead` prefixes, **deprecated**.

- `notebooks/comfirm_rotation.py`: 
    - previously the `main.py`, comfirmed the necessity of this project.
    - found the quality of `./data/2022-11-08_beads_equal_pixel_size/*.nd2` is not good enough. 
- `notebooks/map_centroids.py`:
    - original mapping between the points, 3 ways of determining the centroid: simple mean, weighted mean, maximum (within segmented masks)
- `notebooks/check_results.py`: stack the images of the unchanged camera images and the transformed spectral images, to see the results of the `spectral-512_camera-1024_{color}_weighted_Affine` transforms. *Seems **outdated** and not used by "new beads"*.
- `notebooks/apply_transf_blue.py`: After everything is done, apply the transforms obtained from this project.
