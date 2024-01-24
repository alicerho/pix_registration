# Pixel Registration Between Spectral Detector and Camera

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