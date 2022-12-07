# Pixel Registration Between Spectral Detector and Camera

## Data 

- `2022-11-08_beads_equal_pixel_size`
    - Only 1 field of view.
    - `field-0_focus_TRITC.nd2` is a single z-slice
    - Camera images are `71*2048*2044` z-stack. Beads did not show up in DAPI and CFP. 
    - Spectral images are `71*512*512` z-stack. 
    - The change of z center had not been noticed. But z stack should handle it.
- `2022-11-10_beadsEqualPixel`
    - Acquisition not very successful, the beads moved.
    - Camera images are `71*2048*2044`.
    - In field-0, the red spectra image with equal pixel size. Well taken. But there is only one red spectral image that has the same pixel size. 
    - In field-1, I only took red channel spectra and camera, with single z. In the zoomed spectra image the beads are too few and close to each other. **Not usable**.
    - In field-2, all image are single-sliced. 
- `2022-11-14_forAnnualReport`
    - Images of cells rather than beads. 
    - "longInterval" in file names meant the camera images were taken after all spectral images were taken.
    - Camera images were blury in FITC, YFP and TRITC channels. 
    - **Not usable** for training or registration.
