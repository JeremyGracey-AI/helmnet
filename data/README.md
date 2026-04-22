# Dataset

The HelmNet dataset is **not committed to this repository**. It must be obtained separately and placed in this directory before running the notebook.

## Expected layout

```
data/
├── images.npy     # NumPy array of image tensors
└── labels.csv     # Class labels aligned to images.npy
```

## Dataset characteristics

- **Total images:** 4,125
- **Classes:** `With Helmet` (3,161) · `Without Helmet` (964)
- **Class imbalance:** 3.3 : 1 (addressed at training time via class weighting and augmentation)
- **Domain:** Construction sites, factories, and industrial settings with variation in lighting, camera angle, and worker posture

## Obtaining the data

This dataset was provided as part of coursework and is not redistributed here. If you are replicating this work with your own dataset, ensure it follows the same structure:

- `images.npy` — a `float32` array of shape `(N, H, W, 3)` or `(N, H, W)` with values in `[0, 255]` or `[0, 1]`
- `labels.csv` — a single-column CSV where `0` = *Without Helmet* and `1` = *With Helmet*, with row order matching `images.npy`

If your raw data is a directory of image files instead of a pre-packed NumPy array, the preprocessing cells in the notebook show how to build the `.npy` / `.csv` pair from a folder tree.
