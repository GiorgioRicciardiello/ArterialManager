# ImageOverlap Library

This library provides a comprehensive pipeline for the **intensity-weighted colocalization and quantification of vascular networks** from fluorescent microscopy images. It is designed to process paired images (Green/Red channels) to assess vessel structure and overlap.

<img src="/library/ImageOverlap/img.png" alt="isolated" width="200"/>


## 🚀 Key Features

*   **Advanced Preprocessing:** 
    *   **Wavelet-Based Background subtraction (WBNS):** Removes uneven illumination and noise while preserving vessel structures.
    *   **Vessel Enhancement:** Uses Frangi vesselness filtering to highlight tubular structures.
*   **Overlap Quantification:**
    *   **Structural Overlap:** Skeletonization and intersection of vessel networks.
    *   **Intensity-Weighted Overlap:** Supports multiple strategies (product, minimum, geometric mean, Pearson-weighted) to quantify colocalization strength.
*   **Automated Metrics:** Calculates Manders' coefficients, Pearson correlation, Dice/Jaccard indices, and custom intensity-weighted scores.
*   **Batch Processing:** Parallelized execution for high-throughput analysis of image folders.

---

## 📂 File Structure

*   `wavelet_overlap.py`: **Core Method**. Contains the `coloc_vessels_with_wbns` function which runs the full analysis pipeline on a single image pair.
*   `overlap_images_parallel_pair_level.py`: **Entry Point**. Batch processing script that scans directories, identifies image pairs, and runs the analysis in parallel.
*   `metrics.py`: Computes all quantitative metrics.
*   `display.py`: Handles generation of visualization plots and overlays.
*   `utils.py`: Helper functions for normalization, filtering, and skeletonization.
*   `interpretation_metrics.md`: Documentation explaining the meaning of each calculated metric.

---

## 🛠️ Usage

### 1. Batch Processing (Recommended)
Use `overlap_images_parallel_pair_level.py` to process an entire folder of experiments.

**Expected Input Structure:**
A main directory containing subfolders. Each subfolder should contain paired images with the naming convention:
*   `Name_C=0.jpg` (Green Channel)
*   `Name_C=1.jpg` (Red Channel)

**Run:**
```bash
python -m library.ImageOverlap.overlap_images_parallel_pair_level
```
*   *Configuration:* Adjust `base_path` and `path_out` within the `if __name__ == "__main__":` block or via `config/config.py`.
```python
# -------------------------------------------------------------------------
from library.ImageOverlap.wavelet_overlap import coloc_vessels_with_wbns
from config.config import CONFIG
from multiprocessing import cpu_count

if __name__ == "__main__":
    base_path = CONFIG.get("paths")["local_images"]
    path_out = CONFIG.get("paths")["local_images_output"].joinpath("processed_overlap")

    SEQUENTIAL = False  # set True to disable parallelization per pair

    n_jobs = 2
    if n_jobs <= 0:
        n_jobs = max(cpu_count() - 2, 1)

    print(f"\n[INFO] Parallel mode per-pair: {'ON' if not SEQUENTIAL else 'OFF'}")
    print(f"[INFO] Using {n_jobs} workers")

    if SEQUENTIAL:
        run_batch_overlap_skip(base_path, path_out, n_jobs=1)
    else:
        run_batch_overlap_skip(base_path, path_out, n_jobs=n_jobs)
```


### 2. Single Pair Analysis
You can run the analysis on a single pair of images using the core function:

```python
from library.ImageOverlap.wavelet_overlap import coloc_vessels_with_wbns
from pathlib import Path
metrics, components = coloc_vessels_with_wbns(
    path_green="path/to/image_C=0.jpg",
    path_red="path/to/image_C=1.jpg",
    path_output=Path("./results"),
    cell_id="Experiment_1",
    intensity_overlap="geo_mean",
    show=True
)
```

---

## 📊 Outputs

For each processed image pair, the library generates:

1.  **Visualizations (`/subplots`):**
    *   RGB Overlays
    *   Heatmaps of intensity overlap (Gold/Yellow highlights)
    *   Binary masks and skeleton plots
    *   `_grid_overview.png`: A comprehensive summary panel combined in one image.
2.  **Data:**
    *   `*_metrics.xlsx`: Detailed metrics for the specific image pair.
    *   `*_components.pkl.gz`: Compressed pickle file containing raw data, masks, skeletons, and derived maps for further analysis.
3.  **Aggregated Results:**
    *   `all_metrics_with_paths.csv`: A summary table containing metrics for **all** processed images in the batch.

---

## 📖 Metrics Interpretation
For a detailed explanation of the output metrics (e.g., *M1, M2, Pearson, Overlap_Intensity*), please refer to [interpretation_metrics.md](./interpretation_metrics.md).
