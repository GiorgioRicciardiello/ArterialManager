"""
=============================================================
Colocalization of Red and Green Vessel Networks (Batch Script)
=============================================================

This script automates the analysis of vascular colocalization
between two fluorescent channels (green = C=0, red = C=1)
across multiple experimental folders.

Parallelization now occurs per valid image pair (not per folder).

Workflow
--------
1. Input discovery:
   - Recursively scans the base image folder for subdirectories.
   - Each subdirectory is expected to contain paired images:
       * Green channel (C=0)
       * Red channel (C=1)
   - MERGE and scale artifacts (e.g., "MERGE", "jpgscale") are excluded.

2. Preprocessing (per channel):
   - Wavelet-based background and noise subtraction (WBNS).
   - Vessel enhancement with Frangi vesselness.
   - Binarization and skeletonization of vessel structures.

3. Overlap detection:
   - Structural overlap: skeleton dilation and intersection.
   - Intensity-based overlap: product, minimum, or geometric mean of
     vessel intensities, optionally weighted by Pearson correlation.

4. Metrics:
   - Manders coefficients (fractional colocalization).
   - Pearson correlation within overlap regions.
   - Dice and Jaccard indices on vessel skeletons.

5. Outputs:
   - For each cell:
       * Visualization panels (raw channels, overlays, heatmaps).
       * Metrics tables (.csv, per cell and global summary).
       * Pickled components for reproducibility.
   - Aggregated metrics across all cells saved to:
       overlap_images/all_metrics.csv

Usage
-----
Run as a standalone script:

    python colocalization_batch.py

Configuration:
- Adjust base path and output path in `CONFIG["paths"]`.
- Parallelization: number of workers set in joblib Parallel(n_jobs=...).

Notes
-----
- Assumes every valid cell has exactly two channels (C=0, C=1).
- Assertions ensure no cell is missing a channel.
- Designed for batch processing of fluorescence microscopy images
  (e.g., vessel colocalization experiments).
"""

from config.config import CONFIG
import pathlib
from tqdm import tqdm
from typing import Dict
import pandas as pd
import os
from pathlib import Path
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from library.ImageOverlap.wavelet_overlap import coloc_vessels_with_wbns


# -------------------------------------------------------------------------
# Utility: find valid pairs
# -------------------------------------------------------------------------
def find_pairs_in_folder(folder_cell: pathlib.Path, path_out: pathlib.Path) -> Dict[str, Dict[str, pathlib.Path]]:
    """Identify valid image pairs (C=0 and C=1) in a folder and log issues."""

    def _remove_types(folder_cell: pathlib.Path):
        renamed_files = []
        for img in folder_cell.glob("*.jpg"):
            if "_ C=" in img.name:
                corrected_name = img.name.replace("_ C=", "_C=")
                new_path = img.with_name(corrected_name)
                try:
                    img.rename(new_path)
                    renamed_files.append({
                        "folder": folder_cell.stem,
                        "original_name": img.name,
                        "new_name": corrected_name
                    })
                    print(f"✅ Renamed: {img.name} → {corrected_name}")
                except Exception as e:
                    print(f"⚠️ Failed to rename {img.name}: {e}")
        return renamed_files

    path_out_logs = path_out.joinpath('logs_pairs', folder_cell.stem)
    renamed_files = _remove_types(folder_cell)

    all_imgs = [
        p for p in folder_cell.glob("*.jpg")
        if "MERGE" not in p.name and "jpgscale" not in p.name
    ]
    print(f'🔍 Found {len(all_imgs)} images in {folder_cell.stem}')

    pairs = {}
    error_imgs = []
    incomplete = []

    for img in all_imgs:
        try:
            base_name = img.stem.split("_C=")[0]
            channel = img.stem.split("_C=")[1]
            pairs.setdefault(base_name, {})[channel] = img
        except Exception as e:
            error_imgs.append({
                "folder": folder_cell.stem,
                "filename": img.name,
                "issue": "malformed filename",
                "error_message": str(e)
            })
            print(f"⚠️ Skipping malformed filename: {img.name} → {e}")

    valid_pairs = {}
    for name, chans in pairs.items():
        if set(chans.keys()) == {"0", "1"}:
            valid_pairs[name] = chans
        else:
            incomplete.append({
                "folder": folder_cell.stem,
                "cell_id": name,
                "found_channels": list(chans.keys())
            })

    # Save logs
    if any([renamed_files, error_imgs, incomplete]):
        path_out_logs.mkdir(parents=True, exist_ok=True)
    if renamed_files:
        pd.DataFrame(renamed_files).to_csv(path_out_logs / "renamed_files.csv", index=False)
    if error_imgs:
        pd.DataFrame(error_imgs).to_csv(path_out_logs / "malformed_filenames.csv", index=False)
    if incomplete:
        pd.DataFrame(incomplete).to_csv(path_out_logs / "missing_pairs.csv", index=False)

    print(f'✅ Found {len(valid_pairs)} complete pairs in {folder_cell.stem}')
    return valid_pairs


# -------------------------------------------------------------------------
# Parallelized per-pair processing
# -------------------------------------------------------------------------
def process_folder(folder_cell: pathlib.Path, path_out: pathlib.Path, n_jobs: int = 8):
    """Process all valid pairs in a folder using parallel processing per pair."""
    print(f'process_folder: {folder_cell.stem}')
    pairs = find_pairs_in_folder(folder_cell=folder_cell, path_out=path_out)

    results = []
    error_logs = []

    def _process_single_pair(cell_id: str, channels: Dict[str, pathlib.Path]):
        out_path_tmp = path_out / folder_cell.stem / cell_id
        out_path_tmp.mkdir(parents=True, exist_ok=True)

        try:
            metrics, components = coloc_vessels_with_wbns(
                path_green=channels["0"],
                path_red=channels["1"],
                path_output=out_path_tmp,
                cell_id=cell_id,
                resolution_px=4,
                noise_lvl=1,
                skeleton_radius_px=3,
                intensity_overlap="geo_mean",
                show=False
            )
            return {
                "cell_id": cell_id,
                "folder": str(folder_cell.stem),
                "path": str(folder_cell),
                "path_green": str(channels["0"]),
                "path_red": str(channels["1"]),
                **metrics
            }, None
        except Exception as e:
            return None, {
                "folder": folder_cell.stem,
                "cell_id": cell_id,
                "path_green": str(channels.get("0")),
                "path_red": str(channels.get("1")),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }

    # --- Run in parallel across valid pairs ---
    parallel_outputs = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_pair)(cell_id, chans)
        for cell_id, chans in tqdm(pairs.items(), desc=f"{folder_cell.stem}", leave=False)
        if "0" in chans and "1" in chans
    )

    for res, err in parallel_outputs:
        if res:
            results.append(res)
        if err:
            error_logs.append(err)

    # Save errors
    if error_logs:
        path_out_logs = path_out / folder_cell.stem / 'logs_img_processing'
        path_out_logs.mkdir(parents=True, exist_ok=True)
        error_log_path = path_out_logs / "processing_errors.csv"
        df_errors = pd.DataFrame(error_logs)
        if error_log_path.exists():
            df_errors.to_csv(error_log_path, mode="a", index=False, header=False)
        else:
            df_errors.to_csv(error_log_path, index=False)
        print(f"📝 Logged {len(df_errors)} processing errors to {error_log_path}")

    return results


# -------------------------------------------------------------------------
# Batch runners
# -------------------------------------------------------------------------
def run_batch_overlap(base_path: Path, path_out: Path, n_jobs: int = 8):
    """Process all folders, parallelizing per image pair inside each folder."""
    path_out.mkdir(parents=True, exist_ok=True)
    all_folders = [p for p in base_path.iterdir() if p.is_dir()]

    all_results = []
    for folder_cell in tqdm(all_folders, desc="Processing folders sequentially"):
        results = process_folder(folder_cell, path_out, n_jobs=n_jobs)
        all_results.extend(results)

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(path_out / "all_metrics_with_paths.csv", index=False)
    print(f"✅ Saved metrics to {path_out / 'all_metrics_with_paths.csv'}")
    return df_all


def run_batch_overlap_skip(base_path: Path, path_out: Path, n_jobs: int = 8):
    """Skip already processed folders."""
    path_out.mkdir(parents=True, exist_ok=True)
    all_folders = [p for p in base_path.glob("*") if p.is_dir()]

    to_process = []
    for folder in all_folders:
        if (path_out / folder.stem).exists():
            print(f"⏭️ Skipping {folder.stem} (already processed).")
        else:
            to_process.append(folder)

    print(f"📂 Found {len(all_folders)} folders, processing {len(to_process)} new ones.")

    all_results = []
    for folder_cell in tqdm(to_process, desc="Processing new folders"):
        results = process_folder(folder_cell, path_out, n_jobs=n_jobs)
        all_results.extend(results)

    if all_results:
        df_all = pd.DataFrame(all_results)
        csv_path = path_out / "all_metrics_with_paths.csv"
        if csv_path.exists():
            df_all.to_csv(csv_path, mode="a", index=False, header=False)
        else:
            df_all.to_csv(csv_path, index=False)
        print(f"✅ Updated metrics at {csv_path}")
        return df_all
    else:
        print("⚠️ No new folders processed.")
        return pd.DataFrame()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
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
