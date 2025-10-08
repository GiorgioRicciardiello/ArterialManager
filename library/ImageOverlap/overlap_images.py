"""
=============================================================
Colocalization of Red and Green Vessel Networks (Batch Script)
=============================================================

This script automates the analysis of vascular colocalization
between two fluorescent channels (green = C=0, red = C=1)
across multiple experimental folders.

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

Inputs
------
- CONFIG["paths"]['data']/imgs : Root folder containing subdirectories
  with image pairs.
- Image format: *.jpg (other formats can be extended).

Outputs
-------
- overlap_images/<folder>/<cell_id>/ : Per-cell results.
- overlap_images/all_metrics.csv : Summary of all metrics.

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
from typing import Dict, Tuple, List, Optional
import pandas as pd
import pickle
import gzip

from pathlib import Path
from joblib import Parallel, delayed
from library.ImageOverlap.wavelet_overlap import coloc_vessels_with_wbns


def find_pairs_in_folder(folder_cell: pathlib.Path, path_out: pathlib.Path) -> Dict[str, Dict[str, pathlib.Path]]:
    """
    Identify valid image pairs (C=0 and C=1) in a folder and log issues as structured CSVs.

    :param folder_cell: Path to the folder containing the cell images (e.g., AD1 - NPBB20)
    :param path_out: Path to the output folder where logs will be saved
    :return: Dictionary of valid image pairs
    """

    def _remove_types(folder_cell: pathlib.Path):
        """
        Rename image files in the folder to fix malformed channel tags like '_ C=' → '_C='.

        :param folder_cell: Path to the folder containing image files
        """
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

    # create the logs output
    path_out_logs = path_out.joinpath('logs_pairs')
    # for each folder cell have a cell folder, we mkdir only if there are logs to create
    path_out_logs = path_out_logs.joinpath(folder_cell.stem)

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

    # Save logs only if there are entries
    if any([renamed_files, error_imgs, incomplete]):
        path_out_logs.mkdir(parents=True, exist_ok=True)

    # Save logs as structured CSVs
    if renamed_files:
        df_renamed = pd.DataFrame(renamed_files)
        df_renamed.to_csv(path_out_logs / "renamed_files.csv", index=False)
        print(f"📝 Logged {len(df_renamed)} renamed files to renamed_files.csv")

    if error_imgs:
        # folder logs for the cell is created, only if there are files to log
        df_errors = pd.DataFrame(error_imgs)
        df_errors.to_csv(path_out_logs / "malformed_filenames.csv", index=False)
        print(f"⚠️ Logged {len(df_errors)} malformed filenames to malformed_filenames.csv")

    if incomplete:
        # folder logs for the cell is created, only if there are files to log
        df_incomplete = pd.DataFrame(incomplete)
        df_incomplete.to_csv(path_out_logs / "missing_pairs.csv", index=False)
        print(f"⚠️ Logged {len(df_incomplete)} incomplete pairs to missing_pairs.csv")

    print(f'✅ Found {len(valid_pairs)} complete pairs in {folder_cell.stem}')
    return valid_pairs


def process_folder(folder_cell:pathlib.Path, path_out:pathlib.Path):
    """Process all pairs inside a folder_cell."""
    pairs = find_pairs_in_folder(folder_cell=folder_cell, path_out=path_out)

    results = []
    error_logs = []
    # tqdm inside each folder (progress per cell)
    for cell_id, channels in tqdm(pairs.items(), desc=f"{folder_cell.stem}", leave=False):
        if "0" in channels and "1" in channels:
            path_green = channels["0"]  # C=0 → green
            path_red = channels["1"]    # C=1 → red

            # create subfolder in output
            out_path_tmp = path_out / folder_cell.stem / cell_id
            out_path_tmp.mkdir(parents=True, exist_ok=True)

            try:
                metrics, components = coloc_vessels_with_wbns(
                    path_green=path_green,
                    path_red=path_red,
                    path_output=out_path_tmp,
                    cell_id=cell_id,
                    resolution_px=4,
                    noise_lvl=1,
                    skeleton_radius_px=3,
                    intensity_overlap="geo_mean",
                    show=False
                )
                # Merge metadata with metrics into one dict
                result_row = {
                    "cell_id": cell_id,
                    "folder": str(folder_cell.stem),
                    "path": str(folder_cell),
                    "path_green": str(path_green),
                    "path_red": str(path_red),
                    **metrics
                }
                results.append(result_row)
            except Exception as e:
                print(f"⚠️ Error in {cell_id}: {e}")
                error_logs.append({
                    "folder": folder_cell.stem,
                    "cell_id": cell_id,
                    "path_green": str(path_green),
                    "path_red": str(path_red),
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })

    # Save errors if any occurred
    if error_logs:
        # create directory
        path_out_logs = path_out / folder_cell.stem / 'logs_img_processing'
        path_out_logs.mkdir(parents=True, exist_ok=True)
        error_log_path = path_out_logs / "processing_errors.csv"
        # log the errors
        df_errors = pd.DataFrame(error_logs)
        # Append or create new CSV
        if error_log_path.exists():
            df_errors.to_csv(error_log_path, mode="a", index=False, header=False)
        else:
            df_errors.to_csv(error_log_path, index=False)

        print(f"📝 Logged {len(df_errors)} processing errors to {error_log_path}")


    return results


def run_batch_overlap(base_path: pathlib.Path, path_out: pathlib.Path, n_jobs: int = 8):
    """
    Run batch colocalization for all folders inside base_path.

    Parameters
    ----------
    base_path : Path
        Root folder containing all subfolders with paired images (C=0, C=1).
    path_out : Path
        Output folder where results will be stored.
    n_jobs : int
        Number of parallel workers (joblib).

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for all processed cells.
    """
    path_out.mkdir(parents=True, exist_ok=True)
    # Get all the folders in the base path
    all_folders = [p for p in base_path.iterdir() if p.is_dir()]
    # Parallel processing of the different images
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_folder)(folder_cell, path_out)
        for folder_cell in tqdm(all_folders, desc="Processing folders")
    )

    flat_results = [row for sublist in all_results for row in sublist]
    df_all = pd.DataFrame(flat_results)

    # Save results
    df_all.to_csv(path_out / "all_metrics_with_paths.csv", index=False)
    print(f"✅ Saved metrics to {path_out / 'all_metrics_with_paths.csv'}")

    return df_all


def run_batch_overlap_skip(base_path: Path, path_out: Path, n_jobs: int = 8):
    """
    Run batch colocalization but skip folders already processed.

    Parameters
    ----------
    base_path : Path
        Root folder containing all subfolders with paired images (C=0, C=1).
    path_out : Path
        Output folder where results will be stored.
    n_jobs : int
        Number of parallel workers (joblib).
    """
    path_out.mkdir(parents=True, exist_ok=True)

    # All candidate folders only top-level folders
    all_folders = [p for p in base_path.glob("*") if p.is_dir()]

    # Skip ones already processed in path_out
    to_process = []
    for folder in all_folders:
        out_folder = path_out / folder.stem
        if out_folder.exists():
            print(f"⏭️ Skipping {folder.stem} (already processed).")
        else:
            to_process.append(folder)

    print(f"📂 Found {len(all_folders)} folders, processing {len(to_process)} new ones.")

    # Parallel processing
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_folder)(folder_cell, path_out)
        for folder_cell in tqdm(to_process, desc="Processing new folders")
    )

    flat_results = [row for sublist in all_results for row in sublist]
    if flat_results:
        df_all = pd.DataFrame(flat_results)
        # Append or save results
        csv_path = path_out / "all_metrics_with_paths.csv"
        if csv_path.exists():
            df_all.to_csv(csv_path, mode="a", index=False, header=False)  # append
        else:
            df_all.to_csv(csv_path, index=False)
        print(f"✅ Updated metrics at {csv_path}")
        return df_all
    else:
        print("⚠️ No new folders processed.")
        return pd.DataFrame()


def run_batch_overlap_skip_no_parallel(base_path: Path, path_out: Path):
    """
    Sequential version of run_batch_overlap_skip — skips already processed folders.
    """
    path_out.mkdir(parents=True, exist_ok=True)

    all_folders = [p for p in base_path.glob("*") if p.is_dir()]

    to_process = []
    for folder in all_folders:
        out_folder = path_out / folder.stem
        if out_folder.exists():
            print(f"⏭️ Skipping {folder.stem} (already processed).")
        else:
            to_process.append(folder)

    print(f"📂 Found {len(all_folders)} folders, processing {len(to_process)} new ones (sequential).")

    all_results = []
    for folder_cell in tqdm(to_process, desc="Processing new folders sequentially"):
        results = process_folder(folder_cell, path_out)
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

if __name__ == "__main__":
    # %% Define input and output path
    base_path = CONFIG["paths"]['data'].joinpath('imgs')
    path_out = CONFIG['paths']['outputs_imgs'].joinpath('overlap_images')
    # if parallel processing:
    # run_batch_overlap(base_path, path_out, n_jobs=8)
    # if sequential processing:
    run_batch_overlap_skip_no_parallel(base_path, path_out)


