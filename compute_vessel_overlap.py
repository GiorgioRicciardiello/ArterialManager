"""
=============================================================
🔬 Vascular Colocalization Batch Runner (External Drive Version)
=============================================================

This script automates the batch analysis of vascular colocalization
between paired fluorescent images (green = C=0, red = C=1) stored
on an external hard drive.

It is designed for collaborators who have the dataset saved on
different drives (e.g., D:, E:, F:). The script automatically
searches for the correct path across all available drives.

Workflow
--------
1. Define the relative path of your dataset:
       "Vascular images/FINAL/Colored/Same contrast"
2. The script will:
   - Search across all drives (A–Z) for that path.
   - Process only top-level folders (ignores nested ones).
   - Skip any folders already processed (output exists).
3. Results are saved to a sibling folder named:
       "<drive>:/Vascular images/FINAL/Colored/processed_overlap"

Outputs
-------
- Per-cell results inside:
      processed_overlap/<cell_folder>/
- Aggregated CSV of all results:
      processed_overlap/all_metrics_with_paths.csv

Usage
-----
1. Place this script inside your project folder.
2. Run from the command line (Windows PowerShell or CMD):

       python run_overlap_drive.py

3. Make sure your external HDD/SSD is plugged in and accessible.

Notes
-----
- Assumes paired .jpg images exist in each folder:
    * C=0 → green channel
    * C=1 → red channel
- Skips MERGE and scaled artifacts (jpgscale).
- Skips already processed folders to save time.
- Requires conda/pip environment with dependencies installed.

=============================================================
"""

from library.ImageOverlap.overlap_images import run_batch_overlap_skip
from pathlib import Path
import string

def find_base_folder(relative_path:Path) -> Path:
    """
    Search for the relative_path across all drives (Windows).
    Example: 'Vascular images/FINAL/Colored/Same contrast'
    """
    for drive in string.ascii_uppercase:  # A:, B:, C: ... Z:
        drive_path = Path(f"{drive}:/")
        candidate = drive_path / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {relative_path} on any drive.")


if __name__ == "__main__":
    # 🔍 Dataset location (relative path inside the drive)
    relative_path = Path("Vascular images/FINAL/Colored/Same contrast")

    # Locate the dataset folder automatically
    path_input = find_base_folder(relative_path)
    assert path_input.exists(), f"Path not found: {path_input}"

    # Output goes into the same drive, under "processed_overlap"
    path_out = path_input.parent.joinpath("processed_overlap")
    path_out.mkdir(parents=True, exist_ok=True)

    # 🚀 Run colocalization in parallel
    run_batch_overlap_skip(base_path=path_input,
                           path_out=path_out,
                           n_jobs=8)

    # %% Running on Minerva
    path_input = Path(r'/sc/arion/projects/vascbrain/giocrm/VascularImages/SameContrast')
    path_out = path_input.parent.joinpath("processed_overlap")
    path_out.mkdir(parents=True, exist_ok=True)

    run_batch_overlap_skip(base_path=path_input,
                           path_out=path_out,
                           n_jobs=2)  # for minerva, 2 folders are processed at the same time within the .sh job

