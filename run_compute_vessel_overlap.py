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
import socket
from config.config import CONFIG
import sys
import os

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



def is_minerva() -> bool:
    """Detect if running inside the Minerva HPC environment."""
    hostname = socket.gethostname()
    cwd = Path.cwd()

    # Match typical Minerva path structure and hostname
    if "arion" in str(cwd) or str(cwd).startswith("/sc/arion"):
        return True
    if "minerva" in hostname.lower() or "arion" in hostname.lower() or hostname.startswith("li") or hostname.startswith("ln"):
        return True
    if os.environ.get("LSB_JOBID") is not None:  # LSF scheduler variable always present
        return True
    return False


def main(n_jobs: int | None = None):
    """Run vascular colocalization batch analysis on local system."""
    path_input = CONFIG.get("paths")["local_images"]
    path_out = CONFIG.get("paths")["local_images_output"].joinpath("processed_overlap")

    if n_jobs is None:
        n_jobs = max(os.cpu_count() - 5, 1)  # reserve 5 cores

    print(f"[LOCAL] Using {n_jobs} CPUs")
    print(f"[INPUT]  {path_input}")
    print(f"[OUTPUT] {path_out}")

    path_out.mkdir(parents=True, exist_ok=True)

    run_batch_overlap_skip(base_path=path_input, path_out=path_out, n_jobs=n_jobs)


if __name__ == "__main__":
    # Optional CLI argument for n_jobs
    # user_n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n_jobs=None)
