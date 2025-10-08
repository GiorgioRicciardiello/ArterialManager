#!/bin/bash
#BSUB -J overlap_batch[1-43]         # Job array: one job per folder
#BSUB -P acc_vascbrain
#BSUB -q general                     # general is safer than gpu unless you actually use CUDA
#BSUB -n 4                           # 4 cores per job
#BSUB -R "rusage[mem=16000]"         # 8 GB RAM per job
#BSUB -R "span[hosts=1]"             # keep cores on same host
#BSUB -W 8:00                        # walltime (adjust as needed)
#BSUB -oo /sc/arion/projects/vascbrain/giocrm/VascularImages/ArterialManager/minerva/output/overlap_batch_%I.out
#BSUB -eo /sc/arion/projects/vascbrain/giocrm/VascularImages/ArterialManager/minerva/output/overlap_batch_%I.err
#BSUB -L /bin/bash

# -------------------------------
# Setup
# -------------------------------
module purge
module load anaconda3/2024.06

source activate imgai

echo "Running on $(hostname)"
echo "Job index: $LSB_JOBINDEX"
echo "CPUs: $LSB_DJOB_NUMPROC"

# -------------------------------
# Paths
# -------------------------------
BASE_PATH=/sc/arion/projects/vascbrain/giocrm/VascularImages/SameContrast
OUT_PATH=/sc/arion/projects/vascbrain/giocrm/VascularImages/processed_overlap

# Make sure output dir exists
mkdir -p $OUT_PATH

# -------------------------------
# Select the folder for this job
# -------------------------------
# Get list of all folders under BASE_PATH (only top-level)
FOLDER=$(ls -d $BASE_PATH/*/ | sed -n ${LSB_JOBINDEX}p)

echo "Processing folder: $FOLDER"

# -------------------------------
# Run Python for this folder
# -------------------------------
python -u - <<EOF
from pathlib import Path
from library.ImageOverlap.overlap_images import process_folder

folder = Path("$FOLDER".strip())
out_path = Path("$OUT_PATH") / folder.stem
out_path.mkdir(parents=True, exist_ok=True)

results = process_folder(folder, out_path)
print("✅ Done with", folder, "→ results:", len(results))
EOF
