"""
Author: Giorgio Ricciardiello
        giorgio.ricciardiellomejia@mountsinai.edu
configurations parameters for the paths
"""
import pathlib
from pathlib import Path

import pathlib
from datetime import datetime

# === Root project path ===
ROOT_PATH = pathlib.Path(__file__).resolve().parents[1]

# === Config dictionary ===
CONFIG = {
    "paths": {
        "root": ROOT_PATH,
        "data": ROOT_PATH / "data",
        "path_tables": ROOT_PATH / "data" / "tables",
        "path_imgs": ROOT_PATH / "data" / "imgs",
        "path_feature_imgs": ROOT_PATH / "data" / "feature_imgs",

        "outputs_imgs": ROOT_PATH / "results" / 'proc_imgs',
        "outputs_tabs": ROOT_PATH / "results" / 'master_table',

        "checkpoints": ROOT_PATH / "outputs" / "checkpoints",
        "predictions": ROOT_PATH / "outputs" / "predictions",
        "visualizations": ROOT_PATH / "outputs" / "visualizations",
        "models": ROOT_PATH / "models",


        'local_images':  Path(r'C:\Users\riccig01\Documents\SameContrast'),
        'local_images_output': Path(r'C:\Users\riccig01\Documents'),

        'results': ROOT_PATH / "results",

    },
    "green_img": {
        "param": {}
    },
    "red_img": {
        "param": {}
    },

    'conda_prompt' : r"C:\Users\riccig01\anaconda3\Scripts\activate.bat",
    'env_name' :"imgai_env",
}


# # Auto-create directories so they always exist
# for path in CONFIG["paths"].values():
#     pathlib.Path(path).mkdir(parents=True, exist_ok=True)


