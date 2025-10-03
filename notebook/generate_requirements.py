import subprocess
import pkg_resources
from pathlib import Path

# List of packages you know are used in the repo
USED_PACKAGES = [
    "pandas",
    "numpy",
    "openpyxl",
    "xlsxwriter",
    "regex",
    "streamlit",
    "tk",
    "ipykernel",
    "jupyter",
    "pycwa",
    "seaborn",
    "matplotlib",
    "scikit-image",
    "scipy",
    "joblib",
    "opencv-python"
]

def generate_requirements(output_file="requirements.txt"):
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    lines = []
    for pkg in USED_PACKAGES:
        key = pkg.lower()
        if key in installed:
            lines.append(f"{pkg}=={installed[key]}")
        else:
            print(f"⚠️  {pkg} not found in current environment, skipping.")

    Path(output_file).write_text("\n".join(lines))
    print(f"✅ Requirements written to {output_file}")

if __name__ == "__main__":
    generate_requirements()
