"""
===============================================================================
ArterialManager GUI Launcher
===============================================================================
Author: Giorgio Ricciardiello Mejia
Email:  giorgio.ricciardiellomejia@mountsinai.edu
Date:   2025-10-29
Project: ArterialManager – Angiogenesis Research Data Toolkit
-------------------------------------------------------------------------------

Description:
------------
Graphical interface to launch the main ArterialManager modules without using
the command line. The GUI runs each pipeline inside the pre-existing
Conda environment 'imgai' through an Anaconda Prompt window.

Modules:
---------
1️⃣ Table Manager           → Streamlit app for Excel table processing
2️⃣ Table Visualizer        → Streamlit app for data visualization
3️⃣ Vessel Overlap Analysis → Python script for vessel colocalization
4️⃣ Overlay Viewer (HTML)   → Opens static HTML viewer in browser

Behavior:
----------
• Opens a new terminal window (Conda shell) for each process.
• Automatically activates the 'imgai' environment.
• Changes working directory to the project root (from CONFIG).
• Keeps the terminal open for log inspection.

Dependencies:
-------------
- tkinter (built-in)
- subprocess, os, webbrowser (built-in)
- Streamlit (installed in 'imgai' env)
- config.config (for ROOT_PATH detection)

Usage:
------
Run the launcher from any terminal or by double-clicking:

    python src/app_launcher.py

Press a button to execute the desired module. No terminal input required.

===============================================================================
"""


import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import webbrowser
# -------------------------------------------------------------------
# FIX IMPORT PATHS AUTOMATICALLY (robust launcher)
# -------------------------------------------------------------------
import sys
from pathlib import Path

# Get absolute path of this file: .../ArterialManager/src/app_launcher.py
_THIS_FILE = Path(__file__).resolve()

# Project root = parent of src/
PROJECT_ROOT = _THIS_FILE.parent.parent

# Add root to sys.path if not already included
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now imports ALWAYS work:
from config.config import CONFIG
# -------------------------------------------------------------------

ROOT_PATH = CONFIG["paths"]["root"]

# Absolute paths to scripts
PATHS = {
    "Table Manager": ROOT_PATH / "src" / "app_streamlit.py",
    "Table Visualizer": ROOT_PATH / "src" / "app_plot.py",
    "Vessel Overlap Analysis": ROOT_PATH / "src" / "run_overlap.py",
    "Overlay Viewer (HTML)": ROOT_PATH / "src" / "app_overlay_compare.html",
}

# Adjust this path if Anaconda is installed elsewhere
# CONDA_PROMPT = r"C:\ProgramData\anaconda3\Scripts\activate.bat"
CONDA_PROMPT = r"C:\Users\riccig01\anaconda3\Scripts\activate.bat"
ENV_NAME = "imgai_env"
# python -m streamlit run src/app_launcher.py

# def run_conda_in_terminal(app_name):
#     """Launch inside an Anaconda Prompt with conda activate imgai."""
#     script_path = PATHS[app_name]
#     if not script_path.exists():
#         messagebox.showerror("Error", f"{app_name} not found:\n{script_path}")
#         return
#
#     os.chdir(ROOT_PATH)
#
#     # Build the conda command string
#     if "streamlit" in str(script_path):
#         run_cmd = f"python -m streamlit run \"{script_path}\""
#     else:
#         run_cmd = f"python \"{script_path}\""
#
#     # Complete command: open new terminal, activate, run, keep open
#     cmd = f'start "" cmd /K "{CONDA_PROMPT} {ENV_NAME} && cd /d {ROOT_PATH} && {run_cmd}"'
#
#     try:
#         subprocess.Popen(cmd, shell=True)
#         messagebox.showinfo("Launching", f"{app_name} started in Conda env '{ENV_NAME}'.")
#     except Exception as e:
#         messagebox.showerror("Launch Error", str(e))

def run_conda_in_terminal(app_name):
    script_path = PATHS[app_name]
    if not script_path.exists():
        messagebox.showerror("Error", f"{app_name} not found:\n{script_path}")
        return

    os.chdir(ROOT_PATH)

    # Streamlit apps must be launched via python -m streamlit
    streamlit_apps = {"app_streamlit", "app_plot"}

    if script_path.stem in streamlit_apps:
        run_cmd = f"python -m streamlit run \"{script_path}\""
    else:
        run_cmd = f"python \"{script_path}\""

    cmd = f'start "" cmd /K "{CONDA_PROMPT} {ENV_NAME} && cd /d {ROOT_PATH} && {run_cmd}"'

    try:
        subprocess.Popen(cmd, shell=True)
        messagebox.showinfo("Launching", f"{app_name} started in Conda env '{ENV_NAME}'.")
    except Exception as e:
        messagebox.showerror("Launch Error", str(e))



def open_html_viewer():
    """Open HTML viewer directly."""
    html_path = PATHS["Overlay Viewer (HTML)"]
    if not html_path.exists():
        messagebox.showerror("Error", f"HTML file not found:\n{html_path}")
        return
    webbrowser.open_new_tab(f"file://{html_path}")
    messagebox.showinfo("Opened", "Overlay Viewer launched in your browser.")


# --- GUI ---
root = tk.Tk()
root.title("ArterialManager Launcher")
root.geometry("460x400")
root.configure(bg="#f9f9f9")

tk.Label(root, text="🧠 ArterialManager Launcher", font=("Segoe UI", 16, "bold"), bg="#f9f9f9").pack(pady=20)
tk.Label(root, text=f"Environment: {ENV_NAME}", bg="#f9f9f9", fg="#444").pack(pady=5)
tk.Label(root, text=f"Project path:\n{ROOT_PATH}", bg="#f9f9f9", fg="#666", font=("Segoe UI", 9)).pack(pady=5)

for app in list(PATHS.keys())[:3]:
    tk.Button(
        root,
        text=f"Run {app}",
        width=38,
        height=2,
        bg="#0052cc",
        fg="white",
        command=lambda a=app: run_conda_in_terminal(a),
    ).pack(pady=8)

tk.Button(
    root,
    text="Open Overlay Viewer (HTML)",
    width=38,
    height=2,
    bg="#009688",
    fg="white",
    command=open_html_viewer,
).pack(pady=8)

tk.Label(
    root,
    text="Developed by Giorgio Ricciardiello Mejia",
    font=("Segoe UI", 9),
    bg="#f9f9f9",
    fg="#777",
).pack(side="bottom", pady=10)

root.mainloop()
