# ArterialManager

📊 **ArterialManager** is a Python-based toolkit for **angiogenesis research data processing**.
It includes two complementary pipelines:

1. **Table Manager** → merges, normalizes, and formats **Angiotool**, **Cell Count**, and **Sample** Excel files into a clean master dataset.
2. **Streamlit Visualizer** → Uses the results from 'Table Manager' and generates a plot for visualization 
3. **Vessel Overlap Analysis** → performs **intensity-weighted and binary colocalization analysis** of red and green vessel networks from microscopy images.

Both pipelines can be run via scripts, Streamlit apps, or Colab notebooks.

---

## 🚀 Features

### 📊 Table Manager

* Automatically detects and loads input Excel tables:

  * **Angiotool** output
  * **Cell count** table
  * **Sample metadata**
* Cleans and reformats headers
* Normalizes vessel metrics by cell count
* Produces a styled **Excel master table** with frozen headers


### 📈 Table Visualizer
* Generate plots with the use of the 

### 🧬 Vessel Overlap Analysis

* Preprocesses microscopy images:

  * **Wavelet-based background & noise subtraction (WBNS)**
  * **Frangi vesselness enhancement**
  * **Skeletonization of vessel structures**
* Computes both:

  * **Intensity-based overlap** (Manders, Pearson, Dice, Jaccard on weighted signals)
  * **Binary overlap** (pure intersection of vessel masks)
* Generates high-quality **visual overlays**:

  * Green channel, red channel, and yellow overlap highlights
  * Exported per-cell and summarized into a global CSV/Excel table

---

## 🏗 Project Structure

```
ArterialManager/
├── config/
│   └── config.py                # Centralized path configuration
├── library/
│   ├── TableCreator/
│   │   └── generate_tab_dataset.py   # Core table pipeline
│   └── ImageOverlap/
│       └── wavelet_overlap.py        # Vessel overlap analysis
├── src/
│   ├── app_streamlit.py          # Streamlit app for table pipeline
│   ├── app_plot.py               # Streamlit app for interactive plots
│   └── run_overlap.py            # CLI script for vessel overlap analysis
├── data/
│   ├── tables/                   # Input Excel tables
│   └── imgs/                     # Input microscopy images (C=0 green, C=1 red)
├── results/
│   ├── master_table/             # Processed tables
│   └── overlap_images/           # Processed overlays + metrics
├── environment.yml               # Conda environment definition
└── README.md                     # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone <your_repo_url>
   cd ArterialManager
   ```

2. Create the environment:

   ```bash
   conda env create -f environment.yml
   conda activate arterial_manager
   ```

---

## ▶️ Usage

### 1. Run the Table Manager (Excel processing)

**Streamlit Web App**:

```bash
streamlit run src/app_streamlit.py
```

**Tkinter Desktop App**:

```bash
python src/app_tkinter.py
```

Inputs: `data/tables/`
Outputs: `results/master_table/`

---

### 2. Run the Vessel Overlap Analysis

**Python script (local or Colab):**

```bash
python src/run_overlap.py --input ./data/imgs --output ./results/overlap_images
```

* Input images must follow the convention:

  * `*_C=0.jpg` → green channel
  * `*_C=1.jpg` → red channel
  * Excludes `MERGE` and `jpgscale` files

* Outputs:

  * Per-cell subfolders with overlays (`.png`)
  * Metrics saved as `<cell_id>_metrics.xlsx`
  * Combined summary table: `all_metrics.csv`

**Colab (recommended for Google Drive datasets):**

```python
!git clone https://github.com/you/ArterialManager.git
%cd ArterialManager
from google.colab import drive
drive.mount('/content/drive')

!python src/run_overlap.py --input /content/drive/MyDrive/YourImages --output /content/drive/MyDrive/Results
```

---

## 📊 Output Examples

### Table Manager:

```
<StudyName> Angiotools Formated <timestamp>.xlsx
```

### Vessel Overlap:

* RGB overlays (green + red + yellow highlights)
* Heatmaps of intensity overlap
* Binary overlap masks
* Metrics (Manders, Pearson, Dice, Jaccard)

---

## 👤 Author

**Giorgio Ricciardiello Mejia**
📧 [giorgio.ricciardiellomejia@mountsinai.edu](mailto:giorgio.ricciardiellomejia@mountsinai.edu)

