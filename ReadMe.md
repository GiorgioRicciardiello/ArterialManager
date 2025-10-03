# ArterialManager

📊 **ArterialManager** is a Python-based tool for processing angiogenesis data tables.
It merges, normalizes, and formats **Angiotool**, **Cell Count**, and **Sample** Excel files into a clean, ready-to-analyze master table.
The project includes both a **Streamlit web app** and a **desktop (Tkinter) GUI** for easy use.

---

## 🚀 Features

* Automatically detects and loads input Excel tables:

  * **Angiotool** output
  * **Cell count** table
  * **Sample metadata**
* Cleans and reformats headers
* Extracts **study names** and **timepoints** from filenames
* Normalizes vessel metrics by cell count
* Merges into a single output table with consistent structure
* Saves results as a styled **Excel file** with auto-sized columns and frozen headers
* Run via:

  * **Streamlit Web App** (browser interface with preview + download)
  * **Tkinter Desktop GUI** (local window with preview)

---

## 🏗 Project Structure

```
ArterialManager/
├── config/
│   └── config.py              # Centralized path configuration
├── library/
│   └── TableCreator/
│       └── generate_tab_dataset.py  # Core data pipeline
├── src/
│   └── app_streamlit.py       # Streamlit web app
├── data/
│   └── tables/                # Input Excel tables (Angiotool, Cell Count, Sample)
├── results/
│   └── master_table/          # Processed outputs
├── environment.yml            # Conda environment definition
└── README.md                  # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository and navigate into it:

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

### 1. Run the Streamlit Web App

From project root:

```bash
streamlit run app_streamlit.py
```

* Input tables are read from:
  `data/tables/`
* Outputs are written to:
  `results/master_table/`
* A preview of the merged table is shown in the browser
* Download the final Excel file with a single click

---

### 2. Run the Tkinter Desktop App (optional)

```bash
python src/app_tkinter.py
```

* Select input and output folders via a GUI window
* Run the pipeline and preview results inside the app

---

## 📂 Input Files

Place the following Excel files in `data/tables/`:

* `*Angiotool*.xlsx`
* `*Cell count*.xlsx`
* `*Sample*.xlsx`

The filenames must contain these keywords so the tool can detect them automatically.

---

## 📊 Output

Results are saved to `results/master_table/` with filenames like:

```
<StudyName> Angiotools Formated <timestamp>.xlsx
```

The output Excel includes:

* Study name
* Sample metadata
* Vessel features (raw + normalized by cell count)
* Formatted table style

---

## 📊 Interactive Plots

ArterialManager now includes an additional **Streamlit app** dedicated to **data visualization**:

* Located at:  
  `src/app_plot.py`

* Run it with:  
  ```bash
  streamlit run src/app_plot.py


## 🛠 Development Notes

* Paths are configured in **`config/config.py`**
* The core logic is in **`library/TableCreator/generate_tab_dataset.py`**
* The Streamlit app is the main entry point for users

---

## 👤 Author

**Giorgio Ricciardiello Mejia**
📧 [giorgio.ricciardiellomejia@mountsinai.edu](mailto:giorgio.ricciardiellomejia@mountsinai.edu)

