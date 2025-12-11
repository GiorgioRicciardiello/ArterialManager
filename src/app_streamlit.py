"""
Streamlit constructor to tun the table
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
# Force project root into sys.path
ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))
from config.config import CONFIG
from library.TableCreator.generate_tab_dataset import GenerateDataset


st.title("📊 Generate Dataset Tool (Using CONFIG paths)")

# Extract paths from CONFIG
INPUT_DIR = CONFIG.get("paths")['path_tables']
OUTPUT_DIR = CONFIG.get("paths")['outputs_tabs']

st.markdown(f"""
This app automatically uses the paths defined in your **CONFIG** file:

- **Input folder (tables):** `{INPUT_DIR}`  
- **Output folder (results):** `{OUTPUT_DIR}`  
""")

if st.button("Run Pipeline 🚀"):
    try:
        # Run your pipeline
        generator = GenerateDataset(path_tables=INPUT_DIR, output_path=OUTPUT_DIR)
        df = generator.run()

        # Show preview of the resulting DataFrame
        st.success("✅ Processing completed successfully!")
        st.dataframe(df.head(50))  # Show first 50 rows for preview

        # Find the most recent Excel file saved in output
        excel_files = sorted(OUTPUT_DIR.glob("*.xlsx"),
                             key=lambda f: f.stat().st_mtime,
                             reverse=True)

        if excel_files:
            for excel_file in excel_files:
                with open(excel_file, "rb") as f:
                    st.download_button(
                        f"⬇️ Download {excel_file.stem}",
                        f,
                        file_name=excel_file.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
