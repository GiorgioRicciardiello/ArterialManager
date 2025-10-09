from config.config import CONFIG
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import re
from datetime import datetime
import numpy as np


class GenerateDataset:
    def __init__(self, path_tables: Path,
                 output_path: Path, ):
        """
        We will be building the angitool table section by section until we get the final output and get rid of the
        other columns

            create_tab = GenerateDataset(path_tables=CONFIG.get("paths")['path_tables'],
                                 output_path= CONFIG.get("paths")['outputs_tabs'])
            create_tab.run()


        :param path_tables:
        """

        self.path_tables = path_tables
        self.output_path = output_path

        self.keywords = {
            "angiotool": "angiotool",  # table from Angio Tool
            "cell_count": "cell count",  # cell count
            "sample": "sample"
        }
        self.key_tab_cell_count: str = 'cell_count'
        self.key_tab_angiotool:str = 'angiotool'
        self.key_tab_sample:str = 'sample'

        self.tables: Dict[str, pd.DataFrame] = {}

        # cols to normalize by cell count
        self.cols_norm = ['Vessels Area',
                     'Total Number of Junctions',
                     'Junctions Density',
                     'Total Vessels Length',
                     'Total Number of End Points']

        # columns that the output will have
        self.cols_formated = list(
            {'Study Name', 'Sample Name', "Image Name", "Timepoint", 'Timepoint_datetime', "Cell type", "Density", "Cell count",
             "Vessels Area", "Total Number of Junctions Normalize", "Total Number of Junctions",
             "Junctions Density Normalize", "Junctions Density", "Total Vessels Length Normalize",
             "Total Vessels Length", "Total Number of End Points Normalize", "Total Number of End Points"})


    def run(self) -> pd.DataFrame:
        self.tables = self._search_tables()
        self._format_angiotool()
        self._format_cellcount()
        self._format_sample()
        df_formated = self._generate_formated_cell()
        return df_formated

    # -------------------------------
    # STEP 1: SEARCH TABLES
    # -------------------------------
    def _search_tables(self) -> Dict[str, List[pd.DataFrame]]:
        """
        Search for specific Excel tables in a folder and load them into DataFrames.


        Sample frame: Sample name	Cell line	Sample density
        cell count frame: File	Cells
        angiotool: multi columns header
        Parameters
        ----------
        path_tables : Path
            Directory containing Excel files.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary with keys ('angiotool', 'cell_count', 'sample')
            and corresponding DataFrames if found, else None.
        """
        tables = {k: None for k in self.keywords}
        excel_files = list(self.path_tables.glob("*.xls*"))

        if not excel_files:
            raise FileNotFoundError(f"No Excel files found in {self.path_tables}")

        for file in excel_files:
            fname = file.name.lower()
            for key, kw in self.keywords.items():
                if kw in fname:
                    try:
                        tables[key] = pd.read_excel(file)
                        print(f"✅ Loaded {key} from {file.name}")
                    except Exception as e:
                        print(f"⚠️ Could not read {file.name}: {e}")

        missing = [k for k, v in tables.items() if v is None]
        if missing:
            raise ValueError(f"Missing required tables: {missing}")

        return tables


    # -------------------------------
    # STEP 2: FORMAT ANGIOTOOL
    # -------------------------------
    def _format_angiotool(self, key_tab:str='angiotool') :
        """
        Reformat the angiotool table:
        - Use the 3rd row as the header
        - Drop the rows above it
        Create the pivot column sample_name to pivot with the other tables
        """
        def _get_time_point_from_name(x: str) -> str:
            """
            Extract time point from image filename.
            Example: 'JC3_A1_1_00d07h32m.tif_green.jpg' -> '00d07h32m'
            """
            # safer with regex: look for pattern like 00d07h32m
            match = re.search(r"\d{2}d\d{2}h\d{2}m", x)
            if match:
                return match.group(0)
            return ""

        def _get_study_name_from_name(x: str) -> str:
            """
            Extract study name from filename.
            Example: 'JC3_A1_1_00d07h32m.tif_green.jpg' -> 'JC3'
            """
            return x.split("_")[0] if "_" in x else x

        col_time_point = 'Timepoint'
        col_study_name = 'Study Name'

        df = self.tables.get(key_tab)
        if df is None or df.empty:
            raise ValueError("❌ AngioTool table is empty or missing.")

        # Ensure enough rows for header
        if len(df) < 4:
            raise ValueError("❌ AngioTool table does not contain enough rows.")


        # Reset headers
        row_header = 3
        df.columns = df.iloc[row_header]
        df = df.iloc[row_header + 1:].reset_index(drop=True)

        # Validate required column
        if "Image Name" not in df.columns:
            raise KeyError("❌ Column 'Image Name' missing in AngioTool table.")

        # 2. Create the col_time_point columns
        df[col_time_point] = df["Image Name"].apply(_get_time_point_from_name)

        # --- Create datetime version of the timepoint ---
        def _convert_timepoint_to_datetime(tp: str) -> Optional[datetime]:
            """
            Convert strings like '00d07h32m' into datetime objects (days, hours, minutes).
            Returns None if the format is invalid.
            """
            match = re.match(r"(\d{2})d(\d{2})h(\d{2})m", tp)
            if match:
                days, hours, minutes = map(int, match.groups())
                return datetime(2000, 1, 1) + pd.Timedelta(days=days, hours=hours, minutes=minutes)
            return pd.NaT

        df["Timepoint_datetime"] = df[col_time_point].apply(_convert_timepoint_to_datetime)

        # Get the study name

        # Get the study name
        df[col_study_name] = df["Image Name"].apply(_get_study_name_from_name)

        # Ensure consistent study name
        self.study_name = df["Study Name"].unique()
        if len(self.study_name) > 1:
            raise ValueError(f"⚠️ Multiple study names found: {self.study_name}. Remove this code section of keep only "
                             f"one study name.")
        self.study_name = self.study_name[0]

        # cell name independent to the time stamp
        df['sample_name'] = df["Image Name"].apply(self._get_cell_name)
        self.tables[key_tab] = df

    @staticmethod
    def _get_cell_name(x: str) -> str:
        """
        Extract cell name by cutting right before the timepoint (XXdXXhXXm).
        Example:
          'JC3_A1_1_00d07h32m.tif_green' -> 'JC3_A1_1'
        """
        name = Path(x).stem  # remove extension (.jpg, .tif, etc.)

        # Find the timepoint pattern
        match = re.search(r"\d{2}d\d{2}h\d{2}m", name)
        if match:
            return name[: match.start() - 1]  # cut before "_00d..."
        return name

    # -------------------------------
    # STEP 3: FORMAT CELL COUNT
    # -------------------------------
    def _format_cellcount(self, key_tab_cell_count:str='cell_count'):
        """
        Format the cell count tables with columns:
                | File	| Cells
        Creates the sample_name pivot column for the merge
        :param key_tab_cell_count:
        :return:
        """
        df = self.tables.get(key_tab_cell_count)
        if df is None or df.empty:
            raise ValueError("❌ Cell count table is empty or missing.")

        if "File" not in df.columns:
            raise KeyError("❌ Column 'File' missing in Cell Count table.")

        df["sample_name"] = df["File"].apply(self._get_cell_name)
        self.tables[key_tab_cell_count] = df

    def _format_sample(self):
        """
        Pivot column for merging the tables is the sample_name columns, standardize
        :return:
        """
        df = self.tables.get(self.key_tab_sample)
        if df is None or df.empty:
            raise ValueError("❌ Sample table is empty or missing.")

        if "Sample name" not in df.columns:
            raise KeyError("❌ Column 'Sample name' missing in Sample table.")

        self.tables[self.key_tab_sample] = df.rename(columns={"Sample name": "sample_name"})


    # -------------------------------
    # STEP 5: MERGE + NORMALIZE
    # -------------------------------
    def _generate_formated_cell(self) -> pd.DataFrame :
        """
        Merge the three different table sourcs into a single table and compute normalizations
        :return:
        """
        df = pd.merge(
            left=self.tables[self.key_tab_cell_count],
            right=self.tables[self.key_tab_angiotool],
            on="sample_name",
            how="right",
        )

        df = pd.merge(
            left=df,
            right=self.tables[self.key_tab_sample],
            on="sample_name",
            how="right",
        )

        # Normalize vessel features
        for col_to_norm in self.cols_norm:
            if col_to_norm in df.columns:
                new_col = f"{col_to_norm} Normalize"
                df[new_col] = df[col_to_norm] / df["Cells"].replace(0, np.nan)
                if new_col not in self.cols_formated:
                    self.cols_formated.append(new_col)
            else:
                print(f"⚠️ Column {col_to_norm} missing, skipping normalization.")


        # Final structure
        df.rename(
            columns={
                "sample_name": "Sample Name",
                "Cell line": "Cell type",
                "Cells": "Cell count",
                "Sample density": "Density",
            },
            inplace=True,
        )

        df = df[self.cols_formated]
        self._save_results(df)
        return df

    # -------------------------------
    # STEP 6: SAVE TO EXCEL
    # -------------------------------
    def _save_results(self, df: pd.DataFrame) -> None:
        timestamp = datetime.now().strftime("%d_%m_%Y_%H%M%S")
        file_path = self.output_path.joinpath(
            f"{self.study_name} Angiotools Formated {timestamp}.xlsx"
        )

        try:
            with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Results")

                worksheet = writer.sheets["Results"]

                # Auto-adjust column widths
                for i, col in enumerate(df.columns):
                    try:
                        max_len = int(df[col].astype(str).map(len).max())
                    except Exception:
                        max_len = len(str(col))
                    col_width = max(max_len, len(str(col))) + 2
                    worksheet.set_column(i, i, col_width)

                # Convert to Excel table
                (max_row, max_col) = df.shape
                worksheet.add_table(
                    0,
                    0,
                    max_row,
                    max_col - 1,
                    {
                        "columns": [{"header": col} for col in df.columns],
                        "name": "ResultsTable",
                        "style": "Table Style Medium 9",
                    },
                )

                # Freeze header row
                worksheet.freeze_panes(1, 0)

            print(f"✅ Formatted Table for Study {self.study_name} saved at:\n\t{file_path}")
        except PermissionError:
            print(f"❌ Could not save file. It might be open in Excel: {file_path}")


if __name__ == "__main__":
    generator = GenerateDataset(path_tables=CONFIG.get('paths')['path_tables'],
                                output_path=CONFIG.get('paths')['outputs_tabs'])

    df = generator.run()
