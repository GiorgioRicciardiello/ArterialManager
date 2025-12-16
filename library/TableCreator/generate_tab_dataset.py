"""
The class points to the directory data/tables, where it expects to find for each cell the tables
- Angiotool.xlsx
- Cell count.xlsx
- Sample.xlsx


Categorical columns:
cell type
condition
Study Name
"""
from config.config import CONFIG
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import re
from datetime import datetime
import numpy as np
import warnings

# from src.olds_test.app_plot import fname


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

        self.col_time_point = 'Timepoint'
        self.col_study_name = 'Study Name'
        self.well_id = 'well_id'

        self.tables: Dict[str, pd.DataFrame] = {}

        # cols to normalize by cell count
        self.cols_norm = ['Vessels Area',
                     'Total Number of Junctions',
                     'Junctions Density',
                     'Total Vessels Length',
                     'Total Number of End Points']

        # columns that the output will have
        self.cols_formated = ['Study Name', 'Sample Name', self.well_id, "Image Name",
                              self.col_time_point, f'{self.col_time_point}_datetime',
                              "Cell type", "Density", "Condition", "Cell count",
                              "Vessels Area", "Total Number of Junctions Normalize", "Total Number of Junctions",
                              "Junctions Density Normalize", "Junctions Density", "Total Vessels Length Normalize",
                              "Total Vessels Length", "Total Number of End Points Normalize",
                              "Total Number of End Points"]


    def run(self) -> pd.DataFrame:

        df_cell_types = self._get_cell_type_files()
        formated_collection = {}
        for cell_type in df_cell_types['cell_type'].unique():
            # cell_type = 'ODQ22'
            excel_files = df_cell_types[df_cell_types['cell_type'] == cell_type]['file'].tolist()
            print(f"Processing cell type: {cell_type}")
            self.tables = self._collect_tables(excel_files=excel_files)
            self._format_angiotool()
            self._format_cellcount()
            self._format_sample()
            # self._check_typos_columns()
            df_formated = self._generate_formated_cell()
            formated_collection[cell_type] = df_formated

        # concatenate in a single table
        df_formated_merged = pd.concat(formated_collection)

        # Get unique labels from level 0 (in order of appearance)
        first_level_unique = df_formated_merged.index.get_level_values(0).unique()
        joined_names = "_".join(first_level_unique)

        df_formated_merged = df_formated_merged.reset_index(drop=False, inplace=False)
        df_formated_merged = df_formated_merged.rename(columns={'level_0': 'Experiment Name',
                                           'level_1': 'Experiment Row Idx'})
        self._save_results(df=df_formated_merged, study_name=joined_names)
        return df_formated


    def _check_typos_columns(self, tab_keys=None):
        """
        Ensure all tables have the same set of sample_name values.
        If any table differs, raise ValueError indicating precise differences.

        Parameters
        ----------
        tab_keys : list[str] or None
            If provided, only check these keys (e.g., ['angiotool', 'cell_count', 'sample']).
            If None, check all keys present in self.tables.

        Raises
        ------
        ValueError
            If sample_name sets differ across tables or sample_name column is missing.
        """

        # Choose which tables to check
        if tab_keys is None:
            keys_to_check = list(self.tables.keys())
        else:
            keys_to_check = [k for k in tab_keys if k in self.tables]

        if not keys_to_check:
            raise ValueError("No tables to check: 'self.tables' is empty or none of the requested keys are present.")

        # Helper: normalize sample_name series -> set
        def normalize_series_to_set(s: pd.Series) -> set:
            # Convert to string, strip spaces, lowercase
            s = s.astype(str).str.strip().str.lower()
            # Convert the literal "nan" (produced by astype(str)) back to np.nan, then drop
            s = s.replace({"nan": np.nan})
            s = s.dropna()
            # Drop empties
            s = s[s != ""]
            return set(s)

        # Build normalized sets per table
        table_sets = {}
        for key in keys_to_check:
            tab = self.tables.get(key)
            if tab is None:
                raise ValueError(f"Table '{key}' not found in self.tables.")

            if 'sample_name' not in tab.columns:
                raise ValueError(f"Table '{key}' is missing required column 'sample_name'.")

            table_sets[key] = normalize_series_to_set(tab['sample_name'])

        # If all sets are equal, we are done
        sets_list = list(table_sets.values())
        all_equal = all(s == sets_list[0] for s in sets_list[1:])
        if all_equal:
            return  # Success: no differences

        # Otherwise, compute a detailed difference report
        # Use the union of all names as the reference universe
        universe = set().union(*sets_list)

        details_lines = []
        for key, s in table_sets.items():
            missing = sorted(universe - s)  # present in others, missing here
            extras = sorted(s - (universe - s))  # unique names only this table has (relative extras)

            # More explicit "extras" relative to *overall* presence:
            # extras_here = s - set().union(*(table_sets[k2] for k2 in table_sets if k2 != key))
            extras_here = sorted(
                s - set().union(*(table_sets[k2] for k2 in table_sets if k2 != key))
            )

            # Prefer the explicit extras_here (names present only in this table)
            details_lines.append(
                f"- Table '{key}':\n"
                f"    Missing (in others but not here): {missing if missing else 'None'}\n"
                f"    Extras (only in this table): {extras_here if extras_here else 'None'}"
            )

        # Additionally, show pairwise differences for precision
        pairwise_lines = []
        keys = keys_to_check
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                s1, s2 = table_sets[k1], table_sets[k2]
                only_k1 = sorted(s1 - s2)
                only_k2 = sorted(s2 - s1)
                if only_k1 or only_k2:
                    pairwise_lines.append(
                        f"  • {k1} vs {k2}:\n"
                        f"      Only in {k1}: {only_k1 if only_k1 else 'None'}\n"
                        f"      Only in {k2}: {only_k2 if only_k2 else 'None'}"
                    )

        raise ValueError(
            "Mismatch in 'sample_name' sets across tables.\n\n"
            "Per-table differences:\n"
            + "\n".join(details_lines)
            + ("\n\nPairwise differences:\n" + "\n".join(pairwise_lines) if pairwise_lines else ""))



    def _get_cell_type_files(self) -> pd.DataFrame:
        """
        Retrieves and processes a list of Excel files within a specified directory, categorizing them
        by cell types and conditions. Ensures that each cell type corresponds to exactly three distinct
        conditions.

        :raises FileNotFoundError: If no Excel files are found in the specified directory.
        :raises ValueError: If any cell type does not have exactly three distinct conditions.

        :return: A pandas DataFrame containing the processed list of files with columns for cell type,
                 condition, file path, and normalized condition names.
        :rtype: pandas.DataFrame
        """
        excel_files = [
            f for f in self.path_tables.glob("*")
            if f.suffix.lower() in [".xls", ".xlsx"]
        ]

        if not excel_files:
            raise FileNotFoundError(f"No Excel files found in {self.path_tables}")

        # we can have the 3 files for multiple experiments, they are names as e.g., ODQ21 .., JC7 ...
        cell_types = [[file.name.partition(' ')[0],file.name.partition(' ')[2], file]  for file in excel_files]
        df_cell_types = pd.DataFrame(cell_types, columns=['cell_type', 'condition', 'file'])

        # Normalize condition names to avoid case/spacing mismatches
        df_norm = df_cell_types.copy()
        df_norm['condition_norm'] = (
            df_norm['condition'].astype(str).str.strip().str.lower()
        )

        # Count distinct conditions per cell_type
        cond_counts = (
            df_norm.groupby('cell_type')['condition_norm']
            .nunique()
            .reset_index(name='n_unique_conditions')
        )

        # Raise if any cell_type does not have exactly 3 distinct conditions
        offenders = cond_counts[cond_counts['n_unique_conditions'] != 3]
        if not offenders.empty:
            details = (
                df_norm.groupby(['cell_type', 'condition_norm'])
                .size()
                .reset_index(name='n_files')
                .sort_values(['cell_type', 'condition_norm'])
            )
            raise ValueError(
                "Each cell_type must have exactly 3 distinct conditions (angiotools, cell count and sample). \n"
                "You are missing a file(s)\n\n"
                f"Offending cell_types:\n{offenders.to_string(index=False)}\n\n"
                f"Details per condition:\n{details.to_string(index=False)}"
            )
        return df_cell_types

    # -------------------------------
    # STEP 1: SEARCH TABLES
    # -------------------------------
    def _collect_tables(self, excel_files:List[Path]) -> Dict[str, pd.DataFrame]:
        """
        Collect the the tables from the specific cell type and store them a Dict of DataFrames.

        :param excel_files: List of Excel files to search for tables.

        -------
        Dict[str, pd.DataFrame]
            Dictionary with keys ('angiotool', 'cell_count', 'sample')
            and corresponding DataFrames if found, else None.
        """
        tables = {k: None for k in self.keywords}

        if not excel_files:
            raise FileNotFoundError(f"No Excel files found in {self.path_tables}")

        for file in excel_files:
            fname = file.name.lower()
            print(fname)

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

        df = self.tables.get(key_tab)
        if df is None or df.empty:
            raise ValueError("AngioTool table is empty or missing.")

        # Ensure enough rows for header
        if len(df) < 4:
            raise ValueError("AngioTool table does not contain enough rows.")


        # Reset headers
        row_header = 3
        df.columns = df.iloc[row_header]
        df = df.iloc[row_header + 1:].reset_index(drop=True)

        # Validate required column
        if "Image Name" not in df.columns:
            raise KeyError("Column 'Image Name' missing in AngioTool table.")

        # 2. Create the col_time_point columns
        df[self.col_time_point] = df["Image Name"].apply(self._get_time_point_from_name)

        # --- Create datetime version of the timepoint ---

        df[f"{self.col_time_point}_datetime"] = df[self.col_time_point].apply(self._convert_timepoint_to_datetime)

        # Get the study name

        # Get the study name
        df[self.col_study_name] = df["Image Name"].apply(self._get_study_name_from_name)

        # Ensure consistent study name
        self.study_name = df["Study Name"].unique()
        if len(self.study_name) > 1:
            raise ValueError(f"⚠️ Multiple study names found: {self.study_name}. Remove this code section of keep only "
                             f"one study name.")
        self.study_name = self.study_name[0]

        # cell name independent to the time stamp
        df['sample_name'] = df["Image Name"].apply(self._get_cell_name)

        col_numeric = ['Explant Area',
                       'Vessels Area',
                       'Vessels Percentage Area',
                       'Total Number of Junctions',
                       'Junctions Density',
                       'Average Vessels Length',
                       'Total Number of End Points',
                       'Average Vessel Diameter',
                       'Medial E Lacunarity',
                       'E Lacunarity Gradient',
                       'Medial F Lacunarity',
                       'F Lacunarity Gradient',]
        for col in col_numeric:
            df[col] = df[col].astype(float)

        self.tables[key_tab] = df

    @staticmethod
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


    @staticmethod
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

    @staticmethod
    def _get_study_name_from_name(x: str) -> str:
        """
        Extract study name from filename.
        Example: 'JC3_A1_1_00d07h32m.tif_green.jpg' -> 'JC3'
        """
        return x.split("_")[0] if "_" in x else x


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
        We need to extract the time of the image so we can later merge with the Angiotool
        Creates the sample_name pivot column for the merge
        :param key_tab_cell_count:
        :return:
        """
        df = self.tables.get(key_tab_cell_count)
        if df is None or df.empty:
            raise ValueError("Cell count table is empty or missing.")

        if "File" not in df.columns:
            raise KeyError("Column 'File' missing in Cell Count table.")

        df["sample_name"] = df["File"].apply(self._get_cell_name)

        #  Create the col_time_point columns
        df[self.col_time_point] = df["File"].apply(self._get_time_point_from_name)

        # --- Create datetime version of the timepoint ---
        df[f"{self.col_time_point}_datetime"] = df[self.col_time_point].apply(self._convert_timepoint_to_datetime)


        self.tables[key_tab_cell_count] = df

    def _format_sample(self):
        """
        Pivot column for merging the tables is the sample_name columns, standardize
        :return:
        """
        df = self.tables.get(self.key_tab_sample)
        if df is None or df.empty:
            raise ValueError("Sample table is empty or missing.")

        if "Sample name" not in df.columns:
            raise KeyError("Column 'Sample name' missing in Sample table.")
        df["Sample name"] = df["Sample name"].replace(' ', '')
        self.tables[self.key_tab_sample] = df.rename(columns={"Sample name": "sample_name"})


    # -------------------------------
    # STEP 5: MERGE + NORMALIZE
    # -------------------------------
    def _generate_formated_cell(self) -> pd.DataFrame :
        """
        Merge the three different table sourcs into a single table and compute normalizations
        It also saves the table with the pre-specified columns
        :return:
        """

        angiotool = set(self.tables[self.key_tab_angiotool]['sample_name'])
        cellcount = set(self.tables[self.key_tab_cell_count]['sample_name'])

        # Symmetric difference: elements found in exactly one set (not both)
        diff = angiotool ^ cellcount

        if len(diff) > 0:
            missing_in_angiotool = cellcount - angiotool
            missing_in_cellcount = angiotool - cellcount
            warnings.warn(
                "Sample name mismatch between Angiotool and Cell count.\n"
                f"Missing in Angiotool: {sorted(missing_in_angiotool)}\n"
                f"Missing in Cell count: {sorted(missing_in_cellcount)}",
                category=UserWarning,
                stacklevel=2
            )

        df_ang = self.tables[self.key_tab_angiotool].copy()
        # df_ang.sort_values(by=[self.col_study_name, 'sample_name', self.col_time_point], inplace=True)
        df_ang_count = df_ang.groupby(by=[self.col_study_name, 'sample_name']).count()

        # df_cell = self.tables[self.key_tab_cell_count].copy()

        df = pd.merge(
            left=self.tables[self.key_tab_angiotool],
            right=self.tables[self.key_tab_cell_count],
            # on=["sample_name", self.col_time_point],
            on=["sample_name"],
            how="left",
        )
        col_drop = [col for col in df.columns if col.endswith('_y')]
        df = df.drop(columns=col_drop)
        col_rename = {col: col.replace('_x', '') for col in df.columns if col.endswith('_x')}
        df = df.rename(columns=col_rename)

        df_count = df.groupby(by=[self.col_study_name, 'sample_name']).count()

        # Compare counts of each DataFrame by Study Name and Sample Name. Otherwise, we have duplicates in the merge
        df_ang_count_grouped = df_ang_count.groupby(['Study Name', 'sample_name']).size()
        df_count_grouped = df_count.groupby(['Study Name', 'sample_name']).size()

        if df_ang_count_grouped.equals(df_count_grouped):
            print("The counts match between both DataFrames.")
        else:
            print("The counts do not match. Investigate further.")

        sample = set(self.tables[self.key_tab_sample]['sample_name'])
        out  = set(df['sample_name'])
        diff = sample ^ out
        if len(diff) > 0:
            missing_in_sample = out - sample
            missing_in_out = sample - out
            warnings.warn(
                "Sample name mismatch between Final Frame and Sample.\n"
                f"Missing in Angiotool: {sorted(missing_in_sample)}\n"
                f"Missing in Cell count: {sorted(missing_in_out)}",
                category=UserWarning,
                stacklevel=2
            )

        # df_sample = self.tables[self.key_tab_sample].copy()
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
                "Sample condition": "Condition"
            },
            inplace=True,
        )
        # create columns if missing, not all the sample columns are the same, we can have density of condition
        if 'Density' not in df.columns:
            df['Density'] = np.nan

        if 'Condition' not in df.columns:
            df['Condition'] = np.nan

        df = self._get_well_id(df)
        df = df[self.cols_formated]

        df = df.sort_values(by=[self.col_study_name, self.col_time_point])

        self._save_results(df)
        return df

    def _get_well_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.well_id] = df['Sample Name'].str.extract(r'_(?=[A-H])([A-H]\d{1,2})_')[0]
        return df

    # -------------------------------
    # STEP 6: SAVE TO EXCEL
    # -------------------------------
    def _save_results(self, df: pd.DataFrame,
                      study_name: str = None) -> None:
        if study_name is None:
            study_name = self.study_name

        timestamp = datetime.now().strftime("%d_%m_%Y_%H%M%S")
        file_path = self.output_path.joinpath(
            f"{study_name} Angiotools Formated {timestamp}.xlsx"
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

            print(f"✅ Formatted Table for Study {study_name} saved at:\n\t{file_path}")
        except PermissionError:
            print(f"Could not save file. It might be open in Excel: {file_path}")


if __name__ == "__main__":
    generator = GenerateDataset(path_tables=CONFIG.get('paths')['path_tables'],
                                output_path=CONFIG.get('paths')['outputs_tabs'])

    # generator = GenerateDataset(path_tables=CONFIG.get('paths')['data'].joinpath('sample_tab'),
    #                             output_path=CONFIG.get('paths')['data'].joinpath('sample_tab_output'))


    df = generator.run()
