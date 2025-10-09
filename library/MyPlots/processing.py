# ---------- data_utils.py ----------
import pandas as pd
import numpy as np
import re
from datetime import datetime

# ----------------------------------
# LOAD & PREPROCESS
# ----------------------------------

def load_df(file, sheet_name=None):
    """
    Load CSV/XLSX/TSV file into DataFrame and perform auto-cleaning.
    Also detects AngioTool-style timepoint strings like '00d07h32m'.
    """
    if file is None:
        return None

    name = getattr(file, "name", "")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    elif name.lower().endswith(".tsv"):
        df = pd.read_csv(file, sep="\t")
    elif name.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(file, sheet_name=sheet_name or 0)
    else:
        raise ValueError("Unsupported file type.")

    # --- Auto-detect and parse timepoints ---
    if "Timepoint" in df.columns and "Timepoint_datetime" not in df.columns:
        df["Timepoint_datetime"] = df["Timepoint"].apply(_convert_timepoint_to_datetime)

    # --- Auto-sort if datetime exists ---
    if "Timepoint_datetime" in df.columns:
        df = df.sort_values("Timepoint_datetime").reset_index(drop=True)

    return df


def _convert_timepoint_to_datetime(tp: str):
    """
    Convert '00d07h32m' → datetime (anchor date 2000-01-01).
    """
    if not isinstance(tp, str):
        return pd.NaT
    match = re.match(r"(\d{2})d(\d{2})h(\d{2})m", tp)
    if match:
        d, h, m = map(int, match.groups())
        return datetime(2000, 1, 1) + pd.Timedelta(days=d, hours=h, minutes=m)
    return pd.NaT


def aggregate_replicates(df: pd.DataFrame, x: str, y: str, hue: str|None):
    """
    Compute mean ± SEM for replicates grouped by (x, hue).
    Returns summarized dataframe.
    """
    group_cols = [x] + ([hue] if hue else [])
    agg = df.groupby(group_cols)[y].agg(['mean', 'sem']).reset_index()
    agg.rename(columns={'mean': y, 'sem': f"{y}_sem"}, inplace=True)
    return agg
