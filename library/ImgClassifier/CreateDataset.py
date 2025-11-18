import pandas as pd
from pathlib import Path
from  typing import List

def make_ml_feature_table(path_in: str | Path,
                            path_out: str | Path,
                            id_col: str = "_cell",
                            path_imgs: str | Path = None,
                            img_suffix: str = "_overlay_green_red.png",
                            overwrite: bool = False) -> pd.DataFrame:
    """
    Convert long-format colocalization metrics into one-row-per-cell ML feature table.

    Parameters
    ----------
    path_in : str or Path
        Input Excel/CSV file with ['Metric', 'Value', 'method', 'cell'] columns.
    path_out : str or Path
        Output Excel file path.
    overwrite : bool, default False
        If False and file exists, function returns existing file path instead of overwriting.

    Returns
    -------
    pd.DataFrame or None
        Wide-format DataFrame if created, else None if skipped.
    """

    def include_img_path(df: pd.DataFrame,
                           id_col: str = "cell",
                           path_imgs: str | Path = None,
                           img_suffix: str = "_overlay_green_red.png") -> pd.DataFrame:
        """
        Recursively search for image files matching f"{cell}{img_suffix}" in nested folders.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing cell identifiers.
        id_col : str, default "_cell"
            Column name with unique cell identifiers.
        path_imgs : str or Path
            Base directory where images are stored (may contain nested folders).
        img_suffix : str, default "_overlay_green_red.png"
            Filename suffix pattern.

        Returns
        -------
        pd.DataFrame
            DataFrame with new column 'img_path' containing full path or None if not found.
        """
        path_imgs = Path(path_imgs)
        df = df.copy()

        img_paths = []
        for cell in df[id_col]:
            target_name = f"{cell}{img_suffix}"
            # search recursively in all subfolders
            match = next(path_imgs.rglob(target_name), None)
            img_paths.append(str(match) if match else None)

        df["img_path"] = img_paths
        return df

    path_in, path_out = Path(path_in), Path(path_out)

    if path_out.exists() and not overwrite:
        print(f"⚠️ File already exists, skipping: {path_out}")
        return pd.read_excel(path_out)

    # read input
    df = pd.read_excel(path_in)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # pivot without aggregation
    wide = (
        df.pivot(index="cell", columns=["method", "Metric"], values="Value")
          .reset_index()
    )
    # flatten multi-index
    wide.columns = [
        "cell" if c[0] == "" else f"{c[1]}_{c[0]}" for c in wide.columns.to_flat_index()
    ]
    wide = wide.reindex(sorted(wide.columns), axis=1)
    wide.rename(columns={"_cell": id_col}, inplace=True)

    # sanity check
    assert df[id_col].nunique() == wide[id_col].nunique(), "Cell count mismatch."

    # include image path
    wide = include_img_path(df=wide,
                            id_col=id_col,
                            path_imgs=path_imgs,
                            img_suffix=img_suffix
                            )

    # save output
    wide.to_excel(path_out, index=False)
    print(f"✅ ML feature table saved to {path_out}")
    return wide

def summarize_ml_features(df: pd.DataFrame, features:List[str]) -> pd.DataFrame:
    """
    Compute first-order statistics for each numeric feature column in an ML table.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format ML feature table (one row per sample).

    Returns
    -------
    pd.DataFrame
        Summary statistics for each numeric feature:
        [nan_count, mean, median, std, min, max, skew].
    """
    stats = []
    numeric_df = df[features]

    for col in numeric_df.columns:
        s = numeric_df[col]
        desc = {
            "feature": col,
            "count": s.count(),
            "nan_count": s.isna().sum(),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "skew": s.skew()
        }
        stats.append(desc)

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values(by="feature").reset_index(drop=True)
    return stats_df

