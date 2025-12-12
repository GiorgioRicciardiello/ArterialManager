# ---------- app.py ----------
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import time
from pathlib import Path
import sys
import uuid

# Add project root (ArterialManager) to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from library.MyPlots.processing import load_df, aggregate_replicates
from library.MyPlots.visualization import plot_with_sem, prettify_label, plot_grouped_data_interactive


# --- PAGE CONFIG ---
st.set_page_config(page_title="AngioTool Visualizer", layout="wide")
st.title("🧬 AngioTool Experiment Visualizer")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1️⃣ Load Data")
    file = st.file_uploader("Upload AngioTool output (CSV/XLSX)", type=["csv", "tsv", "xls", "xlsx"])
    sheet = st.text_input("Sheet name (optional)", "")
    df = load_df(file, sheet or None)
    if df is None:
        st.info("Upload a dataset to begin.")
        st.stop()
    st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # --- Smart defaults ---
    cols = sorted(df.columns.tolist(), key=str.lower)
    default_x = "Timepoint_datetime" if "Timepoint_datetime" in cols else cols[0]
    default_y = "Vessels Area Normalize" if "Vessels Area Normalize" in cols else (
        cols[1] if len(cols) > 1 else cols[0])
    default_hue = "Cell type" if "Cell type" in cols else None

    # --- Selection boxes (safe indices) ---
    plot_type = st.radio("Plot Type", ["Standard (Simple/SEM)", "Grouped Grid Cell Type x Condition"])

    if plot_type == "Standard (Simple/SEM)":
        x = st.selectbox("X-axis", options=cols, index=cols.index(default_x) if default_x in cols else 0)

        y_options = [c for c in cols if c != x]
        y_default_index = y_options.index(default_y) if default_y in y_options else 0
        y = st.selectbox("Y-axis", options=y_options, index=y_default_index)

        hue_options = ["(none)"] + [c for c in cols if c not in (x, y)]
        hue_default_index = (hue_options.index(default_hue) if default_hue in hue_options else 0)
        hue = st.selectbox("Color group (optional)", options=hue_options, index=hue_default_index)
        if hue == "(none)":
            hue = None
            
        st.header("3️⃣ Options")
        show_sem = st.checkbox("Aggregate replicates (mean ± SEM)", value=False)
        add_trend = st.checkbox("Add linear trendline", value=False)
        
    else: # Grouped Grid
        x = st.selectbox("X-axis (Time/Numeric)", options=cols, index=cols.index(default_x) if default_x in cols else 0)
        y = st.selectbox("Y-axis (Value)", options=[c for c in cols if c != x], index=0)
        
        hue_options = [c for c in cols if c not in (x, y)]
        # Default Subplot Group -> Cell type
        # Default Color Group -> Condition (or second available)
        
        subplot_col = st.selectbox("Subplot Group (Rows/Cols)", options=hue_options, index=0)
        color_col = st.selectbox("Color Group (Legend)", options=hue_options, index=1 if len(hue_options)>1 else 0)
        
        st.header("3️⃣ Options")
        add_trend = st.checkbox("Add linear trendline", value=True)


# --- DATA PREP & PLOT ---
st.subheader("📊 Plot Preview")

if plot_type == "Standard (Simple/SEM)":
    if show_sem:
        df_plot = aggregate_replicates(df, x, y, hue)
        fig = plot_with_sem(df_plot, x, y, hue)
    else:
        df_plot = df.copy()
        import plotly.express as px
        trend = "ols" if add_trend else None
        fig = px.scatter(df_plot, x=x, y=y, color=hue if hue else None, trendline=trend)
        fig.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis_title=prettify_label(x),
            yaxis_title=prettify_label(y),
        )
else:
    # Grouped Grid
    fig = plot_grouped_data_interactive(
        df,
        cat_col1=subplot_col,
        cat_col2=color_col,
        x_col=x,
        y_col=y,
        add_trend_line=add_trend
    )

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, 
                key=f"plot_{uuid.uuid4()}")

# --- EXPORT ---
st.divider()
st.subheader("💾 Save / Export")
title = st.text_input("Figure title", value=f"{y} vs {x}")
fname = st.text_input("Base filename", value=f"plot_{int(time.time())}")

meta = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "x": x, "y": y, 
    "plot_type": plot_type,
    "title": title,
    "source_file": getattr(file, "name", "unknown"),
}

if st.button("💾 Save PNG + JSON"):
    from plotly.io import to_image
    png_bytes = to_image(fig, format="png", scale=2)
    with open(f"{fname}.png", "wb") as f:
        f.write(png_bytes)
    with open(f"{fname}.json", "w") as f:
        json.dump(meta, f, indent=2)
    st.success(f"Saved {fname}.png and {fname}.json")

if st.download_button("⬇️ Download PNG", data=fig.to_image(format="png"),
                      file_name=f"{fname}.png", mime="image/png"):
    st.success("Download started!")

# --- Optional: export processed data ---
if st.button("📤 Export processed data as Excel"):
    # Re-generate processed df if needed or just export raw?
    # For now, export what we loaded as base, maybe filtered?
    # Simple is best:
    df.to_excel(f"{fname}_data.xlsx", index=False)
    st.success(f"Exported {fname}_data.xlsx")

