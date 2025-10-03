
import io
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# Optional (linear trendline): requires statsmodels in requirements
try:
    import statsmodels.api as sm  # noqa: F401
    _HAS_SM = True
except Exception:
    _HAS_SM = False

# ------------- Helpers -------------

@st.cache_data
def load_df(file, sheet_name=None):
    if file is None:
        return None
    name = getattr(file, "name", "")
    if name.lower().endswith(".csv"):
        return pd.read_csv(file)
    if name.lower().endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    if name.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(file, sheet_name=sheet_name or 0)
    # Fallback: try pandas sniffer
    return pd.read_csv(file)

def is_integer_series(s: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        # Check if all finite values are close to ints
        vals = s.dropna().values
        if len(vals) == 0:
            return False
        return np.allclose(vals, np.round(vals))
    return False

def classify_series(s: pd.Series) -> str:
    """Return one of: 'continuous', 'binary', 'ordinal', 'categorical'"""
    # Booleans → binary
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    # Numeric
    if pd.api.types.is_numeric_dtype(s):
        nunique = s.nunique(dropna=True)
        # All integers with small unique set → binary/ordinal
        if is_integer_series(s):
            if nunique <= 2:
                return "binary"
            if 3 <= nunique <= 12:
                return "ordinal"
        # Otherwise consider continuous if enough unique spread
        if nunique > 15:
            return "continuous"
        # Default fallback for small unique numeric
        if nunique <= 2:
            return "binary"
        return "ordinal"
    # Non-numeric
    nunique = s.nunique(dropna=True)
    if nunique <= 2:
        return "binary"
    return "categorical"

def coerce_dtype_by_role(df: pd.DataFrame, role_map: dict) -> pd.DataFrame:
    """Coerce dtypes so plotting behaves nicely (e.g., ordinal as categorical with order)."""
    out = df.copy()
    for col, role in role_map.items():
        if col not in out.columns:
            continue
        if role in ("binary", "ordinal", "categorical"):
            # Cast to category to ensure discrete axes / grouping
            out[col] = out[col].astype("category")
            if role == "binary":
                # Ensure consistent ordering if possible
                try:
                    # Keep 0/1 or False/True order if present
                    cats = list(out[col].cat.categories)
                    if set(cats) == {0, 1}:
                        out[col] = out[col].cat.reorder_categories([0, 1], ordered=True)
                    elif set(map(str, cats)) == {"0", "1"}:
                        out[col] = out[col].cat.reorder_categories(["0", "1"], ordered=True)
                except Exception:
                    pass
        elif role == "continuous":
            # Try to make numeric
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def auto_plot(df: pd.DataFrame, x: str, y: str, hue: str|None, role_map: dict, add_trend: bool):
    x_role = role_map.get(x, classify_series(df[x]))
    y_role = role_map.get(y, classify_series(df[y]))
    hue_role = None
    if hue:
        hue_role = role_map.get(hue, classify_series(df[hue]))

    # Cast dtypes sensibly
    dfp = coerce_dtype_by_role(df[[c for c in [x,y,hue] if c]], {k:v for k,v in role_map.items() if k in [x,y,hue]})

    # Case 1: both continuous → scatter
    if x_role == "continuous" and y_role == "continuous":
        trend = "ols" if add_trend else None
        fig = px.scatter(dfp, x=x, y=y, color=hue if hue else None, trendline=trend)
        fig.update_layout(margin=dict(l=0,r=0,t=50,b=0))
        return fig, "scatter"

    # Case 2: one continuous, one binary/ordinal → box + strip overlay (jitter)
    roles = {x: x_role, y: y_role}
    cont = None
    disc = None
    for col, role in roles.items():
        if role == "continuous":
            cont = col
        if role in ("binary","ordinal","categorical"):
            disc = col

    if cont and disc:
        # Base box
        fig = px.box(dfp, x=disc, y=cont, color=hue if hue else None, points=False)
        # Overlay strip (jitter)
        strip = px.strip(dfp, x=disc, y=cont, color=hue if hue else None)
        for tr in strip.data:
            tr.update(marker=dict(opacity=0.35))  # set transparency here
            fig.add_trace(tr)
        for tr in strip.data:
            fig.add_trace(tr)
        fig.update_layout(margin=dict(l=0,r=0,t=50,b=0))
        return fig, "box+strip"

    # Fallback: both discrete → count plot
    fig = px.histogram(dfp, x=x, color=hue if hue else None, barmode="group")
    fig.update_layout(margin=dict(l=0,r=0,t=50,b=0))
    return fig, "histogram"

def add_annotations(fig: go.Figure, annos: list[dict]):
    for a in annos:
        fig.add_annotation(
            x=a["x"], y=a["y"],
            text=a["text"],
            showarrow=a.get("showarrow", True),
            arrowhead=2, ax=a.get("ax", 0), ay=a.get("ay", -40),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="black",
            borderwidth=0.5,
            font=dict(size=12)
        )

def fig_to_png_bytes(fig: go.Figure) -> bytes:
    # Requires "kaleido" in requirements
    return fig.to_image(format="png", scale=2)

# ------------- App UI -------------

st.set_page_config(page_title="PlotLab – Research Plotter", layout="wide")
st.title("🧪 Experiment Visualizer")

with st.sidebar:
    st.header("1) Load data")
    data_file = st.file_uploader("Upload CSV/XLSX", type=["csv","tsv","xls","xlsx"])
    sheet_name = st.text_input("Excel sheet (optional)", value="")
    df = load_df(data_file, sheet_name or None)
    if df is not None and len(df) > 0:
        st.success(f"Loaded shape: {df.shape}")
        st.caption("Tip: You can override detected variable roles below.")
    else:
        st.info("Upload a dataset to begin.")
        st.stop()

    st.header("2) Select variables")

    cols = df.columns.tolist()
    x = st.selectbox("X", options=cols)
    y = st.selectbox("Y", options=[c for c in cols if c != x])
    hue = st.selectbox("Hue (optional)", options=["(none)"] + [c for c in cols if c not in (x,y)])
    if hue == "(none)":
        hue = None

    # Auto detect roles
    detected = {c: classify_series(df[c]) for c in [x,y] + ([hue] if hue else [])}
    st.caption("Auto-detected roles: " + ", ".join([f"{k}: {v}" for k,v in detected.items()]))

    st.subheader("Override roles (optional)")
    role_options = ["continuous","binary","ordinal","categorical"]
    role_map = {}
    for c in [x, y] + ([hue] if hue else []):
        role_map[c] = st.selectbox(f"{c} role", options=role_options, index=role_options.index(detected[c]))

    add_trend = False
    if role_map[x] == role_map[y] == "continuous":
        add_trend = st.toggle("Add linear trendline (OLS)", value=False, help="Requires statsmodels.")

    st.header("3) Annotations")
    if "annotations" not in st.session_state:
        st.session_state["annotations"] = []

    with st.expander("Add annotation", expanded=False):
        ann_x = st.text_input("x", value="")
        ann_y = st.text_input("y", value="")
        ann_text = st.text_area("label", value="")
        colA, colB = st.columns(2)
        with colA:
            add_ann = st.button("➕ Add")
        with colB:
            clear_ann = st.button("🗑️ Clear all")

        if add_ann:
            try:
                x_val = float(ann_x)
                y_val = float(ann_y)
                st.session_state["annotations"].append({"x": x_val, "y": y_val, "text": ann_text, "showarrow": True})
                st.success("Annotation added.")
            except Exception as e:
                st.error(f"Could not add annotation. Ensure x/y are numeric. Details: {e}")
        if clear_ann:
            st.session_state["annotations"] = []
            st.success("Cleared annotations.")

st.subheader("Preview")
fig, plot_kind = auto_plot(df, x, y, hue, role_map, add_trend)
add_annotations(fig, st.session_state["annotations"])

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.divider()

st.subheader("Save figure")
title = st.text_input("Figure title (optional)", value=f"{plot_kind.capitalize()} of {y} by {x}")
meta = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "x": x, "y": y, "hue": hue,
    "roles": role_map,
    "plot_kind": plot_kind,
    "annotations": st.session_state["annotations"],
    "title": title,
}
fname = st.text_input("Base filename (no extension)", value=f"plot_{int(time.time())}")
col1, col2 = st.columns(2)
with col1:
    save_png = st.button("💾 Save PNG + JSON")
with col2:
    dl_fig = st.download_button("⬇️ Download PNG now", data=fig_to_png_bytes(fig), file_name=f"{fname}.png", mime="image/png")

if save_png:
    try:
        # Title in layout
        if title:
            fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left"})
        png_bytes = fig_to_png_bytes(fig)
        with open(f"{fname}.png", "wb") as f:
            f.write(png_bytes)
        with open(f"{fname}.json", "w") as f:
            json.dump(meta, f, indent=2)
        st.success(f"Saved {fname}.png and {fname}.json in current working directory.")
    except Exception as e:
        st.error(f"Save failed: {e}")
