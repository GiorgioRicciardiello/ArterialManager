"""
Run the UMAP visualization with clik and display dynamic image
from terminal cd to this folder and then run the command

python app_umap_image_viewer.py

"""
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import base64
import io
import base64


# ==========================================================
# Utils
# ==========================================================
def encode_image(path):
    img = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# ==========================================================
# Load data (REPLACE with your real loading logic)
# ==========================================================
# results must contain:
# ['UMAP1', 'UMAP2', 'Outcome', 'CompositeScore', 'img_path']

results = pd.read_csv(
    r'C:\Users\riccig01\OneDrive\Projects\MtSinai\Fanny\ArterialManager\library\ImgClassifier\results.csv')

# Encode images once (important for performance)
results["img_b64"] = results["img_path"].apply(encode_image)


# ==========================================================
# Build UMAP figure
# ==========================================================
fig = px.scatter(
    results,
    x="UMAP1",
    y="UMAP2",
    color="Outcome",
    symbol="Outcome",
    size="CompositeScore",
    size_max=18,
    opacity=0.8,
)

fig.update_traces(
    marker=dict(line=dict(width=0.5, color="black")),
    hovertemplate=(
        "<b>Outcome:</b> %{customdata[0]}<br>"
        "<b>Composite score:</b> %{marker.size:.2f}<extra></extra>"
    )
)

fig.update_layout(
    margin=dict(l=40, r=40, t=40, b=40),
    legend_title_text="Outcome",
)


# ==========================================================
# Dash app
# ==========================================================
app = Dash(__name__)
server = app.server  # for deployment (gunicorn, etc.)

app.layout = html.Div(
    style={
        "display": "flex",
        "height": "100vh",
        "padding": "10px",
        "boxSizing": "border-box",
    },
    children=[

        # ---------------- Left: UMAP ----------------
        html.Div(
            style={"width": "60%", "height": "100%"},
            children=[
                dcc.Graph(
                    id="umap-graph",
                    figure=fig,
                    style={"height": "100%"},
                    clear_on_unhover=False,
                )
            ],
        ),

        # ---------------- Right: Image ----------------
        html.Div(
            style={
                "width": "40%",
                "height": "100%",
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center",
                "backgroundColor": "#000",
            },
            children=[
                html.Img(
                    id="image-viewer",
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "100%",
                        "objectFit": "contain",
                    },
                )
            ],
        ),
    ],
)


# ==========================================================
# Callbacks
# ==========================================================
@app.callback(
    Output("image-viewer", "src"),
    Input("umap-graph", "hoverData"),
)
def update_image(hoverData):
    if hoverData is None:
        return None

    point_idx = hoverData["points"][0]["pointIndex"]
    img_b64 = results.iloc[point_idx]["img_b64"]

    if img_b64 is None:
        return None

    return f"data:image/png;base64,{img_b64}"


# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    app.run(
        debug=True,
        port=8050,
    )