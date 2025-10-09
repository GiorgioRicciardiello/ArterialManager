# ---------- plot_utils.py ----------
import plotly.express as px
import plotly.graph_objects as go

def prettify_label(label: str) -> str:
    """
    Clean axis labels for display.
    Example: 'Vessels Area Normalize' → 'Vessels Area (Normalized)'
    """
    label = label.replace("_", " ").title()
    if "Normalize" in label:
        label = label.replace("Normalize", "(Normalized)")
    return label


def plot_with_sem(df, x, y, hue=None):
    """
    Plot line + error band if *_sem column exists (used after replicate aggregation).
    """
    y_sem = f"{y}_sem"
    fig = go.Figure()

    if hue and hue in df.columns:
        groups = df[hue].unique()
        for g in groups:
            sub = df[df[hue] == g]
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y],
                mode='lines+markers', name=str(g)
            ))
            if y_sem in sub.columns:
                fig.add_trace(go.Scatter(
                    x=list(sub[x]) + list(sub[x])[::-1],
                    y=list(sub[y] + sub[y_sem]) + list(sub[y] - sub[y_sem])[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,200,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{g} ±SEM",
                    showlegend=False
                ))
    else:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='lines+markers', name=y))
        if y_sem in df.columns:
            fig.add_trace(go.Scatter(
                x=list(df[x]) + list(df[x])[::-1],
                y=list(df[y] + df[y_sem]) + list(df[y] - df[y_sem])[::-1],
                fill='toself',
                fillcolor='rgba(0,100,200,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False
            ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis_title=prettify_label(x),
        yaxis_title=prettify_label(y),
    )
    return fig
