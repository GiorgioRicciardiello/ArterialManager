# ---------- plot_utils.py ----------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.colors as pcolors



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
        color_map = px.colors.qualitative.Set1  # Use a color palette
        for i, g in enumerate(groups):
            # cycle colors if groups > len(palette)
            c = color_map[i % len(color_map)]
            
            sub = df[df[hue] == g]
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y],
                mode='lines+markers', name=str(g), line=dict(color=c),
                legendgroup=str(g)  # Ensure highlight toggles with line
            ))
            if y_sem in sub.columns:
                # Handle color formats (Hex or RGB string)
                if c.startswith("#"):
                    rgb = colors.hex_to_rgb(c)
                elif c.startswith("rgb"):
                    # Extract numbers from "rgb(r, g, b)"
                    rgb = tuple(map(int, c.replace("rgb(", "").replace(")", "").split(",")))
                else:
                    # Fallback or other formats (e.g. named colors) - default to grey if unknown
                    rgb = (128, 128, 128)

                fillcolor = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.1)'

                fig.add_trace(go.Scatter(
                    x=list(sub[x]) + list(sub[x])[::-1],
                    y=list(sub[y] + sub[y_sem]) + list(sub[y] - sub[y_sem])[::-1],
                    fill='toself',
                    fillcolor=fillcolor,
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{g} ±SEM",
                    showlegend=False,
                    legendgroup=str(g)  # Ensure highlight toggles with line
                ))
    else:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='lines+markers', name=y))
        if y_sem in df.columns:
            fig.add_trace(go.Scatter(
                x=list(df[x]) + list(df[x])[::-1],
                y=list(df[y] + df[y_sem]) + list(df[y] - df[y_sem])[::-1],
                fill='toself',
                fillcolor='rgba(0,100,200,0.1)',  # Default color
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                legendgroup='no_hue'  # Treat this as one group for toggling
            ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis_title=prettify_label(x),
        yaxis_title=prettify_label(y),
    )
    return fig


def plot_grouped_data_interactive(df:pd.DataFrame,
                                  cat_col1: str = 'Cell type',
                                  cat_col2: str = 'Condition',
                                  y_col: str = 'Vessels Area Normalize',
                                  x_col: str = 'Timepoint_datetime',
                                  add_trend_line=False):
    """
    Interactive Plotly version of plot_grouped_data_with_optional_trend.
    """
    # Clean Data
    plot_df = df.copy()

    # Handle Datetime X-Axis (Convert to Numbers for Plotting/Regression)
    x_plot_col = 'x_numeric_seconds'
    is_datetime = pd.api.types.is_datetime64_any_dtype(plot_df[x_col])

    if is_datetime:
        start_time = plot_df[x_col].min()
        plot_df[x_plot_col] = (plot_df[x_col] - start_time).dt.total_seconds()
    else:
        plot_df[x_plot_col] = plot_df[x_col]

    # Grouping
    unique_groups = sorted(plot_df[cat_col1].unique())
    unique_conditions = sorted(plot_df[cat_col2].unique())

    # Layout calc: Fit on screen
    # Try to use up to 3 columns to reduce height
    max_cols = 3
    n_cols = min(len(unique_groups), max_cols)
    n_rows = (len(unique_groups) + n_cols - 1) // n_cols

    titles = [str(g) for g in unique_groups]

    # Shared axes allow for better comparison and cleaner look
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        shared_yaxes=True,  # Critical for comparison
        shared_xaxes=True,  # Reduces clutter
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # Color palette (Set1 match)
    colors = pcolors.qualitative.Set1
    color_map = {cond: colors[i % len(colors)] for i, cond in enumerate(unique_conditions)}

    # Aggregation
    agg_df = plot_df.groupby([cat_col1, cat_col2, x_plot_col, x_col])[y_col].agg(['mean', 'std']).reset_index()

    for i, group in enumerate(unique_groups):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        group_data = agg_df[agg_df[cat_col1] == group]

        for condition in unique_conditions:
            cond_data = group_data[group_data[cat_col2] == condition]
            if cond_data.empty:
                continue

            c = color_map[condition]

            # 1. Error Bars + Points
            fig.add_trace(
                go.Scatter(
                    x=cond_data[x_col],
                    y=cond_data['mean'],
                    error_y=dict(
                        type='data',
                        array=cond_data['std'],
                        visible=True,
                        thickness=1.5,
                        width=3,
                        color=c
                    ),
                    mode='markers',
                    marker=dict(
                        color=c,
                        size=8,
                        line=dict(width=1, color='white')  # Professional touch
                    ),
                    name=str(condition),
                    legendgroup=str(condition),
                    showlegend=(i == 0)  # Only show legend once
                ),
                row=row, col=col
            )

            # 2. Trend Line (Linear)
            if add_trend_line:
                raw_cond_data = plot_df[(plot_df[cat_col1] == group) & (plot_df[cat_col2] == condition)]

                if len(raw_cond_data) > 1:
                    x_nums = raw_cond_data[x_plot_col]
                    y_vals = raw_cond_data[y_col]

                    # Remove NaNs
                    valid_mask = ~np.isnan(x_nums) & ~np.isnan(y_vals)
                    if valid_mask.sum() > 1:
                        z = np.polyfit(x_nums[valid_mask], y_vals[valid_mask], 1)
                        p = np.poly1d(z)

                        # Generate trend line points coverage
                        x_range_nums = np.linspace(x_nums.min(), x_nums.max(), 50)

                        if is_datetime:
                            x_trend = [start_time + pd.Timedelta(seconds=val) for val in x_range_nums]
                        else:
                            x_trend = x_range_nums

                        y_trend = p(x_range_nums)

                        fig.add_trace(
                            go.Scatter(
                                x=x_trend,
                                y=y_trend,
                                mode='lines',
                                line=dict(color=c, width=2, dash='solid'),  # Solid or dash
                                name=f"{condition} Trend",
                                legendgroup=str(condition),
                                showlegend=False,
                                hoverinfo='skip',
                                opacity=0.8
                            ),
                            row=row, col=col
                        )

    # Professional Styling
    fig.update_layout(
        height=300 * n_rows,  # Compact height
        template="plotly_white",
        font=dict(family="Arial", size=12, color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,  # Place above plots to save vertical space? Or bottom?
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=20, t=60, b=50),  # Tighter margins
        # Uniform axes style
    )

    # Improve grids
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5', linecolor='black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E5E5', linecolor='black')

    return fig



def plot_grouped_data_with_optional_trend(df: pd.DataFrame,
                                          cat_col1: str ='Cell type',
                                          cat_col2: str ='Condition',
                                          y_col: str = 'Vessels Area Normalize',
                                          x_col: str = 'Timepoint_datetime',
                                          add_trend_line: bool = False,
                                          max_cols: int = 4,
                                          font_scale: float = 1.0,
                                          title: str = None,
                                          save_path: str = None):
    """
    Plots grouped data with matched colors and shows ALL x-axis ticks
    derived from the base column (removing '_datetime').
    """
    # 1. Apply Global Font Scaling
    sns.set(style="whitegrid", font_scale=font_scale)

    # Create a copy to avoid modifying the original dataframe
    plot_df = df.copy()

    # 2. Handle Datetime X-Axis (Convert to Numbers for Plotting)
    x_plot_col = 'x_numeric_seconds'

    # Determine the 'base' column name for labels (e.g. 'Timepoint' from 'Timepoint_datetime')
    # If x_col doesn't have '_datetime', it just uses the column name as is.
    base_label_col = x_col.split('_datetime')[0]

    # Convert Datetime to numeric for regression/plotting
    if pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
        start_time = plot_df[x_col].min()
        plot_df[x_plot_col] = (plot_df[x_col] - start_time).dt.total_seconds()
    else:
        # Fallback if not datetime (treat as raw numeric)
        plot_df[x_plot_col] = plot_df[x_col]

    # 3. Create Tick Mapping (Numeric X -> Base Column Label)
    if 'time' in x_col.lower() and pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
        # --- FIX: Create a mapping of UNIQUE pairs ---
        # We need exactly one label per unique numeric tick
        unique_map = plot_df[[x_plot_col, x_col]].drop_duplicates(subset=[x_plot_col]).sort_values(x_plot_col)

        unique_ticks = unique_map[x_plot_col].values
        # Format only the unique datetime values to HH:MM
        unique_labels = unique_map[x_col].dt.strftime('%H:%M').values
    else:
        unique_ticks = None
        unique_labels = None

    # 4. Aggregate Data (Mean + Std)
    agg_df = plot_df.groupby([cat_col1, cat_col2, x_plot_col])[y_col].agg(['mean', 'std']).reset_index()

    # Get unique groups (subplots) and conditions (colors)
    unique_groups = plot_df[cat_col1].unique()
    unique_conditions = plot_df[cat_col2].unique()

    # Define Palette
    palette = sns.color_palette("Set1", len(unique_conditions))
    color_map = dict(zip(unique_conditions, palette))

    # Calculate grid layout
    n_cols = min(len(unique_groups), max_cols)
    n_rows = (len(unique_groups) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 5))

    # Flatten axes
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # --- Plotting Loop ---
    for i, group in enumerate(unique_groups):
        ax = axes[i]

        # Filter data for this group
        group_agg = agg_df[agg_df[cat_col1] == group]
        group_raw = plot_df[plot_df[cat_col1] == group]

        for condition in unique_conditions:
            color = color_map[condition]

            # A. Plot Mean Points + Error Bars
            cond_agg = group_agg[group_agg[cat_col2] == condition]

            if not cond_agg.empty:
                ax.errorbar(
                    x=cond_agg[x_plot_col],
                    y=cond_agg['mean'],
                    yerr=cond_agg['std'],
                    fmt='o',
                    color=color,
                    ecolor=color,
                    capsize=3 * font_scale,
                    alpha=0.8,
                    label=condition
                )

            # B. Optional Trend Line
            if add_trend_line:
                cond_raw = group_raw[group_raw[cat_col2] == condition]
                if not cond_raw.empty:
                    sns.regplot(
                        data=cond_raw,
                        x=x_plot_col,
                        y=y_col,
                        scatter=False,
                        ax=ax,
                        color=color,
                        ci=None,
                        line_kws={'linewidth': 2 * font_scale}
                    )

        # Set titles and labels
        ax.set_title(f"{group}")
        ax.set_xlabel(base_label_col)
        ax.set_ylabel(y_col)

        # Apply Custom X-Ticks from Base Column
        if unique_ticks is not None:
            # FORCE all ticks to appear by setting them explicitly
            ax.set_xticks(unique_ticks)
            ax.set_xticklabels(unique_labels, rotation=90, ha='center')
            # I removed the MaxNLocator block here so it won't hide any ticks

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # --- Global Legend ---
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='lower center',
        ncol=len(unique_conditions),
        bbox_to_anchor=(0.5, 0.0)
    )

    # Add Main Title
    if title:
        fig.suptitle(title, fontsize=16 * font_scale, y=1.02)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

# if __name__ == "__main__":
#     path = Path(r'C:\Users\riccig01\OneDrive\Projects\MtSinai\Fanny\ArterialManager\data\sample_tab_output')
#
#     # Find the first file that starts with 'OD2.1' and ends with .xlsx
#     matches = list(path.glob('ODQ2.1*.xlsx'))
#     if not matches:
#         raise FileNotFoundError("No .xlsx files starting with 'OD2.1' found in the folder.")
#
#     # If there may be multiple, pick the most recently modified:
#     file_to_read = max(matches, key=lambda p: p.stat().st_mtime)
#
#     df = pd.read_excel(file_to_read)
#     cols = ['Cell type', 'Condition', 'Timepoint', 'Timepoint_datetime', 'Vessels Area Normalize']
#     df_plot = df[cols]
#
#     sns.set(style="whitegrid")
#
#     group_a = 'Cell type'
#     group_b = 'Condition'
#     x_axis = 'Timepoint_datetime'
#     y_axis = 'Vessels Area Normalize'
#
#
#     # # Call the function with your desired columns
#     plot_grouped_data_with_optional_trend(
#         df=df_plot,
#         cat_col1=group_b,       # Plots separate graphs for Condition
#         cat_col2=group_a,       # Plots separate colors for Cell Type
#         y_col=y_axis,
#         x_col=x_axis,
#         add_trend_line=True,
#         font_scale=1.2,         # <--- Change this float to resize fonts (e.g. 0.8 or 1.5)
#         title="Vessel Area Over Time",
#         save_path=None
#     )
#
#

