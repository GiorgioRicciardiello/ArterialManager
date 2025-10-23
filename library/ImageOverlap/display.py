"""
Make the plots as separate functions

The plot function will return the ax so we can use them in subplots and they will also save
the separate subplot as an independent image. This is useful for the upcoming pipelines.


"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cv2
import matplotlib.colors as mcolors
from pathlib import Path
from library.ImageOverlap.utils import _normalize01,_edges_to_rgba

# %% Helpers

def _save_axis(ax: Axes,
               output_dir: Path = None,
               filename: str = None,
               dpi: int = 300) -> None:
    """Save only the area of a given axis to an image file."""
    if output_dir is None or filename is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = ax.figure
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(output_dir / filename, bbox_inches=extent, dpi=dpi)

# =====================================================================
# %% Original colors fluorescent
# =====================================================================
def plot_green_fluorescent(ax: Axes,
                           img_green_color: np.ndarray,
                           heatmap_intensity_gold_rgba: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """Overlay original green image with gold intensity heatmap."""
    ax.imshow(img_green_color, alpha=0.9)
    ax.imshow(heatmap_intensity_gold_rgba)
    ax.set_title("Green Channel + Gold Overlap", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


def plot_red_fluorescent(ax: Axes,
                         img_red_rgb: np.ndarray,
                         output_dir: Path = None,
                         filename: str = None) -> Axes:
    """Plot red fluorescent channel."""
    ax.imshow(img_red_rgb)
    ax.set_title("Red Channel (Fluorescent)", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


# =====================================================================
# %% Norm colors
# =====================================================================
def plot_green_norm(ax: Axes,
                    g_norm: np.ndarray,
                    output_dir: Path = None,
                    filename: str = None) -> Axes:
    """Plot normalized green channel."""
    ax.imshow(g_norm, cmap="Greens")
    ax.set_title("Green (Normalized)", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


def plot_red_norm(ax: Axes,
                  r_norm: np.ndarray,
                  output_dir: Path = None,
                  filename: str = None) -> Axes:
    """Plot normalized red channel."""
    ax.imshow(r_norm, cmap="Reds")
    ax.set_title("Red (Normalized)", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


# =====================================================================
# %% Binary mask
# =====================================================================
def plot_red_binary_mask(ax: Axes,
                         r_mask: np.ndarray,
                         output_dir: Path = None,
                         filename: str = None) -> Axes:
    """Display binary red mask."""
    ax.imshow(r_mask, cmap="Reds")
    ax.set_title("Red Binary Mask", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


def plot_green_binary_mask(ax: Axes,
                           g_mask: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """Display binary green mask."""
    ax.imshow(g_mask, cmap="Greens")
    ax.set_title("Green Binary Mask", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


# =====================================================================
# %% Skeleton
# =====================================================================
def plot_green_skeleton(ax: Axes,
                        g_skel: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """Display green skeleton."""
    ax.imshow(g_skel, cmap="Greens")
    ax.set_title("Green Skeleton", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


def plot_red_skeleton(ax: Axes,
                      r_skel: np.ndarray,
                      output_dir: Path = None,
                      filename: str = None) -> Axes:
    """Display red skeleton."""
    ax.imshow(r_skel, cmap="Reds")
    ax.set_title("Red Skeleton", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


# =====================================================================
# %% Overlaps
# =====================================================================
def plot_overlap_binary(ax: Axes,
                        heatmap_binary: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """Show binary overlap map."""
    ax.imshow(heatmap_binary, cmap="gray")
    ax.set_title("Binary Overlap", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


def plot_overlap_binary_edges(ax: Axes,
                              img_green_color: np.ndarray,
                              g_edges: np.ndarray,
                              heatmap_binary_gold_rgba: np.ndarray,
                              output_dir: Path = None,
                              filename: str = None) -> Axes:
    """Show binary overlap with edges and gold overlay."""
    ax.imshow(img_green_color, alpha=0.9)
    edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
    ax.imshow(edges_rgba)
    ax.imshow(heatmap_binary_gold_rgba)
    ax.set_title("Overlap Binary + Edges", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


def plot_green_red_overlay(ax: Axes,
                           g_norm: np.ndarray,
                           r_masked: np.ndarray,
                           green_u8: np.ndarray,
                           g_edges: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """Overlay green and red channels with edge highlights."""
    ax.imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)
    ax.imshow(g_norm, cmap="Greens", alpha=0.9)
    ax.imshow(r_masked, cmap="Reds", alpha=0.6)
    edges_rgba = _edges_to_rgba(g_edges, hex_color="#00FF00", alpha=0.5)
    ax.imshow(edges_rgba)
    ax.set_title("Green + Red Overlay", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


def plot_overlap_vessel_mask(ax: Axes,
                             vessel_mask: np.ndarray,
                             output_dir: Path = None,
                             filename: str = None) -> Axes:
    """Show vessel mask overlay."""
    ax.imshow(vessel_mask, cmap="gray")
    ax.set_title("Combined Vessel Mask", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax


def plot_overlap_frangi(ax: Axes,
                        overlay: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """Display Frangi-enhanced RGB overlay."""
    ax.imshow(overlay)
    ax.set_title("RGB Overlay (Frangi Enhanced)", fontsize=12)
    ax.axis("off")
    _save_axis(ax, output_dir, filename)
    return ax



def plot_overlay_green_red(ax: Axes,
                           green_u8: np.ndarray,
                           red_u8: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """
    Plot boosted green + red overlay on black background and optionally save the axis.
    """
    ax.imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)
    g_boost = np.clip(_normalize01(green_u8) * 1.2, 0, 1)
    ax.imshow(_normalize01(g_boost), cmap="Greens", alpha=0.9)
    ax.imshow(_normalize01(red_u8), cmap="Reds", alpha=0.6)
    ax.axis("off")
    plt.tight_layout()
    _save_axis(ax, output_dir, filename)
    return ax


def plot_overlay_binary_edges(ax: Axes,
                              img_green_color: np.ndarray,
                              g_edges: np.ndarray,
                              heatmap_binary_gold_rgba: np.ndarray,
                              output_dir: Path = None,
                              filename: str = None) -> Axes:
    """
    Plot green image with magenta edges and gold binary overlap, optionally save the axis.
    """
    ax.imshow(img_green_color, alpha=0.9)
    edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
    ax.imshow(edges_rgba)
    ax.imshow(heatmap_binary_gold_rgba)
    ax.set_title("Overlap Binary", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    _save_axis(ax, output_dir, filename)
    return ax


# %%
def plot_intensity_overlap_norm(ax: Axes, g_norm: np.ndarray,
                                heatmap_rgba: np.ndarray) -> Axes:
    """Overlay normalized green image with gold intensity heatmap."""
    ax.imshow(g_norm, cmap="Greens", alpha=0.9)
    ax.imshow(heatmap_rgba)
    ax.set_title("Intensity Overlap: Green Normalized", fontsize=12)
    ax.axis("off")
    return ax


def plot_intensity_heatmap(ax: Axes, heatmap: np.ndarray) -> Axes:
    """Display intensity heatmap with colorbar."""
    im = ax.imshow(heatmap, cmap="viridis")
    ax.set_title("Intensity Overlap: Heatmap", fontsize=12)
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Overlap Intensity", fontsize=10)
    return ax


def plot_rgb_overlay(ax: Axes, overlay: np.ndarray) -> Axes:
    """Display RGB overlay image."""
    ax.imshow(overlay)
    ax.set_title("RGB Overlay")
    ax.axis("off")
    return ax



def plot_heatmap_plasma(ax: Axes, heatmap: np.ndarray) -> Axes:
    """Display heatmap using plasma colormap."""
    ax.imshow(heatmap, cmap="plasma")
    ax.set_title("Intensity Heatmap")
    ax.axis("off")
    return ax


def plot_binary_overlap(ax: Axes, binary_map: np.ndarray) -> Axes:
    """Display binary overlap mask."""
    ax.imshow(binary_map, cmap="gray")
    ax.set_title("Binary Overlap")
    ax.axis("off")
    return ax



def plot_red_mask(ax: Axes, r_mask: np.ndarray) -> Axes:
    """Display binary red mask."""
    ax.imshow(r_mask, cmap="Reds")
    ax.set_title("Red Binary Mask", fontsize=12)
    ax.axis("off")
    return ax


def plot_binary_overlay(ax: Axes, img_green_color: np.ndarray,
                        g_edges: np.ndarray, heatmap_binary_gold_rgba: np.ndarray) -> Axes:
    """Overlay green image, magenta edges, and gold binary overlap."""
    ax.imshow(img_green_color, alpha=0.9)
    edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
    ax.imshow(edges_rgba)
    ax.imshow(heatmap_binary_gold_rgba)
    ax.set_title("Overlap Binary", fontsize=12)
    ax.axis("off")
    return ax

