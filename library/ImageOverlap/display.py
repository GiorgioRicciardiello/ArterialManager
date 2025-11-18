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
from typing import Tuple, Optional
from pathlib import Path
from library.ImageOverlap.utils import _normalize01,_edges_to_rgba


# =====================================================================
# %% Original colors fluorescent
# =====================================================================
def plot_green_fluorescent(ax: Axes,
                           img_green_color: np.ndarray,
                           heatmap_intensity_gold_rgba: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """
    Overlay original green image with gold intensity heatmap.
    Displays in subplot and saves a full-resolution version preserving
    exact dimensions, colors, and alpha.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(img_green_color, alpha=0.9)
    ax.imshow(heatmap_intensity_gold_rgba)
    ax.set_title('Green Channel')
    ax.axis("off")

    # ---- save full-resolution image ----
    if output_dir and filename:
        h, w = img_green_color.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(img_green_color, alpha=0.9)
        ax_full.imshow(heatmap_intensity_gold_rgba)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_red_fluorescent(ax: Axes,
                         img_red_rgb: np.ndarray,
                         output_dir: Path = None,
                         filename: str = None) -> Axes:
    """
    Plot red fluorescent channel.
    Displays in subplot and saves a full-resolution version preserving
    native image dimensions and color fidelity.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(img_red_rgb)
    ax.set_title('Red Channel')
    ax.axis("off")

    # ---- save full-resolution image ----
    if output_dir and filename:
        h, w = img_red_rgb.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(img_red_rgb)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax



# =====================================================================
# %% Norm colors
# =====================================================================
def plot_green_norm(ax: Axes,
                    g_norm: np.ndarray,
                    output_dir: Path = None,
                    filename: str = None) -> Axes:
    """
    Plot normalized green channel in the provided subplot (ax),
    and optionally save a full-resolution copy preserving native dimensions.
    """
    # ----- draw inside subplot grid -----
    ax.imshow(g_norm, cmap="Greens")
    ax.set_title('Green Normalized')
    ax.axis("off")

    # ----- save full-resolution figure -----
    if output_dir and filename:
        h, w = g_norm.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(g_norm, cmap="Greens")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_red_norm(ax: Axes,
                  r_norm: np.ndarray,
                  output_dir: Path = None,
                  filename: str = None) -> Axes:
    """
    Plot normalized red channel.
    Displays in subplot grid and saves a full-resolution copy
    preserving native dimensions and color mapping.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(r_norm, cmap="Reds")
    ax.set_title('Red Normalized')
    ax.axis("off")

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = r_norm.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(r_norm, cmap="Reds")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax

# =====================================================================
# %% Binary mask
# =====================================================================
def plot_red_binary_mask(ax: Axes,
                         r_mask: np.ndarray,
                         output_dir: Path = None,
                         filename: str = None) -> Axes:
    """
    Display binary red mask.
    Shows in subplot grid and saves a full-resolution copy.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(r_mask, cmap="Reds")
    ax.set_title('Red Binary Mask')
    ax.axis("off")

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = r_mask.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(r_mask, cmap="Reds")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_green_binary_mask(ax: Axes,
                           g_mask: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """
    Display binary green mask.
    Shows in subplot grid and saves a full-resolution copy.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(g_mask, cmap="Greens")
    ax.set_title('Green Binary Mask')
    ax.axis("off")

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = g_mask.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(g_mask, cmap="Greens")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


# =====================================================================
# %% Skeleton
# =====================================================================
def plot_green_skeleton(ax: Axes,
                        g_skel: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """
    Display green skeleton in subplot and save full-resolution copy.
    Preserves native pixel dimensions and color mapping.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(g_skel, cmap="Greens")
    ax.axis("off")
    ax.set_title('Green Skeleton')

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = g_skel.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(g_skel, cmap="Greens")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_red_skeleton(ax: Axes,
                      r_skel: np.ndarray,
                      output_dir: Path = None,
                      filename: str = None) -> Axes:
    """
    Display red skeleton in subplot and save full-resolution copy.
    Preserves native pixel dimensions and color mapping.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(r_skel, cmap="Reds")
    ax.axis("off")
    ax.set_title('Red Skeleton')

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = r_skel.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(r_skel, cmap="Reds")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


# =====================================================================
# %% Adaptive threshold
# =====================================================================


# ---------------------------------------------------------------------
# Bernsen plots
# ---------------------------------------------------------------------
def plot_green_bernsen(ax: Axes,
                       mask_g_bernsen: np.ndarray,
                       output_dir: Path = None,
                       filename: str = None) -> Axes:
    """Display Bernsen green mask."""
    ax.imshow(mask_g_bernsen, cmap="Greens")
    ax.set_title("Green – Bernsen")
    ax.axis("off")

    if output_dir and filename:
        h, w = mask_g_bernsen.shape
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        axf = fig.add_axes([0, 0, 1, 1])
        axf.imshow(mask_g_bernsen, cmap="Greens")
        axf.axis("off")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_red_bernsen(ax: Axes,
                     mask_r_bernsen: np.ndarray,
                     output_dir: Path = None,
                     filename: str = None) -> Axes:
    """Display Bernsen red mask."""
    ax.imshow(mask_r_bernsen, cmap="Reds")
    ax.set_title("Red – Bernsen")
    ax.axis("off")

    if output_dir and filename:
        h, w = mask_r_bernsen.shape
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        axf = fig.add_axes([0, 0, 1, 1])
        axf.imshow(mask_r_bernsen, cmap="Reds")
        axf.axis("off")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_overlap_bernsen(ax: Axes,
                         mask_g_bernsen: np.ndarray,
                         mask_r_bernsen: np.ndarray,
                         output_dir: Path = None,
                         filename: str = None) -> Axes:
    """Display Bernsen composite overlap (yellow = overlap)."""
    rgb = np.zeros((*mask_g_bernsen.shape, 3), dtype=float)
    rgb[..., 1] = mask_g_bernsen
    rgb[..., 0] = mask_r_bernsen
    ax.imshow(rgb)
    ax.set_title("Composite Overlap – Bernsen")
    ax.axis("off")

    if output_dir and filename:
        h, w = mask_g_bernsen.shape
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        axf = fig.add_axes([0, 0, 1, 1])
        axf.imshow(rgb)
        axf.axis("off")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


# ---------------------------------------------------------------------
# Niblack plots
# ---------------------------------------------------------------------
def plot_green_niblack(ax: Axes,
                       mask_g_niblack: np.ndarray,
                       output_dir: Path = None,
                       filename: str = None) -> Axes:
    """Display Niblack green mask."""
    ax.imshow(mask_g_niblack, cmap="Greens")
    ax.set_title("Green – Niblack")
    ax.axis("off")
    if output_dir and filename:
        h, w = mask_g_niblack.shape
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        axf = fig.add_axes([0, 0, 1, 1])
        axf.imshow(mask_g_niblack, cmap="Greens")
        axf.axis("off")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_red_niblack(ax: Axes,
                     mask_r_niblack: np.ndarray,
                     output_dir: Path = None,
                     filename: str = None) -> Axes:
    """Display Niblack red mask."""
    ax.imshow(mask_r_niblack, cmap="Reds")
    ax.set_title("Red – Niblack")
    ax.axis("off")
    if output_dir and filename:
        h, w = mask_r_niblack.shape
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        axf = fig.add_axes([0, 0, 1, 1])
        axf.imshow(mask_r_niblack, cmap="Reds")
        axf.axis("off")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_overlap_niblack(ax: Axes,
                         mask_g_niblack: np.ndarray,
                         mask_r_niblack: np.ndarray,
                         output_dir: Path = None,
                         filename: str = None) -> Axes:
    """Display Niblack composite overlap (yellow = overlap)."""
    rgb = np.zeros((*mask_g_niblack.shape, 3), dtype=float)
    rgb[..., 1] = mask_g_niblack
    rgb[..., 0] = mask_r_niblack
    ax.imshow(rgb)
    ax.set_title("Composite Overlap – Niblack")
    ax.axis("off")
    if output_dir and filename:
        h, w = mask_g_niblack.shape
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        axf = fig.add_axes([0, 0, 1, 1])
        axf.imshow(rgb)
        axf.axis("off")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax



# =====================================================================
# %% Overlaps
# =====================================================================
def plot_overlap_binary(ax: Axes,
                        heatmap_binary: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """
    Show binary overlap map in subplot and save a full-resolution copy.
    Preserves the exact pixel dimensions and grayscale intensity.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(heatmap_binary, cmap="gray")
    ax.set_title('Binary Overlay')
    ax.axis("off")

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = heatmap_binary.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(heatmap_binary, cmap="gray")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_overlap_binary_edges(ax: Axes,
                              img_green_color: np.ndarray,
                              g_edges: np.ndarray,
                              heatmap_binary_gold_rgba: np.ndarray,
                              output_dir: Path = None,
                              filename: str = None) -> Axes:
    """
    Show binary overlap with edges and gold overlay.
    Displays in subplot and saves a full-resolution version preserving alpha and color.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(img_green_color, alpha=0.9)
    edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
    ax.imshow(edges_rgba)
    ax.imshow(heatmap_binary_gold_rgba)
    ax.axis("off")
    ax.set_title('Overlap Binary + Edges')

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = img_green_color.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(img_green_color, alpha=0.9)
        # ax_full.imshow(edges_rgba)  # hide edges in separate overlay
        ax_full.imshow(heatmap_binary_gold_rgba)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_green_red_overlay(ax: Axes,
                           g_norm: np.ndarray,
                           r_masked: np.ndarray,
                           green_u8: np.ndarray,
                           g_edges: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """
    Display overlay in the subplot (ax) and save a full-resolution copy
    rendered independently to preserve native dimensions.
    """
    # ----- render in the subplot (for grid display) -----
    ax.imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)
    ax.imshow(g_norm, cmap="Greens", alpha=0.9)
    ax.imshow(r_masked, cmap="Reds", alpha=0.6)
    edges_rgba = _edges_to_rgba(g_edges, hex_color="#00FF00", alpha=0.5)
    ax.imshow(edges_rgba)
    ax.axis("off")
    ax.set_title('Green + Red Overlay')


    # ----- optional save at full native resolution -----
    if output_dir and filename:
        h, w = green_u8.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)
        ax_full.imshow(g_norm, cmap="Greens", alpha=0.9)
        ax_full.imshow(r_masked, cmap="Reds", alpha=0.6)
        # ax_full.imshow(edges_rgba)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_overlap_vessel_mask(ax: Axes,
                             vessel_mask: np.ndarray,
                             output_dir: Path = None,
                             filename: str = None) -> Axes:
    """
    Show vessel mask overlay in subplot and save full-resolution copy.
    Preserves native pixel dimensions and grayscale fidelity.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(vessel_mask, cmap="gray")
    ax.axis("off")
    ax.set_title('Combined Vessel Mask')

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = vessel_mask.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(vessel_mask, cmap="gray")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_overlap_frangi(ax: Axes,
                        overlay: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """
    Display Frangi-enhanced RGB overlay in subplot and save a full-resolution version.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(overlay)
    ax.set_title('Overlay Frangi Enhanced')
    ax.axis("off")

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = overlay.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(overlay)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_overlay_green_red(ax: Axes,
                           green_u8: np.ndarray,
                           red_u8: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """
    Plot boosted green + red overlay on black background.
    Displays in subplot grid and saves a full-resolution copy.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)
    g_boost = np.clip(_normalize01(green_u8) * 1.2, 0, 1)
    ax.imshow(_normalize01(g_boost), cmap="Greens", alpha=0.9)
    ax.imshow(_normalize01(red_u8), cmap="Reds", alpha=0.6)
    ax.axis("off")

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = green_u8.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)
        ax_full.imshow(_normalize01(g_boost), cmap="Greens", alpha=0.9)
        ax_full.imshow(_normalize01(red_u8), cmap="Reds", alpha=0.6)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax


def plot_overlay_binary_edges(ax: Axes,
                              img_green_color: np.ndarray,
                              g_edges: np.ndarray,
                              heatmap_binary_gold_rgba: np.ndarray,
                              output_dir: Path = None,
                              filename: str = None) -> Axes:
    """
    Plot green image with magenta edges and gold binary overlap.
    Displays in subplot grid and saves a full-resolution copy preserving native dimensions and alpha.
    """
    # ---- draw inside subplot grid ----
    ax.imshow(img_green_color, alpha=0.9)
    edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
    ax.imshow(edges_rgba)
    ax.imshow(heatmap_binary_gold_rgba)
    ax.axis("off")

    # ---- save full-resolution version ----
    if output_dir and filename:
        h, w = img_green_color.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(img_green_color, alpha=0.9)
        ax_full.imshow(edges_rgba)
        ax_full.imshow(heatmap_binary_gold_rgba)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)

    return ax



# %%
def plot_intensity_overlap_norm(ax: Axes,
                                g_norm: np.ndarray,
                                heatmap_rgba: np.ndarray,
                                output_dir: Path = None,
                                filename: str = None) -> Axes:
    """Overlay normalized green image with gold intensity heatmap."""
    # ---- subplot display ----
    ax.imshow(g_norm, cmap="Greens", alpha=0.9)
    ax.imshow(heatmap_rgba)
    ax.axis("off")

    # ---- full-resolution save ----
    if output_dir and filename:
        h, w = g_norm.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(g_norm, cmap="Greens", alpha=0.9)
        ax_full.imshow(heatmap_rgba)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_intensity_heatmap(ax: Axes,
                           heatmap: np.ndarray,
                           output_dir: Path = None,
                           filename: str = None) -> Axes:
    """Display intensity heatmap without colorbar; saves full resolution."""
    ax.imshow(heatmap, cmap="viridis")
    ax.axis("off")

    if output_dir and filename:
        h, w = heatmap.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(heatmap, cmap="viridis")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_rgb_overlay(ax: Axes,
                     overlay: np.ndarray,
                     output_dir: Path = None,
                     filename: str = None) -> Axes:
    """Display RGB overlay image."""
    ax.imshow(overlay)
    ax.axis("off")

    if output_dir and filename:
        h, w = overlay.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(overlay)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_heatmap_plasma(ax: Axes,
                        heatmap: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """Display heatmap using plasma colormap."""
    ax.imshow(heatmap, cmap="plasma")
    ax.axis("off")

    if output_dir and filename:
        h, w = heatmap.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(heatmap, cmap="plasma")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_binary_overlap(ax: Axes,
                        binary_map: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """Display binary overlap mask."""
    ax.imshow(binary_map, cmap="gray")
    ax.axis("off")

    if output_dir and filename:
        h, w = binary_map.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(binary_map, cmap="gray")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_red_mask(ax: Axes,
                  r_mask: np.ndarray,
                  output_dir: Path = None,
                  filename: str = None) -> Axes:
    """Display binary red mask."""
    ax.imshow(r_mask, cmap="Reds")
    ax.axis("off")

    if output_dir and filename:
        h, w = r_mask.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(r_mask, cmap="Reds")
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax


def plot_binary_overlay(ax: Axes,
                        img_green_color: np.ndarray,
                        g_edges: np.ndarray,
                        heatmap_binary_gold_rgba: np.ndarray,
                        output_dir: Path = None,
                        filename: str = None) -> Axes:
    """Overlay green image, magenta edges, and gold binary overlap."""
    ax.imshow(img_green_color, alpha=0.9)
    edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
    ax.imshow(edges_rgba)
    ax.imshow(heatmap_binary_gold_rgba)
    ax.axis("off")

    if output_dir and filename:
        h, w = img_green_color.shape[:2]
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_full = fig.add_axes([0, 0, 1, 1])
        ax_full.imshow(img_green_color, alpha=0.9)
        ax_full.imshow(edges_rgba)
        ax_full.imshow(heatmap_binary_gold_rgba)
        ax_full.axis("off")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        cv2.imwrite(str(Path(output_dir) / filename),
                    cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA))
        plt.close(fig)
    return ax
