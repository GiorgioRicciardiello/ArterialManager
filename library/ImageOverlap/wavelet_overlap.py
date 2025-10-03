"""
=============================================================
Intensity-Weighted Colocalization of Red and Green Vessel Networks
=============================================================

This module provides a pipeline for analyzing colocalization between
two vascular fluorescent channels (green and red). It combines
wavelet-based background removal, vessel enhancement, and
structural + intensity-based overlap quantification, and produces
both visualization panels and quantitative metrics.

Overview
--------
1. **Preprocessing (per channel):**
   - Wavelet-based background and noise subtraction (WBNS).
   - Vessel enhancement using Frangi vesselness filtering.
   - Thresholding and skeletonization of vessel structures.

2. **Overlap detection:**
   - *Structural overlap*: skeleton dilation and intersection.
   - *Intensity-based overlap*: overlap strength defined by product,
     minimum, geometric mean, or Pearson-weighted combination of
     vessel intensities.

3. **Metrics:**
   - Manders’ coefficients (M1, M2) → fractional channel colocalization.
   - Pearson correlation restricted to overlap regions.
   - Dice and Jaccard indices computed from vessel skeletons.

4. **Outputs:**
   - RGB overlays and gold-highlighted overlap maps.
   - Heatmaps of intensity-weighted and binary overlap.
   - Quantitative metrics saved as CSV/Excel files.
   - Serialized component data (pickle.gz) for reproducibility.

Input
-----
- `path_green` : Path to green fluorescence channel image (C=0).
- `path_red`   : Path to red fluorescence channel image (C=1).
- Supported formats: JPG, PNG, TIFF (grayscale or RGB).

Output
------
- Visualization panels (per cell):
    - Normalized green and red channels
    - RGB overlay
    - Intensity-based overlap maps (heatmaps, gold highlights)
    - Binary overlap masks
- Metrics tables (`*_metrics.xlsx` per cell, global CSV if batched).
- Compressed pickle files with raw, processed, and derived components.

Usage
-----
Example:

    metrics, components = coloc_vessels_with_wbns(
        path_green="AD1_1_C=0.jpg",
        path_red="AD1_1_C=1.jpg",
        path_output=Path("./results/AD1_1"),
        cell_id="AD1_1",
        resolution_px=4,
        noise_lvl=1,
        skeleton_radius_px=3,
        intensity_overlap="geo_mean",
        show=True
    )

Notes
-----
- Assumes each cell has exactly two input images: one green (C=0) and one red (C=1).
- Background subtraction is tuned to remove uneven illumination and speckle
  while preserving vascular mid-frequency structures.
- Multiple overlap strategies are implemented to support different biological
  hypotheses (strict product, permissive min, balanced geo_mean, correlation-weighted).
- Visualization overlays highlight overlap in **yellow/gold** for interpretability.

"""

import pathlib
import pandas as pd
import pickle
import gzip
from config.config import CONFIG
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology
from skimage.util import img_as_float
from skimage.filters import frangi
from scipy.ndimage import gaussian_filter
from pywt import wavedecn, waverecn
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colors as mcolors
from typing import Dict, Optional, List, Tuple
from joblib import Parallel, delayed


# -----------------------------
# Helper functions
# -----------------------------
def _normalize01(x):
    x = x.astype(np.float32)
    mn, mx = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip((x - mn) / max(mx - mn, 1e-6), 0, 1)
    return x


def _wavelet_bg_subtraction(image, resolution_px=4, noise_lvl=1):
    """

    Wavelet-based background & noise subtraction (2D or 3D stack).

    The image is decomposed into low-frequency (background) and high-frequency (details/noise) components

    Low levels capture smooth background → uneven illumination, autofluorescence, out-of-focus haze.

    High levels capture finer structures → vessel edges, textures

    Wavelet-based background and noise subtraction.

    This function removes uneven illumination, autofluorescence background,
    and high-frequency speckle noise from microscopy images while preserving
    mid-frequency structures such as vessels.

    The algorithm uses a multilevel wavelet decomposition (db1) to separate
    image content into different frequency bands:

    - Low-frequency (approximation) → smooth background and illumination bias
    - High-frequency (detail) → fine-scale speckle noise
    - Mid-frequency → structures of interest (e.g., vessels)

    Workflow:
    1. Decompose the image with wavelets (wavedecn).
    2. Estimate background by reconstructing coarse levels and applying Gaussian smoothing.
    3. Estimate noise by reconstructing only the fine detail levels (controlled by `noise_lvl`).
    4. Subtract background and noise estimates from the raw image.
    5. Clip to non-negative values.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (2D or 3D stack).
    resolution_px : int, optional
        Approximate FWHM of the point spread function (PSF) in pixels.
        Controls how many wavelet levels are treated as background.
    noise_lvl : int, optional
        Number of wavelet detail levels to treat as noise. Higher values
        remove more fine-scale structure.

    Returns
    -------
    result : np.ndarray (float32)
        Denoised and background-corrected image suitable for vessel
        enhancement and colocalization analysis.
    """

    num_levels = int(np.ceil(np.log2(resolution_px)))

    coeffs = wavedecn(image, 'db1', level=None)
    coeffs2 = coeffs.copy()

    # Remove background (low-frequency details)
    # This reconstructs only the coarse approximation → smooth “background estimate”.
    for BGlvl in range(1, num_levels):
        coeffs[-BGlvl] = {k: np.zeros_like(v) for k, v in coeffs[-BGlvl].items()}
    Background = waverecn(coeffs, 'db1')
    Background = gaussian_filter(Background, sigma=2 ** num_levels)

    # Estimate noise
    # Keeps only fine detail coefficients → “noise estimate”.
    coeffs2[0] = np.ones_like(coeffs2[0])
    for lvl in range(1, len(coeffs2) - noise_lvl):
        coeffs2[lvl] = {k: np.zeros_like(v) for k, v in coeffs2[lvl].items()}
    Noise = waverecn(coeffs2, 'db1')

    # Subtract background & noise
    # Removes uneven background and high-frequency speckle noise.
    # Preserves mid-frequency structures, i.e. vessels.
    result = image - Background - Noise
    result[result < 0] = 0

    return result.astype(np.float32)


def _vessel_enhance(gray_u8):
    """CLAHE + Frangi vesselness + normalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray_u8)
    g = cv2.GaussianBlur(g, (0, 0), 0.8)

    v = frangi(img_as_float(g),
               sigmas=np.linspace(1, 3, 5),
               black_ridges=False)
    v = _normalize01(v)
    return (v * 255).astype(np.uint8), v


def _skeletonize_from_vesselness(v_float):
    """Threshold vesselness -> clean -> skeleton."""
    t = filters.threshold_otsu(gaussian_filter(v_float, 1.0))
    bw = (v_float >= t).astype(np.uint8)
    bw = morphology.remove_small_objects(bw.astype(bool), 30).astype(np.uint8)
    skel = morphology.skeletonize(bw > 0).astype(np.uint8)
    return bw, skel


def _manders_coeffs(g, r, overlap_mask):
    g_sum = np.sum(g)
    r_sum = np.sum(r)
    m1 = float(np.sum(r * overlap_mask) / (r_sum + 1e-8))
    m2 = float(np.sum(g * overlap_mask) / (g_sum + 1e-8))
    return m1, m2

def _save_subplots_individually(fig, ax_grid, titles, output_dir, dpi=300):
    """
    Save each subplot exactly as rendered in the figure.

    Parameters
    ----------
    fig : matplotlib Figure
        The parent figure object.
    ax_grid : 2D array-like of matplotlib Axes
        Grid of subplot axes.
    titles : list of str
        Filenames (without extension) for each subplot.
    output_dir : Path or str
        Directory to save the images.
    dpi : int
        Resolution in dots per inch.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, ax_row in enumerate(ax_grid):
        for j, ax in enumerate(ax_row):
            idx = i * len(ax_row) + j
            title = titles[idx] if idx < len(titles) else f"subplot_{i}_{j}"

            # Save only this Axes using its bounding box
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(output_dir / f"{title}.png",
                        dpi=dpi,
                        bbox_inches=extent,
                        pad_inches=0)


def _edges_to_rgba(edge_mask, hex_color="#00FF00", alpha=0.8):
    """
    Convert a binary edge mask to an RGBA overlay with given hex color.
    Neon green (#00FF00) → high contrast vs black, but blends with vessel body if vessels are already green.
    Cyan / turquoise (#00FFFF) → works very well against green vessels and gold overlap.
    White (#FFFFFF) → super clear, but can be too harsh.
    Magenta (#FF00FF) → great for biomedical overlays, stands out strongly.
    Orange (#FFA500) → warm contrast if you want edges distinct from gold blobs.
    """
    # Convert hex → (R,G,B)
    r, g, b = mcolors.to_rgb(hex_color)

    rgba = np.zeros((*edge_mask.shape, 4))
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = edge_mask.astype(float) * alpha
    return rgba


def _compute_overlap_metrics(G:np.ndarray,
                             R:np.ndarray,
                             g_skel:np.ndarray,
                             r_skel:np.ndarray,
                             mask_overlap:np.ndarray,
                             vessel_mask:np.ndarray,
                             label:str="intensity"):
    """
    Compute Manders, Pearson, Dice, Jaccard for given overlap mask.

    Parameters
    ----------
    G, R : np.ndarray
        Green and red vessel intensity images (uint8).
    g_skel, r_skel : np.ndarray
        Skeletonized vessel masks (binary).
    mask_overlap : np.ndarray
        Overlap mask (float32 or bool).
    vessel_mask : np.ndarray
        Union of vessel regions (bool).
    label : str
        Prefix for metric names (e.g. "intensity", "binary").

    Returns
    -------
    dict : {metric_name: value}
    """

    # Manders coefficients
    m1, m2 = _manders_coeffs(G.astype(np.float32),
                             R.astype(np.float32),
                             mask_overlap.astype(np.float32))

    # Pearson restricted to overlap region
    pearson = (np.corrcoef(G[mask_overlap > 0].ravel(),
                           R[mask_overlap > 0].ravel())[0, 1]
               if mask_overlap.sum() > 10 else np.nan)

    # Skeleton Dice & Jaccard (always the same for both overlap types, but keep for clarity)
    inter = np.logical_and(g_skel, r_skel).sum()
    union = np.logical_or(g_skel, r_skel).sum() + 1e-8
    dice = 2 * inter / (g_skel.sum() + r_skel.sum() + 1e-8)
    jaccard = inter / union

    return {
        f"Manders_M1_{label}": m1,
        f"Manders_M2_{label}": m2,
        f"Pearson_{label}": float(pearson),
        f"Dice_skeletons_{label}": dice,
        f"Jaccard_skeletons_{label}": jaccard
    }

# -----------------------------
# Main unified function
# -----------------------------
def coloc_vessels_with_wbns(path_green:pathlib.Path,
                            path_red:pathlib.Path,
                            path_output:pathlib.Path,
                            cell_id:Optional[str] = 'Cell',
                            resolution_px:Optional[int]=4,
                            noise_lvl=1,
                            skeleton_radius_px=2,
                            intensity_overlap:str='geo_mean',
                            output_path:pathlib.Path=None,
                            show=True):
    """
    Vessel colocalization with WBNS preprocessing.

    Parameters
    ----------
    path_green, path_red : pathlib.Path
        Path of the green vessel and the red vessel.
    resolution_px : int
        Resolution parameter for WBNS (FWHM in px).
    noise_lvl : int
        Noise suppression level (WBNS).
    skeleton_radius_px : int
        Tolerance for calling skeletons overlapping.
    intensity_overlap : str
        Which intensity overlap should be applied. Options are: 'product', 'minimum', 'geo_mean', 'pearson_weighted'
    show : bool
        If True, show overlay + heatmap_intensity.

    Returns
    -------
    overlay : RGB image (uint8)
    heatmap_intensity : float32 in [0,1]
    masks : dict of binary masks
    metrics : dict of colocalization scores
    """
    # -- read image in the different formats we need
    img_green_color = cv2.imread(str(path_green), cv2.IMREAD_COLOR)
    img_red_color = cv2.imread(str(path_red), cv2.IMREAD_COLOR)

    # Raw grayscale images (uint8).
    green_u8 = cv2.imread(str(path_green), cv2.IMREAD_GRAYSCALE)
    red_u8 = cv2.imread(str(path_red), cv2.IMREAD_GRAYSCALE)

    # normalize
    g_norm = _normalize01(green_u8)
    r_norm = _normalize01(red_u8)

    # masks -> show red only where green
    mask_overlap = g_norm > 0
    r_masked = np.zeros_like(r_norm)
    r_masked[mask_overlap] = r_norm[mask_overlap]

    # --- Step 1: WBNS background/noise subtraction ---
    g_clean = _wavelet_bg_subtraction(green_u8, resolution_px, noise_lvl)
    r_clean = _wavelet_bg_subtraction(red_u8, resolution_px, noise_lvl)

    # --- Step 2: Vessel enhancement (Frangi) ---
    g_enh_u8, g_v = _vessel_enhance((g_clean * 255 / _normalize01(g_clean).max()).astype(np.uint8))
    r_enh_u8, r_v = _vessel_enhance((r_clean * 255 / _normalize01(r_clean).max()).astype(np.uint8))

    # --- Step 3: Skeletons ---
    g_bw, g_skel = _skeletonize_from_vesselness(g_v)
    r_bw, r_skel = _skeletonize_from_vesselness(r_v)
    vessel_mask = (g_bw | r_bw).astype(bool)

    # --- Step 4: Overlap detection ---
    # Structural overlap: skeleton proximity
    # define the filter
    # se = morphology.disk(skeleton_radius_px)
    se = morphology.disk(1)  # small structuring element

    r_dil = morphology.binary_dilation(r_skel, se)
    g_dil = morphology.binary_dilation(g_skel, se)
    near_overlap = (g_skel & r_dil) | (r_skel & g_dil)

    # Structural detection -> mask and edges
    g_mask = (green_u8 > 0).astype(np.uint8)  # green vessel presence
    r_mask = (red_u8 > 0).astype(np.uint8)  # red contrast presence

    # Edges with High-pass filter (extract edges from g_mask)  or Sobel for strong edges
    g_edges = cv2.Laplacian(g_mask.astype(np.uint8), cv2.CV_64F)
    g_edges = np.abs(g_edges)
    g_edges = (g_edges > 0).astype(np.uint8)  # binarize edges

    r_edges = cv2.Laplacian(r_mask.astype(np.uint8), cv2.CV_64F)
    r_edges = np.abs(r_edges)
    r_edges = (r_edges > 0).astype(np.uint8)  # binarize edges

    # --- Intensity-based overlap ---
    if intensity_overlap == 'product':
        # Method 1: product
        overlap_strength = g_v * r_v

    elif intensity_overlap == 'minimum':
        # Method 2: minimum
        overlap_strength = np.minimum(g_v, r_v)

    elif intensity_overlap == 'geo_mean':
        # Method 3: geometric mean
        overlap_strength = np.sqrt(g_v * r_v)

    elif intensity_overlap == 'pearson_weighted':
        # Method 4: Pearson-weighted overlap
        mean_g = g_v[g_mask].mean() if g_mask.sum() > 0 else g_v.mean()
        mean_r = r_v[r_mask].mean() if r_mask.sum() > 0 else r_v.mean()
        overlap_strength = (g_v - mean_g) * (r_v - mean_r)
        overlap_strength = np.clip(overlap_strength, 0, None)  # keep only positive correlations

    elif intensity_overlap == 'adaptive_weighted':
        # Method 5: Adaptive weighting with vessel skeletons
        base_strength = g_v * r_v
        overlap_strength = _normalize01(base_strength) * (0.5 + 0.5 * near_overlap.astype(np.float32))

    else:
        raise ValueError(f"Unknown intensity_overlap method: {intensity_overlap}")

    # --- Final normalized heatmap (applies skeleton constraint if desired) ---
    heatmap_intensity = _normalize01(overlap_strength) * near_overlap.astype(np.float32)

    heatmap_intensity_gold_rgba = np.zeros((*heatmap_intensity.shape, 4))
    heatmap_intensity_gold_rgba[..., 0] = 1.0
    heatmap_intensity_gold_rgba[..., 1] = 0.84
    heatmap_intensity_gold_rgba[..., 2] = 0.0
    heatmap_intensity_gold_rgba[..., 3] = heatmap_intensity  # directly scale alpha by overlap strength

    # heatmap_binary = np.logical_and(g_mask, r_mask).astype(np.float32)
    heatmap_binary = (g_mask > 0) & (r_mask > 0)

    # gold RGBA overlay (use mask as alpha channel)
    heatmap_binary_gold_rgba = np.zeros((*heatmap_binary.shape, 4))
    heatmap_binary_gold_rgba[..., 0] = 1.0  # Red channel
    heatmap_binary_gold_rgba[..., 1] = 0.84  # Green channel
    heatmap_binary_gold_rgba[..., 2] = 0.0  # Blue channel (RGB ~ gold)
    heatmap_binary_gold_rgba[..., 3] = heatmap_binary.astype(float) * 0.8  # alpha for overlap


    # --- Step 5: Overlay ---
    # normalize vessel channels _v are Frangi-enhanced vesselness maps
    # Multiplying by 255 converts to 8-bit intensity image
    G = (255 * _normalize01(g_v)).astype(np.uint8)
    R = (255 * _normalize01(r_v)).astype(np.uint8)
    Y = (255 * _normalize01(heatmap_intensity)).astype(np.uint8)

    overlay = np.zeros((*G.shape, 3), dtype=np.uint8)  # blank RGB image
    overlay[..., 1] = G  # green channel vesselness map
    overlay[..., 2] = R  # red channel vesselness map
    boost = (Y > 0)  # intensity overlap map - composite green–red image.
    # Finds pixels where overlap intensity exists
    overlay[..., 1][boost] = np.clip(overlay[..., 1][boost] + Y[boost] // 2, 0, 255)
    overlay[..., 2][boost] = np.clip(overlay[..., 2][boost] + Y[boost] // 2, 0, 255)
    # where overlap is strong, both red and green brighten → visually combining into yellow/orange highlights.

    # --- Step 6: Metrics ---
    metrics_int = _compute_overlap_metrics(G=G, R=R,
                                           g_skel=g_skel,
                                           r_skel=r_skel,
                                           mask_overlap=heatmap_intensity,
                                           vessel_mask=vessel_mask,
                                           label="intensity")

    metrics_bin = _compute_overlap_metrics(G=G, R=R,
                                           g_skel=g_skel,
                                           r_skel=r_skel,
                                           mask_overlap=heatmap_binary,
                                           vessel_mask=vessel_mask,
                                           label="binary")

    # merge both dictionaries
    metrics = {**metrics_int, **metrics_bin}

    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    # Extract method and clean up metric names
    df_metrics["method"] = df_metrics["Metric"].apply(
        lambda x: "intensity" if x.endswith("_intensity") else "binary"
    )

    df_metrics["Metric"] = df_metrics["Metric"].str.replace("_intensity", "", regex=False)
    df_metrics["Metric"] = df_metrics["Metric"].str.replace("_binary", "", regex=False)


    df_metrics.to_excel(path_output.joinpath(f'{cell_id}_metrics.xlsx'), index=False)
    # --- Step 7: Store the data ---
    components = {
        "green": {
            "raw": {
                "intensity": G.astype(np.uint8),  # vesselness map
                "intensity_norm": _normalize01(G).astype(np.float32),
                "mask": g_mask.astype(np.uint8)  # binary vessel presence
            },
            "derived": {
                "skeleton": g_skel.astype(np.uint8),  # vessel skeleton
                "edges": g_edges.astype(np.uint8)  # vessel edges
            }
        },

        "red": {
            "raw": {
                "intensity": R.astype(np.uint8),
                "intensity_norm": _normalize01(R).astype(np.float32),
                "mask": r_mask.astype(np.uint8)
            },
            "derived": {
                "skeleton": r_skel.astype(np.uint8),
                "edges": r_edges.astype(np.uint8)
            }
        },

        "overlap": {
            "raw": {
                "intensity": heatmap_intensity.astype(np.float32),
                "binary": heatmap_binary.astype(np.uint8),
                "vessel_mask": vessel_mask.astype(np.uint8)
            },
            "derived": {
                "overlay": overlay.astype(np.uint8),  # RGB composite
                "near_overlap": near_overlap.astype(np.uint8),  # skeleton proximity overlap
                "r_masked": r_masked.astype(np.uint8),  # red restricted to green
                "gold_intensity": heatmap_intensity_gold_rgba,  # RGBA gold overlay (float)
                "gold_binary": heatmap_binary_gold_rgba  # RGBA gold overlay (binary)
            }
        },

        "meta": {
            "intensity_overlap": intensity_overlap,
            "resolution_px": resolution_px,
            "noise_lvl": noise_lvl,
            "skeleton_radius_px": skeleton_radius_px
        }
    }
    if path_output:
        with gzip.open(path_output.joinpath(f"{cell_id}_components.pkl.gz"), "wb") as f:
            pickle.dump(components, f)

    # --- Step 8: Visualization ---
    if path_output or show:
        # Normalize images for display
        h_norm = _normalize01(heatmap_intensity)
        h_bin = _normalize01(heatmap_binary)

        fig, ax = plt.subplots(4, 3, figsize=(18, 15))
        # ---------------- Row 1: raw images ----------------
        # - Raw green (on black background)
        ax[0, 0].imshow(g_norm , cmap="Greens")
        ax[0, 0].set_title("Green (Normalized)", fontsize=12)
        ax[0, 0].axis("off")

        # - Raw red (on black background)
        ax[0, 1].imshow(r_norm, cmap="Reds")
        ax[0, 1].set_title("Red (Normalized)", fontsize=12)
        ax[0, 1].axis("off")

        # - Green + Red overlay (black background)
        ax[0, 2].imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)  # gray black canvas
        # ax[0, 2].set_facecolor("black")
        ax[0, 2].imshow(g_norm , cmap="Greens", alpha=0.9)
        ax[0, 2].imshow(r_masked, cmap="Reds", alpha=0.6)
        edges_rgba = _edges_to_rgba(g_edges, hex_color="#00FF00", alpha=0.5)
        ax[0, 2].imshow(edges_rgba)
        ax[0, 2].set_title("Green + Red overlay", fontsize=12)
        ax[0, 2].axis("off")

        # ---------------- Row 2: Intensity-based overlap----------------
        # 1. Vessels + gold overlap
        ax[1, 0].imshow(img_green_color, alpha=0.9)
        ax[1, 0].imshow(heatmap_intensity_gold_rgba)
        ax[1, 0].set_title("Intensity Overlap: Green Original", fontsize=12)
        ax[1, 0].axis("off")

        # 2. Green vessels + gold highlights (same gold RGBA overlay)
        ax[1, 1].imshow(g_norm, cmap="Greens", alpha=0.9)
        ax[1, 1].imshow(heatmap_intensity_gold_rgba)
        ax[1, 1].set_title("Intensity Overlap: Green Normalized", fontsize=12)
        ax[1, 1].axis("off")

        # 3. Heatmap view
        im = ax[1, 2].imshow(heatmap_intensity, cmap="viridis")  # or "inferno"/"cividis"
        ax[1, 2].set_title("Intensity Overlap: Heatmap", fontsize=12)
        ax[1, 2].axis("off")
        cbar = plt.colorbar(im, ax=ax[1, 2], fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Overlap Intensity", fontsize=10)

        # ---------------- Row 3: Overlay ----------------

        ax[2, 0].imshow(overlay)
        ax[2, 0].set_title("RGB Overlay")
        ax[2, 0].axis("off")
        ax[2, 1].imshow(heatmap_intensity, cmap="plasma")
        ax[2, 1].set_title("Intensity Heatmap")
        ax[2, 1].axis("off")
        ax[2, 2].imshow(heatmap_binary, cmap="gray")
        ax[2, 2].set_title("Binary Overlap")
        ax[2, 2].axis("off")

        # ---------------- Row 4: Binary overlap ----------------
        # Green mask
        ax[3, 0].imshow(g_mask, cmap="Greens")
        ax[3, 0].set_title("Green Binary Mask", fontsize=12)
        ax[3, 0].axis("off")

        # Red mask
        ax[3, 1].imshow(r_mask, cmap="Reds")
        ax[3, 1].set_title("Red Binary Mask", fontsize=12)
        ax[3, 1].axis("off")

        # Heatmap + green edges + gold overlap
        # ax[2, 2].imshow(h_bin, cmap="inferno")
        # ax[2, 2].imshow(g_edges, cmap="cool", alpha=0.7)
        # ax[2, 2].imshow(heatmap_binary_gold_rgba)  # gold where overlap occurs
        # ax[2, 2].set_title("Binary Heatmap + Edges + Gold Overlap", fontsize=12)
        # ax[2, 2].axis("off")

        ax[3, 2].imshow(img_green_color, alpha=0.9)
        edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
        ax[3, 2].imshow(edges_rgba)
        ax[3, 2].imshow(heatmap_binary_gold_rgba)
        ax[3, 2].axis("off")
        ax[3, 2].set_title("Overlap Binary", fontsize=12)

        plt.tight_layout()

    if path_output:
        plt.savefig(path_output.joinpath(f'{cell_id}_gird_images'), bbox_inches="tight", pad_inches=0, dpi=300)

        # save the plots as separate images
        titles = [
            "film_green_norm", "film_red_norm", "film_green_red_overlap",
            "intensity_overlap_original_green", "intensity_overlap_green_norm", "intensity_overlap_heatmap",
            'overlay_rgb', 'overlay_intensity', 'overlay_binary',
            "binary_green", 'binary_red', 'binary_overlap'
        ]
        output_dir = path_output.joinpath('separate_images')
        output_dir.mkdir(parents=True, exist_ok=True)

        _save_subplots_individually(fig, ax, titles, output_dir=output_dir, dpi=300)
    if show:
        plt.show()

    if output_path:
        # save the best image
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        ax.imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)  # gray black canvas
        g_boost = np.clip(_normalize01(green_u8) * 1.2, 0, 1)  # scale and cli
        ax.imshow(_normalize01(g_boost), cmap="Greens", alpha=0.9)
        ax.imshow(_normalize01(red_u8), cmap="Reds", alpha=0.6)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path_output.joinpath(f'{cell_id}_overlay_green_red'), bbox_inches="tight", pad_inches=0, dpi=300)
        plt.show()
        plt.close()


    return metrics, components
