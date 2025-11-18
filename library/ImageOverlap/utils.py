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
# from library.ImageOverlap.display import *

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