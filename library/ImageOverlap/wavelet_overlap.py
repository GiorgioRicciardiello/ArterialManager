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

https://github.com/NienhausLabKIT/HuepfelM/blob/master/WBNS/python_script/WBNS.py
"""

import pathlib
import pandas as pd
import pickle
import gzip
from skimage import morphology
from library.ImageOverlap.display import *
from library.ImageOverlap.metrics import (_compute_overlap_metrics,
                                          _compute_coloc_metrics_ml,)
from library.ImageOverlap.utils import (_normalize01,
                                        _wavelet_bg_subtraction,
                                        _vessel_enhance,
                                        _skeletonize_from_vesselness)
import gc
import cv2
from skimage.filters import threshold_niblack
import mahotas
import inspect


def plot_histogram_from_array(image_array):
    """
    Plot a histogram of pixel intensities from a NumPy array (grayscale image).

    Parameters:
    image_array (numpy.ndarray): Grayscale image as a NumPy array.
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    # Flatten the array to 1D
    pixels = image_array.ravel()

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(pixels, bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title('Histogram of Pixel Intensities')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



import gc
import matplotlib.pyplot as plt

def _clear_memory_keep(*keep):
    """Close all figures and delete all locals except those in `keep`."""
    plt.close('all')
    frame = inspect.currentframe().f_back
    for k in list(frame.f_locals.keys()):
        if k not in keep:
            try:
                del frame.f_locals[k]
            except:
                pass
    gc.collect()


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
                            show=False):
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
    img_red_rgb = cv2.cvtColor(img_red_color, cv2.COLOR_BGR2RGB)


    # Raw grayscale images (uint8).
    green_u8 = cv2.imread(str(path_green), cv2.IMREAD_GRAYSCALE)
    red_u8 = cv2.imread(str(path_red), cv2.IMREAD_GRAYSCALE)
    # plt.imshow(green_u8); plt.show()
    # plt.imshow(r_clean); plt.show()

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
    # plt.imshow(g_clean); plt.show()
    # plt.imshow(r_clean); plt.show()

    # --- Step 2: Vessel enhancement (Frangi) ---
    g_enh_u8, g_v = _vessel_enhance((g_clean * 255 / _normalize01(g_clean).max()).astype(np.uint8))
    r_enh_u8, r_v = _vessel_enhance((r_clean * 255 / _normalize01(r_clean).max()).astype(np.uint8))

    # plt.imshow(g_enh_u8); plt.show()
    # plt.imshow(r_enh_u8); plt.show()

    # --- Step 3: Skeletons ---
    g_bw, g_skel = _skeletonize_from_vesselness(g_v)
    r_bw, r_skel = _skeletonize_from_vesselness(r_v)
    vessel_mask = (g_bw | r_bw).astype(bool)

    # plt.imshow(g_bw); plt.show()
    # plt.imshow(r_bw); plt.show()

    # plt.imshow(g_skel); plt.show()
    # plt.imshow(r_skel); plt.show()

    # --- Step 4: Overlap detection ---
    # Structural overlap: skeleton proximity
    # define the filter
    # se = morphology.disk(skeleton_radius_px)
    se = morphology.disk(1)  # small structuring element

    # IF dilate before overlap then do:
    # r_dil = morphology.binary_dilation(r_skel, se)
    # g_dil = morphology.binary_dilation(g_skel, se)
    # near_overlap = (g_skel & r_dil) | (r_skel & g_dil)
    # Exact (pixel-perfect) overlap, remove the dilation
    near_overlap = g_skel & r_skel

    # Structural detection -> mask and edges
    g_mask = (green_u8 > 0).astype(np.uint8)  # green vessel presence
    r_mask = (red_u8 > 0).astype(np.uint8)  # red contrast presence

    # plt.imshow(g_mask); plt.show()
    # plt.imshow(r_mask); plt.show()


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

    # plt.imshow(G); plt.show()
    # plt.imshow(R); plt.show()

    overlay = np.zeros((*G.shape, 3), dtype=np.uint8)  # blank RGB image
    overlay[..., 1] = G  # green channel vesselness map
    overlay[..., 2] = R  # red channel vesselness map
    boost = (Y > 0)  # intensity overlap map - composite green–red image.
    # Finds pixels where overlap intensity exists
    overlay[..., 1][boost] = np.clip(overlay[..., 1][boost] + Y[boost] // 2, 0, 255)
    overlay[..., 2][boost] = np.clip(overlay[..., 2][boost] + Y[boost] // 2, 0, 255)
    # where overlap is strong, both red and green brighten → visually combining into yellow/orange highlights.

    # --- Step 5b: Adaptive thresholding ---
    # --- Bernsen thresholding (contrast-adaptive) ---
    mask_g_bernsen = mahotas.thresholding.bernsen(f=g_norm,
                                                  radius=2,
                                                  contrast_threshold=25,
                                                  gthresh=128) > 0

    mask_r_bernsen = mahotas.thresholding.bernsen(f=r_norm,
                                                  radius=2,
                                                  contrast_threshold=25,
                                                  gthresh=182) > 0

    # plt.imshow(mask_g_bernsen); plt.show()
    # plt.imshow(mask_r_bernsen); plt.show()

    # --- Niblack thresholding (local mean ± k*std) ---
    thr_g_niblack = threshold_niblack(G, window_size=25, k=-0.2)
    thr_r_niblack = threshold_niblack(R, window_size=25, k=-0.2)
    mask_g_niblack = G > thr_g_niblack
    mask_r_niblack = R > thr_r_niblack


    # plt.imshow(mask_g_niblack); plt.show()
    # plt.imshow(mask_r_niblack); plt.show()

    # --- Adaptive thresholding: Overlap calculation ---
    overlap_bernsen = np.logical_and(mask_g_bernsen, mask_r_bernsen)
    overlap_niblack = np.logical_and(mask_g_niblack, mask_r_niblack)


    # --- Step 6: Metrics Heatmap Intensity and Binary ---
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

    # --- Step 6b: complete metrics  ---
    metrics_abs = _compute_coloc_metrics_ml(
        G=G, R=R,
        g_norm=g_norm, g_clean=g_clean, r_norm=r_norm, r_clean=r_clean,
        g_mask=g_mask, r_mask=r_mask,
        g_v=g_v, r_v=r_v,
        g_skel=g_skel, r_skel=r_skel,
        vessel_mask=vessel_mask, near_overlap=near_overlap,
        overlap_strength=overlap_strength, heatmap_intensity=heatmap_intensity,
        heatmap_binary=heatmap_binary, overlay=overlay,
        mask_g_bernsen=mask_g_bernsen, mask_r_bernsen=mask_r_bernsen, overlap_bernsen=overlap_bernsen,
        mask_g_niblack=mask_g_niblack, mask_r_niblack=mask_r_niblack, overlap_niblack=overlap_niblack,
        show=show
    )

    # --- Step 7: Merge all dictionaries ---
    metrics = {**metrics_int, **metrics_bin, **metrics_abs,}

    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    df_metrics["method"] = df_metrics["Metric"].apply(
        lambda x: (
            "local_bernsen" if "bernsen" in x else
            "local_niblack" if "niblack" in x else
            "intensity" if x.endswith("_intensity") else
            "binary" if x.endswith("_binary") else
            "absolute"
        )
    )

    df_metrics["Metric"] = (df_metrics["Metric"]
                            .str.replace("_intensity", "", regex=False)
                            .str.replace("_binary", "", regex=False)
                            .str.replace("_bernsen", "", regex=False)
                            .str.replace("_niblack", "", regex=False))
    df_metrics.to_excel(path_output.joinpath(f'{cell_id}_metrics.xlsx'), index=False)

    # plt.imshow(g_skel); plt.show()
    # plt.imshow(r_skel); plt.show()


    # --- Step 9: Store the data ---
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
                "gold_binary": heatmap_binary_gold_rgba,  # RGBA gold overlay (binary)
                "bernsen": overlap_bernsen,
                "niblack": overlap_niblack,
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

    # --- Step 10: Visualization ---
    # ------------------------------------------------------------
    # Always save plots; only show interactively if show=True
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        7, 3,
        figsize=(8, 10),
        constrained_layout=False
    )
    plt.subplots_adjust(wspace=0.0001, hspace=0.5)

    gray_background = 0.8
    fig.patch.set_facecolor(f"{gray_background}")
    for a in ax.ravel():
        a.set_facecolor(f"{gray_background}")

    # --- prepare output folder for subplots ---
    output_dir = path_output.joinpath("subplots") if path_output else None
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Row 1 : Green channel ----------
    plot_green_fluorescent(ax[0, 0], img_green_color, heatmap_intensity_gold_rgba,
                           output_dir, f"{cell_id}_green_gold.png")
    plot_green_norm(ax[0, 1], g_norm,
                    output_dir, f"{cell_id}_green_norm.png")
    plot_green_binary_mask(ax[0, 2], g_mask,
                           output_dir, f"{cell_id}_green_mask.png")

    # ---------- Row 2 : Red channel ----------
    plot_red_fluorescent(ax[1, 0], img_red_rgb,
                         output_dir, f"{cell_id}_red_gold.png")
    plot_red_norm(ax[1, 1], r_norm,
                  output_dir, f"{cell_id}_red_norm.png")
    plot_red_binary_mask(ax[1, 2], r_mask,
                         output_dir, f"{cell_id}_red_mask.png")

    # ---------- Row 3 : Overlays ----------
    plot_green_red_overlay(ax[2, 0], g_norm, r_masked,
                           green_u8, g_edges,
                           output_dir, f"{cell_id}_green_red_overlay.png")
    plot_overlap_frangi(ax[2, 1], overlay,
                        output_dir, f"{cell_id}_overlay_frangi.png")
    plot_overlap_binary(ax[2, 2], heatmap_binary,
                        output_dir, f"{cell_id}_overlap_binary.png")

    # ---------- Row 4 : Binary + edges ----------
    plot_overlap_binary_edges(ax[3, 0], img_green_color,
                              g_edges, heatmap_binary_gold_rgba,
                              output_dir, f"{cell_id}_overlap_edges.png")
    plot_overlap_vessel_mask(ax[3, 1], vessel_mask,
                             output_dir, f"{cell_id}_vessel_mask.png")
    plot_green_skeleton(ax[3, 2], g_skel,
                        output_dir, f"{cell_id}_green_skeleton.png")

    # ---------- Row 5 : Skeletons and overlap ----------
    plot_red_skeleton(ax[4, 0], r_skel,
                      output_dir, f"{cell_id}_red_skeleton.png")
    plot_overlap_binary(ax[4, 1], near_overlap.astype(float),
                        output_dir, f"{cell_id}_near_overlap.png")
    plot_overlap_frangi(ax[4, 2], overlay,
                        output_dir, f"{cell_id}_frangi_overlay.png")

    # ---------- Row 6 : Adaptive thresholds (Bernsen) ----------
    plot_green_bernsen(ax[5, 0], mask_g_bernsen,
                       output_dir, f"{cell_id}_green_bernsen.png")
    plot_red_bernsen(ax[5, 1], mask_r_bernsen,
                     output_dir, f"{cell_id}_red_bernsen.png")
    plot_overlap_bernsen(ax[5, 2], mask_g_bernsen, mask_r_bernsen,
                         output_dir, f"{cell_id}_overlap_bernsen.png")

    # ---------- Row 7 : Adaptive thresholds (Niblack) ----------
    plot_green_niblack(ax[6, 0], mask_g_niblack,
                       output_dir, f"{cell_id}_green_niblack.png")
    plot_red_niblack(ax[6, 1], mask_r_niblack,
                     output_dir, f"{cell_id}_red_niblack.png")
    plot_overlap_niblack(ax[6, 2], mask_g_niblack, mask_r_niblack,
                         output_dir, f"{cell_id}_overlap_niblack.png")

    plt.tight_layout()

    # --- Save grid figure ---
    if path_output:
        fig.savefig(path_output.joinpath(f"{cell_id}_grid_overview.png"),
                    bbox_inches="tight", pad_inches=0, dpi=300)

    if show:
        # Enable interactive mode and display
        plt.ion()
        plt.show(block=True)

    plt.close(fig)

    # --- Optional: save single overlay summaries ---
    if path_output:
        for name, plot_func, args in [
            (f"{cell_id}_overlay_green_red.png", plot_overlay_green_red, (green_u8, red_u8)),
            (f"{cell_id}_overlay_binary_edges.png", plot_overlay_binary_edges,
             (img_green_color, g_edges, heatmap_binary_gold_rgba)),
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
            plot_func(ax, *args, output_dir=path_output, filename=name)
            if show:
                plt.ion()
                plt.show(block=True)
            plt.close(fig)
    # Clean memory
    plt.close('all')
    cv2.destroyAllWindows()
    gc.collect()

    try:
        del img_green_color, img_red_color, img_red_rgb, \
            red_u8, green_u8, \
            G, R, \
            g_norm, g_clean, r_norm, r_clean, \
            g_mask, r_mask, \
            g_edges, r_edges, \
            g_v, r_v, \
            g_skel, r_skel, \
            vessel_mask, near_overlap, \
            overlap_strength, heatmap_intensity, \
            heatmap_binary, overlay, \
            mask_g_bernsen, mask_r_bernsen, overlap_bernsen, \
            mask_g_niblack, mask_r_niblack, overlap_niblack, \
            thr_g_niblack, thr_r_niblack, \
            heatmap_intensity_gold_rgba, heatmap_binary_gold_rgba
    except NameError as e:
        print(f"Trying to clean memory failed: {e}")

    # _clear_memory_keep("metrics", "components")

    return metrics, components


