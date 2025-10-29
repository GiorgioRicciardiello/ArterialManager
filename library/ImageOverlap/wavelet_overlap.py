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
from config.config import CONFIG
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure, filters, morphology
from typing import Dict, Optional, List, Tuple
from library.ImageOverlap.display import *
from library.ImageOverlap.utils import (_normalize01,
                                        _wavelet_bg_subtraction,
                                        _save_subplots_individually,
                                        _vessel_enhance,
                                        _skeletonize_from_vesselness,
                                        _manders_coeffs,
                                        _edges_to_rgba)
import cv2
from scipy.ndimage import distance_transform_edt as edt


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




def _compute_coloc_metrics_ml(
    G:np.ndarray,
    R:np.ndarray,
    g_norm:np.ndarray,
    g_clean:np.ndarray,
    r_norm:np.ndarray,
    r_clean:np.ndarray,
    g_v:np.ndarray, r_v:np.ndarray,
    g_skel:np.ndarray, r_skel:np.ndarray,
    vessel_mask:np.ndarray,
    near_overlap:np.ndarray,
    overlap_strength:np.ndarray,
    heatmap_intensity:np.ndarray,
    heatmap_binary:np.ndarray,
    overlay:np.ndarray
):
    """
    Compute quantitative features describing structural and intensity-based
    colocalization between red and green vascular networks.

    Returns
    -------
    metrics : dict
        Dictionary with >40 scalar metrics. Each key is interpretable and
        normalized for ML or comparative analysis.

    Interpretation summary
    ----------------------
    * `_abs`  = absolute sum of pixel intensities or counts (depends on metric).
    * `_pct`  = normalized by total image area → comparable across samples.
    * 'weighted_*' = normalized by total signal intensity of each channel.
    * Values ~1 indicate strong or complete colocalization; values ~0 indicate poor overlap.
    """

    H, W = g_norm.shape
    N = float(H * W)
    def _pct(x):
        """Return mean intensity or binary fraction over total pixels."""
        return float(np.asarray(x, float).sum()) / (N + 1e-8)

    # ==============================================================
    # 1. CHANNEL ENERGY METRICS (global signal levels)
    # --------------------------------------------------------------
    # Quantify total and normalized intensity before and after cleaning and Frangi enhancement.
    # Useful to detect staining strength, uneven exposure, or processing bias.
    # ==============================================================

    g_norm_abs, g_clean_abs = g_norm.sum(), g_clean.sum()
    r_norm_abs, r_clean_abs = r_norm.sum(), r_clean.sum()
    g_v_abs, r_v_abs = g_v.sum(), r_v.sum()

    g_norm_pct, g_clean_pct = _pct(g_norm), _pct(g_clean)
    r_norm_pct, r_clean_pct = _pct(r_norm), _pct(r_clean)
    g_v_pct, r_v_pct = _pct(g_v), _pct(r_v)

    # ==============================================================
    # 2. OVERLAP INTENSITY SUMS
    # --------------------------------------------------------------
    # Represent total coincidence between the two channels.
    # "overlap_strength" = pixelwise product/geometric mean of vessel intensities.
    # ==============================================================

    sum_overlap_strength = float(overlap_strength.sum())
    sum_heat_int = float(heatmap_intensity.sum())
    sum_heat_bin = float(heatmap_binary.sum())
    sum_vessel = float(vessel_mask.sum())

    sum_green, sum_red = float(G.sum()), float(R.sum())

    # ==============================================================
    # 3. INTENSITY-WEIGHTED FRACTIONS
    # --------------------------------------------------------------
    # Normalized by total channel energy. Values 0–1.
    # Show what fraction of each channel’s total signal coincides with the other.
    # ==============================================================

    weighted_overlap_green = sum_overlap_strength / (sum_green + 1e-8)  # fraction of green overlapping red
    weighted_overlap_red = sum_overlap_strength / (sum_red + 1e-8)      # fraction of red overlapping green
    weighted_overlap_mean = 2 * sum_overlap_strength / (sum_green + sum_red + 1e-8)  # symmetric average
    green_red_ratio = (sum_green + 1e-8) / (sum_red + 1e-8)  # global channel intensity balance
    # If > 1→ green channel dominates (brighter overall).
    # If < 1 → red channel dominates (weaker green signal).
    red_green_ratio = 1/green_red_ratio  # global channel intensity balance

    # ==============================================================
    # 4. COSINE SIMILARITY (INTENSITY SHAPE AGREEMENT)
    # --------------------------------------------------------------
    # Measures alignment of intensity vectors ignoring scale.
    # 1 → identical spatial intensity pattern; 0 → orthogonal (no relation).
    # ==============================================================

    vm = vessel_mask.astype(bool)
    Gm, Rm = G[vm].astype(float), R[vm].astype(float)
    cos_sim = (Gm @ Rm) / (np.linalg.norm(Gm) * np.linalg.norm(Rm) + 1e-12) if vm.any() else np.nan

    # ==============================================================
    # 5. HISTOGRAM OVERLAP METRICS (DISTRIBUTIONAL SIMILARITY)
    # --------------------------------------------------------------
    # Bhattacharyya Coefficient and Hellinger Distance between intensity histograms.
    # BC ∈ [0,1]: 1 → identical distributions; 0 → disjoint.
    # Hellinger ∈ [0,1]: 0 → identical; 1 → dissimilar.
    # ==============================================================

    nbins = 64
    if vm.any():
        p, _ = np.histogram(Gm, bins=nbins, range=(0,255), density=True)
        q, _ = np.histogram(Rm, bins=nbins, range=(0,255), density=True)
        bhattacharyya = float(np.sqrt(p * q).sum())
        hellinger = float(np.sqrt(max(0.0, 1.0 - bhattacharyya)))
    else:
        bhattacharyya, hellinger = np.nan, np.nan

    # ==============================================================
    # 6. MUTUAL INFORMATION (INFORMATION SHARED)
    # --------------------------------------------------------------
    # Captures nonlinear intensity dependencies.
    # Larger MI ⇒ higher information overlap beyond linear correlation.
    # ==============================================================

    if vm.any():
        joint, _, _ = np.histogram2d(Gm, Rm, bins=nbins, range=[[0,255],[0,255]], density=True)
        pj = joint + 1e-12
        pi, pj2 = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
        mutual_information = float((pj * (np.log(pj) - np.log(pi) - np.log(pj2))).sum())
    else:
        mutual_information = np.nan

    # ==============================================================
    # 7. LI’s INTENSITY CORRELATION QUOTIENT (ICQ)
    # --------------------------------------------------------------
    # Counts how often deviations of G and R intensities have the same sign.
    # Range: [-0.5, +0.5]; positive → colocalized; negative → segregated.
    # ==============================================================

    if vm.any():
        Gc, Rc = Gm - Gm.mean(), Rm - Rm.mean()
        icq = float(np.mean(np.sign(Gc * Rc)))
    else:
        icq = np.nan

    # ==============================================================
    # 8. BINARY OVERLAP METRICS (DETECTION-STYLE)
    # --------------------------------------------------------------
    # Precision/Recall/F1 between binary masks of red and green.
    # Useful for model learning or segmentation accuracy estimation.
    # ==============================================================

    r_bw = R > 0
    g_bw = G > 0
    tp = float(np.logical_and(r_bw, g_bw).sum())
    precision = tp / (r_bw.sum() + 1e-8)
    recall = tp / (g_bw.sum() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # ==============================================================
    # 9. SKELETON DISTANCE METRICS (GEOMETRIC AGREEMENT)
    # --------------------------------------------------------------
    # ASSD: average symmetric distance between skeletons.
    # Hausdorff: maximum surface separation (worst-case).
    # Units: pixels. Smaller = better colocalization.
    # ==============================================================

    def _assd(a, b):
        if not a.any() or not b.any(): return np.nan
        da, db = edt(~a), edt(~b)
        return float((da[b].mean() + db[a].mean()) / 2)

    def _hausdorff(a, b):
        if not a.any() or not b.any(): return np.nan
        return float(max(edt(~b)[a].max(), edt(~a)[b].max()))

    assd_skel = _assd(g_skel.astype(bool), r_skel.astype(bool))
    hausdorff_skel = _hausdorff(g_skel.astype(bool), r_skel.astype(bool))

    # ==============================================================
    # 10. SKELETON MORPHOLOGY METRICS
    # --------------------------------------------------------------
    # Endpoints = vessel tips (connectivity=1)
    # Branchpoints = vessel junctions (connectivity≥3)
    # Useful for assessing structural complexity and connectivity alignment.
    # ==============================================================

    kernel = np.ones((3,3), np.uint8)
    def _end_branch(skel):
        n = cv2.filter2D(skel.astype(np.uint8), -1, kernel) - skel.astype(np.uint8)
        ends = int(((n == 1) & (skel == 1)).sum())
        branches = int(((n >= 3) & (skel == 1)).sum())
        return ends, branches

    g_end, g_branch = _end_branch(g_skel)
    r_end, r_branch = _end_branch(r_skel)

    # ==============================================================
    # 11. AGGREGATE RESULTS
    # --------------------------------------------------------------
    # Combine all scalar features into a single dict for downstream analysis.
    # ==============================================================

    return {
        # channel signal
        "g_norm_abs": g_norm_abs, "g_norm_pct": g_norm_pct,
        "g_clean_abs": g_clean_abs, "g_clean_pct": g_clean_pct,
        "r_norm_abs": r_norm_abs, "r_norm_pct": r_norm_pct,
        "r_clean_abs": r_clean_abs, "r_clean_pct": r_clean_pct,
        "g_v_abs": g_v_abs, "g_v_pct": g_v_pct,
        "r_v_abs": r_v_abs, "r_v_pct": r_v_pct,

        # overlap intensities
        "sum_overlap_strength": sum_overlap_strength,
        "sum_heatmap_intensity": sum_heat_int,
        "sum_heatmap_binary": sum_heat_bin,
        "frac_heatmap_intensity": sum_heat_int / (sum_vessel + 1e-8),
        "frac_heatmap_binary": sum_heat_bin / (sum_vessel + 1e-8),

        "sum_vessel_mask": float(vessel_mask.sum()),
        "sum_overlay": float(overlay.sum()),
        "sum_near_overlap": float(near_overlap.sum()),

        # weighted overlap measures
        "weighted_overlap_green": weighted_overlap_green,
        "weighted_overlap_red": weighted_overlap_red,
        "weighted_overlap_mean": weighted_overlap_mean,
        "green_red_ratio": green_red_ratio,
        'red_green_ratio': red_green_ratio,


        # distribution / information metrics
        "cosine_similarity": cos_sim,
        "bhattacharyya_coeff": bhattacharyya,
        "hellinger_distance": hellinger,
        "mutual_information": mutual_information,
        "icq": icq,

        # detection-style binary metrics
        "precision": precision,
        "recall": recall,
        "f1": f1,

        # skeleton geometry and topology
        "g_skel_len_abs": float(g_skel.sum()), "g_skel_len_pct": _pct(g_skel),
        "r_skel_len_abs": float(r_skel.sum()), "r_skel_len_pct": _pct(r_skel),
        "skel_overlap_len_abs": float((g_skel & r_skel).sum()),
        "skel_overlap_len_pct": _pct(g_skel & r_skel),
        "assd_skeleton": assd_skel,
        "hausdorff_skeleton": hausdorff_skel,
        "g_endpoints": g_end, "g_branchpoints": g_branch,
        "r_endpoints": r_end, "r_branchpoints": r_branch,
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
    img_red_rgb = cv2.cvtColor(img_red_color, cv2.COLOR_BGR2RGB)

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

    # IF dilate before overlap then do:
    # r_dil = morphology.binary_dilation(r_skel, se)
    # g_dil = morphology.binary_dilation(g_skel, se)
    # near_overlap = (g_skel & r_dil) | (r_skel & g_dil)
    # Exact (pixel-perfect) overlap, remove the dilation
    near_overlap = g_skel & r_skel

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

    # --- Step 6b: Absolute and weighted overlap measures ---
    metrics_abs = _compute_coloc_metrics_ml(
        G, R,
        g_norm, g_clean, r_norm, r_clean,
        g_v, r_v,
        g_skel, r_skel,
        vessel_mask, near_overlap,
        overlap_strength, heatmap_intensity,
        heatmap_binary, overlay
    )

    # merge both dictionaries
    metrics = {**metrics_int, **metrics_bin, **metrics_abs}


    df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    df_metrics["method"] = df_metrics["Metric"].apply(
        lambda x: "intensity" if x.endswith("_intensity") else
                  ("binary" if x.endswith("_binary") else "absolute")
    )
    df_metrics["Metric"] = (df_metrics["Metric"]
                            .str.replace("_intensity", "", regex=False)
                            .str.replace("_binary", "", regex=False))

    df_metrics.sort_values(by=["method", 'Metric'], ascending=[False,True], inplace=True)
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
        fig, ax = plt.subplots(5, 3, figsize=(18, 20))
        output_dir = path_output.joinpath("subplots") if path_output else None

        # ---------- Row 1 : Green channel ----------
        plot_green_fluorescent(ax[0, 0], img_green_color,
                               heatmap_intensity_gold_rgba,
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

        plt.tight_layout()

        if path_output:
            fig.savefig(path_output.joinpath(f"{cell_id}_grid_overview.png"),
                        bbox_inches="tight", pad_inches=0, dpi=300)

        if show:
            plt.show()

    # ---------------------
    if path_output:
        plt.savefig(path_output.joinpath(f'{cell_id}_gird_images.png'), bbox_inches="tight", pad_inches=0, dpi=300)

    if show:
        plt.show()

    if path_output:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
        plot_overlay_green_red(ax, green_u8, red_u8,
                               output_dir=path_output, filename=f"{cell_id}_overlay_green_red.png")
        plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
        plot_overlay_binary_edges(ax, img_green_color, g_edges, heatmap_binary_gold_rgba,
                                  output_dir=path_output, filename=f"{cell_id}_overlay_binary_edges.png")
        plt.show()

        plt.close(fig)


    return metrics, components



# def coloc_vessels_with_wbns(path_green:pathlib.Path,
#                             path_red:pathlib.Path,
#                             path_output:pathlib.Path,
#                             cell_id:Optional[str] = 'Cell',
#                             resolution_px:Optional[int]=4,
#                             noise_lvl=1,
#                             skeleton_radius_px=2,
#                             intensity_overlap:str='geo_mean',
#                             output_path:pathlib.Path=None,
#                             show=True):
#     """
#     Vessel colocalization with WBNS preprocessing.
#
#     Parameters
#     ----------
#     path_green, path_red : pathlib.Path
#         Path of the green vessel and the red vessel.
#     resolution_px : int
#         Resolution parameter for WBNS (FWHM in px).
#     noise_lvl : int
#         Noise suppression level (WBNS).
#     skeleton_radius_px : int
#         Tolerance for calling skeletons overlapping.
#     intensity_overlap : str
#         Which intensity overlap should be applied. Options are: 'product', 'minimum', 'geo_mean', 'pearson_weighted'
#     show : bool
#         If True, show overlay + heatmap_intensity.
#
#     Returns
#     -------
#     overlay : RGB image (uint8)
#     heatmap_intensity : float32 in [0,1]
#     masks : dict of binary masks
#     metrics : dict of colocalization scores
#     """
#     # -- read image in the different formats we need
#     img_green_color = cv2.imread(str(path_green), cv2.IMREAD_COLOR)
#     img_red_color = cv2.imread(str(path_red), cv2.IMREAD_COLOR)
#     img_red_rgb = cv2.cvtColor(img_red_color, cv2.COLOR_BGR2RGB)
#
#     # Raw grayscale images (uint8).
#     green_u8 = cv2.imread(str(path_green), cv2.IMREAD_GRAYSCALE)
#     red_u8 = cv2.imread(str(path_red), cv2.IMREAD_GRAYSCALE)
#
#     # normalize
#     g_norm = _normalize01(green_u8)
#     r_norm = _normalize01(red_u8)
#
#     # masks -> show red only where green
#     mask_overlap = g_norm > 0
#     r_masked = np.zeros_like(r_norm)
#     r_masked[mask_overlap] = r_norm[mask_overlap]
#
#     # --- Step 1: WBNS background/noise subtraction ---
#     g_clean = _wavelet_bg_subtraction(green_u8, resolution_px, noise_lvl)
#     r_clean = _wavelet_bg_subtraction(red_u8, resolution_px, noise_lvl)
#
#     # --- Step 2: Vessel enhancement (Frangi) ---
#     g_enh_u8, g_v = _vessel_enhance((g_clean * 255 / _normalize01(g_clean).max()).astype(np.uint8))
#     r_enh_u8, r_v = _vessel_enhance((r_clean * 255 / _normalize01(r_clean).max()).astype(np.uint8))
#
#     # --- Step 3: Skeletons ---
#     g_bw, g_skel = _skeletonize_from_vesselness(g_v)
#     r_bw, r_skel = _skeletonize_from_vesselness(r_v)
#     vessel_mask = (g_bw | r_bw).astype(bool)
#
#     # --- Step 4: Overlap detection ---
#     # Structural overlap: skeleton proximity
#     # define the filter
#     # se = morphology.disk(skeleton_radius_px)
#     se = morphology.disk(1)  # small structuring element
#
#     # IF dilate before overlap then do:
#     # r_dil = morphology.binary_dilation(r_skel, se)
#     # g_dil = morphology.binary_dilation(g_skel, se)
#     # near_overlap = (g_skel & r_dil) | (r_skel & g_dil)
#     # Exact (pixel-perfect) overlap, remove the dilation
#     near_overlap = g_skel & r_skel
#
#     # Structural detection -> mask and edges
#     g_mask = (green_u8 > 0).astype(np.uint8)  # green vessel presence
#     r_mask = (red_u8 > 0).astype(np.uint8)  # red contrast presence
#
#     # Edges with High-pass filter (extract edges from g_mask)  or Sobel for strong edges
#     g_edges = cv2.Laplacian(g_mask.astype(np.uint8), cv2.CV_64F)
#     g_edges = np.abs(g_edges)
#     g_edges = (g_edges > 0).astype(np.uint8)  # binarize edges
#
#     r_edges = cv2.Laplacian(r_mask.astype(np.uint8), cv2.CV_64F)
#     r_edges = np.abs(r_edges)
#     r_edges = (r_edges > 0).astype(np.uint8)  # binarize edges
#
#     # --- Intensity-based overlap ---
#     if intensity_overlap == 'product':
#         # Method 1: product
#         overlap_strength = g_v * r_v
#
#     elif intensity_overlap == 'minimum':
#         # Method 2: minimum
#         overlap_strength = np.minimum(g_v, r_v)
#
#     elif intensity_overlap == 'geo_mean':
#         # Method 3: geometric mean
#         overlap_strength = np.sqrt(g_v * r_v)
#
#     elif intensity_overlap == 'pearson_weighted':
#         # Method 4: Pearson-weighted overlap
#         mean_g = g_v[g_mask].mean() if g_mask.sum() > 0 else g_v.mean()
#         mean_r = r_v[r_mask].mean() if r_mask.sum() > 0 else r_v.mean()
#         overlap_strength = (g_v - mean_g) * (r_v - mean_r)
#         overlap_strength = np.clip(overlap_strength, 0, None)  # keep only positive correlations
#
#     elif intensity_overlap == 'adaptive_weighted':
#         # Method 5: Adaptive weighting with vessel skeletons
#         base_strength = g_v * r_v
#         overlap_strength = _normalize01(base_strength) * (0.5 + 0.5 * near_overlap.astype(np.float32))
#
#     else:
#         raise ValueError(f"Unknown intensity_overlap method: {intensity_overlap}")
#
#     # --- Final normalized heatmap (applies skeleton constraint if desired) ---
#     heatmap_intensity = _normalize01(overlap_strength) * near_overlap.astype(np.float32)
#
#     heatmap_intensity_gold_rgba = np.zeros((*heatmap_intensity.shape, 4))
#     heatmap_intensity_gold_rgba[..., 0] = 1.0
#     heatmap_intensity_gold_rgba[..., 1] = 0.84
#     heatmap_intensity_gold_rgba[..., 2] = 0.0
#     heatmap_intensity_gold_rgba[..., 3] = heatmap_intensity  # directly scale alpha by overlap strength
#
#     # heatmap_binary = np.logical_and(g_mask, r_mask).astype(np.float32)
#     heatmap_binary = (g_mask > 0) & (r_mask > 0)
#
#     # gold RGBA overlay (use mask as alpha channel)
#     heatmap_binary_gold_rgba = np.zeros((*heatmap_binary.shape, 4))
#     heatmap_binary_gold_rgba[..., 0] = 1.0  # Red channel
#     heatmap_binary_gold_rgba[..., 1] = 0.84  # Green channel
#     heatmap_binary_gold_rgba[..., 2] = 0.0  # Blue channel (RGB ~ gold)
#     heatmap_binary_gold_rgba[..., 3] = heatmap_binary.astype(float) * 0.8  # alpha for overlap
#
#
#     # --- Step 5: Overlay ---
#     # normalize vessel channels _v are Frangi-enhanced vesselness maps
#     # Multiplying by 255 converts to 8-bit intensity image
#     G = (255 * _normalize01(g_v)).astype(np.uint8)
#     R = (255 * _normalize01(r_v)).astype(np.uint8)
#     Y = (255 * _normalize01(heatmap_intensity)).astype(np.uint8)
#
#     overlay = np.zeros((*G.shape, 3), dtype=np.uint8)  # blank RGB image
#     overlay[..., 1] = G  # green channel vesselness map
#     overlay[..., 2] = R  # red channel vesselness map
#     boost = (Y > 0)  # intensity overlap map - composite green–red image.
#     # Finds pixels where overlap intensity exists
#     overlay[..., 1][boost] = np.clip(overlay[..., 1][boost] + Y[boost] // 2, 0, 255)
#     overlay[..., 2][boost] = np.clip(overlay[..., 2][boost] + Y[boost] // 2, 0, 255)
#     # where overlap is strong, both red and green brighten → visually combining into yellow/orange highlights.
#
#     # --- Step 6: Metrics ---
#     metrics_int = _compute_overlap_metrics(G=G, R=R,
#                                            g_skel=g_skel,
#                                            r_skel=r_skel,
#                                            mask_overlap=heatmap_intensity,
#                                            vessel_mask=vessel_mask,
#                                            label="intensity")
#
#     metrics_bin = _compute_overlap_metrics(G=G, R=R,
#                                            g_skel=g_skel,
#                                            r_skel=r_skel,
#                                            mask_overlap=heatmap_binary,
#                                            vessel_mask=vessel_mask,
#                                            label="binary")
#
#     # merge both dictionaries
#     metrics = {**metrics_int, **metrics_bin}
#
#     df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
#     # Extract method and clean up metric names
#     df_metrics["method"] = df_metrics["Metric"].apply(
#         lambda x: "intensity" if x.endswith("_intensity") else "binary"
#     )
#
#     df_metrics["Metric"] = df_metrics["Metric"].str.replace("_intensity", "", regex=False)
#     df_metrics["Metric"] = df_metrics["Metric"].str.replace("_binary", "", regex=False)
#
#
#     df_metrics.to_excel(path_output.joinpath(f'{cell_id}_metrics.xlsx'), index=False)
#     # --- Step 7: Store the data ---
#     components = {
#         "green": {
#             "raw": {
#                 "intensity": G.astype(np.uint8),  # vesselness map
#                 "intensity_norm": _normalize01(G).astype(np.float32),
#                 "mask": g_mask.astype(np.uint8)  # binary vessel presence
#             },
#             "derived": {
#                 "skeleton": g_skel.astype(np.uint8),  # vessel skeleton
#                 "edges": g_edges.astype(np.uint8)  # vessel edges
#             }
#         },
#
#         "red": {
#             "raw": {
#                 "intensity": R.astype(np.uint8),
#                 "intensity_norm": _normalize01(R).astype(np.float32),
#                 "mask": r_mask.astype(np.uint8)
#             },
#             "derived": {
#                 "skeleton": r_skel.astype(np.uint8),
#                 "edges": r_edges.astype(np.uint8)
#             }
#         },
#
#         "overlap": {
#             "raw": {
#                 "intensity": heatmap_intensity.astype(np.float32),
#                 "binary": heatmap_binary.astype(np.uint8),
#                 "vessel_mask": vessel_mask.astype(np.uint8)
#             },
#             "derived": {
#                 "overlay": overlay.astype(np.uint8),  # RGB composite
#                 "near_overlap": near_overlap.astype(np.uint8),  # skeleton proximity overlap
#                 "r_masked": r_masked.astype(np.uint8),  # red restricted to green
#                 "gold_intensity": heatmap_intensity_gold_rgba,  # RGBA gold overlay (float)
#                 "gold_binary": heatmap_binary_gold_rgba  # RGBA gold overlay (binary)
#             }
#         },
#
#         "meta": {
#             "intensity_overlap": intensity_overlap,
#             "resolution_px": resolution_px,
#             "noise_lvl": noise_lvl,
#             "skeleton_radius_px": skeleton_radius_px
#         }
#     }
#     if path_output:
#         with gzip.open(path_output.joinpath(f"{cell_id}_components.pkl.gz"), "wb") as f:
#             pickle.dump(components, f)
#
#     # --- Step 8: Visualization ---
#     if path_output or show:
#         # Normalize images for display
#         h_norm = _normalize01(heatmap_intensity)
#         h_bin = _normalize01(heatmap_binary)
#
#         fig, ax = plt.subplots(4, 3, figsize=(18, 15))
#         # ---------------- Row 1: raw images ----------------
#         # - Raw green (on black background)
#         ax[0, 0].imshow(g_norm , cmap="Greens")
#         ax[0, 0].set_title("Green (Normalized)", fontsize=12)
#         ax[0, 0].axis("off")
#
#         # - Raw red (on black background)
#         ax[0, 1].imshow(r_norm, cmap="Reds")
#         ax[0, 1].set_title("Red (Normalized)", fontsize=12)
#         ax[0, 1].axis("off")
#
#         # - Green + Red overlay (black background)
#         ax[0, 2].imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)  # gray black canvas
#         # ax[0, 2].set_facecolor("black")
#         ax[0, 2].imshow(g_norm , cmap="Greens", alpha=0.9)
#         ax[0, 2].imshow(r_masked, cmap="Reds", alpha=0.6)
#         edges_rgba = _edges_to_rgba(g_edges, hex_color="#00FF00", alpha=0.5)
#         ax[0, 2].imshow(edges_rgba)
#         ax[0, 2].set_title("Green + Red overlay", fontsize=12)
#         ax[0, 2].axis("off")
#
#         # ---------------- Row 2: Intensity-based overlap----------------
#         # 1. Vessels + gold overlap
#         ax[1, 0].imshow(img_green_color, alpha=0.9)
#         ax[1, 0].imshow(heatmap_intensity_gold_rgba)
#         ax[1, 0].set_title("Intensity Overlap: Green Original", fontsize=12)
#         ax[1, 0].axis("off")
#
#         # 2. Green vessels + gold highlights (same gold RGBA overlay)
#         ax[1, 1].imshow(g_norm, cmap="Greens", alpha=0.9)
#         ax[1, 1].imshow(heatmap_intensity_gold_rgba)
#         ax[1, 1].set_title("Intensity Overlap: Green Normalized", fontsize=12)
#         ax[1, 1].axis("off")
#
#         # 3. Heatmap view
#         im = ax[1, 2].imshow(heatmap_intensity, cmap="viridis")  # or "inferno"/"cividis"
#         ax[1, 2].set_title("Intensity Overlap: Heatmap", fontsize=12)
#         ax[1, 2].axis("off")
#         cbar = plt.colorbar(im, ax=ax[1, 2], fraction=0.046, pad=0.04)
#         cbar.set_label("Normalized Overlap Intensity", fontsize=10)
#
#         # ---------------- Row 3: Overlay ----------------
#
#         ax[2, 0].imshow(overlay)
#         ax[2, 0].set_title("RGB Overlay")
#         ax[2, 0].axis("off")
#         ax[2, 1].imshow(heatmap_intensity, cmap="plasma")
#         ax[2, 1].set_title("Intensity Heatmap")
#         ax[2, 1].axis("off")
#         ax[2, 2].imshow(heatmap_binary, cmap="gray")
#         ax[2, 2].set_title("Binary Overlap")
#         ax[2, 2].axis("off")
#
#         # ---------------- Row 4: Binary overlap ----------------
#         # Green mask
#         ax[3, 0].imshow(g_mask, cmap="Greens")
#         ax[3, 0].set_title("Green Binary Mask", fontsize=12)
#         ax[3, 0].axis("off")
#
#         # Red mask
#         ax[3, 1].imshow(r_mask, cmap="Reds")
#         ax[3, 1].set_title("Red Binary Mask", fontsize=12)
#         ax[3, 1].axis("off")
#
#         # Heatmap + green edges + gold overlap
#         # ax[2, 2].imshow(h_bin, cmap="inferno")
#         # ax[2, 2].imshow(g_edges, cmap="cool", alpha=0.7)
#         # ax[2, 2].imshow(heatmap_binary_gold_rgba)  # gold where overlap occurs
#         # ax[2, 2].set_title("Binary Heatmap + Edges + Gold Overlap", fontsize=12)
#         # ax[2, 2].axis("off")
#
#         ax[3, 2].imshow(img_green_color, alpha=0.9)
#         edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
#         ax[3, 2].imshow(edges_rgba)
#         ax[3, 2].imshow(heatmap_binary_gold_rgba)
#         ax[3, 2].axis("off")
#         ax[3, 2].set_title("Overlap Binary", fontsize=12)
#
#         plt.tight_layout()
#
#     # ---------------------
#     # --- Step 8: Visualization ---
#
#     img_green_color = cv2.imread(str(path_green), cv2.IMREAD_COLOR)
#     img_red_color = cv2.imread(str(path_red), cv2.IMREAD_COLOR)
#     plt.imshow(img_red_color, alpha=0.9)
#     plt.show()
#
#     # Convert BGR to RGB for correct display in matplotlib
#     img_rgb = cv2.cvtColor(img_red_color, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.axis('off')  # Optional: hides axis ticks
#     plt.title('Red Fluorescent Image')
#     plt.show()
#
#
#     # Convert BGR to RGB for correct display in matplotlib
#     img_rgb = cv2.cvtColor(img_green_color, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     plt.axis('off')  # Optional: hides axis ticks
#     plt.title('Red Fluorescent Image')
#     plt.show()
#
#
#
#     if path_output or show:
#         fig, ax = plt.subplots(5, 3, figsize=(18, 20))
#
#         # ---------- Row 1 : Green channel ----------
#         ax[0, 0].imshow(img_green_color, alpha=0.9)
#         ax[0, 0].imshow(heatmap_intensity_gold_rgba)
#         ax[0, 0].set_title("Green Channel + Gold Overlap", fontsize=12)
#         ax[0, 0].axis("off")
#
#         ax[0, 1].imshow(g_norm, cmap="Greens")
#         ax[0, 1].set_title("Green (Normalized)", fontsize=12)
#         ax[0, 1].axis("off")
#
#         ax[0, 2].imshow(g_mask, cmap="Greens")
#         ax[0, 2].set_title("Green Binary Mask", fontsize=12)
#         ax[0, 2].axis("off")
#
#         # ---------- Row 2 : Red channel ----------
#         ax[1, 0].imshow(img_red_rgb)
#         ax[1, 0].imshow(heatmap_intensity_gold_rgba)
#         ax[1, 0].set_title("Red Channel", fontsize=12)
#         ax[1, 0].axis("off")
#
#         ax[1, 1].imshow(r_norm, cmap="Reds")
#         ax[1, 1].set_title("Red (Normalized)", fontsize=12)
#         ax[1, 1].axis("off")
#
#         ax[1, 2].imshow(r_mask, cmap="Reds")
#         ax[1, 2].set_title("Red Binary Mask", fontsize=12)
#         ax[1, 2].axis("off")
#
#         # ---------- Row 3 : Raw overlay ----------
#         ax[2, 0].imshow(g_norm, cmap="Greens", alpha=0.9)
#         ax[2, 0].imshow(r_masked, cmap="Reds", alpha=0.6)
#         ax[2, 0].set_title("Green + Red Overlay", fontsize=12)
#         ax[2, 0].axis("off")
#
#         ax[2, 1].imshow(overlay)
#         ax[2, 1].set_title("RGB Overlay (Frangi Enhanced)", fontsize=12)
#         ax[2, 1].axis("off")
#
#         im = ax[2, 2].imshow(heatmap_intensity, cmap="viridis")
#         ax[2, 2].set_title("Overlap Intensity Heatmap", fontsize=12)
#         ax[2, 2].axis("off")
#         cbar = plt.colorbar(im, ax=ax[2, 2], fraction=0.046, pad=0.04)
#         cbar.set_label("Normalized Overlap", fontsize=10)
#
#         # ---------- Row 4 : Binary overlays ----------
#         ax[3, 0].imshow(heatmap_binary, cmap="gray")
#         ax[3, 0].set_title("Binary Overlap", fontsize=12)
#         ax[3, 0].axis("off")
#
#         ax[3, 1].imshow(img_green_color, alpha=0.9)
#         edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
#         ax[3, 1].imshow(edges_rgba)
#         ax[3, 1].imshow(heatmap_binary_gold_rgba)
#         ax[3, 1].set_title("Overlap Binary + Edges", fontsize=12)
#         ax[3, 1].axis("off")
#
#         ax[3, 2].imshow(vessel_mask, cmap="gray")
#         ax[3, 2].set_title("Combined Vessel Mask", fontsize=12)
#         ax[3, 2].axis("off")
#
#         # ---------- Row 5 : Summary / remaining visuals ----------
#         ax[4, 0].imshow(g_skel, cmap="Greens")
#         ax[4, 0].set_title("Green Skeleton", fontsize=12)
#         ax[4, 0].axis("off")
#
#         ax[4, 1].imshow(r_skel, cmap="Reds")
#         ax[4, 1].set_title("Red Skeleton", fontsize=12)
#         ax[4, 1].axis("off")
#
#         ax[4, 2].imshow(near_overlap, cmap="hot")
#         ax[4, 2].set_title("Skeleton Overlap Map", fontsize=12)
#         ax[4, 2].axis("off")
#
#         plt.tight_layout()
#
#     # ---------------------
#     if path_output:
#         plt.savefig(path_output.joinpath(f'{cell_id}_gird_images.png'), bbox_inches="tight", pad_inches=0, dpi=300)
#
#         # save the plots as separate images
#         titles = [
#             "film_green_norm", "film_red_norm", "film_green_red_overlap",
#             "intensity_overlap_original_green", "intensity_overlap_green_norm", "intensity_overlap_heatmap",
#             'overlay_rgb', 'overlay_intensity', 'overlay_binary',
#             "binary_green", 'binary_red', 'binary_overlap'
#         ]
#         output_dir = path_output.joinpath('separate_images')
#         output_dir.mkdir(parents=True, exist_ok=True)
#
#         _save_subplots_individually(fig, ax, titles, output_dir=output_dir, dpi=300)
#     if show:
#         plt.show()
#
#     if path_output:
#         # save the best image
#         fig = plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
#         ax = fig.add_subplot(111)
#         ax.imshow(np.zeros_like(green_u8), cmap="gray", vmin=0, vmax=1)  # gray black canvas
#         g_boost = np.clip(_normalize01(green_u8) * 1.2, 0, 1)  # scale and cli
#         ax.imshow(_normalize01(g_boost), cmap="Greens", alpha=0.9)
#         ax.imshow(_normalize01(red_u8), cmap="Reds", alpha=0.6)
#         ax.axis("off")
#         plt.tight_layout()
#         plt.savefig(path_output.joinpath(f'{cell_id}_overlay_green_red.png'), bbox_inches="tight", pad_inches=0, dpi=300)
#         plt.show()
#         plt.close()
#
#         fig = plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
#         ax = fig.add_subplot(111)
#         ax.imshow(img_green_color, alpha=0.9)
#         edges_rgba = _edges_to_rgba(g_edges, hex_color="#FF00FF", alpha=0.8)
#         ax.imshow(edges_rgba)
#         ax.imshow(heatmap_binary_gold_rgba)
#         ax.axis("off")
#         ax.set_title("Overlap Binary", fontsize=12)
#         plt.tight_layout()
#         plt.savefig(path_output.joinpath(f'{cell_id}_overlay_green_red_binary.png'), bbox_inches="tight", pad_inches=0, dpi=300)
#         plt.show()
#         plt.close()
#
#     return metrics, components
