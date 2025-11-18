"""
A module for computing quantitative metrics related to colocalization between red
and green vascular networks in biomedical images.

This module provides a set of computational metrics, including correlation, overlap
coefficients, intensity-weighted fractions, and distributional similarity, to assess
signal colocalization and spatial patterns of image regions. All metrics are designed
to support biological research, where understanding structural and intensity-based
relationships is critical.

Functions
---------

_compute_overlap_metrics:
    Computes Manders coefficients, Pearson correlation, Dice, and Jaccard indices
    for provided overlap masks and skeletonized masks.

_frac:
    Computes the fraction of true values in a binary mask relative to the total image area.

_compute_coloc_metrics_ml:
    Computes a comprehensive set of colocalization metrics for machine learning or
    comparative analysis, including intensity sums, weighted overlaps, cosine similarity,
    histogram overlaps, and mutual information.

Notes
-----
The module heavily utilizes mathematical libraries like NumPy and employs specialized
libraries for image processing and intensity analysis.
"""
import numpy as np
from library.ImageOverlap.display import *
from library.ImageOverlap.utils import _manders_coeffs
import cv2
from scipy.ndimage import distance_transform_edt as edt
import pandas as pd
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
import mahotas
from typing import Tuple, Dict
from matplotlib import colors


def _compute_overlap_metrics(
    G: np.ndarray,
    R: np.ndarray,
    g_skel: np.ndarray,
    r_skel: np.ndarray,
    mask_overlap: np.ndarray,
    vessel_mask: np.ndarray,
    label: str = "intensity",
) -> dict:
    """
    Compute intensity- and geometry-based colocalization metrics.

    Parameters
    ----------
    G, R : np.ndarray
        Green and red intensity images (uint8).
    g_skel, r_skel : np.ndarray
        Binary skeletonized vessel masks.
    mask_overlap : np.ndarray
        Overlap map (float for intensity-weighted, bool for binary).
    vessel_mask : np.ndarray
        Combined vessel area (union of both channels).
    label : str
        Label suffix ('intensity' or 'binary').

    Returns
    -------
    dict
        Dictionary of quantitative overlap metrics.
    """

    # --- Convert intensity masks safely ---
    mask_f = mask_overlap.astype(np.float32)
    mask_b = mask_overlap > 0

    # --- Manders coefficients (fraction of signal overlapping) ---
    m1, m2 = _manders_coeffs(G.astype(np.float32),
                             R.astype(np.float32),
                             mask_f)

    # --- Pearson correlation restricted to overlap region ---
    pearson = (np.corrcoef(G[mask_b].ravel(), R[mask_b].ravel())[0, 1]
               if mask_b.sum() > 10 else np.nan)

    # --- Binary overlap measures ---
    inter = np.logical_and(g_skel, r_skel).sum()
    union = np.logical_or(g_skel, r_skel).sum() + 1e-8
    dice = 2 * inter / (g_skel.sum() + r_skel.sum() + 1e-8)
    jaccard = inter / union
    overlap_coeff = inter / (min(g_skel.sum(), r_skel.sum()) + 1e-8)
    volume_similarity = 1 - abs(g_skel.sum() - r_skel.sum()) / (g_skel.sum() + r_skel.sum() + 1e-8)

    # --- Intensity-based continuous overlap ---
    numerator = np.sum(G * R)
    tanimoto = numerator / (np.sum(G**2) + np.sum(R**2) - numerator + 1e-8)
    cosine_sim = numerator / (np.sqrt(np.sum(G**2)) * np.sqrt(np.sum(R**2)) + 1e-8)

    # --- Area-normalized overlaps ---
    frac_overlap_vessel = mask_b.sum() / (vessel_mask.sum() + 1e-8)
    frac_overlap_green = mask_b.sum() / (G.size + 1e-8)
    frac_overlap_red = mask_b.sum() / (R.size + 1e-8)

    return {
        f"Manders_M1_{label}": float(m1),
        f"Manders_M2_{label}": float(m2),
        f"Pearson_{label}": float(pearson),
        f"Dice_skeletons_{label}": float(dice),
        f"Jaccard_skeletons_{label}": float(jaccard),
        f"OverlapCoeff_{label}": float(overlap_coeff),
        f"VolumeSimilarity_{label}": float(volume_similarity),
        f"Tanimoto_{label}": float(tanimoto),
        f"CosineSimilarity_{label}": float(cosine_sim),
        f"FracOverlap_Vessel_{label}": float(frac_overlap_vessel),
        f"FracOverlap_Green_{label}": float(frac_overlap_green),
        f"FracOverlap_Red_{label}": float(frac_overlap_red)
    }


def _frac(mask:np.ndarray, N:int):
    return float(mask.sum()) / (N + 1e-8)


def _compute_coloc_metrics_ml(
    G:np.ndarray,
    R:np.ndarray,
    g_norm:np.ndarray,
    g_mask:np.ndarray,
    r_mask:np.ndarray,
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
    overlay:np.ndarray,
    mask_g_bernsen: np.ndarray,
    mask_r_bernsen: np.ndarray,
    overlap_bernsen: np.ndarray,
    mask_g_niblack: np.ndarray,
    mask_r_niblack: np.ndarray,
    overlap_niblack: np.ndarray,
    ci_thresh: Optional[float] = 0.5,
    assd_thresh: Optional[float] = 2,
    show:Optional[bool] = False,
):
    """
    Calculates a comprehensive set of colocalization metrics between two image channels, typically used
    in microscopy signal analysis. This includes global intensity metrics, overlap measures,
    distributional comparisons, and threshold-centered binary metrics to assess the spatial and
    intensity relationships between the green (G) and red (R) channels.

    These metrics are crucial in quantifying characteristics like channel energy, spatial agreement,
    signal distributional similarity, and binary overlap precision.

    :param G: 2D array representing the green channel image intensities.
    :param R: 2D array representing the red channel image intensities.
    :param g_norm: Normalized green channel pixel intensities post preprocessing.
    :param g_mask: Binary mask of green channel values.
    :param r_mask: Binary mask of red channel values.
    :param g_clean: Cleaned green channel intensities after noise removal or postprocessing.
    :param r_norm: Normalized red channel pixel intensities post preprocessing.
    :param r_clean: Cleaned red channel intensities after noise removal or postprocessing.
    :param g_v: Green channel intensity after vessel-detection enhancement.
    :param r_v: Red channel intensity after vessel-detection enhancement.
    :param g_skel: Skeletonized (thinned) representation of the green channel vessels.
    :param r_skel: Skeletonized (thinned) representation of the red channel vessels.
    :param vessel_mask: Binary mask indicating vessel regions detected across channels.
    :param near_overlap: Refined overlap metric approximating closeness between vessel boundaries.
    :param overlap_strength: Combined pixelwise product of green and red intensities to assess strong overlaps.
    :param heatmap_intensity: Heatmap denoting intensity-based channel overlaps for G and R.
    :param heatmap_binary: Binary region map identifying designated overlap areas in regions of interest.
    :param overlay: Visualization array combining both channels for easier inspection of overlaps.
    :param ci_thresh: Contrast intensity threshold for certain binary metrics (default 0.5).
    :param assd_thresh: Average symmetric surface distance threshold for binary mask comparison (default 2).
    :param mask_g_bernsen: Binary mask for the green channel produced using the Bernsen thresholding method.
    :type mask_g_bernsen: np.ndarray
    :param mask_r_bernsen: Binary mask for the red channel produced using the Bernsen thresholding method.
        :type mask_r_bernsen: np.ndarray
    :param overlap_bernsen: Binary mask representing the overlap of green and red channels under the Bernsen thresholding method.
        :type overlap_bernsen: np.ndarray
    :param mask_g_niblack: Binary mask for the green channel produced using the Niblack thresholding method.
        :type mask_g_niblack: np.ndarray
    :param mask_r_niblack: Binary mask for the red channel produced using the Niblack thresholding method.
        :type mask_r_niblack: np.ndarray
    :param overlap_niblack: Binary mask representing the overlap of green and red channels under the Niblack thresholding method.
        :type overlap_niblack: np.ndarray
    :return: Dictionary containing various colocalization metrics categorized into multiple domains.

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
    # 8b. LOCAL THRESHOLD METRICS (Bernsen, Niblack)
    # --------------------------------------------------------------
    # Local adaptive segmentation applied to raw intensity images.
    # Captures vessel density under spatially varying illumination.
    # ==============================================================

    try:
        H, W = mask_g_bernsen.shape
        N = H * W
        metrics_local = {
            "frac_green_bernsen": _frac(mask=mask_g_bernsen, N=N),
            "frac_red_bernsen": _frac(mask=mask_r_bernsen, N=N),
            "frac_overlap_bernsen": _frac(mask=overlap_bernsen, N=N),
            "frac_green_niblack": _frac(mask=mask_g_niblack, N=N),
            "frac_red_niblack": _frac(mask=mask_r_niblack, N=N),
            "frac_overlap_niblack": _frac(mask=overlap_niblack, N=N),
        }
    except Exception as e:
        metrics_local = {"local_threshold_error": str(e)}

    # ==============================================================
    # 8c. GEOMETRIC METRICS
    # --------------------------------------------------------------
    #  Deterministic geometric classification of vessel colocalization
    # ==============================================================
    # --- Containment index ---
    try:
        intersection = np.logical_and(g_mask, r_mask)
        ci = intersection.sum() / (r_mask.sum() + 1e-8)

        # --- Distance maps ---
        da = edt(~g_mask)  # distance to nearest green vessel
        db = edt(~r_mask)  # distance to nearest red vessel
        # --- Deterministic metrics ---
        assd = (da[r_mask].mean() + db[g_mask].mean()) / 2
        # hd95 = np.percentile(np.concatenate([da[r_mask], db[g_mask]]), 95)

        # --- Decision ---
        # significant = (ci >= ci_thresh) and (hd95 <= hd_thresh) and (assd <= assd_thresh)
        significant = (ci >= ci_thresh) and (assd <= assd_thresh)
        classification = "Significant" if significant else "Weak"

        metrics_assd = {
            "containment_index": float(ci),
            "assd": float(assd),
            # "hausdorff95": float(hd95),
            "significant_overlap": significant,
            "classification": classification
        }
        masked_da = np.where(r_mask, da, np.nan)  # show distances only at red pixels
        masked_db = np.where(g_mask, db, np.nan)  # show distances only at green pixels
    except Exception as e:
        metrics_assd = {"geometry_classification_error": str(e)}

    # --- Optional visualization ---
    if show:
        # The bright yellow regions indicate far-apart structure
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].imshow(g_mask + r_mask, cmap="gray")
        ax[0].set_title("Vessel masks (G + R)")
        ax[0].axis("off")

        # plot_overlap_binary_edges(ax[1], img_green_color,
        #                           g_edges,
        #                           heatmap_binary_gold_rgba,)


        im1 = ax[1].imshow(masked_da, cmap="plasma")
        ax[1].set_title("Distance → nearest green")
        ax[1].axis("off")
        ax[1].contour(masked_da, levels=[10, 20, 50], colors='white', linewidths=0.5)
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="px")

        im2 = ax[2].imshow(masked_db, cmap="plasma")
        ax[2].set_title("Distance → nearest red")
        ax[2].axis("off")
        ax[2].contour(masked_db, levels=[10, 20, 50], colors='white', linewidths=0.5)
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, label="px")

        fig.suptitle(f"CI={ci:.2f}, ASSD={assd:.2f}, {classification}", fontsize=11)
        plt.tight_layout()
        plt.show()

    # ==============================================================
    # 8d. LOCAL METRICS FROM BERNSEN AND NIBLACK METHODS
    # --------------------------------------------------------------
    #     Computes local metrics for image masks using predefined methods. The local
    #     metrics computed include the fractional coverage of green and red masks
    #     as well as their overlap for Bernsen and Niblack methods. The function
    #     analyzes the input image masks and calculates fractional metrics to provi
    # ==============================================================




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

    metrics =  {
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

    metrics.update(metrics_local)
    metrics.update(metrics_assd)

    return metrics




def classify_from_metrics(df_metrics: pd.DataFrame,
                          ci_thresh: float = 0.5,
                          dice_thresh: float = 0.6,
                          assd_thresh: float = 5,
                          hd_thresh: float = 10):
    """
    Classify vessel overlap significance from precomputed metrics dataframe.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        DataFrame containing at least the following keys:
        - containment_index or overlap_pct_red
        - Dice_skeletons_intensity or Dice_skeletons_binary
        - assd_skeleton
        - hausdorff_skeleton
    ci_thresh, dice_thresh, assd_thresh, hd_thresh : float
        Thresholds for containment, Dice, ASSD, and Hausdorff metrics.

    Returns
    -------
    dict
        Classification summary with extracted metric values.
    """

    # --- Retrieve relevant metrics safely ---
    def get_metric(name_list):
        for name in name_list:
            row = df_metrics.loc[df_metrics["Metric"].str.contains(name, case=False, regex=True)]
            if not row.empty:
                return float(row["Value"].values[0])
        return np.nan

    ci = get_metric(["containment_index", "overlap_pct_red"])
    dice = get_metric(["dice", "Dice_skeletons_intensity", "Dice_skeletons_binary"])
    jaccard = get_metric(["jaccard"])
    assd = get_metric(["assd_skeleton"])
    hd = get_metric(["hausdorff_skeleton"])
    pearson = get_metric(["pearson_intensity", "pearson_binary"])

    # --- Classification rules ---
    overlap_good = (ci >= ci_thresh) or (dice >= dice_thresh) or (jaccard >= 0.5)
    geometry_good = (assd <= assd_thresh) or (hd <= hd_thresh)

    if overlap_good and geometry_good:
        classification = "Significant"
    elif overlap_good and not geometry_good:
        classification = "Misaligned"
    elif not overlap_good and geometry_good:
        classification = "Sparse alignment"
    else:
        classification = "Weak"

    return {
        "containment_index": ci,
        "dice": dice,
        "jaccard": jaccard,
        "assd": assd,
        "hausdorff95": hd,
        "pearson_corr": pearson,
        "classification": classification
    }


