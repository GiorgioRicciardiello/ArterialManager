import cv2
import numpy as np
from matplotlib import pyplot as plt
from config.config import CONFIG
import matplotlib.gridspec as gridspec
from typing import Dict, Tuple, List
from pathlib import Path

def to_green(img_gray: np.ndarray) -> np.ndarray:
    """Map grayscale intensities to green channel."""
    img_rgb = np.zeros((*img_gray.shape, 3), dtype=np.uint8)
    img_rgb[:, :, 1] = img_gray
    return img_rgb

def to_red(img_gray: np.ndarray) -> np.ndarray:
    """Map grayscale intensities to red channel."""
    img_rgb = np.zeros((*img_gray.shape, 3), dtype=np.uint8)
    img_rgb[:, :, 2] = img_gray
    return img_rgb


def save_images(img_dict, out_dir="outputs", color="green"):
    """
    Save processed images from dictionary as PNGs with chosen color mapping.
    Supports grayscale, RGB, and RGBA.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for title, img in img_dict.items():
        # Sanitize key for filename
        fname = (
            title.replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
                 .replace("=", "")
                 .replace(",", "")
        )
        path = out_dir / f"{fname}.png"

        # If grayscale -> colorize
        if len(img.shape) == 2:
            if color.lower() == "green":
                img_rgb = to_green(img)
            elif color.lower() == "red":
                img_rgb = to_red(img)
            else:
                raise ValueError("color must be 'green' or 'red'")
            cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        elif img.shape[2] == 3:  # RGB
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        elif img.shape[2] == 4:  # RGBA
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))

        print(f"[INFO] Saved: {path}")


def plot_images_with_hist(img_dict:Dict[str, np.ndarray],
                          figsize:Tuple[int, int]=(14,8)):
    """
    Pair plot of the image and the histogram to evaluate the methods
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, len(img_dict), height_ratios=[1, 1])

    for i, (title, im) in enumerate(img_dict.items()):
        # Image
        ax_img = fig.add_subplot(gs[0, i])
        if len(im.shape) == 2:  # grayscale
            ax_img.imshow(to_green(im))
        elif im.shape[2] == 3:  # RGB
            ax_img.imshow(im)
        elif im.shape[2] == 4:  # RGBA
            ax_img.imshow(im)
        ax_img.set_title(title)
        ax_img.axis("off")

        # Histogram
        ax_hist = fig.add_subplot(gs[1, i])
        if len(im.shape) == 2:
            ax_hist.hist(im.ravel(), bins=256, color='green', alpha=0.7)
        elif im.shape[2] == 4:
            # Show histogram of alpha (transparency)
            ax_hist.hist(im[:, :, 3].ravel(), bins=256, color='gray', alpha=0.7)
            ax_hist.set_title(f"{title} Alpha Histogram")
        else:
            ax_hist.hist(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).ravel(),
                         bins=256, color='green', alpha=0.7)
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Pixel Count (log)")
        ax_hist.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


#
# def plot_images_with_hist(img_dict:Dict[str, np.ndarray],
#                           figsize:Tuple[int, int]=(16,10),
#                           img_height_ratio:int=3,
#                           hist_height_ratio:int=1):
#     """
#     Pair plot of the image and the histogram to evaluate the methods.
#
#     Parameters
#     ----------
#     img_dict : dict
#         Dictionary of {title: image}.
#     figsize : tuple
#         Overall figure size.
#     img_height_ratio : int
#         Relative height for image subplots.
#     hist_height_ratio : int
#         Relative height for histogram subplots.
#     """
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(2, len(img_dict),
#                            height_ratios=[img_height_ratio, hist_height_ratio])
#
#     for i, (title, im) in enumerate(img_dict.items()):
#         # --- Image ---
#         ax_img = fig.add_subplot(gs[0, i])
#         # if len(im.shape) == 2:  # grayscale
#         #     ax_img.imshow(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
#         # elif im.shape[2] == 3:  # RGB
#         ax_img.imshow(im)
#         # elif im.shape[2] == 4:  # RGBA
#         #     ax_img.imshow(im)
#         ax_img.set_title(title)
#         ax_img.axis("off")
#
#         # --- Histogram ---
#         ax_hist = fig.add_subplot(gs[1, i])
#         if len(im.shape) == 2:
#             ax_hist.hist(im.ravel(), bins=256, color='green', alpha=0.7)
#         elif im.shape[2] == 4:
#             # Show histogram of alpha (transparency)
#             ax_hist.hist(im[:, :, 3].ravel(), bins=256, color='gray', alpha=0.7)
#             ax_hist.set_title(f"{title} Alpha Histogram")
#         else:
#             ax_hist.hist(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).ravel(),
#                          bins=256, color='green', alpha=0.7)
#         ax_hist.set_yscale("log")
#         ax_hist.set_xlabel("Intensity")
#         ax_hist.set_ylabel("Pixel Count (log)")
#         ax_hist.grid(True, alpha=0.4)
#
#     plt.tight_layout()
#     plt.show()


def translucent_artery_misty(img,
                             blur_ksize=7,
                             max_val=180,
                             alpha_inside=0.4,
                             alpha_edges=0.7):
    """
    Make artery structures look like a translucent mist.
    Interiors are softened, edges slightly stronger.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (arteries bright, background dark).
    blur_ksize : int
        Gaussian blur kernel size (odd number).
    max_val : int
        Maximum intensity cap.
    alpha_inside : float
        Transparency factor for artery interiors (0=transparent, 1=opaque).
    alpha_edges : float
        Transparency factor for artery borders (stronger than interior).

    Returns
    -------
    img_misty : np.ndarray
        Grayscale image with misty translucent arteries.
    """

    # Step 1: Vessel mask via Otsu threshold
    _, mask = cv2.threshold(img, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Blur to soften interior
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # Step 3: Edge detection
    edges = cv2.Canny(img, 50, 120)

    # Step 4: Weighted combination
    img_flat = np.zeros_like(img, dtype=np.float32)
    img_flat[mask > 0] = img_blur[mask > 0] * alpha_inside
    img_flat[edges > 0] = np.maximum(img_flat[edges > 0],
                                     img[edges > 0] * alpha_edges)

    # Step 5: Cap and normalize
    img_capped = np.minimum(img_flat, max_val)
    img_misty = cv2.normalize(img_capped, None,
                              0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_misty


def translucent_artery_misty_alpha(img,
                             blur_ksize=7,
                             max_val=180,
                             alpha_min=0.1,
                             alpha_max=0.7,
                             density_ksize=15):
    """
    Make artery structures look like a translucent mist with adaptive transparency.
    Alpha is inversely proportional to local pixel density.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (arteries bright, background dark).
    blur_ksize : int
        Gaussian blur kernel size (odd number).
    max_val : int
        Maximum intensity cap.
    alpha_min : float
        Minimum transparency factor (most transparent).
    alpha_max : float
        Maximum transparency factor (least transparent).
    density_ksize : int
        Kernel size for local density estimation (odd number).

    Returns
    -------
    rgba : np.ndarray
        RGBA image where alpha varies adaptively by density.
    """

    # --- Step 1: Vessel mask (Otsu threshold)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 2: Smooth interiors (blur for mist effect)
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # --- Step 3: Local density map (mean filter over neighborhood)
    density = cv2.blur(img.astype(np.float32), (density_ksize, density_ksize))

    # Normalize density 0–1
    density_norm = cv2.normalize(density, None, 0, 1.0, cv2.NORM_MINMAX)

    # --- Step 4: Compute adaptive alpha
    # More density = lower alpha (more transparent)
    alpha_map = alpha_max - density_norm * (alpha_max - alpha_min)

    # Apply mask → background alpha = 0
    alpha_map[mask == 0] = 0

    # Scale to 0–255
    alpha_channel = (alpha_map * 255).astype(np.uint8)

    # --- Step 5: Cap and normalize intensity for visual channel
    img_capped = np.minimum(img_blur, max_val)
    img_final = cv2.normalize(img_capped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- Step 6: Build RGBA image
    rgba = np.zeros((*img.shape, 4), dtype=np.uint8)
    rgba[:, :, 1] = img_final     # Green channel
    rgba[:, :, 3] = alpha_channel # Adaptive transparency

    return rgba


def translucent_artery_rgba(img, blur_ksize=7, alpha_inside=0.2, alpha_edges=0.6):
    """
    Return arteries as RGBA image with real transparency.
    """

    # Otsu mask
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Smooth interiors
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # Extract edges
    edges = cv2.Canny(img, 50, 120)

    # Build alpha map
    alpha = np.zeros_like(img, dtype=np.float32)
    alpha[mask > 0] = alpha_inside
    alpha[edges > 0] = alpha_edges

    # Normalize to 0–255
    alpha_channel = (alpha * 255).astype(np.uint8)

    # Green channel for arteries
    g_channel = img_blur.astype(np.uint8)

    # RGBA: background black + transparency
    rgba = np.zeros((*img.shape, 4), dtype=np.uint8)
    rgba[:, :, 1] = g_channel   # Green
    rgba[:, :, 3] = alpha_channel  # Alpha

    return rgba


def process_image(img,
                  alpha=0.4,
                  clip_percentile=99,
                  gamma=3.0,
                  include_misty=True,
                  include_rgba=True):
    """
    Evaluate different processing methods (contrast reduction, translucency),
    ignoring the black background using a vessel mask.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image (arteries bright, background black).
    alpha : float
        Global scaling for "Uniform + Translucent".
    clip_percentile : int
        Percentile clipping for "Uniform + Translucent".
    gamma : float
        Gamma correction factor.
    percentile : int
        For percentile rescaling (if enabled).
    include_misty : bool
        If True, also compute misty low-pass version.
    include_rgba : bool
        If True, also compute RGBA transparency version.

    Returns
    -------
    results : dict
        Dictionary of {method_name: processed_image}.
        Images may be grayscale, RGB, or RGBA depending on method.
    """
    results = {}

    # --- Step 0: Vessel mask (ignore black background)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Helper to apply mask safely
    def apply_mask(arr):
        out = np.zeros_like(img, dtype=np.uint8)
        out[mask > 0] = arr[mask > 0]
        return out

    # --- Method 0: Original
    results["Original"] = img

    # --- Method 1: Gamma correction
    img_gamma = np.array(255 * (img / 255.0) ** gamma, dtype=np.uint8)
    results[f"Gamma (γ={gamma})"] = apply_mask(img_gamma)

    # --- Method 2: Histogram Equalization
    img_eq = cv2.equalizeHist(img)
    results["Equalized"] = apply_mask(img_eq)

    # --- Method 3: Uniform + Translucent
    img_translucent = cv2.convertScaleAbs(img_eq, alpha=alpha, beta=0)
    clip_val = np.percentile(img_translucent[mask > 0], clip_percentile)  # only vessels
    img_uniform = np.clip(img_translucent, 0, clip_val)
    img_uniform = cv2.normalize(img_uniform, None, 0, 255, cv2.NORM_MINMAX)
    results[f"Uniform + Translucent (α={alpha}, p={clip_percentile})"] = apply_mask(img_uniform)

    # --- Optional Method 4: Misty (low-pass translucent)
    if include_misty:
        misty = translucent_artery_misty(img,
                                         blur_ksize=7,
                                         max_val=150,
                                         alpha_inside=0.2,
                                         alpha_edges=0.6)
        results["LPF Misty"] = apply_mask(misty)

        misty_local = translucent_artery_misty_alpha(img,
                                           blur_ksize=7,
                                           max_val=150,
                                           alpha_min=0.1,
                                           alpha_max=0.7,
                                           density_ksize=15)

        def apply_mas_mink(arr):
            return np.where(mask > 0, arr, 0).astype(np.uint8)
        results["LPF Misty Local"] = apply_mas_mink(misty_local)


    # --- Optional Method 5: True RGBA transparency
    if include_rgba:
        rgba = translucent_artery_rgba(img,
                                       blur_ksize=7,
                                       alpha_inside=0.2,
                                       alpha_edges=0.6)
        results["LPF Transl (RGBA)"] = rgba  # already has alpha → no need apply_mask

    return results

def generate_red_dots(img, num_dots=200, dot_radius=2, out_path="red_dots.png"):
    """
    Generate an image with only red dots inside the vessel mask of 'img'.
    Vessels themselves are removed (black background).

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image (arteries bright, background dark).
    num_dots : int
        Number of random dots to draw.
    dot_radius : int
        Radius of each dot in pixels.
    out_path : str or Path
        Path to save the output PNG.

    Returns
    -------
    out_img : np.ndarray
        RGB image with only red dots (black background).
    """
    # Step 1: Vessel mask (Otsu threshold)
    _, mask = cv2.threshold(img, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Collect valid vessel coordinates
    ys, xs = np.where(mask > 0)
    coords = list(zip(xs, ys))

    if len(coords) == 0:
        raise ValueError("Mask is empty, no vessels detected!")

    # Step 3: Random sample points
    idx = np.random.choice(len(coords),
                           size=min(num_dots, len(coords)),
                           replace=False)
    chosen_points = [coords[i] for i in idx]

    # Step 4: Create black RGB canvas
    h, w = img.shape
    out_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Step 5: Draw red dots
    for (x, y) in chosen_points:
        cv2.circle(out_img, (x, y), dot_radius, (0, 0, 255), -1)  # BGR: red

    # Step 6: Save PNG
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img)

    print(f"[INFO] Saved {len(chosen_points)} red dots at: {out_path}")
    return out_img



def mistify_vessels(img,
                    blur_ksize=5,
                    sat_thresh=180,
                    del_fraction=0.3,
                    max_val=200,
                    alpha_inside=0.3,
                    alpha_edges=0.7):
    """
    Make vessels misty by removing saturated patches.

    Parameters
    ----------
    img : np.ndarray
        Grayscale vessel image.
    blur_ksize : int
        Gaussian blur size for smoothing.
    sat_thresh : int
        Intensity above which a contour is considered saturated.
    del_fraction : float
        Fraction of pixels randomly deleted in saturated areas.
    max_val : int
        Cap for vessel brightness.
    alpha_inside : float
        Transparency factor inside contours.
    alpha_edges : float
        Transparency factor at vessel edges.

    Returns
    -------
    rgba : np.ndarray
        RGBA misty image with transparency applied.
    """

    # --- Step 1: Vessel mask (binary)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 2: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Step 3: Copy original for modification
    mod = img.copy()

    for cnt in contours:
        # Bounding box for contour
        x, y, w, h = cv2.boundingRect(cnt)
        roi = mod[y:y+h, x:x+w]

        # Measure mean intensity
        mean_val = cv2.mean(roi, mask[y:y+h, x:x+w])[0]

        # If saturated: randomly delete pixels
        if mean_val > sat_thresh:
            rand_mask = np.random.rand(h, w) < del_fraction
            roi[rand_mask] = 0
            mod[y:y+h, x:x+w] = roi

    # --- Step 4: Smooth the modified image
    mod_blur = cv2.GaussianBlur(mod, (blur_ksize, blur_ksize), 0)

    # --- Step 5: Edge detection
    edges = cv2.Canny(mod, 50, 120)

    # --- Step 6: Build RGBA with adaptive alpha
    alpha = np.zeros_like(img, dtype=np.float32)
    alpha[mask > 0] = alpha_inside
    alpha[edges > 0] = alpha_edges
    alpha_channel = (alpha * 255).astype(np.uint8)

    # Cap intensity
    mod_capped = np.minimum(mod_blur, max_val)
    mod_norm = cv2.normalize(mod_capped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Build RGBA (green vessel, transparent background)
    rgba = np.zeros((*img.shape, 4), dtype=np.uint8)
    rgba[:, :, 1] = mod_norm
    rgba[:, :, 3] = alpha_channel

    return rgba


import cv2
import numpy as np

def misty_hollow_preserve(original_img, density_thresh=0.5, removal_prob=0.3, blur_ksize=7):
    """
    Preserve the original image but hollow out dense vessel interiors.
    Hollowing is blended back into the original instead of overwriting everything.

    Parameters
    ----------
    original_img : np.ndarray
        Input grayscale vessel image.
    density_thresh : float
        Threshold (0–1). Contours above this density will be hollowed.
    removal_prob : float
        Probability of randomly deleting pixels in dense contours.
    blur_ksize : int
        Gaussian blur kernel size for mist effect.

    Returns
    -------
    misty_preserve : np.ndarray
        Image with misty hollow regions but preserved original elsewhere.
    """

    # Step 1: Vessel mask
    _, mask = cv2.threshold(original_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy original to preserve most areas
    misty_preserve = original_img.copy()

    # Step 3: Work only inside dense contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = mask[y:y+h, x:x+w].copy()

        # Density of filled region
        density = np.sum(roi > 0) / (w * h + 1e-6)

        if density > density_thresh:
            # Random deletion (hollowing)
            noise = (np.random.rand(h, w) < removal_prob).astype(np.uint8) * 255
            hollow = cv2.bitwise_and(roi, cv2.bitwise_not(noise))

            # Keep edges stronger
            edges = cv2.Canny(roi, 50, 150)
            hollow = cv2.bitwise_or(hollow, edges)

            # Blur hollow effect
            if blur_ksize > 1:
                hollow = cv2.GaussianBlur(hollow, (blur_ksize, blur_ksize), 0)

            # Only overwrite pixels where we hollowed, rest stays original
            region = misty_preserve[y:y+h, x:x+w]
            region[hollow > 0] = hollow[hollow > 0]
            misty_preserve[y:y+h, x:x+w] = region

    return misty_preserve





def misty_hsv_alpha(img,
                    sat_thresh=100,
                    sat_scale=0.7,
                    val_thresh=50,
                    blur_ksize=5):
    """
    Make a misty translucent effect in HSV space.
    Transparency (alpha) is adapted from saturation & brightness.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale or RGB image. If grayscale, will be expanded to RGB.
    sat_thresh : int
        Saturation threshold above which we start hollowing.
    sat_scale : float
        Factor to reduce alpha in high-saturation regions (0–1).
    val_thresh : int
        Optional brightness cutoff to suppress very dark areas.
    blur_ksize : int
        Kernel size for smoothing the alpha channel.

    Returns
    -------
    rgba : np.ndarray
        Misty RGBA image with adaptive alpha.
    """

    # --- Step 1: Ensure RGB
    if len(img.shape) == 2:  # grayscale
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = img.copy()

    # --- Step 2: Convert to HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # --- Step 3: Build adaptive alpha
    # Start fully opaque where V > val_thresh
    alpha = np.where(V > val_thresh, 1.0, 0.0).astype(np.float32)

    # Reduce alpha in highly saturated regions
    sat_factor = np.clip((S.astype(np.float32) / 255.0), 0, 1)
    alpha *= (1 - sat_factor * sat_scale)

    # Smooth alpha (mist effect)
    alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)

    # Scale alpha to 0–255
    alpha_channel = (alpha * 255).astype(np.uint8)

    # --- Step 4: Combine into RGBA (keeping RGB as is)
    rgba = np.dstack([rgb, alpha_channel])

    return rgba


if __name__ == "__main__":
    path_img = CONFIG["paths"]['data'].joinpath('imgs', 'Bad', 'AD1_5.1_C=0.jpg')
    path_out = CONFIG['paths']['outputs_imgs']
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)


    img_dict = process_image(img,
                  alpha=0.3,
                  clip_percentile=80,
                  gamma=3.0,
                  include_misty=True,
                  include_rgba=True)


    # Plot
    plot_images_with_hist(img_dict, figsize=(22, 6))


    save_images(img_dict, out_dir=path_out)

    img_dict = {}
    img_dict['original_img'] = img

    misty_rgba = mistify_vessels(img,
                                 blur_ksize=7,
                                 sat_thresh=170,
                                 del_fraction=0.4)
    img_dict['misty_rgba'] = misty_rgba

    img_dict['misty_by_contours'] =  misty_hollow_preserve(
                                        img,
                                        density_thresh=0.45,   # lower threshold → more areas hollowed
                                        removal_prob=0.85,     # high probability → more deletion
                                        blur_ksize=5           # smoother mist effect
                                    )


    plot_images_with_hist(img_dict, figsize=(16, 6))


    # %% HSV test
    # Read image in color
    img_bgr = cv2.imread(str(path_img), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {path_img}")

    # Convert to grayscale for mask generation
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold to get mask of bright (green) regions
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to RGBA
    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)


    # Function to adjust alpha only on mask
    def apply_alpha(img_rgba, mask, alpha_scale=0.5):
        """ Apply alpha intensity only to mask regions """
        img_out = img_rgba.copy()
        img_out[:, :, 3] = 255  # default opaque
        img_out[mask > 0, 3] = int(255 * alpha_scale)  # apply alpha to masked areas
        return img_out


    # Try with different alpha values
    alphas = [1.0, 0.7, 0.4, 0.1]
    results = [apply_alpha(img_rgba, mask, a) for a in alphas]

    # Plot results
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(16, 6))

    # Original
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Alpha variations
    for i, (res, a) in enumerate(zip(results, alphas)):
        axes[i + 1].imshow(cv2.cvtColor(res, cv2.COLOR_BGRA2RGBA))
        axes[i + 1].set_title(f"Alpha = {a}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()

    # %%
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load image
    img_bgr = cv2.imread(str(path_img), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {path_img}")

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- Full vessel mask (Otsu threshold) ---
    _, mask_full = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Adaptive high-intensity mask (percentile) ---
    vals = img_gray[img_gray > 0]  # ignore background
    thresh_val = np.percentile(vals, 90)  # keep top 10% brightest
    _, mask_high = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)

    print(f"Adaptive threshold (90th percentile) = {thresh_val}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask_full, cmap='gray')
    axes[1].set_title("Full Vessel Mask (Otsu)")
    axes[1].axis("off")

    axes[2].imshow(mask_high, cmap='gray')
    axes[2].set_title("High-Intensity Mask (Top 10%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


    # %%
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load image
    img_bgr = cv2.imread(str(path_img), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive high-intensity mask (top 10%)
    vals = img_gray[img_gray > 0]
    thresh_val = np.percentile(vals, 90)
    _, mask_high = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)

    # ---- Create cloudy effect ----
    # Normalize mask to [0,1] and blur it
    mask_float = mask_high.astype(np.float32) / 255.0
    cloudy_mask = cv2.GaussianBlur(mask_float, (35, 35), 15)  # strong blur for mist

    # Expand mask to 3 channels
    cloudy_mask_3c = np.dstack([cloudy_mask] * 3)

    # Overlay with transparency (like fog)
    alpha = 0.1  # global opacity
    overlay = (img_rgb * (1 - alpha * cloudy_mask_3c) +
               np.array([0, 255, 0]) * (alpha * cloudy_mask_3c)).astype(np.uint8)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # axes[1].imshow(mask_high, cmap="gray")
    # axes[1].set_title("High-intensity Mask")
    # axes[1].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Cloudy / Misty Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # %% Filter method


    def donut_kernel(size=51, max_val=0.7):
        """
        Create a donut-shaped kernel:
        - Center = 0
        - Increase with radius up to max_val
        - Border = 0
        """
        k = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)

        # Normalize distance
        dist = dist / dist.max()

        # Build profile: center=0, mid radius= max_val, border=0
        profile = -4 * (dist - 0.5) ** 2 + 1  # parabola peaked at center
        profile = np.clip(profile, 0, 1)

        # Scale to max_val
        k = profile * max_val
        return k


    # Example kernel
    kernel = donut_kernel(101, max_val=0.2)

    plt.imshow(kernel, cmap="viridis")
    plt.colorbar()
    plt.title("Donut Kernel")
    plt.show()

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Step 1: Load image ---
    img_bgr = cv2.imread(str(path_img), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to grayscale for mask
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # High-intensity mask (top 10%)
    vals = img_gray[img_gray > 0]
    thresh_val = np.percentile(vals, 90)
    _, mask_high = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)

    # --- Step 2: Estimate contour thickness automatically ---
    contours, _ = cv2.findContours(mask_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
        perims = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 10]
        avg_thickness = int(np.mean([a / p if p > 0 else 1 for a, p in zip(areas, perims)]))
    else:
        avg_thickness = 15  # fallback

    kernel_size = max(15, avg_thickness * 4 | 1)  # make sure odd and large enough


    # --- Step 3: Create donut kernel ---
    def donut_kernel(size=51, max_val=0.2):
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        dist = dist / dist.max()

        # smooth bell-shaped profile (0 at center and border, peak mid-radius)
        profile = -4 * (dist - 0.5) ** 2 + 1
        profile = np.clip(profile, 0, 1)
        return (profile * max_val).astype(np.float32)


    kernel = donut_kernel(kernel_size, max_val=0.2)

    # --- Step 4: Apply subtraction ---
    filtered = cv2.filter2D(mask_high.astype(np.float32), -1, kernel)
    mask_sub = cv2.subtract(mask_high.astype(np.float32), filtered)
    mask_sub = np.clip(mask_sub, 0, 255).astype(np.uint8)

    # --- Step 5: Overlay effect on original image ---
    # make green mist from mask_sub
    cloudy_mask = cv2.GaussianBlur(mask_sub.astype(np.float32) / 255.0, (25, 25), 10)
    cloudy_mask_3c = np.dstack([cloudy_mask] * 3)

    alpha = 0.6
    overlay = (img_rgb * (1 - alpha * cloudy_mask_3c) +
               np.array([0, 255, 0]) * (alpha * cloudy_mask_3c)).astype(np.uint8)

    # --- Step 6: Show results ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_rgb);
    axes[0, 0].set_title("Original");
    axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_high, cmap="gray");
    axes[0, 1].set_title("High-Intensity Mask");
    axes[0, 1].axis("off")
    axes[0, 2].imshow(kernel, cmap="viridis");
    axes[0, 2].set_title(f"Donut Kernel (size={kernel_size})");
    axes[0, 2].axis("off")

    axes[1, 0].imshow(filtered, cmap="gray");
    axes[1, 0].set_title("Filtered Component");
    axes[1, 0].axis("off")
    axes[1, 1].imshow(mask_sub, cmap="gray");
    axes[1, 1].set_title("Mask After Subtraction");
    axes[1, 1].axis("off")
    axes[1, 2].imshow(overlay);
    axes[1, 2].set_title("Overlay on Original");
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


    # %%
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------
    # Parameters
    # -------------------
    alpha = 0.6  # transparency of mist overlay
    percentile_thresh = 90  # high-intensity threshold (top X% brightest)
    min_area_fraction = 0.05  # keep objects >5% of mean area

    # -------------------
    # Load and prepare
    # -------------------
    img_bgr = cv2.imread(path_img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {path_img}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # -------------------
    # High-intensity mask (percentile)
    # -------------------
    vals = img_gray[img_gray > 0]
    thresh_val = np.percentile(vals, percentile_thresh)
    _, mask_high = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)

    # -------------------
    # Preprocess mask (clean + close)
    # -------------------
    kernel_small = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_high, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # -------------------
    # Estimate vessel thickness for kernel size
    # -------------------
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    avg_thickness = 1  # default fallback
    if contours:
        valid_ratios = []
        for c in contours:
            a = cv2.contourArea(c)
            print(a)
            if a > 10:  # skip tiny speckles
                p = cv2.arcLength(c, True)
                if p > 0:
                    valid_ratios.append(a / p)
        if len(valid_ratios) > 0:
            avg_thickness = int(np.mean(valid_ratios))


    # -------------------
    # Donut kernel builder
    # -------------------
    def donut_kernel(size=51, max_val=0.2):
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        dist = dist / dist.max()
        profile = -4 * (dist - 0.5) ** 2 + 1  # parabola (peak mid-radius)
        profile = np.clip(profile, 0, 1)
        return (profile * max_val).astype(np.float32)


    kernel = donut_kernel(kernel_size, max_val=0.2)

    # -------------------
    # Apply convolution + safe subtraction
    # -------------------
    filtered = cv2.filter2D(mask_clean.astype(np.float32), -1, kernel)
    filtered_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, filtered_bin = cv2.threshold(filtered_norm, 30, 255, cv2.THRESH_BINARY)

    mask_sub = cv2.subtract(mask_clean, cv2.min(mask_clean, filtered_bin))

    # -------------------
    # Size filtering to keep only large structures
    # -------------------
    contours, _ = cv2.findContours(mask_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_sub)
    areas = [cv2.contourArea(c) for c in contours]
    if areas:
        min_area = min_area_fraction * np.mean(areas)
        for c in contours:
            if cv2.contourArea(c) > min_area:
                cv2.drawContours(mask_final, [c], -1, 255, -1)


    # -------------------
    # Overlay before vs after
    # -------------------
    def overlay_mask(image_rgb, mask, alpha=0.6, color=(0, 255, 0)):
        mask_float = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (25, 25), 10)
        mask_3c = np.dstack([mask_float] * 3)
        return (image_rgb * (1 - alpha * mask_3c) + np.array(color) * (alpha * mask_3c)).astype(np.uint8)


    overlay_before = overlay_mask(img_rgb, mask_clean, alpha=alpha, color=(0, 255, 0))
    overlay_after = overlay_mask(img_rgb, mask_final, alpha=alpha, color=(255, 0, 0))  # red for after

    # -------------------
    # Show results
    # -------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_rgb);
    axes[0, 0].set_title("Original");
    axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_clean, cmap="gray");
    axes[0, 1].set_title("Clean Mask Before");
    axes[0, 1].axis("off")
    axes[0, 2].imshow(kernel, cmap="viridis");
    axes[0, 2].set_title(f"Donut Kernel (size={kernel_size})");
    axes[0, 2].axis("off")

    axes[1, 0].imshow(mask_final, cmap="gray");
    axes[1, 0].set_title("Mask After Subtraction");
    axes[1, 0].axis("off")
    axes[1, 1].imshow(overlay_before);
    axes[1, 1].set_title("Overlay Before (Green)");
    axes[1, 1].axis("off")
    axes[1, 2].imshow(overlay_after);
    axes[1, 2].set_title("Overlay After (Red)");
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

    # %% Smart walk contours
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt


    def refine_contours(mask, img_shape):
        """
        Refine contours so that open vessel contours touching the black background
        are closed deterministically by bridging the shortest gap.
        """
        h, w = img_shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        refined_mask = np.zeros_like(mask)

        for cnt in contours:
            # Check if contour touches border
            touches_border = np.any(
                (cnt[:, 0, 0] <= 0) | (cnt[:, 0, 0] >= w - 1) |
                (cnt[:, 0, 1] <= 0) | (cnt[:, 0, 1] >= h - 1)
            )

            if not touches_border:
                # Already closed contour -> keep
                cv2.drawContours(refined_mask, [cnt], -1, 255, -1)
            else:
                # Find all contour points that lie on the border
                border_points = [tuple(pt[0]) for pt in cnt if
                                 pt[0][0] <= 0 or pt[0][0] >= w - 1 or pt[0][1] <= 0 or pt[0][1] >= h - 1]

                if len(border_points) >= 2:
                    # Compute pair with minimum distance
                    dists = []
                    for i in range(len(border_points)):
                        for j in range(i + 1, len(border_points)):
                            d = np.linalg.norm(np.array(border_points[i]) - np.array(border_points[j]))
                            dists.append((d, border_points[i], border_points[j]))

                    _, p1, p2 = min(dists, key=lambda x: x[0])

                    # Draw original contour
                    cv2.drawContours(refined_mask, [cnt], -1, 255, -1)

                    # Close contour with a line between closest border points
                    cv2.line(refined_mask, p1, p2, 255, thickness=1)
                    cv2.fillPoly(refined_mask, [np.array([p1, p2])], 255)

                else:
                    # fallback: just draw contour
                    cv2.drawContours(refined_mask, [cnt], -1, 255, -1)

        return refined_mask


    # -------------------------------
    # Example usage
    # -------------------------------
    # Assume mask_high is already defined from thresholding
    mask_refined = refine_contours(mask_high, img_gray.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(mask_high, cmap='gray');
    axes[0].set_title("Raw High-Intensity Mask");
    axes[0].axis("off")
    axes[1].imshow(mask_refined, cmap='gray');
    axes[1].set_title("Refined Contours");
    axes[1].axis("off")
    axes[2].imshow(img_gray, cmap='gray');
    axes[2].contour(mask_refined, colors='r');
    axes[2].set_title("Overlay");
    axes[2].axis("off")
    plt.show()

    # %% Smart walk full script
    # -------------------
    # Parameters
    # -------------------
    alpha = 0.6  # transparency of mist overlay
    percentile_thresh = 90  # high-intensity threshold (top X% brightest)
    min_area_fraction = 0.05  # keep objects >5% of mean area

    # -------------------
    # Load and prepare
    # -------------------
    img_bgr = cv2.imread(path_img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {path_img}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # -------------------
    # High-intensity mask (percentile)
    # -------------------
    vals = img_gray[img_gray > 0]
    thresh_val = np.percentile(vals, percentile_thresh)
    _, mask_high = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)

    # -------------------
    # Preprocess mask (clean + close)
    # -------------------
    kernel_small = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_high, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_small, iterations=2)


    # -------------------
    # Refinement: close open contours deterministically
    # -------------------
    def refine_contours(mask, img_shape):
        h, w = img_shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_mask = np.zeros_like(mask)

        for cnt in contours:
            touches_border = np.any(
                (cnt[:, 0, 0] <= 0) | (cnt[:, 0, 0] >= w - 1) |
                (cnt[:, 0, 1] <= 0) | (cnt[:, 0, 1] >= h - 1)
            )
            if not touches_border:
                cv2.drawContours(refined_mask, [cnt], -1, 255, -1)
            else:
                border_points = [tuple(pt[0]) for pt in cnt if
                                 pt[0][0] <= 0 or pt[0][0] >= w - 1 or pt[0][1] <= 0 or pt[0][1] >= h - 1]
                if len(border_points) >= 2:
                    dists = [(np.linalg.norm(np.array(p1) - np.array(p2)), p1, p2)
                             for i, p1 in enumerate(border_points) for p2 in border_points[i + 1:]]
                    _, p1, p2 = min(dists, key=lambda x: x[0])
                    cv2.drawContours(refined_mask, [cnt], -1, 255, -1)
                    cv2.line(refined_mask, p1, p2, 255, thickness=1)
                else:
                    cv2.drawContours(refined_mask, [cnt], -1, 255, -1)

        return refined_mask


    mask_refined = refine_contours(mask_clean, img_gray.shape)

    # -------------------
    # Size filtering (keep only large structures)
    # -------------------
    contours, _ = cv2.findContours(mask_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = np.zeros_like(mask_refined)
    areas = [cv2.contourArea(c) for c in contours]
    if areas:
        min_area = min_area_fraction * np.mean(areas)
        for c in contours:
            if cv2.contourArea(c) > min_area:
                cv2.drawContours(mask_filtered, [c], -1, 255, -1)

    mask_filtered = mask_refined
    # -------------------
    # Estimate vessel thickness for kernel size
    # -------------------
    avg_thickness = 1
    if contours:
        valid_ratios = []
        for c in contours:
            a = cv2.contourArea(c)
            if a > 10:
                p = cv2.arcLength(c, True)
                if p > 0:
                    valid_ratios.append(a / p)
        if len(valid_ratios) > 0:
            avg_thickness = int(np.mean(valid_ratios))

    kernel_size = max(15, (avg_thickness * 4) | 1)


    # -------------------
    # Donut kernel builder
    # -------------------
    def donut_kernel(size=51, max_val=0.2):
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        dist = dist / dist.max()
        profile = -4 * (dist - 0.5) ** 2 + 1
        profile = np.clip(profile, 0, 1)
        return (profile * max_val).astype(np.float32)


    kernel = donut_kernel(kernel_size, max_val=0.2)

    # -------------------
    # Apply convolution + safe subtraction
    # -------------------
    filtered = cv2.filter2D(mask_filtered.astype(np.float32), -1, kernel)
    filtered_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, filtered_bin = cv2.threshold(filtered_norm, 30, 255, cv2.THRESH_BINARY)

    mask_sub = cv2.subtract(mask_filtered, cv2.min(mask_filtered, filtered_bin))


    # -------------------
    # Overlay before vs after
    # -------------------
    def overlay_mask(image_rgb, mask, alpha=0.6, color=(0, 255, 0)):
        mask_float = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (25, 25), 10)
        mask_3c = np.dstack([mask_float] * 3)
        return (image_rgb * (1 - alpha * mask_3c) + np.array(color) * (alpha * mask_3c)).astype(np.uint8)


    overlay_before = overlay_mask(img_rgb, mask_filtered, alpha=alpha, color=(0, 255, 0))
    overlay_after = overlay_mask(img_rgb, mask_sub, alpha=alpha, color=(255, 0, 0))

    # -------------------
    # Show results
    # -------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_rgb);
    axes[0, 0].set_title("Original");
    axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_filtered, cmap="gray");
    axes[0, 1].set_title("Filtered Mask Before");
    axes[0, 1].axis("off")
    axes[0, 2].imshow(kernel, cmap="viridis");
    axes[0, 2].set_title(f"Donut Kernel (size={kernel_size})");
    axes[0, 2].axis("off")

    axes[1, 0].imshow(mask_sub, cmap="gray");
    axes[1, 0].set_title("Mask After Subtraction");
    axes[1, 0].axis("off")
    axes[1, 1].imshow(overlay_before);
    axes[1, 1].set_title("Overlay Before (Green)");
    axes[1, 1].axis("off")
    axes[1, 2].imshow(overlay_after);
    axes[1, 2].set_title("Overlay After (Red)");
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()



    # %% Overlay images

    def add_alpha_from_mask(img, mask_color=(0, 0, 0)):
        """
        Convert RGB image to RGBA with transparency where the image matches mask_color.

        Parameters
        ----------
        img : np.ndarray
            Input RGB image (H,W,3).
        mask_color : tuple
            RGB color to treat as transparent (default = black).

        Returns
        -------
        rgba : np.ndarray
            Image with alpha channel.
        """
        # Ensure RGB
        if img.shape[2] != 3:
            raise ValueError("Input must be RGB (H,W,3).")

        # Build mask where pixel != mask_color
        mask = np.any(img != mask_color, axis=-1).astype(np.uint8) * 255

        # Add alpha channel
        rgba = np.dstack([img, mask])
        return rgba


    import cv2
    import numpy as np
    import matplotlib.pyplot as plt




    def overlay_images(background_rgba, foreground_rgba, out_path=None, show=False):
        """
        Alpha-blend two RGBA images using Porter-Duff 'over' operator.
        Foreground goes over background.
        """

        # Ensure float in [0,1]
        bg = background_rgba.astype(np.float32) / 255.0
        fg = foreground_rgba.astype(np.float32) / 255.0

        # Split into color + alpha
        Cb, Ab = bg[..., :3], bg[..., 3:4]
        Cf, Af = fg[..., :3], fg[..., 3:4]

        # Premultiplied colors
        Cb_premul = Cb * Ab
        Cf_premul = Cf * Af

        # Output alpha
        Ao = Af + Ab * (1 - Af)

        # Output color (avoid div by 0)
        Co = np.where(
            Ao > 0,
            (Cf_premul + Cb_premul * (1 - Af)) / Ao,
            0
        )

        # Recombine
        out = np.concatenate([Co, Ao], axis=-1)
        out = (out * 255).astype(np.uint8)

        # Save
        if out_path:
            cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA))

        # Show side-by-side
        if show:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(cv2.cvtColor(background_rgba, cv2.COLOR_BGRA2RGBA))
            axes[0].set_title("Background (Green)")
            axes[0].axis("off")
            axes[1].imshow(cv2.cvtColor(foreground_rgba, cv2.COLOR_BGRA2RGBA))
            axes[1].set_title("Foreground (Red Dots)")
            axes[1].axis("off")
            axes[2].imshow(cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA))
            axes[2].set_title("Overlay (Premultiplied Alpha)")
            axes[2].axis("off")
            plt.tight_layout()
            plt.show()

        return out


    generate_red_dots(img=img, out_path=path_out.joinpath('red_dots.png'))

    # Convert black → transparent

    green_img = cv2.imread(path_out.joinpath('Equalized.png'), cv2.IMREAD_UNCHANGED)
    red_dots = cv2.imread(path_out.joinpath('red_dots.png'), cv2.IMREAD_UNCHANGED)

    red_dots_rgba = add_alpha_from_mask(red_dots, mask_color=(0, 0, 0))
    green_dots_rgba = add_alpha_from_mask(green_img, mask_color=(0, 0, 0))


    composite = overlay_images(background_rgba=green_dots_rgba,
                               foreground_rgba=red_dots_rgba,
                               out_path=None,
                               show=True)


