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



def process_image(img, alpha=0.4, clip_percentile=99, gamma=3.0, percentile=95):
    """
    Evaluate different processing methods
    :param img:
    :param alpha:
    :param clip_percentile:
    :param gamma:
    :param percentile:
    :return:
    """
    results = {}

    # Original
    results["Original"] = img

    # Method 1: Gamma correction (dims bright areas)
    img_gamma = np.array(255 * (img / 255.0) ** gamma, dtype=np.uint8)
    results[f"Gamma (γ={gamma})"] = img_gamma

    # Method 2: Logarithmic compression
    # c = 255 / np.log(1 + np.max(img))
    # img_log = (c * np.log(1 + img.astype(np.float32))).astype(np.uint8)
    # results["Log Compression"] = img_log

    # Method 3: Percentile rescaling
    # p_val = np.percentile(img, percentile)
    # img_clip = np.clip(img, 0, p_val)
    # img_rescaled = cv2.normalize(img_clip, None, 0, 255, cv2.NORM_MINMAX)
    # results[f"Percentile Rescaling (p={percentile})"] = img_rescaled

    # Method 4: Histogram Equalization
    img_eq = cv2.equalizeHist(img)
    results["Equalized"] = img_eq

    # Method 5: Uniform + Translucent (equalization + rescaling + clipping)
    img_translucent = cv2.convertScaleAbs(img_eq, alpha=alpha, beta=0)
    clip_val = np.percentile(img_translucent, clip_percentile)
    img_uniform = np.clip(img_translucent, 0, clip_val)
    img_uniform = cv2.normalize(img_uniform, None, 0, 255, cv2.NORM_MINMAX)
    results[f"Uniform + Translucent (α={alpha}, p={clip_percentile})"] = img_uniform

    return results


def translucent_artery(img, blur_ksize=5, max_green=180,  alpha=0.5, preserve_edges=True):
    """
    Make artery structures translucent green while preserving contours.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (arteries bright, background dark).
    blur_ksize : int
        Kernel size for Gaussian blur (odd number).
    max_green : int
        Maximum green intensity allowed (0-255).
    preserve_edges : bool
        If True, overlays edges to maintain contours.

    Returns
    -------
    img_gray : np.ndarray
        Processed grayscale image (ready for green mapping).
    """

    # Step 1: Blur (low-pass filter)
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    # Step 2: Optionally preserve edges
    if preserve_edges:
        edges = cv2.Canny(img, threshold1=30, threshold2=100)
        img_blur = cv2.addWeighted(img_blur, 0.9, edges, 0.3, 0)

    # Step 3: Normalize and cap intensity
    img_norm = cv2.normalize(img_blur, None, 0, max_green, cv2.NORM_MINMAX)

    # Step 4: Apply "cloudy transparency" by blending with black
    mask = img > 0

    # Step 5: Apply cloudy transparency only on artery pixels
    img_cloudy = np.zeros_like(img_norm, dtype=np.uint8)
    img_cloudy[mask] = (img_norm[mask] * alpha).astype(np.uint8)

    return img_cloudy  # return grayscale, plotting will colorize


def translucent_artery_misty(img, blur_ksize=7, max_val=180, alpha_inside=0.4, alpha_edges=0.7,):
        """
        Make artery structures look like a translucent mist.
        Interiors are flattened, edges are slightly stronger.

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
        color : str
            'green' or 'red' for output coloring.

        Returns
        -------
        img_rgb : np.ndarray
            RGB image with misty translucent arteries.
        """

        # Step 1: Detect vessels (Otsu threshold for robust masking)
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 2: Smooth interiors (blurred version of input)
        img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

        # Step 3: Extract edges
        edges = cv2.Canny(img, 50, 120)

        # Step 4: Build translucent image
        img_flat = np.zeros_like(img, dtype=np.float32)
        img_flat[mask > 0] = img_blur[mask > 0] * alpha_inside
        img_flat[edges > 0] = img[edges > 0] * alpha_edges  # edges stronger

        # Step 5: Cap and normalize
        img_capped = np.minimum(img_flat, max_val)
        img_final = cv2.normalize(img_capped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return img_final


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



def plot_images_with_hist(img_dict:Dict[str, np.ndarray],
                          figsize:Tuple[int, int]=(14,8)):
    """
    Pair plot of the image and the histogram to evaluate the methods
    :param img_dict:
    :return:
    """
    images = list(img_dict.values())
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, len(images), height_ratios=[1, 1])

    for i, (title, im) in enumerate(img_dict.items()):
        # Image
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(to_green(im))
        ax_img.set_title(title)
        ax_img.axis("off")

        # Histogram
        ax_hist = fig.add_subplot(gs[1, i])
        ax_hist.hist(im.ravel(), bins=256, color='green', alpha=0.7)
        ax_hist.set_yscale("log")
        ax_hist.set_title(f"{title} Histogram")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Pixel Count (log)")
        ax_hist.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()



def save_images(img_dict, out_dir="outputs", color="green"):
    """
    Save processed images from dictionary as PNGs with chosen color mapping.

    Parameters
    ----------
    img_dict : dict
        Dictionary of {title: image}, values can be grayscale or RGB.
    out_dir : str or Path
        Directory where images will be saved.
    color : str
        'green' or 'red' (applies only if image is grayscale).
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
        else:
            img_rgb = img  # already RGB

        # Save (convert RGB -> BGR for cv2)
        cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved: {path}")


if __name__ == "__main__":
    path_img = CONFIG["paths"]['data'].joinpath('imgs', 'Bad', 'AD1_5.1_C=0.jpg')
    path_out = CONFIG['paths']['outputs_imgs']
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

    img_dict = process_image(img, alpha=0.4, clip_percentile=99)


    img_dict["Low-pass + Capped Green"] = translucent_artery(img,
                                                             blur_ksize=1 ,
                                                             max_green=120,
                                                             alpha=1,
                                                             preserve_edges=True)


    misty_green = translucent_artery_misty(img,
                                     blur_ksize=7,
                                     max_val=160,
                                     alpha_inside=0.1,
                                     alpha_edges=0.6, )
    img_dict['LPF Misty'] = misty_green

    translucent = translucent_artery_rgba(img, blur_ksize=7, alpha_inside=0.2, alpha_edges=0.6)
    img_dict['LPF Transl'] = translucent

    # Plot
    plot_images_with_hist(img_dict, figsize=(22, 6))


    # TODO: apply this method https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image


    save_images(img_dict, out_dir=path_out)


    # %%


    # === Load image ===
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

    # --- Step 1: Normalize intensity (clip extreme bright regions) ---
    # Clip intensities at 99th percentile to suppress overly bright pixels
    percentile = 95
    clip_val = np.percentile(img, percentile)
    img_clipped = np.clip(img, 0, clip_val)
    img_norm = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX)

    # --- Step 2: Apply CLAHE (adaptive histogram equalization) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_norm.astype(np.uint8))

    # --- Step 3: Optional gamma correction for smoothing ---
    gamma = 3.5  # <1 = brighter darks, >1 = dimmer brights
    img_gamma = np.array(255 * (img_clahe / 255.0) ** gamma, dtype=np.uint8)


    # --- Function to convert grayscale to green colormap ---
    def to_green(img_gray):
        img_color = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
        img_color[:, :, 1] = img_gray  # put grayscale in green channel
        return img_color


    # --- Images to visualize ---
    images = [img, img_norm, img_gamma]
    titles = ["Original", f"Clipped {percentile} + Normalized", f"CLAHE + Gamma {gamma:.2f}"]

    # === Plot images and histograms ===
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, len(images), height_ratios=[1, 1])  # equal height

    for i, (im, title) in enumerate(zip(images, titles)):
        # Image
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(to_green(im))
        ax_img.set_title(title)
        ax_img.axis("off")

        # Histogram
        ax_hist = fig.add_subplot(gs[1, i])
        ax_hist.hist(im.ravel(), bins=256, color='green', alpha=0.7)
        ax_hist.set_yscale("log")
        ax_hist.set_title(f"{title} Histogram")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Pixel Count (log)")
        ax_hist.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    # %%
    img = cv2.imread(str(path_img), cv2.IMREAD_GRAYSCALE)

    # --- Step 1: Strong compression of highlights ---
    gamma = 3.0  # much higher than before → makes bright regions translucent
    img_gamma = np.array(255 * (img / 255.0) ** gamma, dtype=np.uint8)

    # --- Step 2: Global intensity scaling ---
    alpha = 1.2  # scale factor (<1 = dimmer overall)
    img_scaled = cv2.convertScaleAbs(img_gamma, alpha=alpha, beta=0)

    # --- Step 3: Optional log compression ---
    c = 255 / np.log(1 + np.max(img))
    img_log = (c * np.log(1 + img.astype(np.float32))).astype(np.uint8)
    img_log = cv2.convertScaleAbs(img_log, alpha=alpha, beta=0)  # also dim

    # --- Function: grayscale → green ---
    def to_green(img_gray):
        out = np.zeros((*img_gray.shape, 3), dtype=np.uint8)
        out[:, :, 1] = img_gray
        return out

    # === Visualization ===
    images = [img, img_scaled, img_log]
    titles = ["Original", f"Gamma {gamma:.1f} + Scaling {alpha:.1f}", "Log Compression + Scaling"]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, len(images), height_ratios=[1, 1])

    for i, (im, title) in enumerate(zip(images, titles)):
        # Image
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(to_green(im))
        ax_img.set_title(title)
        ax_img.axis("off")

        # Histogram
        ax_hist = fig.add_subplot(gs[1, i])
        ax_hist.hist(im.ravel(), bins=256, color='green', alpha=0.7)
        ax_hist.set_yscale("log")
        ax_hist.set_title(f"{title} Histogram")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Pixel Count (log)")
        ax_hist.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

    # %%
    img = cv2.imread(str(path_img), cv2.IMREAD_GRAYSCALE)

    # --- Step 1: Histogram Equalization (uniform distribution) ---
    img_eq = cv2.equalizeHist(img)

    # --- Step 2: Rescale to make translucent (reduce intensity) ---
    alpha = 0.1  # reduce brightness globally
    img_translucent = cv2.convertScaleAbs(img_eq, alpha=alpha, beta=0)

    # --- Step 3: Optional clipping to avoid very bright arteries ---
    clip_val = np.percentile(img_translucent, 99)
    img_uniform = np.clip(img_translucent, 0, clip_val)
    img_uniform = cv2.normalize(img_uniform, None, 0, 255, cv2.NORM_MINMAX)

    # --- Function: grayscale → green ---
    def to_green(img_gray):
        out = np.zeros((*img_gray.shape, 3), dtype=np.uint8)
        out[:, :, 1] = img_gray
        return out

    # === Visualization ===
    images = [img, img_eq, img_uniform]
    titles = ["Original", "Equalized", "Uniform + Translucent"]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, len(images), height_ratios=[1, 1])

    for i, (im, title) in enumerate(zip(images, titles)):
        # Image
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(to_green(im))
        ax_img.set_title(title)
        ax_img.axis("off")

        # Histogram
        ax_hist = fig.add_subplot(gs[1, i])
        ax_hist.hist(im.ravel(), bins=256, color='green', alpha=0.7)
        ax_hist.set_yscale("log")
        ax_hist.set_title(f"{title} Histogram")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Pixel Count (log)")
        ax_hist.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()
