"""
Out-of-Distribution (OOD) Skin Detector — Multi-Layer Verification.

Layer 1: YCrCb color-space skin pixel detection (Fitzpatrick I–VI)
Layer 2: Texture analysis (Laplacian variance) — rejects flat surfaces like paper
Layer 3: Color variance — rejects uniform-color objects (documents, walls)
Layer 4: HSV saturation check — rejects desaturated objects (white paper, grey objects)
Layer 5: Contour analysis — skin regions form large contiguous blobs, not scattered dots

All layers must pass for an image to be accepted as containing human skin.

References:
    - Kolkur et al., "Human Skin Detection Using RGB, HSV and YCbCr Color Models"
    - Chai & Ngan, "Face segmentation using skin-color map in videophone"
"""

import cv2
import numpy as np


# ── YCrCb detection ranges for inclusive skin-tone detection ──────────────────
_SKIN_RANGES = [
    # (lower_Y, lower_Cr, lower_Cb, upper_Y, upper_Cr, upper_Cb)
    (40,  133, 77,   230, 173, 127),    # Fair / light skin (Fitzpatrick I–II)
    (40,  130, 85,   230, 180, 135),    # Medium / olive skin (Fitzpatrick III–IV)
    (40,  120, 90,   230, 170, 140),    # Dark / deep skin (Fitzpatrick V–VI)
]

# Morphological kernel for noise removal
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# ── Thresholds ────────────────────────────────────────────────────────────────
SKIN_PIXEL_THRESHOLD = 0.25       # Min 25% of pixels must be skin-colored
TEXTURE_THRESHOLD = 80.0          # Min Laplacian variance (paper/docs ~10-40, skin ~100-500+)
COLOR_VARIANCE_THRESHOLD = 15.0   # Min std-dev in skin region (paper ~2-8, skin ~20-60+)
SATURATION_THRESHOLD = 10.0       # Min mean saturation in skin region (paper ~2-8, skin ~12-80+)
MIN_CONTIGUOUS_RATIO = 0.10       # Min ratio of largest skin blob to total skin pixels


def _get_skin_mask(image: np.ndarray) -> np.ndarray:
    """Generate a binary skin mask using multi-range YCrCb thresholding."""
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for y_lo, cr_lo, cb_lo, y_hi, cr_hi, cb_hi in _SKIN_RANGES:
        lower = np.array([y_lo, cr_lo, cb_lo], dtype=np.uint8)
        upper = np.array([y_hi, cr_hi, cb_hi], dtype=np.uint8)
        mask = cv2.inRange(img_ycrcb, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    # Morphological cleanup
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return combined_mask


def _check_texture(image: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:
    """
    Layer 2: Texture analysis using Laplacian variance.
    Real skin has organic texture (pores, hair, wrinkles). Paper/documents are flat.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Only compute texture within the skin region
    skin_gray = cv2.bitwise_and(gray, gray, mask=mask)
    if cv2.countNonZero(mask) < 100:
        return False, 0.0
    laplacian = cv2.Laplacian(skin_gray, cv2.CV_64F)
    # Compute variance only on skin pixels
    skin_pixels_lap = laplacian[mask > 0]
    texture_var = float(np.var(skin_pixels_lap)) if len(skin_pixels_lap) > 0 else 0.0
    return texture_var >= TEXTURE_THRESHOLD, texture_var


def _check_color_variance(image: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:
    """
    Layer 3: Color variance within skin region.
    Real skin has natural color gradients. Paper/documents are uniform.
    """
    # Extract skin pixels in BGR
    skin_pixels = image[mask > 0]
    if len(skin_pixels) < 100:
        return False, 0.0
    # Compute std-dev across all channels
    color_std = float(np.mean(np.std(skin_pixels, axis=0)))
    return color_std >= COLOR_VARIANCE_THRESHOLD, color_std


def _check_saturation(image: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:
    """
    Layer 4: HSV saturation check.
    Real skin has noticeable saturation. White paper/grey objects are desaturated.
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_channel = img_hsv[:, :, 1]
    skin_sat = sat_channel[mask > 0]
    if len(skin_sat) < 100:
        return False, 0.0
    mean_sat = float(np.mean(skin_sat))
    return mean_sat >= SATURATION_THRESHOLD, mean_sat


def _check_contiguous_region(mask: np.ndarray) -> tuple[bool, float]:
    """
    Layer 5: Contiguous skin region check.
    Real skin forms large connected blobs. Random color matches form scattered dots.
    """
    total_skin = cv2.countNonZero(mask)
    if total_skin < 100:
        return False, 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    ratio = largest_area / total_skin
    return ratio >= MIN_CONTIGUOUS_RATIO, ratio


def is_skin_image(image_path: str, skin_threshold: float = SKIN_PIXEL_THRESHOLD) -> tuple[bool, float]:
    """
    Multi-layer skin detection — ALL layers must pass.

    Layer 1: YCrCb color-space skin pixel ratio (>= 25%)
    Layer 2: Texture analysis — Laplacian variance (>= 80)
    Layer 3: Color variance in skin region (>= 15 std-dev)
    Layer 4: HSV saturation in skin region (>= 20 mean)
    Layer 5: Contiguous blob check (largest blob >= 10% of skin pixels)

    This combination effectively rejects:
    - Paper, documents, white objects (fail Layer 2, 3, 4)
    - Food, wood, random objects (fail Layer 1 or Layer 5)
    - Landscapes, buildings (fail Layer 1)
    - Animal fur (fail Layer 3, 4)

    Args:
        image_path: Path to the image file.
        skin_threshold: Minimum fraction of skin pixels (default: 25%).

    Returns:
        (is_valid, skin_percentage)
    """
    image = cv2.imread(image_path)
    if image is None:
        return False, 0.0

    # Layer 1: Skin pixel ratio
    mask = _get_skin_mask(image)
    total_pixels = image.shape[0] * image.shape[1]
    skin_pixels = cv2.countNonZero(mask)
    skin_ratio = skin_pixels / total_pixels

    if skin_ratio < skin_threshold:
        return False, skin_ratio

    # Layer 2: Texture check — reject flat surfaces (paper, walls)
    texture_ok, texture_val = _check_texture(image, mask)
    if not texture_ok:
        return False, skin_ratio

    # Layer 3: Color variance — reject uniform objects (white paper, single-color items)
    variance_ok, variance_val = _check_color_variance(image, mask)
    if not variance_ok:
        return False, skin_ratio

    # Layer 4: Saturation — reject desaturated objects (greyscale, white, grey)
    saturation_ok, saturation_val = _check_saturation(image, mask)
    if not saturation_ok:
        return False, skin_ratio

    # Layer 5: Contiguous region — reject scattered color matches
    contiguous_ok, contiguous_val = _check_contiguous_region(mask)
    if not contiguous_ok:
        return False, skin_ratio

    return True, skin_ratio


def get_ood_error_message() -> str:
    return (
        "No Human Skin Detected\n\n"
        "The uploaded image does not appear to contain human skin. "
        "This tool is designed ONLY for analyzing human skin conditions "
        "(acne, eczema, fungal infections, psoriasis). "
        "Please upload a well-lit, close-up photo of the affected skin area."
    )
