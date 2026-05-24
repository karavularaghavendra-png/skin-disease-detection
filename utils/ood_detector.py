"""
Out-of-Distribution (OOD) Skin Detector.

Uses multi-range YCrCb color-space thresholding to detect whether an image
contains enough human skin pixels. Supports the full Fitzpatrick skin tone
scale (I–VI) by using three overlapping detection ranges.

Includes morphological filtering to reduce false positives from objects that
happen to share similar color ranges (wood, food, animals, etc.).

References:
    - Kolkur et al., "Human Skin Detection Using RGB, HSV and YCbCr Color Models"
    - Chai & Ngan, "Face segmentation using skin-color map in videophone"
"""

import cv2
import numpy as np


# ── YCrCb detection ranges for inclusive skin-tone detection ──────────────────
# Range 1: Fair skin (Fitzpatrick I–II)
# Range 2: Medium skin (Fitzpatrick III–IV)
# Range 3: Dark skin (Fitzpatrick V–VI)
_SKIN_RANGES = [
    # (lower_Y, lower_Cr, lower_Cb, upper_Y, upper_Cr, upper_Cb)
    # Y minimum of 40 prevents black/very-dark non-skin pixels from matching
    (40,  133, 77,   230, 173, 127),    # Fair / light skin
    (40,  130, 85,   230, 180, 135),    # Medium / olive skin
    (40,  120, 90,   230, 170, 140),    # Dark / deep skin
]

# Morphological kernel for noise removal
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def is_skin_image(image_path: str, skin_threshold: float = 0.25) -> tuple[bool, float]:
    """
    Heuristic-based skin detection using multi-range YCrCb thresholding
    with morphological filtering for noise removal.

    Uses three overlapping ranges to detect skin across the full Fitzpatrick
    scale (I–VI), avoiding bias against darker skin tones.

    Morphological open+close removes scattered false positive pixels from
    non-skin objects (wood, food, animals) that happen to share skin color.

    Args:
        image_path: Path to the image file.
        skin_threshold: Minimum fraction of skin pixels required (default: 25%).

    Returns:
        (is_valid, skin_percentage)
    """
    image = cv2.imread(image_path)
    if image is None:
        return False, 0.0

    # Convert to YCrCb color space (standard for skin detection)
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Combine masks from all skin-tone ranges (union)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for y_lo, cr_lo, cb_lo, y_hi, cr_hi, cb_hi in _SKIN_RANGES:
        lower = np.array([y_lo, cr_lo, cb_lo], dtype=np.uint8)
        upper = np.array([y_hi, cr_hi, cb_hi], dtype=np.uint8)
        mask = cv2.inRange(img_ycrcb, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Morphological filtering: remove small noise blobs (open) then
    # fill small holes in genuine skin regions (close).
    # This eliminates scattered false positives from non-skin objects.
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    # Calculate percentage of skin pixels after cleanup
    skin_pixels = cv2.countNonZero(combined_mask)
    total_pixels = image.shape[0] * image.shape[1]
    skin_ratio = skin_pixels / total_pixels

    return skin_ratio >= skin_threshold, skin_ratio


def get_ood_error_message() -> str:
    return (
        "No Human Skin Detected\n\n"
        "The uploaded image does not appear to contain human skin. "
        "This tool is designed ONLY for analyzing human skin conditions "
        "(acne, eczema, fungal infections, psoriasis). "
        "Please upload a well-lit, close-up photo of the affected skin area."
    )
