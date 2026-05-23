"""
Out-of-Distribution (OOD) Skin Detector.

Uses multi-range YCrCb color-space thresholding to detect whether an image
contains enough human skin pixels. Supports the full Fitzpatrick skin tone
scale (I–VI) by using three overlapping detection ranges.

References:
    - Kolkur et al., "Human Skin Detection Using RGB, HSV and YCbCr Color Models"
    - Chai & Ngan, "Face segmentation using skin-color map in videophone"
"""

import cv2
import numpy as np


# ── YCrCb detection ranges for inclusive skin-tone detection ──────────────
# Range 1: Fair skin (Fitzpatrick I–II)
# Range 2: Medium skin (Fitzpatrick III–IV)
# Range 3: Dark skin (Fitzpatrick V–VI)
_SKIN_RANGES = [
    # (lower_Y, lower_Cr, lower_Cb, upper_Y, upper_Cr, upper_Cb)
    # Y minimum of 30 prevents black/very-dark non-skin pixels from matching
    (30,  133, 77,   255, 173, 127),    # Fair / light skin
    (30,  130, 85,   255, 180, 135),    # Medium / olive skin
    (30,  120, 90,   255, 170, 140),    # Dark / deep skin
]


def is_skin_image(image_path: str, skin_threshold: float = 0.15) -> tuple[bool, float]:
    """
    Heuristic-based skin detection using multi-range YCrCb thresholding.

    Uses three overlapping ranges to detect skin across the full Fitzpatrick
    scale (I–VI), avoiding bias against darker skin tones.

    Args:
        image_path: Path to the image file.
        skin_threshold: Minimum fraction of skin pixels required (default: 15%).

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

    # Calculate percentage of skin pixels
    skin_pixels = cv2.countNonZero(combined_mask)
    total_pixels = image.shape[0] * image.shape[1]
    skin_ratio = skin_pixels / total_pixels

    return skin_ratio >= skin_threshold, skin_ratio


def get_ood_error_message() -> str:
    return (
        "🔍 **No Skin Detected**\n\n"
        "The uploaded image does not appear to be a clear photo of human skin. "
        "Our AI is specialized only for dermatological analysis. "
        "Please upload a well-lit, close-up photo of the affected area."
    )
