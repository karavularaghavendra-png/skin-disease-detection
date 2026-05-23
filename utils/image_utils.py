"""
Image Utility Module for Skin Disease Detection.
Provides quality checks for uploaded images (blur, brightness, resolution).
"""

import cv2
import numpy as np


def check_image_quality(image_path_or_array):
    """
    Analyzes image quality and returns a list of warnings.

    Returns:
        is_valid (bool), warnings (list of str)
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None:
        return False, ["Cannot read image file."]

    warnings = []
    h, w = img.shape[:2]

    # 1. Resolution Check
    if h < 224 or w < 224:
        warnings.append(
            "Image resolution is too low. Please upload a higher quality photo."
        )

    # 2. Blur Detection (Laplacian Variance)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        warnings.append(
            "Image appears blurry. Please ensure the camera is focused on the skin area."
        )

    # 3. Brightness/Lighting Check
    avg_brightness = np.mean(gray)
    if avg_brightness < 40:
        warnings.append(
            "Image is too dark. Please take a photo in a well-lit environment."
        )
    elif avg_brightness > 230:
        warnings.append(
            "Image is too bright/overexposed. Avoid direct flash or bright sunlight."
        )

    return len(warnings) == 0, warnings
