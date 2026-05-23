"""Tests for production safeguards: OOD detection and image quality gating."""

import os
import pytest
import cv2
import numpy as np
from utils.ood_detector import is_skin_image
from utils.image_utils import check_image_quality


def test_ood_detector_with_skin(tmp_path):
    """Test that a skin-like image passes the OOD detector."""
    # Create a small skin-colored patch (YCrCb: approx skin ranges)
    # Using Y: 150, Cr: 150, Cb: 100
    skin_patch_ycrcb = np.full((100, 100, 3), [150, 150, 100], dtype=np.uint8)
    skin_patch = cv2.cvtColor(skin_patch_ycrcb, cv2.COLOR_YCrCb2BGR)

    test_path = str(tmp_path / "test_skin.jpg")
    cv2.imwrite(test_path, skin_patch)

    is_valid, ratio = is_skin_image(test_path)
    assert is_valid is True
    assert ratio > 0.9  # Should be nearly 100%


def test_ood_detector_with_non_skin(tmp_path):
    """Test that a non-skin image (e.g. black) fails the OOD detector."""
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_path = str(tmp_path / "test_black.jpg")
    cv2.imwrite(test_path, black_img)

    is_valid, ratio = is_skin_image(test_path)
    assert is_valid is False
    assert ratio < 0.1


def test_ood_detector_dark_skin(tmp_path):
    """Test that dark skin tones (Fitzpatrick V–VI) pass the OOD detector.

    This validates the multi-range fix for skin-tone bias.
    YCrCb values typical of darker skin: Y~100, Cr~145, Cb~115.
    """
    dark_skin_ycrcb = np.full((100, 100, 3), [100, 145, 115], dtype=np.uint8)
    dark_skin_bgr = cv2.cvtColor(dark_skin_ycrcb, cv2.COLOR_YCrCb2BGR)

    test_path = str(tmp_path / "test_dark_skin.jpg")
    cv2.imwrite(test_path, dark_skin_bgr)

    is_valid, ratio = is_skin_image(test_path)
    assert is_valid is True, f"Dark skin tone rejected! ratio={ratio:.3f}"
    assert ratio > 0.5, f"Dark skin detection too low: {ratio:.3f}"


def test_quality_gate_blur(tmp_path):
    """Test that a blurred image is detected."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Add some noise but then blur it heavily
    cv2.randn(img, (128, 128, 128), (50, 50, 50))
    blurred = cv2.GaussianBlur(img, (51, 51), 0)

    test_path = str(tmp_path / "test_blurred.jpg")
    cv2.imwrite(test_path, blurred)

    is_valid, warnings = check_image_quality(test_path)
    assert is_valid is False
    assert any("blurry" in w.lower() for w in warnings)


def test_quality_gate_dark_image(tmp_path):
    """Test that a very dark image triggers a brightness warning."""
    dark_img = np.full((300, 300, 3), 20, dtype=np.uint8)

    test_path = str(tmp_path / "test_dark.jpg")
    cv2.imwrite(test_path, dark_img)

    is_valid, warnings = check_image_quality(test_path)
    assert is_valid is False
    assert any("dark" in w.lower() for w in warnings)


if __name__ == "__main__":
    pytest.main([__file__])
