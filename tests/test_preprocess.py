"""Tests for the image preprocessing pipeline."""
import numpy as np
import pytest

from preprocess import preprocess_single_image as preprocess_image


def test_output_shape(tmp_path):
    """preprocess_image must output (1, 224, 224, 3) float32."""
    from PIL import Image
    img = Image.fromarray(
        np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    )
    path = tmp_path / "test.jpg"
    img.save(path)

    result = preprocess_image(str(path))
    assert result.shape == (1, 224, 224, 3), (
        f"Expected (1,224,224,3), got {result.shape}"
    )
    assert result.dtype == np.float32


def test_pixel_values_normalised(tmp_path):
    """Pixel values should be in [0, 1] range after preprocessing."""
    from PIL import Image
    img = Image.fromarray(
        np.full((100, 100, 3), 255, dtype=np.uint8)
    )
    path = tmp_path / "white.jpg"
    img.save(path)
    result = preprocess_image(str(path))
    assert result.max() <= 1.0, "Pixels not normalised to [0,1]"
    assert result.min() >= 0.0
