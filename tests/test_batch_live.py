"""Extended batch endpoint tests — covers edge cases not in test_api_integration.py.

Tests:
    1. Invalid non-image file gracefully returns error key (not crash)
    2. Mixed valid + invalid files in same batch
    3. Batch endpoint rejects unauthenticated requests
    4. Single file batch works correctly

Run:
    python -m pytest tests/test_batch_live.py -v
"""

import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient


CLASS_NAMES = ["acne", "eczema", "fungal", "normal", "psoriasis"]


def _make_test_image_bytes():
    """Create a valid JPEG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def _make_mock_model(class_idx=2, confidence=0.85):
    """Create a mock TF model."""
    model = MagicMock()
    fake = np.zeros((1, 5), dtype=np.float32)
    fake[0, class_idx] = confidence
    for i in range(5):
        if i != class_idx:
            fake[0, i] = (1.0 - confidence) / 4
    model.predict.return_value = fake
    return model


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    try:
        from api import app
        return TestClient(app)
    except ImportError:
        pytest.skip("API module not importable")


AUTH_HEADERS = {"Authorization": "Bearer dev-key-not-for-production"}


def test_batch_invalid_file_returns_error_key(client):
    """Send 1 invalid non-image file — result should have 'error' key, not crash."""
    fake_data = b"This is not an image at all, just plain text."
    files = [("files", ("not_image.jpg", fake_data, "image/jpeg"))]

    mock_model = _make_mock_model(class_idx=1, confidence=0.80)
    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()
        try:
            response = client.post(
                "/predict/batch", files=files, headers=AUTH_HEADERS
            )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200, f"Expected 200 (graceful), got {response.status_code}"
    data = response.json()
    assert data["total"] == 1
    result = data["results"][0]
    assert "error" in result, f"Expected 'error' key in result for invalid file: {result}"


def test_batch_mixed_valid_and_invalid(client):
    """Send a mix of valid and invalid files — valid should succeed, invalid should error."""
    valid_image = _make_test_image_bytes()
    invalid_data = b"not-an-image"

    files = [
        ("files", ("good.jpg", valid_image, "image/jpeg")),
        ("files", ("bad.jpg", invalid_data, "image/jpeg")),
        ("files", ("good2.jpg", valid_image, "image/jpeg")),
    ]

    mock_model = _make_mock_model(class_idx=1, confidence=0.80)
    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()
        try:
            response = client.post(
                "/predict/batch", files=files, headers=AUTH_HEADERS
            )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3

    # First and third should have disease, second should have error
    assert "disease" in data["results"][0], f"First result should succeed: {data['results'][0]}"
    assert "error" in data["results"][1], f"Second result should error: {data['results'][1]}"
    assert "disease" in data["results"][2], f"Third result should succeed: {data['results'][2]}"


def test_batch_requires_auth(client):
    """Batch endpoint requires API key authentication."""
    files = [("files", ("skin.jpg", _make_test_image_bytes(), "image/jpeg"))]
    response = client.post("/predict/batch", files=files)
    assert response.status_code in [401, 403], f"Expected 401/403, got {response.status_code}"


def test_batch_single_file(client):
    """Batch endpoint works correctly with a single file."""
    test_image = _make_test_image_bytes()
    files = [("files", ("single.jpg", test_image, "image/jpeg"))]

    mock_model = _make_mock_model(class_idx=3, confidence=0.92)
    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()
        try:
            response = client.post(
                "/predict/batch", files=files, headers=AUTH_HEADERS
            )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["disease"] == "normal"

