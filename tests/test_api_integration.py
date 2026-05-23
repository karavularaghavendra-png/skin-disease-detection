"""Integration tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import io


def _make_test_image_bytes():
    """Create a test image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def _make_mock_model(class_idx=2, confidence=0.85):
    """Create a mock TF model for testing."""
    model = MagicMock()
    fake = np.zeros((1, 5), dtype=np.float32)
    fake[0, class_idx] = confidence
    for i in range(5):
        if i != class_idx:
            fake[0, i] = (1.0 - confidence) / 4
    model.predict.return_value = fake
    return model


CLASS_NAMES = ["acne", "eczema", "fungal", "normal", "psoriasis"]


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    try:
        from api import app
        return TestClient(app)
    except ImportError:
        pytest.skip("API module not importable")


# ─────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


# ─────────────────────────────────────────────────────────────
# /predict endpoint (auth-protected)
# ─────────────────────────────────────────────────────────────

def test_predict_endpoint_success(client):
    """Test successful prediction with valid image using mocked model."""
    test_image = _make_test_image_bytes()
    mock_model = _make_mock_model(class_idx=2, confidence=0.85)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            # Also mock OOD and quality checks so random image passes
            with patch("utils.ood_detector.is_skin_image", return_value=(True, 0.75)):
                with patch("utils.image_utils.check_image_quality", return_value=(True, [])):
                    response = client.post(
                        "/predict",
                        files={"file": ("test.jpg", test_image, "image/jpeg")},
                        headers={"Authorization": "Bearer dev-key-not-for-production"}
                    )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "fungal"
    assert "display_name" in data
    assert "confidence" in data
    assert "severity" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert "symptoms" in data
    assert "recommendations" in data
    assert "quality_warnings" in data
    assert "disclaimer" in data
    # New reliability fields
    assert "is_reliable" in data
    assert isinstance(data["is_reliable"], bool)
    assert "prediction_entropy" in data
    assert "prediction_margin" in data


def test_predict_endpoint_no_auth(client):
    """Test prediction fails without authentication."""
    test_image = _make_test_image_bytes()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", test_image, "image/jpeg")}
    )

    assert response.status_code in [401, 403]


def test_predict_endpoint_invalid_auth(client):
    """Test prediction fails with invalid API key."""
    test_image = _make_test_image_bytes()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        headers={"Authorization": "Bearer invalid-key"}
    )

    assert response.status_code == 401


def test_predict_endpoint_unsupported_format(client):
    """Test prediction fails with unsupported file format."""
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        headers={"Authorization": "Bearer dev-key-not-for-production"}
    )

    assert response.status_code == 400


def test_predict_endpoint_file_too_large(client):
    """Test prediction fails with oversized file."""
    large_data = b"x" * (11 * 1024 * 1024)  # 11MB

    response = client.post(
        "/predict",
        files={"file": ("large.jpg", large_data, "image/jpeg")},
        headers={"Authorization": "Bearer dev-key-not-for-production"}
    )

    assert response.status_code == 413


def test_predict_endpoint_ood_rejection(client):
    """Test that non-skin images are rejected with 422."""
    test_image = _make_test_image_bytes()
    mock_model = _make_mock_model(class_idx=0, confidence=0.90)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            # Mock OOD to reject the image
            with patch("utils.ood_detector.is_skin_image", return_value=(False, 0.05)):
                with patch("utils.image_utils.check_image_quality", return_value=(True, [])):
                    response = client.post(
                        "/predict",
                        files={"file": ("test.jpg", test_image, "image/jpeg")},
                        headers={"Authorization": "Bearer dev-key-not-for-production"}
                    )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 422
    assert "skin" in response.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────
# /predict/batch endpoint
# ─────────────────────────────────────────────────────────────

def test_batch_predict_endpoint(client):
    """Test batch prediction endpoint."""
    test_images = [_make_test_image_bytes() for _ in range(3)]
    mock_model = _make_mock_model(class_idx=1, confidence=0.80)

    files = [
        ("files", ("test1.jpg", test_images[0], "image/jpeg")),
        ("files", ("test2.jpg", test_images[1], "image/jpeg")),
        ("files", ("test3.jpg", test_images[2], "image/jpeg")),
    ]

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            response = client.post(
                "/predict/batch",
                files=files,
                headers={"Authorization": "Bearer dev-key-not-for-production"}
            )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total" in data
    assert data["total"] == 3
    assert len(data["results"]) == 3


def test_batch_predict_endpoint_too_many_files(client):
    """Test batch prediction fails with too many files."""
    test_images = [_make_test_image_bytes() for _ in range(25)]

    files = [
        ("files", (f"test{i}.jpg", img, "image/jpeg"))
        for i, img in enumerate(test_images)
    ]

    response = client.post(
        "/predict/batch",
        files=files,
        headers={"Authorization": "Bearer dev-key-not-for-production"}
    )

    assert response.status_code == 400


# ─────────────────────────────────────────────────────────────
# Root redirect and static serving
# ─────────────────────────────────────────────────────────────

def test_root_redirects_to_ui(client):
    """GET / should redirect to /static/index.html."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert "/static/index.html" in response.headers.get("location", "")


# ─────────────────────────────────────────────────────────────
# /predict/web endpoint (no auth)
# ─────────────────────────────────────────────────────────────

def test_predict_web_no_auth_required(client):
    """POST /predict/web should work without an API key."""
    test_image = _make_test_image_bytes()
    mock_model = _make_mock_model(class_idx=2, confidence=0.85)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            with patch("utils.ood_detector.is_skin_image", return_value=(True, 0.75)):
                with patch("utils.image_utils.check_image_quality", return_value=(True, [])):
                    response = client.post(
                        "/predict/web",
                        files={"file": ("test.jpg", test_image, "image/jpeg")},
                    )
        finally:
            load_model_cached.cache_clear()

    # Should NOT return 401/403 (no auth required)
    assert response.status_code == 200
    data = response.json()

    # Verify frontend-friendly response shape
    assert data["disease"] == "fungal"
    assert "display_name" in data
    assert "confidence" in data
    assert "description" in data
    assert "severity" in data
    assert "symptoms" in data
    assert "recommendations" in data
    assert "top_predictions" in data
    assert "quality_warnings" in data
    assert "disclaimer" in data
    assert isinstance(data["confidence"], float)
    assert isinstance(data["symptoms"], list)
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["top_predictions"], list)
    assert len(data["top_predictions"]) >= 1


def test_predict_web_response_shape(client):
    """Validate the full response shape of /predict/web for frontend rendering."""
    test_image = _make_test_image_bytes()
    mock_model = _make_mock_model(class_idx=4, confidence=0.75)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            with patch("utils.ood_detector.is_skin_image", return_value=(True, 0.80)):
                with patch("utils.image_utils.check_image_quality", return_value=(True, [])):
                    response = client.post(
                        "/predict/web",
                        files={"file": ("test.jpg", test_image, "image/jpeg")},
                    )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200
    data = response.json()

    # Top prediction should be psoriasis (index 4)
    assert data["disease"] == "psoriasis"
    assert data["display_name"] == "Psoriasis"

    # Specialist field for web endpoint
    assert "specialist" in data
    assert "severity_colour" in data

    # Reliability fields
    assert "is_reliable" in data
    assert isinstance(data["is_reliable"], bool)
    assert "prediction_entropy" in data
    assert "prediction_margin" in data

    # Top predictions list must have confidence values
    for pred in data["top_predictions"]:
        assert "disease" in pred
        assert "confidence" in pred
        assert isinstance(pred["confidence"], float)


def test_predict_web_low_confidence_warning(client):
    """Test that low-confidence predictions generate reliability warnings."""
    test_image = _make_test_image_bytes()
    # Create a model with low confidence (30%) — should trigger warnings
    mock_model = _make_mock_model(class_idx=0, confidence=0.30)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            with patch("utils.ood_detector.is_skin_image", return_value=(True, 0.60)):
                with patch("utils.image_utils.check_image_quality", return_value=(True, [])):
                    response = client.post(
                        "/predict/web",
                        files={"file": ("test.jpg", test_image, "image/jpeg")},
                    )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200
    data = response.json()

    # Should be marked as unreliable
    assert data["is_reliable"] is False

    # Should have reliability warnings in quality_warnings
    assert len(data["quality_warnings"]) > 0
    assert any("confidence" in w.lower() or "uncertain" in w.lower()
               for w in data["quality_warnings"])


def test_predict_web_quality_warnings(client):
    """Test that image quality warnings are included in the response."""
    test_image = _make_test_image_bytes()
    mock_model = _make_mock_model(class_idx=0, confidence=0.90)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            with patch("utils.ood_detector.is_skin_image", return_value=(True, 0.60)):
                with patch("utils.image_utils.check_image_quality",
                           return_value=(False, ["Image appears blurry."])):
                    response = client.post(
                        "/predict/web",
                        files={"file": ("test.jpg", test_image, "image/jpeg")},
                    )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert len(data["quality_warnings"]) > 0
    assert "blurry" in data["quality_warnings"][0].lower()


def test_predict_web_unsupported_format(client):
    """POST /predict/web with a .txt file should return 400."""
    response = client.post(
        "/predict/web",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_web_file_too_large(client):
    """POST /predict/web with an oversized file should return 413."""
    large_data = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/predict/web",
        files={"file": ("large.jpg", large_data, "image/jpeg")},
    )
    assert response.status_code == 413


def test_predict_web_tta_metadata(client):
    """Verify TTA metadata fields are present in /predict/web response."""
    test_image = _make_test_image_bytes()
    mock_model = _make_mock_model(class_idx=1, confidence=0.90)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            with patch("utils.ood_detector.is_skin_image", return_value=(True, 0.80)):
                with patch("utils.image_utils.check_image_quality", return_value=(True, [])):
                    response = client.post(
                        "/predict/web",
                        files={"file": ("test.jpg", test_image, "image/jpeg")},
                    )
        finally:
            load_model_cached.cache_clear()

    assert response.status_code == 200
    data = response.json()

    # TTA metadata must be present
    assert data["tta_used"] is True
    assert isinstance(data["tta_passes"], int)
    assert data["tta_passes"] == 8
    assert isinstance(data["tta_agreement"], float)
    assert 0 <= data["tta_agreement"] <= 100
    assert isinstance(data["tta_disagreement"], float)
    assert isinstance(data["latency_ms"], float)
    assert data["latency_ms"] > 0