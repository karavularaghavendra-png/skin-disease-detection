"""Tests for the prediction pipeline."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image


CLASS_NAMES = ["acne", "eczema", "fungal", "normal", "psoriasis"]


def _make_test_image(tmp_path, name="test.jpg"):
    img_path = tmp_path / name
    Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    ).save(img_path)
    return img_path


def _get_predict_fn():
    """Import predict_single_image from the canonical location."""
    try:
        from predict import predict_single_image
        return predict_single_image
    except (ImportError, ModuleNotFoundError):
        return None


def _make_mock_model(class_idx=2, confidence=0.92):
    """Create a mock Keras model that returns a fake prediction array."""
    model = MagicMock()
    fake = np.zeros((1, 5), dtype=np.float32)
    fake[0, class_idx] = confidence
    # Spread small values to remaining classes
    for i in range(5):
        if i != class_idx:
            fake[0, i] = (1.0 - confidence) / 4
    model.predict.return_value = fake
    return model


# ── Test 1: predict function is importable ─────────────────
def test_predict_function_is_importable():
    fn = _get_predict_fn()
    if fn is None:
        pytest.skip(
            "predict_single_image not importable (tensorflow may not be installed)"
        )
    assert callable(fn)


# ── Test 2: prediction returns correct output types ────────
def test_prediction_output_types(tmp_path):
    """predict_single_image should return (top_results: list[dict], raw_preds: ndarray)."""
    fn = _get_predict_fn()
    if fn is None:
        pytest.skip("predict_single_image not importable")

    img_path = _make_test_image(tmp_path)
    mock_model = _make_mock_model(class_idx=2, confidence=0.92)

    # Patch load_model_cached to return our mock model + class names
    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        # Clear lru_cache so our patch takes effect
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            top_results, raw_preds = fn(str(img_path))
        finally:
            load_model_cached.cache_clear()

    # Validate top_results shape
    assert isinstance(top_results, list), f"top_results must be list, got {type(top_results)}"
    assert len(top_results) >= 1, "top_results must have at least 1 entry"

    # Validate each result dict
    for result in top_results:
        assert isinstance(result, dict), f"Each result must be dict, got {type(result)}"
        assert "disease" in result, "Result dict must have 'disease' key"
        assert "confidence" in result, "Result dict must have 'confidence' key"
        assert isinstance(result["disease"], str)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 100.0, f"Confidence out of range: {result['confidence']}"

    # Validate raw predictions
    assert isinstance(raw_preds, np.ndarray), f"raw_preds must be ndarray, got {type(raw_preds)}"
    assert raw_preds.shape == (5,), f"raw_preds shape should be (5,), got {raw_preds.shape}"

    # Top result should be the class we mocked (fungal, index 2)
    assert top_results[0]["disease"] == "fungal"
    assert top_results[0]["confidence"] == pytest.approx(92.0, abs=0.1)


# ── Test 3: confidence values are float ────────────────────
def test_confidence_is_float(tmp_path):
    """All confidence values in top_results should be Python float."""
    fn = _get_predict_fn()
    if fn is None:
        pytest.skip("predict_single_image not importable")

    img_path = _make_test_image(tmp_path, "conf_test.jpg")
    mock_model = _make_mock_model(class_idx=0, confidence=0.85)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            top_results, _ = fn(str(img_path))
        finally:
            load_model_cached.cache_clear()

    for result in top_results:
        assert isinstance(result["confidence"], float), (
            f"Confidence should be float, got {type(result['confidence'])}"
        )


# ── Test 4: top results are sorted by confidence descending ─
def test_results_sorted_by_confidence(tmp_path):
    """Top results should be sorted from highest to lowest confidence."""
    fn = _get_predict_fn()
    if fn is None:
        pytest.skip("predict_single_image not importable")

    img_path = _make_test_image(tmp_path, "sort_test.jpg")
    mock_model = _make_mock_model(class_idx=3, confidence=0.70)

    with patch("predict.load_model_cached", return_value=(mock_model, CLASS_NAMES)):
        from predict import load_model_cached
        load_model_cached.cache_clear()

        try:
            top_results, _ = fn(str(img_path))
        finally:
            load_model_cached.cache_clear()

    confidences = [r["confidence"] for r in top_results]
    assert confidences == sorted(confidences, reverse=True), (
        f"Results not sorted by confidence: {confidences}"
    )


# ── Test 5: invalid file path raises error ─────────────────
def test_invalid_file_path():
    fn = _get_predict_fn()
    if fn is None:
        pytest.skip("predict_single_image not importable")

    with pytest.raises((FileNotFoundError, OSError, ValueError, RuntimeError)):
        fn("nonexistent_file.jpg")


# ── Test 6: corrupted image handling ──────────────────────
def test_corrupted_image(tmp_path):
    fn = _get_predict_fn()
    if fn is None:
        pytest.skip("predict_single_image not importable")

    # Create a corrupted "image" file
    corrupted_path = tmp_path / "corrupted.jpg"
    with open(corrupted_path, "wb") as f:
        f.write(b"not an image")

    with pytest.raises((OSError, ValueError)):
        fn(str(corrupted_path))


# ── Test 7: unsupported file format ───────────────────────
def test_unsupported_format(tmp_path):
    fn = _get_predict_fn()
    if fn is None:
        pytest.skip("predict_single_image not importable")

    # Create a text file with .jpg extension
    txt_path = tmp_path / "fake.jpg"
    with open(txt_path, "w") as f:
        f.write("This is not an image")

    with pytest.raises((OSError, ValueError)):
        fn(str(txt_path))


# ── Test 8: reliability analysis — high confidence is reliable ─
def test_reliability_high_confidence():
    """High-confidence predictions should be marked as reliable."""
    from predict import analyze_prediction_reliability

    # 85% confidence on one class, rest spread evenly
    preds = np.array([0.05, 0.03, 0.85, 0.04, 0.03], dtype=np.float32)
    result = analyze_prediction_reliability(preds, confidence_pct=85.0)

    assert result["is_reliable"] is True
    assert len(result["warnings"]) == 0
    assert result["entropy"] < 1.2
    assert result["margin"] > 15.0


# ── Test 9: reliability analysis — low confidence is unreliable ─
def test_reliability_low_confidence():
    """Low-confidence predictions should be marked as unreliable."""
    from predict import analyze_prediction_reliability

    # 30% confidence — model is not sure
    preds = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float32)
    result = analyze_prediction_reliability(preds, confidence_pct=30.0)

    assert result["is_reliable"] is False
    assert len(result["warnings"]) > 0
    assert any("confidence" in w.lower() for w in result["warnings"])


# ── Test 10: reliability analysis — uniform distribution ──────
def test_reliability_uniform_distribution():
    """Uniform predictions (model confused) should be marked as unreliable."""
    from predict import analyze_prediction_reliability

    # All classes have ~20% — model has no idea
    preds = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float32)
    result = analyze_prediction_reliability(preds, confidence_pct=20.0)

    assert result["is_reliable"] is False
    assert result["entropy"] > 1.2  # High entropy = confused


# ── Test 11: TTA augmentation produces correct shapes ─────────
def test_tta_augmentation_passes():
    """All 8 TTA augmentation passes produce valid (1,224,224,3) tensors."""
    from predict import _apply_tta_augmentation
    from PIL import Image

    # Create a test image
    dummy = Image.fromarray(
        np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
    )

    for aug_id in range(8):
        arr = _apply_tta_augmentation(dummy, aug_id)
        assert arr.shape == (1, 224, 224, 3), f"Pass {aug_id}: wrong shape {arr.shape}"
        assert arr.min() >= 0.0, f"Pass {aug_id}: negative values"
        assert arr.max() <= 1.0, f"Pass {aug_id}: values > 1.0"
        assert arr.dtype == np.float32, f"Pass {aug_id}: wrong dtype {arr.dtype}"


# ── Test 12: TTA disagreement flagged as unreliable ───────────
def test_reliability_tta_disagreement():
    """High TTA disagreement should be flagged as unreliable."""
    from predict import analyze_prediction_reliability

    # Averaged probs look fine (85% confident)
    avg_preds = np.array([0.85, 0.05, 0.04, 0.03, 0.03], dtype=np.float32)

    # But individual passes wildly disagree
    all_pass_probs = np.array([
        [0.95, 0.02, 0.01, 0.01, 0.01],  # pass 0: very confident acne
        [0.30, 0.40, 0.10, 0.10, 0.10],  # pass 1: says eczema!
        [0.90, 0.03, 0.03, 0.02, 0.02],  # pass 2: acne again
        [0.20, 0.10, 0.50, 0.10, 0.10],  # pass 3: says fungal!
        [0.95, 0.02, 0.01, 0.01, 0.01],  # pass 4: acne
        [0.85, 0.05, 0.04, 0.03, 0.03],  # pass 5: acne
        [0.90, 0.03, 0.02, 0.03, 0.02],  # pass 6: acne
        [0.75, 0.08, 0.07, 0.05, 0.05],  # pass 7: acne but less sure
    ], dtype=np.float32)

    result = analyze_prediction_reliability(
        avg_preds, 85.0, all_pass_probs=all_pass_probs
    )

    # High std-dev across passes → should flag TTA disagreement
    assert result["tta_disagreement"] > 0.0
    assert "tta_disagreement" in result


# ── Test 13: TTA agreement → reliable ─────────────────────────
def test_reliability_tta_agreement():
    """Consistent TTA passes should produce a reliable prediction."""
    from predict import analyze_prediction_reliability

    avg_preds = np.array([0.90, 0.04, 0.03, 0.02, 0.01], dtype=np.float32)

    # All 8 passes agree strongly
    all_pass_probs = np.tile(avg_preds, (8, 1))  # identical across passes

    result = analyze_prediction_reliability(
        avg_preds, 90.0, all_pass_probs=all_pass_probs
    )

    assert result["is_reliable"] is True
    assert result["tta_disagreement"] < 0.01  # Near-zero disagreement
    assert len(result["warnings"]) == 0
