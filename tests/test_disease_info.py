"""Tests for utils/disease_info.py"""
import pytest

from utils.disease_info import get_disease_info, get_severity, DISEASE_DATABASE


ALL_CLASSES = ["acne", "eczema", "fungal", "normal", "psoriasis"]


def test_all_classes_return_known_condition():
    """Every model output class must map to a real entry."""
    for cls in ALL_CLASSES:
        info = get_disease_info(cls)
        assert info is not None, f"get_disease_info returned None for {cls!r}"
        name = info.get("display_name", "")
        assert name != "Unknown Condition", (
            f"Class {cls!r} maps to Unknown Condition. "
            "Add it to DISEASE_DATABASE."
        )


def test_all_classes_have_required_keys():
    """Each entry must have display_name, symptoms, recommendations, base_severity."""
    required_keys = {"display_name", "symptoms", "recommendations", "base_severity"}
    for cls in ALL_CLASSES:
        info = get_disease_info(cls)
        missing = required_keys - set(info.keys())
        assert not missing, f"{cls!r} is missing keys: {missing}"


def test_get_severity_returns_tuple():
    """get_severity() must return (label, colour_hex) tuple."""
    for confidence in [30.0, 60.0, 85.0, 95.0]:
        result = get_severity(confidence)
        assert isinstance(result, tuple) and len(result) == 2, (
            f"get_severity({confidence}) must return (label, colour), got {result!r}"
        )

    # With disease name — tests richer path
    for cls in ALL_CLASSES:
        result = get_severity(80.0, cls)
        assert isinstance(result, tuple) and len(result) == 2


def test_no_empty_symptom_lists():
    for cls in ALL_CLASSES:
        info = get_disease_info(cls)
        assert len(info.get("symptoms", [])) > 0, (
            f"{cls!r} has an empty symptoms list"
        )
