"""Tests for medication_map.py"""
import pytest

from utils.medication_map import MEDICATION_MAP

ALL_CLASSES = ["acne", "eczema", "fungal", "normal", "psoriasis"]


def test_all_classes_have_medications():
    for cls in ALL_CLASSES:
        assert cls in MEDICATION_MAP, f"No medication entry for {cls!r}"
        assert len(MEDICATION_MAP[cls]) > 0


def test_medication_entries_have_required_keys():
    for cls, meds in MEDICATION_MAP.items():
        for med in meds:
            assert "name" in med, f"Missing 'name' in {cls} medication"
            assert "use" in med, f"Missing 'use' in {cls} medication"
