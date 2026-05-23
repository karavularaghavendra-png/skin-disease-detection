"""Shared fixtures for the test suite."""
import sys
from pathlib import Path
import numpy as np
import pytest

# Make src/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_image_array():
    """224x224 RGB image as float32 numpy array."""
    return np.random.rand(224, 224, 3).astype("float32")


@pytest.fixture
def sample_image_batch(sample_image_array):
    """Batched version of sample_image_array."""
    return np.expand_dims(sample_image_array, 0)
