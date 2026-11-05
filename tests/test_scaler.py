"""Unit tests for feature scaling anti-data leakage behavior."""
import numpy as np
import pytest
from src.preprocessing.scaler import FeatureScaler


def test_scaler_fit_transform() -> None:
    """Tests that scaler transforms data to approximate zero mean and unit variance."""
    np.random.seed(42)
    data = np.random.normal(loc=50.0, scale=10.0, size=(1000, 5))
    scaler = FeatureScaler()
    scaled = scaler.fit_transform(data)

    assert scaler.is_fitted is True
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-7)


def test_scaler_transform_unfitted_raises() -> None:
    """Tests that transforming without fitting raises RuntimeError."""
    scaler = FeatureScaler()
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(RuntimeError):
        scaler.transform(data)


def test_scaler_anti_leakage() -> None:
    """Tests that test data does not affect scaler mean and scale parameters."""
    train_data = np.array([[10.0, 10.0], [20.0, 20.0]])
    test_data = np.array([[1000.0, 1000.0], [2000.0, 2000.0]])

    scaler = FeatureScaler()
    scaler.fit(train_data)

    initial_mean = scaler.scaler.mean_.copy()
    initial_scale = scaler.scaler.scale_.copy()

    # Transform test data
    scaler.transform(test_data)

    # Assert train statistics remained unchanged
    np.testing.assert_array_equal(scaler.scaler.mean_, initial_mean)
    np.testing.assert_array_equal(scaler.scaler.scale_, initial_scale)
