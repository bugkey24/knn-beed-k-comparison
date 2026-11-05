"""Unit tests for data loading, schema, and split integrity."""
import numpy as np
import pandas as pd
import pytest
from src.data.loader import load_beed_dataset
from src.data.validator import validate_beed_dataset, ValidationError
from src.data.splitter import split_dataset


def test_load_beed_dataset_success() -> None:
    """Tests loading the BEED dataset CSV."""
    df = load_beed_dataset("beed-bangalore-eeg-epilepsy-dataset/BEED_Data.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (8000, 17)


def test_load_beed_dataset_file_not_found() -> None:
    """Tests that missing dataset path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_beed_dataset("non_existent_path.csv")


def test_validate_beed_dataset_valid() -> None:
    """Tests validation on the real dataset."""
    df = load_beed_dataset("beed-bangalore-eeg-epilepsy-dataset/BEED_Data.csv")
    assert validate_beed_dataset(df) is True


def test_validate_beed_dataset_missing_column() -> None:
    """Tests validation error when target column is missing."""
    df = pd.DataFrame({"X1": [1, 2], "X2": [3, 4]})
    with pytest.raises(ValidationError):
        validate_beed_dataset(df)


def test_validate_beed_dataset_with_nulls() -> None:
    """Tests validation failure when dataset contains missing values."""
    df = load_beed_dataset("beed-bangalore-eeg-epilepsy-dataset/BEED_Data.csv").copy()
    df.iloc[0, 0] = np.nan
    with pytest.raises(ValidationError):
        validate_beed_dataset(df)


def test_split_dataset_stratification_and_shape() -> None:
    """Tests stratified split maintains proportions and dimensions."""
    df = load_beed_dataset("beed-bangalore-eeg-epilepsy-dataset/BEED_Data.csv")
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.20, random_state=42, stratify=True)

    assert X_train.shape == (6400, 16)
    assert X_test.shape == (1600, 16)
    assert len(y_train) == 6400
    assert len(y_test) == 1600

    # Verify stratified balance in test set (each class should have exactly 400 instances)
    counts = y_test.value_counts()
    for count in counts:
        assert count == 400
