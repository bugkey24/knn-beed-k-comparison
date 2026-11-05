"""Dataset validation module."""
from typing import List
import pandas as pd


class ValidationError(Exception):
    """Raised when dataset integrity or schema validation fails."""
    pass


def validate_beed_dataset(
    df: pd.DataFrame,
    expected_rows: int = 8000,
    expected_features: int = 16,
    target_column: str = "y",
    expected_classes: int = 4
) -> bool:
    """Validates schema, null counts, row counts, and class balance of BEED dataset.

    Args:
        df: DataFrame to validate.
        expected_rows: Expected number of rows.
        expected_features: Expected number of feature columns.
        target_column: Name of the target label column.
        expected_classes: Expected number of unique classes.

    Returns:
        True if all checks pass.

    Raises:
        ValidationError: If any assertion or schema rule fails.
    """
    # 1. Null check
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        raise ValidationError(f"Dataset contains {null_counts} null/missing values.")

    # 2. Row count check
    if len(df) != expected_rows:
        raise ValidationError(f"Expected {expected_rows} rows, found {len(df)}.")

    # 3. Target column check
    if target_column not in df.columns:
        raise ValidationError(f"Target column '{target_column}' is missing from dataset.")

    # 4. Feature columns check
    expected_feature_cols = [f"X{i}" for i in range(1, expected_features + 1)]
    missing_cols = set(expected_feature_cols) - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing expected feature columns: {missing_cols}")

    # 5. Class balance check
    class_counts = df[target_column].value_counts()
    if len(class_counts) != expected_classes:
        raise ValidationError(f"Expected {expected_classes} classes, found {len(class_counts)}.")

    expected_per_class = expected_rows // expected_classes
    for cls, count in class_counts.items():
        if count != expected_per_class:
            raise ValidationError(
                f"Class '{cls}' count is {count}, expected balanced count of {expected_per_class}."
            )

    return True
