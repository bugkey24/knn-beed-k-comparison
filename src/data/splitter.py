"""Dataset partitioning module."""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    target_column: str = "y",
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits dataset into train and test sets using stratified sampling.

    Args:
        df: Input DataFrame containing features and target column.
        target_column: Name of target column.
        test_size: Fraction of samples assigned to test set.
        random_state: Random seed for reproducibility.
        stratify: Whether to use stratified sampling based on the target.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_labels = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )

    return X_train, X_test, y_train, y_test
