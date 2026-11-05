"""Model training iterator across hyperparameter grid."""
import time
from typing import Dict, List, Tuple
import numpy as np
from src.models.knn import KNNClassifierModel


def train_and_evaluate_k(
    k: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    metric: str = "euclidean",
    weights: str = "uniform"
) -> Tuple[KNNClassifierModel, np.ndarray, np.ndarray, float]:
    """Fits KNN for a specific K and returns predictions plus elapsed execution time.

    Args:
        k: Number of nearest neighbors.
        X_train: Scaled training features.
        y_train: Training labels.
        X_test: Scaled testing features.
        metric: Distance metric.
        weights: Weighting function.

    Returns:
        Tuple of (fitted_model, y_train_pred, y_test_pred, inference_time_sec).
    """
    model = KNNClassifierModel(n_neighbors=k, metric=metric, weights=weights)
    model.fit(X_train, y_train)

    # Measure test prediction latency
    start_time = time.perf_counter()
    y_test_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start_time

    # Train prediction for overfitting assessment
    y_train_pred = model.predict(X_train)

    return model, y_train_pred, y_test_pred, inference_time
