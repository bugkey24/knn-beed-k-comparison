"""Unit tests for models, evaluation metrics, and full pipeline integration."""
import numpy as np
import pytest
from src.config import load_config
from src.models.knn import KNNClassifierModel
from src.models.trainer import train_and_evaluate_k
from src.evaluation.metrics import compute_metrics
from src.pipeline import run_pipeline


def test_knn_fit_and_predict_shape() -> None:
    """Tests KNNClassifierModel fitting and prediction output dimensions."""
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])

    model = KNNClassifierModel(n_neighbors=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (2,)
    assert list(preds) == [0, 1]


def test_compute_metrics_accuracy() -> None:
    """Tests compute_metrics against known ground truth."""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 0])  # 3 out of 4 correct

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["confusion_matrix"].shape == (4, 4)


def test_end_to_end_pipeline() -> None:
    """Smoke test running pipeline end-to-end."""
    config = load_config("config/config.yaml")
    results = run_pipeline(config)

    assert "comparison_df" in results
    assert "best_k" in results
    assert results["best_k"] in config.model.k_values
    assert results["best_accuracy"] > 0.80
    assert results["comparison_df"].shape[0] == len(config.model.k_values)
