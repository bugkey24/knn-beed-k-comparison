"""Evaluation metrics computation module."""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro"
) -> Dict[str, Any]:
    """Computes multiclass classification metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        average: Averaging strategy for multi-class metrics ('macro', 'weighted', etc.).

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
