"""KNN Model encapsulation."""
from typing import Literal
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifierModel:
    """Wrapper around scikit-learn's KNeighborsClassifier."""

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform"
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNNClassifierModel":
        """Fits the KNN classifier on scaled training data."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for input samples."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for input samples."""
        return self.model.predict_proba(X)
