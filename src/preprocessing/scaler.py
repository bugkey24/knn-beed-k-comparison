"""Feature scaling module with strict anti-data leakage enforcement."""
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """StandardScaler wrapper enforcing strict separation of fit and transform."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame | np.ndarray) -> "FeatureScaler":
        """Fits the scaler exclusively on training data.

        Args:
            X_train: Training features.

        Returns:
            Self (fitted scaler).
        """
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforms features using previously computed statistics from training data.

        Args:
            X: Features to transform.

        Returns:
            Standardized numpy array.

        Raises:
            RuntimeError: If transform is called prior to fit.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on training data before transforming.")
        return self.scaler.transform(X)

    def fit_transform(self, X_train: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fits on training data and transforms training features in one step.

        Args:
            X_train: Training features.

        Returns:
            Standardized training numpy array.
        """
        self.fit(X_train)
        return self.transform(X_train)
