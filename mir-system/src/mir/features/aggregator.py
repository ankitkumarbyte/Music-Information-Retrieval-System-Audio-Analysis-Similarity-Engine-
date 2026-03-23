"""
Feature Aggregator
==================
Aggregates per-frame feature matrices into fixed-size
statistical vectors for use in classical ML models (SVM, RF, XGBoost).

Supports: mean, std, min, max, median, skewness, kurtosis, percentiles.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


class FeatureAggregator:
    """
    Aggregate a (D, T) feature matrix into a 1-D statistical vector.

    Parameters
    ----------
    stats : list[str]
        Statistics to compute per feature dimension.
        Supported: "mean", "std", "min", "max", "median",
                   "skew", "kurtosis", "p25", "p75", "range".
    """

    SUPPORTED = frozenset({
        "mean", "std", "min", "max", "median",
        "skew", "kurtosis", "p25", "p75", "range",
    })

    def __init__(
        self,
        stats: list[str] | None = None,
    ) -> None:
        self.stats = stats or ["mean", "std", "min", "max"]
        invalid = set(self.stats) - self.SUPPORTED
        if invalid:
            raise ValueError(f"Unknown stats: {invalid}. Supported: {self.SUPPORTED}")

    def aggregate(self, matrix: np.ndarray) -> np.ndarray:
        """
        Aggregate a 2-D feature matrix.

        Parameters
        ----------
        matrix : np.ndarray  shape (D, T)
            Feature matrix with D dimensions and T time frames.

        Returns
        -------
        np.ndarray  shape (D * len(stats),)
        """
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        parts = []
        for stat in self.stats:
            parts.append(self._compute(stat, matrix))

        return np.concatenate(parts, axis=0).astype(np.float32)

    def aggregate_dict(self, feature_dict: dict) -> np.ndarray:
        """
        Aggregate multiple feature matrices from a features dictionary.

        Parameters
        ----------
        feature_dict : dict
            Keys are feature names, values are (D, T) np.ndarrays.

        Returns
        -------
        np.ndarray  shape (sum of aggregated dims,)
        """
        vectors = []
        for name, matrix in feature_dict.items():
            if isinstance(matrix, np.ndarray):
                vectors.append(self.aggregate(matrix))
        return np.concatenate(vectors, axis=0) if vectors else np.array([], dtype=np.float32)

    # ------------------------------------------------------------------
    # Stat computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute(stat: str, matrix: np.ndarray) -> np.ndarray:
        """Compute a stat across the time axis (axis=1) for each feature row."""
        match stat:
            case "mean":
                return matrix.mean(axis=1)
            case "std":
                return matrix.std(axis=1)
            case "min":
                return matrix.min(axis=1)
            case "max":
                return matrix.max(axis=1)
            case "median":
                return np.median(matrix, axis=1)
            case "skew":
                return skew(matrix, axis=1)
            case "kurtosis":
                return kurtosis(matrix, axis=1)
            case "p25":
                return np.percentile(matrix, 25, axis=1)
            case "p75":
                return np.percentile(matrix, 75, axis=1)
            case "range":
                return matrix.max(axis=1) - matrix.min(axis=1)
            case _:
                raise ValueError(f"Unknown stat: '{stat}'")

    def output_dim(self, feature_dim: int) -> int:
        """Calculate output vector length for a given feature dimensionality."""
        return feature_dim * len(self.stats)
