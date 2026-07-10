"""Principal Component Analysis implemented with NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class PCA:
    """Linear projection onto directions of maximum variance."""

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        self.n_components = n_components
        self.components_: FloatArray | None = None
        self.mean_: FloatArray | None = None
        self.explained_variance_: FloatArray | None = None
        self.explained_variance_ratio_: FloatArray | None = None

    def fit(self, X: ArrayLike) -> "PCA":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.n_components > X_arr.shape[1]:
            raise ValueError("n_components cannot exceed n_features")

        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_
        covariance = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        total_variance = float(np.sum(eigenvalues))
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components, dtype=float)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        assert self.mean_ is not None and self.components_ is not None
        return (X_arr - self.mean_) @ self.components_.T

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)

    def _check_is_fitted(self) -> None:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA instance is not fitted")
