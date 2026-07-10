"""K-Means clustering implemented with Lloyd's algorithm."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class KMeans:
    """K-Means with k-means++ initialization and multiple restarts."""

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
        n_init: int = 10,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if tol < 0:
            raise ValueError("tol cannot be negative")
        if n_init < 1:
            raise ValueError("n_init must be at least 1")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.centroids_: FloatArray | None = None
        self.labels_: NDArray[np.int_] | None = None
        self.inertia_: float | None = None

    def fit(self, X: ArrayLike) -> "KMeans":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X_arr.shape[0] < self.n_clusters:
            raise ValueError("n_clusters cannot exceed n_samples")

        rng = np.random.default_rng(self.random_state)
        best_inertia = float("inf")
        best_centroids: FloatArray | None = None
        best_labels: NDArray[np.int_] | None = None

        for _ in range(self.n_init):
            centroids = self._initialize_centroids(X_arr, rng)
            labels = np.zeros(X_arr.shape[0], dtype=int)
            for _iter in range(self.max_iter):
                distances = self._squared_distances(X_arr, centroids)
                labels = np.argmin(distances, axis=1)
                new_centroids = centroids.copy()
                for cluster in range(self.n_clusters):
                    mask = labels == cluster
                    if np.any(mask):
                        new_centroids[cluster] = np.mean(X_arr[mask], axis=0)
                    else:
                        farthest = np.argmax(np.min(distances, axis=1))
                        new_centroids[cluster] = X_arr[farthest]
                shift = float(np.linalg.norm(new_centroids - centroids))
                centroids = new_centroids
                if shift <= self.tol:
                    break

            distances = self._squared_distances(X_arr, centroids)
            labels = np.argmin(distances, axis=1)
            inertia = float(np.sum(distances[np.arange(X_arr.shape[0]), labels]))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()

        assert best_centroids is not None and best_labels is not None
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        assert self.centroids_ is not None
        return np.argmin(self._squared_distances(X_arr, self.centroids_), axis=1)

    def _initialize_centroids(
        self,
        X: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first = int(rng.integers(0, X.shape[0]))
        centroids[0] = X[first]
        closest_distances = self._squared_distances(X, centroids[:1]).ravel()
        for cluster in range(1, self.n_clusters):
            total = float(np.sum(closest_distances))
            if total <= 0:
                index = int(rng.integers(0, X.shape[0]))
            else:
                probabilities = closest_distances / total
                index = int(rng.choice(X.shape[0], p=probabilities))
            centroids[cluster] = X[index]
            closest_distances = np.minimum(
                closest_distances,
                self._squared_distances(X, centroids[cluster : cluster + 1]).ravel(),
            )
        return centroids

    @staticmethod
    def _squared_distances(X: FloatArray, centroids: FloatArray) -> FloatArray:
        return np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    def _check_is_fitted(self) -> None:
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("KMeans instance is not fitted")
