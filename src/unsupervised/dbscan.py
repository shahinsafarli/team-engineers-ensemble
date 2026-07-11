"""DBSCAN clustering implemented with brute-force neighborhoods."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class DBSCAN:
    """Density-based clustering with ``-1`` used for noise points."""

    def __init__(self, eps: float, min_samples: int) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self.eps = eps
        self.min_samples = min_samples
        self.labels_: NDArray[np.int_] | None = None

    def fit(self, X: ArrayLike) -> "DBSCAN":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        n_samples = X_arr.shape[0]
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0
        neighborhoods = [self._region_query(X_arr, i) for i in range(n_samples)]

        for point in range(n_samples):
            if visited[point]:
                continue
            visited[point] = True
            neighbors = neighborhoods[point]
            if neighbors.size < self.min_samples:
                labels[point] = -1
                continue
            self._expand_cluster(
                labels,
                visited,
                neighborhoods,
                point,
                neighbors,
                cluster_id,
            )
            cluster_id += 1

        self.labels_ = labels
        return self

    def _expand_cluster(
        self,
        labels: NDArray[np.int_],
        visited: NDArray[np.bool_],
        neighborhoods: list[NDArray[np.int_]],
        point: int,
        neighbors: NDArray[np.int_],
        cluster_id: int,
    ) -> None:
        labels[point] = cluster_id
        seeds = list(neighbors.tolist())
        # ``seeds_seen`` mirrors the contents of ``seeds`` but as a set, so
        # membership tests below are O(1) instead of the O(len(seeds)) scan
        # a plain ``in seeds`` list check would require. Without it, the
        # frontier expansion is effectively O(n^2) (or worse) on datasets
        # with large/dense clusters, since every newly discovered neighbor
        # re-scans the whole growing ``seeds`` list.
        seeds_seen = set(seeds)
        index = 0
        while index < len(seeds):
            candidate = seeds[index]
            if not visited[candidate]:
                visited[candidate] = True
                candidate_neighbors = neighborhoods[candidate]
                if candidate_neighbors.size >= self.min_samples:
                    for neighbor in candidate_neighbors:
                        neighbor_id = int(neighbor)
                        if neighbor_id not in seeds_seen:
                            seeds_seen.add(neighbor_id)
                            seeds.append(neighbor_id)
            if labels[candidate] == -1:
                labels[candidate] = cluster_id
            index += 1

    def _region_query(self, X: NDArray[np.float64], point_index: int) -> NDArray[np.int_]:
        distances = np.linalg.norm(X - X[point_index], axis=1)
        return np.where(distances <= self.eps)[0]
