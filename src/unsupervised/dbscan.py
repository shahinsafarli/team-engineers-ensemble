"""Exact DBSCAN clustering backed by a lazy balanced KD-tree."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(slots=True)
class _KDNode:
    lower: FloatArray
    upper: FloatArray
    indices: IntArray | None = None
    axis: int = 0
    split: float = 0.0
    left: "_KDNode | None" = None
    right: "_KDNode | None" = None


class _ExactKDTree:
    """Private exact spatial index for radius and nearest-neighbor queries."""

    def __init__(self, X: ArrayLike, leaf_size: int = 32) -> None:
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        self.leaf_size = leaf_size
        self.root = self._build(np.arange(self.X.shape[0], dtype=int))

    def query_radius(self, point: ArrayLike, radius: float) -> IntArray:
        if radius < 0:
            raise ValueError("radius must be non-negative")
        point_arr = np.asarray(point, dtype=float)
        if point_arr.shape != (self.X.shape[1],):
            raise ValueError("query point dimensionality does not match the fitted data")
        radius_squared = float(radius * radius)
        matches: list[int] = []
        self._query_radius(self.root, point_arr, radius_squared, matches)
        return np.asarray(sorted(matches), dtype=int)

    def kth_distance(self, point: ArrayLike, k: int) -> float:
        """Return the exact distance to the kth nearest row, counting from one."""

        if k < 1 or k > self.X.shape[0]:
            raise ValueError("k must be between 1 and the number of fitted samples")
        point_arr = np.asarray(point, dtype=float)
        if point_arr.shape != (self.X.shape[1],):
            raise ValueError("query point dimensionality does not match the fitted data")
        heap: list[tuple[float, int]] = []
        self._query_knn(self.root, point_arr, k, heap)
        return float(np.sqrt(max(0.0, -heap[0][0])))

    def _build(self, indices: IntArray) -> _KDNode:
        points = self.X[indices]
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        if indices.size <= self.leaf_size:
            return _KDNode(lower=lower, upper=upper, indices=indices.copy())

        spreads = upper - lower
        axis = int(np.argmax(spreads))
        order = np.argsort(points[:, axis], kind="mergesort")
        sorted_indices = indices[order]
        middle = sorted_indices.size // 2
        left_indices = sorted_indices[:middle]
        right_indices = sorted_indices[middle:]
        split = float(self.X[right_indices[0], axis])
        return _KDNode(
            lower=lower,
            upper=upper,
            axis=axis,
            split=split,
            left=self._build(left_indices),
            right=self._build(right_indices),
        )

    def _query_radius(
        self,
        node: _KDNode,
        point: FloatArray,
        radius_squared: float,
        matches: list[int],
    ) -> None:
        if self._bbox_distance_squared(point, node) > radius_squared:
            return
        if node.indices is not None:
            differences = self.X[node.indices] - point
            squared = np.einsum("ij,ij->i", differences, differences)
            matches.extend(node.indices[squared <= radius_squared].tolist())
            return
        if node.left is not None:
            self._query_radius(node.left, point, radius_squared, matches)
        if node.right is not None:
            self._query_radius(node.right, point, radius_squared, matches)

    def _query_knn(
        self,
        node: _KDNode,
        point: FloatArray,
        k: int,
        heap: list[tuple[float, int]],
    ) -> None:
        threshold = -heap[0][0] if len(heap) == k else np.inf
        if self._bbox_distance_squared(point, node) > threshold:
            return
        if node.indices is not None:
            differences = self.X[node.indices] - point
            squared = np.einsum("ij,ij->i", differences, differences)
            for index, distance_squared in zip(node.indices, squared):
                item = (-float(distance_squared), -int(index))
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            return

        left = node.left
        right = node.right
        if point[node.axis] >= node.split:
            left, right = right, left
        if left is not None:
            self._query_knn(left, point, k, heap)
        if right is not None:
            self._query_knn(right, point, k, heap)

    @staticmethod
    def _bbox_distance_squared(point: FloatArray, node: _KDNode) -> float:
        below = np.maximum(node.lower - point, 0.0)
        above = np.maximum(point - node.upper, 0.0)
        delta = below + above
        return float(np.dot(delta, delta))


class DBSCAN:
    """Density-based clustering with exact lazy neighborhoods and ``-1`` noise."""

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
        if n_samples == 0:
            self.labels_ = np.empty(0, dtype=int)
            return self
        tree = _ExactKDTree(X_arr)
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for point in range(n_samples):
            if visited[point]:
                continue
            visited[point] = True
            neighbors = tree.query_radius(X_arr[point], self.eps)
            if neighbors.size < self.min_samples:
                continue
            self._expand_cluster(labels, visited, tree, X_arr, point, neighbors, cluster_id)
            cluster_id += 1

        self.labels_ = labels
        return self

    def _expand_cluster(
        self,
        labels: NDArray[np.int_],
        visited: NDArray[np.bool_],
        tree: _ExactKDTree,
        X: FloatArray,
        point: int,
        neighbors: IntArray,
        cluster_id: int,
    ) -> None:
        labels[point] = cluster_id
        seeds = [int(index) for index in neighbors]
        seeds_seen = set(seeds)
        position = 0
        while position < len(seeds):
            candidate = seeds[position]
            if not visited[candidate]:
                visited[candidate] = True
                candidate_neighbors = tree.query_radius(X[candidate], self.eps)
                if candidate_neighbors.size >= self.min_samples:
                    for neighbor in candidate_neighbors:
                        neighbor_id = int(neighbor)
                        if neighbor_id not in seeds_seen:
                            seeds_seen.add(neighbor_id)
                            seeds.append(neighbor_id)
            if labels[candidate] == -1:
                labels[candidate] = cluster_id
            position += 1


def kth_neighbor_distances(X: ArrayLike, k: int) -> FloatArray:
    """Return sorted exact kth-nonneighbor distances without an n-by-n array.

    The query row itself occupies nearest-neighbor position one, so requesting
    ``k=5`` returns the fifth *other* point, matching the previous implementation.
    """

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if k < 1 or k >= X_arr.shape[0]:
        raise ValueError("k must be at least 1 and smaller than the number of samples")
    tree = _ExactKDTree(X_arr)
    distances = np.fromiter(
        (tree.kth_distance(point, k + 1) for point in X_arr),
        dtype=float,
        count=X_arr.shape[0],
    )
    return np.sort(distances)
