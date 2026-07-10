"""Weighted CART decision tree classifier implemented from scratch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass
class Node:
    """A single CART node."""

    value: FloatArray
    samples: int
    impurity: float
    weighted_samples: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    impurity_decrease: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    """Binary-split CART classifier for continuous features.

    The implementation supports weighted samples so that the same tree
    can serve as an AdaBoost weak learner. Class labels may be numeric
    or strings; internally they are encoded and decoded on prediction.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be None or a non-negative integer")
        if min_samples_split < 1:
            raise ValueError("min_samples_split must be at least 1")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Node | None = None
        self.classes_: NDArray[np.object_] | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self._rng = np.random.default_rng(random_state)
        self._feature_importances: FloatArray | None = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "DecisionTree":
        """Fit the tree using exhaustive weighted impurity reduction."""

        X_arr = self._validate_X(X)
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("DecisionTree cannot be fit on an empty dataset")

        classes, y_encoded = np.unique(y_arr, return_inverse=True)
        self.classes_ = classes.astype(object)
        self.n_classes_ = len(classes)
        self.n_features_in_ = X_arr.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._feature_importances = np.zeros(self.n_features_in_, dtype=float)

        if sample_weight is None:
            weights = np.ones(X_arr.shape[0], dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.ndim != 1 or weights.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight must have shape (n_samples,)")
            if np.any(weights < 0):
                raise ValueError("sample_weight cannot contain negative values")
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError("sample_weight must be finite and have positive sum")

        self.root_ = self._build_tree(X_arr, y_encoded.astype(int), weights, depth=0)
        total = float(np.sum(self._feature_importances))
        if total > 0:
            self._feature_importances = self._feature_importances / total
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.object_]:
        """Return the most likely class for each row in X."""

        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._classes()[indices]

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Return empirical class probabilities from reached leaves."""

        self._check_is_fitted()
        X_arr = self._validate_X(X, fitting=False)
        probabilities = np.empty((X_arr.shape[0], self.n_classes_), dtype=float)
        assert self.root_ is not None
        for i, row in enumerate(X_arr):
            node = self._traverse(row, self.root_)
            total = float(np.sum(node.value))
            if total <= 0:
                probabilities[i] = np.full(self.n_classes_, 1.0 / self.n_classes_)
            else:
                probabilities[i] = node.value / total
        return probabilities

    @property
    def depth(self) -> int:
        """Maximum number of edges from root to any leaf."""

        self._check_is_fitted()
        assert self.root_ is not None
        return self._depth(self.root_)

    @property
    def n_leaves(self) -> int:
        """Number of terminal nodes."""

        self._check_is_fitted()
        assert self.root_ is not None
        return self._n_leaves(self.root_)

    def feature_importances(self) -> FloatArray:
        """Normalized total weighted impurity decrease per feature."""

        self._check_is_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    def __repr__(self) -> str:
        if self.root_ is None:
            return "DecisionTree(unfitted)"
        if self.depth > 4:
            return (
                "DecisionTree("
                f"depth={self.depth}, leaves={self.n_leaves}, "
                f"criterion='{self.criterion}')"
            )
        lines: list[str] = []
        self._append_repr(self.root_, lines, indent=0)
        return "\n".join(lines)

    def _build_tree(
        self,
        X: FloatArray,
        y: IntArray,
        weights: FloatArray,
        depth: int,
    ) -> Node:
        counts = self._weighted_class_counts(y, weights)
        impurity = self._impurity_from_counts(counts)
        node = Node(
            value=counts,
            samples=len(y),
            impurity=impurity,
            weighted_samples=float(np.sum(weights)),
        )

        if self._should_stop(X, y, depth, impurity):
            return node

        split = self._best_split(X, y, weights, impurity)
        if split is None:
            return node

        feature_index, threshold, gain, left_mask = split
        right_mask = ~left_mask
        node.feature_index = feature_index
        node.threshold = threshold
        node.impurity_decrease = max(gain, 0.0) * node.weighted_samples
        assert self._feature_importances is not None
        self._feature_importances[feature_index] += node.impurity_decrease
        node.left = self._build_tree(X[left_mask], y[left_mask], weights[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], weights[right_mask], depth + 1)
        return node

    def _should_stop(
        self,
        X: FloatArray,
        y: IntArray,
        depth: int,
        impurity: float,
    ) -> bool:
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if X.shape[0] < self.min_samples_split:
            return True
        if impurity <= 1e-12:
            return True
        if np.all(X == X[0]):
            return True
        return np.unique(y).size == 1

    def _best_split(
        self,
        X: FloatArray,
        y: IntArray,
        weights: FloatArray,
        parent_impurity: float,
    ) -> tuple[int, float, float, NDArray[np.bool_]] | None:
        n_samples, _ = X.shape
        if n_samples <= 1:
            return None
        total_counts = self._weighted_class_counts(y, weights)
        total_weight = float(np.sum(total_counts))
        if total_weight <= 0:
            return None

        best_gain = -float("inf")
        minimum_gain = 1e-12
        best_feature: int | None = None
        best_threshold: float | None = None
        best_left_mask: NDArray[np.bool_] | None = None
        one_hot = np.eye(self.n_classes_, dtype=float)[y] * weights[:, None]

        for feature_index in self._candidate_features():
            values = X[:, feature_index]
            order = np.argsort(values, kind="mergesort")
            x_sorted = values[order]
            cumulative = np.cumsum(one_hot[order], axis=0)
            left_counts = cumulative[:-1]
            right_counts = total_counts[None, :] - left_counts
            left_weights = np.sum(left_counts, axis=1)
            right_weights = total_weight - left_weights
            valid = (
                (x_sorted[:-1] != x_sorted[1:])
                & (left_weights > 0)
                & (right_weights > 0)
            )
            if not np.any(valid):
                continue

            left_impurity = self._impurity_many(left_counts)
            right_impurity = self._impurity_many(right_counts)
            child_impurity = (
                left_weights / total_weight * left_impurity
                + right_weights / total_weight * right_impurity
            )
            gains = np.where(valid, parent_impurity - child_impurity, -np.inf)
            split_pos = int(np.argmax(gains))
            gain = float(gains[split_pos])
            if gain >= minimum_gain and gain > best_gain:
                threshold = float((x_sorted[split_pos] + x_sorted[split_pos + 1]) / 2.0)
                best_gain = gain
                best_feature = int(feature_index)
                best_threshold = threshold

        if best_feature is None or best_threshold is None:
            return None
        best_left_mask = X[:, best_feature] <= best_threshold
        if best_left_mask.all() or (~best_left_mask).all():
            return None
        return best_feature, best_threshold, best_gain, best_left_mask

    def _candidate_features(self) -> Iterable[int]:
        n_features = self.n_features_in_
        max_features = self._resolve_max_features(n_features)
        if max_features >= n_features:
            return range(n_features)
        return self._rng.choice(n_features, size=max_features, replace=False).tolist()

    def _resolve_max_features(self, n_features: int) -> int:
        value = self.max_features
        if value is None:
            return n_features
        if isinstance(value, int):
            if value < 1:
                raise ValueError("max_features as int must be positive")
            return min(value, n_features)
        if value == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if value == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be int, 'sqrt', 'log2', or None")

    def _impurity_from_counts(self, counts: FloatArray) -> float:
        total = float(np.sum(counts))
        if total <= 0:
            return 0.0
        probabilities = counts / total
        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities**2))
        nonzero = probabilities > 0
        return float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero])))

    def _impurity_many(self, counts: FloatArray) -> FloatArray:
        totals = np.sum(counts, axis=1)
        probabilities = np.divide(
            counts,
            totals[:, None],
            out=np.zeros_like(counts, dtype=float),
            where=totals[:, None] > 0,
        )
        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities**2, axis=1)
        log_prob = np.zeros_like(probabilities)
        mask = probabilities > 0
        log_prob[mask] = np.log2(probabilities[mask])
        return -np.sum(probabilities * log_prob, axis=1)

    def _weighted_class_counts(self, y: IntArray, weights: FloatArray) -> FloatArray:
        return np.bincount(y, weights=weights, minlength=self.n_classes_).astype(float)

    def _traverse(self, row: FloatArray, node: Node) -> Node:
        current = node
        while not current.is_leaf:
            assert current.feature_index is not None
            assert current.threshold is not None
            if row[current.feature_index] <= current.threshold:
                assert current.left is not None
                current = current.left
            else:
                assert current.right is not None
                current = current.right
        return current

    def _depth(self, node: Node) -> int:
        if node.is_leaf:
            return 0
        assert node.left is not None and node.right is not None
        return 1 + max(self._depth(node.left), self._depth(node.right))

    def _n_leaves(self, node: Node) -> int:
        if node.is_leaf:
            return 1
        assert node.left is not None and node.right is not None
        return self._n_leaves(node.left) + self._n_leaves(node.right)

    def _append_repr(self, node: Node, lines: list[str], indent: int) -> None:
        prefix = "  " * indent
        counts = np.array2string(node.value, precision=2, separator=", ")
        if node.is_leaf:
            lines.append(
                f"{prefix}Leaf(samples={node.samples}, "
                f"{self.criterion}={node.impurity:.4f}, value={counts})"
            )
            return
        lines.append(
            f"{prefix}Node(x[{node.feature_index}] <= {node.threshold:.6g}, "
            f"samples={node.samples}, {self.criterion}={node.impurity:.4f}, "
            f"value={counts})"
        )
        assert node.left is not None and node.right is not None
        self._append_repr(node.left, lines, indent + 1)
        self._append_repr(node.right, lines, indent + 1)

    def _validate_X(self, X: ArrayLike, fitting: bool = True) -> FloatArray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if not np.isfinite(X_arr).all():
            raise ValueError("X must contain only finite values")
        if not fitting and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}"
            )
        return X_arr

    def _classes(self) -> NDArray[np.object_]:
        self._check_is_fitted()
        assert self.classes_ is not None
        return self.classes_

    def _check_is_fitted(self) -> None:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("DecisionTree instance is not fitted")


class DecisionStump(DecisionTree):
    """Depth-1 decision tree used as AdaBoost's weak learner."""

    def __init__(self, criterion: str = "gini", random_state: int | None = None) -> None:
        super().__init__(max_depth=1, criterion=criterion, random_state=random_state)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> "DecisionStump":
        """Fit a weighted stump using vectorized split evaluation."""

        X_arr = self._validate_X(X)
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        classes, y_encoded = np.unique(y_arr, return_inverse=True)
        self.classes_ = classes.astype(object)
        self.n_classes_ = len(classes)
        self.n_features_in_ = X_arr.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._feature_importances = np.zeros(self.n_features_in_, dtype=float)

        if sample_weight is None:
            weights = np.ones(X_arr.shape[0], dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.ndim != 1 or weights.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight must have shape (n_samples,)")
            if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError("sample_weight must be finite, non-negative, and positive-sum")

        parent_counts = self._weighted_class_counts(y_encoded.astype(int), weights)
        parent_impurity = self._impurity_from_counts(parent_counts)
        root = Node(
            value=parent_counts,
            samples=X_arr.shape[0],
            impurity=parent_impurity,
            weighted_samples=float(np.sum(weights)),
        )
        if self._should_stop(X_arr, y_encoded.astype(int), depth=0, impurity=parent_impurity):
            self.root_ = root
            return self

        split = self._best_stump_split(X_arr, y_encoded.astype(int), weights, parent_impurity)
        if split is None:
            self.root_ = root
            return self

        feature_index, threshold, gain, left_mask = split
        right_mask = ~left_mask
        left_counts = self._weighted_class_counts(y_encoded[left_mask].astype(int), weights[left_mask])
        right_counts = self._weighted_class_counts(y_encoded[right_mask].astype(int), weights[right_mask])
        root.feature_index = feature_index
        root.threshold = threshold
        root.impurity_decrease = max(gain, 0.0) * root.weighted_samples
        self._feature_importances[feature_index] = root.impurity_decrease
        root.left = Node(
            value=left_counts,
            samples=int(np.sum(left_mask)),
            impurity=self._impurity_from_counts(left_counts),
            weighted_samples=float(np.sum(weights[left_mask])),
        )
        root.right = Node(
            value=right_counts,
            samples=int(np.sum(right_mask)),
            impurity=self._impurity_from_counts(right_counts),
            weighted_samples=float(np.sum(weights[right_mask])),
        )
        self.root_ = root
        total = float(np.sum(self._feature_importances))
        if total > 0:
            self._feature_importances = self._feature_importances / total
        return self

    def _best_stump_split(
        self,
        X: FloatArray,
        y: IntArray,
        weights: FloatArray,
        parent_impurity: float,
    ) -> tuple[int, float, float, NDArray[np.bool_]] | None:
        best_gain = -float("inf")
        best_feature: int | None = None
        best_threshold: float | None = None
        total_counts = self._weighted_class_counts(y, weights)
        total_weight = float(np.sum(total_counts))
        one_hot = np.eye(self.n_classes_, dtype=float)[y] * weights[:, None]

        for feature_index in self._candidate_features():
            values = X[:, feature_index]
            order = np.argsort(values, kind="mergesort")
            x_sorted = values[order]
            cumulative = np.cumsum(one_hot[order], axis=0)
            left_counts = cumulative[:-1]
            right_counts = total_counts[None, :] - left_counts
            left_weights = np.sum(left_counts, axis=1)
            right_weights = total_weight - left_weights
            valid = (
                (x_sorted[:-1] != x_sorted[1:])
                & (left_weights > 0)
                & (right_weights > 0)
            )
            if not np.any(valid):
                continue
            left_impurity = self._impurity_many(left_counts)
            right_impurity = self._impurity_many(right_counts)
            child_impurity = (
                left_weights / total_weight * left_impurity
                + right_weights / total_weight * right_impurity
            )
            gains = parent_impurity - child_impurity
            gains = np.where(valid, gains, -np.inf)
            split_pos = int(np.argmax(gains))
            gain = float(gains[split_pos])
            if gain > 1e-12 and gain > best_gain:
                best_gain = gain
                best_feature = int(feature_index)
                best_threshold = float((x_sorted[split_pos] + x_sorted[split_pos + 1]) / 2.0)

        if best_feature is None or best_threshold is None:
            return None
        left_mask = X[:, best_feature] <= best_threshold
        if left_mask.all() or (~left_mask).all():
            return None
        return best_feature, best_threshold, best_gain, left_mask

    def _impurity_many(self, counts: FloatArray) -> FloatArray:
        totals = np.sum(counts, axis=1)
        probabilities = np.divide(
            counts,
            totals[:, None],
            out=np.zeros_like(counts, dtype=float),
            where=totals[:, None] > 0,
        )
        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities**2, axis=1)
        log_prob = np.zeros_like(probabilities)
        mask = probabilities > 0
        log_prob[mask] = np.log2(probabilities[mask])
        return -np.sum(probabilities * log_prob, axis=1)
