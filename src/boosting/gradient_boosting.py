"""A compact binary Gradient Boosting classifier for the bonus task."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class _RegressionStump:
    def __init__(self) -> None:
        self.feature_index: int | None = None
        self.threshold: float | None = None
        self.left_value: float = 0.0
        self.right_value: float = 0.0

    def fit(self, X: FloatArray, residual: FloatArray) -> "_RegressionStump":
        best_loss = float("inf")
        for feature in range(X.shape[1]):
            values = X[:, feature]
            order = np.argsort(values, kind="mergesort")
            x_sorted = values[order]
            r_sorted = residual[order]
            csum = np.cumsum(r_sorted)
            csum_sq = np.cumsum(r_sorted**2)
            total_sum = csum[-1]
            total_sq = csum_sq[-1]
            n = X.shape[0]

            for split_pos in range(n - 1):
                if x_sorted[split_pos] == x_sorted[split_pos + 1]:
                    continue
                left_n = split_pos + 1
                right_n = n - left_n
                left_sum = csum[split_pos]
                left_sq = csum_sq[split_pos]
                right_sum = total_sum - left_sum
                right_sq = total_sq - left_sq
                left_loss = left_sq - (left_sum**2 / left_n)
                right_loss = right_sq - (right_sum**2 / right_n)
                loss = float(left_loss + right_loss)
                if loss < best_loss:
                    best_loss = loss
                    self.feature_index = feature
                    self.threshold = float((x_sorted[split_pos] + x_sorted[split_pos + 1]) / 2)
                    self.left_value = float(left_sum / left_n)
                    self.right_value = float(right_sum / right_n)
        if self.feature_index is None:
            self.feature_index = 0
            self.threshold = float(np.median(X[:, 0]))
            self.left_value = float(np.mean(residual))
            self.right_value = self.left_value
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        X_arr = np.asarray(X, dtype=float)
        assert self.feature_index is not None and self.threshold is not None
        return np.where(
            X_arr[:, self.feature_index] <= self.threshold,
            self.left_value,
            self.right_value,
        )


class GradientBoostingClassifier:
    """Binary logistic gradient boosting with regression stumps."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.classes_: NDArray[np.object_] | None = None
        self.estimators_: list[_RegressionStump] = []
        self.init_score_: float = 0.0
        self.train_loss_: list[float] = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> "GradientBoostingClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        if classes.size != 2:
            raise ValueError("GradientBoostingClassifier supports binary labels only")
        self.classes_ = classes.astype(object)
        y_bin = (y_arr == self.classes_[1]).astype(float)
        positive_rate = float(np.clip(np.mean(y_bin), 1e-6, 1 - 1e-6))
        self.init_score_ = float(np.log(positive_rate / (1 - positive_rate)))
        scores = np.full(X_arr.shape[0], self.init_score_, dtype=float)
        self.estimators_ = []
        self.train_loss_ = []

        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(scores)
            residual = y_bin - probabilities
            stump = _RegressionStump().fit(X_arr, residual)
            scores += self.learning_rate * stump.predict(X_arr)
            self.estimators_.append(stump)
            self.train_loss_.append(self._log_loss(y_bin, self._sigmoid(scores)))
        return self

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        scores = np.full(X_arr.shape[0], self.init_score_, dtype=float)
        for stump in self.estimators_:
            scores += self.learning_rate * stump.predict(X_arr)
        p1 = self._sigmoid(scores)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: ArrayLike) -> NDArray[np.object_]:
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    @staticmethod
    def _sigmoid(scores: FloatArray) -> FloatArray:
        scores = np.clip(scores, -30, 30)
        return 1.0 / (1.0 + np.exp(-scores))

    @staticmethod
    def _log_loss(y_true: FloatArray, probabilities: FloatArray) -> float:
        probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
        return float(
            -np.mean(
                y_true * np.log(probabilities)
                + (1 - y_true) * np.log(1 - probabilities)
            )
        )

    def _check_is_fitted(self) -> None:
        if self.classes_ is None or not self.estimators_:
            raise RuntimeError("GradientBoostingClassifier instance is not fitted")


