"""AdaBoost classifiers built from the project DecisionTree stumps."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.trees.decision_tree import DecisionStump


FloatArray = NDArray[np.float64]


class AdaBoostClassifier:
    """Discrete SAMME AdaBoost with optional SAMME.R scoring.

    The default ``algorithm="SAMME"`` matches the project brief. The
    optional ``"SAMME.R"`` mode is included for the multiclass bonus and
    uses the same weighted decision-stump learner.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        random_state: int | None = None,
        algorithm: str = "SAMME",
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if algorithm not in {"SAMME", "SAMME.R"}:
            raise ValueError("algorithm must be 'SAMME' or 'SAMME.R'")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state
        self.algorithm = algorithm

        self.estimators_: list[DecisionStump] = []
        self.classes_: NDArray[np.object_] | None = None
        self.n_classes_: int = 0
        self._estimator_weights: list[float] = []
        self._estimator_errors: list[float] = []

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AdaBoostClassifier":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("AdaBoost cannot be fit on an empty dataset")

        classes = np.unique(y_arr)
        if len(classes) < 2:
            raise ValueError("AdaBoost requires at least two classes")
        self.classes_ = classes.astype(object)
        self.n_classes_ = len(classes)
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.array([class_to_index[label] for label in y_arr], dtype=int)

        sample_weight = np.full(X_arr.shape[0], 1.0 / X_arr.shape[0], dtype=float)
        self.estimators_ = []
        self._estimator_weights = []
        self._estimator_errors = []

        for m in range(self.n_estimators):
            stump = DecisionStump(
                criterion=self.criterion,
                random_state=None if self.random_state is None else self.random_state + m,
            )
            stump.fit(X_arr, y_arr, sample_weight=sample_weight)
            prediction = stump.predict(X_arr)
            incorrect = prediction != y_arr
            error = float(np.dot(sample_weight, incorrect) / np.sum(sample_weight))
            error = float(np.clip(error, 1e-12, 1.0 - 1e-12))

            if self.algorithm == "SAMME.R":
                estimator_weight = self.learning_rate
                self.estimators_.append(stump)
                self._estimator_weights.append(estimator_weight)
                self._estimator_errors.append(error)
                proba = np.clip(stump.predict_proba(X_arr), 1e-12, 1.0)
                y_coding = np.full_like(proba, -1.0 / (self.n_classes_ - 1), dtype=float)
                y_coding[np.arange(X_arr.shape[0]), y_encoded] = 1.0
                update = -self.learning_rate * (
                    (self.n_classes_ - 1) / self.n_classes_
                ) * np.sum(y_coding * np.log(proba), axis=1)
                sample_weight *= np.exp(update)
            else:
                max_acceptable_error = 1.0 - 1.0 / self.n_classes_
                if error >= max_acceptable_error:
                    if not self.estimators_:
                        raise ValueError(
                            "The first weak learner is no better than chance; "
                            "cannot fit AdaBoost."
                        )
                    break
                alpha = self.learning_rate * (
                    np.log((1.0 - error) / error) + np.log(self.n_classes_ - 1)
                )
                self.estimators_.append(stump)
                self._estimator_weights.append(float(alpha))
                self._estimator_errors.append(error)
                sample_weight *= np.exp(alpha * incorrect)

            weight_sum = float(np.sum(sample_weight))
            if not np.isfinite(weight_sum) or weight_sum <= 0:
                break
            sample_weight /= weight_sum
            if error <= 1e-12:
                break

        if not self.estimators_:
            raise RuntimeError("AdaBoost did not fit any weak learner")
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.object_]:
        scores = self._decision_scores(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        scores = self._decision_scores(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @property
    def estimator_weights(self) -> FloatArray:
        self._check_is_fitted()
        return np.asarray(self._estimator_weights, dtype=float)

    @property
    def estimator_errors(self) -> FloatArray:
        self._check_is_fitted()
        return np.asarray(self._estimator_errors, dtype=float)

    def staged_predict(self, X: ArrayLike) -> Iterator[NDArray[np.object_]]:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        assert self.classes_ is not None
        scores = np.zeros((X_arr.shape[0], self.n_classes_), dtype=float)
        for estimator, alpha in zip(self.estimators_, self._estimator_weights):
            scores += self._estimator_scores(estimator, X_arr, alpha)
            yield self.classes_[np.argmax(scores, axis=1)]

    def _decision_scores(self, X: ArrayLike) -> FloatArray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        scores = np.zeros((X_arr.shape[0], self.n_classes_), dtype=float)
        for estimator, alpha in zip(self.estimators_, self._estimator_weights):
            scores += self._estimator_scores(estimator, X_arr, alpha)
        return scores

    def _estimator_scores(
        self,
        estimator: DecisionStump,
        X: FloatArray,
        alpha: float,
    ) -> FloatArray:
        if self.algorithm == "SAMME.R":
            proba = np.clip(estimator.predict_proba(X), 1e-12, 1.0)
            log_proba = np.log(proba)
            centered = log_proba - np.mean(log_proba, axis=1, keepdims=True)
            return alpha * (self.n_classes_ - 1) * centered

        prediction = estimator.predict(X)
        assert self.classes_ is not None
        class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        indices = np.array([class_to_index[label] for label in prediction], dtype=int)
        scores = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        scores[np.arange(X.shape[0]), indices] += alpha
        return scores

    def _check_is_fitted(self) -> None:
        if not self.estimators_ or self.classes_ is None:
            raise RuntimeError("AdaBoostClassifier instance is not fitted")

