"""Random Forest classifier implemented with project DecisionTree models."""

from __future__ import annotations

from multiprocessing import get_context
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.trees.decision_tree import DecisionTree


FloatArray = NDArray[np.float64]


def _fit_tree_job(args: tuple[Any, ...]) -> tuple[DecisionTree, NDArray[np.int_]]:
    (
        X,
        y,
        sample_indices,
        oob_indices,
        max_depth,
        min_samples_split,
        max_features,
        random_state,
    ) = args
    tree = DecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion="gini",
        max_features=max_features,
        random_state=random_state,
    )
    tree.fit(X[sample_indices], y[sample_indices])
    return tree, oob_indices


class RandomForestClassifier:
    """Bootstrap aggregation over randomized CART trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | str | None = "sqrt",
        min_samples_split: int = 2,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if min_samples_split < 1:
            raise ValueError("min_samples_split must be at least 1")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        if oob_score and not bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimators_: list[DecisionTree] = []
        self.oob_indices_: list[NDArray[np.int_]] = []
        self.classes_: NDArray[np.object_] | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self._oob_score: float | None = None
        self._feature_importances: FloatArray | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomForestClassifier":
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("RandomForest cannot be fit on an empty dataset")

        self.classes_ = np.unique(y_arr).astype(object)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_arr.shape[1]
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.n_estimators)

        jobs: list[tuple[Any, ...]] = []
        all_indices: NDArray[np.int_] = np.arange(X_arr.shape[0], dtype=int)
        for tree_idx in range(self.n_estimators):
            sample_indices: NDArray[np.int_]
            oob_indices: NDArray[np.int_]
            if self.bootstrap:
                sample_indices = np.asarray(
                    rng.integers(0, X_arr.shape[0], size=X_arr.shape[0]),
                    dtype=int,
                )
                in_bag = np.zeros(X_arr.shape[0], dtype=bool)
                in_bag[sample_indices] = True
                oob_indices = all_indices[~in_bag]
            else:
                sample_indices = all_indices
                oob_indices = np.array([], dtype=int)
            jobs.append(
                (
                    X_arr,
                    y_arr,
                    sample_indices,
                    oob_indices,
                    self.max_depth,
                    self.min_samples_split,
                    self.max_features,
                    int(seeds[tree_idx]),
                )
            )

        if self.n_jobs == 1:
            fitted = [_fit_tree_job(job) for job in jobs]
        else:
            worker_count = None if self.n_jobs == -1 else self.n_jobs
            with get_context("spawn").Pool(processes=worker_count) as pool:
                fitted = pool.map(_fit_tree_job, jobs)

        self.estimators_ = [tree for tree, _ in fitted]
        self.oob_indices_ = [oob for _, oob in fitted]
        self._feature_importances = self._compute_feature_importances()
        self._oob_score = self._compute_oob_score(X_arr, y_arr) if self.oob_score else None
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.object_]:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}"
            )
        votes = self._tree_vote_counts(X_arr)
        assert self.classes_ is not None
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}"
            )
        proba = np.zeros((X_arr.shape[0], self.n_classes_), dtype=float)
        for tree in self.estimators_:
            proba += self._aligned_tree_proba(tree, X_arr)
        return proba / len(self.estimators_)

    @property
    def oob_score_(self) -> float:
        if self._oob_score is None:
            raise AttributeError("OOB score is only available when oob_score=True")
        return self._oob_score

    @property
    def feature_importances_(self) -> FloatArray:
        self._check_is_fitted()
        assert self._feature_importances is not None
        return self._feature_importances.copy()

    def _aligned_tree_proba(self, tree: DecisionTree, X: FloatArray) -> FloatArray:
        assert self.classes_ is not None
        tree_proba = tree.predict_proba(X)
        aligned = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        tree_classes = tree.classes_
        assert tree_classes is not None
        for tree_col, label in enumerate(tree_classes):
            forest_col = np.where(self.classes_ == label)[0]
            if forest_col.size:
                aligned[:, forest_col[0]] = tree_proba[:, tree_col]
        return aligned

    def _compute_oob_score(self, X: FloatArray, y: NDArray[Any]) -> float:
        votes = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        counts = np.zeros(X.shape[0], dtype=int)
        for tree, oob_indices in zip(self.estimators_, self.oob_indices_):
            if oob_indices.size == 0:
                continue
            tree_predictions = tree.predict(X[oob_indices])
            for row_offset, label in enumerate(tree_predictions):
                forest_col = np.where(self.classes_ == label)[0]
                if forest_col.size:
                    votes[oob_indices[row_offset], forest_col[0]] += 1.0
            counts[oob_indices] += 1
        mask = counts > 0
        if not np.any(mask):
            return float("nan")
        assert self.classes_ is not None
        predictions = self.classes_[np.argmax(votes[mask], axis=1)]
        return float(np.mean(predictions == y[mask]))

    def _tree_vote_counts(self, X: FloatArray) -> FloatArray:
        assert self.classes_ is not None
        votes = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for tree in self.estimators_:
            predictions = tree.predict(X)
            for class_index, label in enumerate(self.classes_):
                votes[:, class_index] += predictions == label
        return votes

    def _compute_feature_importances(self) -> FloatArray:
        importances = np.zeros(self.n_features_in_, dtype=float)
        for tree in self.estimators_:
            importances += tree.feature_importances()
        total = float(np.sum(importances))
        if total > 0:
            return importances / total
        return importances

    def _check_is_fitted(self) -> None:
        if not self.estimators_ or self.classes_ is None:
            raise RuntimeError("RandomForestClassifier instance is not fitted")
