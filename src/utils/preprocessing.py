"""Preprocessing helpers used by experiments and tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


class StandardScaler:
    """Small NumPy equivalent of sklearn's standard scaler."""

    def __init__(self) -> None:
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None

    def fit(self, X: ArrayLike) -> "StandardScaler":
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X_arr, axis=0)
        scale = np.std(X_arr, axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler instance is not fitted")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)


def stratified_train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[FloatArray, FloatArray, NDArray[np.object_], NDArray[np.object_]]:
    """Split arrays while preserving class proportions."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(y_arr):
        indices = np.where(y_arr == label)[0]
        rng.shuffle(indices)
        if indices.size == 1:
            n_test = 0
        else:
            n_test = max(1, int(round(indices.size * test_size)))
            n_test = min(n_test, indices.size - 1)
        test_indices.extend(indices[:n_test].tolist())
        train_indices.extend(indices[n_test:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return (
        X_arr[train_indices],
        X_arr[test_indices],
        y_arr[train_indices].astype(object),
        y_arr[test_indices].astype(object),
    )


def random_oversample(
    X: ArrayLike,
    y: ArrayLike,
    random_state: int | None = None,
    target_ratio: float = 1.0,
) -> tuple[FloatArray, NDArray[np.object_]]:
    """Oversample minority classes toward the majority-class count.

    ``target_ratio=1.0`` balances every class to the majority count.
    Smaller values, such as 0.25, create a lighter treatment that is
    useful for severe-imbalance experiments with very rare classes.
    """

    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be in (0, 1]")

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    rng = np.random.default_rng(random_state)
    labels, counts = np.unique(y_arr, return_counts=True)
    majority_count = int(np.max(counts))
    minimum_target_count = max(1, int(np.ceil(majority_count * target_ratio)))
    selected: list[int] = []
    for label in labels:
        indices = np.where(y_arr == label)[0]
        target_count = max(indices.size, minimum_target_count)
        if indices.size < target_count:
            extra = rng.choice(indices, size=target_count - indices.size, replace=True)
            indices = np.concatenate([indices, extra])
        selected.extend(indices.tolist())
    rng.shuffle(selected)
    return X_arr[selected], y_arr[selected].astype(object)


def stratified_subsample(
    X: ArrayLike,
    y: ArrayLike,
    max_samples: int,
    random_state: int | None = None,
) -> tuple[FloatArray, NDArray[np.object_]]:
    """Return a deterministic stratified subset with at most max_samples rows."""

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    if X_arr.shape[0] <= max_samples:
        return X_arr, y_arr.astype(object)
    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    for label in np.unique(y_arr):
        indices = np.where(y_arr == label)[0]
        share = indices.size / X_arr.shape[0]
        take = max(1, int(round(max_samples * share)))
        take = min(take, indices.size)
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(selected) > max_samples:
        selected = rng.choice(selected, size=max_samples, replace=False).tolist()
    rng.shuffle(selected)
    return X_arr[selected], y_arr[selected].astype(object)


def flip_labels(
    y: ArrayLike,
    noise_fraction: float,
    random_state: int | None = None,
) -> NDArray[np.object_]:
    """Randomly replace a fraction of labels with another class."""

    if not 0 <= noise_fraction <= 1:
        raise ValueError("noise_fraction must be in [0, 1]")
    y_arr = np.asarray(y).astype(object)
    labels = np.unique(y_arr)
    if labels.size < 2:
        return y_arr.copy()
    rng = np.random.default_rng(random_state)
    corrupted = y_arr.copy()
    n_flip = int(round(noise_fraction * y_arr.shape[0]))
    if n_flip == 0:
        return corrupted
    flip_indices = rng.choice(y_arr.shape[0], size=n_flip, replace=False)
    for index in flip_indices:
        alternatives = labels[labels != corrupted[index]]
        corrupted[index] = rng.choice(alternatives)
    return corrupted


def class_distribution(y: ArrayLike) -> dict[str, int]:
    labels, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}

