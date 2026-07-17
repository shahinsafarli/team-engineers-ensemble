"""Preprocessing helpers used by experiments and tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ObjectArray = NDArray[np.object_]


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


class MixedTypePreprocessor:
    """Train-fitted numeric/categorical transformer with no row deletion.

    Numeric columns use median imputation followed by standard scaling.
    Categorical columns use mode imputation and deterministic one-hot encoding.
    Categories absent from the fitted rows map to all-zero one-hot columns.
    """

    def __init__(
        self,
        numeric_columns: tuple[int, ...],
        categorical_columns: tuple[int, ...] = (),
    ) -> None:
        if set(numeric_columns).intersection(categorical_columns):
            raise ValueError("numeric_columns and categorical_columns must not overlap")
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.numeric_medians_: FloatArray | None = None
        self.numeric_mean_: FloatArray | None = None
        self.numeric_scale_: FloatArray | None = None
        self.categorical_modes_: tuple[str, ...] | None = None
        self.categories_: tuple[tuple[str, ...], ...] | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: ArrayLike) -> "MixedTypePreprocessor":
        X_arr = np.asarray(X)
        self._validate_input(X_arr)
        self.n_features_in_ = X_arr.shape[1]

        numeric = self._numeric_matrix(X_arr)
        if numeric.shape[1]:
            with np.errstate(all="ignore"):
                medians = np.nanmedian(numeric, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            numeric = np.where(np.isnan(numeric), medians, numeric)
            mean = np.mean(numeric, axis=0)
            scale = np.std(numeric, axis=0)
            scale[scale == 0] = 1.0
        else:
            medians = np.empty(0, dtype=float)
            mean = np.empty(0, dtype=float)
            scale = np.empty(0, dtype=float)
        self.numeric_medians_ = medians.astype(float)
        self.numeric_mean_ = mean.astype(float)
        self.numeric_scale_ = scale.astype(float)

        modes: list[str] = []
        categories: list[tuple[str, ...]] = []
        for column in self.categorical_columns:
            values = [self._categorical_value(value) for value in X_arr[:, column]]
            observed = [value for value in values if value is not None]
            if observed:
                labels, counts = np.unique(np.asarray(observed, dtype=str), return_counts=True)
                mode = str(labels[np.argmax(counts)])
            else:
                mode = "<missing>"
            imputed = [mode if value is None else value for value in values]
            modes.append(mode)
            categories.append(tuple(str(value) for value in np.unique(np.asarray(imputed, dtype=str))))
        self.categorical_modes_ = tuple(modes)
        self.categories_ = tuple(categories)
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        if (
            self.numeric_medians_ is None
            or self.numeric_mean_ is None
            or self.numeric_scale_ is None
            or self.categorical_modes_ is None
            or self.categories_ is None
            or self.n_features_in_ is None
        ):
            raise RuntimeError("MixedTypePreprocessor instance is not fitted")
        X_arr = np.asarray(X)
        self._validate_input(X_arr)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the fitted data")

        numeric = self._numeric_matrix(X_arr)
        if numeric.shape[1]:
            numeric = np.where(np.isnan(numeric), self.numeric_medians_, numeric)
            numeric = (numeric - self.numeric_mean_) / self.numeric_scale_
        encoded: list[FloatArray] = [numeric]
        for position, column in enumerate(self.categorical_columns):
            mode = self.categorical_modes_[position]
            values = np.asarray(
                [mode if (value := self._categorical_value(raw)) is None else value for raw in X_arr[:, column]],
                dtype=str,
            )
            column_categories = self.categories_[position]
            one_hot = np.zeros((X_arr.shape[0], len(column_categories)), dtype=float)
            for category_index, category in enumerate(column_categories):
                one_hot[:, category_index] = values == category
            encoded.append(one_hot)
        return np.concatenate(encoded, axis=1) if encoded else np.empty((X_arr.shape[0], 0), dtype=float)

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)

    def _numeric_matrix(self, X: NDArray[np.generic]) -> FloatArray:
        if X.dtype.kind in {"b", "i", "u", "f"}:
            return np.asarray(X[:, self.numeric_columns], dtype=float)
        result = np.empty((X.shape[0], len(self.numeric_columns)), dtype=float)
        for output_column, input_column in enumerate(self.numeric_columns):
            for row, value in enumerate(X[:, input_column]):
                if self._is_missing(value):
                    result[row, output_column] = np.nan
                else:
                    try:
                        result[row, output_column] = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Non-numeric value {value!r} in numeric column {input_column}"
                        ) from exc
        return result

    @staticmethod
    def _validate_input(X: NDArray[np.generic]) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return True
        return str(value).strip() in {"", "?"}

    @classmethod
    def _categorical_value(cls, value: object) -> str | None:
        if cls._is_missing(value):
            return None
        return str(value).strip()


def stratified_train_test_indices(
    y: ArrayLike,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Return deterministic stratified train/test row indices."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
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
    return np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int)


def stratified_train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> tuple[NDArray[np.generic], NDArray[np.generic], ObjectArray, ObjectArray]:
    """Split arrays while preserving class proportions."""

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    train_indices, test_indices = stratified_train_test_indices(y_arr, test_size, random_state)
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
    """Return an exact deterministic largest-remainder stratified subset."""

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    selected = stratified_subsample_indices(y_arr, max_samples, random_state)
    return np.asarray(X_arr[selected], dtype=float), y_arr[selected].astype(object)


def largest_remainder_allocation(y: ArrayLike, n_samples: int) -> dict[object, int]:
    """Allocate exactly ``n_samples`` proportionally using Hamilton's method."""

    y_arr = np.asarray(y)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_samples > y_arr.shape[0]:
        raise ValueError("n_samples cannot exceed the available rows")
    labels, counts = np.unique(y_arr, return_counts=True)
    quotas = counts.astype(float) * n_samples / y_arr.shape[0]
    allocation = np.floor(quotas).astype(int)
    remaining = int(n_samples - allocation.sum())
    if remaining:
        fractional = quotas - allocation
        order = np.lexsort((np.arange(labels.size), -fractional))
        for position in order[:remaining]:
            allocation[position] += 1
    if int(allocation.sum()) != n_samples or np.any(allocation > counts):
        raise RuntimeError("largest-remainder allocation failed to produce a valid exact sample")
    return {label.item() if isinstance(label, np.generic) else label: int(count) for label, count in zip(labels, allocation)}


def stratified_subsample_indices(
    y: ArrayLike,
    max_samples: int,
    random_state: int | None = None,
) -> NDArray[np.int_]:
    """Return unique deterministic row indices for an exact stratified subset."""

    y_arr = np.asarray(y)
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if y_arr.shape[0] <= max_samples:
        return np.arange(y_arr.shape[0], dtype=int)
    if max_samples > y_arr.shape[0]:
        raise ValueError("max_samples cannot exceed the available rows")
    rng = np.random.default_rng(random_state)
    allocation = largest_remainder_allocation(y_arr, max_samples)
    selected: list[int] = []
    for label in np.unique(y_arr):
        indices = np.where(y_arr == label)[0]
        label_key = label.item() if isinstance(label, np.generic) else label
        take = allocation[label_key]
        if take:
            selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    rng.shuffle(selected)
    selected_arr = np.asarray(selected, dtype=int)
    if selected_arr.shape[0] != max_samples or np.unique(selected_arr).shape[0] != max_samples:
        raise RuntimeError("stratified subsampling did not produce exact unique row indices")
    return selected_arr


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
