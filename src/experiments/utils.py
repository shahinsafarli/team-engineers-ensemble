"""Utilities for the reproducible experiment runner."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import ttest_rel


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: NDArray[np.float64] | NDArray[np.object_]
    y: NDArray[np.object_]
    source: str
    description: str
    severe_imbalance: bool = False
    high_dimensional: bool = False
    numeric_columns: tuple[int, ...] | None = None
    categorical_columns: tuple[int, ...] = ()
    source_version: str = ""
    selection_rule: str = ""
    preprocessing: str = "Numeric median imputation and standard scaling fitted on training rows only."
    raw_samples: int | None = None

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("DatasetBundle.X must be a 2D array")
        if self.y.ndim != 1 or self.y.shape[0] != self.X.shape[0]:
            raise ValueError("DatasetBundle.y must be one-dimensional and aligned with X")
        n_features = self.X.shape[1]
        numeric = tuple(range(n_features)) if self.numeric_columns is None else self.numeric_columns
        categorical = self.categorical_columns
        if len(set(numeric).intersection(categorical)):
            raise ValueError("numeric_columns and categorical_columns must not overlap")
        if set(numeric).union(categorical) != set(range(n_features)):
            raise ValueError("numeric_columns and categorical_columns must cover every feature")

    @property
    def resolved_numeric_columns(self) -> tuple[int, ...]:
        """Return explicit numeric indices even for an all-numeric bundle."""

        if self.numeric_columns is None:
            return tuple(range(self.X.shape[1]))
        return self.numeric_columns


def dataset_fingerprint(dataset: DatasetBundle) -> str:
    """Return a stable content fingerprint for cache/provenance validation."""

    digest = hashlib.sha256()
    digest.update(dataset.name.encode("utf-8"))
    digest.update(str(dataset.X.shape).encode("ascii"))
    digest.update(str(dataset.y.shape).encode("ascii"))
    X_arr = np.asarray(dataset.X)
    if X_arr.dtype == object:
        for row in X_arr:
            digest.update("\x1f".join("<missing>" if value is None else str(value) for value in row).encode("utf-8"))
            digest.update(b"\x1e")
    else:
        digest.update(np.ascontiguousarray(X_arr).tobytes())
    digest.update("\x1f".join(str(value) for value in np.asarray(dataset.y)).encode("utf-8"))
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def ensure_output_dirs(root: Path) -> tuple[Path, Path]:
    figures_dir = root / "figures"
    results_dir = root / "data" / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, results_dir


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def aligned_proba(
    model: Any,
    X: ArrayLike,
    labels: ArrayLike,
) -> NDArray[np.float64] | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = np.asarray(model.predict_proba(X), dtype=float)
    target_labels = np.asarray(labels).astype(object)
    target_keys = target_labels.astype(str)
    model_labels = getattr(model, "classes_", target_labels)
    model_labels = np.asarray(model_labels).astype(object)
    model_keys = model_labels.astype(str)
    if proba.shape[1] == target_labels.size and np.array_equal(model_keys, target_keys):
        return proba
    aligned = np.zeros((proba.shape[0], target_labels.size), dtype=float)
    for source_col, label in enumerate(model_keys):
        target_col = np.where(target_keys == label)[0]
        if target_col.size:
            aligned[:, target_col[0]] = proba[:, source_col]
    row_sums = aligned.sum(axis=1, keepdims=True)
    missing = row_sums.ravel() <= 0
    if np.any(missing):
        aligned[missing] = 1.0 / target_labels.size
        row_sums = aligned.sum(axis=1, keepdims=True)
    return aligned / row_sums


def mean_std_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_keys)
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        out = {column: value for column, value in zip(group_keys, key)}
        for metric in metric_keys:
            values = np.array([float(row[metric]) for row in group_rows], dtype=float)
            out[f"{metric}_mean"] = float(np.nanmean(values))
            out[f"{metric}_std"] = float(np.nanstd(values))
            out[f"{metric}_pm"] = f"{np.nanmean(values):.3f} +/- {np.nanstd(values):.3f}"
        summary.append(out)
    return summary


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment for a family of p-values.

    Returns adjusted p-values in the same order as ``p_values``. This is
    the standard correction to apply when several hypothesis tests are
    reported together (e.g. one paired t-test per dataset), so that the
    family-wise false-positive rate is controlled rather than the
    per-comparison rate.

    Non-finite entries (``NaN``, ``+inf``, ``-inf``) represent an undefined
    test outcome -- e.g. ``scipy.stats.ttest_rel`` returns ``NaN`` when the
    paired differences have zero variance -- and are excluded from the
    correction family so they cannot lower the correction factor applied
    to the real p-values. Their positions are preserved and returned as
    ``NaN``, since an undefined test must never be reported as evidence of
    significance.
    """

    finite_indices = [i for i, p in enumerate(p_values) if math.isfinite(p)]
    n = len(finite_indices)
    adjusted = [float("nan")] * len(p_values)
    order = sorted(finite_indices, key=lambda i: p_values[i])
    running_max = 0.0
    for rank, i in enumerate(order):
        candidate = (n - rank) * p_values[i]
        running_max = max(running_max, candidate)
        adjusted[i] = min(running_max, 1.0)
    return adjusted


def paired_ttest_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    pair_column: str,
    pair_values: tuple[str, str],
    fold_key: str,
    metric_keys: list[str],
) -> list[dict[str, Any]]:
    """Paired t-test between two ``pair_column`` levels, matched by fold.

    Reuses per-fold rows already produced by a cross-validation experiment
    (e.g. ``head_to_head_cv.csv`` rows) instead of recomputing anything.
    ``scipy.stats.ttest_rel`` is appropriate here because both models in
    ``pair_values`` are evaluated on the exact same folds, so the two
    samples are naturally paired rather than independent.

    Every ``(group, metric)`` combination is a separate hypothesis test, so
    with more than one group this function is testing a family of
    hypotheses per metric. Each metric's p-values are therefore also
    Holm-Bonferroni corrected across all groups (e.g. across datasets) and
    reported alongside the raw p-value as ``{metric}_p_value_holm``.
    """

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_keys)
        grouped.setdefault(key, []).append(row)
    model_a, model_b = pair_values
    keys = list(grouped.keys())
    results: list[dict[str, Any]] = []
    raw_p_by_metric: dict[str, list[float]] = {metric: [] for metric in metric_keys}
    for key in keys:
        group_rows = grouped[key]
        by_model: dict[str, dict[Any, dict[str, Any]]] = {model_a: {}, model_b: {}}
        for row in group_rows:
            if row[pair_column] in by_model:
                by_model[row[pair_column]][row[fold_key]] = row
        shared_folds = sorted(set(by_model[model_a]) & set(by_model[model_b]))
        out = {column: value for column, value in zip(group_keys, key)}
        out["model_a"] = model_a
        out["model_b"] = model_b
        out["n_folds"] = len(shared_folds)
        for metric in metric_keys:
            a_values = np.array([float(by_model[model_a][fold][metric]) for fold in shared_folds], dtype=float)
            b_values = np.array([float(by_model[model_b][fold][metric]) for fold in shared_folds], dtype=float)
            t_stat, p_value = ttest_rel(a_values, b_values)
            out[f"{metric}_mean_diff"] = float(np.mean(a_values - b_values))
            out[f"{metric}_t_stat"] = float(t_stat)
            out[f"{metric}_p_value"] = float(p_value)
            raw_p_by_metric[metric].append(float(p_value))
        results.append(out)

    for metric in metric_keys:
        adjusted = holm_bonferroni(raw_p_by_metric[metric])
        for out, adj_p in zip(results, adjusted):
            out[f"{metric}_p_value_holm"] = adj_p

    return results


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
