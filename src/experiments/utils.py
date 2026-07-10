"""Utilities for the reproducible experiment runner."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: NDArray[np.float64]
    y: NDArray[np.object_]
    source: str
    description: str
    severe_imbalance: bool = False
    high_dimensional: bool = False


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

