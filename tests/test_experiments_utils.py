"""Unit tests for src/experiments/utils.py.

These were added to close a real coverage gap: the previous .coveragerc
omitted src/experiments/* entirely, which hid the fact that the runner
utilities were never exercised by the test suite. See REPRODUCIBILITY.md
and README.md "Testing note" for the full explanation.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.experiments.utils import (
    DatasetBundle,
    aligned_proba,
    ensure_output_dirs,
    mean_std_rows,
    project_root,
    slugify,
    write_csv,
    write_json,
)


def test_project_root_points_at_repo_root():
    root = project_root()
    assert (root / "src").is_dir()
    assert (root / "tests").is_dir()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Breast Cancer Wisconsin", "breast_cancer_wisconsin"),
        ("Digits High-Dimensional", "digits_high_dimensional"),
        ("  Weird__Spacing!! ", "weird_spacing"),
        ("ALLCAPS", "allcaps"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_ensure_output_dirs_creates_both(tmp_path):
    figures_dir, results_dir = ensure_output_dirs(tmp_path)
    assert figures_dir == tmp_path / "figures"
    assert results_dir == tmp_path / "data" / "results"
    assert figures_dir.is_dir()
    assert results_dir.is_dir()


def test_dataset_bundle_defaults():
    bundle = DatasetBundle(
        name="Toy",
        X=np.zeros((2, 2)),
        y=np.array([0, 1], dtype=object),
        source="synthetic",
        description="unit-test bundle",
    )
    assert bundle.severe_imbalance is False
    assert bundle.high_dimensional is False


def test_write_csv_roundtrip_and_empty(tmp_path):
    path = tmp_path / "nested" / "out.csv"
    rows = [
        {"a": 1, "b": "x"},
        {"a": 2, "c": "y"},  # ragged: introduces a new column, missing "b"
    ]
    write_csv(path, rows)
    text = path.read_text(encoding="utf-8")
    assert "a,b,c" in text
    assert "1,x," in text

    empty_path = tmp_path / "empty.csv"
    write_csv(empty_path, [])
    assert empty_path.read_text(encoding="utf-8") == ""


def test_write_csv_sanitizes_nan_and_inf(tmp_path):
    path = tmp_path / "nan.csv"
    write_csv(path, [{"value": float("nan")}, {"value": float("inf")}, {"value": 1.5}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1] in ("", '""')  # NaN sanitized to blank
    assert lines[2] in ("", '""')  # inf sanitized to blank
    assert lines[3] == "1.5"


def test_write_json_handles_numpy_and_nan(tmp_path):
    path = tmp_path / "out.json"
    payload = {
        "arr": np.array([1, 2, 3]),
        "scalar": np.float64(2.5),
        "nested": {1: [np.int64(4), float("nan"), (1, 2)]},
        "plain": "ok",
    }
    write_json(path, payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["arr"] == [1, 2, 3]
    assert loaded["scalar"] == 2.5
    assert loaded["nested"]["1"][0] == 4
    assert loaded["nested"]["1"][1] is None
    assert loaded["nested"]["1"][2] == [1, 2]
    assert loaded["plain"] == "ok"


class _FakeModelWithProba:
    def __init__(self, classes, proba):
        self.classes_ = np.asarray(classes, dtype=object)
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return self._proba


class _FakeModelNoProba:
    def predict(self, X):
        return np.zeros(len(X))


def test_aligned_proba_returns_none_without_predict_proba():
    assert aligned_proba(_FakeModelNoProba(), np.zeros((3, 2)), np.array([0, 1])) is None


def test_aligned_proba_passthrough_when_already_aligned():
    model = _FakeModelWithProba([0, 1], [[0.7, 0.3], [0.2, 0.8]])
    aligned = aligned_proba(model, np.zeros((2, 2)), np.array([0, 1]))
    assert np.allclose(aligned, [[0.7, 0.3], [0.2, 0.8]])


def test_aligned_proba_reorders_and_fills_missing_classes():
    # Model only ever saw class "1" in this bootstrap; target space has 0,1,2.
    model = _FakeModelWithProba(["1"], [[1.0], [1.0]])
    aligned = aligned_proba(model, np.zeros((2, 2)), np.array([0, 1, 2]))
    assert aligned.shape == (2, 3)
    assert np.allclose(aligned.sum(axis=1), 1.0)
    assert np.allclose(aligned[:, 1], 1.0)


def test_aligned_proba_uniform_fallback_when_row_sum_zero():
    # Model reports classes that don't intersect target labels at all.
    model = _FakeModelWithProba(["9"], [[1.0], [1.0]])
    aligned = aligned_proba(model, np.zeros((2, 2)), np.array([0, 1]))
    assert np.allclose(aligned, 0.5)


def test_mean_std_rows_groups_and_formats():
    rows = [
        {"dataset": "A", "model": "X", "accuracy": 0.8},
        {"dataset": "A", "model": "X", "accuracy": 0.9},
        {"dataset": "A", "model": "Y", "accuracy": 0.5},
    ]
    summary = mean_std_rows(rows, group_keys=["dataset", "model"], metric_keys=["accuracy"])
    by_model = {row["model"]: row for row in summary}
    assert math.isclose(by_model["X"]["accuracy_mean"], 0.85, rel_tol=1e-6)
    assert "+/-" in by_model["X"]["accuracy_pm"]
    assert by_model["Y"]["accuracy_std"] == 0.0
