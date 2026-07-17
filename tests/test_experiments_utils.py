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
    dataset_fingerprint,
    ensure_output_dirs,
    holm_bonferroni,
    mean_std_rows,
    paired_ttest_rows,
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
        ("MNIST2Class", "mnist2class"),
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
    assert bundle.resolved_numeric_columns == (0, 1)
    assert bundle.categorical_columns == ()
    assert bundle.raw_samples is None


def test_dataset_bundle_mixed_metadata_and_fingerprint_are_stable():
    X = np.array([[1.0, "A"], [None, "B"]], dtype=object)
    bundle = DatasetBundle(
        name="Mixed",
        X=X,
        y=np.array([0, 1], dtype=object),
        source="fixture",
        source_version="v1",
        selection_rule="all rows",
        description="mixed fixture",
        numeric_columns=(0,),
        categorical_columns=(1,),
        preprocessing="train-fitted fixture transform",
        raw_samples=2,
    )

    assert bundle.resolved_numeric_columns == (0,)
    assert dataset_fingerprint(bundle) == dataset_fingerprint(bundle)
    changed = DatasetBundle(
        name="Mixed",
        X=np.array([[1.0, "A"], [None, "C"]], dtype=object),
        y=bundle.y,
        source="fixture",
        description="mixed fixture",
        numeric_columns=(0,),
        categorical_columns=(1,),
    )
    assert dataset_fingerprint(changed) != dataset_fingerprint(bundle)


def test_dataset_bundle_rejects_invalid_shapes_and_column_metadata():
    base = {
        "name": "Invalid fixture",
        "source": "unit test",
        "description": "validation branches",
    }
    with pytest.raises(ValueError, match="2D"):
        DatasetBundle(X=np.zeros(2), y=np.array([0, 1], dtype=object), **base)
    with pytest.raises(ValueError, match="aligned"):
        DatasetBundle(X=np.zeros((2, 2)), y=np.array([0], dtype=object), **base)
    with pytest.raises(ValueError, match="overlap"):
        DatasetBundle(
            X=np.zeros((2, 2)),
            y=np.array([0, 1], dtype=object),
            numeric_columns=(0, 1),
            categorical_columns=(1,),
            **base,
        )
    with pytest.raises(ValueError, match="every feature"):
        DatasetBundle(
            X=np.zeros((2, 2)),
            y=np.array([0, 1], dtype=object),
            numeric_columns=(0,),
            **base,
        )


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


def test_paired_ttest_rows_matches_folds_and_detects_effect():
    rows = [
        {"dataset": "A", "fold": 1, "model": "Random Forest", "accuracy": 0.90},
        {"dataset": "A", "fold": 2, "model": "Random Forest", "accuracy": 0.92},
        {"dataset": "A", "fold": 3, "model": "Random Forest", "accuracy": 0.91},
        {"dataset": "A", "fold": 1, "model": "AdaBoost", "accuracy": 0.80},
        {"dataset": "A", "fold": 2, "model": "AdaBoost", "accuracy": 0.81},
        {"dataset": "A", "fold": 3, "model": "AdaBoost", "accuracy": 0.79},
        {"dataset": "A", "fold": 1, "model": "Single Tree", "accuracy": 0.5},
    ]
    result = paired_ttest_rows(
        rows,
        group_keys=["dataset"],
        pair_column="model",
        pair_values=("Random Forest", "AdaBoost"),
        fold_key="fold",
        metric_keys=["accuracy"],
    )
    assert len(result) == 1
    row = result[0]
    assert row["n_folds"] == 3
    assert math.isclose(row["accuracy_mean_diff"], 0.11, rel_tol=1e-6)
    assert row["accuracy_t_stat"] > 0
    assert 0 <= row["accuracy_p_value"] <= 1


def test_paired_ttest_rows_zero_difference_gives_nan_stat():
    rows = [
        {"dataset": "A", "fold": 1, "model": "Random Forest", "accuracy": 0.9},
        {"dataset": "A", "fold": 2, "model": "Random Forest", "accuracy": 0.9},
        {"dataset": "A", "fold": 1, "model": "AdaBoost", "accuracy": 0.9},
        {"dataset": "A", "fold": 2, "model": "AdaBoost", "accuracy": 0.9},
    ]
    result = paired_ttest_rows(
        rows,
        group_keys=["dataset"],
        pair_column="model",
        pair_values=("Random Forest", "AdaBoost"),
        fold_key="fold",
        metric_keys=["accuracy"],
    )
    row = result[0]
    assert row["accuracy_mean_diff"] == 0.0
    assert math.isnan(row["accuracy_t_stat"])


def test_paired_ttest_rows_adds_holm_adjusted_p_value_per_metric():
    rows = [
        {"dataset": "A", "fold": 1, "model": "Random Forest", "accuracy": 0.95},
        {"dataset": "A", "fold": 2, "model": "Random Forest", "accuracy": 0.97},
        {"dataset": "A", "fold": 1, "model": "AdaBoost", "accuracy": 0.80},
        {"dataset": "A", "fold": 2, "model": "AdaBoost", "accuracy": 0.83},
        {"dataset": "B", "fold": 1, "model": "Random Forest", "accuracy": 0.90},
        {"dataset": "B", "fold": 2, "model": "Random Forest", "accuracy": 0.94},
        {"dataset": "B", "fold": 1, "model": "AdaBoost", "accuracy": 0.91},
        {"dataset": "B", "fold": 2, "model": "AdaBoost", "accuracy": 0.93},
    ]
    result = paired_ttest_rows(
        rows,
        group_keys=["dataset"],
        pair_column="model",
        pair_values=("Random Forest", "AdaBoost"),
        fold_key="fold",
        metric_keys=["accuracy"],
    )
    by_dataset = {row["dataset"]: row for row in result}
    # Holm adjustment is never smaller than the raw p-value.
    for row in result:
        assert row["accuracy_p_value_holm"] >= row["accuracy_p_value"] - 1e-12
        assert 0 <= row["accuracy_p_value_holm"] <= 1
    # The larger raw p-value (dataset B, near-identical models) should not
    # end up with a smaller adjusted p-value than dataset A's.
    assert by_dataset["B"]["accuracy_p_value_holm"] >= by_dataset["A"]["accuracy_p_value_holm"]


def test_holm_bonferroni_matches_hand_computed_example():
    # Classic textbook-style example: three p-values, alpha irrelevant here
    # since this checks the adjusted values themselves.
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    # Sorted ascending: 0.01 (rank0, x3), 0.03 (rank1, x2), 0.04 (rank2, x1),
    # each enforced non-decreasing.
    assert math.isclose(adjusted[0], 0.03, rel_tol=1e-9)  # 0.01 * 3
    assert math.isclose(adjusted[2], 0.06, rel_tol=1e-9)  # 0.03 * 2
    assert math.isclose(adjusted[1], 0.06, rel_tol=1e-9)  # 0.04 * 1, but monotonicity floors it at 0.06


def test_holm_bonferroni_caps_at_one():
    adjusted = holm_bonferroni([0.9, 0.95])
    assert all(value <= 1.0 for value in adjusted)


def test_holm_bonferroni_preserves_nan_position_and_value():
    # A single undefined test alongside no other comparisons: NaN must stay
    # NaN, not collapse to 0.0 or 1.0.
    adjusted = holm_bonferroni([float("nan")])
    assert len(adjusted) == 1
    assert math.isnan(adjusted[0])


def test_holm_bonferroni_excludes_nan_from_correction_family():
    # Regression test for the documented example: NaN must not participate
    # in (and must not shrink the correction factor for) the finite family.
    adjusted = holm_bonferroni([0.001, float("nan"), 0.04])
    assert math.isnan(adjusted[1])
    assert math.isclose(adjusted[0], 0.002, rel_tol=1e-9)
    assert math.isclose(adjusted[2], 0.04, rel_tol=1e-9)


def test_holm_bonferroni_finite_only_results_unchanged_by_nan_handling():
    # With no NaNs present, behavior must be identical to the pre-fix
    # implementation (same values as test_holm_bonferroni_matches_hand_computed_example).
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    assert math.isclose(adjusted[0], 0.03, rel_tol=1e-9)
    assert math.isclose(adjusted[2], 0.06, rel_tol=1e-9)
    assert math.isclose(adjusted[1], 0.06, rel_tol=1e-9)
    assert not any(math.isnan(value) for value in adjusted)
