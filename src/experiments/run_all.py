"""Run every experiment required by the final-project statement.

The script writes CSV tables under ``data/results`` and publication
figures under ``figures``. It intentionally uses sklearn only for data
loading, metrics, cross-validation utilities, t-SNE, and explicitly
marked reference baselines.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

# Keep numerical-library thread pools deterministic and prevent shutdown hangs.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure console logging for the experiment driver.

    Progress messages previously went through bare ``print()`` calls,
    which gives no level/timestamp information and can't be redirected,
    filtered, or silenced independently of the script's actual output
    files. This sets up a standard ``logging`` console handler instead
    so progress messages behave like normal application logs.
    """

    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.boosting.gradient_boosting import GradientBoostingClassifier
from src.experiments.datasets import load_all_datasets
from src.experiments.utils import (
    DatasetBundle,
    aligned_proba,
    dataset_fingerprint,
    ensure_output_dirs,
    mean_std_rows,
    paired_ttest_rows,
    slugify,
    write_csv,
    write_json,
)
from src.metrics.evaluation import bias_variance_01, classification_metrics
from src.trees.decision_tree import DecisionStump, DecisionTree
from src.unsupervised.dbscan import DBSCAN, kth_neighbor_distances
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA
from src.utils.preprocessing import (
    MixedTypePreprocessor,
    class_distribution,
    flip_labels,
    random_oversample,
    stratified_train_test_indices,
)


# Experiment-wide hyperparameter constants. Every experiment function below
# reuses these instead of repeating literal values, so there is a single
# place to change the AdaBoost learning rate, Random Forest feature-subset
# strategy, or the severe-imbalance oversampling ratio. See report.tex,
# "Hyperparameter and Design Choices" for the justification of each value.
ADABOOST_LEARNING_RATE = 0.6
RANDOM_FOREST_MAX_FEATURES = "sqrt"
SEVERE_IMBALANCE_TARGET_RATIO = 0.25
CANONICAL_DATASET_ROWS = {
    "Breast Cancer Wisconsin": 569,
    "Adult Income": 48_842,
    "Covertype": 50_000,
    "MNIST2Class": 14_780,
}
RESULT_TABLE_ROWS = {
    "baseline_metrics.csv": 16,
    "adaboost_scaling.csv": 84,
    "random_forest_estimators.csv": 28,
    "random_forest_depth.csv": 32,
    "head_to_head_cv.csv": 80,
    "head_to_head_summary.csv": 16,
    "head_to_head_significance.csv": 4,
    "noise_robustness.csv": 24,
    "bias_variance.csv": 4,
    "unsupervised_summary.csv": 4,
    "gradient_boosting_bonus.csv": 2,
}


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_fit: np.ndarray
    y_fit: np.ndarray


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Avoid network access and require all raw/interim caches to exist.",
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Validate and reuse a complete matching canonical result set instead of fitting models.",
    )
    parser.add_argument(
        "--recompute-heavy",
        action="store_true",
        help="Backward-compatible alias for the default full-recomputation behavior.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Workers for custom and sklearn Random Forest fits (use 1 on limited-memory Windows systems).",
    )
    parser.add_argument("--_rf-depth-worker-input", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_rf-depth-worker-output", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a positive integer")
    if args.reuse_results and args.recompute_heavy:
        parser.error("--reuse-results and --recompute-heavy cannot be combined")

    if args._rf_depth_worker_input is not None:
        if args._rf_depth_worker_output is None:
            raise ValueError("RF depth worker output path is required")
        run_rf_depth_worker(
            Path(args._rf_depth_worker_input),
            Path(args._rf_depth_worker_output),
        )
        return

    figures_dir, results_dir = ensure_output_dirs(ROOT)
    datasets = load_datasets(args.seed, skip_downloads=args.skip_downloads)
    metadata = build_dataset_metadata(datasets, args.seed)
    if args.reuse_results:
        validate_reusable_outputs(results_dir, figures_dir, metadata, args.seed)
        logger.info("Validated and reused the complete canonical result set in %s", results_dir)
        return

    write_json(results_dir / "dataset_metadata.json", metadata)
    timings: dict[str, float] = {}

    logger.info("Running baselines...")
    baseline_rows = _timed(timings, "baselines", run_baselines, datasets, results_dir, args.seed)
    logger.info("Running AdaBoost scaling...")
    _timed(timings, "adaboost_scaling", run_adaboost_scaling, datasets, figures_dir, results_dir, args.seed)
    logger.info("Running Random Forest scaling...")
    _timed(
        timings,
        "random_forest_scaling",
        run_rf_scaling,
        datasets,
        figures_dir,
        results_dir,
        args.seed,
        args.n_jobs,
    )
    logger.info("Running head-to-head cross-validation...")
    head_to_head_rows = _timed(
        timings,
        "head_to_head_cv",
        run_head_to_head,
        datasets,
        results_dir,
        args.seed,
        args.n_jobs,
    )
    logger.info("Running noise robustness...")
    _timed(
        timings,
        "noise_robustness",
        run_noise_robustness,
        datasets,
        figures_dir,
        results_dir,
        args.seed,
        args.n_jobs,
    )
    logger.info("Running Breast Cancer bias-variance decomposition...")
    _timed(
        timings,
        "bias_variance",
        run_bias_variance,
        datasets[0],
        figures_dir,
        results_dir,
        args.seed,
        args.n_jobs,
    )
    logger.info("Running unsupervised analysis...")
    _timed(timings, "unsupervised", run_unsupervised, datasets, figures_dir, results_dir, args.seed)
    logger.info("Running Gradient Boosting bonus...")
    _timed(
        timings,
        "gradient_boosting_bonus",
        run_gradient_boosting_bonus,
        datasets[0],
        figures_dir,
        results_dir,
        args.seed,
    )

    write_json(
        results_dir / "run_summary.json",
        {
            "seed": args.seed,
            "datasets": metadata,
            "baseline_rows": len(baseline_rows),
            "head_to_head_rows": len(head_to_head_rows),
            "run_mode": "full_recomputation",
            "n_jobs": args.n_jobs,
            "experiment_contract": experiment_contract(args.seed, metadata),
            "timings_seconds": timings,
            "outputs": {
                "results_dir": "data/results",
                "figures_dir": "figures",
            },
        },
    )
    validate_generated_outputs(results_dir, figures_dir)
    logger.info("Wrote results to %s", results_dir)
    logger.info("Wrote figures to %s", figures_dir)


def _timed(timings: dict[str, float], name: str, function: Any, *args: Any) -> Any:
    started = time.perf_counter()
    result = function(*args)
    timings[name] = round(time.perf_counter() - started, 3)
    return result


def build_dataset_metadata(datasets: list[DatasetBundle], seed: int) -> dict[str, Any]:
    actual = {dataset.name: int(dataset.X.shape[0]) for dataset in datasets}
    if actual != CANONICAL_DATASET_ROWS:
        raise RuntimeError(f"Dataset contract mismatch: expected {CANONICAL_DATASET_ROWS}, got {actual}")
    return {
        dataset.name: {
            "source": dataset.source,
            "source_version": dataset.source_version,
            "description": dataset.description,
            "selection_rule": dataset.selection_rule,
            "samples": int(dataset.X.shape[0]),
            "raw_samples": dataset.raw_samples,
            "features": int(dataset.X.shape[1]),
            "class_distribution": class_distribution(dataset.y),
            "preprocessing": dataset.preprocessing,
            "severe_imbalance": dataset.severe_imbalance,
            "high_dimensional": dataset.high_dimensional,
            "seed": seed,
            "fingerprint_sha256": dataset_fingerprint(dataset),
        }
        for dataset in datasets
    }


def experiment_contract(seed: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "dataset_names_and_rows": CANONICAL_DATASET_ROWS,
        "dataset_fingerprints": {
            name: values["fingerprint_sha256"] for name, values in metadata.items()
        },
        "grids": {
            "adaboost_estimators": [1, *range(10, 201, 10)],
            "random_forest_estimators": [1, 10, 25, 50, 100, 150, 200],
            "random_forest_depth": [1, 2, 3, 5, 8, 12, 16, 20],
            "cross_validation_folds": 5,
            "noise_fractions": [0.05, 0.1, 0.2],
            "bias_variance_bootstraps": 100,
            "dbscan_k": 5,
        },
        "expected_table_rows": RESULT_TABLE_ROWS,
    }


def load_datasets(seed: int, skip_downloads: bool = False) -> list[DatasetBundle]:
    """Compatibility wrapper around the isolated authoritative loaders."""

    return load_all_datasets(seed=seed, skip_downloads=skip_downloads, root=ROOT)


def prepare_index_split(
    dataset: DatasetBundle,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> SplitData:
    """Fit preprocessing on train indices and apply treatment to train only."""

    preprocessor = MixedTypePreprocessor(
        dataset.resolved_numeric_columns,
        dataset.categorical_columns,
    )
    X_train_scaled = preprocessor.fit_transform(dataset.X[train_indices])
    X_test_scaled = preprocessor.transform(dataset.X[test_indices])
    y_train = dataset.y[train_indices].astype(object)
    y_test = dataset.y[test_indices].astype(object)
    if dataset.severe_imbalance:
        X_fit, y_fit = random_oversample(
            X_train_scaled,
            y_train,
            random_state=seed,
            target_ratio=SEVERE_IMBALANCE_TARGET_RATIO,
        )
    else:
        X_fit, y_fit = X_train_scaled, y_train
    return SplitData(X_train_scaled, X_test_scaled, y_train, y_test, X_fit, y_fit)


def prepare_split(dataset: DatasetBundle, seed: int, test_size: float = 0.2) -> SplitData:
    train_indices, test_indices = stratified_train_test_indices(
        dataset.y,
        test_size=test_size,
        random_state=seed,
    )
    return prepare_index_split(dataset, train_indices, test_indices, seed)


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    pred = model.predict(X_test)
    proba = aligned_proba(model, X_test, labels)
    return classification_metrics(y_test, pred, proba, labels=labels)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read an existing result table for validation/replotting."""

    csv_module = __import__("csv")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv_module.DictReader(handle))


def validate_and_replot_heavy_outputs(results_dir: Path, figures_dir: Path) -> list[dict[str, Any]]:
    """Validate exact canonical long-sweep CSVs and regenerate aggregate figures."""

    expected_tables: dict[str, tuple[set[str], int]] = {
        "random_forest_estimators.csv": ({"dataset", "n_estimators", "test_accuracy", "oob_accuracy"}, 28),
        "random_forest_depth.csv": ({"dataset", "max_depth", "test_accuracy", "oob_accuracy"}, 32),
        "head_to_head_cv.csv": ({"dataset", "fold", "model", "accuracy", "macro_f1", "auc_roc"}, 80),
        "head_to_head_summary.csv": ({"dataset", "model", "accuracy_mean", "accuracy_std"}, 16),
        "noise_robustness.csv": ({"dataset", "noise_fraction", "model", "accuracy"}, 24),
        "bias_variance.csv": ({"dataset", "model", "bias_squared", "variance", "expected_loss"}, 4),
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    for filename, (required_columns, exact_rows) in expected_tables.items():
        rows = read_csv_rows(results_dir / filename)
        if len(rows) != exact_rows:
            raise RuntimeError(f"{filename} has {len(rows)} rows; expected exactly {exact_rows}")
        if rows:
            missing = required_columns.difference(rows[0].keys())
            if missing:
                raise RuntimeError(f"{filename} is missing required columns: {sorted(missing)}")
            expected_names = (
                {"Breast Cancer Wisconsin"}
                if filename == "bias_variance.csv"
                else set(CANONICAL_DATASET_ROWS)
            )
            actual_names = {str(row["dataset"]) for row in rows}
            if actual_names != expected_names:
                raise RuntimeError(
                    f"{filename} dataset set mismatch: expected {sorted(expected_names)}, got {sorted(actual_names)}"
                )
        loaded[filename] = rows

    replot_random_forest_scaling(
        loaded["random_forest_estimators.csv"],
        loaded["random_forest_depth.csv"],
        figures_dir,
    )
    replot_noise_robustness(loaded["noise_robustness.csv"], figures_dir)
    replot_bias_variance(loaded["bias_variance.csv"], figures_dir)
    return loaded["head_to_head_cv.csv"]


def validate_reusable_outputs(
    results_dir: Path,
    figures_dir: Path,
    metadata: dict[str, Any],
    seed: int,
) -> None:
    """Reject stale reuse whenever data, seed, grids, rows, or figures differ."""

    metadata_path = results_dir / "dataset_metadata.json"
    summary_path = results_dir / "run_summary.json"
    if not metadata_path.exists() or not summary_path.exists():
        raise RuntimeError("--reuse-results requires dataset_metadata.json and run_summary.json")
    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if stored_metadata != metadata:
        raise RuntimeError("Stored dataset metadata/fingerprints do not match the loaded four-dataset contract")
    stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_contract = experiment_contract(seed, metadata)
    if stored_summary.get("experiment_contract") != expected_contract:
        raise RuntimeError("Stored seed, dataset contract, grids, or fingerprints are stale")
    validate_generated_outputs(results_dir, figures_dir)
    validate_and_replot_heavy_outputs(results_dir, figures_dir)


def validate_generated_outputs(results_dir: Path, figures_dir: Path) -> None:
    """Require exact canonical result-table rows, dataset sets, and 23 figures."""

    four_dataset_tables = {
        "baseline_metrics.csv",
        "adaboost_scaling.csv",
        "random_forest_estimators.csv",
        "random_forest_depth.csv",
        "head_to_head_cv.csv",
        "head_to_head_summary.csv",
        "head_to_head_significance.csv",
        "noise_robustness.csv",
        "unsupervised_summary.csv",
    }
    for filename, expected_rows in RESULT_TABLE_ROWS.items():
        path = results_dir / filename
        if not path.exists():
            raise RuntimeError(f"Missing required result table: {filename}")
        rows = read_csv_rows(path)
        if len(rows) != expected_rows:
            raise RuntimeError(f"{filename} has {len(rows)} rows; expected exactly {expected_rows}")
        expected_names = (
            set(CANONICAL_DATASET_ROWS)
            if filename in four_dataset_tables
            else {"Breast Cancer Wisconsin"}
        )
        actual_names = {str(row.get("dataset", "")) for row in rows}
        if actual_names != expected_names:
            raise RuntimeError(
                f"{filename} dataset set mismatch: expected {sorted(expected_names)}, got {sorted(actual_names)}"
            )

    expected_figures = {
        "adaboost_scaling.pdf",
        "random_forest_estimators.pdf",
        "random_forest_depth.pdf",
        "noise_robustness.pdf",
        "bias_variance.pdf",
        "gradient_boosting_bonus.pdf",
        "mnist2class_tsne_bonus.pdf",
    }
    for dataset_name in CANONICAL_DATASET_ROWS:
        slug = slugify(dataset_name)
        for suffix in ("pca_scree", "kmeans_elbow", "dbscan_kdistance", "pca_clusters"):
            expected_figures.add(f"{slug}_{suffix}.pdf")
    actual_figures = {path.name for path in figures_dir.glob("*.pdf")}
    if actual_figures != expected_figures:
        missing = sorted(expected_figures - actual_figures)
        extra = sorted(actual_figures - expected_figures)
        raise RuntimeError(f"Figure contract mismatch; missing={missing}, stale_or_extra={extra}")


def _as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _dataset_order(rows: list[dict[str, Any]]) -> list[str]:
    order: list[str] = []
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in order:
            order.append(dataset)
    return order


def _dataset_figure_axes(
    n_datasets: int,
    sharey: bool = True,
) -> tuple[Any, list[Any]]:
    """Create a readable 2x2 aggregate layout for the four-study contract."""

    if n_datasets < 1:
        raise ValueError("At least one dataset is required for an aggregate figure")
    if n_datasets == 1:
        fig, axis = plt.subplots(figsize=(5, 3.5))
        return fig, [axis]
    fig, grid = plt.subplots(2, 2, figsize=(10, 7), sharey=sharey)
    axes = list(np.asarray(grid).ravel())
    for axis in axes[n_datasets:]:
        axis.set_visible(False)
    return fig, axes[:n_datasets]


def replot_random_forest_scaling(
    estimator_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    datasets = _dataset_order(estimator_rows)
    fig_n, axes_n = _dataset_figure_axes(len(datasets))
    fig_d, axes_d = _dataset_figure_axes(len(datasets))
    for ax_n, ax_d, dataset in zip(axes_n, axes_d, datasets):
        rows_n = [row for row in estimator_rows if row["dataset"] == dataset]
        rows_n.sort(key=lambda row: int(float(row["n_estimators"])))
        n_grid = [int(float(row["n_estimators"])) for row in rows_n]
        test_acc = [_as_float(row, "test_accuracy") for row in rows_n]
        oob_acc = [_as_float(row, "oob_accuracy") for row in rows_n]
        ax_n.plot(n_grid, test_acc, marker="o", label="Test")
        ax_n.plot(n_grid, oob_acc, marker="s", label="OOB")
        ax_n.set_title(dataset)
        ax_n.set_xlabel("Trees")
        ax_n.grid(True, alpha=0.25)

        rows_d = [row for row in depth_rows if row["dataset"] == dataset]
        rows_d.sort(key=lambda row: int(float(row["max_depth"])))
        depth_grid = [int(float(row["max_depth"])) for row in rows_d]
        depth_acc = [_as_float(row, "test_accuracy") for row in rows_d]
        ax_d.plot(depth_grid, depth_acc, marker="o")
        ax_d.set_title(dataset)
        ax_d.set_xlabel("Max depth")
        ax_d.grid(True, alpha=0.25)
    axes_n[0].set_ylabel("Accuracy")
    axes_n[len(datasets) - 1].legend(loc="best", fontsize=8)
    axes_d[0].set_ylabel("Accuracy")
    fig_n.tight_layout()
    fig_d.tight_layout()
    fig_n.savefig(figures_dir / "random_forest_estimators.pdf")
    fig_d.savefig(figures_dir / "random_forest_depth.pdf")
    plt.close(fig_n)
    plt.close(fig_d)


def replot_noise_robustness(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    datasets = _dataset_order(rows)
    fig, axes = _dataset_figure_axes(len(datasets))
    for ax, dataset in zip(axes, datasets):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        models: list[str] = []
        for row in dataset_rows:
            model = str(row["model"])
            if model not in models:
                models.append(model)
        for model in models:
            model_rows = [row for row in dataset_rows if row["model"] == model]
            model_rows.sort(key=lambda row: float(row["noise_fraction"]))
            x = [float(row["noise_fraction"]) for row in model_rows]
            y = [float(row["accuracy"]) for row in model_rows]
            ax.plot(x, y, marker="o", label=model)
        ax.set_title(dataset)
        ax.set_xlabel("Label-noise fraction")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Clean-test accuracy")
    axes[len(datasets) - 1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "noise_robustness.pdf")
    plt.close(fig)


def replot_bias_variance(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(rows))
    bias = [float(row["bias_squared"]) for row in rows]
    variance = [float(row["variance"]) for row in rows]
    width = 0.35
    ax.bar(x - width / 2, bias, width, label="Bias^2")
    ax.bar(x + width / 2, variance, width, label="Variance")
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["model"]) for row in rows], rotation=10, ha="right")
    ax.set_ylabel("0-1 decomposition component")
    ax.set_title("Bias-variance on Breast Cancer Wisconsin")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "bias_variance.pdf")
    plt.close(fig)


def fit_estimator(model: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit sklearn reference models with string labels; keep project labels raw."""

    if model.__class__.__module__.startswith("sklearn."):
        return model.fit(X, np.asarray(y).astype(str))
    return model.fit(X, y)


def run_baselines(
    datasets: list[DatasetBundle],
    results_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        split = prepare_split(dataset, seed)
        labels = np.unique(dataset.y).astype(object)
        models: list[tuple[str, Any]] = [
            (
                "Decision Tree (ours)",
                DecisionTree(criterion="entropy", random_state=seed),
            ),
            (
                "Decision Stump (ours)",
                DecisionStump(criterion="entropy", random_state=seed),
            ),
            (
                "Decision Tree (sklearn reference)",
                SklearnDecisionTreeClassifier(criterion="entropy", random_state=seed),
            ),
        ]
        for name, model in models:
            fit_estimator(model, split.X_fit, split.y_fit)
            metrics = evaluate_model(model, split.X_test, split.y_test, labels)
            rows.append({"dataset": dataset.name, "model": name, **metrics})

        own_acc = rows[-3]["accuracy"]
        sklearn_acc = rows[-1]["accuracy"]
        rows.append(
            {
                "dataset": dataset.name,
                "model": "Tree accuracy gap vs sklearn",
                "accuracy": round(abs(float(own_acc) - float(sklearn_acc)), 6),
                "macro_f1": np.nan,
                "auc_roc": np.nan,
            }
        )
    write_csv(results_dir / "baseline_metrics.csv", rows)
    return rows


def run_adaboost_scaling(
    datasets: list[DatasetBundle],
    figures_dir: Path,
    results_dir: Path,
    seed: int,
) -> None:
    rows: list[dict[str, Any]] = []
    fig, axes = _dataset_figure_axes(len(datasets))
    for ax, dataset in zip(axes, datasets):
        split = prepare_split(dataset, seed)
        model = AdaBoostClassifier(n_estimators=200, learning_rate=ADABOOST_LEARNING_RATE, random_state=seed)
        model.fit(split.X_fit, split.y_fit)
        train_errors: list[float] = []
        test_errors: list[float] = []
        rounds: list[int] = []
        staged_train = list(model.staged_predict(split.X_train))
        staged_test = list(model.staged_predict(split.X_test))
        checkpoints = [1, *range(10, 201, 10)]
        for round_index in checkpoints:
            effective = min(round_index, len(staged_train))
            train_pred = staged_train[effective - 1]
            test_pred = staged_test[effective - 1]
            train_error = 1.0 - float(np.mean(train_pred == split.y_train))
            test_error = 1.0 - float(np.mean(test_pred == split.y_test))
            rounds.append(round_index)
            train_errors.append(train_error)
            test_errors.append(test_error)
            rows.append(
                {
                    "dataset": dataset.name,
                    "n_estimators": round_index,
                    "effective_estimators": effective,
                    "train_error": train_error,
                    "test_error": test_error,
                }
            )
        ax.plot(rounds, train_errors, marker="o", label="Train error")
        ax.plot(rounds, test_errors, marker="s", label="Test error")
        ax.set_title(dataset.name)
        ax.set_xlabel("Boosting rounds")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Error")
    axes[len(datasets) - 1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "adaboost_scaling.pdf")
    plt.close(fig)
    write_csv(results_dir / "adaboost_scaling.csv", rows)


def run_rf_scaling(
    datasets: list[DatasetBundle],
    figures_dir: Path,
    results_dir: Path,
    seed: int,
    n_jobs: int = 1,
) -> None:
    estimator_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    n_grid = [1, 10, 25, 50, 100, 150, 200]
    depth_grid = [1, 2, 3, 5, 8, 12, 16, 20]

    fig_n, axes_n = _dataset_figure_axes(len(datasets))
    fig_d, axes_d = _dataset_figure_axes(len(datasets))

    for ax_n, ax_d, dataset in zip(axes_n, axes_d, datasets):
        logger.info("RF scaling: %s", dataset.name)
        split = prepare_split(dataset, seed)
        forest = RandomForestClassifier(
            n_estimators=max(n_grid),
            max_depth=None,
            max_features=RANDOM_FOREST_MAX_FEATURES,
            oob_score=True,
            n_jobs=n_jobs,
            random_state=seed,
        ).fit(split.X_fit, split.y_fit)
        test_acc: list[float] = []
        oob_acc: list[float] = []
        for n_estimators in n_grid:
            logger.debug("n_estimators=%d", n_estimators)
            pred = rf_prefix_predict(forest, split.X_test, n_estimators)
            acc = float(np.mean(pred == split.y_test))
            oob = rf_prefix_oob_score(forest, split.X_fit, split.y_fit, n_estimators)
            test_acc.append(acc)
            oob_acc.append(oob)
            estimator_rows.append(
                {
                    "dataset": dataset.name,
                    "n_estimators": n_estimators,
                    "test_accuracy": acc,
                    "oob_accuracy": oob,
                }
            )
        ax_n.plot(n_grid, test_acc, marker="o", label="Test")
        ax_n.plot(n_grid, oob_acc, marker="s", label="OOB")
        ax_n.set_title(dataset.name)
        ax_n.set_xlabel("Trees")
        ax_n.grid(True, alpha=0.25)

        depth_acc: list[float] = []
        for depth in depth_grid:
            logger.debug("max_depth=%s", depth)
            metrics = fit_rf_depth_in_worker(
                split.X_fit,
                split.y_fit,
                split.X_test,
                split.y_test,
                depth=depth,
                seed=seed + depth,
                n_jobs=n_jobs,
                results_dir=results_dir,
            )
            acc = metrics["test_accuracy"]
            depth_acc.append(acc)
            depth_rows.append(
                {
                    "dataset": dataset.name,
                    "max_depth": depth,
                    "test_accuracy": acc,
                    "oob_accuracy": metrics["oob_accuracy"],
                }
            )
        ax_d.plot(depth_grid, depth_acc, marker="o")
        ax_d.set_title(dataset.name)
        ax_d.set_xlabel("Max depth")
        ax_d.grid(True, alpha=0.25)
        del forest
        gc.collect()

    axes_n[0].set_ylabel("Accuracy")
    axes_n[len(datasets) - 1].legend(loc="best", fontsize=8)
    axes_d[0].set_ylabel("Accuracy")
    fig_n.tight_layout()
    fig_d.tight_layout()
    fig_n.savefig(figures_dir / "random_forest_estimators.pdf")
    fig_d.savefig(figures_dir / "random_forest_depth.pdf")
    plt.close(fig_n)
    plt.close(fig_d)
    write_csv(results_dir / "random_forest_estimators.csv", estimator_rows)
    write_csv(results_dir / "random_forest_depth.csv", depth_rows)
    gc.collect()


def fit_rf_depth_in_worker(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    depth: int,
    seed: int,
    n_jobs: int,
    results_dir: Path,
) -> dict[str, float]:
    """Fit one depth-scaling forest in a short-lived process.

    The from-scratch trees are intentionally transparent rather than C-optimized.
    Isolating the heaviest Random Forest depth fits keeps the required
    one-command experiment runner reliable across local machines, CI, and
    notebook kernels that may otherwise retain native numerical-library state.
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=results_dir) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        input_path = tmp_dir / f"rf_depth_{depth}_input.npz"
        output_path = tmp_dir / f"rf_depth_{depth}_output.json"
        np.savez_compressed(
            input_path,
            X_fit=np.asarray(X_fit, dtype=float),
            y_fit=np.asarray(y_fit).astype(str),
            X_test=np.asarray(X_test, dtype=float),
            y_test=np.asarray(y_test).astype(str),
            depth=np.array([depth], dtype=int),
            seed=np.array([seed], dtype=int),
            n_jobs=np.array([n_jobs], dtype=int),
        )
        child_env = os.environ.copy()
        for thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            child_env[thread_env] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_rf-depth-worker-input",
            str(input_path),
            "--_rf-depth-worker-output",
            str(output_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr_text = ""
        try:
            for _ in range(14_400):  # Up to two hours for full-data single-worker fits.
                if output_path.exists() and output_path.stat().st_size > 0:
                    break
                if process.poll() is not None:
                    stderr_text = process.stderr.read() if process.stderr is not None else ""
                    break
                __import__("time").sleep(0.5)
            else:
                process.kill()
                raise TimeoutError(f"RF depth worker timed out for depth={depth}")

            if not output_path.exists() or output_path.stat().st_size == 0:
                return_code = process.poll()
                raise RuntimeError(
                    f"RF depth worker failed for depth={depth} with code {return_code}: {stderr_text}"
                )
            payload = output_path.read_text(encoding="utf-8")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            if process.stderr is not None:
                process.stderr.close()
    loaded = __import__("json").loads(payload)
    return {"test_accuracy": float(loaded["test_accuracy"]), "oob_accuracy": float(loaded["oob_accuracy"])}


def run_rf_depth_worker(input_path: Path, output_path: Path) -> None:
    """Worker entry point for one Random Forest max-depth experiment."""

    payload = np.load(input_path, allow_pickle=False)
    X_fit = payload["X_fit"].astype(float)
    y_fit = payload["y_fit"].astype(str)
    X_test = payload["X_test"].astype(float)
    y_test = payload["y_test"].astype(str)
    depth = int(payload["depth"][0])
    seed = int(payload["seed"][0])
    n_jobs = int(payload["n_jobs"][0]) if "n_jobs" in payload.files else 1
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        max_features=RANDOM_FOREST_MAX_FEATURES,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=seed,
    ).fit(X_fit, y_fit)
    pred = model.predict(X_test).astype(str)
    write_json(
        output_path,
        {
            "test_accuracy": float(np.mean(pred == y_test)),
            "oob_accuracy": model.oob_score_,
        },
    )


def rf_prefix_predict(
    forest: RandomForestClassifier,
    X: np.ndarray,
    n_estimators: int,
) -> np.ndarray:
    """Predict using hard majority vote from the first ``n_estimators`` trees."""

    assert forest.classes_ is not None
    votes = np.zeros((X.shape[0], forest.n_classes_), dtype=float)
    for tree in forest.estimators_[:n_estimators]:
        predictions = tree.predict(X)
        for class_index, label in enumerate(forest.classes_):
            votes[:, class_index] += predictions == label
    return forest.classes_[np.argmax(votes, axis=1)]


def rf_prefix_oob_score(
    forest: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
) -> float:
    """OOB accuracy from hard votes over the first ``n_estimators`` trees."""

    assert forest.classes_ is not None
    votes = np.zeros((X.shape[0], forest.n_classes_), dtype=float)
    counts = np.zeros(X.shape[0], dtype=int)
    for tree, oob in zip(forest.estimators_[:n_estimators], forest.oob_indices_[:n_estimators]):
        if oob.size == 0:
            continue
        predictions = tree.predict(X[oob])
        for class_index, label in enumerate(forest.classes_):
            votes[oob, class_index] += predictions == label
        counts[oob] += 1
    mask = counts > 0
    if not np.any(mask):
        return float("nan")
    pred = forest.classes_[np.argmax(votes[mask], axis=1)]
    return float(np.mean(pred == y[mask]))


def run_head_to_head(
    datasets: list[DatasetBundle],
    results_dir: Path,
    seed: int,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        labels, counts = np.unique(dataset.y, return_counts=True)
        n_splits = min(5, int(np.min(counts)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_labels = dataset.y.astype(str)
        for fold, (train_idx, test_idx) in enumerate(cv.split(dataset.X, split_labels), start=1):
            split = prepare_index_split(dataset, train_idx, test_idx, seed + fold)

            models: list[tuple[str, Any]] = [
                ("Single Tree", DecisionTree(random_state=seed + fold)),
                (
                    "AdaBoost",
                    AdaBoostClassifier(
                        n_estimators=100,
                        learning_rate=ADABOOST_LEARNING_RATE,
                        random_state=seed + fold,
                    ),
                ),
                (
                    "Random Forest",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_features=RANDOM_FOREST_MAX_FEATURES,
                        oob_score=True,
                        n_jobs=n_jobs,
                        random_state=seed + fold,
                    ),
                ),
                (
                    "sklearn RF reference",
                    SklearnRandomForestClassifier(
                        n_estimators=100,
                        max_features=RANDOM_FOREST_MAX_FEATURES,
                        n_jobs=n_jobs,
                        random_state=seed + fold,
                    ),
                ),
            ]
            for model_name, model in models:
                fit_estimator(model, split.X_fit, split.y_fit)
                metrics = evaluate_model(model, split.X_test, split.y_test, labels.astype(object))
                rows.append(
                    {
                        "dataset": dataset.name,
                        "fold": fold,
                        "model": model_name,
                        **metrics,
                    }
                )
    summary = mean_std_rows(
        rows,
        group_keys=["dataset", "model"],
        metric_keys=["accuracy", "macro_f1", "auc_roc"],
    )
    significance = paired_ttest_rows(
        rows,
        group_keys=["dataset"],
        pair_column="model",
        pair_values=("Random Forest", "AdaBoost"),
        fold_key="fold",
        metric_keys=["accuracy", "macro_f1", "auc_roc"],
    )
    write_csv(results_dir / "head_to_head_cv.csv", rows)
    write_csv(results_dir / "head_to_head_summary.csv", summary)
    write_csv(results_dir / "head_to_head_significance.csv", significance)
    return rows


def run_noise_robustness(
    datasets: list[DatasetBundle],
    figures_dir: Path,
    results_dir: Path,
    seed: int,
    n_jobs: int = 1,
) -> None:
    noise_levels = [0.05, 0.10, 0.20]
    rows: list[dict[str, Any]] = []
    fig, axes = _dataset_figure_axes(len(datasets))
    for ax, dataset in zip(axes, datasets):
        split = prepare_split(dataset, seed)
        model_points: dict[str, list[float]] = {"AdaBoost": [], "Random Forest": []}
        for noise in noise_levels:
            y_noisy = flip_labels(split.y_train, noise, random_state=seed + int(noise * 100))
            if dataset.severe_imbalance:
                X_fit, y_fit = random_oversample(
                    split.X_train,
                    y_noisy,
                    random_state=seed,
                    target_ratio=SEVERE_IMBALANCE_TARGET_RATIO,
                )
            else:
                X_fit, y_fit = split.X_train, y_noisy
            models: list[tuple[str, Any]] = [
                (
                    "AdaBoost",
                    AdaBoostClassifier(n_estimators=100, learning_rate=ADABOOST_LEARNING_RATE, random_state=seed),
                ),
                (
                    "Random Forest",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_features=RANDOM_FOREST_MAX_FEATURES,
                        n_jobs=n_jobs,
                        random_state=seed,
                    ),
                ),
            ]
            for name, model in models:
                model.fit(X_fit, y_fit)
                pred = model.predict(split.X_test)
                acc = float(np.mean(pred == split.y_test))
                model_points[name].append(acc)
                rows.append(
                    {
                        "dataset": dataset.name,
                        "noise_fraction": noise,
                        "model": name,
                        "accuracy": acc,
                    }
                )
        for name, values in model_points.items():
            ax.plot(noise_levels, values, marker="o", label=name)
        ax.set_title(dataset.name)
        ax.set_xlabel("Label-noise fraction")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Clean-test accuracy")
    axes[len(datasets) - 1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "noise_robustness.pdf")
    plt.close(fig)
    write_csv(results_dir / "noise_robustness.csv", rows)


def run_bias_variance(
    dataset: DatasetBundle,
    figures_dir: Path,
    results_dir: Path,
    seed: int,
    n_jobs: int = 1,
) -> None:
    if dataset.name != "Breast Cancer Wisconsin":
        logger.warning("Bias-variance is specified for Breast Cancer Wisconsin; received %s", dataset.name)
    split = prepare_split(dataset, seed)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    model_predictions: dict[str, list[np.ndarray]] = {
        "Decision Stump": [],
        "Single Tree": [],
        "AdaBoost": [],
        "Random Forest": [],
    }
    for replicate in range(100):
        indices = rng.integers(0, split.X_train.shape[0], size=split.X_train.shape[0])
        X_boot = split.X_train[indices]
        y_boot = split.y_train[indices]
        models: list[tuple[str, Any]] = [
            (
                "Decision Stump",
                DecisionStump(random_state=seed + replicate),
            ),
            (
                "Single Tree",
                DecisionTree(random_state=seed + replicate),
            ),
            (
                "AdaBoost",
                AdaBoostClassifier(n_estimators=50, learning_rate=ADABOOST_LEARNING_RATE, random_state=seed + replicate),
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=25,
                    max_features=RANDOM_FOREST_MAX_FEATURES,
                    n_jobs=n_jobs,
                    random_state=seed + replicate,
                ),
            ),
        ]
        for name, model in models:
            model.fit(X_boot, y_boot)
            model_predictions[name].append(model.predict(split.X_test))

    for name, predictions in model_predictions.items():
        summary = bias_variance_01(np.asarray(predictions), split.y_test)
        rows.append({"dataset": dataset.name, "model": name, **summary})

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(rows))
    bias = [row["bias_squared"] for row in rows]
    variance = [row["variance"] for row in rows]
    width = 0.35
    ax.bar(x - width / 2, bias, width, label="Bias^2")
    ax.bar(x + width / 2, variance, width, label="Variance")
    ax.set_xticks(x)
    ax.set_xticklabels([row["model"] for row in rows], rotation=10, ha="right")
    ax.set_ylabel("0-1 decomposition component")
    ax.set_title("Bias-variance on Breast Cancer Wisconsin")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "bias_variance.pdf")
    plt.close(fig)
    write_csv(results_dir / "bias_variance.csv", rows)


def run_unsupervised(
    datasets: list[DatasetBundle],
    figures_dir: Path,
    results_dir: Path,
    seed: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        slug = slugify(dataset.name)
        preprocessor = MixedTypePreprocessor(
            dataset.resolved_numeric_columns,
            dataset.categorical_columns,
        )
        X_scaled = preprocessor.fit_transform(dataset.X)
        labels = dataset.y
        max_components = min(X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca_full = PCA(n_components=max_components).fit(X_scaled)
        assert pca_full.explained_variance_ratio_ is not None
        cumulative = np.cumsum(pca_full.explained_variance_ratio_)
        components_90 = int(np.searchsorted(cumulative, 0.90) + 1)
        plot_scree(cumulative, dataset.name, figures_dir / f"{slug}_pca_scree.pdf")

        pca_2 = PCA(n_components=2)
        X_2d = pca_2.fit_transform(X_scaled)

        inertias: list[float] = []
        ari_by_k: list[float] = []
        best_k = 1
        best_kmeans_ari = -1.0
        best_kmeans_labels = np.zeros(X_scaled.shape[0], dtype=int)
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=5).fit(X_2d)
            assert kmeans.inertia_ is not None and kmeans.labels_ is not None
            ari = float(adjusted_rand_score(labels, kmeans.labels_))
            inertias.append(kmeans.inertia_)
            ari_by_k.append(ari)
            if ari > best_kmeans_ari:
                best_kmeans_ari = ari
                best_k = k
                best_kmeans_labels = kmeans.labels_.copy()
        plot_elbow(inertias, dataset.name, figures_dir / f"{slug}_kmeans_elbow.pdf")

        kth_distances = kth_neighbor_distances(X_2d, k=5)
        eps_values = np.percentile(kth_distances, [60, 70, 80, 90, 95])
        best_dbscan_ari = -1.0
        best_eps = float(eps_values[0])
        best_dbscan_labels = np.full(X_scaled.shape[0], -1, dtype=int)
        best_noise = 1.0
        for eps in eps_values:
            dbscan = DBSCAN(eps=float(eps), min_samples=5).fit(X_2d)
            assert dbscan.labels_ is not None
            ari = float(adjusted_rand_score(labels, dbscan.labels_))
            noise = float(np.mean(dbscan.labels_ == -1))
            score = ari - 0.05 * noise
            if score > best_dbscan_ari - 0.05 * best_noise:
                best_dbscan_ari = ari
                best_eps = float(eps)
                best_dbscan_labels = dbscan.labels_.copy()
                best_noise = noise
        plot_k_distance(kth_distances, best_eps, dataset.name, figures_dir / f"{slug}_dbscan_kdistance.pdf")
        plot_pca_scatter(
            X_2d,
            labels,
            best_kmeans_labels,
            best_dbscan_labels,
            dataset.name,
            figures_dir / f"{slug}_pca_clusters.pdf",
        )

        if dataset.name == "MNIST2Class":
            plot_tsne(X_scaled, labels, seed, figures_dir / "mnist2class_tsne_bonus.pdf")

        rows.append(
            {
                "dataset": dataset.name,
                "components_for_90pct_variance": components_90,
                "kmeans_best_k": best_k,
                "kmeans_ari": best_kmeans_ari,
                "dbscan_eps": best_eps,
                "dbscan_ari": best_dbscan_ari,
                "dbscan_noise_fraction": best_noise,
            }
        )
    write_csv(results_dir / "unsupervised_summary.csv", rows)


def plot_scree(cumulative: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    ax.axhline(0.90, color="tab:red", linestyle="--", linewidth=1)
    ax.set_title(f"PCA scree: {title}")
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_elbow(inertias: list[float], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(range(1, 11), inertias, marker="o")
    ax.set_title(f"K-Means elbow: {title}")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_k_distance(distances: np.ndarray, eps: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(np.arange(1, len(distances) + 1), distances)
    ax.axhline(eps, color="tab:red", linestyle="--", linewidth=1, label=f"eps={eps:.2f}")
    ax.set_title(f"DBSCAN k-distance: {title}")
    ax.set_xlabel("Sorted samples")
    ax.set_ylabel("5th-neighbor distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pca_scatter(
    X_2d: np.ndarray,
    true_labels: np.ndarray,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    panels = [
        ("True labels", true_labels),
        ("K-Means", kmeans_labels),
        ("DBSCAN", dbscan_labels),
    ]
    for ax, (panel_title, labels) in zip(axes, panels):
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.astype(float), s=12, cmap="tab10", alpha=0.8)
        ax.set_title(panel_title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tsne(X: np.ndarray, y: np.ndarray, seed: int, path: Path) -> None:
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y.astype(float), s=8, cmap="tab10", alpha=0.7)
    ax.set_title("t-SNE bonus: MNIST2Class (all 14,780 rows)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_gradient_boosting_bonus(
    dataset: DatasetBundle,
    figures_dir: Path,
    results_dir: Path,
    seed: int,
) -> None:
    split = prepare_split(dataset, seed)
    labels = np.unique(dataset.y).astype(object)
    models: list[tuple[str, Any]] = [
        (
            "AdaBoost",
            AdaBoostClassifier(n_estimators=100, learning_rate=ADABOOST_LEARNING_RATE, random_state=seed),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, random_state=seed),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, model in models:
        model.fit(split.X_fit, split.y_fit)
        rows.append({"dataset": dataset.name, "model": name, **evaluate_model(model, split.X_test, split.y_test, labels)})

    gbm = models[1][1]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(np.arange(1, len(gbm.train_loss_) + 1), gbm.train_loss_, color="tab:green")
    ax.set_title("Gradient Boosting log-loss")
    ax.set_xlabel("Boosting rounds")
    ax.set_ylabel("Training log-loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "gradient_boosting_bonus.pdf")
    plt.close(fig)
    write_csv(results_dir / "gradient_boosting_bonus.csv", rows)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Some native numerical libraries can leave non-daemon worker threads alive
    # after large from-scratch tree sweeps. The experiment outputs are already
    # closed and flushed above, so exit explicitly to keep one-command
    # reproduction from hanging at interpreter shutdown on such systems.
    os._exit(0)
