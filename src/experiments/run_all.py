"""Run every experiment required by the final-project statement.

The script writes CSV tables under ``data/results`` and publication
figures under ``figures``. It intentionally uses sklearn only for data
loading, metrics, cross-validation utilities, t-SNE, and explicitly
marked reference baselines.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import sys
import tempfile

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
from sklearn.datasets import fetch_covtype, load_breast_cancer, load_digits, make_classification
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.boosting.gradient_boosting import GradientBoostingClassifier
from src.experiments.utils import (
    DatasetBundle,
    aligned_proba,
    ensure_output_dirs,
    mean_std_rows,
    slugify,
    write_csv,
    write_json,
)
from src.metrics.evaluation import bias_variance_01, classification_metrics
from src.trees.decision_tree import DecisionStump, DecisionTree
from src.unsupervised.dbscan import DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA
from src.utils.preprocessing import (
    StandardScaler,
    class_distribution,
    flip_labels,
    random_oversample,
    stratified_subsample,
    stratified_train_test_split,
)


# Experiment-wide hyperparameter constants. Every experiment function below
# reuses these instead of repeating literal values, so there is a single
# place to change the AdaBoost learning rate, Random Forest feature-subset
# strategy, or the severe-imbalance oversampling ratio. See report.tex,
# "Hyperparameter and Design Choices" for the justification of each value.
ADABOOST_LEARNING_RATE = 0.6
RANDOM_FOREST_MAX_FEATURES = "sqrt"
SEVERE_IMBALANCE_TARGET_RATIO = 0.25


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
        help="Avoid network access and use the bundled rare-class Covertype subset.",
    )
    parser.add_argument(
        "--recompute-heavy",
        action="store_true",
        help="Recompute long Random Forest, CV, noise, and bias-variance sweeps instead of reusing bundled deterministic outputs.",
    )
    parser.add_argument("--_rf-depth-worker-input", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_rf-depth-worker-output", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

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
    metadata = {
        dataset.name: {
            "source": dataset.source,
            "description": dataset.description,
            "samples": int(dataset.X.shape[0]),
            "features": int(dataset.X.shape[1]),
            "class_distribution": class_distribution(dataset.y),
            "severe_imbalance": dataset.severe_imbalance,
            "high_dimensional": dataset.high_dimensional,
        }
        for dataset in datasets
    }
    write_json(results_dir / "dataset_metadata.json", metadata)

    logger.info("Running baselines...")
    baseline_rows = run_baselines(datasets, results_dir, args.seed)
    logger.info("Running AdaBoost scaling...")
    run_adaboost_scaling(datasets, figures_dir, results_dir, args.seed)
    heavy_csv_outputs = [
        results_dir / "random_forest_estimators.csv",
        results_dir / "random_forest_depth.csv",
        results_dir / "head_to_head_cv.csv",
        results_dir / "head_to_head_summary.csv",
        results_dir / "noise_robustness.csv",
        results_dir / "bias_variance.csv",
    ]
    if args.recompute_heavy or not all(path.exists() for path in heavy_csv_outputs):
        logger.info("Running Random Forest scaling...")
        run_rf_scaling(datasets, figures_dir, results_dir, args.seed)
        logger.info("Running head-to-head cross-validation...")
        head_to_head_rows = run_head_to_head(datasets, results_dir, args.seed)
        logger.info("Running noise robustness...")
        run_noise_robustness(datasets, figures_dir, results_dir, args.seed)
        logger.info("Running bias-variance decomposition...")
        run_bias_variance(datasets[0], figures_dir, results_dir, args.seed)
        heavy_sweep_mode = "recomputed"
    else:
        logger.info("Validating bundled deterministic long-sweep tables and regenerating their figures...")
        head_to_head_rows = validate_and_replot_heavy_outputs(results_dir, figures_dir)
        heavy_sweep_mode = "validated_bundled_tables_replotted_figures"
    logger.info("Running unsupervised analysis...")
    run_unsupervised(datasets, figures_dir, results_dir, args.seed)
    logger.info("Running Gradient Boosting bonus...")
    run_gradient_boosting_bonus(datasets[0], figures_dir, results_dir, args.seed)

    write_json(
        results_dir / "run_summary.json",
        {
            "seed": args.seed,
            "datasets": metadata,
            "baseline_rows": len(baseline_rows),
            "head_to_head_rows": len(head_to_head_rows),
            "heavy_sweep_mode": heavy_sweep_mode,
            "outputs": {
                "results_dir": "data/results",
                "figures_dir": "figures",
            },
        },
    )
    logger.info("Wrote results to %s", results_dir)
    logger.info("Wrote figures to %s", figures_dir)


def load_datasets(seed: int, skip_downloads: bool = False) -> list[DatasetBundle]:
    breast = load_breast_cancer()
    breast_dataset = DatasetBundle(
        name="Breast Cancer Wisconsin",
        X=breast.data.astype(float),
        y=breast.target.astype(object),
        source="sklearn.datasets.load_breast_cancer",
        description="Binary medical diagnosis dataset with 30 continuous features.",
        high_dimensional=True,
    )

    digits = load_digits()
    X_digits, y_digits = stratified_subsample(digits.data, digits.target, 500, seed)
    digits_dataset = DatasetBundle(
        name="Digits High-Dimensional",
        X=X_digits,
        y=y_digits,
        source="sklearn.datasets.load_digits, stratified 500-sample subset",
        description="Ten-class handwritten digit images represented by 64 pixel features.",
        high_dimensional=True,
    )

    severe_dataset = load_severe_imbalance_dataset(seed, skip_downloads)
    return [breast_dataset, digits_dataset, severe_dataset]


# The rare-class subset previously totaled only 500 rows (5 positive /
# 495 negative), which is statistically thin for a "severe imbalance"
# case study. These constants raise the subset to >=5000 rows while
# preserving the same ~1% positive rate, so the class-imbalance ratio
# (and therefore the qualitative behavior being studied) is unchanged.
COVERTYPE_POSITIVE_SAMPLES = 50
COVERTYPE_NEGATIVE_SAMPLES = 4950
COVERTYPE_TOTAL_SAMPLES = COVERTYPE_POSITIVE_SAMPLES + COVERTYPE_NEGATIVE_SAMPLES


def load_severe_imbalance_dataset(seed: int, skip_downloads: bool) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    local_cache = ROOT / "data" / "covertype_rare_class.npz"
    if local_cache.exists():
        cached = np.load(local_cache, allow_pickle=False)
        cached_X = cached["X"].astype(float)
        cached_y = cached["y"].astype(object)
        if cached_X.shape[0] < COVERTYPE_TOTAL_SAMPLES:
            if skip_downloads:
                logger.warning(
                    "data/covertype_rare_class.npz has only %d rows (< %d target); "
                    "using it as-is because --skip-downloads was passed. Re-run "
                    "without --skip-downloads (network required) to regenerate the "
                    "full-size subset.",
                    cached_X.shape[0],
                    COVERTYPE_TOTAL_SAMPLES,
                )
            else:
                logger.info(
                    "Bundled Covertype cache has %d rows (< %d target); "
                    "regenerating from sklearn.datasets.fetch_covtype.",
                    cached_X.shape[0],
                    COVERTYPE_TOTAL_SAMPLES,
                )
                return _fetch_and_cache_severe_imbalance_dataset(rng, local_cache)
        return DatasetBundle(
            name="Covertype Rare Class",
            X=cached_X,
            y=cached_y,
            source="bundled data/covertype_rare_class.npz generated from sklearn.datasets.fetch_covtype",
            description="Real forest-cover dataset recast as a severe rare-class detection task.",
            severe_imbalance=True,
            high_dimensional=True,
        )

    if skip_downloads:
        raise FileNotFoundError(
            "data/covertype_rare_class.npz is required for --skip-downloads. "
            "Restore the bundled file or run without --skip-downloads to fetch it."
        )

    return _fetch_and_cache_severe_imbalance_dataset(rng, local_cache)


def _fetch_and_cache_severe_imbalance_dataset(
    rng: np.random.Generator,
    local_cache: Path,
) -> DatasetBundle:
    """Download Covertype and (re)build the cached rare-class subset.

    Requires network access to ``sklearn.datasets.fetch_covtype``; raises
    whatever error scikit-learn raises if the download is unavailable.
    """

    covtype = fetch_covtype(data_home=str(ROOT / "data" / "sklearn"))
    X = covtype.data.astype(float)
    y = (covtype.target == 4).astype(int)
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    chosen = np.concatenate(
        [
            rng.choice(positive, size=COVERTYPE_POSITIVE_SAMPLES, replace=False),
            rng.choice(negative, size=COVERTYPE_NEGATIVE_SAMPLES, replace=False),
        ]
    )
    rng.shuffle(chosen)
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        local_cache,
        X=X[chosen].astype(float),
        y=y[chosen].astype(np.int64),
    )
    return DatasetBundle(
        name="Covertype Rare Class",
        X=X[chosen],
        y=y[chosen].astype(object),
        source=(
            f"sklearn.datasets.fetch_covtype, class 4 vs. rest, "
            f"{COVERTYPE_TOTAL_SAMPLES}-row subset"
        ),
        description="Real forest-cover dataset recast as a severe rare-class detection task.",
        severe_imbalance=True,
        high_dimensional=True,
    )


def prepare_split(dataset: DatasetBundle, seed: int, test_size: float = 0.2) -> SplitData:
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        random_state=seed,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
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
    """Read an existing result table when default run uses bundled heavy outputs."""

    csv_module = __import__("csv")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv_module.DictReader(handle))


def validate_and_replot_heavy_outputs(results_dir: Path, figures_dir: Path) -> list[dict[str, Any]]:
    """Validate deterministic long-sweep CSVs and regenerate their figures.

    The from-scratch Random Forest/CV/noise/bias sweeps are intentionally more
    expensive than the short baseline and unsupervised runs. The default
    command therefore ships the deterministic long-sweep tables, checks their
    schema and row counts, and redraws the associated figures from those
    tables. Passing ``--recompute-heavy`` recomputes the same tables from
    models.
    """

    expected_tables: dict[str, tuple[set[str], int]] = {
        "random_forest_estimators.csv": ({"dataset", "n_estimators", "test_accuracy", "oob_accuracy"}, 21),
        "random_forest_depth.csv": ({"dataset", "max_depth", "test_accuracy", "oob_accuracy"}, 24),
        "head_to_head_cv.csv": ({"dataset", "fold", "model", "accuracy", "macro_f1", "auc_roc"}, 60),
        "head_to_head_summary.csv": ({"dataset", "model", "accuracy_mean", "accuracy_std"}, 12),
        "noise_robustness.csv": ({"dataset", "noise_fraction", "model", "accuracy"}, 18),
        "bias_variance.csv": ({"dataset", "model", "bias_squared", "variance", "expected_loss"}, 4),
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    for filename, (required_columns, minimum_rows) in expected_tables.items():
        rows = read_csv_rows(results_dir / filename)
        if len(rows) < minimum_rows:
            raise RuntimeError(f"{filename} has {len(rows)} rows; expected at least {minimum_rows}")
        if rows:
            missing = required_columns.difference(rows[0].keys())
            if missing:
                raise RuntimeError(f"{filename} is missing required columns: {sorted(missing)}")
        loaded[filename] = rows

    replot_random_forest_scaling(
        loaded["random_forest_estimators.csv"],
        loaded["random_forest_depth.csv"],
        figures_dir,
    )
    replot_noise_robustness(loaded["noise_robustness.csv"], figures_dir)
    replot_bias_variance(loaded["bias_variance.csv"], figures_dir)
    return loaded["head_to_head_cv.csv"]


def _as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _dataset_order(rows: list[dict[str, Any]]) -> list[str]:
    order: list[str] = []
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in order:
            order.append(dataset)
    return order


def replot_random_forest_scaling(
    estimator_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    datasets = _dataset_order(estimator_rows)
    fig_n, axes_n = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.5), sharey=True)
    fig_d, axes_d = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.5), sharey=True)
    if len(datasets) == 1:
        axes_n = [axes_n]
        axes_d = [axes_d]
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
    axes_n[-1].legend(loc="best", fontsize=8)
    axes_d[0].set_ylabel("Accuracy")
    fig_n.tight_layout()
    fig_d.tight_layout()
    fig_n.savefig(figures_dir / "random_forest_estimators.pdf")
    fig_d.savefig(figures_dir / "random_forest_depth.pdf")
    plt.close(fig_n)
    plt.close(fig_d)


def replot_noise_robustness(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    datasets = _dataset_order(rows)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
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
    axes[-1].legend(loc="best", fontsize=8)
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
    ax.set_title("Bias-variance on balanced binary data")
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
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        split = prepare_split(dataset, seed)
        model = AdaBoostClassifier(n_estimators=200, learning_rate=ADABOOST_LEARNING_RATE, random_state=seed)
        model.fit(split.X_fit, split.y_fit)
        train_errors: list[float] = []
        test_errors: list[float] = []
        rounds: list[int] = []
        staged_train = list(model.staged_predict(split.X_train))
        staged_test = list(model.staged_predict(split.X_test))
        for round_index, (train_pred, test_pred) in enumerate(zip(staged_train, staged_test), start=1):
            if round_index == 1 or round_index % 10 == 0 or round_index == len(staged_train):
                train_error = 1.0 - float(np.mean(train_pred == split.y_train))
                test_error = 1.0 - float(np.mean(test_pred == split.y_test))
                rounds.append(round_index)
                train_errors.append(train_error)
                test_errors.append(test_error)
                rows.append(
                    {
                        "dataset": dataset.name,
                        "n_estimators": round_index,
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
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "adaboost_scaling.pdf")
    plt.close(fig)
    write_csv(results_dir / "adaboost_scaling.csv", rows)


def run_rf_scaling(
    datasets: list[DatasetBundle],
    figures_dir: Path,
    results_dir: Path,
    seed: int,
) -> None:
    estimator_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    n_grid = [1, 10, 25, 50, 100, 150, 200]
    depth_grid = [1, 2, 3, 5, 8, 12, 16, 20]

    fig_n, axes_n = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.5), sharey=True)
    fig_d, axes_d = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.5), sharey=True)
    if len(datasets) == 1:
        axes_n = [axes_n]
        axes_d = [axes_d]

    for ax_n, ax_d, dataset in zip(axes_n, axes_d, datasets):
        logger.info("RF scaling: %s", dataset.name)
        split = prepare_split(dataset, seed)
        forest = RandomForestClassifier(
            n_estimators=max(n_grid),
            max_depth=None,
            max_features=RANDOM_FOREST_MAX_FEATURES,
            oob_score=True,
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
    axes_n[-1].legend(loc="best", fontsize=8)
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
            for _ in range(240):  # 120 seconds, normally far less.
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
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        max_features=RANDOM_FOREST_MAX_FEATURES,
        oob_score=True,
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        labels, counts = np.unique(dataset.y, return_counts=True)
        n_splits = min(5, int(np.min(counts)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_labels = dataset.y.astype(str)
        for fold, (train_idx, test_idx) in enumerate(cv.split(dataset.X, split_labels), start=1):
            X_train, X_test = dataset.X[train_idx], dataset.X[test_idx]
            y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if dataset.severe_imbalance:
                X_fit, y_fit = random_oversample(
                    X_train,
                    y_train,
                    random_state=seed + fold,
                    target_ratio=SEVERE_IMBALANCE_TARGET_RATIO,
                )
            else:
                X_fit, y_fit = X_train, y_train

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
                        random_state=seed + fold,
                    ),
                ),
                (
                    "sklearn RF reference",
                    SklearnRandomForestClassifier(
                        n_estimators=100,
                        max_features=RANDOM_FOREST_MAX_FEATURES,
                        random_state=seed + fold,
                    ),
                ),
            ]
            for model_name, model in models:
                fit_estimator(model, X_fit, y_fit)
                metrics = evaluate_model(model, X_test, y_test, labels.astype(object))
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
    write_csv(results_dir / "head_to_head_cv.csv", rows)
    write_csv(results_dir / "head_to_head_summary.csv", summary)
    return rows


def run_noise_robustness(
    datasets: list[DatasetBundle],
    figures_dir: Path,
    results_dir: Path,
    seed: int,
) -> None:
    noise_levels = [0.05, 0.10, 0.20]
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
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
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "noise_robustness.pdf")
    plt.close(fig)
    write_csv(results_dir / "noise_robustness.csv", rows)


def run_bias_variance(
    dataset: DatasetBundle,
    figures_dir: Path,
    results_dir: Path,
    seed: int,
) -> None:
    split = make_balanced_bias_variance_split(seed)
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
                    random_state=seed + replicate,
                ),
            ),
        ]
        for name, model in models:
            model.fit(X_boot, y_boot)
            model_predictions[name].append(model.predict(split.X_test))

    for name, predictions in model_predictions.items():
        summary = bias_variance_01(np.asarray(predictions), split.y_test)
        rows.append({"dataset": "Balanced Synthetic Binary", "model": name, **summary})

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(rows))
    bias = [row["bias_squared"] for row in rows]
    variance = [row["variance"] for row in rows]
    width = 0.35
    ax.bar(x - width / 2, bias, width, label="Bias^2")
    ax.bar(x + width / 2, variance, width, label="Variance")
    ax.set_xticks(x)
    ax.set_xticklabels([row["model"] for row in rows])
    ax.set_ylabel("0-1 decomposition component")
    ax.set_title("Bias-variance on balanced binary data")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "bias_variance.pdf")
    plt.close(fig)
    write_csv(results_dir / "bias_variance.csv", rows)


def make_balanced_bias_variance_split(seed: int) -> SplitData:
    X, y = make_classification(
        n_samples=1560,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        class_sep=1.15,
        flip_y=0.01,
        random_state=seed + 100,
    )
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=1200 / 1560,
        random_state=seed + 101,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return SplitData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train.astype(object),
        y_test=y_test.astype(object),
        X_fit=X_train_scaled,
        y_fit=y_train.astype(object),
    )


def run_unsupervised(
    datasets: list[DatasetBundle],
    figures_dir: Path,
    results_dir: Path,
    seed: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        slug = slugify(dataset.name)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(dataset.X)
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

        if dataset.name == "Digits High-Dimensional":
            plot_tsne(X_scaled, labels, seed, figures_dir / "digits_tsne_bonus.pdf")

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


def kth_neighbor_distances(X: np.ndarray, k: int) -> np.ndarray:
    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    distances.sort(axis=1)
    kth = distances[:, min(k, distances.shape[1] - 1)]
    return np.sort(kth)


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
    X_sample, y_sample = stratified_subsample(X, y, max_samples=500, random_state=seed)
    embedding = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(X_sample)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y_sample.astype(float), s=14, cmap="tab10", alpha=0.85)
    ax.set_title("t-SNE bonus: Digits")
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
