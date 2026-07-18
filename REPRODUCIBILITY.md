# Reproducibility Notes

## Canonical environment

- Python 3.12
- Dependencies pinned by the ranges in `requirements.txt`
- Canonical seed: 42 for selection, splits, CV, corruption, clustering, and models
- Default Random Forest worker count: `min(4, os.cpu_count())`; use `--n-jobs 1` on limited-memory Windows systems

Create a clean environment with:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## First-run downloads and caches

```bash
bash download_data.sh
```

The Python loader is the single source of truth. It combines both Adult files, filters OpenML dataset 554 to all zeros/ones, and applies the seed-42 exact largest-remainder Covertype selection. The shell helper does not duplicate these rules.

Local-only cache paths are:

- `data/raw/`: authoritative Adult, Covertype, and OpenML source downloads;
- `data/sklearn/`: scikit-learn dataset cache used by supported loader fallbacks;
- `data/interim/`: validated seed-42 Covertype and MNIST2Class selections.

All three paths are ignored. Loaders fail loudly if source dimensions, labels, or locked class counts differ.

## Full recomputation

```bash
python src/experiments/run_all.py --seed 42
```

This is the default and recomputes every model-level result. `--recompute-heavy` remains accepted for backward compatibility but is no longer needed.

For a prepared offline cache:

```bash
python src/experiments/run_all.py --seed 42 --skip-downloads --n-jobs 1
```

`--skip-downloads` requires the ignored authoritative caches to exist and fails when they are absent. No execution path reduces the locked dataset sizes.

## Strict result reuse

```bash
python src/experiments/run_all.py --seed 42 --skip-downloads --reuse-results
```

Reuse validates exact dataset names and sizes, seed, source/selection metadata, content fingerprints, hyperparameter grids, CSV row counts/dataset sets, and the exact 23-figure filename set. Any mismatch stops with an error.

## Leakage controls

For every supervised split or CV fold:

```text
raw authoritative rows
-> deterministic train/test indices
-> fit imputation/scaling/one-hot categories on training rows
-> transform held-out rows
-> oversample training rows only when configured
-> fit every compared model on the identical prepared fold
```

Adult missing values are imputed rather than dropped. Unseen categorical values map to all-zero one-hot groups. The unsupervised pipeline fits its transform on complete data because no held-out score is reported.

## Expected generated artifacts

`data/results/` contains:

| File | Data rows |
|---|---:|
| `baseline_metrics.csv` | 16 |
| `adaboost_scaling.csv` | 84 |
| `random_forest_estimators.csv` | 28 |
| `random_forest_depth.csv` | 32 |
| `head_to_head_cv.csv` | 80 |
| `head_to_head_summary.csv` | 16 |
| `head_to_head_significance.csv` | 4 |
| `noise_robustness.csv` | 24 |
| `bias_variance.csv` | 4 |
| `unsupervised_summary.csv` | 4 |
| `gradient_boosting_bonus.csv` | 2 |

It also contains `dataset_metadata.json` and `run_summary.json`, including sources/versions, selection rules, class distributions, preprocessing, fingerprints, timings, worker count, and run mode.

`figures/` contains exactly 23 PDFs: six aggregate/bonus plots, four unsupervised plots per dataset, and `mnist2class_tsne_bonus.pdf` computed from all 14,780 rows.

## Verification

```bash
python -m pytest -q
python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=60 -q
python -m mypy src --ignore-missing-imports
python -m ruff check src tests
RUN_SLOW_TESTS=1 python -m pytest tests/test_dbscan_performance.py -q
```

Before release, run the default pipeline once with prepared caches, compare tracked outputs to that run, validate the notebook, compile all three TeX deliverables, and visually inspect every PDF page/slide.
