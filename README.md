# Ensemble Methods: Boosting vs. Bagging

Machine Learning final project for AI Academy, National AI Center.

Team members: Shahin Safarli, Gulnisa Abdurahmanli, Jeyhuna Sevdiyeva, Seljan Khasiyeva,
Suleyman Allahverdiyev (5 members).

> Status: repository scaffold only. Algorithm implementations, experiments, report, and
> presentation land through reviewed Pull Requests over the following days — see the
> project's Git/GitHub workflow document for the day-by-day plan.

## Setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## One-command reproduction (once experiments land)

```bash
python src/experiments/run_all.py
```

The runner will write result tables to `data/results/` and figures to `figures/`.

## Project layout

```
src/
  trees/          decision tree, tree-based wrappers
  boosting/        AdaBoost, Gradient Boosting (bonus)
  bagging/         Random Forest
  unsupervised/    PCA, K-Means, DBSCAN
  utils/           preprocessing helpers
  metrics/         evaluation helpers
  experiments/     run_all.py pipeline and shared experiment utilities
tests/             unit tests mirroring src/
report/            IEEE-style report (report.tex, report.pdf)
presentation/      defense slides (presentation.tex, presentation.pdf)
notebooks/         exploratory notebook(s)
figures/           generated figures (populated by run_all.py)
data/results/      generated result tables (populated by run_all.py)
contribution_report.tex / .pdf   signed per-member contribution statement
```

## Implementation policy

`sklearn` is used only for dataset loading, metrics, cross-validation utilities, t-SNE,
and explicitly marked sanity-check reference baselines. Decision Tree, AdaBoost, Random
Forest, PCA, K-Means, DBSCAN, and the bonus Gradient Boosting model are implemented from
scratch in `src/`.

## Verification (run before every PR)

```bash
python -m pytest -q
python -m mypy src --ignore-missing-imports
python -m ruff check src tests
```

CI (lint, type check, test, coverage) is added to this repository later in the workflow,
once the core implementations exist — see the workflow document, Day 5.
