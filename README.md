# Ensemble Methods: Boosting vs. Bagging

Machine Learning final project for AI Academy, National AI Center.

Team members: Shahin Safarli, Gulnisa Abdurahmanli, Jeyhuna Sevdiyeva, Seljan Khasiyeva, and Suleyman Allahverdiyev.

The repository contains from-scratch Decision Tree, AdaBoost, Random Forest, PCA, K-Means, DBSCAN, and bonus Gradient Boosting implementations. Scikit-learn is limited to authoritative dataset loading, metrics/CV utilities, t-SNE, and explicitly labelled reference baselines.

## Setup

Python 3.12 is the canonical environment.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Prepare the first-run caches from the official sources:

```bash
bash download_data.sh
```

This writes only to ignored `data/raw/`, `data/sklearn/`, and `data/interim/` paths. No raw dataset or selected-data cache belongs in Git.

## Canonical reproduction

The default command is a true full model-level recomputation:

```bash
python src/experiments/run_all.py --seed 42
```

Use up to four Random Forest workers by default. On a limited-memory Windows machine, prefer:

```bash
python src/experiments/run_all.py --seed 42 --n-jobs 1
```

After a successful first run, network access can be disabled while retaining full recomputation:

```bash
python src/experiments/run_all.py --seed 42 --skip-downloads --n-jobs 1
```

Reusing committed outputs is opt-in and strict:

```bash
python src/experiments/run_all.py --seed 42 --skip-downloads --reuse-results
```

Reuse is rejected if dataset names, exact sizes, seed, experiment grids, content fingerprints, table rows, or figure names differ. `--recompute-heavy` remains accepted as a backward-compatible alias for the default recomputation behavior.

The complete run is intentionally expensive: it uses all required rows, 5-fold comparisons, 100 Breast Cancer bootstrap replicates, full-data clustering, and full-data MNIST2Class t-SNE. Runtime depends strongly on CPU, memory, and `--n-jobs`; no command-line option reduces the study data.

## Locked dataset contract

| Dataset | Rows | Target | Selection |
|---|---:|---|---|
| Breast Cancer Wisconsin | 569 | Binary | Full scikit-learn/UCI dataset |
| Adult Income | 48,842 | Binary | All 32,561 `adult.data` and 16,281 `adult.test` rows |
| Covertype | 50,000 | Seven-class | Seed-42 exact largest-remainder stratified selection from 581,012 rows |
| MNIST2Class | 14,780 | 0 vs 1 | All 6,903 zeros and 7,877 ones from OpenML `mnist_784` v1 |

Authoritative records: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype), and [OpenML dataset 554](https://www.openml.org/api/v1/json/data/554).

The locked Covertype allocation is 18,230 / 24,380 / 3,077 / 236 / 817 / 1,495 / 1,765 for classes 1 through 7. The loader fails if raw sizes, labels, or these counts change.

Adult remains object-valued until a fold is split. Numeric median imputation/scaling and categorical mode imputation/one-hot categories are fitted on training rows only; unseen held-out categories become all-zero indicator groups. All 48,842 source rows are retained. For unsupervised analysis only, the same transformation is fitted on the complete dataset because there is no held-out evaluation.

Covertype oversampling is a training-fold-only treatment shared by every compared supervised model. It does not alter the 50,000-row source sample.

## Outputs

The runner creates eleven CSV tables plus `dataset_metadata.json` and `run_summary.json` under `data/results/`. Exact expected CSV data-row counts are 16, 84, 28, 32, 80, 16, 4, 24, 4, 4, and 2 for the baseline, AdaBoost scaling, RF estimator scaling, RF depth scaling, CV, CV summary, CV significance, noise, bias-variance, unsupervised, and Gradient Boosting tables respectively.

`head_to_head_significance.csv` holds one Random Forest vs.\ AdaBoost paired $t$-test per dataset (accuracy, macro-F1, and AUC-ROC), each with both a raw and a Holm-Bonferroni-adjusted p-value across the four datasets. See "Head-to-Head Comparison" in `report.tex` for the exploratory, small-sample caveats around these numbers.

Exactly 23 PDFs are generated under `figures/`: six aggregate/bonus figures, four unsupervised figures for each dataset, and full-data `mnist2class_tsne_bonus.pdf`.

## Verification

```bash
python -m pytest -q
python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=60 -q
python -m mypy src --ignore-missing-imports
python -m ruff check src tests
```

The opt-in 50,000-point DBSCAN smoke test is:

```bash
RUN_SLOW_TESTS=1 python -m pytest tests/test_dbscan_performance.py -q
```

## Repository layout

```text
src/
  trees/              DecisionTree, DecisionStump, assignment-path wrappers
  boosting/           AdaBoost and bonus GradientBoostingClassifier
  bagging/            RandomForestClassifier
  unsupervised/       PCA, KMeans, exact KD-tree-backed DBSCAN
  experiments/        authoritative loaders, runner, and output utilities
  metrics/            scoring helpers
  utils/              fitted preprocessing and sampling helpers
tests/                unit, parity, pipeline, and opt-in performance checks
figures/              generated canonical experiment figures
data/results/         generated canonical tables and provenance
report/               IEEE report source and compiled PDF
presentation/         defense slide source and compiled PDF
notebooks/            bonus interactive exploration notebook
contribution_report.tex / contribution_report.pdf
```

## Deliverables and integrity

The final tagged repository should include source, tests, requirements, documentation, generated result/figure artifacts, the IEEE report, the 12-slide defense deck, and the contribution report, but no raw datasets, local caches, ZIPs, coverage files, or LaTeX auxiliaries.

Substantial AI assistance was used to implement, debug, and polish this submission. Every team member should review, run, and be able to defend the code, experiments, and written claims assigned to them.
