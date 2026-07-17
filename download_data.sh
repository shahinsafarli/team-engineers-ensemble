#!/usr/bin/env bash
# Optional dataset helper for the ML final project.
# Run from the repository root: bash download_data.sh
set -euo pipefail

mkdir -p data/raw data/sklearn data/interim

cat <<'MSG'
Preparing the four authoritative sources:
  1. Breast Cancer Wisconsin: full scikit-learn packaged dataset (569 rows)
  2. Adult Income: UCI adult.data + adult.test (48,842 rows)
  3. Covertype: full UCI/scikit-learn source, then Python's seed-42 exact
     largest-remainder seven-class selection (50,000 rows)
  4. MNIST2Class: OpenML mnist_784 v1, all zero/one rows (14,780 rows)

The Python loaders validate raw sizes, labels, and locked class counts. This
script does not duplicate either the Covertype sampling rule or MNIST filter.
MSG

python -m src.experiments.datasets --seed 42

cat <<'MSG'
Caches are ready under ignored data/raw, data/sklearn, and data/interim paths.
To run without network access afterward:
  python src/experiments/run_all.py --skip-downloads
MSG
