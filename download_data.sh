#!/usr/bin/env bash
# Optional dataset helper for the ML final project.
# Run from the repository root: bash download_data.sh
set -euo pipefail

mkdir -p data

cat <<'MSG'
The project uses:
  1. Breast Cancer Wisconsin via sklearn.datasets.load_breast_cancer
  2. Digits via sklearn.datasets.load_digits
  3. A bundled rare-class Covertype subset at data/covertype_rare_class.npz

No extra download is required for normal reproduction:
  python src/experiments/run_all.py --skip-downloads

To refresh the Covertype subset from sklearn instead, delete
data/covertype_rare_class.npz and run:
  python src/experiments/run_all.py
MSG
