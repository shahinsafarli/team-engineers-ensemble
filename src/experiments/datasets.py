"""Authoritative loaders for the four locked study datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_covtype, load_breast_cancer

from src.experiments.utils import DatasetBundle, project_root
from src.utils.preprocessing import class_distribution, stratified_subsample_indices


logger = logging.getLogger(__name__)

ADULT_TRAIN_ROWS = 32_561
ADULT_TEST_ROWS = 16_281
ADULT_ROWS = ADULT_TRAIN_ROWS + ADULT_TEST_ROWS
ADULT_CLASS_COUNTS = {"0": 37_155, "1": 11_687}
ADULT_NUMERIC_COLUMNS = (0, 2, 4, 10, 11, 12)
ADULT_CATEGORICAL_COLUMNS = (1, 3, 5, 6, 7, 8, 9, 13)
ADULT_URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
}
ADULT_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"

COVERTYPE_RAW_ROWS = 581_012
COVERTYPE_ARCHIVE_URL = "https://ndownloader.figshare.com/files/5976039"
COVERTYPE_ROWS = 50_000
COVERTYPE_RAW_CLASS_COUNTS = {
    "1": 211_840,
    "2": 283_301,
    "3": 35_754,
    "4": 2_747,
    "5": 9_493,
    "6": 17_367,
    "7": 20_510,
}
COVERTYPE_CLASS_COUNTS = {
    "1": 18_230,
    "2": 24_380,
    "3": 3_077,
    "4": 236,
    "5": 817,
    "6": 1_495,
    "7": 1_765,
}

MNIST_RAW_ROWS = 70_000
MNIST_ROWS = 14_780
MNIST_CLASS_COUNTS = {"0": 6_903, "1": 7_877}
MNIST_ARFF_URL = "https://www.openml.org/data/v1/download/52667/mnist_784.arff"


def load_all_datasets(
    seed: int = 42,
    skip_downloads: bool = False,
    root: Path | None = None,
) -> list[DatasetBundle]:
    """Load and validate the exact four-dataset contract in canonical order."""

    data_root = project_root() if root is None else root
    datasets = [
        load_breast_cancer_dataset(),
        load_adult_income(data_root, skip_downloads=skip_downloads),
        load_covertype_dataset(data_root, seed=seed, skip_downloads=skip_downloads),
        load_mnist2class(data_root, skip_downloads=skip_downloads),
    ]
    expected = {
        "Breast Cancer Wisconsin": 569,
        "Adult Income": ADULT_ROWS,
        "Covertype": COVERTYPE_ROWS,
        "MNIST2Class": MNIST_ROWS,
    }
    actual = {dataset.name: int(dataset.X.shape[0]) for dataset in datasets}
    if actual != expected:
        raise RuntimeError(f"Four-dataset contract mismatch: expected {expected}, got {actual}")
    return datasets


def load_breast_cancer_dataset() -> DatasetBundle:
    raw = load_breast_cancer()
    X = np.asarray(raw.data, dtype=float)
    y = np.asarray(raw.target, dtype=object)
    _assert_shape("Breast Cancer Wisconsin", X, y, rows=569, features=30)
    _assert_class_counts("Breast Cancer Wisconsin", y, {"0": 212, "1": 357})
    return DatasetBundle(
        name="Breast Cancer Wisconsin",
        X=X,
        y=y,
        source="scikit-learn load_breast_cancer (UCI Wisconsin Diagnostic Breast Cancer)",
        source_version="scikit-learn packaged full dataset",
        selection_rule="All 569 rows; no sampling.",
        description="Binary malignant/benign diagnosis data with 30 continuous features.",
        high_dimensional=True,
        raw_samples=569,
    )


def load_adult_income(root: Path, skip_downloads: bool = False) -> DatasetBundle:
    raw_dir = root / "data" / "raw"
    paths = {name: raw_dir / name for name in ADULT_URLS}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        if skip_downloads:
            raise FileNotFoundError(
                f"{missing[0]} is required by --skip-downloads; run once without the flag to cache Adult."
            )
        archive = raw_dir / "adult.zip"
        if not archive.exists():
            _download(ADULT_ARCHIVE_URL, archive)
        _extract_adult_archive(archive, paths)

    train_X, train_y = _parse_adult_file(paths["adult.data"], is_test=False)
    test_X, test_y = _parse_adult_file(paths["adult.test"], is_test=True)
    if train_X.shape[0] != ADULT_TRAIN_ROWS or test_X.shape[0] != ADULT_TEST_ROWS:
        raise RuntimeError(
            "Adult source row-count mismatch: "
            f"expected {ADULT_TRAIN_ROWS}+{ADULT_TEST_ROWS}, got {train_X.shape[0]}+{test_X.shape[0]}"
        )
    X = np.vstack([train_X, test_X]).astype(object)
    y = np.concatenate([train_y, test_y]).astype(object)
    _assert_shape("Adult Income", X, y, rows=ADULT_ROWS, features=14)
    _assert_class_counts("Adult Income", y, ADULT_CLASS_COUNTS)
    return DatasetBundle(
        name="Adult Income",
        X=X,
        y=y,
        source="UCI Adult adult.data + adult.test",
        source_version="UCI dataset 2; original train/test files",
        selection_rule="All 32,561 training and 16,281 test records combined; no rows dropped.",
        description="Binary income classification with six numeric and eight categorical raw features.",
        numeric_columns=ADULT_NUMERIC_COLUMNS,
        categorical_columns=ADULT_CATEGORICAL_COLUMNS,
        preprocessing=(
            "Training-only numeric median imputation and scaling; training-only categorical mode "
            "imputation and one-hot encoding; unseen categories map to all-zero indicator groups."
        ),
        raw_samples=ADULT_ROWS,
    )


def load_covertype_dataset(
    root: Path,
    seed: int = 42,
    skip_downloads: bool = False,
) -> DatasetBundle:
    interim = root / "data" / "interim" / f"covertype_50000_seed{seed}.npz"
    if interim.exists():
        cached = np.load(interim, allow_pickle=False)
        X = np.asarray(cached["X"], dtype=float)
        y = np.asarray(cached["y"], dtype=object)
    else:
        raw_archive = root / "data" / "raw" / "covtype.data.gz"
        if raw_archive.exists():
            with gzip.open(raw_archive, "rt", encoding="ascii") as handle:
                raw_matrix = np.loadtxt(handle, delimiter=",", dtype=float)
            X_raw = raw_matrix[:, :-1]
            y_raw = raw_matrix[:, -1].astype(int)
        else:
            if skip_downloads:
                raise FileNotFoundError(
                    f"{interim} or {raw_archive} is required by --skip-downloads; "
                    "run once without the flag to cache Covertype."
                )
            try:
                _download(COVERTYPE_ARCHIVE_URL, raw_archive)
                with gzip.open(raw_archive, "rt", encoding="ascii") as handle:
                    raw_matrix = np.loadtxt(handle, delimiter=",", dtype=float)
                X_raw = raw_matrix[:, :-1]
                y_raw = raw_matrix[:, -1].astype(int)
            except (OSError, urllib.error.URLError):
                raw = fetch_covtype(data_home=str(root / "data" / "sklearn"))
                X_raw = np.asarray(raw.data, dtype=float)
                y_raw = np.asarray(raw.target)
        _assert_shape("raw Covertype", X_raw, y_raw, rows=COVERTYPE_RAW_ROWS, features=54)
        _assert_class_counts("raw Covertype", y_raw, COVERTYPE_RAW_CLASS_COUNTS)
        indices = stratified_subsample_indices(y_raw, COVERTYPE_ROWS, random_state=seed)
        X = X_raw[indices]
        y = y_raw[indices].astype(object)
        interim.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(interim, X=X.astype(np.float32), y=np.asarray(y, dtype=np.int8))

    _assert_shape("Covertype", X, y, rows=COVERTYPE_ROWS, features=54)
    _assert_class_counts("Covertype", y, COVERTYPE_CLASS_COUNTS)
    return DatasetBundle(
        name="Covertype",
        X=X,
        y=y,
        source="scikit-learn fetch_covtype (UCI Covertype)",
        source_version="UCI dataset 31 / scikit-learn fetch_covtype",
        selection_rule=(
            f"Seed {seed} exact largest-remainder stratified sample of 50,000 from all 581,012 rows."
        ),
        description="Seven-class forest cover type classification with 54 numeric/binary features.",
        severe_imbalance=True,
        high_dimensional=True,
        raw_samples=COVERTYPE_RAW_ROWS,
    )


def load_mnist2class(root: Path, skip_downloads: bool = False) -> DatasetBundle:
    interim = root / "data" / "interim" / "mnist2class_openml554_v1.npz"
    if interim.exists():
        cached = np.load(interim, allow_pickle=False)
        X = np.asarray(cached["X"], dtype=float)
        y = np.asarray(cached["y"], dtype=object)
    else:
        raw_archive = root / "data" / "raw" / "mnist_784_openml554_v1.arff"
        if raw_archive.exists():
            try:
                X, y = _parse_mnist_arff(raw_archive)
            except (OSError, RuntimeError, UnicodeError):
                if skip_downloads:
                    raise RuntimeError(
                        f"{raw_archive} is incomplete or invalid; rerun without --skip-downloads to repair it."
                    )
                raw_archive.unlink()
        if not raw_archive.exists():
            if skip_downloads:
                raise FileNotFoundError(
                    f"{interim} or {raw_archive} is required by --skip-downloads; "
                    "run once without the flag to cache MNIST2Class."
                )
            _download(MNIST_ARFF_URL, raw_archive)
            X, y = _parse_mnist_arff(raw_archive)
        interim.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(interim, X=X.astype(np.uint8), y=np.asarray(y, dtype=np.int8))

    _assert_shape("MNIST2Class", X, y, rows=MNIST_ROWS, features=784)
    _assert_class_counts("MNIST2Class", y, MNIST_CLASS_COUNTS)
    return DatasetBundle(
        name="MNIST2Class",
        X=X,
        y=y,
        source="OpenML data_id=554, mnist_784 version 1",
        source_version="OpenML dataset 554 v1",
        selection_rule="All 6,903 zeros and all 7,877 ones; no sampling.",
        description="Binary handwritten digit classification using all MNIST zeros and ones.",
        high_dimensional=True,
        raw_samples=MNIST_RAW_ROWS,
    )


def _parse_adult_file(path: Path, is_test: bool) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[object]] = []
    targets: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if not row or row[0].lstrip().startswith("|"):
                continue
            if len(row) != 15:
                raise RuntimeError(f"Malformed Adult row in {path}: expected 15 fields, got {len(row)}")
            values = [value.strip() for value in row]
            target = values[-1].removesuffix("." if is_test else "")
            if target not in {"<=50K", ">50K"}:
                raise RuntimeError(f"Unexpected Adult target {target!r} in {path}")
            parsed: list[object] = []
            for index, value in enumerate(values[:-1]):
                if value == "?":
                    parsed.append(None)
                elif index in ADULT_NUMERIC_COLUMNS:
                    parsed.append(float(value))
                else:
                    parsed.append(value)
            features.append(parsed)
            targets.append(int(target == ">50K"))
    return np.asarray(features, dtype=object), np.asarray(targets, dtype=object)


def _download(url: str, path: Path, headers: dict[str, str] | None = None) -> None:
    logger.info("Downloading %s to %s", url, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    request_headers = {
        "User-Agent": "ensemble-methods-reproducibility/1.0",
        "Accept": "*/*",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            expected_size_text = response.headers.get("Content-Length")
            bytes_written = 0
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                bytes_written += len(chunk)
            if expected_size_text is not None and bytes_written != int(expected_size_text):
                raise OSError(
                    f"Incomplete download from {url}: expected {expected_size_text} bytes, got {bytes_written}"
                )
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def _extract_adult_archive(archive: Path, paths: dict[str, Path]) -> None:
    with zipfile.ZipFile(archive) as zipped:
        names = {Path(name).name: name for name in zipped.namelist()}
        for filename, target in paths.items():
            if filename not in names:
                raise RuntimeError(f"Official Adult archive is missing {filename}")
            partial = target.with_suffix(target.suffix + ".part")
            try:
                with zipped.open(names[filename]) as source, partial.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                partial.replace(target)
            finally:
                if partial.exists():
                    partial.unlink()


def _parse_mnist_arff(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rt", encoding="ascii", newline="") as handle:
        return _parse_mnist_arff_lines(handle)


def _parse_mnist_arff_gzip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a gzip fixture or legacy cache; new downloads use identity transfer."""

    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        return _parse_mnist_arff_lines(handle)


def _parse_mnist_arff_lines(handle: io.TextIOBase) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((MNIST_ROWS, 784), dtype=np.uint8)
    y = np.empty(MNIST_ROWS, dtype=np.int8)
    raw_rows = 0
    selected_rows = 0
    in_data = False
    for line in handle:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if not in_data:
            if stripped.lower() == "@data":
                in_data = True
            continue
        raw_rows += 1
        feature_text, target_text = stripped.rsplit(",", 1)
        target = int(float(target_text))
        if target not in {0, 1}:
            continue
        values = np.fromstring(feature_text, sep=",", dtype=np.uint8)
        if values.shape != (784,):
            raise RuntimeError(
                f"Malformed OpenML MNIST row {raw_rows}: expected 784 features, got {values.size}"
            )
        if selected_rows >= MNIST_ROWS:
            raise RuntimeError("OpenML MNIST contains more zero/one rows than expected")
        X[selected_rows] = values
        y[selected_rows] = target
        selected_rows += 1
    if not in_data or raw_rows != MNIST_RAW_ROWS or selected_rows != MNIST_ROWS:
        raise RuntimeError(
            "OpenML mnist_784 v1 row mismatch: "
            f"expected 70000 raw and 14780 selected, got {raw_rows} raw and {selected_rows} selected"
        )
    return X, y.astype(object)


def _assert_shape(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    rows: int,
    features: int,
) -> None:
    if X.shape != (rows, features) or y.shape != (rows,):
        raise RuntimeError(
            f"{name} shape mismatch: expected X={(rows, features)}, y={(rows,)}, got X={X.shape}, y={y.shape}"
        )


def _assert_class_counts(name: str, y: np.ndarray, expected: dict[str, int]) -> None:
    actual = class_distribution(y)
    if actual != expected:
        raise RuntimeError(f"{name} class distribution mismatch: expected {expected}, got {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache and validate all four authoritative datasets.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for dataset in load_all_datasets(seed=args.seed):
        print(
            f"{dataset.name}: rows={dataset.X.shape[0]}, features={dataset.X.shape[1]}, "
            f"classes={class_distribution(dataset.y)}"
        )


if __name__ == "__main__":
    main()
