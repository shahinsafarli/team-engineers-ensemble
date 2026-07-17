from __future__ import annotations

import gzip
import io

import numpy as np
import pytest

from src.experiments import datasets
from src.experiments.utils import DatasetBundle
from src.utils.preprocessing import class_distribution


def _bundle(name: str, rows: int) -> DatasetBundle:
    return DatasetBundle(
        name=name,
        X=np.zeros((rows, 1), dtype=float),
        y=np.resize(np.array([0, 1], dtype=object), rows),
        source="mocked authoritative loader",
        description="unit-test bundle",
    )


def test_load_all_datasets_enforces_exact_four_dataset_contract(monkeypatch, tmp_path):
    expected = [
        _bundle("Breast Cancer Wisconsin", 569),
        _bundle("Adult Income", 48_842),
        _bundle("Covertype", 50_000),
        _bundle("MNIST2Class", 14_780),
    ]
    monkeypatch.setattr(datasets, "load_breast_cancer_dataset", lambda: expected[0])
    monkeypatch.setattr(datasets, "load_adult_income", lambda root, skip_downloads: expected[1])
    monkeypatch.setattr(
        datasets,
        "load_covertype_dataset",
        lambda root, seed, skip_downloads: expected[2],
    )
    monkeypatch.setattr(datasets, "load_mnist2class", lambda root, skip_downloads: expected[3])

    loaded = datasets.load_all_datasets(seed=42, skip_downloads=True, root=tmp_path)

    assert [(item.name, item.X.shape[0]) for item in loaded] == [
        ("Breast Cancer Wisconsin", 569),
        ("Adult Income", 48_842),
        ("Covertype", 50_000),
        ("MNIST2Class", 14_780),
    ]


def test_parse_adult_handles_test_header_periods_and_missing_values(tmp_path):
    path = tmp_path / "adult.test"
    path.write_text(
        "|1x3 Cross validator\n"
        "25, Private, 1234, Bachelors, 13, Never-married, ?, Sales, White, Male, 0, 0, 40, ?, <=50K.\n"
        "40, ?, 999, HS-grad, 9, Married-civ-spouse, Craft-repair, Husband, White, Male, 0, 0, 50, US, >50K.\n",
        encoding="utf-8",
    )

    X, y = datasets._parse_adult_file(path, is_test=True)

    assert X.shape == (2, 14)
    assert y.tolist() == [0, 1]
    assert X[0, 6] is None and X[0, 13] is None and X[1, 1] is None
    assert X[0, 0] == 25.0


def test_adult_loader_combines_all_rows_and_drops_none(monkeypatch, tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    for filename in datasets.ADULT_URLS:
        (raw_dir / filename).write_text("cached", encoding="utf-8")

    train_X = np.zeros((datasets.ADULT_TRAIN_ROWS, 14), dtype=object)
    test_X = np.zeros((datasets.ADULT_TEST_ROWS, 14), dtype=object)
    train_X[0, 1] = None
    test_X[0, 3] = None
    train_y = np.array([0] * 24_720 + [1] * 7_841, dtype=object)
    test_y = np.array([0] * 12_435 + [1] * 3_846, dtype=object)

    def fake_parse(path, is_test):
        return (test_X, test_y) if is_test else (train_X, train_y)

    monkeypatch.setattr(datasets, "_parse_adult_file", fake_parse)
    bundle = datasets.load_adult_income(tmp_path, skip_downloads=True)

    assert bundle.X.shape == (48_842, 14)
    assert bundle.y.shape == (48_842,)
    assert class_distribution(bundle.y) == datasets.ADULT_CLASS_COUNTS
    assert bundle.categorical_columns == datasets.ADULT_CATEGORICAL_COLUMNS


def test_downloadless_loaders_fail_loudly_without_caches(tmp_path):
    for loader in (
        lambda: datasets.load_adult_income(tmp_path, skip_downloads=True),
        lambda: datasets.load_covertype_dataset(tmp_path, seed=42, skip_downloads=True),
        lambda: datasets.load_mnist2class(tmp_path, skip_downloads=True),
    ):
        try:
            loader()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("loader silently proceeded without the required cache")


def test_streaming_mnist_arff_parser_keeps_only_zero_and_one(monkeypatch, tmp_path):
    path = tmp_path / "mnist.arff.gz"
    zeros = ",".join(["0"] * 784)
    ones = ",".join(["1"] * 784)
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write("@relation mnist_784\n@attribute pixel1 numeric\n@data\n")
        handle.write(f"{zeros},0\n")
        handle.write(f"{ones},5\n")
    monkeypatch.setattr(datasets, "MNIST_RAW_ROWS", 2)
    monkeypatch.setattr(datasets, "MNIST_ROWS", 1)

    X, y = datasets._parse_mnist_arff_gzip(path)

    assert X.shape == (1, 784)
    assert y.tolist() == [0]

    plain_path = tmp_path / "mnist.arff"
    plain_path.write_text(
        "@relation mnist_784\n@attribute pixel1 numeric\n@data\n"
        f"{zeros},0\n{ones},5\n",
        encoding="ascii",
    )
    plain_X, plain_y = datasets._parse_mnist_arff(plain_path)
    assert np.array_equal(plain_X, X)
    assert plain_y.tolist() == [0]


def test_download_rejects_incomplete_content_length(monkeypatch, tmp_path):
    class Response(io.BytesIO):
        headers = {"Content-Length": "4"}

    monkeypatch.setattr(
        datasets.urllib.request,
        "urlopen",
        lambda request, timeout: Response(b"abc"),
    )
    target = tmp_path / "source.arff"

    with pytest.raises(OSError, match="Incomplete download"):
        datasets._download("https://example.invalid/source.arff", target)

    assert not target.exists()
    assert not target.with_suffix(".arff.part").exists()
