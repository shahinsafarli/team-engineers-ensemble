"""Opt-in full-scale spatial-index smoke test.

Run with ``RUN_SLOW_TESTS=1 python -m pytest tests/test_dbscan_performance.py -q``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.unsupervised.dbscan import DBSCAN


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") != "1", reason="set RUN_SLOW_TESTS=1")
def test_dbscan_50000_point_deterministic_smoke():
    rng = np.random.default_rng(42)
    X = rng.uniform(size=(50_000, 2))
    first = DBSCAN(eps=0.0006, min_samples=3).fit(X).labels_
    second = DBSCAN(eps=0.0006, min_samples=3).fit(X).labels_

    assert first.shape == (50_000,)
    assert np.array_equal(first, second)
