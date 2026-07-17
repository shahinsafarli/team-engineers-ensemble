import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from src.metrics.evaluation import bias_variance_01, classification_metrics
from src.unsupervised import dbscan as dbscan_module
from src.unsupervised.dbscan import DBSCAN, _ExactKDTree, kth_neighbor_distances
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA
from src.utils.preprocessing import (
    StandardScaler,
    MixedTypePreprocessor,
    class_distribution,
    flip_labels,
    largest_remainder_allocation,
    random_oversample,
    stratified_subsample,
    stratified_subsample_indices,
    stratified_train_test_split,
)


def test_standard_scaler_and_stratified_split():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([0] * 10 + [1] * 10)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
    )
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    assert X_test.shape[0] == 4
    assert set(y_test.tolist()) == {0, 1}
    assert np.allclose(X_scaled.mean(axis=0), 0)


def test_oversample_flip_and_distribution_helpers():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 0, 0, 0, 0, 1])

    X_over, y_over = random_oversample(X, y, random_state=0)
    flipped = flip_labels(y, 0.5, random_state=1)

    assert X_over.shape[0] == 10
    assert class_distribution(y_over) == {"0": 5, "1": 5}
    assert np.sum(flipped != y.astype(object)) == 3


def test_largest_remainder_subsample_is_exact_unique_and_deterministic():
    counts = {1: 211_840, 2: 283_301, 3: 35_754, 4: 2_747, 5: 9_493, 6: 17_367, 7: 20_510}
    y = np.concatenate([np.full(count, label, dtype=np.int8) for label, count in counts.items()])
    expected = {1: 18_230, 2: 24_380, 3: 3_077, 4: 236, 5: 817, 6: 1_495, 7: 1_765}

    assert largest_remainder_allocation(y, 50_000) == expected
    first = stratified_subsample_indices(y, 50_000, random_state=42)
    second = stratified_subsample_indices(y, 50_000, random_state=42)

    assert np.array_equal(first, second)
    assert first.size == np.unique(first).size == 50_000
    assert class_distribution(y[first]) == {str(label): count for label, count in expected.items()}


def test_stratified_subsample_preserves_exact_requested_size():
    X = np.arange(66, dtype=float).reshape(33, 2)
    y = np.array([0] * 20 + [1] * 10 + [2] * 3)
    X_sample, y_sample = stratified_subsample(X, y, 17, random_state=7)

    assert X_sample.shape == (17, 2)
    assert y_sample.shape == (17,)
    assert class_distribution(y_sample) == {"0": 10, "1": 5, "2": 2}


def test_mixed_type_preprocessor_imputes_without_leakage_or_row_drops():
    X_train = np.array(
        [[1.0, "A"], [3.0, "A"], [None, "B"], [5.0, None]],
        dtype=object,
    )
    X_test = np.array([[1000.0, "unseen"], [None, None]], dtype=object)
    transformer = MixedTypePreprocessor((0,), (1,)).fit(X_train)

    train = transformer.transform(X_train)
    test = transformer.transform(X_test)

    assert train.shape[0] == X_train.shape[0]
    assert test.shape[0] == X_test.shape[0]
    assert np.all(np.isfinite(train)) and np.all(np.isfinite(test))
    assert np.isclose(train[:, 0].mean(), 0.0)
    assert transformer.numeric_medians_.tolist() == [3.0]
    assert transformer.categorical_modes_ == ("A",)
    assert np.all(test[0, 1:] == 0.0)  # category appeared only in held-out rows
    assert test[1, 1:].sum() == 1.0  # missing value is imputed to fitted mode

    repeated = MixedTypePreprocessor((0,), (1,)).fit_transform(X_train)
    assert np.array_equal(train, repeated)


def test_pca_orders_variance_and_transforms():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    X = np.column_stack([x, 0.2 * x + rng.normal(scale=0.05, size=100)])

    pca = PCA(n_components=2).fit(X)
    transformed = pca.transform(X)

    assert transformed.shape == (100, 2)
    assert pca.explained_variance_ratio_[0] > pca.explained_variance_ratio_[1]
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)


def test_kmeans_finds_two_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([-2, 0], 0.2, size=(20, 2)),
            rng.normal([2, 0], 0.2, size=(20, 2)),
        ]
    )

    model = KMeans(n_clusters=2, random_state=0).fit(X)

    assert model.labels_.shape == (40,)
    assert model.inertia_ < 10


def test_dbscan_marks_noise_and_clusters():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
        ]
    )

    labels = DBSCAN(eps=0.25, min_samples=2).fit(X).labels_

    assert set(labels.tolist()) == {0, -1}


def test_exact_kdtree_radius_queries_match_brute_force():
    rng = np.random.default_rng(4)
    fixtures = [
        rng.normal(size=(80, 3)),
        np.vstack([np.zeros((12, 2)), np.ones((12, 2))]),
        np.column_stack([np.arange(60), np.zeros(60)]).astype(float),
        rng.normal(scale=0.01, size=(70, 2)),
    ]
    for X in fixtures:
        tree = _ExactKDTree(X, leaf_size=8)
        radius = 0.2 if X.shape[1] == 2 else 1.0
        for index in range(0, X.shape[0], max(1, X.shape[0] // 9)):
            brute = np.where(np.linalg.norm(X - X[index], axis=1) <= radius)[0]
            assert np.array_equal(tree.query_radius(X[index], radius), brute)


def test_kth_neighbor_distances_match_brute_force_with_duplicates():
    rng = np.random.default_rng(8)
    X = np.vstack([rng.normal(size=(45, 3)), np.zeros((5, 3))])
    brute = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    brute.sort(axis=1)

    assert np.allclose(kth_neighbor_distances(X, k=5), np.sort(brute[:, 5]))


def test_dbscan_labels_match_sklearn_on_sparse_dense_and_duplicate_fixtures():
    rng = np.random.default_rng(11)
    fixtures = [
        (rng.normal(size=(100, 2)), 0.35, 4),
        (rng.normal(scale=0.03, size=(90, 2)), 0.08, 5),
        (np.vstack([np.zeros((20, 2)), np.ones((20, 2)), [[5.0, 5.0]]]), 0.01, 3),
    ]
    for X, eps, min_samples in fixtures:
        expected = SklearnDBSCAN(eps=eps, min_samples=min_samples, algorithm="brute").fit_predict(X)
        actual = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_
        assert np.array_equal(actual, expected)


def test_dbscan_never_requests_a_quadratic_array(monkeypatch):
    rng = np.random.default_rng(15)
    X = rng.normal(size=(600, 2))
    original_empty = np.empty
    original_zeros = np.zeros

    def guarded_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) >= 2 and shape[0] == shape[1] == X.shape[0]:
            raise AssertionError("quadratic allocation requested")
        return original_empty(shape, *args, **kwargs)

    def guarded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) >= 2 and shape[0] == shape[1] == X.shape[0]:
            raise AssertionError("quadratic allocation requested")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(dbscan_module.np, "empty", guarded_empty)
    monkeypatch.setattr(dbscan_module.np, "zeros", guarded_zeros)
    assert DBSCAN(eps=0.15, min_samples=4).fit(X).labels_.shape == (600,)
    assert kth_neighbor_distances(X, k=5).shape == (600,)


def test_metrics_helpers():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
    metrics = classification_metrics(y_true, y_pred, proba, labels=np.array([0, 1]))
    bv = bias_variance_01(
        np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]]),
        y_true,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["auc_roc"] == 1.0
    assert set(bv) == {"bias_squared", "variance", "expected_loss"}
