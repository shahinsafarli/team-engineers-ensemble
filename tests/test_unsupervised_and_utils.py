import numpy as np

from src.metrics.evaluation import bias_variance_01, classification_metrics
from src.unsupervised.dbscan import DBSCAN
from src.unsupervised.kmeans import KMeans
from src.unsupervised.pca import PCA
from src.utils.preprocessing import (
    StandardScaler,
    class_distribution,
    flip_labels,
    random_oversample,
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
