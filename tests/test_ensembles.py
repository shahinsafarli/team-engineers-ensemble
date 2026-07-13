import numpy as np

from src.bagging.random_forest import RandomForestClassifier
from src.boosting.adaboost import AdaBoostClassifier
from src.boosting.gradient_boosting import GradientBoostingClassifier


def make_binary_data(seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-1.0, scale=0.6, size=(50, 2))
    X1 = rng.normal(loc=1.0, scale=0.6, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


def test_adaboost_properties_and_staged_predictions():
    X, y = make_binary_data()

    model = AdaBoostClassifier(n_estimators=20, learning_rate=0.8, random_state=3).fit(X, y)
    staged = list(model.staged_predict(X))

    assert len(staged) == len(model.estimator_weights)
    assert len(model.estimator_errors) == len(model.estimator_weights)
    assert model.predict_proba(X[:5]).shape == (5, 2)
    assert np.mean(model.predict(X) == y) >= 0.9


def test_adaboost_samme_r_multiclass_bonus_path():
    rng = np.random.default_rng(2)
    X = np.vstack(
        [
            rng.normal([0, 0], 0.25, size=(20, 2)),
            rng.normal([2, 0], 0.25, size=(20, 2)),
            rng.normal([1, 2], 0.25, size=(20, 2)),
        ]
    )
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)

    model = AdaBoostClassifier(
        n_estimators=10,
        algorithm="SAMME.R",
        random_state=4,
    ).fit(X, y)

    assert model.predict_proba(X).shape == (60, 3)
    assert np.mean(model.predict(X) == y) >= 0.8


def test_random_forest_oob_and_feature_importances():
    X, y = make_binary_data()

    model = RandomForestClassifier(
        n_estimators=25,
        max_depth=5,
        oob_score=True,
        random_state=5,
    ).fit(X, y)

    assert model.predict_proba(X[:7]).shape == (7, 2)
    assert 0.0 <= model.oob_score_ <= 1.0
    assert np.isclose(model.feature_importances_.sum(), 1.0)
    assert np.mean(model.predict(X) == y) >= 0.9


def test_random_forest_predict_uses_hard_majority_vote():
    class FakeTree:
        def __init__(self, predictions):
            self._predictions = np.asarray(predictions, dtype=object)

        def predict(self, X):
            return self._predictions[: len(X)]

    forest = RandomForestClassifier(n_estimators=3)
    forest.classes_ = np.array([0, 1], dtype=object)
    forest.n_classes_ = 2
    forest.n_features_in_ = 1
    forest.estimators_ = [
        FakeTree([0, 0]),
        FakeTree([0, 1]),
        FakeTree([1, 1]),
    ]

    assert np.array_equal(forest.predict(np.array([[0.0], [1.0]])), np.array([0, 1], dtype=object))


def test_gradient_boosting_bonus_classifier():
    X, y = make_binary_data()

    model = GradientBoostingClassifier(n_estimators=30, learning_rate=0.3).fit(X, y)

    assert model.predict_proba(X[:3]).shape == (3, 2)
    assert len(model.train_loss_) == 30
    assert model.train_loss_[-1] <= model.train_loss_[0]
