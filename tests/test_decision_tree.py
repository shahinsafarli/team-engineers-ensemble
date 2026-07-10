import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

from src.trees.decision_tree import DecisionStump, DecisionTree
from src.utils.preprocessing import StandardScaler, stratified_train_test_split


def test_tree_learns_simple_axis_aligned_split():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTree(max_depth=1, random_state=7).fit(X, y)

    assert np.array_equal(tree.predict(X), y)
    assert tree.depth == 1
    assert tree.n_leaves == 2
    assert np.isclose(tree.predict_proba([[0.1], [0.9]]).sum(axis=1), 1).all()
    assert np.isclose(tree.feature_importances().sum(), 1.0)


def test_tree_handles_two_level_positive_gain_splits():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1])

    tree = DecisionTree(max_depth=2, random_state=0).fit(X, y)

    assert np.array_equal(tree.predict(X), y)


def test_tree_does_not_split_when_impurity_does_not_improve():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    tree = DecisionTree(max_depth=2, random_state=0).fit(X, y)

    assert tree.depth == 0
    assert tree.n_leaves == 1


def test_tree_edge_cases_single_label_and_depth_zero():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])

    single_label = DecisionTree(random_state=0).fit(X, y)
    root_only = DecisionTree(max_depth=0, random_state=0).fit(X, np.array([0, 1, 1]))

    assert np.array_equal(single_label.predict(X), y)
    assert root_only.depth == 0
    assert root_only.n_leaves == 1


def test_weighted_stump_respects_sample_weights():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    weights = np.array([0.05, 0.05, 0.45, 0.45])

    stump = DecisionStump(random_state=0).fit(X, y, sample_weight=weights)

    assert stump.depth == 1
    assert np.array_equal(stump.predict([[2.5], [0.5]]), np.array([1, 0], dtype=object))


def test_tree_matches_sklearn_reference_within_two_points():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        random_state=42,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ours = DecisionTree(criterion="entropy", random_state=42).fit(X_train, y_train)
    reference = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(
        X_train,
        y_train.astype(str),
    )

    ours_acc = np.mean(ours.predict(X_test).astype(str) == y_test.astype(str))
    ref_acc = np.mean(reference.predict(X_test) == y_test.astype(str))
    assert abs(ours_acc - ref_acc) <= 0.02
