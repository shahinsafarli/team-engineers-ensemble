"""Evaluation helpers shared by experiment scripts."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def safe_roc_auc(
    y_true: ArrayLike,
    y_proba: ArrayLike | None,
    labels: ArrayLike | None = None,
) -> float:
    """Compute binary or OVR macro AUC, returning NaN when undefined.

    AUC is undefined in several well-known situations (a single class
    present, a probability matrix that doesn't line up with the label
    set, non-finite probabilities, etc.). Each of those is checked for
    explicitly below so the NaN fallback path is reached deliberately
    rather than by catching whatever ``ValueError`` scikit-learn happens
    to raise. A narrow ``except`` remains only as a last-resort net for
    genuinely unanticipated scikit-learn errors, and it logs a warning
    (rather than swallowing the failure silently) so such cases are
    still visible during a run.
    """

    if y_proba is None:
        return float("nan")

    y_true_arr = np.asarray(y_true).astype(str)
    proba = np.asarray(y_proba, dtype=float)

    if labels is None:
        label_arr = np.unique(y_true_arr)
    else:
        label_arr = np.asarray(labels).astype(str)

    # Precondition 1: AUC needs at least two classes actually present.
    present = np.unique(y_true_arr)
    if present.size < 2:
        return float("nan")

    # Precondition 2: every observed label must be one of the declared
    # labels, or scikit-learn's OVR/label-alignment logic will fail.
    if not np.isin(present, label_arr).all():
        return float("nan")

    # Precondition 3: probabilities must be finite and have a column for
    # each candidate label (or, for the binary shortcut, at least one
    # column) -- otherwise indexing/alignment below is meaningless.
    if proba.ndim != 2 or not np.all(np.isfinite(proba)):
        return float("nan")
    if label_arr.size > 2 and proba.shape[1] < label_arr.size:
        return float("nan")
    if label_arr.size == 2 and proba.shape[1] not in (1, 2):
        return float("nan")

    try:
        if label_arr.size == 2:
            positive_col = 1 if proba.shape[1] > 1 else 0
            return float(roc_auc_score(y_true_arr, proba[:, positive_col]))
        return float(
            roc_auc_score(
                y_true_arr,
                proba,
                labels=label_arr,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        # All known failure modes are filtered out above, so reaching
        # this point means scikit-learn rejected the inputs for a reason
        # not covered by the explicit checks. Log it instead of silently
        # returning NaN so the run's logs surface the anomaly.
        logger.warning(
            "safe_roc_auc: roc_auc_score raised ValueError after passing "
            "all precondition checks (n_labels=%d, proba.shape=%s); "
            "returning NaN.",
            label_arr.size,
            proba.shape,
        )
        return float("nan")


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: ArrayLike | None = None,
    labels: ArrayLike | None = None,
) -> dict[str, float]:
    """Return the three metrics required by the project brief."""

    y_true_arr = np.asarray(y_true).astype(str)
    y_pred_arr = np.asarray(y_pred).astype(str)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "auc_roc": safe_roc_auc(y_true, y_proba, labels=labels),
    }


def mean_std(values: list[float] | NDArray[np.float64]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{np.nanmean(arr):.3f} +/- {np.nanstd(arr):.3f}"


def bias_variance_01(
    predictions: ArrayLike,
    y_true: ArrayLike,
) -> dict[str, float]:
    """Breiman-style 0-1 classification bias/variance summary.

    ``predictions`` has shape (n_models, n_samples). Bias is the error
    of the main prediction; variance is the expected disagreement with
    that main prediction.
    """

    pred = np.asarray(predictions)
    y_arr = np.asarray(y_true)
    if pred.ndim != 2:
        raise ValueError("predictions must be a 2D array")
    main_predictions = []
    for col in pred.T:
        labels, counts = np.unique(col, return_counts=True)
        main_predictions.append(labels[np.argmax(counts)])
    main = np.asarray(main_predictions)
    bias_sq = np.mean(main != y_arr)
    variance = np.mean(pred != main[None, :])
    loss = np.mean(pred != y_arr[None, :])
    return {
        "bias_squared": float(bias_sq),
        "variance": float(variance),
        "expected_loss": float(loss),
    }


