"""Calibration + accuracy metrics for categorical confidence.

Pure numpy. These operate on stacked predictive probability matrices
``probs: (N, C)`` and integer ``labels: (N,)``. The mixture-specific
sampled-NLL lives on :class:`~mixture.MixtureOfDirichlets`; the dataset-level
wrapper :func:`mean_sampled_nll` is here so all reportable metrics sit together.
"""

from __future__ import annotations

import numpy as np

from .mixture import MixtureOfDirichlets


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy of ``argmax(probs)`` against ``labels``."""
    return float((probs.argmax(axis=1) == labels).mean())


def roc_auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Binary ROC-AUC via the rank-sum (Mann-Whitney U) identity. No sklearn.

    ``scores`` is a 1-D array of decision scores (higher = more "positive");
    ``positives`` is a boolean array marking the positive class. Returns the
    probability that a random positive outranks a random negative; ties count
    half. Returns NaN if either class is empty (AUC undefined).
    """
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups so ties contribute 0.5.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def error_detection_auroc(band: np.ndarray, correct: np.ndarray) -> float:
    """How well the uncertainty band flags the model's *wrong* answers.

    This grades the UNCERTAINTY ESTIMATE itself, not the point prediction: a good
    band is large when the answer is wrong and small when it's right. We treat
    "answer is wrong" as the positive class and the band as the alarm score, then
    take ROC-AUC. 0.5 = the band is noise (no relationship to error); 1.0 = the
    band perfectly ranks every wrong answer above every right one. NaN if every
    answer is right or every answer is wrong (undefined).
    """
    band = np.asarray(band, dtype=float)
    wrong = ~np.asarray(correct, dtype=bool)
    return roc_auc(band, wrong)


def risk_coverage(band: np.ndarray, correct: np.ndarray) -> list[dict]:
    """Risk-coverage curve: abstain on the highest-band claims first.

    Sort by band ascending (most-confident first); for each coverage level report
    the accuracy on the retained (lowest-band) fraction. If the band is
    informative, accuracy should rise as coverage shrinks. Returns a list of
    {coverage, n, accuracy} from full coverage down to ~20%.
    """
    band = np.asarray(band, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    order = band.argsort()  # most confident (smallest band) first
    sorted_correct = correct[order]
    n = len(band)
    out = []
    for frac in (1.0, 0.8, 0.6, 0.4, 0.2):
        k = max(1, int(round(n * frac)))
        out.append(
            {"coverage": frac, "n": k, "accuracy": float(sorted_correct[:k].mean())}
        )
    return out


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Top-class ECE with equal-width bins over ``[0, 1]``.

    ``confidences`` is the predicted probability of the predicted class;
    ``correct`` is a boolean array of whether that prediction was right.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error vs the one-hot label."""
    n, _ = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean ``-log p_y`` using the predictive mean (insensitive to ``s``)."""
    n = len(labels)
    p_y = probs[np.arange(n), labels]
    return float(-np.log(np.clip(p_y, 1e-12, 1.0)).mean())


def mean_sampled_nll(
    mixtures_and_labels: list[tuple[MixtureOfDirichlets, int]],
    s: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Mean sampled-NLL across a dataset at concentration ``s``.

    This is the ``s``-sensitive metric where the Dirichlet mixture (config D) is
    expected to beat the point mixture (config C); the two share a predictive
    mean and therefore tie on accuracy / ECE / Brier.
    """
    vals = [
        mix.with_s(s).sampled_nll(y, n_samples, rng) for mix, y in mixtures_and_labels
    ]
    return float(np.mean(vals)) if vals else float("nan")


def reliability_curve(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Per-bin (mean confidence, empirical accuracy, count) for plotting."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    conf_out: list[float] = []
    acc_out: list[float] = []
    count_out: list[float] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count_out.append(float(mask.sum()))
        if mask.sum() == 0:
            conf_out.append(float((lo + hi) / 2))
            acc_out.append(0.0)
        else:
            conf_out.append(float(confidences[mask].mean()))
            acc_out.append(float(correct[mask].mean()))
    return {"confidence": conf_out, "accuracy": acc_out, "count": count_out}
