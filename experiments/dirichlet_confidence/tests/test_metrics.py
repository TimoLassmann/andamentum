"""Offline unit tests for the metrics module."""

from __future__ import annotations

import numpy as np

from experiments.dirichlet_confidence.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    mean_sampled_nll,
    negative_log_likelihood,
    reliability_curve,
)
from experiments.dirichlet_confidence.mixture import MixtureOfDirichlets


def test_accuracy_perfect_and_zero():
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    assert accuracy(probs, np.array([0, 1])) == 1.0
    assert accuracy(probs, np.array([1, 0])) == 0.0


def test_brier_zero_for_perfect_confident():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert brier_score(probs, np.array([0, 1])) == 0.0


def test_brier_positive_for_wrong():
    probs = np.array([[0.0, 1.0, 0.0]])
    # squared error: (0-1)^2 + (1-0)^2 = 2
    assert brier_score(probs, np.array([0])) == 2.0


def test_ece_perfectly_calibrated_is_zero():
    # 100 predictions all at confidence 1.0 and all correct -> ECE 0.
    conf = np.ones(100)
    correct = np.ones(100, dtype=bool)
    assert expected_calibration_error(conf, correct) == 0.0


def test_ece_detects_overconfidence():
    # Confidence 0.9 but only 50% correct -> ECE ~ 0.4.
    conf = np.full(100, 0.9)
    correct = np.array([True, False] * 50)
    assert expected_calibration_error(conf, correct) > 0.3


def test_negative_log_likelihood():
    probs = np.array([[0.5, 0.5]])
    nll = negative_log_likelihood(probs, np.array([0]))
    assert np.isclose(nll, -np.log(0.5))


def test_sampled_nll_is_s_sensitive():
    # A 2-component disagreeing mixture; the true label is the minority.
    means = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    ids = np.array(["a", "b"])
    mix = MixtureOfDirichlets(means=means, model_ids=ids, s=50.0)
    rng = np.random.default_rng(0)
    nll_small_s = mean_sampled_nll([(mix, 2)], s=2.0, n_samples=4000, rng=rng)
    nll_large_s = mean_sampled_nll([(mix, 2)], s=500.0, n_samples=4000, rng=rng)
    # Different s gives a different sampled-NLL (the whole point of fitting s).
    assert not np.isclose(nll_small_s, nll_large_s, atol=1e-3)


def test_reliability_curve_shape():
    conf = np.linspace(0, 1, 50)
    correct = conf > 0.5
    curve = reliability_curve(conf, correct, n_bins=5)
    assert len(curve["confidence"]) == 5
    assert len(curve["accuracy"]) == 5
    assert sum(curve["count"]) == 50
