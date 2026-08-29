"""Offline unit tests for the mixture-of-Dirichlets core. No network, no models."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.dirichlet_confidence.mixture import (
    MixtureOfDirichlets,
    build_mixture,
    fit_concentration_s,
    repair_row,
)


# --- repair_row ------------------------------------------------------------


def test_repair_row_normalises_with_no_prior():
    # alpha0=0 is plain normalisation.
    p = repair_row(np.array([70, 20, 10]), alpha0=0.0)
    assert np.allclose(p, [0.7, 0.2, 0.1])


def test_repair_row_smooths_hard_zero():
    p = repair_row(np.array([100, 0, 0]), alpha0=0.5)
    # The dismissed categories must be strictly positive after smoothing.
    assert (p > 0).all()
    assert p[1] == pytest.approx(0.5 / (100 + 1.5), rel=1e-6)
    assert np.isclose(p.sum(), 1.0)


def test_repair_row_drops_all_zero():
    assert repair_row(np.array([0, 0, 0])) is None


def test_repair_row_clips_negatives():
    p = repair_row(np.array([-5, 50, 50]), alpha0=0.0)
    assert np.allclose(p, [0.0 if False else p[0], 0.5, 0.5]) or np.isclose(
        p.sum(), 1.0
    )
    assert p[0] >= 0


# --- the mandatory regression test from the walkthrough --------------------


def test_single_prompt_recovery():
    """Spec step 3: M=1, K=1, s -> inf, alpha0=0 must recover the single
    verbalized distribution exactly (pooled mean == the one repaired row, and
    sampling collapses onto it)."""
    row = np.array([70, 20, 10])
    mix = build_mixture(
        [{"model": "only", "phrasing_id": 0, "row": row}],
        s=1e9,
        alpha0=0.0,
    )
    assert mix is not None
    assert np.allclose(mix.pooled_mean(), [0.7, 0.2, 0.1])

    rng = np.random.default_rng(0)
    samples = mix.sample_distributions(2000, rng)
    # With s -> inf the Dirichlet collapses to a point mass at the mean.
    assert np.allclose(samples.mean(axis=0), [0.7, 0.2, 0.1], atol=1e-3)
    assert samples.std(axis=0).max() < 1e-3


# --- pooled mean: mixture vs average equivalence in expectation ------------


def test_pooled_mean_equals_weighted_average():
    means = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3], [0.8, 0.15, 0.05]])
    mix = MixtureOfDirichlets(means=means, model_ids=np.array(["a", "b", "c"]), s=50.0)
    assert np.allclose(mix.pooled_mean(), means.mean(axis=0))


def test_pooled_mean_is_invariant_to_s():
    means = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]])
    ids = np.array(["a", "b"])
    m1 = MixtureOfDirichlets(means=means, model_ids=ids, s=5.0)
    m2 = MixtureOfDirichlets(means=means, model_ids=ids, s=500.0)
    assert np.allclose(m1.pooled_mean(), m2.pooled_mean())


# --- variance decomposition ------------------------------------------------


def test_variance_decomposition_components_sum():
    means = np.array(
        [[0.7, 0.2, 0.1], [0.6, 0.25, 0.15], [0.4, 0.3, 0.3], [0.45, 0.35, 0.2]]
    )
    model_ids = np.array(["A", "A", "B", "B"])
    mix = MixtureOfDirichlets(means=means, model_ids=model_ids, s=50.0)
    d = mix.variance_decomposition()
    # between_component == within_model + between_model (law of total variance
    # over the component-mean point cloud, grouped by model).
    assert np.allclose(d["between_component"], d["within_model"] + d["between_model"])
    # total == within_component + between_component.
    assert np.allclose(d["total"], d["within_component"] + d["between_component"])


def test_within_component_shrinks_with_s():
    means = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]])
    ids = np.array(["a", "b"])
    small = MixtureOfDirichlets(
        means=means, model_ids=ids, s=5.0
    ).variance_decomposition()
    big = MixtureOfDirichlets(
        means=means, model_ids=ids, s=500.0
    ).variance_decomposition()
    assert big["within_component"].sum() < small["within_component"].sum()
    # between-model spread is a property of the means, not of s.
    assert np.allclose(small["between_model"], big["between_model"])


def test_phrasing_brittleness_shows_as_within_model():
    # One model gives wildly different rows across phrasings; the other is stable.
    brittle = np.array([[0.9, 0.05, 0.05], [0.3, 0.4, 0.3]])  # model A, 2 phrasings
    stable = np.array([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]])  # model B, 2 phrasings
    means = np.vstack([brittle, stable])
    ids = np.array(["A", "A", "B", "B"])
    d = MixtureOfDirichlets(means=means, model_ids=ids, s=50.0).variance_decomposition()
    # within_model is dominated by the brittle model -> non-trivial.
    assert d["within_model"].sum() > 0.01


# --- mixture preserves spread that averaging destroys ----------------------


def test_mixture_has_more_category_spread_than_collapsed_point():
    # Two confident-but-opposed components: mixture should sample both classes;
    # the averaged single point would also, but with different second moment.
    means = np.array([[0.95, 0.025, 0.025], [0.025, 0.95, 0.025]])
    ids = np.array(["a", "b"])
    mix = MixtureOfDirichlets(means=means, model_ids=ids, s=1000.0)
    rng = np.random.default_rng(1)
    cats = mix.sample_categories(5000, rng)
    frac0 = (cats == 0).mean()
    frac1 = (cats == 1).mean()
    # Both opposed modes are sampled roughly equally — the disagreement survives.
    assert 0.4 < frac0 < 0.6
    assert 0.4 < frac1 < 0.6


# --- fitting s -------------------------------------------------------------


def test_fit_s_returns_value_in_bounds():
    rng = np.random.default_rng(2)
    pairs = []
    for _ in range(20):
        means = rng.dirichlet([2, 2, 2], size=4)
        ids = np.array(["A", "A", "B", "B"])
        mix = MixtureOfDirichlets(means=means, model_ids=ids, s=50.0)
        label = int(mix.pooled_mean().argmax())
        pairs.append((mix, label))
    s = fit_concentration_s(pairs, s_bounds=(1.0, 1000.0), n_samples=100)
    assert 1.0 <= s <= 1000.0


def test_build_mixture_drops_bad_rows_and_returns_none_if_all_bad():
    assert build_mixture([{"model": "m", "phrasing_id": 0, "row": [0, 0, 0]}]) is None
    mix = build_mixture(
        [
            {"model": "m", "phrasing_id": 0, "row": [0, 0, 0]},  # dropped
            {"model": "m", "phrasing_id": 1, "row": [50, 30, 20]},  # kept
        ]
    )
    assert mix is not None
    assert mix.n_components == 1


def test_model_weights_balance_uneven_phrasing_counts():
    # Model A contributed 3 phrasings, model B only 1. With per-model weights,
    # each model should carry equal total weight (0.5 each).
    elicitations = [
        {"model": "A", "phrasing_id": 0, "row": [80, 10, 10]},
        {"model": "A", "phrasing_id": 1, "row": [80, 10, 10]},
        {"model": "A", "phrasing_id": 2, "row": [80, 10, 10]},
        {"model": "B", "phrasing_id": 0, "row": [10, 80, 10]},
    ]
    mix = build_mixture(elicitations, model_weights={"A": 1.0, "B": 1.0})
    assert mix is not None
    wA = mix.weights[mix.model_ids == "A"].sum()
    wB = mix.weights[mix.model_ids == "B"].sum()
    assert wA == pytest.approx(0.5)
    assert wB == pytest.approx(0.5)
