"""Offline end-to-end test of the benchmark scoring pipeline.

Exercises build_all_mixtures -> evaluate (fit s, A/B/C/D, per-model baselines,
variance decomposition, top-disagreement) on a synthetic cache — no model calls.
"""

from __future__ import annotations

import numpy as np

from experiments.dirichlet_confidence.benchmark import (
    _cache_key,
    build_all_mixtures,
    evaluate,
)
from experiments.dirichlet_confidence.elicitation import n_phrasings
from experiments.dirichlet_confidence.scifact import Example, ID2LABEL


def _synthetic(n=60, seed=0):
    """Build synthetic examples + a cache with model-family bias + phrasing noise."""
    rng = np.random.default_rng(seed)
    models = ["mA", "mB", "mC"]
    biases = {
        "mA": np.array([0.05, -0.02, -0.03]),
        "mB": np.array([-0.05, 0.03, 0.02]),
        "mC": np.array([0.0, -0.01, 0.01]),
    }
    examples: list[Example] = []
    cache: dict[str, list[float] | None] = {}
    for i in range(n):
        label = i % 3
        truth = np.array([0.2, 0.2, 0.2])
        truth[label] = 0.6
        cid = f"item{i}"
        examples.append(
            Example(
                claim_id=cid,
                claim=f"claim {i}",
                evidence="ev",
                label=label,
                label_name=ID2LABEL[label],
            )
        )
        for m in models:
            for k in range(n_phrasings()):
                p = truth + biases[m] + rng.normal(0, 0.03, size=3)
                p = np.clip(p, 0.01, None)
                p = p / p.sum()
                cache[_cache_key(cid, m, k)] = list(np.round(p * 100))
    return examples, cache, models


def test_evaluate_runs_and_orders_metrics_sensibly():
    examples, cache, models = _synthetic(n=60)
    mixtures = build_all_mixtures(examples, cache, models, s_initial=50.0, alpha0=0.5)
    assert all(m is not None for m in mixtures)

    results = evaluate(
        examples, mixtures, models, dev_frac=0.3, n_mc_samples=300, seed=1
    )

    # s fit within bounds.
    assert 1.0 <= results["s_fit"] <= 1000.0

    # All four configs scored on a non-empty test split.
    for key in (
        "A_single",
        "B_multi_phrasing_mean",
        "C_full_point_mixture",
        "D_dirichlet_mixture",
    ):
        assert results[key]["n"] > 0
        assert 0.0 <= results[key]["accuracy"] <= 1.0

    # C and D share a predictive mean -> identical accuracy/ECE/Brier.
    c, d = results["C_full_point_mixture"], results["D_dirichlet_mixture"]
    assert np.isclose(c["accuracy"], d["accuracy"])
    assert np.isclose(c["brier"], d["brier"])
    # But sampled-NLL is reported for both and is a real number.
    assert np.isfinite(c["sampled_nll"]) and np.isfinite(d["sampled_nll"])

    # Per-model section present for every model.
    assert set(results["per_model"]) == set(models)

    # Variance decomposition: between-model spread should be non-zero given the
    # injected model-family bias.
    vd = results["mean_variance_decomposition"]
    assert vd["between_model"] > 0

    # Top disagreement list is populated.
    assert len(results["top_disagreement"]) > 0


def test_synthetic_signal_is_recovered():
    # With a clear majority-class signal, the point mixture should be accurate.
    examples, cache, models = _synthetic(n=90, seed=3)
    mixtures = build_all_mixtures(examples, cache, models, s_initial=50.0, alpha0=0.5)
    results = evaluate(
        examples, mixtures, models, dev_frac=0.3, n_mc_samples=300, seed=2
    )
    assert results["C_full_point_mixture"]["accuracy"] > 0.8
