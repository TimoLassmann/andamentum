"""Offline tests for elicitation plumbing (no model calls)."""

from __future__ import annotations

import numpy as np

from experiments.dirichlet_confidence.elicitation import (
    VerbalizedDistribution,
    n_phrasings,
)
from experiments.dirichlet_confidence.phrasings import PHRASINGS, render


def test_verbalized_distribution_to_row():
    d = VerbalizedDistribution(support=70, contradict=20, noinfo=10)
    assert np.array_equal(d.to_row(), np.array([70.0, 20.0, 10.0]))


def test_n_phrasings_matches_templates():
    assert n_phrasings() == len(PHRASINGS) == 5


def test_render_fills_claim_and_evidence():
    out = render(PHRASINGS[0], claim="CLAIMTEXT", evidence="EVTEXT")
    assert "CLAIMTEXT" in out and "EVTEXT" in out


def test_all_phrasings_render_without_keyerror():
    for p in PHRASINGS:
        out = render(p, claim="c", evidence="e")
        assert "c" in out and "e" in out
