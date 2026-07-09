"""Tests for the deterministic scrutiny evidence-weight.

Covers ``scrutiny_weight.compute_evidence_weight`` — the pure replacement for
the retired ``epistemic_assess_evidence`` LLM categorical (harness-backstops
item 3). Uses lightweight stubs for the evidence surface it reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from andamentum.epistemic.scrutiny_weight import compute_evidence_weight
from andamentum.epistemic.thresholds import (
    SCRUTINY_STRONG_MASS_THRESHOLD,
    SCRUTINY_WEAK_MASS_THRESHOLD,
)


@dataclass
class _Ev:
    """Minimal evidence stub (duck-typed to the module's Protocol)."""

    judgment_distribution: list[float] | None = None
    support_judgment: str | None = None
    quality_score: float | None = None
    corroboration_count: int = 1


def _supporting(quality: float = 0.8, corr: int = 1) -> _Ev:
    return _Ev(
        judgment_distribution=[0.9, 0.05, 0.05],
        support_judgment="supports",
        quality_score=quality,
        corroboration_count=corr,
    )


def _contradicting(quality: float = 0.8, corr: int = 1) -> _Ev:
    return _Ev(
        judgment_distribution=[0.05, 0.9, 0.05],
        support_judgment="contradicts",
        quality_score=quality,
        corroboration_count=corr,
    )


class TestVerdictLabels:
    def test_empty_evidence_is_weak(self) -> None:
        w = compute_evidence_weight([])
        assert w.label == "weak"
        assert w.supporting_mass == 0.0 and w.contradicting_mass == 0.0

    def test_single_supporting_item_is_moderate(self) -> None:
        # One high-quality item clears the weak floor but not the strong cut.
        w = compute_evidence_weight([_supporting()])
        assert w.label == "moderate"

    def test_heavily_corroborated_support_is_strong(self) -> None:
        w = compute_evidence_weight([_supporting(corr=6)])
        assert w.supporting_mass >= SCRUTINY_STRONG_MASS_THRESHOLD
        assert w.label == "strong"

    def test_multiple_supporting_items_reach_strong(self) -> None:
        w = compute_evidence_weight([_supporting(), _supporting(), _supporting()])
        assert w.label == "strong"

    def test_strong_contradiction_still_passes_as_strong(self) -> None:
        # Scrutiny weight is direction-agnostic: strong evidence AGAINST a claim
        # is still strong evidential weight (the direction verdict is downstream).
        w = compute_evidence_weight([_contradicting(corr=6)])
        assert w.label == "strong"
        assert w.contradicting_mass >= SCRUTINY_STRONG_MASS_THRESHOLD

    def test_balanced_conflict_is_conflicting(self) -> None:
        w = compute_evidence_weight([_supporting(), _contradicting()])
        assert w.label == "conflicting"

    def test_dominant_support_with_minor_dissent_is_not_conflicting(self) -> None:
        # A lot of support plus one weak dissent should not read as conflict.
        items = [_supporting(corr=6), _supporting(corr=6), _contradicting(quality=0.3)]
        w = compute_evidence_weight(items)
        assert w.label == "strong"

    def test_mostly_no_bearing_is_weak(self) -> None:
        thin = _Ev(
            judgment_distribution=[0.2, 0.1, 0.7],
            support_judgment="no_bearing",
            quality_score=0.5,
        )
        w = compute_evidence_weight([thin])
        assert w.supporting_mass < SCRUTINY_WEAK_MASS_THRESHOLD
        assert w.label == "weak"


class TestFallbackAndQuality:
    def test_hard_vote_fallback_without_distribution(self) -> None:
        # Evidence with no distribution still contributes via support_judgment.
        e = _Ev(support_judgment="supports", quality_score=0.8, corroboration_count=6)
        w = compute_evidence_weight([e])
        assert w.label == "strong"

    def test_missing_quality_uses_neutral_prior(self) -> None:
        # No quality_score → SCRUTINY_DEFAULT_QUALITY (0.5), still enough for
        # a single item to clear the weak floor.
        e = _Ev(
            judgment_distribution=[0.9, 0.05, 0.05],
            support_judgment="supports",
            quality_score=None,
        )
        w = compute_evidence_weight([e])
        assert w.label in ("moderate", "strong")
        assert w.supporting_mass >= SCRUTINY_WEAK_MASS_THRESHOLD

    def test_low_quality_reduces_mass(self) -> None:
        strong_q = compute_evidence_weight([_supporting(quality=1.0)]).supporting_mass
        weak_q = compute_evidence_weight([_supporting(quality=0.1)]).supporting_mass
        assert weak_q < strong_q
