"""Tests for the deterministic scrutiny evidence-weight.

Covers ``scrutiny_weight.compute_evidence_weight`` — the pure replacement for
the retired ``epistemic_assess_evidence`` LLM categorical (harness-backstops
item 3). Uses lightweight stubs for the evidence surface it reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from andamentum.epistemic.scrutiny_weight import compute_evidence_weight
from andamentum.epistemic.thresholds import (
    JUDGMENT_DOUBT_ESCALATION_THRESHOLD,
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


class TestEvidenceWeightDoubt:
    """``EvidenceWeight.doubt`` — the doubt of the judgments that produced
    the mass, measured in the same pass that produced it.

    Doubt never changes the label. It is a suspicion-raiser routed through
    the lazy-escalation ``Demand`` loop, and the tests below pin the
    properties a future "simplify this" change would break.
    """

    def test_doubt_is_none_when_no_distribution_was_captured(self) -> None:
        """Absence of a measured confidence is not evidence of low confidence.

        Adversarial counter-evidence and pre-Tier-0 data carry a hard
        ``support_judgment`` but no distribution. They contribute MASS
        (the verdict still counts them) but no DOUBT.

        PINS: coercing this to 0.0 would publish certainty the system never
        established; coercing it to 1.0 would escalate on every legacy item.
        Only ``None`` can never escalate.
        """
        items = [
            _Ev(support_judgment="supports", quality_score=0.8, corroboration_count=6),
            _Ev(support_judgment="supports", quality_score=0.8),
        ]
        w = compute_evidence_weight(items)
        assert w.supporting_mass > 0.0, "the items must still carry mass"
        assert w.doubt is None

    def test_doubt_is_none_when_no_directional_mass(self) -> None:
        """All-no_bearing evidence measures nothing about the DIRECTION.

        PINS: doubt is about the judgments that moved the verdict, not
        about every judgment that happened to be made.
        """
        items = [
            _Ev(
                judgment_distribution=[0.0, 0.0, 1.0],
                support_judgment="no_bearing",
                quality_score=0.8,
            )
        ]
        w = compute_evidence_weight(items)
        assert w.doubt is None

    def test_one_hot_judgments_give_near_zero_doubt(self) -> None:
        items = [
            _Ev(
                judgment_distribution=[1.0, 0.0, 0.0],
                support_judgment="supports",
                quality_score=0.8,
            ),
            _Ev(
                judgment_distribution=[1.0, 0.0, 0.0],
                support_judgment="supports",
                quality_score=0.8,
            ),
        ]
        w = compute_evidence_weight(items)
        assert w.doubt == pytest.approx(0.0, abs=0.02)

    def test_flat_judgments_give_doubt_above_the_escalation_threshold(self) -> None:
        """The escalation must be REACHABLE from realistic small-model output.

        A judge that returns 34/33/33 is exactly the "I cannot commit" shape
        local models produce. If that did not clear the threshold, the whole
        mechanism would be dead on arrival.
        """
        items = [
            _Ev(
                judgment_distribution=[0.34, 0.33, 0.33],
                support_judgment="supports",
                quality_score=0.8,
            )
        ]
        w = compute_evidence_weight(items)
        assert w.doubt is not None
        assert w.doubt >= JUDGMENT_DOUBT_ESCALATION_THRESHOLD

    def test_doubt_is_directional_mass_weighted_not_a_plain_mean(self) -> None:
        """Items weigh by how much they moved the verdict.

        Differential, not an absolute number: one confident high-quality
        item plus one flat negligible-quality item must stay LOW; swap the
        quality weights and it must rise. An unweighted mean gives the same
        answer both ways and flips this relation.
        """
        confident = _Ev(
            judgment_distribution=[1.0, 0.0, 0.0],
            support_judgment="supports",
        )
        flat = _Ev(
            judgment_distribution=[0.34, 0.33, 0.33],
            support_judgment="supports",
        )
        confident_dominates = compute_evidence_weight(
            [
                _Ev(**{**confident.__dict__, "quality_score": 1.0}),
                _Ev(**{**flat.__dict__, "quality_score": 0.02}),
            ]
        )
        flat_dominates = compute_evidence_weight(
            [
                _Ev(**{**confident.__dict__, "quality_score": 0.02}),
                _Ev(**{**flat.__dict__, "quality_score": 1.0}),
            ]
        )
        assert confident_dominates.doubt is not None
        assert flat_dominates.doubt is not None
        assert confident_dominates.doubt < flat_dominates.doubt
        # And the ordering is decisive, not a rounding difference.
        assert confident_dominates.doubt < JUDGMENT_DOUBT_ESCALATION_THRESHOLD
        assert flat_dominates.doubt >= JUDGMENT_DOUBT_ESCALATION_THRESHOLD

    def test_confident_opposing_judgments_are_conflicting_but_not_doubtful(
        self,
    ) -> None:
        """Claim-level CONFLICT and item-level DOUBT are different signals.

        Two judges, each certain, pointing opposite ways: the claim is
        conflicting, but no individual judgment was unsure.

        PINS: a future change that derives doubt from the mass split (an
        attractive shortcut) would report high doubt here and fail.
        """
        w = compute_evidence_weight(
            [
                _Ev(
                    judgment_distribution=[1.0, 0.0, 0.0],
                    support_judgment="supports",
                    quality_score=0.8,
                ),
                _Ev(
                    judgment_distribution=[0.0, 1.0, 0.0],
                    support_judgment="contradicts",
                    quality_score=0.8,
                ),
            ]
        )
        assert w.label == "conflicting"
        assert w.doubt == pytest.approx(0.0, abs=0.02)
