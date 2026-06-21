"""Tests for the Tier 0 verbalized-confidence signal.

Covers the pure maths in ``judgment_signal`` and the derived properties +
serialization round-trip on the ``Evidence`` entity.
"""

from __future__ import annotations

import math

import pytest

from andamentum.epistemic.entities.evidence import Evidence
from andamentum.epistemic.judgment_signal import (
    JUDGMENT_CLASSES,
    argmax_verdict,
    distribution_confidence,
    distribution_entropy,
    distribution_is_one_hot,
    normalize_distribution,
)
from andamentum.epistemic.judge import apply_judgment
from andamentum.epistemic.agents.output_models import EvidenceJudgmentOutput
from andamentum.epistemic.thresholds import JUDGMENT_ONE_HOT_THRESHOLD


class TestNormalize:
    def test_normalises_to_sum_one(self) -> None:
        d = normalize_distribution(85, 10, 5)
        assert sum(d) == pytest.approx(1.0)
        assert d[0] == pytest.approx(0.85)

    def test_non_100_sum_is_renormalised(self) -> None:
        # Small models drift off exactly 100; normalise by the actual sum.
        d = normalize_distribution(8, 1, 1)
        assert d == pytest.approx([0.8, 0.1, 0.1])

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            normalize_distribution(-1, 50, 51)

    def test_all_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            normalize_distribution(0, 0, 0)


class TestDerivedSignals:
    def test_argmax_verdict_matches_classes(self) -> None:
        assert argmax_verdict([0.7, 0.2, 0.1]) == "supports"
        assert argmax_verdict([0.2, 0.7, 0.1]) == "contradicts"
        assert argmax_verdict([0.1, 0.2, 0.7]) == "no_bearing"

    def test_argmax_tie_breaks_by_class_order(self) -> None:
        # supports > contradicts > no_bearing on exact ties (deterministic).
        assert argmax_verdict([0.5, 0.5, 0.0]) == "supports"
        assert argmax_verdict([0.0, 0.5, 0.5]) == "contradicts"

    def test_confidence_is_top_mass(self) -> None:
        assert distribution_confidence([0.85, 0.1, 0.05]) == pytest.approx(0.85)

    def test_entropy_bounds(self) -> None:
        # One-hot → 0; uniform → 1 (normalised by log(3)).
        assert distribution_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
        third = 1.0 / 3.0
        assert distribution_entropy([third, third, third]) == pytest.approx(1.0)

    def test_entropy_monotone(self) -> None:
        sharp = distribution_entropy([0.9, 0.05, 0.05])
        flat = distribution_entropy([0.5, 0.3, 0.2])
        assert sharp < flat

    def test_one_hot_threshold(self) -> None:
        assert distribution_is_one_hot([JUDGMENT_ONE_HOT_THRESHOLD, 0.05, 0.0]) is True
        assert distribution_is_one_hot([0.8, 0.15, 0.05]) is False


class TestEvidenceProperties:
    def test_none_distribution_yields_none_signals(self) -> None:
        ev = Evidence(objective_id="o1", source_type="web", source_ref="x")
        assert ev.judgment_distribution is None
        assert ev.judgment_confidence is None
        assert ev.judgment_entropy is None
        assert ev.judgment_one_hot is None

    def test_derived_from_distribution(self) -> None:
        ev = Evidence(
            objective_id="o1",
            source_type="web",
            source_ref="x",
            judgment_distribution=[0.85, 0.10, 0.05],
        )
        assert ev.judgment_confidence == pytest.approx(0.85)
        assert ev.judgment_one_hot is False
        assert ev.judgment_entropy is not None and ev.judgment_entropy > 0.0

    def test_metadata_round_trip(self) -> None:
        ev = Evidence(
            objective_id="o1",
            source_type="web",
            source_ref="x",
            support_judgment="supports",
            judgment_reasoning="because reasons",
            judgment_distribution=[0.7, 0.2, 0.1],
        )
        meta = ev._extra_metadata()
        assert meta["judgment_distribution"] == [0.7, 0.2, 0.1]
        restored = Evidence._from_metadata("", {**meta, "objective_id": "o1"})
        assert restored.judgment_distribution == [0.7, 0.2, 0.1]
        assert restored.judgment_confidence == pytest.approx(0.7)


class TestApplyJudgment:
    def test_apply_writes_verdict_reasoning_distribution(self) -> None:
        judgment = EvidenceJudgmentOutput(
            claim_scope_summary="topic A",
            evidence_scope_summary="topic A specifically",
            in_scope=True,
            reasoning="on-topic, leans supportive",
            belief_supports=80,
            belief_contradicts=15,
            belief_no_bearing=5,
        )
        ev = Evidence(objective_id="o1", source_type="web", source_ref="x")
        apply_judgment(ev, judgment)
        assert ev.support_judgment == "supports"
        assert ev.judgment_reasoning == "on-topic, leans supportive"
        assert ev.judgment_distribution == pytest.approx([0.8, 0.15, 0.05])
        # Stored verdict is exactly the distribution argmax (single source of truth).
        assert (
            ev.support_judgment
            == JUDGMENT_CLASSES[
                max(range(3), key=lambda i: (ev.judgment_distribution or [])[i])
            ]
        )

    def test_classes_ordering_is_canonical(self) -> None:
        assert JUDGMENT_CLASSES == ("supports", "contradicts", "no_bearing")
        assert math.isclose(sum([0.7, 0.2, 0.1]), 1.0)
