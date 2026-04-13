"""Tests for adversarial search modules."""

import pytest

from ..adversarial_query_generator import (
    generate_adversarial_queries,
    detect_domain,
)
from ..adversarial_evaluator import (
    create_counterargument,
    is_valid_criticism,
    get_category_weight,
)
from ..adversarial_balance import (
    calculate_adversarial_balance,
    determine_verdict,
    interpret_balance,
)
from ..primitives import CriticismCategory, Counterargument, CounterargumentQuality


class TestAdversarialQueryGeneration:
    def test_generates_queries(self):
        queries = generate_adversarial_queries("Caffeine improves alertness")
        assert len(queries) > 0
        assert all(isinstance(q, str) for q in queries)

    def test_max_queries_respected(self):
        queries = generate_adversarial_queries("Test claim", max_queries=3)
        assert len(queries) <= 3

    def test_detect_domain_returns_none(self):
        """detect_domain is a stub that always returns None (agent should classify)."""
        assert detect_domain("This drug treatment shows clinical efficacy") is None
        assert (
            detect_domain("The neural network algorithm achieves high accuracy") is None
        )
        assert detect_domain("Something very generic") is None

    def test_domain_specific_queries_included(self):
        queries = generate_adversarial_queries(
            "This drug treatment is effective", claim_domain="biomedical"
        )
        # Should include biomedical-specific queries
        assert any(
            "clinical" in q.lower()
            or "side effects" in q.lower()
            or "retracted" in q.lower()
            for q in queries
        )


class TestCreateCounterargument:
    def test_creates_with_defaults(self):
        ca = create_counterargument(
            summary="The method is flawed",
            source_ref="https://example.com",
            claim_id="c-1",
        )
        assert ca.claim_id == "c-1"
        assert ca.summary == "The method is flawed"
        assert ca.category == CriticismCategory.INTERPRETATION  # default
        assert ca.weight >= 0

    def test_creates_with_pre_evaluated_scores(self):
        quality = CounterargumentQuality(
            relevance=0.9,
            specificity=0.8,
            evidence_backed=0.7,
            source_credibility=0.6,
            novelty=0.5,
        )
        ca = create_counterargument(
            summary="Study failed to replicate",
            source_ref="https://example.com/paper",
            claim_id="c-2",
            category=CriticismCategory.REPLICATION_FAILURE,
            quality=quality,
            match_strength="strong",
        )
        assert ca.category == CriticismCategory.REPLICATION_FAILURE
        assert ca.quality.relevance == 0.9
        assert ca.match_strength == "strong"
        assert ca.weight > 0

    def test_get_category_weight(self):
        w = get_category_weight(CriticismCategory.REPLICATION_FAILURE)
        assert isinstance(w, float)
        assert w > 0


class TestIsValidCriticism:
    def test_ad_hominem_invalid(self):
        ca = Counterargument(
            claim_id="c-1",
            summary="The researcher is a fraud",
            source_ref="https://example.com",
            source_author=None,
            category=CriticismCategory.AD_HOMINEM,
            quality=CounterargumentQuality(
                relevance=0.5,
                specificity=0.5,
                evidence_backed=0.5,
                source_credibility=0.5,
                novelty=0.5,
            ),
            match_strength="strong",
        )
        valid, reason = is_valid_criticism(ca)
        assert not valid
        assert "ad hominem" in reason.lower()


class TestAdversarialBalance:
    def test_balanced_evidence(self):
        balance = calculate_adversarial_balance(5.0, 5.0)
        assert balance == pytest.approx(0.5)

    def test_strongly_supported(self):
        balance = calculate_adversarial_balance(9.0, 1.0)
        assert balance > 0.8

    def test_strongly_challenged(self):
        balance = calculate_adversarial_balance(1.0, 9.0)
        assert balance < 0.2

    def test_no_evidence_is_neutral(self):
        balance = calculate_adversarial_balance(0.0, 0.0)
        assert balance == 0.5

    def test_interpret_balance_strings(self):
        assert "supported" in interpret_balance(0.85).lower()
        assert "contested" in interpret_balance(0.5).lower()
        assert "challenged" in interpret_balance(0.15).lower()

    def test_determine_verdict_supported(self):
        verdict, recommendation, conf = determine_verdict(0.9, [])
        assert verdict == "SUPPORTED"
        assert recommendation == "maintain"

    def test_determine_verdict_low_balance(self):
        verdict, recommendation, conf = determine_verdict(0.1, [])
        assert verdict == "REFUTED"
        assert recommendation == "refute"
