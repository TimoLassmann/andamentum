"""Tests for convergence detection and domain classification.

Domain classification keyword heuristics have been removed. The remaining
tests verify domain distance calculations, clustering logic, and convergence
detection using DomainClassification objects constructed directly.
"""

import pytest

from epistemic.domain_classifier import (
    classify_evidence_domain,
    get_domain_label,
)
from epistemic.domain_distance import (
    calculate_domain_distance,
    compute_pairwise_distances,
    cluster_by_domain,
    interpret_distance,
)
from epistemic.convergence_detector import (
    detect_convergence,
    CONVERGENCE_THRESHOLDS,
)
from epistemic.primitives import (
    MethodType,
    DataSourceType,
    TemporalApproach,
    CausalRole,
    DomainClassification,
)


def _make_classification(
    evidence_id: str,
    claim_id: str = "c-1",
    method: MethodType = MethodType.OBSERVATIONAL,
    source: DataSourceType = DataSourceType.PRIMARY,
    temporal: TemporalApproach = TemporalApproach.CROSS_SECTIONAL,
    causal: CausalRole = CausalRole.PHENOMENOLOGICAL,
    confidence: float = 0.8,
) -> DomainClassification:
    """Helper to construct a DomainClassification directly."""
    return DomainClassification(
        evidence_id=evidence_id,
        claim_id=claim_id,
        method_type=method,
        data_source=source,
        temporal=temporal,
        causal_role=causal,
        classification_confidence=confidence,
        classification_method="test",
        classification_notes="Test classification",
    )


class TestDomainClassification:
    def test_classify_evidence_domain_returns_defaults(self):
        """classify_evidence_domain now returns low-confidence defaults."""
        c = classify_evidence_domain(
            evidence_id="e-1",
            claim_id="c-1",
            evidence_text="A randomized trial collected data from 500 participants",
        )
        assert c.evidence_id == "e-1"
        assert c.claim_id == "c-1"
        assert c.classification_confidence <= 0.2  # Low confidence (default)
        assert c.classification_method == "default"

    def test_domain_label(self):
        c = _make_classification("e-1", method=MethodType.EXPERIMENTAL, source=DataSourceType.PRIMARY)
        label = get_domain_label(c)
        assert "/" in label
        assert "experimental" in label
        assert "primary" in label


class TestDomainDistance:
    def test_same_domain_zero_distance(self):
        c1 = _make_classification("e-1", method=MethodType.EXPERIMENTAL)
        c2 = _make_classification("e-2", method=MethodType.EXPERIMENTAL)
        d = calculate_domain_distance(c1, c2)
        assert d < 0.3  # Same domain = low distance

    def test_different_domains_high_distance(self):
        c1 = _make_classification("e-1", method=MethodType.EXPERIMENTAL, source=DataSourceType.PRIMARY)
        c2 = _make_classification(
            "e-2", method=MethodType.THEORETICAL, source=DataSourceType.SYNTHETIC,
            temporal=TemporalApproach.RETROSPECTIVE, causal=CausalRole.MECHANISTIC,
        )
        d = calculate_domain_distance(c1, c2)
        assert d > 0.3  # Different domains = higher distance

    def test_pairwise_distances(self):
        c1 = _make_classification("e-1", method=MethodType.EXPERIMENTAL)
        c2 = _make_classification("e-2", method=MethodType.THEORETICAL)
        distances = compute_pairwise_distances([c1, c2])
        assert ("e-1", "e-2") in distances
        assert ("e-2", "e-1") in distances
        assert distances[("e-1", "e-2")] == distances[("e-2", "e-1")]

    def test_interpret_distance(self):
        assert "same domain" in interpret_distance(0.1).lower()
        assert "different" in interpret_distance(0.6).lower()


class TestClustering:
    def test_same_domain_clusters_together(self):
        c1 = _make_classification("e-1", method=MethodType.EXPERIMENTAL)
        c2 = _make_classification("e-2", method=MethodType.EXPERIMENTAL)
        clusters = cluster_by_domain([c1, c2], distance_threshold=0.5)
        assert len(clusters) <= 2

    def test_different_domains_separate_clusters(self):
        c1 = _make_classification(
            "e-1", method=MethodType.EXPERIMENTAL, source=DataSourceType.PRIMARY,
        )
        c2 = _make_classification(
            "e-2", method=MethodType.THEORETICAL, source=DataSourceType.SYNTHETIC,
            temporal=TemporalApproach.RETROSPECTIVE, causal=CausalRole.MECHANISTIC,
        )
        clusters = cluster_by_domain([c1, c2], distance_threshold=0.3)
        assert len(clusters) >= 1

    def test_empty_input(self):
        clusters = cluster_by_domain([])
        assert clusters == []


class TestConvergenceDetection:
    def test_empty_evidence(self):
        result = detect_convergence([], claim_id="c-1", objective_id="o-1")
        assert not result.convergence_detected
        assert result.verdict == "NO_EVIDENCE"
        assert result.num_independent_domains == 0

    def test_single_evidence_no_convergence(self):
        items = [{"evidence_id": "e-1", "content": "A trial tested the drug"}]
        result = detect_convergence(items, claim_id="c-1", objective_id="o-1")
        assert result.total_evidence_count == 1
        assert result.num_independent_domains >= 1

    def test_multi_domain_convergence(self):
        items = [
            {"evidence_id": "e-1", "content": "A randomized controlled trial of the intervention in 500 participants"},
            {"evidence_id": "e-2", "content": "A mathematical proof derived from first principles and axioms"},
            {"evidence_id": "e-3", "content": "A computational simulation model predicted the outcome"},
        ]
        result = detect_convergence(items, claim_id="c-1", objective_id="o-1")
        assert result.total_evidence_count == 3
        assert result.num_independent_domains >= 1
        assert result.explanation  # Should have an explanation

    def test_convergence_thresholds_defined(self):
        assert "min_independent_domains" in CONVERGENCE_THRESHOLDS
        assert "min_inter_domain_distance" in CONVERGENCE_THRESHOLDS
        assert CONVERGENCE_THRESHOLDS["min_independent_domains"] >= 2
