"""Tests for detect_convergence's integration with LLM agent inputs."""

from __future__ import annotations

import pytest

from ..convergence_detector import detect_convergence
from ..primitives import (
    DomainClassification,
    MethodType,
    DataSourceType,
    TemporalApproach,
    CausalRole,
)


def _mk_classification(
    eid: str, method: MethodType, source: DataSourceType
) -> DomainClassification:
    return DomainClassification(
        evidence_id=eid,
        claim_id="claim-1",
        method_type=method,
        data_source=source,
        temporal=TemporalApproach.CROSS_SECTIONAL,
        causal_role=CausalRole.PHENOMENOLOGICAL,
        classification_confidence=0.9,
        classification_method="agent",
        classification_notes="test classification",
    )


def test_detect_convergence_uses_precomputed_classifications():
    """When precomputed_classifications is provided, internal classification is skipped
    and the agent's classifications drive the convergence calculation."""
    items = [
        {"evidence_id": "ev1", "content": "experimental study results"},
        {"evidence_id": "ev2", "content": "observational cohort data"},
    ]
    # Force two distinct domain types to ensure clustering separates them
    classifications = [
        _mk_classification("ev1", MethodType.EXPERIMENTAL, DataSourceType.PRIMARY),
        _mk_classification("ev2", MethodType.OBSERVATIONAL, DataSourceType.SECONDARY),
    ]
    result = detect_convergence(
        evidence_items=items,
        claim_id="claim-1",
        objective_id="obj-1",
        precomputed_classifications=classifications,
    )
    # The result should reflect the agent's classifications: ev1 EXPERIMENTAL, ev2 OBSERVATIONAL
    assert len(result.evidence_classifications) == 2
    assert result.evidence_classifications[0].method_type == MethodType.EXPERIMENTAL
    assert result.evidence_classifications[1].method_type == MethodType.OBSERVATIONAL
    # All classifications should carry the "agent" method label (not the heuristic default)
    assert all(
        c.classification_method == "agent" for c in result.evidence_classifications
    )


def test_detect_convergence_validates_classification_count():
    """precomputed_classifications must have one entry per evidence item."""
    items = [
        {"evidence_id": "ev1", "content": "x"},
        {"evidence_id": "ev2", "content": "y"},
    ]
    classifications = [
        _mk_classification("ev1", MethodType.EXPERIMENTAL, DataSourceType.PRIMARY)
    ]
    with pytest.raises(ValueError, match="length"):
        detect_convergence(
            evidence_items=items,
            claim_id="claim-1",
            objective_id="obj-1",
            precomputed_classifications=classifications,
        )


def test_detect_convergence_pairwise_independence_boosts_score():
    """When the LLM judges within-cluster items as independent, independence_score
    is higher than without that signal."""
    # Two items in the same domain cluster (same method, same source)
    items = [
        {"evidence_id": "ev1", "content": "study from lab A"},
        {"evidence_id": "ev2", "content": "study from lab B"},
        {"evidence_id": "ev3", "content": "different domain study"},
    ]
    classifications = [
        _mk_classification("ev1", MethodType.EXPERIMENTAL, DataSourceType.PRIMARY),
        _mk_classification("ev2", MethodType.EXPERIMENTAL, DataSourceType.PRIMARY),
        _mk_classification("ev3", MethodType.OBSERVATIONAL, DataSourceType.SECONDARY),
    ]
    # Agent says ev1 and ev2 (in the same cluster) are actually independent
    pairwise = {("ev1", "ev2"): True}

    with_pairwise = detect_convergence(
        evidence_items=items,
        claim_id="claim-1",
        objective_id="obj-1",
        precomputed_classifications=classifications,
        pairwise_independence=pairwise,
    )
    without_pairwise = detect_convergence(
        evidence_items=items,
        claim_id="claim-1",
        objective_id="obj-1",
        precomputed_classifications=classifications,
    )
    assert with_pairwise.independence_score > without_pairwise.independence_score
    assert with_pairwise.independence_checks.get("intra_cluster_diversity") is True
    assert without_pairwise.independence_checks.get("intra_cluster_diversity") is False
