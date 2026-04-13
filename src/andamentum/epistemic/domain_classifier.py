"""Domain classifier for the epistemic system.

Classifies evidence along domain dimensions for cross-domain convergence analysis.

Classification should be performed by focused agents, not keyword heuristics.
This module retains only pure data-formatting helpers and stubs for functions
that other modules import.

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import Optional, Dict, Any

from .primitives import (
    MethodType,
    DataSourceType,
    TemporalApproach,
    CausalRole,
    DomainClassification,
)


def get_domain_label(classification: DomainClassification) -> str:
    """Generate a human-readable domain label from classification.

    Examples:
    - "experimental/primary/prospective"
    - "observational/meta/retrospective"
    """
    parts = [
        classification.method_type.value,
        classification.data_source.value,
        classification.temporal.value,
    ]
    return "/".join(parts)


def classify_evidence_domain(
    evidence_id: str,
    claim_id: str,
    evidence_text: str,
    evidence_metadata: Optional[Dict[str, Any]] = None,
) -> DomainClassification:
    """Create a default DomainClassification for evidence.

    Previously used keyword heuristics to classify evidence. Now returns
    a default classification with low confidence. Use the
    epistemic_classify_domain agent for accurate classification.

    Args:
        evidence_id: ID of the evidence being classified
        claim_id: ID of the claim this evidence supports
        evidence_text: The text content of the evidence
        evidence_metadata: Optional metadata (unused)

    Returns:
        DomainClassification with default values and low confidence
    """
    return DomainClassification(
        evidence_id=evidence_id,
        claim_id=claim_id,
        method_type=MethodType.OBSERVATIONAL,
        data_source=DataSourceType.PRIMARY,
        temporal=TemporalApproach.CROSS_SECTIONAL,
        causal_role=CausalRole.PHENOMENOLOGICAL,
        classification_confidence=0.1,
        classification_method="default",
        classification_notes="Unclassified — use agent-based classification for accurate results",
    )
