"""Question-type routing configuration.

Maps each QuestionType to:
- Which verification tracks fire (primary, secondary, if_applicable, skip)
- What stage gate thresholds apply

This is pure data — no LLM calls, no framework dependencies.
The routing table is the authoritative spec for conditional verification.

Architecture: Layer 1 (framework-agnostic)
"""

from enum import Enum
from dataclasses import dataclass, field


class TrackActivation(str, Enum):
    """How a verification track is activated for a given question type."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    IF_APPLICABLE = "if_applicable"
    SKIP = "skip"


@dataclass(frozen=True)
class RoutingProfile:
    """Verification and gate configuration for one question type."""

    tracks: dict[str, TrackActivation]
    gate_thresholds: dict[str, dict[str, object]] = field(default_factory=dict)


ROUTING_TABLE: dict[str, RoutingProfile] = {
    "verificatory": RoutingProfile(
        tracks={
            "adversarial": TrackActivation.PRIMARY,
            "convergence": TrackActivation.PRIMARY,
            "deductive": TrackActivation.SECONDARY,
            "computational": TrackActivation.IF_APPLICABLE,
            "argument": TrackActivation.SECONDARY,
            "contrastive": TrackActivation.SKIP,
            "consistency": TrackActivation.SKIP,
        },
        gate_thresholds={
            "supported": {"min_evidence_weighted": 1.0, "min_adversarial_balance": 0.4},
            "provisional": {
                "min_evidence_weighted": 2.0,
                "min_quality_mean": 0.5,
                "requires_convergence": True,
            },
            "robust": {"min_evidence_weighted": 3.0, "min_independent_domains": 2},
        },
    ),
    "explanatory": RoutingProfile(
        tracks={
            "adversarial": TrackActivation.SECONDARY,
            "convergence": TrackActivation.SECONDARY,
            "deductive": TrackActivation.PRIMARY,
            "computational": TrackActivation.IF_APPLICABLE,
            "argument": TrackActivation.PRIMARY,
            "contrastive": TrackActivation.PRIMARY,
            "consistency": TrackActivation.SKIP,
        },
        gate_thresholds={
            "supported": {
                "min_evidence_weighted": 1.0,
                "requires_deductive_validation": True,
            },
            "provisional": {
                "min_evidence_weighted": 2.0,
                "requires_contrastive_superiority": True,
            },
            "robust": {
                "min_evidence_weighted": 2.0,
                "requires_deductive_validation": True,
                "requires_contrastive_superiority": True,
            },
        },
    ),
    "exploratory": RoutingProfile(
        tracks={
            "adversarial": TrackActivation.SKIP,
            "convergence": TrackActivation.SECONDARY,
            "deductive": TrackActivation.SKIP,
            "computational": TrackActivation.SKIP,
            "argument": TrackActivation.SKIP,
            "contrastive": TrackActivation.SKIP,
            "consistency": TrackActivation.PRIMARY,
        },
        gate_thresholds={
            "supported": {"min_evidence_weighted": 0.5},
            "provisional": {
                "min_evidence_weighted": 1.0,
                "requires_cross_claim_consistency": True,
            },
            "robust": {
                "min_evidence_weighted": 2.0,
                "requires_cross_claim_consistency": True,
                "min_independent_domains": 2,
            },
        },
    ),
    "comparative": RoutingProfile(
        tracks={
            "adversarial": TrackActivation.SECONDARY,
            "convergence": TrackActivation.SKIP,
            "deductive": TrackActivation.SECONDARY,
            "computational": TrackActivation.SKIP,
            "argument": TrackActivation.SKIP,
            "contrastive": TrackActivation.PRIMARY,
            "consistency": TrackActivation.PRIMARY,
        },
        gate_thresholds={
            "supported": {
                "min_evidence_weighted": 1.0,
                "requires_symmetric_scrutiny": True,
            },
            "provisional": {
                "min_evidence_weighted": 2.0,
                "requires_contrastive_evaluation": True,
            },
            "robust": {
                "min_evidence_weighted": 2.0,
                "requires_symmetric_scrutiny": True,
                "requires_contrastive_evaluation": True,
            },
        },
    ),
    "predictive": RoutingProfile(
        tracks={
            "adversarial": TrackActivation.SECONDARY,
            "convergence": TrackActivation.SKIP,
            "deductive": TrackActivation.PRIMARY,
            "computational": TrackActivation.PRIMARY,
            "argument": TrackActivation.SKIP,
            "contrastive": TrackActivation.SKIP,
            "consistency": TrackActivation.SKIP,
        },
        gate_thresholds={
            "supported": {
                "min_evidence_weighted": 1.0,
            },
            "provisional": {
                "min_evidence_weighted": 2.0,
            },
            "robust": {
                "min_evidence_weighted": 3.0,
            },
            "actionable": {
                "requires_falsification_criteria": True,
            },
        },
    ),
    "compositional": RoutingProfile(
        tracks={
            "adversarial": TrackActivation.SKIP,
            "convergence": TrackActivation.PRIMARY,
            "deductive": TrackActivation.SKIP,
            "computational": TrackActivation.SKIP,
            "argument": TrackActivation.SKIP,
            "contrastive": TrackActivation.SKIP,
            "consistency": TrackActivation.PRIMARY,
        },
        gate_thresholds={
            "supported": {"min_evidence_weighted": 1.0},
            "provisional": {
                "min_evidence_weighted": 1.5,
                "requires_cross_claim_consistency": True,
                "requires_convergence": True,
            },
            "robust": {
                "min_evidence_weighted": 2.0,
            },
        },
    ),
    "normative": RoutingProfile(
        tracks={
            "adversarial": TrackActivation.SECONDARY,
            "convergence": TrackActivation.SKIP,
            "deductive": TrackActivation.PRIMARY,
            "computational": TrackActivation.SKIP,
            "argument": TrackActivation.PRIMARY,
            "contrastive": TrackActivation.SKIP,
            "consistency": TrackActivation.PRIMARY,
        },
        gate_thresholds={
            "supported": {
                "min_evidence_weighted": 1.0,
                "requires_fact_value_separation": True,
            },
            "provisional": {
                "min_evidence_weighted": 2.0,
                "requires_fact_value_separation": True,
                "requires_deductive_validation": True,
            },
            "robust": {
                "min_evidence_weighted": 2.0,
                "requires_fact_value_separation": True,
                "evaluative_claims_flagged": True,
            },
        },
    ),
}

SECONDARY_TRIGGERS: dict[str, str] = {
    "adversarial": "conflicting_evidence_flag OR adversarial_balance < 0.6",
    "convergence": "evidence_count >= 3 AND domain_count < 2",
    "deductive": "claim_has_logical_structure AND no_deductive_validation_yet",
    "argument": "claim_has_premise_chain AND no_argument_analysis_yet",
}


def get_routing_profile(question_type: str) -> RoutingProfile:
    """Get the routing profile for a question type.

    Raises KeyError if question_type is not in the routing table.
    """
    key = str(question_type)
    if key not in ROUTING_TABLE:
        available = ", ".join(sorted(ROUTING_TABLE))
        raise KeyError(f"Unknown question type: {key}. Available: {available}")
    return ROUTING_TABLE[key]


def get_active_tracks(question_type: str) -> dict[str, TrackActivation]:
    """Get the track activation map for a question type."""
    return get_routing_profile(question_type).tracks


# Provider selection has moved to ``provider_routing.py`` (semantic similarity
# via embeddings). This module now covers only verification-track routing.
