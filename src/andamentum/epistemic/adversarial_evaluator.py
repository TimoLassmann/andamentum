"""Adversarial evaluator for the epistemic system.

Provides factory functions for creating Counterargument objects and
looking up category weights. Quality assessment (relevance, specificity,
evidence-backing, source credibility, novelty) is performed by focused
agents — not keyword heuristics.

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import Optional

from .primitives import (
    CriticismCategory,
    CounterargumentQuality,
    Counterargument,
    CRITICISM_CATEGORY_WEIGHTS,
)


def get_category_weight(category: CriticismCategory) -> float:
    """Get the evidence weight for a criticism category.

    Args:
        category: The criticism category.

    Returns:
        Weight multiplier for this category.
    """
    return CRITICISM_CATEGORY_WEIGHTS.get(category, 1.0)


def create_counterargument(
    summary: str,
    source_ref: str,
    claim_id: str,
    category: CriticismCategory = CriticismCategory.INTERPRETATION,
    quality: Optional[CounterargumentQuality] = None,
    match_strength: str = "partial",
    source_author: Optional[str] = None,
    supporting_evidence: str = "",
) -> Counterargument:
    """Create a Counterargument from pre-evaluated scores.

    All quality assessment (relevance, specificity, evidence-backing,
    source credibility, novelty) and classification should be performed
    by agents before calling this function. This is a thin factory that
    constructs the Counterargument and computes its weight.

    Args:
        summary: Brief summary of the criticism.
        source_ref: URL or DOI of the source.
        claim_id: ID of the claim being challenged.
        category: Pre-classified criticism category (from agent).
        quality: Pre-evaluated quality scores (from agent).
        match_strength: Pre-determined match strength: 'strong', 'partial', 'weak', or 'none'.
        source_author: Optional author of the criticism.
        supporting_evidence: Evidence supporting the criticism.

    Returns:
        A Counterargument object with computed weight.
    """
    if quality is None:
        quality = CounterargumentQuality(
            relevance=0.5,
            specificity=0.5,
            evidence_backed=0.5,
            source_credibility=0.5,
            novelty=0.5,
        )

    counterargument = Counterargument(
        claim_id=claim_id,
        summary=summary,
        source_ref=source_ref,
        source_author=source_author,
        supporting_evidence=supporting_evidence,
        category=category,
        quality=quality,
        match_strength=match_strength,
    )

    counterargument.weight = counterargument.compute_weight()
    return counterargument


def is_valid_criticism(counterargument: Counterargument) -> tuple[bool, str]:
    """Check if a counterargument is valid (not ad hominem, meets threshold).

    Args:
        counterargument: The counterargument to validate.

    Returns:
        Tuple of (is_valid, reason).
    """
    # Ad hominem attacks are never valid
    if counterargument.category == CriticismCategory.AD_HOMINEM:
        return False, "Ad hominem attacks on researchers are not valid criticism"

    # Must pass quality threshold
    if not counterargument.quality.passes_threshold:
        return False, f"Quality score {counterargument.quality.combined_score:.2f} below threshold 2.5"

    # Must have some match to the claim
    if counterargument.match_strength == "none":
        return False, "Counterargument does not match the claim"

    return True, "Valid criticism"
