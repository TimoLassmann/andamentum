"""Adversarial balance calculator for the epistemic system.

Calculates adversarial balance scores and determines verdicts
based on the balance between supporting and adversarial evidence.

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import List, Tuple, Optional
import uuid

from .primitives import (
    Counterargument,
    AdversarialEvidence,
    CriticismCategory,
)


def calculate_adversarial_balance(supporting_weight: float, adversarial_weight: float) -> float:
    """Calculate adversarial balance score.

    From spec Part 6.3:
    balance = supporting / (supporting + adversarial)

    Args:
        supporting_weight: Sum of weights for supporting evidence.
        adversarial_weight: Sum of weights for adversarial evidence.

    Returns:
        Balance score from 0.0 to 1.0.

    Interpretation:
    - > 0.8: Strongly supported (but check for confirmation bias)
    - 0.6-0.8: Moderately supported
    - 0.4-0.6: Contested
    - 0.2-0.4: Weakly supported / likely false
    - < 0.2: Strongly challenged
    """
    total = supporting_weight + adversarial_weight
    if total == 0:
        return 0.5  # No evidence either way - neutral
    return supporting_weight / total


def interpret_balance(balance: float) -> str:
    """Interpret an adversarial balance score.

    Args:
        balance: Balance score from 0.0 to 1.0.

    Returns:
        Human-readable interpretation.
    """
    if balance > 0.95:
        return "Suspiciously uncontested - may indicate insufficient adversarial search"
    elif balance > 0.8:
        return "Strongly supported - no significant counterarguments found"
    elif balance > 0.6:
        return "Moderately supported - some valid criticism but outweighed by support"
    elif balance > 0.4:
        return "Contested - significant criticism exists, claim uncertain"
    elif balance > 0.2:
        return "Weakly supported - adversarial evidence outweighs support"
    else:
        return "Strongly challenged - substantial counterarguments undermine claim"


def determine_verdict(balance: float, counterarguments: List[Counterargument]) -> Tuple[str, str, float]:
    """Determine verdict and recommendation from adversarial results.

    Args:
        balance: Adversarial balance score.
        counterarguments: List of discovered counterarguments.

    Returns:
        Tuple of (verdict, recommendation, confidence).

    Verdicts:
    - SUPPORTED: Claim survived adversarial search
    - CONTESTED: Significant criticism exists
    - CHALLENGED: Strong counterarguments undermine claim
    - REFUTED: Overwhelming adversarial evidence
    """
    # Count strong counterarguments
    strong_counters = sum(1 for c in counterarguments if c.match_strength == "strong" and c.quality.passes_threshold)

    # Check for replication failures (highest impact)
    replication_failures = sum(
        1 for c in counterarguments if c.category == CriticismCategory.REPLICATION_FAILURE and c.quality.passes_threshold
    )

    # Determine verdict
    if replication_failures > 0:
        # Replication failure is very serious
        if replication_failures >= 2:
            return "REFUTED", "refute", 0.85
        else:
            return "CHALLENGED", "weaken", 0.75

    if balance > 0.8:
        if strong_counters == 0:
            return "SUPPORTED", "maintain", 0.9 - (1.0 - balance) * 0.5
        else:
            # Strong counters exist but balance is good
            return "SUPPORTED", "maintain", 0.7

    elif balance > 0.6:
        if strong_counters >= 2:
            return "CONTESTED", "modify", 0.6
        else:
            return "SUPPORTED", "maintain", 0.65

    elif balance > 0.4:
        return "CONTESTED", "modify", 0.5

    elif balance > 0.2:
        return "CHALLENGED", "weaken", 0.6

    else:
        return "REFUTED", "refute", 0.75


def calculate_total_adversarial_weight(counterarguments: List[Counterargument]) -> float:
    """Calculate total weight of adversarial evidence.

    Only counts counterarguments that pass quality threshold.

    Args:
        counterarguments: List of counterarguments.

    Returns:
        Sum of weights for valid counterarguments.
    """
    return sum(c.weight for c in counterarguments if c.quality.passes_threshold and c.match_strength != "none")


def should_flag_for_review(balance: float, counterarguments: List[Counterargument]) -> Tuple[bool, Optional[str]]:
    """Determine if results should be flagged for human review.

    From spec Part 6.3: Flag suspicious patterns.

    Args:
        balance: Adversarial balance score.
        counterarguments: List of counterarguments.

    Returns:
        Tuple of (should_flag, reason).
    """
    # Suspiciously uncontested
    if balance > 0.95 and len(counterarguments) == 0:
        return True, "No counterarguments found - may need broader search"

    # Strong challenge but not demoted
    if balance < 0.3:
        return True, "Strong adversarial evidence - claim may need demotion"

    # Mixed quality counterarguments
    low_quality = sum(1 for c in counterarguments if not c.quality.passes_threshold)
    if low_quality > len(counterarguments) / 2:
        return True, "Many low-quality counterarguments - may need better sources"

    # Ad hominem attacks present (filtered but should note)
    ad_hominems = sum(1 for c in counterarguments if c.category == CriticismCategory.AD_HOMINEM)
    if ad_hominems > 0:
        return True, f"{ad_hominems} ad hominem attacks filtered - topic may be politically charged"

    return False, None


def generate_explanation(
    balance: float, counterarguments: List[Counterargument], verdict: str, recommendation: str
) -> str:
    """Generate human-readable explanation of adversarial search results.

    Args:
        balance: Adversarial balance score.
        counterarguments: List of counterarguments.
        verdict: The determined verdict.
        recommendation: The recommended action.

    Returns:
        Explanation text.
    """
    valid_counters = [c for c in counterarguments if c.quality.passes_threshold and c.match_strength != "none"]
    strong_counters = [c for c in valid_counters if c.match_strength == "strong"]

    parts = []

    # Overall assessment
    parts.append(f"Adversarial balance score: {balance:.2f}")
    parts.append(interpret_balance(balance))

    # Counterargument summary
    if not counterarguments:
        parts.append("No counterarguments discovered during adversarial search.")
    else:
        parts.append(f"Found {len(counterarguments)} potential counterarguments, {len(valid_counters)} meet quality threshold.")

        if strong_counters:
            parts.append(f"Strong challenges: {len(strong_counters)}")
            # Summarize strongest counterarguments
            for c in strong_counters[:3]:  # Top 3
                parts.append(f"  - [{c.category.value}] {c.summary[:100]}...")

    # Recommendation
    if recommendation == "maintain":
        parts.append("Recommendation: Maintain claim with current confidence.")
    elif recommendation == "modify":
        parts.append("Recommendation: Modify claim to address valid criticisms.")
    elif recommendation == "weaken":
        parts.append("Recommendation: Significantly reduce confidence in claim.")
    elif recommendation == "refute":
        parts.append("Recommendation: Consider demoting or refuting claim.")

    return " ".join(parts)


def synthesize_adversarial_result(
    claim_id: str,
    objective_id: str,
    queries_used: List[str],
    counterarguments: List[Counterargument],
    supporting_evidence_weight: float,
) -> AdversarialEvidence:
    """Create complete adversarial evidence record.

    Args:
        claim_id: ID of the claim being tested.
        objective_id: ID of the parent objective.
        queries_used: List of adversarial queries executed.
        counterarguments: List of discovered counterarguments.
        supporting_evidence_weight: Weight of existing supporting evidence.

    Returns:
        Complete AdversarialEvidence record.
    """
    # Calculate adversarial weight
    adversarial_weight = calculate_total_adversarial_weight(counterarguments)

    # Calculate balance
    balance = calculate_adversarial_balance(supporting_evidence_weight, adversarial_weight)

    # Determine verdict
    verdict, recommendation, confidence = determine_verdict(balance, counterarguments)

    # Generate explanation
    explanation = generate_explanation(balance, counterarguments, verdict, recommendation)

    return AdversarialEvidence(
        evidence_id=str(uuid.uuid4()),
        claim_id=claim_id,
        objective_id=objective_id,
        queries_used=queries_used,
        sources_searched=len(counterarguments),
        counterarguments=counterarguments,
        supporting_weight=supporting_evidence_weight,
        adversarial_weight=adversarial_weight,
        adversarial_balance=balance,
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        recommendation=recommendation,
        suggested_modifications=None,
    )
