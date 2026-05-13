"""Adversarial balance calculator for the epistemic system.

Calculates adversarial balance scores and determines verdicts
based on the balance between supporting and adversarial evidence.

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import List, Tuple
import uuid

from .primitives import (
    Counterargument,
    AdversarialEvidence,
    CriticismCategory,
)
from .thresholds import (
    ADVERSARIAL_REFUTED_THRESHOLD,
    ADVERSARIAL_SURVIVED_THRESHOLD,
    ADVERSARIAL_SUSPICIOUS_THRESHOLD,
)


def calculate_adversarial_balance(
    supporting_weight: float, adversarial_weight: float
) -> float:
    """Calculate adversarial balance score.

    From spec Part 6.3:
    balance = supporting / (supporting + adversarial)

    Args:
        supporting_weight: Sum of weights for supporting evidence.
        adversarial_weight: Sum of weights for adversarial evidence.

    Returns:
        Balance score from 0.0 to 1.0.

    Interpretation (Popper-Lakatos three-band; canonical breakpoints
    in epistemic.thresholds):

    - > ADVERSARIAL_SUSPICIOUS_THRESHOLD (0.95): suspiciously
      uncontested — check for insufficient adversarial search
    - >= ADVERSARIAL_SURVIVED_THRESHOLD (0.7): survived adversarial
      challenge (Popperian corroboration)
    - >= ADVERSARIAL_REFUTED_THRESHOLD (0.3): contested (Lakatos:
      counterevidence exists but isn't decisive)
    - < ADVERSARIAL_REFUTED_THRESHOLD (0.3): refuted (Popper:
      adversarial evidence dominates)
    """
    total = supporting_weight + adversarial_weight
    if total == 0:
        return 0.5  # No evidence either way - neutral
    return supporting_weight / total


def interpret_balance(balance: float) -> str:
    """Interpret an adversarial balance score using the canonical
    Popper-Lakatos three-band system.

    The function is a *narrative renderer*: its strings are consumed
    by report writers and CLI panels for human-readable display, not
    by graph routing or gate decisions (those branch on the balance
    value directly via ``epistemic.thresholds`` constants). Bands
    here are derived from the same canonical thresholds so the
    narrative stays consistent with the gating logic.

    Args:
        balance: Balance score from 0.0 to 1.0.

    Returns:
        Human-readable interpretation.
    """
    if balance > ADVERSARIAL_SUSPICIOUS_THRESHOLD:
        return "Suspiciously uncontested — may indicate insufficient adversarial search"
    if balance >= ADVERSARIAL_SURVIVED_THRESHOLD:
        return "Survived adversarial challenge — no decisive counterevidence found"
    if balance < ADVERSARIAL_REFUTED_THRESHOLD:
        return "Refuted — adversarial evidence dominates"
    return "Contested — significant counterevidence exists; claim uncertain"


def determine_verdict(
    balance: float, counterarguments: List[Counterargument]
) -> Tuple[str, str, float]:
    """Determine verdict and recommendation from adversarial results.

    Args:
        balance: Adversarial balance score.
        counterarguments: List of discovered counterarguments.

    Returns:
        Tuple of (verdict, recommendation, confidence).

    Three-band Popper-Lakatos verdict, derived from the canonical
    breakpoints in ``epistemic.thresholds``. The verdict string is
    consumed by reporters and CLI panels (display only — graph
    routing branches on the balance value directly via the gate
    thresholds, not on the verdict string).

    Verdicts (canonical three-band):
    - SUPPORTED: balance >= ADVERSARIAL_SURVIVED_THRESHOLD —
      claim survived adversarial challenge (Popperian corroboration).
    - CONTESTED: REFUTED <= balance < SURVIVED — claim has
      counterevidence but isn't decisively refuted (Lakatos middle).
    - REFUTED: balance < ADVERSARIAL_REFUTED_THRESHOLD — adversarial
      evidence dominates (Popper falsification). Also returned when
      ≥2 quality replication failures are present, regardless of
      balance — replication failure is a structural refutation that
      a balance score may not fully reflect.
    """
    # Count strong counterarguments
    strong_counters = sum(
        1
        for c in counterarguments
        if c.match_strength == "strong" and c.quality.passes_threshold
    )

    # Check for replication failures (highest impact)
    replication_failures = sum(
        1
        for c in counterarguments
        if c.category == CriticismCategory.REPLICATION_FAILURE
        and c.quality.passes_threshold
    )

    # Replication failures are decisive regardless of balance.
    if replication_failures >= 2:
        return "REFUTED", "refute", 0.85
    if replication_failures > 0:
        return "CONTESTED", "weaken", 0.75

    if balance >= ADVERSARIAL_SURVIVED_THRESHOLD:
        # Strong counters temper survival confidence (Lakatos: the
        # research programme survives but isn't unchallenged).
        if strong_counters == 0:
            return "SUPPORTED", "maintain", 0.9 - (1.0 - balance) * 0.5
        return "SUPPORTED", "maintain", 0.7

    if balance < ADVERSARIAL_REFUTED_THRESHOLD:
        return "REFUTED", "refute", 0.75

    # Contested middle band (REFUTED ≤ balance < SURVIVED).
    if strong_counters >= 2:
        return "CONTESTED", "modify", 0.6
    return "CONTESTED", "modify", 0.5


def calculate_total_adversarial_weight(
    counterarguments: List[Counterargument],
) -> float:
    """Calculate total weight of adversarial evidence.

    Only counts counterarguments that pass quality threshold.

    Args:
        counterarguments: List of counterarguments.

    Returns:
        Sum of weights for valid counterarguments.
    """
    return sum(
        c.weight
        for c in counterarguments
        if c.quality.passes_threshold and c.match_strength != "none"
    )


def generate_explanation(
    balance: float,
    counterarguments: List[Counterargument],
    verdict: str,
    recommendation: str,
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
    valid_counters = [
        c
        for c in counterarguments
        if c.quality.passes_threshold and c.match_strength != "none"
    ]
    strong_counters = [c for c in valid_counters if c.match_strength == "strong"]

    parts = []

    # Overall assessment
    parts.append(f"Adversarial balance score: {balance:.2f}")
    parts.append(interpret_balance(balance))

    # Counterargument summary
    if not counterarguments:
        parts.append("No counterarguments discovered during adversarial search.")
    else:
        parts.append(
            f"Found {len(counterarguments)} potential counterarguments, {len(valid_counters)} meet quality threshold."
        )

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
    balance = calculate_adversarial_balance(
        supporting_evidence_weight, adversarial_weight
    )

    # Determine verdict
    verdict, recommendation, confidence = determine_verdict(balance, counterarguments)

    # Generate explanation
    explanation = generate_explanation(
        balance, counterarguments, verdict, recommendation
    )

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
