"""Posterior confidence calculator for completed epistemic inquiries.

Computes posterior confidence by synthesizing two signals:
  1. Per-item counting (induction): logistic(supporting - contradicting)
  2. Integration assessment (abduction): verdict_to_probability(verdict, confidence)

The two signals are blended via weighted model averaging:
  w = n_directional / (n_directional + K)  where K=5
  posterior = w * counting_posterior + (1-w) * integration_posterior

Integration "insufficient" abstains from the blend — counting runs alone.
No LLM calls. No trained weights. Domain-independent by construction.

Architecture: Layer 1 (framework-agnostic, pure computation)

Usage::

    repo = await EpistemicRepository.for_database("my_run", db_dir=Path("./results"))
    posterior = await compute_posterior(repo, objective_id="...")
    if posterior:
        print(f"Posterior confidence: {posterior.posterior:.2%}")
"""

import logging
import math

from pydantic import BaseModel, Field

from .repository import EpistemicRepository

logger = logging.getLogger(__name__)


# Question types where posterior P(Y) is meaningful.
# Comparative is excluded: "Is A better than B?" has three outcomes
# (A better, B better, equivalent), not two. The posterior's binary
# P(Y) framing is misleading when the answer is "no difference."
POSTERIOR_ELIGIBLE: set[str] = {"verificatory", "predictive"}


class PosteriorReport(BaseModel):
    """Posterior probability P(Y) for a yes/no-style research objective.

    Synthesizes per-item evidence counting with integration assessment
    via weighted model averaging. Only meaningful for verificatory and
    predictive questions. For other question types compute_posterior()
    returns None.
    """

    posterior: float = Field(description="Combined P(Y) in [0.0, 1.0]")
    log_odds: int = Field(description="Effective log-odds from combined posterior")
    supporting_count: int = Field(
        description="Total independent supporting evidence across active claims"
    )
    contradicting_count: int = Field(
        description="Total independent contradicting evidence across active claims"
    )
    counting_posterior: float = Field(description="P(Y) from per-item counting alone")
    integration_verdict: str | None = Field(
        default=None,
        description="Integration assessment: 'supports', 'contradicts', 'insufficient', or None",
    )
    integration_confidence: float | None = Field(
        default=None,
        description="Integration confidence 0.0-1.0, or None if not run",
    )
    mode: str = Field(
        default="counting_only",
        description="'counting_only' or 'synthesized'",
    )
    objective_id: str
    question_type: str
    explanation: str


async def compute_posterior(
    repo: EpistemicRepository,
    objective_id: str,
) -> PosteriorReport | None:
    """Compute posterior probability P(Y) by synthesizing counting and integration.

    Per-item counting always runs (the inductive base). Integration assessment
    blends in via weighted model averaging when available. "insufficient" verdicts
    abstain from the blend.

    Args:
        repo: Repository with epistemic run data.
        objective_id: Which inquiry to assess.

    Returns:
        PosteriorReport, or None for ineligible question types.
    """
    # 1. Load objective, check eligibility
    objective = await repo.get_objective(objective_id)
    question_type = objective.question_type
    if question_type is None or question_type not in POSTERIOR_ELIGIBLE:
        return None

    # 2. Load claims and evidence
    claims = await repo.get_claims_for_objective(objective_id)
    evidence = await repo.get_evidence_for_objective(objective_id)
    active_claims = [c for c in claims if not c.abandoned]

    # 3. ALWAYS compute per-item counts (the inductive base)
    supporting = 0
    contradicting = 0
    for claim in active_claims:
        claim_evidence = [
            e
            for e in evidence
            if e.entity_id in claim.evidence_ids
            and not e.invalidated
            and e.cluster_status not in ("corroborative", "deferred")
        ]
        for e in claim_evidence:
            if e.support_judgment == "supports":
                supporting += 1
            elif e.support_judgment == "contradicts":
                contradicting += 1

    # 4. Compute counting posterior
    counting_log_odds = supporting - contradicting
    if abs(counting_log_odds) < 700:
        counting_posterior = 1.0 / (1.0 + math.exp(-counting_log_odds))
    else:
        counting_posterior = 1.0 if counting_log_odds > 0 else 0.0

    # 5. Compute integration signal (when available)
    integrated_claims = [
        c for c in active_claims if c.integrated_assessment is not None
    ]

    # Collect directional integration verdicts (skip "insufficient")
    integration_verdict = None
    integration_confidence = None
    integration_posterior = None

    directional = [
        c
        for c in integrated_claims
        if c.integrated_assessment in ("supports", "contradicts")
    ]

    if directional:
        # For multiple claims with integration, aggregate: each claim is one vote
        int_supporting = sum(
            1 for c in directional if c.integrated_assessment == "supports"
        )
        int_contradicting = sum(
            1 for c in directional if c.integrated_assessment == "contradicts"
        )
        # Average confidence across directional claims
        avg_confidence = sum(
            (c.integrated_confidence or 0.5) for c in directional
        ) / len(directional)
        avg_confidence = max(0.0, min(1.0, avg_confidence))

        # Net direction
        if int_supporting > int_contradicting:
            integration_verdict = "supports"
            integration_confidence = avg_confidence
            integration_posterior = 0.5 + avg_confidence / 2
        elif int_contradicting > int_supporting:
            integration_verdict = "contradicts"
            integration_confidence = avg_confidence
            integration_posterior = 0.5 - avg_confidence / 2
        else:
            # Equal directional votes — treat as insufficient
            integration_verdict = "insufficient"
            integration_confidence = avg_confidence
            integration_posterior = None
    elif integrated_claims:
        # All claims had "insufficient" — record but don't blend
        integration_verdict = "insufficient"
        integration_confidence = sum(
            (c.integrated_confidence or 0.5) for c in integrated_claims
        ) / len(integrated_claims)
        integration_posterior = None

    # 6. Blend: weighted model averaging
    n_directional = supporting + contradicting
    MIXING_K = 5

    if integration_posterior is not None:
        w = n_directional / (n_directional + MIXING_K)
        posterior = w * counting_posterior + (1.0 - w) * integration_posterior
        mode = "synthesized"
    else:
        posterior = counting_posterior
        mode = "counting_only"

    # 7. Compute effective log-odds from blended posterior (for report)
    if posterior <= 0.0:
        log_odds = -700
    elif posterior >= 1.0:
        log_odds = 700
    else:
        log_odds = round(math.log(posterior / (1.0 - posterior)))

    # 8. Build explanation
    parts = []
    parts.append(f"Posterior {posterior:.4f} for {question_type} question.")
    parts.append(
        f"Per-item counting: {supporting} supporting vs {contradicting} contradicting "
        f"(counting posterior {counting_posterior:.4f})."
    )
    if integration_verdict is not None:
        if integration_posterior is not None:
            parts.append(
                f"Integration: {integration_verdict} at confidence "
                f"{integration_confidence:.2f} (integration posterior {integration_posterior:.4f}). "
                f"Blended with counting weight {n_directional / (n_directional + MIXING_K):.2f}."
            )
        else:
            parts.append(
                f"Integration: {integration_verdict} — abstained from blend."
            )

    # Flag disagreement between counting and integration (Doyle TMS)
    if integration_posterior is not None:
        counting_direction = (
            "supports"
            if counting_posterior > 0.5
            else ("contradicts" if counting_posterior < 0.5 else "neutral")
        )
        if integration_verdict != counting_direction and counting_direction != "neutral":
            parts.append(
                f"NOTE: Counting ({counting_direction}) and integration "
                f"({integration_verdict}) disagree."
            )

    explanation = " ".join(parts)

    return PosteriorReport(
        posterior=round(posterior, 6),
        log_odds=log_odds,
        supporting_count=supporting,
        contradicting_count=contradicting,
        counting_posterior=round(counting_posterior, 6),
        integration_verdict=integration_verdict,
        integration_confidence=(
            round(integration_confidence, 4)
            if integration_confidence is not None
            else None
        ),
        mode=mode,
        objective_id=objective_id,
        question_type=question_type,
        explanation=explanation,
    )
