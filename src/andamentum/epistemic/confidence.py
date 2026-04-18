"""Posterior confidence calculator for completed epistemic inquiries.

Computes posterior confidence from evidence direction:
logistic(supporting - contradicting). Measures how strongly the evidence
supports the established claims. Higher = more claims supported, fewer
contradicted.

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

    Aggregates evidence direction across all active claims:
      log_odds = supporting_count - contradicting_count
      posterior = 1 / (1 + exp(-log_odds))

    Only meaningful for verificatory and predictive questions.
    For other question types compute_posterior() returns None.
    """

    posterior: float = Field(description="P(Y) in [0.0, 1.0]")
    log_odds: int = Field(description="supporting - contradicting")
    supporting_count: int = Field(
        description="Total independent supporting evidence across active claims"
    )
    contradicting_count: int = Field(
        description="Total independent contradicting evidence across active claims"
    )
    objective_id: str
    question_type: str
    explanation: str


async def compute_posterior(
    repo: EpistemicRepository,
    objective_id: str,
) -> PosteriorReport | None:
    """Compute posterior probability P(Y) from evidence direction.

    Aggregates evidence across all active (non-abandoned) claims for the
    objective.  Only representative, non-invalidated evidence with a
    directional judgment (supports / contradicts) counts.

    Returns None for question types where a yes/no posterior is not
    meaningful (explanatory, exploratory, etc.).

    Args:
        repo: Repository with epistemic run data.
        objective_id: Which inquiry to assess.

    Returns:
        PosteriorReport, or None for ineligible question types.
    """
    # 1. Load objective to get question_type
    objective = await repo.get_objective(objective_id)
    question_type = objective.question_type

    # 2. Check eligibility
    if question_type is None or question_type not in POSTERIOR_ELIGIBLE:
        return None

    # 3. Load claims and evidence
    claims = await repo.get_claims_for_objective(objective_id)
    evidence = await repo.get_evidence_for_objective(objective_id)

    # 4. Filter to active claims
    active_claims = [c for c in claims if not c.abandoned]

    # 5. Count supporting / contradicting evidence across all active claims
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
            # "no_bearing" or None → ignored

    # 6. Compute log-odds and posterior
    log_odds = supporting - contradicting

    if abs(log_odds) < 700:
        posterior = 1.0 / (1.0 + math.exp(-log_odds))
    else:
        posterior = 1.0 if log_odds > 0 else 0.0

    # 7. Build explanation
    total = supporting + contradicting
    if total == 0:
        explanation = (
            f"Posterior {posterior:.2f} for {question_type} question. "
            f"No directional evidence found (uninformative prior)."
        )
    else:
        explanation = (
            f"Posterior {posterior:.2f} for {question_type} question. "
            f"{supporting} supporting vs {contradicting} contradicting "
            f"evidence (log-odds {log_odds:+d})."
        )

    return PosteriorReport(
        posterior=round(posterior, 6),
        log_odds=log_odds,
        supporting_count=supporting,
        contradicting_count=contradicting,
        objective_id=objective_id,
        question_type=question_type,
        explanation=explanation,
    )
