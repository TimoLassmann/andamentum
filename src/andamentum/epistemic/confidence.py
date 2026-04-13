"""Post-hoc confidence calculator for completed epistemic inquiries.

Reads a completed epistemic database and computes two scores:

1. **Answer confidence** (process completion): checklist of pass/fail checks,
   logistic(passes - failures). Measures how thoroughly the inquiry was conducted.

2. **Posterior P(Y)** (evidential direction): for yes/no-style questions,
   logistic(supporting - contradicting). Measures evidence direction.

No LLM calls. No trained weights. Domain-independent by construction.

Architecture: Layer 1 (framework-agnostic, pure computation)

Usage::

    repo = await EpistemicRepository.for_database("my_run", db_dir=Path("./results"))
    report = await compute_answer_confidence(repo, objective_id="...")
    print(f"Confidence: {report.confidence:.2f} ({report.level})")

    posterior = await compute_posterior(repo, objective_id="...")
    if posterior:
        print(f"P(Y): {posterior.posterior:.2f}")
"""

import logging
import math

from pydantic import BaseModel, Field

from .entities import Claim
from .repository import EpistemicRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer-level confidence (checklist model)
# ---------------------------------------------------------------------------

# Track name -> Claim boolean flag name
TRACK_FLAGS: dict[str, str] = {
    "adversarial": "adversarial_checked",
    "convergence": "convergence_checked",
    "deductive": "deductive_checked",
    "computational": "computational_checked",
    "contrastive": "contrastive_checked",
    "consistency": "consistency_checked",
    # "argument" has no checked flag — intentionally absent
}

# Track name -> epistemological tradition label
TRACK_TRADITIONS: dict[str, str] = {
    "adversarial": "popper",
    "convergence": "wimsatt",
    "deductive": "hempel",
    "computational": "popper",
    "contrastive": "lipton",
    "consistency": "kahneman",
}

# Question types where posterior P(Y) is meaningful.
# Comparative is excluded: "Is A better than B?" has three outcomes
# (A better, B better, equivalent), not two. The posterior's binary
# P(Y) framing is misleading when the answer is "no difference."
POSTERIOR_ELIGIBLE: set[str] = {"verificatory", "predictive"}


class CheckResult(BaseModel):
    """Result of a single answer-confidence check."""

    name: str = Field(
        description="Check identifier, e.g. 'evidence_basis' or 'track:adversarial'"
    )
    tradition: str = Field(
        description="Epistemological tradition, e.g. 'kahneman', 'peirce', ''"
    )
    passed: bool = Field(description="Whether the check passed")
    detail: str = Field(description="Human-readable explanation")


class AnswerConfidenceReport(BaseModel):
    """Checklist-style confidence assessment of a completed epistemic inquiry.

    Each check contributes +1 (pass) or -1 (fail) to log-odds.
    The logistic function converts to a probability.
    """

    objective_id: str
    question_type: str | None = Field(
        description="Question type from objective, or None if unclassified"
    )
    checks: list[CheckResult] = Field(description="All checks that were evaluated")
    passes: int = Field(description="Number of checks that passed")
    failures: int = Field(description="Number of checks that failed")
    log_odds: int = Field(description="passes - failures")
    confidence: float = Field(description="1 / (1 + exp(-log_odds))")
    level: str = Field(description="'high', 'moderate', 'low', or 'insufficient'")
    explanation: str = Field(description="Human-readable summary")


class PosteriorReport(BaseModel):
    """Posterior probability P(Y) for a yes/no-style research objective.

    Aggregates evidence direction across all active claims:
      log_odds = supporting_count - contradicting_count
      posterior = 1 / (1 + exp(-log_odds))

    Only meaningful for verificatory, comparative, and predictive questions.
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


def _track_completed(
    claim: Claim,
    track_name: str,
    flag: str,
    repo: EpistemicRepository,
) -> bool:
    """Check whether a verification track ran on a claim.

    The checked flag is the primary signal, but TMS demotion resets flags
    via Peirce cycling while leaving track results intact.  We accept
    persistent results as proof the investigation happened:

    - adversarial: adversarial_balance is not None
    - convergence: convergence_checked (no cheap sync fallback)
    - others: fall back to the flag only
    """
    if getattr(claim, flag, False):
        return True

    # Flag was reset — check for persistent results that survive demotion
    if track_name == "adversarial" and claim.adversarial_balance is not None:
        return True

    return False


async def compute_answer_confidence(
    repo: EpistemicRepository,
    objective_id: str,
) -> AnswerConfidenceReport:
    """Compute answer-level confidence from a checklist of pass/fail checks.

    1. Loads objective to get question_type
    2. Loads claims, evidence, uncertainties
    3. Filters to active (non-abandoned) claims
    4. Runs universal checks (evidence_basis, scrutiny_complete,
       uncertainties_resolved, belief_maintenance)
    5. Runs routing-dependent checks for each PRIMARY track
    6. Aggregates: log_odds = passes - failures, confidence = sigmoid

    Args:
        repo: Repository with completed epistemic run.
        objective_id: Which inquiry to assess.

    Returns:
        AnswerConfidenceReport with checks, scores, and level.
    """
    from .routing import get_routing_profile, TrackActivation

    # 1. Load objective
    objective = await repo.get_objective(objective_id)
    question_type = objective.question_type

    # 2. Load entities
    claims = await repo.get_claims_for_objective(objective_id)
    evidence = await repo.get_evidence_for_objective(objective_id)
    uncertainties = await repo.get_uncertainties_for_objective(objective_id)

    # 3. Filter to active claims
    active_claims = [c for c in claims if not c.abandoned]

    checks: list[CheckResult] = []

    # ------------------------------------------------------------------
    # 4. Universal checks
    # ------------------------------------------------------------------

    # evidence_basis (Kahneman): at least one active claim has at least one
    # judged, non-invalidated evidence
    has_evidence_basis = False
    for claim in active_claims:
        claim_evidence = [
            e
            for e in evidence
            if e.entity_id in claim.evidence_ids
            and not e.invalidated
            and e.support_judgment is not None
        ]
        if claim_evidence:
            has_evidence_basis = True
            break

    checks.append(
        CheckResult(
            name="evidence_basis",
            tradition="kahneman",
            passed=has_evidence_basis,
            detail="at least one active claim has judged, non-invalidated evidence"
            if has_evidence_basis
            else "no active claim has judged, non-invalidated evidence",
        )
    )

    # scrutiny_complete (Peirce): all active claims have scrutiny_verdict
    # in ("pass", "fail"). Requires active claims to exist.
    if active_claims:
        all_scrutinized = all(
            c.scrutiny_verdict in ("pass", "fail") for c in active_claims
        )
    else:
        all_scrutinized = False

    checks.append(
        CheckResult(
            name="scrutiny_complete",
            tradition="peirce",
            passed=all_scrutinized,
            detail="all active claims have scrutiny verdict"
            if all_scrutinized
            else "not all active claims have scrutiny verdict"
            if active_claims
            else "no active claims",
        )
    )

    # uncertainties_resolved: no unresolved blocking uncertainties
    unresolved_blocking = [
        u for u in uncertainties if u.is_blocking and not u.is_resolved
    ]
    unc_resolved = len(unresolved_blocking) == 0

    checks.append(
        CheckResult(
            name="uncertainties_resolved",
            tradition="",
            passed=unc_resolved,
            detail="no unresolved blocking uncertainties"
            if unc_resolved
            else f"{len(unresolved_blocking)} unresolved blocking uncertainties remain",
        )
    )

    # belief_maintenance (Doyle): no active claim has needs_revalidation=True
    # Requires active claims to exist.
    if active_claims:
        needs_reval = [c for c in active_claims if c.needs_revalidation]
        belief_ok = len(needs_reval) == 0
    else:
        belief_ok = False
        needs_reval = []

    checks.append(
        CheckResult(
            name="belief_maintenance",
            tradition="doyle",
            passed=belief_ok,
            detail="no active claims need revalidation"
            if belief_ok
            else f"{len(needs_reval)} claims need revalidation"
            if active_claims
            else "no active claims",
        )
    )

    # ------------------------------------------------------------------
    # 5. Routing-dependent track checks
    # ------------------------------------------------------------------

    if question_type is not None:
        try:
            profile = get_routing_profile(question_type)
        except KeyError:
            logger.warning(
                "Unknown question type %r — skipping track checks",
                question_type,
            )
            profile = None

        if profile is not None:
            for track_name, activation in profile.tracks.items():
                if activation != TrackActivation.PRIMARY:
                    continue

                # Skip tracks without a checked flag (e.g. "argument")
                flag = TRACK_FLAGS.get(track_name)
                if flag is None:
                    continue

                # Track check passes if active_claims exist AND all
                # active claims have completed the track.
                #
                # Complication: TMS demotion resets checked flags (Peirce
                # cycling) but the track's RESULT persists.  A claim whose
                # adversarial_checked was reset to False after demotion
                # still has adversarial_balance set — proving the check
                # ran.  We accept persistent results as evidence the
                # investigation happened.
                if active_claims:
                    completed = sum(
                        1
                        for c in active_claims
                        if _track_completed(c, track_name, flag, repo)
                    )
                    track_passed = completed == len(active_claims)
                    if track_passed:
                        detail = f"all {len(active_claims)} active claims checked"
                    else:
                        detail = (
                            f"{len(active_claims) - completed}/{len(active_claims)} "
                            f"active claims missing {track_name}"
                        )
                else:
                    track_passed = False
                    detail = "no active claims"

                tradition = TRACK_TRADITIONS.get(track_name, "")
                checks.append(
                    CheckResult(
                        name=f"track:{track_name}",
                        tradition=tradition,
                        passed=track_passed,
                        detail=detail,
                    )
                )

    # ------------------------------------------------------------------
    # 6. Aggregate
    # ------------------------------------------------------------------

    passes = sum(1 for c in checks if c.passed)
    failures = sum(1 for c in checks if not c.passed)
    log_odds = passes - failures

    if abs(log_odds) < 700:
        confidence = 1.0 / (1.0 + math.exp(-log_odds))
    else:
        confidence = 1.0 if log_odds > 0 else 0.0

    level = _level_from_score(confidence)

    failed_names = [c.name.replace("_", " ") for c in checks if not c.passed]
    if failed_names:
        explanation = (
            f"Answer confidence {confidence:.2f} ({level.upper()}). "
            f"{passes} of {len(checks)} checks passed. "
            f"Failed: {', '.join(failed_names)}."
        )
    else:
        explanation = (
            f"Answer confidence {confidence:.2f} ({level.upper()}). "
            f"All {passes} checks passed."
        )

    return AnswerConfidenceReport(
        objective_id=objective_id,
        question_type=question_type,
        checks=checks,
        passes=passes,
        failures=failures,
        log_odds=log_odds,
        confidence=round(confidence, 6),
        level=level,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Posterior P(Y) scoring
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _level_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "moderate"
    if score >= 0.25:
        return "low"
    return "insufficient"
