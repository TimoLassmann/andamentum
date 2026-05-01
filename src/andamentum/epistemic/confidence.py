"""Posterior confidence reporter for completed epistemic inquiries.

Reads the abductive integration agent's verdict on each active claim and
maps it to a probability:

    supports   at confidence c → 0.5 + c/2
    contradicts at confidence c → 0.5 - c/2
    insufficient                → 0.5

Multi-claim objectives aggregate via confidence-weighted averaging. The
abduction agent (epistemic_integrate_evidence) is Peirce-grounded — it
already considers evidence convergence, adversarial outcome, and remaining
uncertainties when forming its verdict. This module honours that verdict
rather than re-deriving it from per-item counts.

Counting only fires as a *fallback* when integration never ran for a
claim (e.g. the pipeline aborted before reaching IntegrateEvidence).
That's the only branch where supporting_count / contradicting_count
materially affect the answer.

The previous design blended a sigmoid-of-counts with the integration
verdict via a tunable mixing constant. With cluster-size weighting (A4)
the blend formula's behaviour shifted in ways the constant could no
longer absorb honestly. Trusting the abduction step — which is where
the philosophical work already lives — removes the constants we could
not justify with held-out data.

Architecture: Layer 1 (framework-agnostic, pure computation)

Usage::

    repo = await EpistemicRepository.for_database("my_run", db_dir=Path("./results"))
    posterior = await compute_posterior(repo, objective_id="...")
    if posterior:
        print(f"Posterior confidence: {posterior.posterior:.2%}")
"""

import logging
import math
from typing import Literal

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

    Primary signal is the abduction agent's verdict (mode="abductive").
    Counting is reported as a diagnostic field and only drives the
    posterior when integration didn't run (mode="counting_fallback").
    Only meaningful for verificatory and predictive questions; other
    question types return None from compute_posterior().
    """

    posterior: float = Field(description="P(Y) in [0.0, 1.0]")
    log_odds: int = Field(description="Effective log-odds from posterior")
    supporting_count: float = Field(
        description=(
            "Diagnostic: total weighted supporting evidence across active "
            "claims. Each representative contributes "
            "``1 + log(corroboration_count)``. Drives the posterior only "
            "in counting_fallback mode (when no claim received an "
            "integration verdict)."
        )
    )
    contradicting_count: float = Field(
        description=(
            "Diagnostic: total weighted contradicting evidence across "
            "active claims. Same weighting as supporting_count."
        )
    )
    counting_posterior: float = Field(
        description=(
            "Diagnostic: P(Y) implied by counting alone "
            "(sigmoid(supporting - contradicting)). For comparison with "
            "the abduction-driven posterior; not used when the abduction "
            "verdict is available."
        )
    )
    integration_verdict: str | None = Field(
        default=None,
        description="Integration assessment: 'supports', 'contradicts', 'insufficient', or None",
    )
    integration_confidence: float | None = Field(
        default=None,
        description="Integration confidence 0.0-1.0, or None if not run",
    )
    mode: str = Field(
        default="abductive",
        description=(
            "'abductive' when the posterior follows the integration "
            "verdict; 'counting_fallback' when no claim received an "
            "integration verdict and the posterior is the sigmoid of "
            "weighted counts."
        ),
    )
    terminal_state: Literal[
        "completed", "retrieval_failed", "oscillation_detected"
    ] = Field(
        default="completed",
        description=(
            "How the investigation terminated. 'completed' for normal runs; "
            "'retrieval_failed' when evidence extraction kept returning empty "
            "content, meaning the posterior is based on insufficient data; "
            "'oscillation_detected' when one or more claims hit the "
            "scrutinise/resolve cycle cap before converging — the posterior "
            "is 0.5 (genuinely uncertain) and the inquiry did not fix belief."
        ),
    )
    objective_id: str
    question_type: str
    explanation: str


async def compute_posterior(
    repo: EpistemicRepository,
    objective_id: str,
    *,
    retrieval_failed: bool = False,
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
    # Retrieval-failed short-circuit: the pipeline flagged that evidence
    # extraction kept returning empty content. Emit an explicit
    # terminal_state report so callers don't mistake an uninformative
    # 0.5 for a "genuinely balanced evidence" conclusion. Still scoped
    # to POSTERIOR_ELIGIBLE question types — for ineligible types,
    # posterior is N/A regardless.
    if retrieval_failed:
        objective = await repo.get_objective(objective_id)
        qt = objective.question_type
        # Multi-seed-claim aware: ``is_verification_task()`` covers BOTH
        # single-seed mode (claim_to_verify set) AND multi-seed mode
        # (decomposition with sub-investigations). Without this, a
        # decomposed run where the parent was classified explanatory
        # would silently drop its retrieval_failed report — same shape as
        # the case-54 silent-loss bug we fixed for single-seed mode.
        is_verification_mode = objective.is_verification_task()
        if not is_verification_mode and (qt is None or qt not in POSTERIOR_ELIGIBLE):
            return None
        return PosteriorReport(
            posterior=0.5,
            log_odds=0,
            supporting_count=0,
            contradicting_count=0,
            counting_posterior=0.5,
            mode="counting_only",
            objective_id=objective_id,
            question_type=qt or "verificatory",
            explanation=(
                "Retrieval failed: evidence extraction returned empty content "
                "at least 3 times consecutively. Posterior defaults to 0.5 "
                "(uninformative); terminal_state='retrieval_failed'."
            ),
            terminal_state="retrieval_failed",
        )

    # 1. Load objective, check eligibility
    #
    # Eligibility has two paths:
    #   (a) question_type is verificatory or predictive — the parent
    #       question's answer is binary, P(Y) maps cleanly.
    #   (b) the objective is in seed_claim mode (claim_to_verify is set) —
    #       the objective is verifying ONE specific claim binary-by-
    #       construction regardless of the parent question_type. This
    #       is the decomposed-children case: a parent classified as
    #       explanatory/exploratory/etc. spawns N seed-claim children,
    #       each of which runs binary verification on its seed. Without
    #       this branch, decomposed runs whose parent was misclassified
    #       (or genuinely non-binary at the parent level) silently lose
    #       their per-child posteriors — caught on smoke_v12_decompose
    #       case 54 where the parent was classified explanatory and
    #       compute_posterior dropped 7 valid integration verdicts.
    #
    # comparative is intentionally excluded from (a) because it has 3+
    # outcomes (A better / B better / equivalent). In seed_claim mode a
    # comparative parent's children still verify specific seed claims,
    # so they pass via (b).
    objective = await repo.get_objective(objective_id)
    question_type = objective.question_type
    # Multi-seed-claim aware (Phase 8 follow-up): ``is_verification_task()``
    # is True for both single-seed mode (claim_to_verify set) AND multi-
    # seed mode (decomposition with sub-investigations). Each branch is
    # binary verification by construction; the parent's classifier output
    # is irrelevant to per-claim posterior eligibility.
    is_verification_mode = objective.is_verification_task()
    if not is_verification_mode and (
        question_type is None or question_type not in POSTERIOR_ELIGIBLE
    ):
        return None
    # PosteriorReport.question_type is required (str). For seed-claim
    # objectives whose parent didn't classify (rare but possible),
    # default to "verificatory" for the report — it's the binary
    # operation the seed-claim machinery is performing.
    if question_type is None:
        question_type = "verificatory"

    # 2. Load claims and evidence
    claims = await repo.get_claims_for_objective(objective_id)
    evidence = await repo.get_evidence_for_objective(objective_id)
    active_claims = [c for c in claims if not c.abandoned]

    # Oscillation handling (Phase 2 of multi-seed-claim refactor):
    #
    # Previous semantic: ANY capped claim short-circuited the entire
    # posterior to oscillation_detected, discarding healthy verdicts on
    # sibling claims. That broke multi-seed-claim's resilience promise —
    # one cycling claim shouldn't take 4 healthy claims down with it.
    #
    # New semantic: filter capped claims out of aggregation. Only emit
    # the all-or-nothing oscillation_detected report when every active
    # claim is capped (nothing left to aggregate). Otherwise, drop the
    # capped subset, aggregate the rest normally, and surface the
    # oscillation as a diagnostic in the explanation. The combiner
    # (CombineClaimVerdicts in Phase 4) will receive the partial signal
    # and apply the combination_rule over the non-capped claims.
    capped = [c for c in active_claims if getattr(c, "cycle_capped", False)]
    if capped and len(capped) == len(active_claims) and active_claims:
        # All-capped: no signal to aggregate. Honest oscillation_detected.
        concern_total = sum(len(c.persistent_concerns) for c in capped)
        capped_summary = ", ".join(
            f"{c.entity_id[:8]}:{len(c.persistent_concerns)}c" for c in capped
        )
        return PosteriorReport(
            posterior=0.5,
            log_odds=0,
            supporting_count=0,
            contradicting_count=0,
            counting_posterior=0.5,
            mode="counting_only",
            objective_id=objective_id,
            question_type=question_type,
            explanation=(
                f"Oscillation detected: ALL {len(capped)} active claim(s) hit "
                f"the scrutiny-resolve cycle cap with {concern_total} "
                f"persistent concerns total ({capped_summary}). Posterior "
                "defaults to 0.5 (uninformative); "
                "terminal_state='oscillation_detected'. The inquiry did not "
                "converge — diagnose by inspecting claim.persistent_concerns "
                "to decide whether cluster-dedup or claim reformulation is "
                "the right architectural follow-up."
            ),
            terminal_state="oscillation_detected",
        )

    # Partial-cap: some claims capped, others have verdicts. Aggregate
    # over the non-capped subset; remember the capped count for the
    # report's explanation.
    n_capped_partial = len(capped)
    if n_capped_partial:
        active_claims = [
            c for c in active_claims if not getattr(c, "cycle_capped", False)
        ]

    # 3. Diagnostic: weighted counts across active claims. These are reported
    # for inspection and used only as the counting fallback when no claim
    # received an integration verdict.
    supporting = 0.0
    contradicting = 0.0
    for claim in active_claims:
        claim_evidence = [
            e
            for e in evidence
            if e.entity_id in claim.evidence_ids
            and not e.invalidated
            and e.cluster_status not in ("corroborative", "deferred")
        ]
        for e in claim_evidence:
            cluster_size = max(1, getattr(e, "corroboration_count", 1) or 1)
            weight = 1.0 + math.log(cluster_size)
            if e.support_judgment == "supports":
                supporting += weight
            elif e.support_judgment == "contradicts":
                contradicting += weight

    counting_log_odds = supporting - contradicting
    if abs(counting_log_odds) < 700:
        counting_posterior = 1.0 / (1.0 + math.exp(-counting_log_odds))
    else:
        counting_posterior = 1.0 if counting_log_odds > 0 else 0.0

    # 4. Honour the abduction agent's verdict per claim and aggregate.
    # Each integrated claim contributes a probability and a confidence
    # weight; the objective-level posterior is the confidence-weighted
    # average. "insufficient" verdicts contribute 0.5 with their own
    # confidence weight (so a confidently-insufficient verdict pulls
    # toward neutral; a low-confidence insufficient verdict barely moves
    # the average).
    integrated_claims = [
        c for c in active_claims if c.integrated_assessment is not None
    ]

    integration_verdict: str | None = None
    integration_confidence: float | None = None

    # Multi-seed-claim aware aggregation (post-audit Commit B):
    # When the Objective has a decomposition with a combination_rule, the
    # headline posterior must honour the rule (AND→min, OR→max,
    # WEIGHTED_AND→weighted mean over per-claim posteriors). The previous
    # confidence-weighted average ignored the rule and could disagree
    # with the rule-aware verdict in CombineClaimVerdicts (e.g. AND over
    # [0.9, 0.35] gives min=0.35 but weighted-average ≈ 0.75 — opposite
    # directions). Now ``compute_posterior`` delegates to
    # ``combine_claim_verdicts`` for decomposed runs so callers see one
    # consistent number.
    decomposition = getattr(objective, "decomposition", None) or {}
    from .graph.combination import resolve_combination_rule

    combination_rule = resolve_combination_rule(objective)
    use_rule_aware = bool(integrated_claims) and bool(combination_rule)

    if use_rule_aware:
        from .graph.combination import (
            combine_claim_verdicts,
            extract_weights_from_decomposition,
        )

        # Order claims by sub_investigation_id per the decomposition so
        # weights align — same alignment CombineClaimVerdicts uses.
        sub_ids_in_order = [
            s.get("id")
            for s in (decomposition.get("sub_investigations") or [])
        ]
        claims_by_sub = {
            c.sub_investigation_id: c
            for c in integrated_claims
            if c.sub_investigation_id is not None
        }
        ordered = [
            claims_by_sub[sid]
            for sid in sub_ids_in_order
            if sid in claims_by_sub
        ]
        # Fall back to the integrated_claims list if no sub_investigation_id
        # alignment is possible (e.g. ProposeClaims path that happened to
        # have a combination_rule from a prior decomposition).
        if not ordered:
            ordered = integrated_claims
            weights = None
        else:
            weights = extract_weights_from_decomposition(
                decomposition, ordered
            )
        assert combination_rule is not None  # guarded by use_rule_aware
        combined = combine_claim_verdicts(ordered, combination_rule, weights=weights)

        if combined.posterior is not None:
            posterior = combined.posterior
            integration_verdict = combined.verdict
            # Average confidence as before — the per-claim confidences
            # remain the diagnostic signal; the rule decides which claims
            # dominate via the combination operation.
            integration_confidence = sum(
                c.integrated_confidence or 0.0 for c in integrated_claims
            ) / len(integrated_claims)
            # combination_rule is non-None inside this branch (the
            # use_rule_aware predicate guarantees it).
            mode = f"rule_aware_{(combination_rule or 'unknown').lower()}"
        else:
            # UNION or no_data: no scalar verdict. Fall back to counting.
            posterior = counting_posterior
            mode = "counting_fallback"
    elif integrated_claims:
        # Per-claim probability + weight (rule-blind path: open-research
        # multi-claim or seed-claim runs without a combination_rule).
        weighted_sum = 0.0
        weight_total = 0.0
        verdict_counts: dict[str, int] = {
            "supports": 0,
            "contradicts": 0,
            "insufficient": 0,
        }
        confidence_sum = 0.0
        for c in integrated_claims:
            verdict = c.integrated_assessment
            confidence = c.integrated_confidence or 0.5
            confidence = max(0.0, min(1.0, confidence))
            confidence_sum += confidence

            if verdict == "supports":
                claim_p = 0.5 + confidence / 2
            elif verdict == "contradicts":
                claim_p = 0.5 - confidence / 2
            else:  # "insufficient"
                claim_p = 0.5

            weighted_sum += claim_p * confidence
            weight_total += confidence
            if verdict in verdict_counts:
                verdict_counts[verdict] += 1

        # If all confidences were exactly 0, fall back to unweighted mean
        # at 0.5 — we have integration verdicts but no information to weight
        # them with.
        if weight_total > 0:
            posterior = weighted_sum / weight_total
        else:
            posterior = 0.5

        # Surface a single objective-level verdict label: the modal
        # directional verdict if one exists, otherwise insufficient.
        if verdict_counts["supports"] > verdict_counts["contradicts"]:
            integration_verdict = "supports"
        elif verdict_counts["contradicts"] > verdict_counts["supports"]:
            integration_verdict = "contradicts"
        else:
            integration_verdict = "insufficient"
        integration_confidence = confidence_sum / len(integrated_claims)
        mode = "abductive"
    else:
        # No claim received an integration verdict — abduction never ran.
        # Fall back to the counting signal as the only available input.
        posterior = counting_posterior
        mode = "counting_fallback"

    # 5. Compute effective log-odds for the report
    if posterior <= 0.0:
        log_odds = -700
    elif posterior >= 1.0:
        log_odds = 700
    else:
        log_odds = round(math.log(posterior / (1.0 - posterior)))

    # 6. Build explanation
    parts = [f"Posterior {posterior:.4f} for {question_type} question."]
    if n_capped_partial:
        parts.append(
            f"NOTE: {n_capped_partial} claim(s) hit the scrutiny-resolve "
            "cycle cap and were excluded from aggregation. Aggregation is "
            "over the non-capped subset only."
        )
    if mode == "abductive":
        parts.append(
            f"Mode: abductive (driven by integration verdict). "
            f"{len(integrated_claims)} claim(s) integrated; "
            f"verdict={integration_verdict}, "
            f"avg confidence {integration_confidence:.2f}."
        )
        # Diagnostic disclosure of the counting signal for transparency
        # (Doyle TMS: surface disagreements between counting and abduction
        # so a reader can see when the literature was split even though
        # the abduction agent committed).
        counting_direction = (
            "supports"
            if counting_posterior > 0.5
            else ("contradicts" if counting_posterior < 0.5 else "neutral")
        )
        if (
            integration_verdict in ("supports", "contradicts")
            and counting_direction != "neutral"
            and integration_verdict != counting_direction
        ):
            parts.append(
                f"NOTE: counting diagnostic ({counting_direction}, "
                f"{supporting:.2f} vs {contradicting:.2f}) disagrees with "
                f"the abductive verdict ({integration_verdict}). The "
                f"abduction step is authoritative; this is informational."
            )
    else:
        parts.append(
            f"Mode: counting_fallback (no claim received an integration "
            f"verdict). Per-item weighted counts: "
            f"{supporting:.2f} supporting vs {contradicting:.2f} contradicting."
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
