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

# Confidence penalties live in ``epistemic.thresholds`` (the canonical
# constants module). Re-exported here so existing callers that imported
# them from ``confidence`` keep working.
from .thresholds import (
    CYCLE_CAP_CONFIDENCE_PENALTY,
    RETRIEVAL_FAILED_CONFIDENCE_PENALTY,
)

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
    terminal_state: Literal["completed", "retrieval_failed", "oscillation_detected"] = (
        Field(
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
    # Retrieval-failed handling: the pipeline flagged that evidence
    # extraction kept returning empty content. The flag prevents *more*
    # extraction work from running, but any IBE chain that fired
    # beforehand produced a legitimate verdict on the evidence already
    # gathered.
    #
    # The previous design short-circuited here to posterior=0.5 with
    # terminal_state="retrieval_failed", discarding any signal already
    # acquired. That repeated the cycle-cap output-layer anti-pattern —
    # the v14 SciFact case 54 was a textbook instance: claim reached
    # SUPPORTED with integrated_assessment="contradicts" at 0.857
    # confidence, then retrieval failed during a later investigation
    # cycle, and the short-circuit zeroed the verdict to 0.5.
    #
    # New rule: let the function run normally, then at the end pull the
    # aggregated posterior toward neutral with
    # RETRIEVAL_FAILED_CONFIDENCE_PENALTY and stamp
    # terminal_state="retrieval_failed". Genuine no-signal runs (no
    # claims, no IA, no evidence) still produce posterior=0.5 by the
    # normal flow.

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

    # Cycle-capped handling — three-way rule (2026-05-04):
    #
    # Cycle_capped is an inquiry-layer safety belt that stops more LLM
    # work on a non-converging claim. The output layer should still
    # surface the signal that was acquired before the cap fired, with
    # provenance flagged in the explanation:
    #
    #   1. capped + integrated_assessment → contribute the verdict with
    #      a confidence penalty (CYCLE_CAP_CONFIDENCE_PENALTY).
    #   2. capped + no IA, evidence one-sided → counting path surfaces
    #      it (no special handling needed; we just stop filtering).
    #   3. capped + no IA, balanced → terminal_state="oscillation_detected"
    #      via the no-signal terminal at the end of this function.
    #
    # The previous all-capped → 0.5 short-circuit conflated "inquiry
    # didn't converge" with "verdict is unstable" — distinct epistemic
    # situations. See
    # docs/superpowers/plans/2026-05-04-confidence-honest-aggregation.md.
    capped = [c for c in active_claims if getattr(c, "cycle_capped", False)]
    n_capped_partial = len(capped)

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
    decomposition = getattr(objective, "decomposition", None)
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
        # Phase 6 of the Move-3 plan: typed Decomposition access.
        sub_ids_in_order = [
            s.id for s in (decomposition.sub_investigations if decomposition else [])
        ]
        claims_by_sub = {
            c.sub_investigation_id: c
            for c in integrated_claims
            if c.sub_investigation_id is not None
        }
        ordered = [
            claims_by_sub[sid] for sid in sub_ids_in_order if sid in claims_by_sub
        ]
        # Fall back to the integrated_claims list if no sub_investigation_id
        # alignment is possible (e.g. ProposeClaims path that happened to
        # have a combination_rule from a prior decomposition).
        if not ordered:
            ordered = integrated_claims
            weights = None
        else:
            weights = extract_weights_from_decomposition(decomposition, ordered)
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
            # Cycle-cap penalty: a capped claim's IBE-certified verdict
            # still informs the headline, just with reduced weight to
            # reflect the non-converged inquiry that produced it.
            if getattr(c, "cycle_capped", False):
                confidence *= CYCLE_CAP_CONFIDENCE_PENALTY
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

    # 5a. Cycle-cap penalty for the counting-fallback path.
    #
    # The integration paths above (rule_blind and rule_aware via
    # combine_claim_verdicts) already apply CYCLE_CAP_CONFIDENCE_PENALTY
    # per-claim when a capped claim has an integrated_assessment.
    # The counting-fallback path is the gap: when a capped claim
    # contributes signal via support_judgment counts only (because
    # PromoteToSupported's cycle_capped filter prevented IBE from
    # running), its raw counts run at full strength. SciFact case 439
    # v15 was the demonstration: cycle_capped=True, no IA, 2 supports
    # in a thin pool → cluster weighting amplified to log_odds 4.20,
    # posterior 0.985 — confident SUP on a gold-NEI claim.
    #
    # The principle is the same as the integration-path penalty:
    # capped means inquiry didn't converge cleanly; whatever signal
    # we have is provisional, surface it with reduced weight. Pulling
    # the counting_posterior toward neutral mirrors the per-claim
    # confidence pull in the integration path.
    if mode == "counting_fallback" and n_capped_partial > 0:
        posterior = 0.5 + (posterior - 0.5) * CYCLE_CAP_CONFIDENCE_PENALTY

    # 5b. Apply retrieval_failed pull-toward-neutral.
    #
    # The flag signals that evidence extraction kept returning empty
    # content, so the inquiry was prevented from gathering more.
    # Whatever verdict the IBE chain produced beforehand is provisional
    # but directional — surface it with a confidence penalty rather
    # than zeroing it. (See SciFact case 54 v14: integrated_assessment
    # was contradicts at 0.857 but the old short-circuit returned
    # posterior=0.5, discarding the verdict.)
    #
    # Stacks multiplicatively with 5a when both apply: a cycle-capped
    # claim under retrieval_failed conditions is doubly provisional,
    # and 0.7 × 0.7 = 0.49 reflects that.
    if retrieval_failed:
        posterior = 0.5 + (posterior - 0.5) * RETRIEVAL_FAILED_CONFIDENCE_PENALTY
        if integration_confidence is not None:
            integration_confidence = max(
                0.0,
                min(1.0, integration_confidence * RETRIEVAL_FAILED_CONFIDENCE_PENALTY),
            )

    # 6. Compute effective log-odds for the report
    if posterior <= 0.0:
        log_odds = -700
    elif posterior >= 1.0:
        log_odds = 700
    else:
        log_odds = round(math.log(posterior / (1.0 - posterior)))

    # 7. Decide terminal_state.
    #
    # Precedence: retrieval_failed > oscillation_detected > completed.
    # retrieval_failed wins because it's a process-level signal about
    # the inquiry's evidence base, not a statement about the verdict's
    # stability — the verdict itself may still be directional (case 54).
    #
    # Genuine oscillation = ALL active claims are capped AND none have
    # an integrated_assessment AND the counting signal is essentially
    # balanced. That's the case the original design was trying to
    # catch — the inquiry oscillated without ever producing a
    # directional verdict OR a one-sided evidence pool. Anything else
    # has signal worth surfacing, so terminal_state stays "completed".
    is_balanced_count = abs(counting_log_odds) < 1.0
    all_capped_no_signal = (
        n_capped_partial > 0
        and n_capped_partial == len(active_claims)
        and not integrated_claims
        and is_balanced_count
    )
    terminal_state: Literal["completed", "retrieval_failed", "oscillation_detected"]
    if retrieval_failed:
        terminal_state = "retrieval_failed"
    elif all_capped_no_signal:
        terminal_state = "oscillation_detected"
    else:
        terminal_state = "completed"

    # 8. Build explanation
    parts = [f"Posterior {posterior:.4f} for {question_type} question."]
    if retrieval_failed:
        parts.append(
            "Retrieval failed: evidence extraction returned empty content "
            "for several consecutive attempts. Posterior reflects signal "
            f"acquired before the failure, with confidence penalty "
            f"{RETRIEVAL_FAILED_CONFIDENCE_PENALTY} (further inquiry was "
            "prevented). terminal_state='retrieval_failed'."
        )
    if all_capped_no_signal:
        concern_total = sum(len(c.persistent_concerns) for c in capped)
        parts.append(
            f"Oscillation detected: ALL {n_capped_partial} active claim(s) "
            f"hit the scrutiny-resolve cycle cap, none produced an integration "
            f"verdict, and counting is balanced ({concern_total} persistent "
            "concerns total). Posterior at 0.5 is the honest outcome."
        )
    elif n_capped_partial:
        # Provenance: cap fired, but signal exists (either via integrated
        # verdicts on capped claims or via one-sided counting). Surface
        # the cap in the explanation rather than zeroing the number.
        # The penalty path differs by mode: integration paths apply it
        # per-claim inside the verdict aggregation; counting_fallback
        # pulls the final posterior toward neutral (5a above).
        applied_at = (
            "counting_posterior pulled toward neutral"
            if mode == "counting_fallback"
            else "per-claim confidence reduced"
        )
        parts.append(
            f"NOTE: {n_capped_partial} claim(s) hit the scrutiny-resolve "
            f"cycle cap; their signal is included with confidence penalty "
            f"{CYCLE_CAP_CONFIDENCE_PENALTY} ({applied_at}; verdict acquired "
            "under non-converged inquiry — provisional but directional)."
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

    # Force posterior to exactly 0.5 in the genuine-oscillation terminal
    # so callers comparing on the value still see the canonical neutral.
    if all_capped_no_signal:
        posterior = 0.5
        log_odds = 0

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
        terminal_state=terminal_state,
    )
