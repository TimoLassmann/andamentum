"""Stage management operations.

Gate-based claim promotion and demotion. PromoteClaimOperation validates
gate requirements before allowing a claim to advance to the next stage.
DemoteClaimOperation handles regression when scrutiny fails or new
blocking uncertainties appear.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim entities
"""

from typing import Literal, cast

from .base import BaseOperation, OperationInput, OperationResult

from ..entities import CapEvaluation, Claim
from ..entities.claim import ClaimStage
from ..thresholds import (
    MECHANICAL_VERDICT_MAX_CONFIDENCE,
    SCRUTINY_STRONG_MASS_THRESHOLD,
)


class PromoteClaimOperation(BaseOperation):
    """Promote claim to next stage with gate validation.

    Validates gate requirements before allowing promotion.
    Records promotion in claim history.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import validate_promotion, get_next_stage

        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        # Determine target stage
        target_stage = get_next_stage(claim.stage)
        if not target_stage:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message=f"Claim at {claim.stage.value} cannot be promoted further",
            )

        # Get objective's question_type for routing-aware gate thresholds
        objective = await self.repo.get("objective", claim.objective_id)
        question_type = objective.question_type

        # Validate gate requirements
        gate_result = await validate_promotion(
            claim, target_stage, self.repo, question_type=question_type
        )

        if not gate_result.passed:
            await self.log_event(
                "gate_failed",
                claim.entity_id,
                {
                    "target_stage": target_stage.value,
                    "reasons": gate_result.blocking_reasons,
                },
            )
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message=gate_result.blocking_reasons[0]
                if gate_result.blocking_reasons
                else "Gate failed",
                validation_errors=gate_result.blocking_reasons,
            )

        # Record promotion (canonical writer: claim.record_promotion).
        old_stage = claim.stage
        claim.record_promotion(
            old_stage,
            target_stage,
            justification=f"Gate requirements met for {target_stage.value}",
        )

        # Compute and set confidence score
        from ..gates import compute_confidence_score, quality_weighted_evidence_sum

        quality_sum = await quality_weighted_evidence_sum(claim, self.repo)
        evidence_count = max(1, claim.evidence_count)
        avg_quality = quality_sum / evidence_count
        claim.confidence_score = compute_confidence_score(
            target_stage, avg_quality, adversarial_balance=claim.adversarial_balance
        )

        await self.repo.save(claim)

        await self.log_event(
            "claim_promoted",
            claim.entity_id,
            {
                "from": old_stage.value,
                "to": target_stage.value,
            },
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] → {target_stage.value}",
        )


class DemoteClaimOperation(BaseOperation):
    """Demote claim to previous stage.

    Used when scrutiny fails or new blocking uncertainties are found.

    NOTE: This does NOT cascade to derived evidence. Normal demotion means
    the claim didn't meet promotion requirements — the evidence is still
    valid and may support other claims. Only TMS-triggered demotion
    (RevalidateClaimOperation) cascades, because there the evidence
    foundation itself has been undermined.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import get_previous_stage

        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        target_stage = get_previous_stage(claim.stage)
        if not target_stage:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="Claim at HYPOTHESIS cannot be demoted",
            )

        old_stage = claim.stage
        claim.record_demotion(
            target_stage, justification="Scrutiny failure or blocking uncertainty"
        )

        await self.repo.save(claim)

        await self.log_event(
            "claim_demoted",
            claim.entity_id,
            {
                "from": old_stage.value,
                "to": target_stage.value,
            },
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Demoted to {target_stage.value}",
        )


class PromoteAsRefutedOperation(BaseOperation):
    """Promote a HYPOTHESIS claim to SUPPORTED with integrated_assessment="contradicts".

    Bypasses the scrutiny gate's "verdict must be pass" rule. Intended as an
    escape valve when the evidence overwhelmingly contradicts the claim —
    without this path, such claims get abandoned and their contradiction
    signal drops out of the posterior entirely.

    Integration LLM is skipped for these claims because integrated_assessment
    is pre-set; the existing "if integrated_assessment is not None: skip"
    check in IntegrateEvidence handles this correctly.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import count_support_contradict, is_refuted_by_evidence

        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )
        if claim.stage != ClaimStage.HYPOTHESIS:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message=f"Refuted-promotion only runs on HYPOTHESIS, not {claim.stage.value}",
            )
        if not await is_refuted_by_evidence(claim, self.repo):
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="Evidence balance does not meet refutation threshold",
            )

        n_sup, n_con = await count_support_contradict(claim, self.repo)
        # Confidence from imbalance; ceilinged because this is a mechanical
        # count, not holistic integration.
        confidence = min(
            MECHANICAL_VERDICT_MAX_CONFIDENCE, n_con / (n_con + max(1, n_sup))
        )

        claim.record_promotion(
            claim.stage,
            ClaimStage.SUPPORTED,
            justification=f"Refuted by evidence: {n_con} contradicts vs {n_sup} supports",
        )
        claim.integrated_assessment = "contradicts"
        claim.integrated_confidence = confidence
        claim.integrated_reasoning = (
            f"Promoted as refuted: evidence balance {n_con} contradicts vs "
            f"{n_sup} supports. Confidence derived mechanically from imbalance; "
            f"integration LLM skipped."
        )
        claim.confidence_score = confidence
        await self.repo.save(claim)

        await self.log_event(
            "claim_promoted_as_refuted",
            claim.entity_id,
            {"n_supports": n_sup, "n_contradicts": n_con, "confidence": confidence},
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] → supported (refuted: {n_con}⊥ / {n_sup}✓)",
        )


class ConcludeAtCycleCapOperation(BaseOperation):
    """Record what the evidence said at the moment the cycle cap fired.

    THE DEFECT THIS EXISTS TO FIX. A claim that cannot pass a stage gate
    retries, hits ``PEIRCE_CYCLE_CAP``, and — before this operation — was
    left with ``integrated_assessment=None`` forever, so the pipeline
    wrote "insufficient evidence" even when the gathered evidence was
    decisive. That is a scheduling side effect masquerading as an
    epistemic conclusion (2026-07-08 audit issue #2, case 781: the
    refuting studies were present; the claim just failed to climb).
    "Insufficient" must be a conclusion ABOUT THE EVIDENCE.

    What it does, deterministically and with **no LLM call**:

    1. Gathers the claim's eligible representative evidence via
       ``gates.eligible_representative_evidence`` — the same set scrutiny
       reasons over, by construction (one shared helper).
    2. Computes ``scrutiny_weight.compute_evidence_weight`` over it — the
       same engine that produces the scrutiny verdict, so the cap does not
       introduce a second, divergent reading of the evidence.
    3. Always records the reading as a ``CapEvaluation`` on the Claim.
    4. When the reading is directional — ``weight.label`` in
       {strong, moderate}, exactly the set for which
       ``ScrutiniseClaimOperation._verdict_from_weight`` returns "pass" —
       AND this operation owns the verdict, writes
       ``integrated_assessment`` / ``integrated_confidence`` /
       ``integrated_reasoning``. ``conflicting`` (genuine disagreement) and
       ``weak`` (too thin) emit no verdict: the cap must never manufacture
       a direction the evidence does not carry.

    CONFIDENCE. ``purity × mass_adequacy``, ceilinged at
    ``MECHANICAL_VERDICT_MAX_CONFIDENCE``, where::

        purity        = (dominant - minority) / (dominant + minority)
        mass_adequacy = min(1, dominant / SCRUTINY_STRONG_MASS_THRESHOLD)

    ``purity`` alone is a *directional-purity* measure and is 1.0 for any
    one-sided pool regardless of size, so on its own it saturated the
    ceiling for a single unopposed item — reporting the system's maximum
    mechanical confidence off one source, above what a 5-vs-1 pool got.
    That is incompatible with "emit the verdict the evidence supports
    WITH LOW CONFIDENCE" and with the Reichenbach ≥2-independent-sources
    doctrine the rest of the module applies. ``mass_adequacy`` discounts
    proportionally until the dominant direction carries the mass the
    system already calls "strong" (``SCRUTINY_STRONG_MASS_THRESHOLD``);
    no new constant is introduced. Both factors are in [0, 1], so the
    "can never be decisive" bound is unchanged.

    VERDICT OWNERSHIP across repeated cap firings. ``_mark_cycle_capped``
    can fire more than once per claim (ResolveUncertainties fires per
    affected claim per resolution), and other operations may certify a
    verdict *between* two firings, so ownership is recorded EXPLICITLY on
    the previous ``CapEvaluation``::

        cap_owns_live_verdict = (
            previous is not None
            and previous.wrote_live_verdict
            and claim.stage == ClaimStage.HYPOTHESIS
        )
        may_write = claim.integrated_assessment is None or cap_owns_live_verdict
        may_clear = cap_owns_live_verdict

    An earlier design inferred ownership from ``cap_evaluation is not
    None``, on the reasoning that no certified verdict can appear after a
    cap record exists. ``PromoteAsRefutedOperation`` breaks that
    reasoning: ``AbandonOrDemote`` dispatches it with no ``cycle_capped``
    filter and it guards only on stage, so cap#1 (non-directional) →
    refute-promotion (``contradicts``, SUPPORTED) → cap#2 would have
    nulled a certified verdict, leaving a SUPPORTED claim with none. The
    stage clause is the second, structural guard: this operation never
    promotes, so a verdict on a claim above HYPOTHESIS was written by a
    promotion path, not by the cap.

    When a later reading turns non-directional AND the cap owns the live
    verdict, it CLEARS what it previously wrote, so its record and its
    verdict never disagree.

    WHY MASS, NOT COUNTS. ``PromoteAsRefutedOperation`` — the other
    mechanical verdict writer — derives its confidence from
    ``gates.count_support_contradict``, i.e. hard integer counts of
    supporting vs contradicting items. This operation deliberately uses
    ``compute_evidence_weight``'s quality- and corroboration-weighted
    *mass* instead, because its stated requirement is to reproduce the
    reading scrutiny performs at the same moment. The two derivations are
    not unified: converting the refutation gate to soft mass would change
    an unbenchmarked promotion path. They share only the
    ``MECHANICAL_VERDICT_MAX_CONFIDENCE`` ceiling — one ceiling, two
    documented derivations.

    WHAT IT DELIBERATELY DOES NOT DO:

    * It does not change ``claim.stage``. A capped claim is one whose
      inquiry did not converge; promoting it would send it to an IBE
      chain that filters capped claims out anyway, leaving it stranded at
      SUPPORTED. Note that leaving the stage alone is NOT by itself
      sufficient to protect the verdict: ``AbandonOrDemote`` promotes
      failed-scrutiny claims of its own accord, and a later demotion runs
      ``record_demotion``, which nulls ``integrated_assessment``. That
      erasure path is closed in ``AbandonOrDemote`` itself, which now
      refuses to demote a cycle-capped claim (see the comment there).
    * It does not pre-apply ``CYCLE_CAP_CONFIDENCE_PENALTY``. That penalty
      is already applied downstream in all three posterior paths; applying
      it here too would recalibrate silently (0.7² = 0.49).
    * It touches no entity other than the Claim (P5).
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import eligible_representative_evidence
        from ..scrutiny_weight import compute_evidence_weight

        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )

        evidence = await eligible_representative_evidence(claim, self.repo)
        weight = compute_evidence_weight(evidence)

        # Ownership is decided against the PRE-EXISTING record, before the
        # fresh CapEvaluation overwrites it. See the class docstring: an
        # explicit ``wrote_live_verdict`` flag, not an inference from the
        # record's existence, because PromoteAsRefutedOperation can certify
        # a verdict between two cap firings.
        previous = claim.cap_evaluation
        cap_owns_live_verdict = (
            previous is not None
            and previous.wrote_live_verdict
            and claim.stage == ClaimStage.HYPOTHESIS
        )
        may_write = claim.integrated_assessment is None or cap_owns_live_verdict
        may_clear = cap_owns_live_verdict

        weight_label: Literal["strong", "moderate", "weak", "conflicting"] = cast(
            Literal["strong", "moderate", "weak", "conflicting"], weight.label
        )
        verdict: Literal["supports", "contradicts"] | None = None
        confidence: float | None = None
        if weight_label in ("strong", "moderate"):
            # The label algebra guarantees strict dominance here: an equal
            # split with enough mass reads as ``conflicting``, and an equal
            # split without it reads as ``weak``.
            dominant = max(weight.supporting_mass, weight.contradicting_mass)
            minority = min(weight.supporting_mass, weight.contradicting_mass)
            verdict = (
                "supports"
                if weight.supporting_mass > weight.contradicting_mass
                else "contradicts"
            )
            purity = (dominant - minority) / (dominant + minority)
            mass_adequacy = min(1.0, dominant / SCRUTINY_STRONG_MASS_THRESHOLD)
            confidence = min(MECHANICAL_VERDICT_MAX_CONFIDENCE, purity * mass_adequacy)

        wrote_verdict = verdict is not None and may_write
        claim.cap_evaluation = CapEvaluation(
            weight_label=weight_label,
            supporting_mass=weight.supporting_mass,
            contradicting_mass=weight.contradicting_mass,
            n_evidence=len(evidence),
            verdict=verdict,
            confidence=confidence,
            wrote_live_verdict=wrote_verdict,
        )

        if wrote_verdict:
            claim.integrated_assessment = verdict
            claim.integrated_confidence = confidence
            claim.integrated_reasoning = (
                f"Concluded at the scrutiny-resolve cycle cap: the "
                f"deterministic evidence reading was {weight_label} "
                f"({weight.supporting_mass:.2f} supporting vs "
                f"{weight.contradicting_mass:.2f} contradicting mass over "
                f"{len(evidence)} eligible item(s)), so the verdict is "
                f"{verdict}. Derived mechanically from the directional "
                f"balance — the IBE chain did not run on this claim "
                f"because its inquiry did not converge."
            )
        elif verdict is None and may_clear:
            # Non-directional reading, and the cap owns the live verdict.
            # Clear what THIS operation wrote on an earlier firing, so its
            # record and its verdict can never disagree. A verdict owned by
            # anyone else is left strictly alone.
            claim.integrated_assessment = None
            claim.integrated_confidence = None
            claim.integrated_reasoning = None

        await self.repo.save(claim)

        await self.log_event(
            "claim_concluded_at_cycle_cap",
            claim.entity_id,
            {
                "weight_label": weight_label,
                "supporting_mass": weight.supporting_mass,
                "contradicting_mass": weight.contradicting_mass,
                "n_evidence": len(evidence),
                "verdict": verdict,
                "confidence": confidence,
                "wrote_live_verdict": wrote_verdict,
                "cap_owns_live_verdict": cap_owns_live_verdict,
            },
        )

        if wrote_verdict:
            # ``confidence`` is non-None whenever ``verdict`` is (the
            # CapEvaluation validator enforces both-or-neither), but format
            # defensively rather than asserting.
            confidence_text = "n/a" if confidence is None else f"{confidence:.2f}"
            message = (
                f"[{claim.statement[:60]}] cap verdict: {verdict} "
                f"(conf={confidence_text}, weight={weight_label})"
            )
        elif verdict is not None:
            message = (
                f"[{claim.statement[:60]}] cap reading {verdict} recorded; "
                f"existing verdict {claim.integrated_assessment} kept "
                "(not written by the cap)"
            )
        elif claim.integrated_assessment is not None:
            message = (
                f"[{claim.statement[:60]}] cap reading was {weight_label} — "
                f"not directional; existing verdict "
                f"{claim.integrated_assessment} kept (not written by the cap)"
            )
        else:
            message = (
                f"[{claim.statement[:60]}] cap reading was {weight_label} — "
                "not directional, no verdict emitted"
            )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=message,
            did_work=wrote_verdict,
        )


class SoftPromoteOperation(BaseOperation):
    """Promote a HYPOTHESIS claim to SUPPORTED with integrated_assessment="insufficient".

    The middle path between PromoteAsRefutedOperation (strong contradicting
    evidence) and AbandonStaleClaimOperation (no signal at all). Used when
    refute-promotion declined but the linked evidence still carries
    directional judgments — promotes the claim to SUPPORTED so it can
    receive the same abductive deliberation (the IBE chain) as any other
    SUPPORTED claim, instead of being erased via abandonment.

    The verdict is NOT pre-set here. SoftPromote is a routing decision
    ("let this claim through the gate"), not an integration verdict.
    Earlier versions of this operation did pre-set
    integrated_assessment="insufficient" — that was a sensible
    optimization when integration was a single rubber-stamp LLM call,
    but it became wrong after the IBE 4-stage refactor (commit
    3affc1f), where the integration step produces qualitatively richer
    output than a hard-coded label. Setting integrated_assessment here
    short-circuits the IBE chain (whose entry condition is
    integrated_assessment is None) and discards real reasoning.

    Returns success=False when n_supports + n_contradicts == 0 so the graph
    falls through to abandon, which is the honest terminal when there is
    genuinely nothing to say.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import count_support_contradict

        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )
        if claim.stage != ClaimStage.HYPOTHESIS:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message=f"Soft-promote only runs on HYPOTHESIS, not {claim.stage.value}",
                did_work=False,
            )

        n_sup, n_con = await count_support_contradict(claim, self.repo)
        if n_sup + n_con == 0:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="No directional evidence; abandonment is appropriate",
                did_work=False,
            )

        claim.record_promotion(
            claim.stage,
            ClaimStage.SUPPORTED,
            justification=(
                f"Soft-promoted: refute threshold not met "
                f"({n_con} contradicts vs {n_sup} supports). "
                f"Promoted to SUPPORTED so the IBE chain can produce a "
                f"calibrated verdict on the directional evidence."
            ),
        )
        # integrated_assessment / integrated_confidence / integrated_reasoning
        # are deliberately left as None. The IBE chain (EnumerateCandidates
        # → ScoreLoveliness → ScoreLikeliness → SelectBestExplanation) will
        # populate them based on the actual evidence pattern.
        await self.repo.save(claim)

        await self.log_event(
            "claim_soft_promoted",
            claim.entity_id,
            {"n_supports": n_sup, "n_contradicts": n_con},
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=(
                f"[{claim.statement[:60]}] → supported "
                f"(soft-promote, deferring verdict to IBE: {n_sup}✓ / {n_con}⊥)"
            ),
        )
