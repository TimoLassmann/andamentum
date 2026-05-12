"""Investigation, prediction, and decision recording operations.

The investigation loop (Peirce inquiry cycling) identifies evidence gaps
when scrutiny produces doubt and creates targeted Evidence stubs.
GeneratePredictionOperation produces testable predictions from robust claims.
RecordDecisionOperation creates Decision entities for actionable claims.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim, Evidence, Decision, Objective entities
"""

from .base import BaseOperation, OperationInput, OperationResult
from .dispatch_gather import dispatch_and_persist_for_text
from ..thresholds import PEIRCE_CYCLE_CAP

from ..entities import (
    Claim,
    ClaimStage,
    Decision,
    Evidence,
    Objective,
)


class GeneratePredictionOperation(BaseOperation):
    """Generate testable predictions from robust claims.

    Predictions are a hallmark of good epistemology (Lakatos: progressive
    research programmes make novel predictions).

    Decomposed into 4 narrow agent steps per prediction:
    1. identify_testable_aspect — finds one testable dimension (run K times)
    2. specify_prediction — specifies prediction details for one aspect
    3. define_falsification — defines what would disprove the prediction
    4. classify_prediction — classifies type and specificity (existing agent)
    """

    entity_type = "claim"
    NUM_ASPECTS = 3  # Number of testable aspects to generate per claim

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.predictions_generated:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Predictions already generated",
            )

        if self.agent_runner:
            # Gather supporting evidence summary for grounded predictions
            evidence_summary = ""
            for eid in claim.evidence_ids:
                ev = await self.repo.get("evidence", eid)
                if isinstance(ev, Evidence) and ev.extracted_content:
                    evidence_summary += ev.extracted_content + "\n"

            # Step 1: Identify testable aspects (narrow agent, run K times)
            # Each call sees previously found aspects to avoid duplicates
            aspects = []
            for i in range(self.NUM_ASPECTS):
                if aspects:
                    prev_text = "\n".join(f"- {a.testable_dimension}" for a in aspects)
                else:
                    prev_text = "(none yet)"

                aspect_result = await self.run_agent(
                    "epistemic_identify_testable_aspect",
                    claim=claim.statement,
                    evidence_summary=evidence_summary
                    if evidence_summary
                    else "[No evidence available]",
                    aspect_number=i + 1,
                    previously_identified=prev_text,
                )
                aspects.append(aspect_result)

            # Steps 2-4: For each aspect, specify -> falsify -> classify
            for aspect in aspects:
                # Step 2: Specify prediction details
                spec_result = await self.run_agent(
                    "epistemic_specify_prediction",
                    testable_dimension=aspect.testable_dimension,
                    claim=claim.statement,
                )

                # Step 3: Define falsification criterion
                fals_result = await self.run_agent(
                    "epistemic_define_falsification",
                    prediction=spec_result.expected_observation,
                    conditions=spec_result.conditions,
                    timeframe=spec_result.timeframe,
                )

                # Step 4: Classify prediction (existing narrow agent)
                class_result = await self.run_agent(
                    "epistemic_classify_prediction",
                    prediction_statement=spec_result.expected_observation,
                    claim_statement=claim.statement,
                )

                # Phase 6 (deferred) of the Move-3 plan: typed Prediction
                # replaces the previous list-of-dicts shape.
                from ..entities.prediction import Prediction

                claim.predictions.append(
                    Prediction(
                        statement=spec_result.expected_observation,
                        type=class_result.prediction_type,
                        specificity=float(class_result.specificity),
                        success_criteria=spec_result.expected_observation,
                        failure_criteria=fals_result.falsification_criterion,
                        time_horizon=spec_result.timeframe,
                        conditions=spec_result.conditions,
                        measurability=spec_result.measurability,
                        observation_type=aspect.observation_type,
                    )
                )

        claim.predictions_generated = True
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Generated {len(claim.predictions)} predictions",
        )


class RecordDecisionOperation(BaseOperation):
    """Record a decision based on an actionable claim.

    Creates a Decision entity linked to the claim. Each actionable claim
    gets at most one decision (guarded by decision_recorded flag).
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.decision_recorded:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Decision already recorded",
            )

        if claim.stage != ClaimStage.ACTIONABLE:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message=f"Claim at {claim.stage.value}, not ACTIONABLE",
            )

        statement = f"Accept claim: {claim.statement}"
        justification = (
            f"Claim reached ACTIONABLE stage with {claim.evidence_count} evidence items"
        )

        if self.agent_runner:
            # Build context: objective + all claims for multi-claim reasoning
            objective_description = ""
            all_claims_text: list[str] = []
            if claim.objective_id:
                obj = await self.repo.get("objective", claim.objective_id)
                if isinstance(obj, Objective):
                    objective_description = obj.description
                # Load all claims for this objective
                all_claims = await self.repo.query(
                    "claim", objective_id=claim.objective_id
                )
                for i, c in enumerate(all_claims):
                    if isinstance(c, Claim) and not c.abandoned:
                        marker = (
                            " <- (this claim)" if c.entity_id == claim.entity_id else ""
                        )
                        all_claims_text.append(
                            f"[{i}] [{c.stage.value}] {c.statement} "
                            f"(evidence: {c.evidence_count}, confidence: {c.confidence_score or 'N/A'}){marker}"
                        )

            result = await self.run_agent(
                "epistemic_record_decision",
                objective_description=objective_description or claim.statement,
                available_claims="\n".join(all_claims_text)
                if all_claims_text
                else f"[0] {claim.statement}",
                claim_count=len(all_claims_text) or 1,
                decision_context=f"Claim '{claim.statement}' has reached ACTIONABLE stage with {claim.evidence_count} evidence items.",
            )
            statement = result.statement
            justification = result.justification

        decision = Decision(
            objective_id=claim.objective_id,
            statement=statement,
            justification=justification,
            claim_ids=[claim.entity_id],
        )
        await self.repo.save(decision)

        claim.decision_recorded = True
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=decision.entity_id,
            message=f"Decision recorded: {statement[:50]}",
            created_entities=[decision.entity_id],
        )


class InvestigateClaimOperation(BaseOperation):
    """Follow-up evidence gather when scrutiny produces doubt.

    Two-layer design:

    1. **Gap analysis.** ``epistemic_investigate_claim`` reads the
       claim, the unresolved scrutiny issues, and the intents proposed
       in earlier rounds, and produces 1-3 natural-language **intents**
       describing fresh evidence-search angles. The agent is held
       responsible for proposing angles that differ from prior intents
       (the prompt makes the previous-intents list visible and
       explicitly instructs against paraphrase).

    2. **Routing.** Each intent is funnelled through
       ``dispatch_and_persist_for_text`` — the same routing layer used
       by the initial gather (``DispatchGatherOperation``). The
       description-driven dispatch agent decides per-provider whether
       to commit or abstain on each intent, and persisted Evidence
       inherits the claim's ``sub_investigation_id`` plus
       ``depends_on_claim_id=claim.entity_id``.

    The previous design (queries-with-source-types + a separate
    ranker that silently overrode the agent's provider choice) was
    decommissioned alongside the legacy gather chain. There is now
    exactly one routing implementation; investigation is purely an
    upstream gap-analysis step that feeds it.

    After ``PEIRCE_CYCLE_CAP`` rounds (in graph state — this operation
    is the per-claim safety duplicate), the claim is abandoned.
    """

    entity_type = "claim"

    def __init__(
        self,
        *args: object,
        providers: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._dispatch_providers: dict[str, object] = providers or {}

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.abandoned:
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="Claim already abandoned",
            )

        # Per-claim Peirce-cycling safety. The graph-state cap is the
        # primary bound; this is defence in depth.
        if claim.investigation_count >= PEIRCE_CYCLE_CAP:
            if claim.scrutiny_verdict == "needs_resolution":
                claim.scrutiny_verdict = "fail"
            claim.abandoned = True
            await self.repo.save(claim)
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message=(
                    f"Investigation exhausted after "
                    f"{claim.investigation_count} attempts, claim abandoned"
                ),
            )

        if not self.agent_runner:
            raise RuntimeError(
                "InvestigateClaimOperation requires an agent_runner — "
                "the gap-analysis agent is the upstream cognitive step."
            )

        if not self._dispatch_providers:
            raise RuntimeError(
                "InvestigateClaimOperation requires a non-empty providers "
                "dict (passed through EpistemicDeps.providers). The "
                "routing layer dispatches each intent to all providers."
            )

        # Existing evidence summaries — only items with extracted content
        # contribute. Invalidated stubs (provider returned nothing on a
        # prior round) are honestly absent from this view.
        evidence_summaries: list[str] = []
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if isinstance(ev, Evidence) and ev.extracted_content:
                evidence_summaries.append(f"[{ev.source_type}] {ev.extracted_content}")

        # Unresolved scrutiny issues only — resolved uncertainties are
        # filtered (otherwise the agent re-targets gaps already closed,
        # see d9bcf1f).
        scrutiny_issues: list[str] = []
        uncertainties = await self.repo.query(
            "uncertainty",
            objective_id=claim.objective_id,
        )
        for u in uncertainties:
            if u.is_resolved:
                continue
            if claim.entity_id in u.affected_claim_ids and u.description:
                scrutiny_issues.append(str(u.description))

        # Prior-round intents — the memory the previous agent didn't have.
        previous_intents = list(claim.investigation_intents)
        previous_intents_text = (
            "\n".join(f"- {intent}" for intent in previous_intents)
            if previous_intents
            else "(none — this is the first investigation round)"
        )

        gap_result = await self.run_agent(
            "epistemic_investigate_claim",
            claim_statement=claim.statement,
            claim_scope=claim.scope,
            existing_evidence="\n".join(evidence_summaries)
            if evidence_summaries
            else "No evidence gathered yet",
            scrutiny_issues="\n".join(scrutiny_issues)
            if scrutiny_issues
            else "No specific issues recorded",
            previous_intents=previous_intents_text,
            scrutiny_verdict=claim.scrutiny_verdict or "unknown",
        )

        new_intents = [s.strip() for s in gap_result.intents if s.strip()]

        # Route each intent through the shared description-driven
        # dispatch path — the same machinery initial gather uses.
        core_runner: object = getattr(
            self.agent_runner, "core_runner", self.agent_runner
        )

        created_entities: list[str] = []
        for intent in new_intents:
            ev_ids = await dispatch_and_persist_for_text(
                self,
                intent,
                objective_id=claim.objective_id,
                providers=self._dispatch_providers,  # type: ignore[arg-type]
                core_runner=core_runner,
                sub_investigation_id=claim.sub_investigation_id,
                depends_on_claim_id=claim.entity_id,
                created_by="investigate_claim",
            )
            created_entities.extend(ev_ids)
            claim.evidence_ids.extend(ev_ids)
            claim.investigation_intents.append(intent)

        claim.investigation_count += 1
        claim.evidence_count = len(claim.evidence_ids)
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=(
                f"Investigation #{claim.investigation_count}: "
                f"{len(new_intents)} intent(s) → "
                f"{len(created_entities)} evidence item(s)"
            ),
            created_entities=created_entities,
        )
