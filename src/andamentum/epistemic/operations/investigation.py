"""Investigation, prediction, and decision recording operations.

The investigation loop (Peirce inquiry cycling) identifies evidence gaps
when scrutiny produces doubt and creates targeted Evidence stubs.
GeneratePredictionOperation produces testable predictions from robust claims.
RecordDecisionOperation creates Decision entities for actionable claims.

Depends on: base (BaseOperation, OperationResult, MAX_INVESTIGATION_ATTEMPTS)
Operates on: Claim, Evidence, Decision, Objective entities
"""

from .base import BaseOperation, MAX_INVESTIGATION_ATTEMPTS, OperationInput, OperationResult

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
            for eid in claim.evidence_ids[:5]:
                ev = await self.repo.get("evidence", eid)
                if isinstance(ev, Evidence) and ev.extracted_content:
                    evidence_summary += ev.extracted_content + "\n"

            # Step 1: Identify testable aspects (narrow agent, run K times)
            # Each call sees previously found aspects to avoid duplicates
            aspects = []
            for i in range(self.NUM_ASPECTS):
                if aspects:
                    prev_text = "\n".join(
                        f"- {a.testable_dimension}" for a in aspects
                    )
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

                prediction_dict = {
                    "statement": spec_result.expected_observation,
                    "type": class_result.prediction_type,
                    "specificity": float(class_result.specificity),
                    "success_criteria": spec_result.expected_observation,
                    "failure_criteria": fals_result.falsification_criterion,
                    "time_horizon": spec_result.timeframe,
                    "conditions": spec_result.conditions,
                    "measurability": spec_result.measurability,
                    "observation_type": aspect.observation_type,
                }
                claim.predictions.append(prediction_dict)

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
                            " <- (this claim)"
                            if c.entity_id == claim.entity_id
                            else ""
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
    """Investigate evidence gaps when scrutiny produces doubt (Peirce inquiry cycling).

    When scrutiny returns "needs_resolution" or "fail" at HYPOTHESIS,
    this operation identifies what evidence is missing and creates
    targeted Evidence stubs for the existing extraction infrastructure.

    After MAX_INVESTIGATION_ATTEMPTS, the claim is abandoned.
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

        if claim.abandoned:
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="Claim already abandoned",
            )

        # Check investigation limit
        if claim.investigation_count >= MAX_INVESTIGATION_ATTEMPTS:
            # Exhausted — force to fail and abandon
            if claim.scrutiny_verdict == "needs_resolution":
                claim.scrutiny_verdict = "fail"
            claim.abandoned = True
            await self.repo.save(claim)

            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message=f"Investigation exhausted after {claim.investigation_count} attempts, claim abandoned",
            )

        # Gather context for the investigation agent
        # Load existing evidence
        evidence_summaries: list[str] = []
        source_types_seen: set[str] = set()
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if isinstance(ev, Evidence) and ev.extracted_content:
                evidence_summaries.append(
                    f"[{ev.source_type}] {ev.extracted_content}"
                )
                source_types_seen.add(ev.source_type)

        # Load scrutiny issues from uncertainties
        scrutiny_issues: list[str] = []
        uncertainties = await self.repo.query(
            "uncertainty",
            objective_id=claim.objective_id,
        )
        for u in uncertainties:
            affected = u.affected_claim_ids
            if claim.entity_id in affected:
                desc = u.description
                if desc:
                    scrutiny_issues.append(str(desc))

        # Run investigation agent
        created_entities: list[str] = []

        if self.agent_runner:
            result = await self.run_agent(
                "epistemic_investigate_claim",
                claim_statement=claim.statement,
                claim_scope=claim.scope,
                existing_evidence="\n".join(evidence_summaries)
                if evidence_summaries
                else "No evidence gathered yet",
                scrutiny_issues="\n".join(scrutiny_issues)
                if scrutiny_issues
                else "No specific issues recorded",
                available_source_types=", ".join(sorted(source_types_seen))
                if source_types_seen
                else "openalex, web_search",
                scrutiny_verdict=claim.scrutiny_verdict or "unknown",
            )

            # Create evidence stubs from agent output
            for eq in result.evidence_queries:
                source_type = (
                    eq.get("source_type", "web_search")
                    if isinstance(eq, dict)
                    else "web_search"
                )
                query = eq.get("query", "") if isinstance(eq, dict) else str(eq)
                if not query:
                    continue

                evidence_stub = Evidence(
                    objective_id=claim.objective_id,
                    source_type=source_type,
                    source_ref=query,
                    extracted=False,
                    created_by="investigate_claim",
                    depends_on_claim_id=claim.entity_id,
                )
                await self.repo.save(evidence_stub)

                # Link to claim
                claim.evidence_ids.append(evidence_stub.entity_id)
                created_entities.append(evidence_stub.entity_id)

        claim.investigation_count += 1
        claim.evidence_count = len(claim.evidence_ids)

        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Investigation #{claim.investigation_count}: created {len(created_entities)} evidence stubs",
            created_entities=created_entities,
        )
