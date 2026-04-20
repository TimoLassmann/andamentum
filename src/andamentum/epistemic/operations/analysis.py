"""Argument analysis and pairwise claim comparison operations.

AnalyzeArgumentOperation assesses reasoning quality (hidden assumptions,
argument strength, logical structure). ContrastiveEvaluationOperation and
CrossClaimConsistencyOperation perform pairwise comparisons between
sibling claims under the same objective.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim, Evidence, Uncertainty entities
"""

from .base import BaseOperation, OperationInput, OperationResult

from ..entities import (
    Claim,
    Uncertainty,
    UncertaintyType,
)


class AnalyzeArgumentOperation(BaseOperation):
    """Analyze argument structure and quality of a claim.

    Distinct from deductive validation: argument analysis assesses the
    *quality of reasoning* (hidden assumptions, argument strength, logical
    structure) rather than formal logical consistency.

    Creates uncertainties for weak arguments.
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

        if claim.argument_analyzed:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already analyzed",
            )

        if self.agent_runner:
            result = await self.run_agent(
                "epistemic_analyze_argument",
                claim=claim.statement,
                scope=claim.scope,
                assumptions=claim.assumptions,
            )

            # Create uncertainties for detected fallacies
            for fallacy in result.fallacies:
                uncertainty = Uncertainty(
                    objective_id=claim.objective_id,
                    uncertainty_type=UncertaintyType.MISSING_PREMISE,
                    description=str(fallacy),
                    affected_claim_ids=[claim.entity_id],
                )
                await self.repo.save(uncertainty)

        claim.argument_analyzed = True
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] argument structure analyzed",
        )


class ContrastiveEvaluationOperation(BaseOperation):
    """Pairwise comparison of competing claims under the same objective.

    For each sibling claim at the same stage, calls the contrastive evaluation
    agent to determine which better explains the evidence. If a claim is
    clearly inferior, creates a blocking uncertainty.

    Used by explanatory and comparative question types.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if claim.contrastive_checked:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Contrastive evaluation already completed",
            )

        if not self.agent_runner:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="No agent runner available",
            )

        # Find sibling claims at same stage, non-abandoned
        siblings = await self.repo.query("claim", objective_id=claim.objective_id)
        siblings = [
            s
            for s in siblings
            if s.entity_id != claim.entity_id
            and s.stage == claim.stage
            and not s.abandoned
        ]

        created_entities: list[str] = []
        inferior_count = 0

        for sibling in siblings:
            # Gather shared evidence text
            shared_evidence_ids = set(claim.evidence_ids) & set(sibling.evidence_ids)
            evidence_texts = []
            for eid in shared_evidence_ids:
                try:
                    ev = await self.repo.get("evidence", eid)
                    if ev.extracted_content:
                        evidence_texts.append(ev.extracted_content)
                except Exception:
                    pass

            shared_evidence = (
                "\n---\n".join(evidence_texts)
                if evidence_texts
                else "No shared evidence"
            )

            result = await self.run_agent(
                "epistemic_contrastive_evaluation",
                claim_a=claim.statement,
                claim_b=sibling.statement,
                shared_evidence=shared_evidence,
            )

            # If this claim is clearly inferior (result says B is better with high confidence)
            if result.better_claim == "B" and result.confidence >= 0.7:
                inferior_count += 1
                uncertainty = Uncertainty(
                    objective_id=claim.objective_id,
                    uncertainty_type=UncertaintyType.SCOPE_DIFFERENCE,
                    description=f"Contrastive evaluation: claim is inferior to '{sibling.statement[:80]}'. Distinguishing observation: {result.distinguishing_observation}",
                    affected_claim_ids=[claim.entity_id],
                    is_blocking=True,
                    created_by="contrastive_evaluation",
                )
                await self.repo.save(uncertainty)
                created_entities.append(uncertainty.entity_id)

            # When claims are equally supported, preserve the distinguishing
            # observation as a non-blocking caveat. This surfaces what evidence
            # WOULD separate them — valuable for the report and future work.
            elif result.better_claim == "neither" and result.distinguishing_observation:
                uncertainty = Uncertainty(
                    objective_id=claim.objective_id,
                    uncertainty_type=UncertaintyType.SCOPE_DIFFERENCE,
                    description=f"Contrastive parity with '{sibling.statement[:80]}': {result.distinguishing_observation}",
                    affected_claim_ids=[claim.entity_id, sibling.entity_id],
                    is_blocking=False,
                    created_by="contrastive_evaluation",
                )
                await self.repo.save(uncertainty)
                created_entities.append(uncertainty.entity_id)

        claim.contrastive_checked = True
        await self.repo.save(claim)

        msg = f"Contrastive evaluation complete: compared with {len(siblings)} siblings"
        if inferior_count > 0:
            msg += f", found inferior in {inferior_count} comparisons"

        return OperationResult(
            success=True,
            entity_id=work.entity_id,
            message=msg,
            created_entities=created_entities,
        )


class CrossClaimConsistencyOperation(BaseOperation):
    """Pairwise consistency check between claims under the same objective.

    For each sibling claim at the same stage, checks whether the claims
    contradict each other. If a conflict is found, creates blocking
    uncertainties on both claims.

    Used by exploratory, comparative, compositional, and normative question types.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if claim.consistency_checked:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Consistency check already completed",
            )

        if not self.agent_runner:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="No agent runner available",
            )

        # Find sibling claims at same stage, non-abandoned
        siblings = await self.repo.query("claim", objective_id=claim.objective_id)
        siblings = [
            s
            for s in siblings
            if s.entity_id != claim.entity_id
            and s.stage == claim.stage
            and not s.abandoned
        ]

        created_entities: list[str] = []
        conflict_count = 0

        for sibling in siblings:
            result = await self.run_agent(
                "epistemic_cross_claim_consistency",
                claim_a=claim.statement,
                claim_b=sibling.statement,
            )

            if result.conflicts:
                conflict_count += 1
                # Create blocking uncertainty on THIS claim
                uncertainty = Uncertainty(
                    objective_id=claim.objective_id,
                    uncertainty_type=UncertaintyType.CONTRADICTION,
                    description=f"Cross-claim conflict with '{sibling.statement[:80]}': {result.tension_point}",
                    affected_claim_ids=[claim.entity_id, sibling.entity_id],
                    is_blocking=True,
                    created_by="cross_claim_consistency",
                )
                await self.repo.save(uncertainty)
                created_entities.append(uncertainty.entity_id)

        claim.consistency_checked = True
        await self.repo.save(claim)

        msg = f"[{claim.statement[:60]}] consistent with {len(siblings)} sibling claims"
        if conflict_count > 0:
            msg = f"[{claim.statement[:60]}] {conflict_count} conflicts with siblings"

        return OperationResult(
            success=True,
            entity_id=work.entity_id,
            message=msg,
            created_entities=created_entities,
        )
