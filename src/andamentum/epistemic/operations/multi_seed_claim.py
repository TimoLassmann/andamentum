"""Multi-Seed-Claim Operation — mint N Claims from a parent objective's
decomposition.

Counterpart to SeedClaimOperation (one claim from claim_to_verify) and
ProposeClaimsOperation (N claims discovered from evidence). This operation
fires when ``objective.decomposition`` is set and mints one Claim per
sub-investigation in the decomposition.

Each minted Claim links only the Evidence whose ``sub_investigation_id``
matches it — the per-claim evidence pool that PlanTaskOperation set up
in multi-seed-claim mode. This avoids the support_judgment-collision
problem: ``Evidence.support_judgment`` is a single scalar, so an Evidence
item linked to claim A only ever needs one verdict (vs. A), not multiple
conflicting ones.

This is the architectural collapse of v0.2 spawning into the v0.1 multi-
claim shape: instead of N child Objectives each running their own graph
with one seed claim, ONE Objective hosts N Claims and the v0.1 multi-
claim machinery (Scrutinize, Investigate, IBE, etc.) handles them.

Architecture: Layer 1 (framework-agnostic)
"""

from __future__ import annotations

from ..entities.claim import Claim, ClaimStage
from ..entities.evidence import Evidence
from ..entities.objective import Objective
from .base import BaseOperation, OperationInput, OperationResult


class MultiSeedClaimOperation(BaseOperation):
    """Mint N Claims from objective.decomposition.sub_investigations.

    Fires from the CreateClaims graph node's third branch (when
    ``objective.decomposition`` is set and ``claim_to_verify`` is not).

    Per-claim evidence linkage: each minted Claim's ``evidence_ids`` is
    populated only with Evidence whose ``sub_investigation_id`` matches.
    Then each (claim, evidence) pair is judged via the same judge agent
    used by SeedClaim/ProposeClaims, with the verdict stored on the
    Evidence item — safe because each Evidence is linked to exactly one
    Claim under this scheme.

    Idempotent: re-running on a parent that already has claims for some
    sub-investigations skips those and only mints the missing ones.
    Designed to compose with reflection-driven decomposition growth (the
    decomposition can gain new sub-investigations between calls).
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
                did_work=False,
            )

        if not objective.decomposition:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="No decomposition set on objective",
                did_work=False,
            )

        # Phase 6 of the Move-3 plan: typed Decomposition access.
        sub_investigations = objective.decomposition.sub_investigations
        if not sub_investigations:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Decomposition has no sub_investigations",
                did_work=False,
            )

        # Find existing claims so we skip already-minted sub-investigations.
        existing_claims = await self.repo.query(
            "claim", objective_id=objective.entity_id
        )
        existing_sub_ids: set[str] = set()
        for c in existing_claims:
            if isinstance(c, Claim) and c.sub_investigation_id:
                existing_sub_ids.add(c.sub_investigation_id)

        # All evidence for this objective. We'll filter per-sub-investigation
        # below using the sub_investigation_id tag PlanTaskOperation wrote.
        all_evidence_raw = await self.repo.query(
            "evidence", objective_id=objective.entity_id
        )
        all_evidence: list[Evidence] = [
            ev for ev in all_evidence_raw if isinstance(ev, Evidence)
        ]

        from ..judge import judge_evidence as _judge

        created: list[str] = []
        judged_total = 0
        skipped_existing = 0

        for sub in sub_investigations:
            sub_id = sub.id
            if not sub_id:
                continue
            if sub_id in existing_sub_ids:
                skipped_existing += 1
                continue

            seed_claim_text = sub.seed_claim
            rationale = sub.rationale
            if not seed_claim_text:
                continue

            # Per-claim evidence pool: only evidence whose
            # sub_investigation_id matches this claim. Extracted-only;
            # un-extracted stubs aren't useful for scrutiny.
            claim_evidence = [
                ev
                for ev in all_evidence
                if ev.sub_investigation_id == sub_id and ev.extracted
            ]
            evidence_ids = [ev.entity_id for ev in claim_evidence]

            claim = Claim(
                objective_id=objective.entity_id,
                statement=seed_claim_text,
                scope=rationale or "Sub-investigation seed",
                stage=ClaimStage.HYPOTHESIS,
                sub_investigation_id=sub_id,
                evidence_ids=evidence_ids,
                evidence_count=len(evidence_ids),
            )
            await self.repo.save(claim)
            created.append(claim.entity_id)

            # Judge each linked evidence vs. this specific claim. Each
            # Evidence is linked to exactly ONE claim under per-claim
            # pool semantics, so support_judgment-as-single-scalar is
            # correct here.
            #
            # Phase 2a of the efficiency plan: judges across the
            # claim_evidence pool are independent (each writes a
            # different Evidence entity), so they run concurrently
            # via asyncio.gather. The AgentRunner's global semaphore
            # bounds in-flight calls (defaults: 1 for Ollama, 8 for
            # cloud) so we don't hammer local servers.
            if self.agent_runner is not None:
                import asyncio

                runner = self.agent_runner  # narrow for closure
                evs_to_judge = [
                    ev
                    for ev in claim_evidence
                    if ev.support_judgment is None and ev.extracted_content
                ]

                async def _judge_one(ev: Evidence) -> None:
                    judgment = await _judge(
                        claim_statement=claim.statement,
                        claim_scope=claim.scope,
                        evidence_content=ev.extracted_content,
                        evidence_source=f"{ev.source_type}: {ev.source_ref}",
                        runner=runner,
                    )
                    ev.support_judgment = judgment.verdict
                    ev.judgment_reasoning = judgment.reasoning
                    await self.repo.save(ev)

                if evs_to_judge:
                    await asyncio.gather(*(_judge_one(ev) for ev in evs_to_judge))
                    judged_total += len(evs_to_judge)

        # Mark objective as having claims proposed (same contract as
        # SeedClaim / ProposeClaims) so CreateClaims can advance.
        if created:
            objective.claims_proposed = True
            objective.phase = "claims_proposed"
            await self.repo.save(objective)

        if not created and skipped_existing == len(sub_investigations):
            # All sub-investigations already had claims — idempotent no-op.
            return OperationResult(
                success=True,
                entity_id=objective.entity_id,
                message=(
                    f"All {skipped_existing} sub-investigations already have "
                    "claims; no new claims minted"
                ),
                did_work=False,
            )

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=(
                f"Multi-seed-claim minted {len(created)} claims "
                f"({judged_total} evidence items judged, "
                f"{skipped_existing} pre-existing sub-investigations skipped)"
            ),
            created_entities=created,
        )
