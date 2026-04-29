"""Abductive integration operation (Peirce + Kahneman + Wimsatt).

Takes ALL evidence for a claim — including no_bearing items — along with
adversarial search outcome and open uncertainties. Produces a holistic
IntegrationAssessment that captures cross-evidence reasoning.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim entities
"""

from .base import BaseOperation, OperationInput, OperationResult
from ..entities import Claim, Evidence, Uncertainty


class AbductiveIntegrationOperation(BaseOperation):
    """Holistic evidence integration (Peirce abduction + Kahneman aggregation)."""

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.integrated_assessment is not None:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already integrated",
            )

        if not self.agent_runner:
            # No agent runner — skip integration
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="Integration skipped (no agent runner)",
            )

        from .claims import LLM_PANEL_CAP, top_n_representatives

        # Build structured brief from investigation results. Filter to
        # representatives only (clustering already collapsed redundant
        # sources into corroborative groups) and cap at LLM_PANEL_CAP
        # highest-quality reps so the integration prompt stays bounded
        # regardless of how many clusters the evidence base produced.
        candidates: list[Evidence] = []
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if not isinstance(ev, Evidence) or ev.invalidated:
                continue
            if ev.cluster_status in ("corroborative", "deferred"):
                continue
            candidates.append(ev)

        supports_items: list[str] = []
        contradicts_items: list[str] = []
        no_bearing_items: list[str] = []
        for ev in top_n_representatives(candidates, LLM_PANEL_CAP):
            content = ev.extracted_content or ""
            cluster_size = max(1, getattr(ev, "corroboration_count", 1) or 1)
            # cluster_size lets the agent apply Mill's method-of-difference
            # reasoning directly: a representative standing for many similar
            # sources is redundant confirmation, not independent evidence.
            summary = (
                f"[{ev.source_type}, cluster_size={cluster_size}] {content}"
            )
            if ev.support_judgment == "supports":
                supports_items.append(summary)
            elif ev.support_judgment == "contradicts":
                contradicts_items.append(summary)
            else:
                no_bearing_items.append(summary)

        # Adversarial outcome
        adversarial_text = "Adversarial search has NOT been conducted."
        if claim.adversarial_checked:
            if claim.adversarial_balance is not None:
                if claim.adversarial_balance >= 0.7:
                    adversarial_text = (
                        f"Adversarial search conducted: NO strong counterevidence "
                        f"found (balance: {claim.adversarial_balance:.2f}). "
                        f"The claim survived active refutation attempts."
                    )
                else:
                    adversarial_text = (
                        f"Adversarial search found significant counterevidence "
                        f"(balance: {claim.adversarial_balance:.2f})."
                    )

        # Open uncertainties
        uncertainties = await self.repo.query(
            "uncertainty",
            objective_id=claim.objective_id,
        )
        open_blocking = [
            u
            for u in uncertainties
            if isinstance(u, Uncertainty)
            and claim.entity_id in u.affected_claim_ids
            and u.resolution is None
            and u.is_blocking
        ]
        unc_text = (
            "\n".join(f"- {u.description}" for u in open_blocking)
            if open_blocking
            else "No unresolved blocking uncertainties."
        )

        # Run integration agent
        result = await self.run_agent(
            "epistemic_integrate_evidence",
            claim_statement=claim.statement,
            claim_scope=claim.scope,
            supporting_evidence="\n\n".join(supports_items) or "None found.",
            contradicting_evidence="\n\n".join(contradicts_items) or "None found.",
            no_bearing_evidence="\n\n".join(no_bearing_items) or "None.",
            adversarial_outcome=adversarial_text,
            open_uncertainties=unc_text,
            evidence_count=len(supports_items)
            + len(contradicts_items)
            + len(no_bearing_items),
            supporting_count=len(supports_items),
            contradicting_count=len(contradicts_items),
            no_bearing_count=len(no_bearing_items),
        )

        # Store assessment
        claim.integrated_assessment = result.verdict
        claim.integrated_confidence = float(result.confidence)
        claim.integrated_reasoning = result.reasoning
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Integration: {result.verdict} (confidence {result.confidence:.2f})",
        )
