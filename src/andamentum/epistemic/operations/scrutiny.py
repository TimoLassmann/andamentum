"""Scrutiny operations (Phase 5).

Skeptic review of claims using a split-agent architecture: evidence
assessment + one-issue-at-a-time identification + deterministic
verdict combination. Creates Uncertainty entities for issues found.

Also contains ``_maybe_advance_phase``, which is shared by scrutiny,
stage_management, and uncertainty modules.

Depends on: base (BaseOperation, OperationResult), claims (select_top_k_evidence)
Operates on: Claim, Evidence, Uncertainty, Objective entities
"""

from typing import TYPE_CHECKING

from .base import BaseOperation, OperationResult
from .claims import select_top_k_evidence

from ..entities import (
    Claim,
    ClaimStage,
    Evidence,
    Objective,
    Uncertainty,
    UncertaintyType,
)
from ..patterns import WorkItem

if TYPE_CHECKING:
    from ..repository import EpistemicRepository


async def _maybe_advance_phase(repo: "EpistemicRepository", objective_id: str) -> None:
    """Advance objective from claims_proposed to claims_done when ready.

    Ready condition (all must be true):
    1. Objective is at phase "claims_proposed"
    2. All claims have been scrutinized (scrutiny_verdict is not None)
    3. No HYPOTHESIS claims with passing scrutiny (should have been promoted)
    4. No unresolved blocking uncertainties

    This is deterministic and idempotent.
    """
    objective = await repo.get("objective", objective_id)
    if not isinstance(objective, Objective) or objective.phase != "claims_proposed":
        return

    claims = await repo.query("claim", objective_id=objective_id)
    if not claims:
        return

    # All non-abandoned claims must have terminal scrutiny verdicts
    # Only "pass" and "fail" are terminal. None means not yet scrutinized;
    # "needs_resolution" means under investigation.
    for c in claims:
        if not isinstance(c, Claim):
            continue
        if c.abandoned:
            continue
        if c.scrutiny_verdict not in ("pass", "fail"):
            return  # Not ready — still under scrutiny or investigation

    # No non-abandoned HYPOTHESIS claims with passing scrutiny
    # (they should have been promoted already)
    for c in claims:
        if (
            isinstance(c, Claim)
            and not c.abandoned
            and c.stage == ClaimStage.HYPOTHESIS
            and c.scrutiny_verdict == "pass"
        ):
            return

    # No unresolved blocking uncertainties
    uncertainties = await repo.query(
        "uncertainty", objective_id=objective_id, resolution=None
    )
    for u in uncertainties:
        if u.is_blocking:
            return

    objective.phase = "claims_done"
    await repo.save(objective)


class ScrutiniseClaimOperation(BaseOperation):
    """Run skeptic review on a claim.

    Uses a split-agent architecture:
    1. epistemic_assess_evidence — evaluates evidence weight (focused judgment)
    2. epistemic_identify_single_issue — identifies issues one at a time (focused judgment)
    3. Deterministic combination — computes verdict from agent outputs

    Updates scrutiny_verdict to "pass", "fail", or "needs_resolution".
    May create Uncertainty entities for issues found.
    """

    entity_type = "claim"

    async def _gather_evidence_summaries(self, claim: Claim) -> list[str]:
        """Gather formatted evidence summaries for agent input."""
        evidence_summaries: list[str] = []
        for eid in claim.evidence_ids:
            try:
                ev = await self.repo.get("evidence", eid)
                if isinstance(ev, Evidence) and ev.extracted_content:
                    if ev.invalidated:
                        continue
                    # Skip corroborative/deferred evidence — only representatives carry to scrutiny
                    if ev.cluster_status in ("corroborative", "deferred"):
                        continue
                    quality_str = (
                        f", quality={ev.quality_score:.2f}" if ev.quality_score else ""
                    )
                    evidence_summaries.append(
                        f"[{ev.source_type}{quality_str}] {ev.source_ref}\n{ev.extracted_content}"
                    )
            except Exception:
                continue
        return evidence_summaries

    # Blocking types that remain blocking even when scrutiny passes.
    # These represent factual contradictions or logical impossibilities that
    # the scrutiny verdict cannot override.
    _ALWAYS_BLOCKING = frozenset(
        {"contradiction", "logical_inconsistency", "physical_implausibility"}
    )

    async def _handle_issues(
        self,
        claim: Claim,
        issues: list[str],
        issue_types: list[str],
        verdict: str = "pass",
    ) -> None:
        """Create uncertainties from issues, handling evidence_corrupted specially.

        issue_types is parallel to issues — each issue has a corresponding type.
        Non-blocking types create non-blocking uncertainties.
        "evidence_corrupted" invalidates evidence rather than creating uncertainty.

        When verdict="pass", blocking types that encode missing information
        ("unknown", "missing_premise") are downgraded to their non-blocking
        equivalents ("evidence_gap", "assumption"). The scrutiny verdict already
        encodes the judgment that the claim is acceptable despite these gaps —
        creating blocking uncertainties on top of a passing verdict would
        override the agent's judgment at the data layer.

        Truly problematic types (contradiction, logical_inconsistency,
        physical_implausibility) remain blocking regardless of verdict.
        """
        for i, issue in enumerate(issues):
            issue_type_str = issue_types[i] if i < len(issue_types) else "unknown"

            # When scrutiny passes, downgrade information-gap blocking types to
            # their non-blocking equivalents. The scrutiny judgment already
            # decided this is acceptable; double-blocking contradicts that.
            if verdict == "pass" and issue_type_str not in self._ALWAYS_BLOCKING:
                if issue_type_str == "unknown":
                    issue_type_str = "evidence_gap"
                elif issue_type_str == "missing_premise":
                    issue_type_str = "assumption"

            # Handle corrupted evidence: invalidate rather than create uncertainty
            if issue_type_str == "evidence_corrupted":
                for eid in claim.evidence_ids:
                    try:
                        ev = await self.repo.get("evidence", eid)
                        if (
                            isinstance(ev, Evidence)
                            and ev.extracted_content
                            and not ev.invalidated
                        ):
                            ev.invalidated = True
                            ev.invalidation_reason = str(issue)
                            await self.repo.save(ev)
                            break  # One invalidation per issue
                    except Exception:
                        continue
                continue

            # Map issue_type string to UncertaintyType enum
            try:
                uncertainty_type = UncertaintyType(issue_type_str)
            except ValueError:
                uncertainty_type = UncertaintyType.UNKNOWN

            uncertainty = Uncertainty(
                objective_id=claim.objective_id,
                uncertainty_type=uncertainty_type,
                description=str(issue),
                affected_claim_ids=[claim.entity_id],
            )
            await self.repo.save(uncertainty)

    MAX_ISSUES = 5  # Maximum per-evidence calls (caps cost for large evidence sets)

    async def _execute_split(
        self,
        claim: Claim,
        evidence_summaries: list[str],
    ) -> str:
        """Execute using split agents + deterministic combination.

        Per-evidence issue identification: each evidence item is checked
        independently against the claim (Kahneman independence principle).
        One additional call checks all evidence together for contradictions.
        All calls run in parallel — no shared context, no anchoring.

        Downstream DeduplicateConcernsOperation handles any duplicate
        uncertainties that result from multiple calls finding the same issue.

        Returns the computed verdict string.
        """
        import asyncio

        evidence_text = (
            "\n\n".join(evidence_summaries)
            if evidence_summaries
            else "[No evidence available]"
        )

        # Agent A: Assess evidence weight (flat output, 4 fields — reliable)
        # Sees ALL evidence for holistic weight assessment.
        assess_result = await self.run_agent(
            "epistemic_assess_evidence",
            claim_id=claim.entity_id,
            claim=claim.statement,
            scope=claim.scope,
            evidence=evidence_text,
            evidence_count=len(evidence_summaries),
        )

        # Agent B: Identify issues — one call per evidence item + one contradiction call.
        # Each per-evidence call sees only ONE evidence item (independent judgment).
        # The contradiction call sees all items but asks only about contradictions.
        found_issues: list[dict[str, object]] = []

        async def _check_single_evidence(
            evidence_item: str,
        ) -> dict[str, object] | None:
            try:
                result = await self.run_agent(
                    "epistemic_identify_single_issue",
                    claim=claim.statement,
                    scope=claim.scope,
                    evidence=evidence_item,
                )
                if result.has_issue:
                    return {
                        "description": result.description,
                        "issue_type": result.issue_type,
                        "reversal_test": result.reversal_test,
                    }
            except Exception:
                pass
            return None

        async def _check_contradictions(all_evidence: str) -> dict[str, object] | None:
            try:
                result = await self.run_agent(
                    "epistemic_identify_single_issue",
                    claim=claim.statement,
                    scope=claim.scope,
                    evidence=all_evidence,
                    focus="Check whether any of the evidence items contradict each other.",
                )
                if result.has_issue:
                    return {
                        "description": result.description,
                        "issue_type": result.issue_type,
                        "reversal_test": result.reversal_test,
                    }
            except Exception:
                pass
            return None

        # Build tasks: one per evidence item (capped at MAX_ISSUES) + one contradiction check
        tasks: list[asyncio.Task[dict[str, object] | None]] = []
        for ev_summary in evidence_summaries[: self.MAX_ISSUES]:
            tasks.append(asyncio.ensure_future(_check_single_evidence(ev_summary)))
        if len(evidence_summaries) >= 2:
            tasks.append(asyncio.ensure_future(_check_contradictions(evidence_text)))

        results = await asyncio.gather(*tasks)
        found_issues = [r for r in results if r is not None]

        # ── Deterministic combination ────────────────────────────────────
        evidence_weight: str = assess_result.evidence_weight

        # Check for blocking issues via reversal_test
        has_blocking = any(iss["reversal_test"] for iss in found_issues)

        # If blocking issues exist but evidence was rated strong, downgrade to moderate
        if has_blocking and evidence_weight == "strong":
            evidence_weight = "moderate"

        # Compute verdict deterministically from evidence weight
        passes_scrutiny = evidence_weight in ("strong", "moderate")
        if passes_scrutiny:
            verdict = "pass"
        elif evidence_weight == "conflicting":
            verdict = "fail"
        else:
            verdict = "needs_resolution"

        # Build parallel issue/type lists for _handle_issues
        issue_descriptions = [str(iss["description"]) for iss in found_issues]
        issue_type_strs = [str(iss["issue_type"]) for iss in found_issues]

        await self._handle_issues(
            claim, issue_descriptions, issue_type_strs, verdict=verdict
        )

        return verdict

    async def execute(self, work: WorkItem) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.scrutiny_verdict is not None:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already scrutinized",
            )

        if self.agent_runner:
            # If this is re-scrutiny after investigation, cluster the claim's
            # evidence first. Investigation may have fetched many new items that
            # need to be reduced to a representative subset before scrutiny.
            if claim.investigation_count > 0:
                all_evidence = []
                for eid in claim.evidence_ids:
                    try:
                        ev = await self.repo.get("evidence", eid)
                        if (
                            isinstance(ev, Evidence)
                            and ev.extracted
                            and ev.extracted_content
                            and not ev.invalidated
                        ):
                            all_evidence.append(ev)
                    except Exception:
                        continue
                if len(all_evidence) >= 2:
                    await select_top_k_evidence(
                        self.repo, all_evidence, embedding_model=self.embedding_model
                    )

            evidence_summaries = await self._gather_evidence_summaries(claim)
            verdict = await self._execute_split(claim, evidence_summaries)
            claim.scrutiny_verdict = verdict
        else:
            # No agent runner - pass by default
            claim.scrutiny_verdict = "pass"

        # Saturation check: detect uninformative investigation cycles.
        # If this is a re-scrutiny after investigation (investigation_count > 0)
        # and the verdict is still "needs_resolution" but all blocking uncertainties
        # have been resolved (even as "Unresolvable"), then investigation is
        # not producing useful information. Mark as saturated.
        if (
            claim.investigation_count > 0
            and claim.scrutiny_verdict == "needs_resolution"
        ):
            blocking_unresolved = await self.repo.query(
                "uncertainty",
                affected_claim_ids__contains=claim.entity_id,
                resolution=None,
            )
            blocking_unresolved = [u for u in blocking_unresolved if u.is_blocking]

            if not blocking_unresolved:
                claim.saturated = True

        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] verdict: {claim.scrutiny_verdict}",
        )
