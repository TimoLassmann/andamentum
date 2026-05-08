"""Scrutiny operations.

Skeptic review of claims using a split-agent architecture: evidence
assessment + one-issue-at-a-time identification + deterministic
verdict combination. Creates Uncertainty entities for issues found.

Depends on: base (BaseOperation, OperationResult), claims (select_top_k_evidence)
Operates on: Claim, Evidence, Uncertainty, Objective entities
"""

import hashlib

from .base import BaseOperation, OperationInput, OperationResult
from .claims import LLM_PANEL_CAP, select_top_k_evidence, top_n_representatives

from ..entities import (
    Claim,
    Evidence,
    Uncertainty,
    UncertaintyType,
)


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

    async def _compute_input_fingerprint(self, claim: Claim) -> str:
        """SHA-256 of the claim+evidence inputs that scrutiny depends on.

        Hashes the claim statement, scope, and the sorted set of evidence_ids
        that are extracted and not invalidated. cluster_status is intentionally
        excluded: it's an internal optimization for which subset gets sent to
        the LLM, not an independent input — the same evidence_id set will
        produce the same scrutiny outcome regardless of how clustering chooses
        representatives.
        """
        eligible_ids: list[str] = []
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if not isinstance(ev, Evidence):
                continue
            if not ev.extracted_content:
                continue
            if ev.invalidated:
                continue
            eligible_ids.append(eid)
        eligible_ids.sort()

        parts = [claim.statement, claim.scope or "", *eligible_ids]
        blob = "\n".join(parts).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    async def _gather_evidence_summaries(self, claim: Claim) -> list[str]:
        """Gather formatted evidence summaries for agent input.

        Caps the returned list at LLM_PANEL_CAP highest-quality
        representatives so the assess_evidence and contradiction prompts
        stay bounded as the underlying evidence base grows.
        """
        candidates: list[Evidence] = []
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if not isinstance(ev, Evidence) or not ev.extracted_content:
                continue
            if ev.invalidated:
                continue
            if ev.cluster_status in ("corroborative", "deferred"):
                continue
            candidates.append(ev)

        evidence_summaries: list[str] = []
        selected = await top_n_representatives(
            candidates,
            LLM_PANEL_CAP,
            claim_text=claim.statement,
            embedding_model=self.embedding_model,
        )
        for ev in selected:
            quality_str = (
                f", quality={ev.quality_score:.2f}" if ev.quality_score else ""
            )
            evidence_summaries.append(
                f"[{ev.source_type}{quality_str}] {ev.source_ref}\n{ev.extracted_content}"
            )
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
            return None

        async def _check_contradictions(all_evidence: str) -> dict[str, object] | None:
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

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )

        # Idempotence: if the inputs that drive scrutiny are unchanged since
        # the last successful pass, the outcome would be identical. Returning
        # early here prevents the operation from minting fresh Uncertainty
        # entities for the same issues every time the graph re-enters the
        # scrutinise → resolve cycle. The Scrutinize graph node clears the
        # fingerprint when it intentionally wants to force a fresh pass.
        if claim.scrutiny_verdict is not None and claim.scrutiny_fingerprint:
            current_fp = await self._compute_input_fingerprint(claim)
            if current_fp == claim.scrutiny_fingerprint:
                return OperationResult(
                    success=True,
                    entity_id=work.entity_id,
                    message=(
                        f"[{claim.statement[:60]}] verdict: "
                        f"{claim.scrutiny_verdict} (inputs unchanged, no-op)"
                    ),
                    did_work=False,
                )

        if claim.scrutiny_verdict is not None:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already scrutinized",
                did_work=False,
            )

        _deferred = 0  # track deferred cluster count for visibility
        if self.agent_runner:
            # If this is re-scrutiny after investigation, cluster the claim's
            # evidence first. Investigation may have fetched many new items that
            # need to be reduced to a representative subset before scrutiny.
            if claim.investigation_count > 0:
                all_evidence = []
                for eid in claim.evidence_ids:
                    ev = await self.repo.get("evidence", eid)
                    if (
                        isinstance(ev, Evidence)
                        and ev.extracted
                        and ev.extracted_content
                        and not ev.invalidated
                    ):
                        all_evidence.append(ev)
                if len(all_evidence) >= 2:
                    _reps, _total, _deferred = await select_top_k_evidence(
                        self.repo, all_evidence, embedding_model=self.embedding_model
                    )

            evidence_summaries = await self._gather_evidence_summaries(claim)
            verdict = await self._execute_split(claim, evidence_summaries)
            claim.scrutiny_verdict = verdict
        else:
            # No agent runner - pass by default
            claim.scrutiny_verdict = "pass"

        # Stamp the fingerprint of the inputs we just scrutinised so a
        # subsequent call with identical inputs short-circuits.
        claim.scrutiny_fingerprint = await self._compute_input_fingerprint(claim)
        await self.repo.save(claim)

        deferred_note = (
            f" ({_deferred} evidence clusters deferred)" if _deferred > 0 else ""
        )
        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] verdict: {claim.scrutiny_verdict}{deferred_note}",
        )
