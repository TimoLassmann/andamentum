"""Synthesis operations (Phase 9).

Freeze snapshot of the epistemic state and generate the final research
report. FreezeSnapshotOperation deduplicates caveats before creating an
immutable snapshot. SynthesizeReportOperation assembles the canonical
output via a writer-validator loop (LLM-written answer) plus
deterministic markdown assembly from entity data.

Depends on: base (BaseOperation, OperationResult, DEDUP_SIMILARITY_THRESHOLD)
Operates on: Objective, Snapshot, Artefact, Claim, Evidence, Uncertainty entities
"""

from typing import Any

from .base import BaseOperation, DEDUP_SIMILARITY_THRESHOLD, OperationResult

from ..entities import (
    Artefact,
    Claim,
    ClaimStage,
    Evidence,
    Objective,
    Snapshot,
    Uncertainty,
)
from ..patterns import WorkItem


class FreezeSnapshotOperation(BaseOperation):
    """Create immutable snapshot of epistemic state."""

    entity_type = "objective"

    async def execute(self, work: WorkItem) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
            )

        # Check if already has snapshot
        if objective.snapshot_id:
            return OperationResult(
                success=True,
                entity_id=objective.entity_id,
                message="Objective already has snapshot",
            )

        # ── Deduplicate caveats before freezing ──────────────────────────
        # Caveats are non-blocking, unresolved uncertainties. Many are
        # near-duplicates from scrutiny running independently on each claim.
        # Group by embedding similarity, keep the medoid (most central),
        # resolve the rest so they don't appear in the snapshot.
        all_caveats = await self.repo.query(
            "uncertainty",
            objective_id=objective.entity_id,
            resolution=None,
        )
        caveats: list[Uncertainty] = [
            u for u in all_caveats
            if isinstance(u, Uncertainty) and not u.is_blocking
        ]

        if len(caveats) >= 2:
            from ..embeddings import embed_texts
            from ..similarity import group_by_similarity, medoid as find_medoid

            if not self.embedding_model:
                raise RuntimeError("embedding_model is required for uncertainty deduplication. Pass embedding_model= to create_operations().")
            caveat_texts = [c.description for c in caveats]
            embeddings = await embed_texts(caveat_texts, model=self.embedding_model)
            groups = group_by_similarity(embeddings, DEDUP_SIMILARITY_THRESHOLD)

            deduped_count = 0
            for group in groups:
                if len(group) < 2:
                    continue
                representative_idx = find_medoid(embeddings, group)
                representative = caveats[representative_idx]
                for idx in group:
                    if idx != representative_idx:
                        caveats[idx].resolve(
                            f"Deduplicated: same theme as [{representative.entity_id}]"
                        )
                        await self.repo.save(caveats[idx])
                        deduped_count += 1

            if deduped_count > 0:
                import logging
                logging.getLogger(__name__).info(
                    "freeze_snapshot: deduped %d/%d caveats into %d groups",
                    deduped_count, len(caveats), len(groups),
                )

        # Get claims at or above minimum stage
        claims = await self.repo.query(
            "claim",
            objective_id=objective.entity_id,
        )
        claim_ids = [c.entity_id for c in claims if isinstance(c, Claim) and not c.abandoned]

        # Get evidence (exclude invalidated, corroborative, and deferred)
        evidence = await self.repo.query(
            "evidence",
            objective_id=objective.entity_id,
            extracted=True,
        )
        evidence_ids = [
            e.entity_id
            for e in evidence
            if not e.invalidated and getattr(e, "cluster_status", "unclustered") not in ("corroborative", "deferred")
        ]

        # Get unresolved uncertainties
        uncertainties = await self.repo.query(
            "uncertainty",
            objective_id=objective.entity_id,
            resolution=None,
        )
        uncertainty_ids = [u.entity_id for u in uncertainties]

        # Create snapshot
        snapshot = Snapshot(
            objective_id=objective.entity_id,
            claim_ids=claim_ids,
            evidence_ids=evidence_ids,
            uncertainty_ids=uncertainty_ids,
            snapshot_type="final",
        )
        await self.repo.save(snapshot)

        # Update objective
        objective.snapshot_id = snapshot.entity_id
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=snapshot.entity_id,
            message=f"Snapshot with {len(claim_ids)} claims",
            created_entities=[snapshot.entity_id],
        )


class SynthesizeReportOperation(BaseOperation):
    """Synthesize report from snapshot using code-driven assembly.

    The artefact is the ONE canonical output of the epistemic system.
    It must be a complete, human-readable research report that also
    contains everything a downstream LLM needs to judge the output.

    Architecture:
    - LLM writes the opening answer (validated by a writer-validator loop)
    - Everything else is assembled deterministically from entity data
    - No truncation — all evidence, claims, uncertainties included in full

    Flow:
    1. Load all entities from the snapshot
    2. Load verification data (adversarial, convergence) per claim
    3. Compute quality signals deterministically
    4. Writer-validator loop: LLM writes answer, validator checks faithfulness
    5. Assemble full markdown report deterministically
    6. Build trace mapping from DB relationships
    """

    entity_type = "snapshot"

    MAX_VALIDATION_ROUNDS = 10

    async def execute(self, work: WorkItem) -> OperationResult:
        snapshot = await self.repo.get("snapshot", work.entity_id)

        if not isinstance(snapshot, Snapshot):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Snapshot",
            )

        if snapshot.artefact_id is not None:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Artefact already compiled",
            )

        objective = await self.repo.get("objective", snapshot.objective_id)

        # Load claims sorted by stage (highest first)
        stage_order = {
            ClaimStage.ACTIONABLE: 0,
            ClaimStage.ROBUST: 1,
            ClaimStage.PROVISIONAL: 2,
            ClaimStage.SUPPORTED: 3,
            ClaimStage.HYPOTHESIS: 4,
        }
        claims: list[Claim] = []
        for cid in snapshot.claim_ids:
            c = await self.repo.get("claim", cid)
            if isinstance(c, Claim):
                claims.append(c)
        claims.sort(key=lambda c: stage_order.get(c.stage, 99))

        # Load evidence (snapshot excludes corroborative/deferred at creation, but filter
        # defensively for both and for evidence invalidated after the snapshot was frozen)
        evidence: list[Evidence] = []
        for eid in snapshot.evidence_ids:
            e = await self.repo.get("evidence", eid)
            if isinstance(e, Evidence) and not e.invalidated and getattr(e, "cluster_status", "unclustered") not in ("corroborative", "deferred"):
                evidence.append(e)

        # Load uncertainties
        uncertainties: list[Uncertainty] = []
        for uid in snapshot.uncertainty_ids:
            u = await self.repo.get("uncertainty", uid)
            if isinstance(u, Uncertainty):
                uncertainties.append(u)

        question = objective.description if isinstance(objective, Objective) else "Research question"

        # Load verification data per claim
        from ..primitives import AdversarialEvidence, ConvergentEvidence

        adversarial_by_claim: dict[str, AdversarialEvidence] = {}
        convergence_by_claim: dict[str, ConvergentEvidence] = {}
        for claim in claims:
            adv = await self.repo.get_adversarial_evidence_for_claim(claim.entity_id)
            if adv is not None:
                adversarial_by_claim[claim.entity_id] = adv
            conv = await self.repo.get_convergent_evidence_for_claim(claim.entity_id)
            if conv is not None:
                convergence_by_claim[claim.entity_id] = conv

        # Compute quality signals deterministically
        quality_signals = self._compute_quality_signals(claims, evidence, uncertainties)

        # Build evidence index for cross-referencing
        evidence_index = {e.entity_id: i + 1 for i, e in enumerate(evidence)}

        # Build data summaries for the writer and validator agents
        data_context = self._build_data_context(
            claims,
            evidence,
            uncertainties,
            adversarial_by_claim,
            convergence_by_claim,
            evidence_index,
            quality_signals,
        )

        # Writer-validator loop
        title = "Research Summary"
        verdict = ""
        answer = ""

        if self.agent_runner:
            title, verdict, answer = await self._writer_validator_loop(
                question,
                data_context,
            )

        # Build markdown report (deterministic — everything except answer)
        build_args = (
            title,
            verdict,
            answer,
            question,
            claims,
            evidence,
            uncertainties,
            adversarial_by_claim,
            convergence_by_claim,
            evidence_index,
            quality_signals,
        )
        content = self._build_markdown(*build_args)
        content_body = self._build_markdown(*build_args, include_quality_signals=False)

        # Build trace deterministically from DB relationships
        trace = self._build_trace(claims, evidence)

        if not content:
            return OperationResult(
                success=False,
                entity_id=snapshot.entity_id,
                message="Failed to generate content",
            )

        # Create artefact
        artefact = Artefact(
            objective_id=snapshot.objective_id,
            snapshot_id=snapshot.entity_id,
            artefact_type=work.metadata.get("artefact_type", "summary"),
            audience_profile=work.metadata.get("audience", "general"),
            content=content,
            content_body=content_body,
            trace=trace,
        )
        await self.repo.save(artefact)

        # Update snapshot
        snapshot.artefact_id = artefact.entity_id
        await self.repo.save(snapshot)

        # Update objective
        if isinstance(objective, Objective):
            objective.artefact_id = artefact.entity_id
            objective.phase = "complete"
            await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=artefact.entity_id,
            message=f"Synthesized {len(content)} chars",
            created_entities=[artefact.entity_id],
        )

    async def _writer_validator_loop(
        self,
        question: str,
        data_context: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Run writer-validator loop until answer is faithful or max rounds reached.

        Returns:
            (title, verdict, answer) tuple
        """
        import logging

        logger = logging.getLogger(__name__)

        title = "Research Summary"
        verdict = ""
        answer = ""
        prior_feedback: list[str] = []

        for round_num in range(1, self.MAX_VALIDATION_ROUNDS + 1):
            # Writer: produce answer
            writer_kwargs: dict[str, Any] = {
                "research_question": question,
                **data_context,
            }
            if answer and prior_feedback:
                writer_kwargs["previous_answer"] = answer
                writer_kwargs["validator_feedback"] = prior_feedback

            result = await self.run_agent("epistemic_write_answer", **writer_kwargs)
            title = result.title or title
            verdict = getattr(result, "verdict", "") or ""
            answer = result.answer or ""

            if not answer:
                break

            # Validator: check faithfulness
            validation = await self.run_agent(
                "epistemic_validate_answer",
                answer=answer,
                research_question=question,
                **data_context,
            )

            approved = validation.approved
            feedback = validation.feedback

            if approved or not feedback:
                logger.info(f"Answer approved after {round_num} round(s)")
                break

            logger.info(f"Validation round {round_num}: {len(feedback)} issue(s) found")
            prior_feedback = feedback

        return title, verdict, answer

    @staticmethod
    def _build_data_context(
        claims: list["Claim"],
        evidence: list["Evidence"],
        uncertainties: list["Uncertainty"],
        adversarial_by_claim: dict[str, Any],
        convergence_by_claim: dict[str, Any],
        evidence_index: dict[str, int],
        quality_signals: dict[str, Any],
    ) -> dict[str, Any]:
        """Build data summaries for writer and validator agents.

        Returns a dict of keyword arguments that both agents receive,
        giving them the same view of the underlying data.
        """
        # Claims with full context
        claim_summaries = []
        for c in claims:
            parts = [f"[{c.stage.value.upper()}] {c.statement}"]
            parts.append(f"  Scope: {c.scope}")
            if c.confidence_score is not None:
                parts.append(f"  Confidence: {c.confidence_score:.2f}")
            if c.scrutiny_verdict:
                parts.append(f"  Scrutiny: {c.scrutiny_verdict}")

            # Verification status
            verifications = []
            if c.adversarial_checked:
                balance = c.adversarial_balance
                verifications.append(f"adversarial (balance: {balance:.2f})" if balance is not None else "adversarial")
            if c.convergence_checked:
                verifications.append("convergence")
            if c.deductive_checked:
                verifications.append("deductive")
            if c.computational_checked:
                verifications.append("computational")
            if verifications:
                parts.append(f"  Verification: {', '.join(verifications)}")

            # Evidence references
            refs = [str(evidence_index[eid]) for eid in c.evidence_ids if eid in evidence_index]
            if refs:
                parts.append(f"  Evidence: [{', '.join(refs)}]")

            if c.abandoned:
                parts.append("  STATUS: ABANDONED")

            claim_summaries.append("\n".join(parts))

        # Evidence summaries — show the system's judgment, not raw source content.
        # The writer agent should reason from our interpretation, not quote sources.
        evidence_summaries = []
        for e in evidence:
            idx = evidence_index.get(e.entity_id, 0)
            qs = e.quality_score
            quality_str = f", quality: {qs:.2f}" if qs is not None else ""
            judgment = f" [{e.support_judgment}]" if e.support_judgment else ""
            reasoning = e.judgment_reasoning or "(not yet assessed)"
            evidence_summaries.append(
                f"[{idx}] ({e.source_type}{quality_str}){judgment} {reasoning}\n  Source: {e.source_ref}"
            )

        # Adversarial results
        adversarial_summaries = []
        for claim_id, adv in adversarial_by_claim.items():
            # Find claim statement for context
            claim_stmt = next((c.statement for c in claims if c.entity_id == claim_id), claim_id[:8])
            parts = [f'Claim: "{claim_stmt}"']
            parts.append(f"  Balance: {adv.adversarial_balance:.2f} ({adv.verdict})")
            if adv.counterarguments:
                parts.append(f"  Counterarguments ({len(adv.counterarguments)}):")
                for ca in adv.counterarguments:
                    parts.append(f"    - {ca.summary} (source: {ca.source_ref})")
            if adv.explanation:
                parts.append(f"  Assessment: {adv.explanation}")
            adversarial_summaries.append("\n".join(parts))

        # Convergence results
        convergence_summaries = []
        for claim_id, conv in convergence_by_claim.items():
            claim_stmt = next((c.statement for c in claims if c.entity_id == claim_id), claim_id[:8])
            parts = [f'Claim: "{claim_stmt}"']
            parts.append(f"  Verdict: {conv.verdict} ({conv.num_independent_domains} independent domains)")
            if conv.convergence_strength > 0:
                parts.append(f"  Convergence strength: {conv.convergence_strength:.2f}")
            if conv.explanation:
                parts.append(f"  Assessment: {conv.explanation}")
            convergence_summaries.append("\n".join(parts))

        # Uncertainties
        blocking = [u.description for u in uncertainties if u.is_blocking and u.resolution is None]
        non_blocking = [u.description for u in uncertainties if not u.is_blocking and u.resolution is None]

        return {
            "claims": claim_summaries,
            "evidence": evidence_summaries,
            "adversarial_results": adversarial_summaries
            if adversarial_summaries
            else ["No adversarial search performed."],
            "convergence_results": convergence_summaries
            if convergence_summaries
            else ["No convergence assessment performed."],
            "blocking_uncertainties": blocking if blocking else ["None."],
            "non_blocking_uncertainties": non_blocking if non_blocking else ["None."],
            "quality_signals": quality_signals,
        }

    @staticmethod
    def _compute_quality_signals(
        claims: list["Claim"],
        evidence: list["Evidence"],
        uncertainties: list["Uncertainty"],
    ) -> dict[str, Any]:
        """Compute structured quality signals deterministically from entities."""
        stage_order = {
            ClaimStage.HYPOTHESIS: 0,
            ClaimStage.SUPPORTED: 1,
            ClaimStage.PROVISIONAL: 2,
            ClaimStage.ROBUST: 3,
            ClaimStage.ACTIONABLE: 4,
        }
        max_stage = "hypothesis"
        confidence_scores: list[float] = []
        scrutiny_passed = 0
        scrutiny_total = 0
        non_abandoned = [c for c in claims if not c.abandoned]

        for claim in non_abandoned:
            stage = claim.stage
            if stage_order.get(stage, 0) > stage_order.get(ClaimStage(max_stage), 0):
                max_stage = stage.value

            if claim.confidence_score is not None:
                confidence_scores.append(claim.confidence_score)

            if claim.scrutiny_verdict is not None:
                scrutiny_total += 1
                if claim.scrutiny_verdict == "pass":
                    scrutiny_passed += 1

        quality_scores: list[float] = []
        for ev in evidence:
            qs = ev.quality_score
            if qs is not None:
                quality_scores.append(qs)

        unresolved = [u for u in uncertainties if not u.is_resolved]
        blocking = [u for u in unresolved if u.is_blocking]

        # Determine confidence level
        source_types = {e.source_type for e in evidence}
        has_external = bool(source_types - {"world_knowledge"})
        supported_plus = sum(1 for c in non_abandoned if stage_order.get(c.stage, 0) >= 1)

        if len(evidence) == 0:
            confidence_level = "none"
        elif not has_external:
            confidence_level = "low"
        elif supported_plus >= len(non_abandoned) * 0.5 and has_external:
            confidence_level = "high"
        elif has_external or supported_plus > 0:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "confidence_level": confidence_level,
            "max_stage": max_stage,
            "claims_established": supported_plus,
            "claims_total": len(non_abandoned),
            "claims_abandoned": sum(1 for c in claims if c.abandoned),
            "scrutiny_pass_rate": (scrutiny_passed / scrutiny_total) if scrutiny_total > 0 else None,
            "mean_confidence_score": (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else None,
            "evidence_count": len(evidence),
            "mean_evidence_quality": (sum(quality_scores) / len(quality_scores)) if quality_scores else None,
            "unresolved_uncertainties": len(unresolved),
            "blocking_uncertainties": len(blocking),
        }

    @staticmethod
    def _build_markdown(
        title: str,
        verdict: str,
        answer: str,
        question: str,
        claims: list["Claim"],
        evidence: list["Evidence"],
        uncertainties: list["Uncertainty"],
        adversarial_by_claim: dict[str, Any],
        convergence_by_claim: dict[str, Any],
        evidence_index: dict[str, int],
        quality_signals: dict[str, Any],
        *,
        include_quality_signals: bool = True,
    ) -> str:
        """Assemble the canonical research report from structured data.

        The answer section is LLM-written (validated by writer-validator loop).
        Everything else is deterministic — assembled from entity fields.

        Args:
            include_quality_signals: If False, omits confidence header,
                per-claim stage/confidence metadata, and Methodology section.
                Use False for benchmark evaluation where these pre-computed
                labels would bias downstream interpreters.
        """
        sections: list[str] = []

        established = quality_signals.get("claims_established", 0)
        total_claims = quality_signals.get("claims_total", 0)
        ev_count = quality_signals.get("evidence_count", 0)

        # === Header ===
        sections.append(f"# {title}\n")
        sections.append(f"> **Research Question:** {question}")
        if include_quality_signals:
            sections.append(
                f"> **Evidence Sources:** {ev_count} | "
                f"**Claims Established:** {established} of {total_claims}"
            )
        sections.append("")

        # === Verdict (one-sentence bottom line) ===
        if verdict:
            sections.append(f"> **Verdict:** {verdict}")
            sections.append("")

        # === LLM-written answer (validated) ===
        if answer:
            sections.append(answer)
            sections.append("")

        # NOTE: Findings, Evidence Sources, Challenges, Convergence,
        # Open Questions, Caveats, and Methodology are NOT appended here.
        # The HTML report renders all of these from structured database
        # entities — producing better formatting, clickable citations,
        # and sequential evidence numbering. The artefact stores only
        # the prose summary (title + verdict + answer).

        return "\n".join(sections)

    @staticmethod
    def _build_trace(claims: list["Claim"], evidence: list["Evidence"]) -> dict[str, list[str]]:
        """Build trace mapping from claim IDs to evidence IDs deterministically."""
        evidence_id_set = {e.entity_id for e in evidence}
        trace: dict[str, list[str]] = {}
        for claim in claims:
            linked = [eid for eid in claim.evidence_ids if eid in evidence_id_set]
            if linked:
                trace[claim.entity_id] = linked
        return trace
