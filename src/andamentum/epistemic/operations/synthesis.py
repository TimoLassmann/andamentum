"""Synthesis operations.

Freeze snapshot of the epistemic state and generate the final research
report. FreezeSnapshotOperation deduplicates caveats before creating an
immutable snapshot. SynthesizeReportOperation assembles the canonical
output via a writer-validator loop (LLM-written answer) plus
deterministic markdown assembly from entity data.

Depends on: base (BaseOperation, OperationResult, DEDUP_SIMILARITY_THRESHOLD)
Operates on: Objective, Snapshot, Artefact, Claim, Evidence, Uncertainty entities
"""

from typing import Any

from .base import (
    BaseOperation,
    DEDUP_SIMILARITY_THRESHOLD,
    OperationInput,
    OperationResult,
)

from ..entities import (
    Artefact,
    Claim,
    ClaimStage,
    Evidence,
    Objective,
    Snapshot,
    Uncertainty,
)
from ..gates import STAGE_HIERARCHY


class FreezeSnapshotOperation(BaseOperation):
    """Create immutable snapshot of epistemic state."""

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
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
            u for u in all_caveats if isinstance(u, Uncertainty) and not u.is_blocking
        ]

        if len(caveats) >= 2:
            from ..embeddings import embed_texts
            from ..similarity import group_by_similarity, medoid as find_medoid

            if not self.embedding_model:
                raise RuntimeError(
                    "embedding_model is required for uncertainty deduplication. Pass embedding_model= to create_operations()."
                )
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
                    deduped_count,
                    len(caveats),
                    len(groups),
                )

        # Get claims at or above minimum stage
        claims = await self.repo.query(
            "claim",
            objective_id=objective.entity_id,
        )
        claim_ids = [
            c.entity_id for c in claims if isinstance(c, Claim) and not c.abandoned
        ]

        # Get evidence (exclude invalidated, corroborative, and deferred)
        evidence = await self.repo.query(
            "evidence",
            objective_id=objective.entity_id,
            extracted=True,
        )
        # Snapshot includes all non-invalidated evidence. The synthesis
        # consumer applies its own LLM_PANEL_CAP via top_n_representatives
        # to bound the prompt; the snapshot itself is a complete record
        # of the evidence base at freeze time, not a pre-filtered slice.
        evidence_ids = [e.entity_id for e in evidence if not e.invalidated]

        # Get unresolved uncertainties
        uncertainties = await self.repo.query(
            "uncertainty",
            objective_id=objective.entity_id,
            resolution=None,
        )
        uncertainty_ids = [u.entity_id for u in uncertainties]

        # Carry the combined verdict from CombineClaimVerdicts (Phase 4)
        # onto the snapshot so SynthesizeReport can present a rule-aware
        # combined view alongside the per-claim narrative. Stored under
        # objective.decomposition.combined_verdict by the graph node;
        # promoted to a top-level snapshot field here. Serialised to
        # dict for the snapshot's persistence layer.
        # Phase 6 of the Move-3 plan: typed Decomposition access.
        combined_verdict = None
        if objective.decomposition and objective.decomposition.combined_verdict:
            combined_verdict = objective.decomposition.combined_verdict.model_dump()

        # Create snapshot
        snapshot = Snapshot(
            objective_id=objective.entity_id,
            claim_ids=claim_ids,
            evidence_ids=evidence_ids,
            uncertainty_ids=uncertainty_ids,
            snapshot_type="final",
            combined_verdict=combined_verdict,
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

    async def execute(self, work: OperationInput) -> OperationResult:
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
        claims: list[Claim] = []
        for cid in snapshot.claim_ids:
            c = await self.repo.get("claim", cid)
            if isinstance(c, Claim):
                claims.append(c)
        claims.sort(key=lambda c: -STAGE_HIERARCHY.get(c.stage, -1))

        # Load evidence (snapshot excludes corroborative/deferred at creation, but filter
        # defensively for both and for evidence invalidated after the snapshot was frozen).
        # Cap at LLM_PANEL_CAP highest-quality reps so the synthesis prompt
        # stays bounded as the underlying evidence base grows.
        from .claims import LLM_PANEL_CAP, top_n_representatives

        snapshot_evidence: list[Evidence] = []
        for eid in snapshot.evidence_ids:
            e = await self.repo.get("evidence", eid)
            if (
                isinstance(e, Evidence)
                and not e.invalidated
                and getattr(e, "cluster_status", "unclustered")
                not in ("corroborative", "deferred")
            ):
                snapshot_evidence.append(e)
        evidence: list[Evidence] = top_n_representatives(
            snapshot_evidence, LLM_PANEL_CAP
        )

        # Load uncertainties
        uncertainties: list[Uncertainty] = []
        for uid in snapshot.uncertainty_ids:
            u = await self.repo.get("uncertainty", uid)
            if isinstance(u, Uncertainty):
                uncertainties.append(u)

        question = (
            objective.description
            if isinstance(objective, Objective)
            else "Research question"
        )

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
            combined_verdict=getattr(snapshot, "combined_verdict", None),
        )

        # Writer-validator loop
        title = "Research Summary"
        verdict = ""
        answer = ""

        if self.agent_runner:
            title, verdict, answer = await self._writer_validator_loop(
                question,
                data_context,
                objective_id=snapshot.objective_id,
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
        *,
        objective_id: str = "",
    ) -> tuple[str, str, str]:
        """Run writer-validator loop until answer is faithful or max rounds reached.

        ``objective_id`` is included in K4 log lines as a per-case
        attribution tag — when 5 SciFact cases run in parallel under
        snakemake their stderr interleaves into one log file, and the
        tag is the only way to attribute a 9-round-cap session to a
        specific case after the fact.

        Returns:
            (title, verdict, answer) tuple
        """
        import json
        import logging
        import time

        logger = logging.getLogger(__name__)

        # Short tag for log attribution — first 12 chars of the
        # objective_id keep grep-friendliness while remaining unique.
        oid_tag = objective_id[:12] if objective_id else "????????????"

        title = "Research Summary"
        verdict = ""
        answer = ""
        prior_feedback: list[str] = []

        # K4 instrumentation: capture per-round LLM-call timing so we
        # can SEE where the synthesis 90s actually goes (writer vs.
        # validator, single round vs. many rounds, prompt size). Logged
        # at WARNING so the CLI's verbose-mode log filter doesn't drop
        # them. Cheap (a few timestamps + dict-size); no behavior change.
        loop_t0 = time.monotonic()
        round_num = 0
        # Approximate input size for the writer/validator prompts.
        # JSON-serialise the data_context to get a stable byte count;
        # this is the input the model has to read on every round.
        try:
            data_context_bytes = len(
                json.dumps(data_context, default=str, ensure_ascii=False)
            )
        except Exception:
            data_context_bytes = -1

        for round_num in range(1, self.MAX_VALIDATION_ROUNDS + 1):
            # Writer: produce answer
            writer_kwargs: dict[str, Any] = {
                "research_question": question,
                **data_context,
            }
            if answer and prior_feedback:
                writer_kwargs["previous_answer"] = answer
                writer_kwargs["validator_feedback"] = prior_feedback

            writer_t0 = time.monotonic()
            result = await self.run_agent("epistemic_write_answer", **writer_kwargs)
            writer_ms = (time.monotonic() - writer_t0) * 1000
            title = result.title or title
            verdict = getattr(result, "verdict", "") or ""
            answer = result.answer or ""
            answer_chars = len(answer)

            if not answer:
                logger.warning(
                    "[synthesis.writer] obj=%s round=%d writer_ms=%d "
                    "data_ctx_bytes=%d answer_chars=0 — empty answer, "
                    "breaking",
                    oid_tag,
                    round_num,
                    int(writer_ms),
                    data_context_bytes,
                )
                break

            # Validator: check faithfulness
            validator_t0 = time.monotonic()
            validation = await self.run_agent(
                "epistemic_validate_answer",
                answer=answer,
                research_question=question,
                **data_context,
            )
            validator_ms = (time.monotonic() - validator_t0) * 1000

            approved = validation.approved
            feedback = validation.feedback

            logger.warning(
                "[synthesis.writer] obj=%s round=%d writer_ms=%d validator_ms=%d "
                "data_ctx_bytes=%d answer_chars=%d approved=%s feedback=%d",
                oid_tag,
                round_num,
                int(writer_ms),
                int(validator_ms),
                data_context_bytes,
                answer_chars,
                approved,
                len(feedback),
            )
            # Observation B: log the actual feedback content so we can
            # see WHY the validator keeps rejecting. Each feedback item
            # gets its own line so grep -F "[synthesis.feedback] obj=<id>"
            # gives the full rejection trace for one case.
            for i, item in enumerate(feedback):
                # Truncate to keep one line per feedback item (~280 chars
                # is a comfortable terminal-line limit).
                truncated = item[:280] + ("…" if len(item) > 280 else "")
                logger.warning(
                    "[synthesis.feedback] obj=%s round=%d item=%d/%d: %s",
                    oid_tag,
                    round_num,
                    i + 1,
                    len(feedback),
                    truncated,
                )

            if approved or not feedback:
                break

            prior_feedback = feedback

        loop_total_s = time.monotonic() - loop_t0
        logger.warning(
            "[synthesis.writer] obj=%s DONE total=%.2fs rounds=%d "
            "max_rounds=%d data_ctx_bytes=%d hit_cap=%s",
            oid_tag,
            loop_total_s,
            round_num,
            self.MAX_VALIDATION_ROUNDS,
            data_context_bytes,
            round_num >= self.MAX_VALIDATION_ROUNDS,
        )

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
        combined_verdict: dict[str, Any] | None = None,
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
                verifications.append(
                    f"adversarial (balance: {balance:.2f})"
                    if balance is not None
                    else "adversarial"
                )
            if c.convergence_checked:
                verifications.append("convergence")
            if c.deductive_checked:
                verifications.append("deductive")
            if c.computational_checked:
                verifications.append("computational")
            if verifications:
                parts.append(f"  Verification: {', '.join(verifications)}")

            # Evidence references
            refs = [
                str(evidence_index[eid])
                for eid in c.evidence_ids
                if eid in evidence_index
            ]
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
            claim_stmt = next(
                (c.statement for c in claims if c.entity_id == claim_id), claim_id[:8]
            )
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
            claim_stmt = next(
                (c.statement for c in claims if c.entity_id == claim_id), claim_id[:8]
            )
            parts = [f'Claim: "{claim_stmt}"']
            parts.append(
                f"  Verdict: {conv.verdict} ({conv.num_independent_domains} independent domains)"
            )
            if conv.convergence_strength > 0:
                parts.append(f"  Convergence strength: {conv.convergence_strength:.2f}")
            if conv.explanation:
                parts.append(f"  Assessment: {conv.explanation}")
            convergence_summaries.append("\n".join(parts))

        # Uncertainties
        blocking = [
            u.description
            for u in uncertainties
            if u.is_blocking and u.resolution is None
        ]
        non_blocking = [
            u.description
            for u in uncertainties
            if not u.is_blocking and u.resolution is None
        ]

        # Combined verdict (multi-seed-claim runs only). Surfaces the
        # rule-aware aggregate verdict so the writer agent can frame its
        # answer around AND/OR/WEIGHTED_AND/UNION semantics rather than
        # narrating per-claim verdicts in isolation. The combiner has
        # already applied the decomposition's combination_rule; without
        # this the writer might produce prose that disagrees with the
        # structured combined verdict on the snapshot.
        combined_verdict_summary = "Not applicable (no decomposition)."
        if combined_verdict:
            posterior = combined_verdict.get("posterior")
            verdict_label = combined_verdict.get("verdict", "n/a")
            rule = combined_verdict.get("combination_rule", "n/a")
            n_capped = combined_verdict.get("n_capped", 0)
            n_no_verdict = combined_verdict.get("n_no_verdict", 0)
            n_abandoned = combined_verdict.get("n_abandoned", 0)
            posterior_str = (
                f"{posterior:.3f}" if isinstance(posterior, (int, float)) else "n/a"
            )
            combined_verdict_summary = (
                f"Combination rule: {rule}; combined verdict: "
                f"{verdict_label}; combined posterior: {posterior_str}. "
                f"Diagnostic counts: capped={n_capped}, no_verdict="
                f"{n_no_verdict}, abandoned={n_abandoned}. "
                "(This aggregate honours the decomposition rule. Frame "
                "your answer around it; per-claim verdicts above are "
                "supporting detail.)"
            )

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
            "combined_verdict": combined_verdict_summary,
        }

    @staticmethod
    def _compute_quality_signals(
        claims: list["Claim"],
        evidence: list["Evidence"],
        uncertainties: list["Uncertainty"],
    ) -> dict[str, Any]:
        """Compute structured quality signals deterministically from entities."""
        max_stage = "hypothesis"
        confidence_scores: list[float] = []
        scrutiny_passed = 0
        scrutiny_total = 0
        non_abandoned = [c for c in claims if not c.abandoned]

        for claim in non_abandoned:
            stage = claim.stage
            if STAGE_HIERARCHY.get(stage, 0) > STAGE_HIERARCHY.get(
                ClaimStage(max_stage), 0
            ):
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
        supported_plus = sum(
            1
            for c in non_abandoned
            if STAGE_HIERARCHY.get(c.stage, 0) >= STAGE_HIERARCHY[ClaimStage.SUPPORTED]
        )

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
            "scrutiny_pass_rate": (scrutiny_passed / scrutiny_total)
            if scrutiny_total > 0
            else None,
            "mean_confidence_score": (sum(confidence_scores) / len(confidence_scores))
            if confidence_scores
            else None,
            "evidence_count": len(evidence),
            "mean_evidence_quality": (sum(quality_scores) / len(quality_scores))
            if quality_scores
            else None,
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
    def _build_trace(
        claims: list["Claim"], evidence: list["Evidence"]
    ) -> dict[str, list[str]]:
        """Build trace mapping from claim IDs to evidence IDs deterministically."""
        evidence_id_set = {e.entity_id for e in evidence}
        trace: dict[str, list[str]] = {}
        for claim in claims:
            linked = [eid for eid in claim.evidence_ids if eid in evidence_id_set]
            if linked:
                trace[claim.entity_id] = linked
        return trace


class SynthesizeInsufficientReportOperation(BaseOperation):
    """Synthesise an artefact for the structurally-insufficient case.

    Companion to ``SynthesizeReportOperation``. Same input shape (a
    snapshot id), same output shape (an Artefact stamped onto the
    Objective), but no LLM call. The body is templated deterministically
    from the snapshot's structural counts and the synthesis-demand
    diagnosis the gate produced (carried on
    ``EpistemicGraphState.synthesis_insufficient_reason``).

    The artefact's ``artefact_type`` is ``"insufficient"`` so downstream
    consumers (CLI, tests, exporters) can distinguish a structural
    "we suspended judgment" from a directional verdict by reading a
    typed field rather than parsing prose. The verdict string is
    fixed: "Insufficient evidence to answer." — Peirce's fallibilism
    encoded in the system's topology, not delegated to an LLM prompt.
    """

    entity_type = "snapshot"

    INSUFFICIENT_VERDICT = "Insufficient evidence to answer."

    async def execute(self, work: OperationInput) -> OperationResult:
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

        claims: list[Claim] = []
        for cid in snapshot.claim_ids:
            c = await self.repo.get("claim", cid)
            if isinstance(c, Claim):
                claims.append(c)
        claims.sort(key=lambda c: -STAGE_HIERARCHY.get(c.stage, -1))

        evidence: list[Evidence] = []
        for eid in snapshot.evidence_ids:
            e = await self.repo.get("evidence", eid)
            if isinstance(e, Evidence) and not e.invalidated:
                evidence.append(e)

        uncertainties: list[Uncertainty] = []
        for uid in snapshot.uncertainty_ids:
            u = await self.repo.get("uncertainty", uid)
            if isinstance(u, Uncertainty):
                uncertainties.append(u)

        question = (
            objective.description
            if isinstance(objective, Objective)
            else "Research question"
        )

        quality_signals = SynthesizeReportOperation._compute_quality_signals(
            claims, evidence, uncertainties
        )

        # The structural reason the gate routed here. None when the
        # operation is invoked outside the normal CheckSynthesisDemand
        # path (e.g., in a test that constructs the Snapshot directly);
        # in that case we still produce a coherent insufficient artefact
        # but the body lacks the demand-level diagnosis.
        reason = work.metadata.get("synthesis_insufficient_reason") or ""

        title = self._build_title(question)
        content = self._build_markdown(
            title=title,
            question=question,
            claims=claims,
            evidence=evidence,
            quality_signals=quality_signals,
            reason=reason,
            include_quality_signals=True,
        )
        content_body = self._build_markdown(
            title=title,
            question=question,
            claims=claims,
            evidence=evidence,
            quality_signals=quality_signals,
            reason=reason,
            include_quality_signals=False,
        )

        trace = SynthesizeReportOperation._build_trace(claims, evidence)

        artefact = Artefact(
            objective_id=snapshot.objective_id,
            snapshot_id=snapshot.entity_id,
            artefact_type="insufficient",
            audience_profile=work.metadata.get("audience", "general"),
            content=content,
            content_body=content_body,
            trace=trace,
        )
        await self.repo.save(artefact)

        snapshot.artefact_id = artefact.entity_id
        await self.repo.save(snapshot)

        if isinstance(objective, Objective):
            objective.artefact_id = artefact.entity_id
            objective.phase = "complete"
            await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=artefact.entity_id,
            message=f"Synthesised insufficient artefact ({len(content)} chars)",
            created_entities=[artefact.entity_id],
        )

    @staticmethod
    def _build_title(question: str) -> str:
        truncated = question.strip()
        if len(truncated) > 80:
            truncated = truncated[:77].rstrip() + "..."
        return f"Insufficient Evidence: {truncated}"

    @classmethod
    def _build_markdown(
        cls,
        *,
        title: str,
        question: str,
        claims: list["Claim"],
        evidence: list["Evidence"],
        quality_signals: dict[str, Any],
        reason: str,
        include_quality_signals: bool,
    ) -> str:
        """Deterministic templated body. No LLM. Surfaces structural
        counts plus the gate's diagnosis so the artefact reads as a
        coherent "system suspended judgment, here's why" — not a
        directional verdict invented from no data."""
        n_claims = quality_signals.get("claims_total", 0)
        n_abandoned = quality_signals.get("claims_abandoned", 0)
        n_capped = sum(1 for c in claims if getattr(c, "cycle_capped", False))
        n_no_verdict = sum(
            1
            for c in claims
            if not c.abandoned
            and not getattr(c, "cycle_capped", False)
            and c.integrated_assessment is None
        )
        n_evidence = len(evidence)
        n_blocking = quality_signals.get("blocking_uncertainties", 0)

        sections: list[str] = []
        sections.append(f"# {title}\n")
        sections.append(f"> **Research Question:** {question}")
        if include_quality_signals:
            sections.append(
                f"> **Evidence Sources:** {n_evidence} | "
                f"**Claims Established:** 0 of {n_claims}"
            )
        sections.append("")
        sections.append(f"> **Verdict:** {cls.INSUFFICIENT_VERDICT}")
        sections.append("")
        sections.append(
            "The investigation completed without reaching an integration "
            "verdict. The system suspends judgment on this question rather "
            "than producing a directional answer the evidence does not "
            "support."
        )
        sections.append("")

        sections.append("## What the system attempted")
        sections.append("")
        sections.append(f"- {n_claims} claim(s) investigated.")
        if n_abandoned:
            sections.append(
                f"- {n_abandoned} claim(s) abandoned (no actionable evidence found)."
            )
        if n_capped:
            sections.append(
                f"- {n_capped} claim(s) reached the per-claim investigation cap."
            )
        if n_no_verdict:
            sections.append(f"- {n_no_verdict} claim(s) had no integration verdict.")
        sections.append(f"- {n_evidence} evidence item(s) gathered overall.")
        if n_blocking:
            sections.append(f"- {n_blocking} blocking uncertainty(ies) identified.")
        sections.append("")

        if reason:
            sections.append("## Why no directional verdict is offered")
            sections.append("")
            sections.append(reason)
            sections.append("")

        sections.append(
            'A directional answer ("yes" or "no") would not be '
            "supported by the evidence base assembled. Further "
            "investigation — different sources, different framing, or "
            "human expert review — is the appropriate next step."
        )

        return "\n".join(sections)
