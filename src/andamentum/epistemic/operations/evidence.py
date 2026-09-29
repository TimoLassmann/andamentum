"""Evidence extraction operations.

Fetches raw evidence from external sources (via EvidenceGatherer protocol)
or agent-based extraction, and scores source quality via OpenAlex lookup
or agent assessment.

Depends on: base (BaseOperation, OperationResult, GatheredEvidence)
Operates on: Evidence entities
"""

from .base import BaseOperation, GatheredEvidence, OperationInput, OperationResult

from ..entities import Evidence


class ExtractEvidenceOperation(BaseOperation):
    """Extract content from an evidence source.

    Takes an evidence stub (``extracted=False``) and populates the
    ``extracted_content`` field by calling the configured evidence
    gatherer.

    The previous design had an "agent fallback" path that called an
    LLM to "extract" content when the gatherer returned empty. That
    fallback was structurally broken: the agent's prompt assumed
    ``source_content`` was provided as input, but the call site never
    fetched any source content — it only passed the source_ref (a
    search query in the empty-gatherer case) and the claim's
    ``objective_description``. With no actual paper to read, the LLM
    fell back on its parametric knowledge plus the claim text, and
    synthesised content that paraphrased the claim. Downstream the
    judge labelled this synthesised text "supports" because it
    matched the claim, producing a closed-loop hallucination where
    the system voted on its own input. Discovered via SciFact case
    781 v25 trace (6 phantom "supports" pieces with claim-paraphrased
    content vs 2 real "contradicts" pieces from PubMed/OpenAlex).

    The fallback is removed entirely. When the gatherer returns no
    results, the evidence stub is marked invalidated honestly. When
    no gatherer is configured at all, the operation raises — there is
    no LLM-only mode that produces non-hallucinated evidence content.
    """

    entity_type = "evidence"

    async def execute(self, work: OperationInput) -> OperationResult:
        evidence = await self.repo.get("evidence", work.entity_id)

        if not isinstance(evidence, Evidence):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Evidence",
            )

        if evidence.extracted:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already extracted",
            )

        import logging as _logging

        _extract_log = _logging.getLogger(__name__)
        _extract_log.info(
            "[extract_evidence] START %s source_type=%s source_ref=%.80s gatherer=%s",
            evidence.entity_id,
            evidence.source_type,
            evidence.source_ref,
            type(self.evidence_gatherer).__name__ if self.evidence_gatherer else "None",
        )

        if not self.evidence_gatherer:
            raise RuntimeError(
                f"[extract_evidence] no evidence_gatherer configured for "
                f"{evidence.entity_id}: ExtractEvidenceOperation requires a "
                f"real gatherer. The agent-only fallback was removed because "
                f"the agent has no source content to extract from and produces "
                f"hallucinated content that paraphrases the claim. Wire a real "
                f"gatherer or skip extraction for this objective."
            )

        gathered = await self.evidence_gatherer.gather(
            evidence.source_type,
            evidence.source_ref,
        )
        _extract_log.info(
            "[extract_evidence] GATHERER returned %d items for %s",
            len(gathered) if gathered else 0,
            evidence.entity_id,
        )

        if not gathered:
            # Provider returned no results. Mark the stub invalidated
            # honestly. The previous "fall through to agent extraction"
            # path produced hallucinated content; see the class docstring
            # and SciFact case 781 v25 forensic.
            evidence.invalidated = True
            evidence.invalidation_reason = (
                f"Provider {evidence.source_type} returned no results "
                f"for query: {evidence.source_ref[:120]}"
            )
            evidence.extracted = True  # don't retry
            await self.repo.save(evidence)
            _extract_log.info(
                "[extract_evidence] EMPTY-RESULT %s — stub invalidated",
                evidence.entity_id,
            )
            return OperationResult(
                success=True,
                entity_id=evidence.entity_id,
                message=(
                    f"No source found by {evidence.source_type} "
                    f"for query — stub invalidated"
                ),
            )

        # Fill original stub with first gathered item
        self._fill_evidence_from_gathered(evidence, gathered[0])
        await self._score_evidence(evidence, gathered[0])
        evidence.extracted = True
        await self.repo.save(evidence)
        _extract_log.info(
            "[extract_evidence] GATHERER SUCCESS %s quality_score=%s",
            evidence.entity_id,
            evidence.quality_score,
        )

        # Create new Evidence entities for remaining items.
        # Propagate sub_investigation_id from the originating stub so
        # multi-seed-claim's per-claim evidence pool (filter on
        # sub_investigation_id at multi_seed_claim.py:126) sees ALL
        # the gatherer's results, not just the first one. Without
        # this, gatherer-extras silently drop out of the per-claim
        # pool — each sub-investigation would see only the 1 stub-
        # tied result per provider instead of the full hit set.
        #
        # Also propagate ``depends_on_claim_id`` from the stub. When
        # the stub was created during an investigation round, it was
        # linked to a specific claim; the extras conceptually inherit
        # the same link, so downstream judging (predicate-driven in
        # ``ExtractNewEvidence``) can find them.
        created_ids = [evidence.entity_id]
        new_evidence_entities: list[Evidence] = []
        for g in gathered[1:]:
            new_ev = Evidence(
                objective_id=evidence.objective_id,
                source_type=g.source_type or evidence.source_type,
                source_ref=g.source_ref,
                extracted=True,
                sub_investigation_id=evidence.sub_investigation_id,
                depends_on_claim_id=evidence.depends_on_claim_id,
            )
            self._fill_evidence_from_gathered(new_ev, g)
            await self._score_evidence(new_ev, g)
            await self.repo.save(new_ev)
            created_ids.append(new_ev.entity_id)
            new_evidence_entities.append(new_ev)

        # Append extras to the originating claim's evidence_ids list.
        # This is the reverse-index linkage MultiSeedClaim, scrutiny,
        # and the gates iterate over. Without it, extras exist with
        # a depends_on_claim_id pointer but the claim doesn't know
        # about them — they appear in repo queries but not in
        # ``claim.evidence_ids``.
        if new_evidence_entities and evidence.depends_on_claim_id:
            claim = await self.repo.get("claim", evidence.depends_on_claim_id)
            from ..entities import Claim

            if isinstance(claim, Claim):
                added = False
                for new_ev in new_evidence_entities:
                    if new_ev.entity_id not in claim.evidence_ids:
                        claim.evidence_ids.append(new_ev.entity_id)
                        added = True
                if added:
                    claim.evidence_count = len(claim.evidence_ids)
                    await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=evidence.entity_id,
            message=f"Extracted {len(gathered)} sources into {len(created_ids)} entities",
            created_entities=created_ids,
        )

    def _fill_evidence_from_gathered(
        self, evidence: Evidence, gathered: GatheredEvidence
    ) -> None:
        """Fill an Evidence entity's content fields from a GatheredEvidence item."""
        evidence.extracted_content = gathered.content
        evidence.source_ref = gathered.source_ref or evidence.source_ref
        evidence.limitations.extend(gathered.limitations)

        # Pass through annotations or AI pre-analysis as supplementary context
        if gathered.structured_data:
            parts: list[str] = []

            # Annotation-based format (from passage extraction)
            if annotations := gathered.structured_data.get("annotations"):
                parts.append(
                    "Evidence pointers:\n" + "\n".join(f"- {a}" for a in annotations)
                )

            # Legacy format (from fallback strategies or other providers)
            if ai_summary := gathered.structured_data.get("ai_summary"):
                parts.append(f"AI Summary: {ai_summary}")
            if key_points := gathered.structured_data.get("key_points"):
                parts.append("Key Points:\n" + "\n".join(f"- {p}" for p in key_points))
            if key_excerpts := gathered.structured_data.get("key_excerpts"):
                parts.append(
                    "Verbatim Excerpts:\n" + "\n".join(f'"{e}"' for e in key_excerpts)
                )

            if parts:
                evidence.experimental_context = "\n\n".join(parts)

    async def _score_evidence(
        self, evidence: Evidence, gathered: GatheredEvidence | None = None
    ) -> None:
        """Score evidence quality. Three paths: OpenAlex -> agent assessment ->
        gatherer-supplied score. Raises if none can produce a score.

        Path 2's near-zero TMS invalidation is GATED by a second opinion. The
        agent score is one sampled point value over four free-floating
        numbers; below 0.10 it used to destroy the evidence outright, so a
        single unlucky draw erased real signal and nothing downstream could
        tell erasure from absence. The evidence is now invalidated only when
        ``llm_judge`` independently and CONFIDENTLY agrees the text is
        substanceless (``overall == 'fails'`` and ``not needs_review`` — the
        label decides, and ``needs_review`` already folds in the doubt
        threshold, so no new numeric cut is introduced here). When the judge
        disagrees or is unsure, the evidence survives WITH its low
        ``quality_score`` intact, so it stays heavily down-weighted
        everywhere. Attenuation, not destruction.
        """
        import logging

        _log = logging.getLogger(__name__)
        _log.info(
            "[_score_evidence] START %s gathered=%s",
            evidence.entity_id,
            gathered is not None,
        )

        # Path 1: Try the bibliometric resolver (currently OpenAlex)
        # if we can extract a DOI / PMID / arXiv identifier.
        #
        # Phase 3 of the efficiency plan: extraction is now done here
        # against BOTH source_ref AND extracted_content[:1000], not
        # just source_ref. Many evidence items have a DOI in their
        # content body but not in the URL; the upstream extraction
        # routes them to Path 1 (free OpenAlex lookup) instead of
        # Path 2 (LLM-based quality assessment), saving an LLM call
        # per item.
        if self.quality_scorer:
            from .identifier_extraction import extract_identifiers

            content_window = (
                evidence.extracted_content[:1000]
                if evidence.extracted_content
                else None
            )
            identifiers = extract_identifiers(evidence.source_ref, content_window)
            qs = await self.quality_scorer.score(
                identifiers, evidence.source_ref, evidence.source_type
            )
            if qs is not None and qs.source != "needs_assessment":
                evidence.quality_score = qs.score
                evidence.quality_metadata = qs.raw_metadata
                _log.info(
                    "[_score_evidence] PATH1 OpenAlex %s score=%s "
                    "(identifiers=doi=%s, pmid=%s, arxiv=%s)",
                    evidence.entity_id,
                    qs.score,
                    identifiers.doi,
                    identifiers.pmid,
                    identifiers.arxiv,
                )
                return

        # Path 2: Agent-based quality assessment
        if self.agent_runner:
            claim_context = ""
            if evidence.objective_id:
                claims = await self.repo.query(
                    "claim", objective_id=evidence.objective_id
                )
                if claims:
                    claim_context = claims[0].statement

            source_header = f"Source: {evidence.source_type} — {evidence.source_ref}"
            if gathered and gathered.quality_metadata:
                provider = gathered.quality_metadata.get(
                    "provider", evidence.source_type
                )
                source_header = f"Source: {provider} — {evidence.source_ref}"
                extra = {
                    k: v
                    for k, v in gathered.quality_metadata.items()
                    if k != "provider"
                }
                if extra:
                    source_header += f"\nMetadata: {extra}"

            content = evidence.extracted_content or (
                gathered.content if gathered else ""
            )
            _log.info(
                "[_score_evidence] PATH2 calling agent for %s content_len=%d",
                evidence.entity_id,
                len(content),
            )
            result = await self.run_agent(
                "epistemic_assess_evidence_quality",
                evidence_content=content,
                source_header=source_header,
                claim_statement=claim_context,
            )
            # Deterministic combination
            score = (
                0.35 * result.source_credibility
                + 0.25 * result.relevance
                + 0.25 * result.specificity
                + 0.15 * result.recency_appropriate
            )
            evidence.quality_score = max(0.05, min(1.0, score))
            evidence.quality_metadata = {
                "source": "agent",
                "source_credibility": result.source_credibility,
                "relevance": result.relevance,
                "specificity": result.specificity,
                "recency_appropriate": result.recency_appropriate,
                "justification": result.justification,
            }
            _log.info(
                "[_score_evidence] PATH2 agent scored %s = %.3f",
                evidence.entity_id,
                evidence.quality_score,
            )

            # TMS trigger: agent-assessed evidence scoring near zero is unreliable.
            # 0.10 threshold means all four quality dimensions averaged near zero.
            # Only applies to agent assessments — not defaults or fallbacks.
            #
            # The threshold no longer decides on its own: it decides only that
            # the question is worth a second opinion (P7 lazy escalation — the
            # judge fires at the categorical cliff, never on the happy path).
            if evidence.quality_score < 0.10:
                await self._confirm_or_retain_low_quality(
                    evidence,
                    content=content,
                    claim_statement=claim_context,
                    agent_justification=result.justification,
                    credibility=result.source_credibility,
                )

            return

        # Path 3: Fall back to gatherer's quality_score if available
        if gathered and gathered.quality_score is not None:
            evidence.quality_score = max(0.05, gathered.quality_score)
            evidence.quality_metadata = {"source": "gatherer_fallback"}
            _log.info(
                "[_score_evidence] PATH3 gatherer_fallback %s score=%s",
                evidence.entity_id,
                evidence.quality_score,
            )
            return

        # No scoring strategy succeeded — fail loud rather than fabricating a default.
        raise RuntimeError(
            f"[_score_evidence] no scorer available for {evidence.entity_id}: "
            f"_score_evidence requires a quality_scorer, an agent_runner, or a "
            f"gatherer-supplied quality_score on the GatheredEvidence. None were "
            f"available. This indicates a wiring bug in graph construction."
        )

    async def _confirm_or_retain_low_quality(
        self,
        evidence: Evidence,
        *,
        content: str,
        claim_statement: str,
        agent_justification: str,
        credibility: float,
    ) -> None:
        """Destroy the evidence only if a second opinion confidently agrees.

        The three same-typed strings (``content``, ``claim_statement``,
        ``agent_justification``) are keyword-only: transposing any pair would
        judge the wrong text against the wrong context, silently.

        WHY THE WIRING CHECK STAYS HERE, AND NOT AT CONSTRUCTION. The raise
        below is data-dependent: it fires on the first near-zero-quality item,
        which on a gathering path is deep inside a persistence loop whose
        enclosing node quarantines on any exception. That made a construction
        gap cost a whole gather rather than one item — and it did, until
        ``judge_model`` was threaded into the two ``_make_op``-bypassing node
        constructions and into ``op.derive(...)`` in
        ``dispatch_and_persist_for_text``. The considered alternative was to
        assert the invariant once at construction (``agent_runner`` set implies
        ``judge_model`` set). It is NOT taken, for two reasons:

        * it asserts a dependency the operation may genuinely not need — an
          item scored through Path 1 (OpenAlex) or Path 3 (gatherer) never
          reaches this gate, and ``create_operations`` builds every operation
          from one dependency set, so the assertion would make a judge model
          mandatory for callers that will never elicit one;
        * the data-dependence is what actually needed removing, and it was
          removed a different way. A construction gap is now structurally
          impossible rather than merely unlikely: ``BaseOperation.derive``
          copies the full dependency set instead of enumerating it, and
          ``test_structural_wiring::TestInlineOperationConstructionsMatchMakeOp``
          pins the two inline node constructions equal to ``_make_op``. What
          remains reachable is a hand-built operation in a caller that then
          drives real low-quality evidence — for which "you built this
          yourself and forgot a dependency" is the honest message.

        The raise also matches the pre-existing contract of that loop: the
        no-scorer raise below has always aborted a gather the same way.

        Raises:
            RuntimeError: when no ``judge_model`` was wired. An operation
                constructed with an ``agent_runner`` but no judge model cannot
                run the gate, and quietly resuming the old unconditional
                destroy would hide the wiring bug behind lost evidence. Same
                fail-loud shape as the no-scorer raise above.
            Whatever ``judge_score`` raises — propagated, so ``_execute_op``
                records it as a loud quarantine rather than swallowing it into
                an invalidation or a pass.
        """
        import logging

        from ..quality_doubt import confirm_low_quality

        _log = logging.getLogger(__name__)

        if not content:
            # Nothing to second-opinion. An empty body is not a judgment call,
            # and paying a model call to judge nothing is pure cost.
            evidence.invalidated = True
            evidence.invalidation_reason = (
                f"Agent quality assessment scored {evidence.quality_score:.3f} "
                f"(credibility={credibility:.2f}) and the extracted content is "
                f"empty — nothing to re-examine: {agent_justification}"
            )
            _log.warning(
                "[_score_evidence] TMS: invalidated %s — empty content at quality %.3f",
                evidence.entity_id,
                evidence.quality_score,
            )
            return

        if not self.judge_model:
            raise RuntimeError(
                f"[_score_evidence] no judge_model wired for {evidence.entity_id}: "
                f"the near-zero quality gate requires an llm_judge model id so a "
                f"single sampled point value cannot destroy evidence on its own. "
                f"An agent_runner is configured but judge_model is None — thread "
                f"it through EpistemicDeps.judge_model / create_operations("
                f"judge_model=...). This indicates a wiring bug in graph "
                f"construction."
            )

        opinion = await confirm_low_quality(
            content=content,
            source_ref=evidence.source_ref,
            claim_statement=claim_statement,
            agent_score=evidence.quality_score if evidence.quality_score else 0.0,
            model=self.judge_model,
        )
        evidence.quality_distribution = opinion.distribution

        # The LABEL decides; the scalar is for ranking. ``needs_review`` already
        # folds in ``doubt >= NEEDS_REVIEW_DOUBT_THRESHOLD``, so a doubtful
        # 'fails' retains the evidence without this module inventing a second
        # doubt cutoff of its own.
        if opinion.overall == "fails" and not opinion.needs_review:
            evidence.invalidated = True
            evidence.invalidation_reason = (
                f"Agent quality assessment scored {evidence.quality_score:.3f} "
                f"(credibility={credibility:.2f}): {agent_justification} — "
                f"confirmed by an independent judge (overall=fails, "
                f"doubt={opinion.doubt:.2f}): {opinion.summary}"
            )
            _log.warning(
                "[_score_evidence] TMS: invalidated %s — quality %.3f, second "
                "opinion confidently agreed (doubt=%.2f)",
                evidence.entity_id,
                evidence.quality_score,
                opinion.doubt,
            )
            return

        _log.warning(
            "[_score_evidence] TMS: RETAINED %s at quality %.3f — second opinion "
            "was overall=%s (doubt=%.2f, needs_review=%s). The low score stands "
            "and still down-weights this item everywhere; it is attenuated, not "
            "destroyed.",
            evidence.entity_id,
            evidence.quality_score,
            opinion.overall,
            opinion.doubt,
            opinion.needs_review,
        )
