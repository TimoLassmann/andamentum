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
        created_ids = [evidence.entity_id]
        for g in gathered[1:]:
            new_ev = Evidence(
                objective_id=evidence.objective_id,
                source_type=g.source_type or evidence.source_type,
                source_ref=g.source_ref,
                extracted=True,
                sub_investigation_id=evidence.sub_investigation_id,
            )
            self._fill_evidence_from_gathered(new_ev, g)
            await self._score_evidence(new_ev, g)
            await self.repo.save(new_ev)
            created_ids.append(new_ev.entity_id)

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
        """Score evidence quality. Three paths: OpenAlex -> agent assessment -> gatherer-supplied score. Raises if none can produce a score."""
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
            identifiers = extract_identifiers(
                evidence.source_ref, content_window
            )
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
            if evidence.quality_score < 0.10:
                evidence.invalidated = True
                evidence.invalidation_reason = (
                    f"Agent quality assessment scored {evidence.quality_score:.3f} "
                    f"(credibility={result.source_credibility:.2f}): {result.justification}"
                )
                _log.warning(
                    "[_score_evidence] TMS: invalidated %s — quality %.3f below threshold",
                    evidence.entity_id,
                    evidence.quality_score,
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
