"""Evidence extraction operations (Phase 3).

Fetches raw evidence from external sources (via EvidenceGatherer protocol)
or agent-based extraction, and scores source quality via OpenAlex lookup
or agent assessment.

Depends on: base (BaseOperation, OperationResult, GatheredEvidence)
Operates on: Evidence, Objective entities
"""

from .base import BaseOperation, GatheredEvidence, OperationResult

from ..entities import (
    Claim,
    ClaimStage,
    Evidence,
    Objective,
)
from ..patterns import WorkItem


class ExtractEvidenceOperation(BaseOperation):
    """Extract content from an evidence source.

    Takes an evidence stub (extracted=False) and populates
    the extracted_content field.

    Evidence gathering strategy:
    1. If evidence_gatherer is available: use it to fetch raw content
    2. If agent_runner is available (no gatherer): use agent for extraction
    3. If neither: placeholder content (for testing)
    """

    entity_type = "evidence"

    async def execute(self, work: WorkItem) -> OperationResult:
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
            "[extract_evidence] START %s source_type=%s source_ref=%.80s gatherer=%s runner=%s",
            evidence.entity_id,
            evidence.source_type,
            evidence.source_ref,
            type(self.evidence_gatherer).__name__ if self.evidence_gatherer else "None",
            type(self.agent_runner).__name__ if self.agent_runner else "None",
        )

        # Strategy 1: Use evidence gatherer (async Python, no LLM)
        if self.evidence_gatherer:
            try:
                gathered = await self.evidence_gatherer.gather(
                    evidence.source_type,
                    evidence.source_ref,
                )
                _extract_log.info(
                    "[extract_evidence] GATHERER returned %d items for %s",
                    len(gathered) if gathered else 0,
                    evidence.entity_id,
                )
            except Exception as e:
                import traceback

                _extract_log.warning(
                    "[extract_evidence] GATHERER FAILED for %s: %r\n%s",
                    evidence.entity_id,
                    e,
                    traceback.format_exc(),
                )
                gathered = None

            if gathered:
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

                # Create new Evidence entities for remaining items
                created_ids = [evidence.entity_id]
                for g in gathered[1:]:
                    new_ev = Evidence(
                        objective_id=evidence.objective_id,
                        source_type=g.source_type or evidence.source_type,
                        source_ref=g.source_ref,
                        extracted=True,
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
            # Gatherer returned nothing or failed — fall through to agent extraction
            _extract_log.info(
                "[extract_evidence] GATHERER empty/failed for %s, falling through to agent",
                evidence.entity_id,
            )

        # Strategy 2: Use agent runner (primary when no gatherer, or fallback when gatherer fails)
        if self.agent_runner:
            _extract_log.info(
                "[extract_evidence] AGENT extraction for %s", evidence.entity_id
            )
            # Load objective description for context
            objective_description = ""
            if evidence.objective_id:
                obj = await self.repo.get("objective", evidence.objective_id)
                if isinstance(obj, Objective):
                    objective_description = obj.description

            result = await self.run_agent(
                "epistemic_extract_evidence",
                source_ref=evidence.source_ref,
                source_type=evidence.source_type,
                objective_description=objective_description or evidence.source_ref,
            )
            evidence.extracted_content = result.content
            evidence.limitations = result.limitations
            _extract_log.info(
                "[extract_evidence] AGENT extracted %d chars for %s",
                len(evidence.extracted_content) if evidence.extracted_content else 0,
                evidence.entity_id,
            )
            # Score via agent (no GatheredEvidence available in this path)
            await self._score_evidence(evidence)

        # Strategy 3: Placeholder (no agent runner available)
        else:
            _extract_log.info(
                "[extract_evidence] PLACEHOLDER for %s (no runner)", evidence.entity_id
            )
            evidence.extracted_content = f"[Content from {evidence.source_ref}]"

        evidence.extracted = True
        # Final guard: ensure quality_score is never None for extracted evidence with content
        if (
            evidence.quality_score is None
            and evidence.extracted_content
            and evidence.extracted_content.strip()
        ):
            evidence.quality_score = 0.1
            evidence.quality_metadata = {"source": "default_minimum"}
            _extract_log.info(
                "[extract_evidence] FINAL GUARD applied default_minimum for %s",
                evidence.entity_id,
            )

        _extract_log.info(
            "[extract_evidence] DONE %s extracted=%s quality_score=%s quality_source=%s content_len=%d",
            evidence.entity_id,
            evidence.extracted,
            evidence.quality_score,
            (evidence.quality_metadata or {}).get("source", "none"),
            len(evidence.extracted_content) if evidence.extracted_content else 0,
        )
        await self.repo.save(evidence)

        # ── Judge evidence if already linked to a claim ────────────────
        # For investigation-created evidence, the claim link exists before
        # extraction. Judge immediately after content is available.
        # For plan-created evidence, no claim exists yet — judging happens
        # inside ProposeClaimsOperation after claims are created.
        if (
            self.agent_runner
            and evidence.extracted_content
            and evidence.support_judgment is None
        ):
            claims = await self.repo.query("claim", objective_id=evidence.objective_id)
            linked_claim = None
            for c in claims:
                if isinstance(c, Claim) and evidence.entity_id in c.evidence_ids:
                    linked_claim = c
                    break

            if linked_claim is not None:
                from ..judge import judge_evidence as _judge

                judgment = await _judge(
                    claim_statement=linked_claim.statement,
                    claim_scope=linked_claim.scope,
                    evidence_content=evidence.extracted_content,
                    evidence_source=f"{evidence.source_type}: {evidence.source_ref}",
                    runner=self.agent_runner,
                )
                verdict = judgment.verdict.lower().strip()
                if verdict not in ("supports", "contradicts", "no_bearing"):
                    verdict = "no_bearing"
                evidence.support_judgment = verdict
                evidence.judgment_reasoning = judgment.reasoning
                await self.repo.save(evidence)

                # TMS trigger: if this verdict is "contradicts" and the claim is
                # already promoted, check whether contradicting evidence now outweighs
                # supporting. validate_current_stage handles the balance check.
                if (
                    verdict == "contradicts"
                    and linked_claim.stage != ClaimStage.HYPOTHESIS
                ):
                    supporting = 0
                    contradicting_count = 0
                    for eid_check in linked_claim.evidence_ids:
                        try:
                            ev_check = await self.repo.get("evidence", eid_check)
                            if ev_check.invalidated:
                                continue
                            j = getattr(ev_check, "support_judgment", None)
                            if j == "supports":
                                supporting += 1
                            elif j == "contradicts":
                                contradicting_count += 1
                        except Exception:
                            pass
                    if (
                        supporting + contradicting_count
                    ) >= 2 and contradicting_count >= supporting:
                        linked_claim.needs_revalidation = True
                        await self.repo.save(linked_claim)
                        _extract_log.info(
                            "TMS: contradicting balance (%d vs %d) triggers revalidation for %s",
                            contradicting_count,
                            supporting,
                            linked_claim.entity_id,
                        )

        return OperationResult(
            success=True,
            entity_id=evidence.entity_id,
            message=f"Extracted {len(evidence.extracted_content)} chars",
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
        """Score evidence quality. Four paths: OpenAlex -> agent assessment -> gatherer fallback -> minimum default."""
        import logging

        _log = logging.getLogger(__name__)
        _log.info(
            "[_score_evidence] START %s gathered=%s",
            evidence.entity_id,
            gathered is not None,
        )

        # Path 1: Try OpenAlex if we have a DOI or PMID identifier
        if self.quality_scorer:
            try:
                qs = await self.quality_scorer.score(
                    evidence.source_ref, evidence.source_type
                )
                if qs is not None and qs.source != "needs_assessment":
                    evidence.quality_score = qs.score
                    evidence.quality_metadata = qs.raw_metadata
                    _log.info(
                        "[_score_evidence] PATH1 OpenAlex %s score=%s",
                        evidence.entity_id,
                        qs.score,
                    )
                    return
            except Exception as e:
                _log.warning(
                    "[_score_evidence] PATH1 OpenAlex failed for %s: %r",
                    evidence.entity_id,
                    e,
                )

        # Path 2: Agent-based quality assessment
        if self.agent_runner:
            try:
                claim_context = ""
                if evidence.objective_id:
                    claims = await self.repo.query(
                        "claim", objective_id=evidence.objective_id
                    )
                    if claims:
                        claim_context = claims[0].statement

                source_header = (
                    f"Source: {evidence.source_type} — {evidence.source_ref}"
                )
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
            except Exception as e:
                import traceback

                _log.warning(
                    "[_score_evidence] PATH2 agent FAILED for %s: %r\n%s",
                    evidence.entity_id,
                    e,
                    traceback.format_exc(),
                )

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

        # Path 4: Minimum default — never leave extracted evidence unscored
        if evidence.extracted_content and evidence.extracted_content.strip():
            evidence.quality_score = 0.1
            evidence.quality_metadata = {"source": "default_minimum"}
            _log.info(
                "[_score_evidence] PATH4 default_minimum %s score=0.1",
                evidence.entity_id,
            )
        else:
            _log.warning(
                "[_score_evidence] NO SCORE for %s — no content to score",
                evidence.entity_id,
            )
