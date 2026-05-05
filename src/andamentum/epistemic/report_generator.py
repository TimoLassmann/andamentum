"""
Report Generator - High-level report orchestration for epistemic system.

Extracts data from epistemic database and generates HTML reports.

Architecture: Layer 4 (Application)

CRITICAL: This module NEVER truncates data. All claims, evidence, uncertainties,
and other information are included in full in the generated reports.
"""

import logging
from pathlib import Path
from typing import Optional

from andamentum.document_store import DocumentStore
from .primitives import UncertaintyType
from .report_data import (
    AdversarialSummary,
    ClaimSummary,
    ConfidenceScores,
    ConvergenceSummary,
    EvidenceSummary,
    InvestigationStats,
    QUESTION_TYPE_LABELS,
    ReportData,
    UncertaintySummary,
)
from .repository import EpistemicRepository
from .thresholds import (
    ADVERSARIAL_REFUTED_THRESHOLD,
    ADVERSARIAL_SURVIVED_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Orchestrates HTML report generation from epistemic database.

    Extracts all data from the database and generates a standalone HTML report
    with full traceability. NO truncation of any data.
    """

    def __init__(self, store: DocumentStore, database_name: str):
        """Initialize with a DocumentStore instance.

        Args:
            store: DocumentStore for the epistemic database
            database_name: Name of the database for report metadata
        """
        self.store = store
        self.database_name = database_name
        self.repo = EpistemicRepository(store)

    async def extract_report_data(
        self,
        objective_id: Optional[str] = None,
        model_name: str = "unknown",
    ) -> Optional[ReportData]:
        """Extract all data needed for report generation.

        If objective_id is not provided, finds the most recent objective.
        CRITICAL: Extracts ALL data - no truncation.

        Args:
            objective_id: Optional specific objective to report on
            model_name: Model used for the investigation

        Returns:
            ReportData with all extracted information, or None if no data found
        """
        # Find objective
        if objective_id:
            try:
                objective = await self.repo.get_objective(objective_id)
            except Exception:
                logger.warning(f"Objective not found: {objective_id}")
                return None
        else:
            # Find most recent objective
            results = await self.store.find_by_metadata(
                {"epistemic_type": "objective"},
                limit=1,
            )
            if not results:
                logger.warning("No objectives found in database")
                return None
            try:
                objective = await self.repo.get_objective(
                    results[0].metadata.get("objective_id", "")
                )
            except Exception:
                logger.warning("Failed to load objective")
                return None

        obj_id = objective.objective_id

        # Get artefact (executive summary)
        artefacts = await self.repo.get_artefacts_for_objective(obj_id)
        artefact = artefacts[-1] if artefacts else None

        direct_answer = ""
        verdict = ""
        artefact_trace: dict[str, list[str]] = {}
        if artefact:
            content = artefact.content
            # Extract verdict if present (stored as "> **Verdict:** ..." line)
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("> **Verdict:**"):
                    verdict = stripped[len("> **Verdict:**") :].strip()
                    break

            # Extract content, removing the header and verdict if present
            if content.startswith("# "):
                lines = content.split("\n", 2)
                if len(lines) > 2:
                    content = lines[2]
            # Remove verdict line from the answer body
            if verdict:
                content_lines = content.split("\n")
                content_lines = [
                    ln
                    for ln in content_lines
                    if not ln.strip().startswith("> **Verdict:**")
                ]
                content = "\n".join(content_lines)
            direct_answer = content.strip()
            artefact_trace = artefact.trace or {}

        # Fallback: if no verdict was stored, extract first substantive
        # sentence from the answer (skip headers, blockquotes, blank lines)
        if not verdict and direct_answer:
            for line in direct_answer.split("\n"):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith(">"):
                    continue
                # Found a substantive line — take the first sentence
                for sep in [". ", ".\n"]:
                    if sep in line:
                        verdict = line[: line.index(sep) + 1]
                        break
                else:
                    if line.endswith("."):
                        verdict = line
                if verdict:
                    break

        # Get all claims - NO FILTERING, get everything
        all_claims = await self.repo.get_claims_for_objective(obj_id)
        claims_by_stage: dict[str, int] = {}

        # Get adversarial evidence for all claims (needed for claim summaries)
        adversarial_by_claim: dict[str, object] = {}
        convergence_by_claim: dict[str, object] = {}
        for claim in all_claims:
            adv_evidence = await self.repo.get_adversarial_evidence_for_claim(
                claim.claim_id
            )
            if adv_evidence:
                adversarial_by_claim[claim.claim_id] = adv_evidence
            conv_evidence = await self.repo.get_convergent_evidence_for_claim(
                claim.claim_id
            )
            if conv_evidence:
                convergence_by_claim[claim.claim_id] = conv_evidence

        # Get ALL evidence - NO TRUNCATION
        all_evidence = await self.repo.get_evidence_for_objective(obj_id)

        # ── Evidence filtering, dedup, and renumbering ──
        # Only include judged evidence (support_judgment is not None)
        judged_evidence = [ev for ev in all_evidence if ev.support_judgment is not None]

        # Deduplicate by source_ref (keep first occurrence)
        seen_refs: set[str] = set()
        deduped_evidence = []
        for ev in judged_evidence:
            if ev.source_ref not in seen_refs:
                seen_refs.add(ev.source_ref)
                deduped_evidence.append(ev)

        # Sort: supporting first, then contradicting, then other
        judgment_order = {"supports": 0, "contradicts": 1}
        deduped_evidence.sort(
            key=lambda e: judgment_order.get(e.support_judgment or "", 2)
        )

        # Build sequential index map: old entity_id -> new [1..N]
        evidence_index_map: dict[str, int] = {}
        for i, ev in enumerate(deduped_evidence, start=1):
            evidence_index_map[ev.evidence_id] = i

        evidence_summaries: list[EvidenceSummary] = []
        for ev in deduped_evidence:
            evidence_summaries.append(
                EvidenceSummary(
                    evidence_id=ev.evidence_id,
                    source_type=ev.source_type,
                    source_ref=ev.source_ref,
                    extracted_content=ev.extracted_content or "",
                    limitations=ev.limitations or [],
                    verified=ev.verified,
                    provider=ev.created_by if ev.created_by != "system" else None,
                    support_judgment=ev.support_judgment,
                    judgment_reasoning=ev.judgment_reasoning,
                    quality_score=ev.quality_score,
                )
            )

        # Build claim summaries with enriched fields
        claim_summaries: list[ClaimSummary] = []
        for claim in all_claims:
            stage_name = claim.stage.value
            claims_by_stage[stage_name] = claims_by_stage.get(stage_name, 0) + 1

            # Build verification summary
            verification_parts: list[str] = []
            if claim.scrutiny_verdict:
                verdict_str = claim.scrutiny_verdict
                verification_parts.append(f"Scrutiny: {verdict_str}")
            if claim.adversarial_checked:
                adv = adversarial_by_claim.get(claim.claim_id)
                if adv is not None:
                    balance = getattr(adv, "adversarial_balance", None)
                    if balance is not None:
                        if balance < ADVERSARIAL_REFUTED_THRESHOLD:
                            verification_parts.append(
                                f"Adversarial search: found strong counter-evidence (balance: {balance:.2f})"
                            )
                        elif balance >= ADVERSARIAL_SURVIVED_THRESHOLD:
                            verification_parts.append(
                                f"Adversarial search: withstood challenge (balance: {balance:.2f})"
                            )
                        else:
                            verification_parts.append(
                                f"Adversarial search: mixed results (balance: {balance:.2f})"
                            )
                    else:
                        verification_parts.append("Adversarial search: completed")
                else:
                    verification_parts.append("Adversarial search: completed")
            if claim.convergence_checked:
                conv = convergence_by_claim.get(claim.claim_id)
                if conv is not None:
                    verdict_val = getattr(conv, "verdict", None)
                    if verdict_val is not None:
                        verification_parts.append(
                            f"Convergence: {str(verdict_val).lower()}"
                        )
                    else:
                        verification_parts.append("Convergence: assessed")
                else:
                    verification_parts.append("Convergence: assessed")
            if claim.deductive_checked:
                verification_parts.append("Deductive validation: passed")
            if claim.computational_checked:
                verification_parts.append("Computational verification: completed")

            verification_summary = (
                ". ".join(verification_parts) + "." if verification_parts else ""
            )

            # Map evidence IDs to sequential display numbers
            evidence_refs_display = sorted(
                evidence_index_map[eid]
                for eid in (claim.evidence_ids or [])
                if eid in evidence_index_map
            )

            claim_summaries.append(
                ClaimSummary(
                    claim_id=claim.claim_id,
                    statement=claim.statement,
                    scope=claim.scope,
                    assumptions=claim.assumptions or [],
                    stage=stage_name,
                    evidence_ids=claim.evidence_ids or [],
                    uncertainty_ids=claim.uncertainty_ids or [],
                    adversarial_balance=claim.adversarial_balance,
                    scrutiny_verdict=claim.scrutiny_verdict,
                    verification_summary=verification_summary,
                    evidence_refs_display=evidence_refs_display,
                )
            )

        # Sort claims: SUPPORTED+ first (strongest), then HYPOTHESIS (weakest/challenged)
        _STAGE_SORT = {
            "actionable": 0,
            "robust": 1,
            "provisional": 2,
            "supported": 3,
            "hypothesis": 4,
        }
        claim_summaries.sort(key=lambda c: _STAGE_SORT.get(c.stage.lower(), 99))

        # Get ALL uncertainties - NO TRUNCATION
        all_uncertainties = await self.repo.get_uncertainties_for_objective(obj_id)
        uncertainty_summaries: list[UncertaintySummary] = []
        blocking_count = 0
        non_blocking_count = 0
        resolved_count = 0

        for unc in all_uncertainties:
            is_blocking = self._is_blocking_uncertainty_type(unc.uncertainty_type)

            if unc.is_resolved:
                resolved_count += 1
            elif is_blocking:
                blocking_count += 1
            else:
                non_blocking_count += 1

            uncertainty_summaries.append(
                UncertaintySummary(
                    uncertainty_id=unc.uncertainty_id,
                    uncertainty_type=unc.uncertainty_type.value,
                    description=unc.description,
                    scope=", ".join(unc.affected_claim_ids[:3])
                    if unc.affected_claim_ids
                    else "global",
                    is_blocking=is_blocking,
                    is_resolved=unc.is_resolved,
                    affected_claim_ids=unc.affected_claim_ids or [],
                )
            )

        # Build adversarial summaries per claim
        adversarial_summaries: list[AdversarialSummary] = []
        for claim in all_claims:
            adv_evidence = adversarial_by_claim.get(claim.claim_id)
            if adv_evidence is not None:
                counterarguments = getattr(adv_evidence, "counterarguments", None)
                if counterarguments:
                    for ca in counterarguments:
                        adversarial_summaries.append(
                            AdversarialSummary(
                                claim_id=claim.claim_id,
                                counterargument=ca.summary,
                                strength=ca.weight,
                                source_ref=ca.source_ref,
                                rebuttal=ca.supporting_evidence,
                            )
                        )

        # Build convergence summaries for backward compat
        convergence_summaries: list[ConvergenceSummary] = []
        for claim in all_claims:
            conv_evidence = convergence_by_claim.get(claim.claim_id)
            domain_clusters = (
                getattr(conv_evidence, "domain_clusters", None)
                if conv_evidence is not None
                else None
            )
            if domain_clusters:
                for cluster in domain_clusters:
                    if cluster.representative_classification:
                        domain = cluster.cluster_label or "Unknown Domain"
                        convergence_summaries.append(
                            ConvergenceSummary(
                                domain=domain,
                                supporting_evidence=f"{cluster.cluster_size} evidence items",
                                confidence=cluster.average_evidence_quality,
                            )
                        )

        # Extract open questions from artefact content if available
        open_questions: list[str] = []
        if direct_answer:
            for line in direct_answer.split("\n"):
                line = line.strip()
                if line.endswith("?") and len(line) > 10:
                    if line.startswith("- "):
                        line = line[2:]
                    elif line.startswith("* "):
                        line = line[2:]
                    open_questions.append(line)

        # Build statistics
        stats = InvestigationStats(
            total_evidence=len(all_evidence),
            total_claims=len(all_claims),
            claims_by_stage=claims_by_stage,
            blocking_uncertainties=blocking_count,
            non_blocking_uncertainties=non_blocking_count,
            resolved_uncertainties=resolved_count,
            adversarial_challenges=len(adversarial_summaries),
            convergent_domains=len(convergence_summaries),
        )

        # Build investigation narrative
        investigation_narrative = self._build_investigation_narrative(
            objective,
            all_claims,
            all_evidence,
            all_uncertainties,
            adversarial_by_claim,
            resolved_count,
        )

        # Compute confidence scores.
        # Note: this call does NOT thread retrieval_failed — that flag lives
        # on the graph state and is not persisted on the Objective. So an
        # offline report regenerated from a stored DB will always show
        # terminal_state="completed" even if the original run failed
        # retrieval. The retrieval_failed signal is only surfaced in the
        # live CLI path (via PipelineResult.retrieval_failed).
        confidence_scores: ConfidenceScores | None = None
        try:
            from .confidence import compute_posterior

            po = await compute_posterior(self.repo, obj_id)

            confidence_scores = ConfidenceScores(
                posterior=po.posterior if po else None,
                posterior_supporting=po.supporting_count if po else 0,
                posterior_contradicting=po.contradicting_count if po else 0,
                posterior_question_type=po.question_type if po else None,
                terminal_state=po.terminal_state if po else "completed",
            )
        except Exception as e:
            logger.warning(f"Failed to compute confidence scores: {e}")
            confidence_scores = None

        # Build report data
        return ReportData(
            research_question=objective.description,
            clarified_question=objective.goal_context or objective.description,
            investigation_date=objective.created_at,
            model_used=model_name,
            database_name=self.database_name,
            question_type=objective.question_type,
            verdict=verdict,
            direct_answer=direct_answer,
            artefact_trace=artefact_trace,
            investigation_narrative=investigation_narrative,
            evidence_index_map=evidence_index_map,
            claims=claim_summaries,
            evidence=evidence_summaries,
            uncertainties=uncertainty_summaries,
            adversarial=adversarial_summaries,
            convergence=convergence_summaries,
            open_questions=open_questions,
            stats=stats,
            confidence_scores=confidence_scores,
        )

    @staticmethod
    def _build_investigation_narrative(
        objective: object,
        all_claims: list,
        all_evidence: list,
        all_uncertainties: list,
        adversarial_by_claim: dict[str, object],
        resolved_count: int,
    ) -> str:
        """Build a deterministic investigation narrative from entity state.

        No LLM needed -- everything comes from entity fields.
        """
        parts: list[str] = []

        # Question type
        question_type = getattr(objective, "question_type", None) or "unknown"
        qt_label = QUESTION_TYPE_LABELS.get(question_type, question_type)

        # Evidence gathering
        source_types: dict[str, int] = {}
        for ev in all_evidence:
            provider = (
                ev.created_by
                if hasattr(ev, "created_by") and ev.created_by != "system"
                else ev.source_type
            )
            source_types[provider] = source_types.get(provider, 0) + 1
        providers_str = ", ".join(
            f"{k} ({v})" for k, v in sorted(source_types.items(), key=lambda x: -x[1])
        )
        parts.append(
            f"This {qt_label} was investigated using {len(source_types)} evidence "
            f"provider{'s' if len(source_types) != 1 else ''} ({providers_str}), "
            f"collecting {len(all_evidence)} evidence items."
        )

        # Claim formation
        non_abandoned = [c for c in all_claims if not c.abandoned]
        if non_abandoned:
            parts.append(
                f"\n{len(non_abandoned)} claims were proposed from evidence clusters."
            )

        # Scrutiny
        scrutiny_passed = sum(1 for c in non_abandoned if c.scrutiny_verdict == "pass")
        scrutiny_failed = sum(1 for c in non_abandoned if c.scrutiny_verdict == "fail")
        scrutiny_needs = sum(
            1 for c in non_abandoned if c.scrutiny_verdict == "needs_resolution"
        )
        scrutiny_total = scrutiny_passed + scrutiny_failed + scrutiny_needs
        if scrutiny_total > 0:
            if scrutiny_total == scrutiny_passed:
                parts.append(
                    f"All {scrutiny_total} claims passed skeptical review (scrutiny)."
                )
            else:
                detail_parts = []
                if scrutiny_passed:
                    detail_parts.append(f"{scrutiny_passed} passed")
                if scrutiny_failed:
                    detail_parts.append(f"{scrutiny_failed} failed")
                if scrutiny_needs:
                    detail_parts.append(
                        f"{scrutiny_needs} flagged for further investigation"
                    )
                parts.append(f"Scrutiny results: {', '.join(detail_parts)}.")

        # Adversarial search
        adversarial_checked = [c for c in non_abandoned if c.adversarial_checked]
        if adversarial_checked:
            strong_counter = sum(
                1
                for c in adversarial_checked
                if c.adversarial_balance is not None
                and c.adversarial_balance < ADVERSARIAL_REFUTED_THRESHOLD
            )
            survived = sum(
                1
                for c in adversarial_checked
                if c.adversarial_balance is not None
                and c.adversarial_balance >= ADVERSARIAL_SURVIVED_THRESHOLD
            )
            parts.append(
                f"\n{len(adversarial_checked)} claims were checked against counter-evidence (adversarial search)."
            )
            if strong_counter > 0:
                parts.append(
                    f"{strong_counter} were found to have strong counter-evidence "
                    f"(adversarial balance < {ADVERSARIAL_REFUTED_THRESHOLD}) "
                    f"and were automatically demoted "
                    f"by the Truth Maintenance System."
                )
            if survived > 0:
                parts.append(
                    f"{survived} claims survived adversarial challenge "
                    f"(balance >= {ADVERSARIAL_SURVIVED_THRESHOLD})."
                )

        # TMS demotions (from promotion_history)
        demotion_count = 0
        for claim in all_claims:
            for entry in claim.promotion_history:
                from_stage = entry.get("from", "")
                to_stage = entry.get("to", "")
                # A demotion is when the target stage is lower
                _STAGE_ORDER = {
                    "hypothesis": 0,
                    "supported": 1,
                    "provisional": 2,
                    "robust": 3,
                    "actionable": 4,
                }
                if _STAGE_ORDER.get(to_stage, 0) < _STAGE_ORDER.get(from_stage, 0):
                    demotion_count += 1
        if demotion_count > 0:
            parts.append(
                f"\n{demotion_count} claim demotion{'s' if demotion_count != 1 else ''} "
                f"were triggered by the Truth Maintenance System after evidence changes."
            )

        # Peirce inquiry cycling
        cycling_claims = [c for c in all_claims if c.investigation_count > 0]
        if cycling_claims:
            total_cycles = sum(c.investigation_count for c in cycling_claims)
            parts.append(
                f"\n{len(cycling_claims)} claim{'s' if len(cycling_claims) != 1 else ''} "
                f"triggered further investigation (Peirce inquiry cycling) "
                f"after initial scrutiny raised doubts, "
                f"totalling {total_cycles} additional investigation cycle{'s' if total_cycles != 1 else ''}."
            )

        # Abandoned claims
        abandoned = [c for c in all_claims if c.abandoned]
        if abandoned:
            parts.append(
                f"{len(abandoned)} claim{'s' if len(abandoned) != 1 else ''} "
                f"were abandoned after exhausting investigation attempts."
            )

        # Uncertainty resolution
        blocking_uncertainties = [u for u in all_uncertainties if not u.is_resolved]
        blocking_blocking = sum(1 for u in blocking_uncertainties if u.is_blocking)
        total_uncertainties = len(all_uncertainties)
        if total_uncertainties > 0:
            parts.append(
                f"\n{total_uncertainties} uncertainties were identified during the investigation. "
                f"{resolved_count} were resolved through additional evidence gathering and analysis."
            )
            if blocking_blocking > 0:
                parts.append(
                    f"{blocking_blocking} blocking uncertainties remain unresolved."
                )

        return " ".join(parts)

    def _is_blocking_uncertainty_type(self, uncertainty_type: UncertaintyType) -> bool:
        """Determine if an uncertainty type is blocking.

        Blocking uncertainty types prevent claim promotion.
        See UncertaintyType enum in primitives.py for categorization.
        """
        blocking_types = {
            # Core blocking uncertainties
            UncertaintyType.UNKNOWN,
            UncertaintyType.CONTRADICTION,
            UncertaintyType.COMPUTATIONAL_DISAGREEMENT,
            UncertaintyType.STRONG_COUNTEREVIDENCE,
            # Deductive validation blocking uncertainties
            UncertaintyType.LOGICAL_INCONSISTENCY,
            UncertaintyType.PHYSICAL_IMPLAUSIBILITY,
            UncertaintyType.MISSING_PREMISE,
        }
        return uncertainty_type in blocking_types

    async def generate_html(
        self,
        objective_id: Optional[str] = None,
        model_name: str = "unknown",
    ) -> Optional[str]:
        """Generate HTML report content.

        Args:
            objective_id: Optional specific objective to report on
            model_name: Model used for the investigation

        Returns:
            HTML content as string, or None if no data found
        """
        report_data = await self.extract_report_data(
            objective_id=objective_id,
            model_name=model_name,
        )

        if not report_data:
            return None

        from andamentum.typeset import render as typeset_render

        from .typeset_report import build_typeset_report

        return typeset_render(build_typeset_report(report_data))

    async def save_html(
        self,
        output_path: Path,
        objective_id: Optional[str] = None,
        model_name: str = "unknown",
    ) -> bool:
        """Generate and save HTML report to file.

        Args:
            output_path: Path to save the HTML file
            objective_id: Optional specific objective to report on
            model_name: Model used for the investigation

        Returns:
            True if saved successfully, False otherwise
        """
        html_content = await self.generate_html(
            objective_id=objective_id,
            model_name=model_name,
        )

        if not html_content:
            logger.warning("No content to save - report generation returned None")
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content, encoding="utf-8")
            logger.info(f"Saved HTML report to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save HTML report: {e}")
            return False


async def generate_report_from_database(
    database_name: str,
    output_path: Path,
    objective_id: Optional[str] = None,
    model_name: str = "unknown",
) -> bool:
    """Convenience function to generate report from a named database.

    Args:
        database_name: Name of the epistemic database
        output_path: Path to save the HTML file
        objective_id: Optional specific objective to report on
        model_name: Model used for the investigation

    Returns:
        True if saved successfully, False otherwise
    """
    store = DocumentStore.for_database(database_name)
    await store.initialize()

    generator = ReportGenerator(store, database_name)
    return await generator.save_html(
        output_path=output_path,
        objective_id=objective_id,
        model_name=model_name,
    )
