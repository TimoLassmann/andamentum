"""Epistemic Synthesis - Always-produce-results synthesis module.

This module generates results from database state, even when operations have failed.
The key principle is HONEST TRANSPARENCY - we always tell the user what worked,
what didn't, and how confident we are in the results.

Three synthesis modes:
1. Full Results - All operations succeeded, high confidence
2. Partial Results - Some operations failed, medium confidence with caveats
3. No Results - All operations failed, acknowledge and explain

Architecture: Layer 1 (Libraries) - framework-agnostic, no model calls

NOTE: This module is ASYNC - all methods query the database directly
instead of using an in-memory cache. This supports multi-agent architectures
where multiple agents may be operating on the same database.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from enum import Enum

from .primitives import ClaimStage

if TYPE_CHECKING:
    from .repository import EpistemicRepository
    from .primitives import Objective, Evidence, Claim, Uncertainty

# Claims at or above SUPPORTED stage are considered "established" for synthesis
MINIMUM_CLAIM_STAGE_FOR_FINDINGS = ClaimStage.SUPPORTED

# Stage ordering for comparison
STAGE_ORDER = {
    ClaimStage.HYPOTHESIS: 0,
    ClaimStage.SUPPORTED: 1,
    ClaimStage.PROVISIONAL: 2,
    ClaimStage.ROBUST: 3,
    ClaimStage.ACTIONABLE: 4,
}


def _claim_meets_minimum_stage(claim: "Claim", minimum: ClaimStage = MINIMUM_CLAIM_STAGE_FOR_FINDINGS) -> bool:
    """Check if a claim's stage is at or above the minimum for synthesis."""
    return STAGE_ORDER.get(claim.stage, 0) >= STAGE_ORDER.get(minimum, 1)


class ResultConfidence(str, Enum):
    """Confidence level in synthesized results."""

    HIGH = "high"  # 3+ evidence sources, no failures
    MEDIUM = "medium"  # 2+ evidence or 1 evidence + claims
    LOW = "low"  # 1 evidence source only
    NONE = "none"  # No evidence at all


@dataclass
class SynthesisResult:
    """Result of synthesis operation.

    Contains the synthesized content along with transparency metadata
    about what worked and what didn't.
    """

    # Main content
    title: str
    summary: str
    findings: List[str] = field(default_factory=list)

    # Evidence and claims used
    evidence_count: int = 0
    claim_count: int = 0
    evidence_summaries: List[str] = field(default_factory=list)

    # Confidence and limitations
    confidence: ResultConfidence = ResultConfidence.NONE
    limitations: List[str] = field(default_factory=list)

    # Transparency about operations
    operations_succeeded: int = 0
    operations_failed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # What would improve confidence
    improvement_suggestions: List[str] = field(default_factory=list)

    # Structured quality signals for downstream consumers (benchmarks, interpreters)
    quality_signals: Optional[Dict[str, Any]] = field(default=None)

    def to_markdown(self) -> str:
        """Convert to markdown format for fallback display.

        Note: The canonical output is the Artefact produced by
        SynthesizeReportOperation. This method is only used as a
        fallback when no artefact exists (e.g., early operation failures).
        """
        sections = []

        sections.append(f"## {self.title}")
        sections.append("")
        sections.append(self.summary)
        sections.append("")

        if self.findings:
            sections.append("### Key Findings")
            sections.append("")
            for finding in self.findings:
                sections.append(f"- {finding}")
            sections.append("")

        sections.append(f"**Confidence:** {self.confidence.value.upper()} | "
                        f"**Evidence:** {self.evidence_count} | "
                        f"**Claims:** {self.claim_count}")
        sections.append("")

        if self.limitations:
            sections.append("### Limitations")
            sections.append("")
            for limitation in self.limitations:
                sections.append(f"- {limitation}")
            sections.append("")

        if self.errors:
            sections.append("### Issues Encountered")
            sections.append("")
            for error in self.errors:
                sections.append(f"- {error}")
            sections.append("")

        return "\n".join(sections)


class EpistemicSynthesizer:
    """Synthesizes results by querying the database directly.

    The synthesizer ALWAYS produces a result, even when operations fail.
    It provides honest transparency about what worked and what didn't.

    This is an ASYNC synthesizer that queries the database directly,
    supporting multi-agent architectures where multiple agents operate
    on the same database.

    Usage:
        synthesizer = EpistemicSynthesizer()
        result = await synthesizer.synthesize(repo, objective_id)
        print(result.to_markdown())
    """

    async def synthesize(
        self,
        repo: "EpistemicRepository",
        objective_id: str,
    ) -> SynthesisResult:
        """Synthesize results by querying the repository.

        Args:
            repo: EpistemicRepository for querying epistemic entities
            objective_id: ID of the objective to synthesize

        Returns:
            SynthesisResult with synthesized content and metadata
        """
        # Query all entities from repository
        objective = await repo.get_objective(objective_id)
        evidence = await repo.get_evidence_for_objective(objective_id)
        claims = await repo.get_claims_for_objective(objective_id)
        artefacts = await repo.get_artefacts_for_objective(objective_id)
        uncertainties = await repo.get_uncertainties_for_objective(objective_id)

        # Build stats from already-queried entities
        claims_by_stage: Dict[str, int] = {}
        for c in claims:
            stage_name = c.stage.value
            claims_by_stage[stage_name] = claims_by_stage.get(stage_name, 0) + 1
        stats = {
            "total_claims": len(claims),
            "total_evidence": len(evidence),
            "claims_by_stage": claims_by_stage,
        }

        # WorkItem error/warning queries (empty if no event log)
        errors: List[str] = []
        warnings: List[str] = []

        # Build context from queried data
        context = self._build_context(
            evidence=evidence,
            claims=claims,
            stats=stats,
            errors=errors,
            warnings=warnings,
            artefacts=artefacts,
        )

        # Determine synthesis path based on what we have
        if context["has_complete_failure"]:
            result = self._synthesize_no_results(objective, context)
        elif context["has_partial_results"]:
            result = self._synthesize_partial_results(evidence, claims, context)
        else:
            result = self._synthesize_full_results(evidence, claims, context)

        # Always compute quality signals from raw entities
        result.quality_signals = self._compute_quality_signals(claims, evidence, uncertainties)
        return result

    def _compute_quality_signals(
        self,
        claims: List["Claim"],
        evidence: List["Evidence"],
        uncertainties: List["Uncertainty"],
    ) -> Dict[str, Any]:
        """Compute structured quality signals from epistemic entities.

        These signals expose the pipeline's internal quality assessment
        in a machine-readable format for downstream consumers (benchmarks,
        interpretation steps, dashboards).
        """
        # --- Claim signals ---
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

        for claim in claims:
            if stage_order.get(claim.stage, 0) > stage_order.get(ClaimStage(max_stage), 0):
                max_stage = claim.stage.value

            if claim.confidence_score is not None:
                confidence_scores.append(claim.confidence_score)

            verdict = claim.scrutiny_verdict
            if verdict is not None:
                scrutiny_total += 1
                if verdict == "pass":
                    scrutiny_passed += 1

        # --- Evidence signals ---
        quality_scores: list[float] = []
        for ev in evidence:
            if ev.quality_score is not None:
                quality_scores.append(ev.quality_score)

        # --- Uncertainty signals ---
        unresolved = [u for u in uncertainties if not u.is_resolved]
        blocking = [u for u in unresolved if u.is_blocking]

        return {
            "max_stage": max_stage,
            "claims_total": len(claims),
            "scrutiny_pass_rate": (scrutiny_passed / scrutiny_total) if scrutiny_total > 0 else None,
            "mean_confidence_score": (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else None,
            "evidence_count": len(evidence),
            "mean_evidence_quality": (sum(quality_scores) / len(quality_scores)) if quality_scores else None,
            "unresolved_uncertainties": len(unresolved),
            "blocking_uncertainties": len(blocking),
        }

    def _build_context(
        self,
        evidence: List["Evidence"],
        claims: List["Claim"],
        stats: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
        artefacts: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Build synthesis context from queried data.

        Args:
            evidence: List of evidence objects
            claims: List of claim objects
            stats: Stats from get_objective_stats()
            errors: Error messages from event_log
            warnings: Warning messages from event_log

        Returns:
            Context dict for synthesis decisions
        """
        evidence_count = len(evidence)
        claim_count = len(claims)
        operations_completed = stats.get("workitems_done", 0)
        operations_failed = stats.get("workitems_failed", 0)
        uncertainty_count = stats.get("uncertainties_active", 0)

        # Calculate evidence source diversity
        source_types = set()
        for ev in evidence:
            source_types.add(ev.source_type)
        has_external_evidence = bool(source_types - {"world_knowledge"})
        all_world_knowledge = source_types == {"world_knowledge"} or not source_types

        # Calculate claim stage distribution
        promoted_claims = 0
        for claim in claims:
            if claim.stage.value in ("supported", "provisional", "robust", "actionable"):
                promoted_claims += 1

        # Calculate confidence level - epistemically honest
        # HIGH: External evidence, promoted claims, low uncertainty ratio
        # MEDIUM: Some external evidence OR promoted claims
        # LOW: Only world_knowledge, all hypothesis, or high uncertainty
        # NONE: No evidence at all

        if evidence_count == 0:
            confidence_level = "none"
        elif all_world_knowledge:
            # World knowledge only = LOW confidence (no external verification)
            confidence_level = "low"
        elif operations_failed > 0 and not has_external_evidence:
            # Failed operations and no external evidence = LOW
            confidence_level = "low"
        elif promoted_claims == 0:
            # No claims promoted beyond hypothesis = cap at MEDIUM
            if has_external_evidence and uncertainty_count < claim_count * 5:
                confidence_level = "medium"
            else:
                confidence_level = "low"
        elif has_external_evidence and promoted_claims >= claim_count * 0.5:
            # External evidence + most claims promoted = HIGH
            confidence_level = "high"
        elif has_external_evidence or promoted_claims > 0:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Determine synthesis flags
        total_ops = operations_completed + operations_failed
        has_failures = operations_failed > 0
        has_successes = operations_completed > 0
        has_complete_failure = has_failures and not has_successes and total_ops > 0
        has_partial_results = has_failures and has_successes

        # Process artefacts if available
        artefact_count = len(artefacts) if artefacts else 0
        latest_artefact = artefacts[-1] if artefacts else None

        return {
            "evidence_count": evidence_count,
            "claim_count": claim_count,  # Total claims (for proportion calculations)
            "supported_claim_count": promoted_claims,  # SUPPORTED+ claims only
            "operations_completed": operations_completed,
            "operations_failed": operations_failed,
            "uncertainty_count": uncertainty_count,
            "confidence_level": confidence_level,
            "has_complete_failure": has_complete_failure,
            "has_partial_results": has_partial_results,
            "has_external_evidence": has_external_evidence,
            "all_world_knowledge": all_world_knowledge,
            "promoted_claims": promoted_claims,
            "errors": errors,
            "warnings": warnings,
            "artefact_count": artefact_count,
            "latest_artefact": latest_artefact,
        }

    def _synthesize_full_results(
        self,
        evidence: List["Evidence"],
        claims: List["Claim"],
        context: Dict[str, Any],
    ) -> SynthesisResult:
        """Synthesize when all operations succeeded.

        This is the happy path - we have evidence, possibly claims,
        and no operation failures.
        """
        # Determine appropriate title based on evidence quality
        if context.get("all_world_knowledge"):
            title = "Preliminary Analysis (World Knowledge Only)"
        elif context.get("promoted_claims", 0) == 0:
            title = "Research Results (Hypotheses - Pending Verification)"
        elif context.get("has_external_evidence"):
            title = "Research Results"
        else:
            title = "Research Results (Limited Evidence)"

        # Filter claims to SUPPORTED+ stage first
        supported_claims = [c for c in claims if _claim_meets_minimum_stage(c)]

        result = SynthesisResult(
            title=title,
            summary=self._build_summary(evidence, claims, full=True),
            evidence_count=context["evidence_count"],
            claim_count=len(supported_claims),  # Only count SUPPORTED+ claims
            operations_succeeded=context["operations_completed"],
            operations_failed=context["operations_failed"],
            confidence=self._map_confidence(context["confidence_level"]),
        )

        # Add findings from filtered claims
        for claim in supported_claims:
            result.findings.append(claim.statement)

        # Add evidence summaries (de-duplicated by source_ref/URL)
        seen_sources: set[str] = set()
        for ev in evidence:
            # De-duplicate by source_ref (URL)
            if ev.source_ref in seen_sources:
                continue
            seen_sources.add(ev.source_ref)

            summary = f"[{ev.source_type}] {ev.source_ref}"
            if ev.extracted_content:
                # Truncate for summary
                content_preview = ev.extracted_content[:100]
                if len(ev.extracted_content) > 100:
                    content_preview += "..."
                summary += f": {content_preview}"
            result.evidence_summaries.append(summary)

        # Add limitations based on evidence quality
        if context.get("all_world_knowledge"):
            result.limitations.append(
                "All evidence derived from LLM world knowledge only - no external sources verified"
            )
        if context.get("promoted_claims", 0) == 0 and context.get("claim_count", 0) > 0:
            result.limitations.append(
                f"All {context['claim_count']} claims remain at HYPOTHESIS stage - pending verification"
            )
        supported_count = context.get("supported_claim_count", 0)
        if supported_count > 0 and context.get("uncertainty_count", 0) > supported_count * 5:
            result.limitations.append(
                f"High uncertainty ratio: {context['uncertainty_count']} open questions for {supported_count} established claims"
            )

        # Add improvement suggestions even for full results
        result.improvement_suggestions = self._get_improvement_suggestions(context)

        # Add any warnings (filter out "Unknown warning" noise)
        raw_warnings = context.get("warnings", [])
        result.warnings = [w for w in raw_warnings if w != "Unknown warning"]

        return result

    def _synthesize_partial_results(
        self,
        evidence: List["Evidence"],
        claims: List["Claim"],
        context: Dict[str, Any],
    ) -> SynthesisResult:
        """Synthesize when some operations failed.

        We have SOMETHING (evidence or claims) but with failures.
        Be transparent about limitations.
        """
        # Filter claims to SUPPORTED+ stage first
        supported_claims = [c for c in claims if _claim_meets_minimum_stage(c)]

        result = SynthesisResult(
            title="Research Results (Partial)",
            summary=self._build_summary(evidence, claims, full=False),
            evidence_count=context["evidence_count"],
            claim_count=len(supported_claims),  # Only count SUPPORTED+ claims
            operations_succeeded=context["operations_completed"],
            operations_failed=context["operations_failed"],
            confidence=self._map_confidence(context["confidence_level"]),
            errors=context.get("errors", []),
            warnings=context.get("warnings", []),
        )

        # Add findings from filtered claims
        for claim in supported_claims:
            result.findings.append(claim.statement)

        # Add evidence summaries (de-duplicated by source_ref/URL)
        seen_sources: set[str] = set()
        for ev in evidence:
            # De-duplicate by source_ref (URL)
            if ev.source_ref in seen_sources:
                continue
            seen_sources.add(ev.source_ref)

            summary = f"[{ev.source_type}] {ev.source_ref}"
            result.evidence_summaries.append(summary)

        # Add limitations based on what failed
        result.limitations = self._get_limitations(evidence, claims, context)

        # Add improvement suggestions
        result.improvement_suggestions = self._get_improvement_suggestions(context)

        return result

    def _synthesize_no_results(
        self,
        objective: Optional["Objective"],
        context: Dict[str, Any],
    ) -> SynthesisResult:
        """Synthesize when all operations failed.

        We have NOTHING - no evidence, no claims. Be honest about it
        and provide useful information about what went wrong.
        """
        objective_desc = objective.description if objective else ""

        result = SynthesisResult(
            title="Research Results (Unable to Complete)",
            summary=f"Unable to find results for: {objective_desc}\n\nAll research operations encountered errors. See below for details.",
            evidence_count=0,
            claim_count=0,
            operations_succeeded=context["operations_completed"],
            operations_failed=context["operations_failed"],
            confidence=ResultConfidence.NONE,
            errors=context.get("errors", []),
            warnings=context.get("warnings", []),
        )

        # Explain what was attempted
        result.limitations = [
            "No evidence sources could be retrieved",
            "Research operations failed - see Issues Encountered below",
            "Results are unavailable for this query",
        ]

        # Provide actionable suggestions
        result.improvement_suggestions = [
            "Try rephrasing the research question",
            "Check network connectivity if web search was attempted",
            "Review error messages for specific failure reasons",
            "Consider breaking the question into smaller, more specific queries",
        ]

        return result

    def _build_summary(
        self,
        evidence: List["Evidence"],
        claims: List["Claim"],
        full: bool,
    ) -> str:
        """Build a summary paragraph based on evidence and claims.

        Note: Claim statements are shown separately in "Key Findings" section,
        so we only provide context here, not duplicate the findings.
        """
        evidence_count = len(evidence)
        # Only count SUPPORTED+ claims (consistent with findings filtering)
        supported_claims = [c for c in claims if _claim_meets_minimum_stage(c)]
        claim_count = len(supported_claims)

        parts = []

        # Opening based on what we found
        if full:
            if claim_count > 0:
                parts.append(f"Based on {evidence_count} evidence source(s), {claim_count} finding(s) were established:")
            else:
                parts.append(f"Based on {evidence_count} evidence source(s):")
        else:
            parts.append(f"Based on partial evidence ({evidence_count} source(s), some operations failed):")

        # Don't duplicate claims here - they're shown in "Key Findings" section

        # If no claims but have evidence, mention that
        if not claims and evidence:
            parts.append("")
            parts.append("Evidence was collected but no specific claims were established yet.")

        return "\n".join(parts)

    def _get_limitations(
        self,
        evidence: List["Evidence"],
        claims: List["Claim"],
        context: Dict[str, Any],
    ) -> List[str]:
        """Generate limitations list based on failures."""
        limitations = []

        failed_ops = context["operations_failed"]
        if failed_ops > 0:
            limitations.append(f"{failed_ops} operation(s) failed during research")

        if context["evidence_count"] < 3:
            limitations.append(f"Limited evidence sources ({context['evidence_count']} found)")

        if context["claim_count"] == 0 and context["evidence_count"] > 0:
            limitations.append("No claims could be established from available evidence")

        for error in context.get("errors", []):
            # Extract operation name from error if possible
            if ":" in error:
                op_name = error.split(":")[0].strip("[]")
                limitations.append(f"{op_name} encountered errors")

        return limitations

    def _get_improvement_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving confidence."""
        suggestions = []

        if context["evidence_count"] < 3:
            suggestions.append("Additional evidence sources would increase confidence")

        if context["claim_count"] == 0:
            suggestions.append("Establishing specific claims would clarify findings")

        if context["confidence_level"] == "low":
            suggestions.append("More comprehensive research would improve reliability")

        if not suggestions:
            suggestions.append("Consider peer review of findings for additional validation")

        return suggestions

    def _map_confidence(self, level: str) -> ResultConfidence:
        """Map string confidence level to enum."""
        mapping = {
            "high": ResultConfidence.HIGH,
            "medium": ResultConfidence.MEDIUM,
            "low": ResultConfidence.LOW,
            "none": ResultConfidence.NONE,
        }
        return mapping.get(level, ResultConfidence.NONE)


async def synthesize_from_database(
    repo: "EpistemicRepository",
    objective_id: str,
) -> SynthesisResult:
    """Convenience function to synthesize results from repository.

    Usage:
        from andamentum.epistemic.synthesis import synthesize_from_database
        result = await synthesize_from_database(repo, objective_id)
        print(result.to_markdown())
    """
    synthesizer = EpistemicSynthesizer()
    return await synthesizer.synthesize(repo, objective_id)
