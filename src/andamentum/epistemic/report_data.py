"""Schema for epistemic report rendering.

Pure data module — no rendering, no I/O. Holds the dataclasses produced
by ``report_generator.extract_report_data`` and consumed by
``audit_report.build_audit_report``. Decoupled from rendering so the
data-extraction layer doesn't have to know what the renderer does with
the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EvidenceSummary:
    """Summary of evidence for report rendering."""

    evidence_id: str
    source_type: str
    source_ref: str
    extracted_content: str
    limitations: list[str] = field(default_factory=list)
    verified: bool = False
    provider: Optional[str] = None
    support_judgment: Optional[str] = None
    judgment_reasoning: Optional[str] = None
    quality_score: Optional[float] = None


@dataclass
class UncertaintySummary:
    """Summary of uncertainty for report rendering."""

    uncertainty_id: str
    uncertainty_type: str
    description: str
    scope: str
    is_blocking: bool
    is_resolved: bool
    affected_claim_ids: list[str] = field(default_factory=list)


@dataclass
class InvestigationRound:
    """One follow-up investigation round on a claim.

    Each entry is one intent the gap-analysis agent proposed during a
    round when scrutiny said the evidence was insufficient. The agent
    names a methodological angle (mechanism, adversarial, replication,
    methodologically-independent, etc.) and the dispatch layer routes
    it to providers. ``evidence_count`` is the reachability signal —
    how many evidence items the routing found for this angle.
    """

    round_index: int  # 1-based — Round 1, Round 2, …
    intent: str
    evidence_count: int


@dataclass
class GateTraceEntry:
    """One row of the per-claim gate trace.

    The gate trace externalises the deterministic checks the system
    applied to a claim — scrutiny, convergence, adversarial balance,
    deductive validation, computational verification, posterior
    decisiveness. Each row carries a human-readable ``required`` value
    (the threshold), an ``observed`` value, and a status word that the
    renderer surfaces as plain text (``satisfied`` / ``failed`` /
    ``skipped``) with no colour signal.

    This is the load-bearing answer to Schneider (2025)'s
    "black box" challenge: the reasoning steps are *named*, the
    thresholds are *named*, and the observed values are *named*.
    """

    name: str  # "scrutiny", "convergence", "adversarial_balance", ...
    routing: str  # "PRIMARY" | "SECONDARY" | "SKIP"
    required: str  # e.g. "pass", "≥ 2 independent sources", "< 0.70"
    observed: str  # e.g. "pass", "14 independent sources", "0.62"
    status: str  # "satisfied" | "failed" | "skipped"
    note: Optional[str] = None  # one-sentence elaboration


@dataclass
class IBECandidate:
    """One candidate explanation from the IBE chain.

    The IBE (Inference to the Best Explanation) step enumerates 2+
    alternative explanations of the evidence, scores each on
    *loveliness* (how well the explanation fits) and *likeliness*
    (prior probability), and selects the candidate with the best
    combined score as the integrated verdict.
    """

    candidate_id: str  # "A", "B", "C", …
    verdict: str  # supports / contradicts / insufficient / …
    description: str
    loveliness: Optional[float] = None
    loveliness_reasoning: Optional[str] = None
    likeliness: Optional[float] = None
    likeliness_reasoning: Optional[str] = None
    chosen: bool = False
    runner_up: bool = False
    gap_loveliness: Optional[float] = None
    gap_likeliness: Optional[float] = None


@dataclass
class ClaimSummary:
    """Summary of claim for report rendering."""

    claim_id: str
    statement: str
    scope: str
    assumptions: list[str]
    stage: str  # HYPOTHESIS, SUPPORTED, PROVISIONAL, ROBUST, ACTIONABLE
    evidence_ids: list[str] = field(default_factory=list)
    uncertainty_ids: list[str] = field(default_factory=list)
    adversarial_balance: Optional[float] = None
    scrutiny_verdict: Optional[str] = None
    verification_summary: str = ""
    evidence_refs_display: list[int] = field(default_factory=list)
    investigation_rounds: list[InvestigationRound] = field(default_factory=list)
    ibe_candidates: list[IBECandidate] = field(default_factory=list)
    integrated_assessment: Optional[str] = None
    integrated_confidence: Optional[float] = None
    # Per-claim audit data added for v2 report: the verdict label that
    # the renderer puts on the badge (closed vocabulary; see
    # ``audit_report._normalised_verdict``), and the gate trace that
    # externalises every deterministic check the claim was subjected
    # to. Both are populated by ``report_generator`` from claim entity
    # state; the renderer never recomputes them.
    verdict_label: str = ""
    gate_trace: list[GateTraceEntry] = field(default_factory=list)


@dataclass
class AdversarialSummary:
    """Summary of adversarial analysis."""

    claim_id: str
    counterargument: str
    strength: float
    source_ref: str
    rebuttal: Optional[str] = None


@dataclass
class ConvergenceSummary:
    """Summary of cross-domain convergence."""

    domain: str
    supporting_evidence: str
    confidence: float


@dataclass
class InvestigationStats:
    """Statistics about the investigation."""

    total_evidence: int = 0
    total_claims: int = 0
    claims_by_stage: dict[str, int] = field(default_factory=dict)
    blocking_uncertainties: int = 0
    non_blocking_uncertainties: int = 0
    resolved_uncertainties: int = 0
    adversarial_challenges: int = 0
    convergent_domains: int = 0
    # Per-evidence judgement breakdown (the audit-trail view of the
    # 42% / 58% directional / no_bearing split shown in dev30 results).
    evidence_supports: int = 0
    evidence_contradicts: int = 0
    evidence_no_bearing: int = 0
    evidence_invalidated: int = 0
    investigation_rounds_total: int = 0


@dataclass
class ConfidenceScores:
    """Answer-level confidence scores for report rendering."""

    posterior: Optional[float] = None
    posterior_supporting: float = 0.0
    posterior_contradicting: float = 0.0
    posterior_question_type: Optional[str] = None
    terminal_state: str = "completed"


@dataclass
class ReportData:
    """Complete data for report rendering."""

    research_question: str
    clarified_question: str
    investigation_date: datetime
    model_used: str
    database_name: str
    direct_answer: str

    question_type: Optional[str] = None
    verdict: str = ""
    artefact_trace: dict[str, list[str]] = field(default_factory=dict)
    investigation_narrative: str = ""
    evidence_index_map: dict[str, int] = field(default_factory=dict)
    claims: list[ClaimSummary] = field(default_factory=list)
    evidence: list[EvidenceSummary] = field(default_factory=list)
    uncertainties: list[UncertaintySummary] = field(default_factory=list)
    adversarial: list[AdversarialSummary] = field(default_factory=list)
    convergence: list[ConvergenceSummary] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    stats: InvestigationStats = field(default_factory=InvestigationStats)
    confidence_scores: Optional[ConfidenceScores] = None
    # Reproducibility metadata for the v2 audit report. Stable across
    # re-renders of the same persisted state — this is the diachronic-
    # justification answer to Schneider (2025): a reader six months from
    # now can re-run the same command and obtain a comparable artefact.
    snapshot_id: Optional[str] = None
    artefact_id: Optional[str] = None
    pipeline_version: str = ""
    pipeline_git_ref: Optional[str] = None
    reproduction_command: str = ""


# Canonical mapping from ``Objective.question_type`` enum values
# (see ``primitives.QuestionType``) to a human-readable label.
QUESTION_TYPE_LABELS: dict[str, str] = {
    "verificatory": "yes/no factual question",
    "explanatory": "explanation or mechanism question",
    "exploratory": "open-ended exploration",
    "comparative": "comparison question",
    "predictive": "prediction or forecast",
    "compositional": "compositional question (parts/factors)",
    "normative": "value judgment or recommendation",
}


__all__ = [
    "AdversarialSummary",
    "ClaimSummary",
    "ConfidenceScores",
    "ConvergenceSummary",
    "EvidenceSummary",
    "GateTraceEntry",
    "IBECandidate",
    "InvestigationRound",
    "InvestigationStats",
    "QUESTION_TYPE_LABELS",
    "ReportData",
    "UncertaintySummary",
]
