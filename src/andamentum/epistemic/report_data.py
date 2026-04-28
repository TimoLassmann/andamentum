"""Schema for epistemic report rendering.

Pure data module — no rendering, no I/O. Holds the dataclasses produced by
``report_generator.extract_report_data`` and consumed by
``typeset_report.build_typeset_report``. Decoupled from any specific output
format so the data extraction layer doesn't have to know which renderer
will run.

This module replaces the dataclass section that used to live in
``html_report.py``. The HTML rendering itself now goes through
``andamentum.typeset`` via the typeset_report adapter.
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


@dataclass
class ConfidenceScores:
    """Answer-level confidence scores for report rendering."""

    posterior: Optional[float] = None
    posterior_supporting: int = 0
    posterior_contradicting: int = 0
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
    "InvestigationStats",
    "QUESTION_TYPE_LABELS",
    "ReportData",
    "UncertaintySummary",
]
