"""Schema for epistemic report rendering.

Pure data module — no atom building, no I/O. Holds the dataclasses
produced by ``report_generator.extract_report_data`` and consumed by
``audit_report.build_audit_report``. Decoupled from rendering so the
data-extraction layer doesn't have to know what the renderer does with
the data.

It also owns the small display VOCABULARY the report surfaces share —
``QUESTION_TYPE_LABELS`` and, since the doubt signal became visible, the
doubt phrasing below. Vocabulary lives here for the same reason the
verdict labels live in ``entities.verdict``: two renderers describing
one measurement in two different sets of words is how a reader ends up
with two different impressions of the same run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .entities.verdict import EpistemicVerdict
from .thresholds import JUDGMENT_DOUBT_ESCALATION_THRESHOLD


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
    #: Normalized entropy of the direction judge's own belief distribution.
    #: None when not measured (adversarial / pre-Tier-0 items).
    judgment_doubt: Optional[float] = None
    # NOT here, deliberately: the quality second opinion's doubt. Only
    # agent-scored evidence that hit the quality cliff ever has one, so
    # rendering it would mark exactly that subset as suspect and leave
    # the un-instrumented majority looking clean — the partial-
    # instrumentation bias ``tests/test_quality_second_opinion.py``
    # exists to prevent, moved off the gate and onto the reader.
    #: True when an INDEPENDENT judge contradicted this item's direction;
    #: False when it confirmed it; None when no confirmation ran. Three
    #: states, deliberately — "not checked" is not "checked and agreed".
    direction_dissent: Optional[bool] = None


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
    #: Directional-mass-weighted mean entropy of the evidence judgments behind
    #: the reading that produced this claim's verdict — scrutiny's normally,
    #: the cap's for a cycle-capped claim (see ``ClaimVerdict.verdict_doubt``,
    #: which decides). Named for the verdict rather than for scrutiny because
    #: it is rendered beside the verdict badge and must describe the same
    #: reading. None when unmeasured, never 0.0 — so a reader can tell "the
    #: judges were sure" from "nothing was measured".
    verdict_doubt: Optional[float] = None
    verification_summary: str = ""
    evidence_refs_display: list[int] = field(default_factory=list)
    investigation_rounds: list[InvestigationRound] = field(default_factory=list)
    ibe_candidates: list[IBECandidate] = field(default_factory=list)
    integrated_assessment: Optional[str] = None
    integrated_confidence: Optional[float] = None
    # Per-claim audit data added for v2 report: the verdict label that
    # the renderer puts on the badge (closed vocabulary; see
    # ``entities.verdict.display_label``), and the gate trace that
    # externalises every deterministic check the claim was subjected
    # to. Both are populated by ``report_generator``; ``verdict_label``
    # is copied off the run's ``ClaimVerdict`` — the renderer never
    # derives it, which is what stops the badges and the headline
    # disagreeing inside one document.
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
    """Answer-level confidence scores for report rendering.

    A projection of ``ReportData.verdict_object.posterior_report`` — not
    a second computation. The offline report used to re-run
    ``compute_posterior`` to fill this in, which is where the documented
    live-vs-offline ``terminal_state`` divergence came from.
    """

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
    # The run's persisted decision record, loaded from
    # objective.snapshot_id → snapshot.verdict. The renderer reads its
    # labels; it does not re-derive a verdict from the posterior and it
    # no longer scrapes one out of the artefact prose.
    verdict_object: Optional["EpistemicVerdict"] = None
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


# ──────────────────────────────────────────────────────────────────────
# Judgment doubt — the shared phrasing for a measurement that is only
# ever a suspicion-raiser
# ──────────────────────────────────────────────────────────────────────
#
# Doubt is measured all over the pipeline (``EvidenceRef.judgment_doubt``
# / ``quality_doubt``, ``ClaimVerdict.scrutiny_doubt``, the
# ``VerdictProvenance`` counts) and every one of those numbers is on ONE
# scale with ONE cutoff — ``JUDGMENT_DOUBT_ESCALATION_THRESHOLD``, kept
# equal to ``llm_judge.signals.NEEDS_REVIEW_DOUBT_THRESHOLD`` by
# ``tests/test_doubt_scale_single_source.py``. The helpers below are the
# render path's single read of that cutoff, so the HTML report and the
# CLI cannot end up drawing the line in two different places or
# describing it in two different sets of words.


DOUBT_SUSPICION_CAVEAT = (
    "Doubt is a suspicion-raiser, not a score: high doubt is a reason to "
    "look closer, and low or unmeasured doubt is not evidence that the "
    "reading is correct."
)
"""The honesty clause that must travel with any rendered doubt number.

Normalized entropy over a judge's verbalized belief distribution says
how unsure THAT judge sounded. It does not calibrate the judge, so a
rendered 0.05 must never read as "verified" — see
``thresholds.JUDGMENT_DOUBT_ESCALATION_THRESHOLD``.
"""


def doubt_is_suspicious(doubt: Optional[float]) -> bool:
    """True only when doubt was MEASURED at or above the one cutoff.

    ``None`` is not suspicious and it is not reassuring — it means
    nothing was measured, which is why every renderer that calls this
    also has to say something about the unmeasured case rather than
    treating a ``False`` here as "fine".
    """
    return doubt is not None and doubt >= JUDGMENT_DOUBT_ESCALATION_THRESHOLD


def format_doubt(doubt: Optional[float]) -> str:
    """Render one doubt measurement for a table cell or an inline note.

    ``None`` renders as ``not measured`` and never as ``0.00`` — the two
    states are the ones a reader is most likely to conflate, and
    conflating them turns "we never checked" into "the judge was sure".
    """
    if doubt is None:
        return "not measured"
    if doubt_is_suspicious(doubt):
        return f"{doubt:.2f} (unsure)"
    return f"{doubt:.2f}"


def judgment_reliability_line(verdict: Optional[EpistemicVerdict]) -> Optional[str]:
    """One sentence on how reliable the judgments behind a run's verdict were.

    Reads the three ``VerdictProvenance`` reliability counts and returns
    ``None`` when none of them raised anything — the same
    speak-only-on-a-signal rule ``report_generator._scrutiny_reliability_note``
    follows for the per-claim gate trace, so a run that measured nothing
    suspicious renders exactly as it did before doubt was surfaced.

    ``n_doubt_unmeasured_claims`` is reported alongside
    ``n_high_doubt_claims`` and never on its own logic, because a bare
    "0 claims were unsure" reads as "all measured, all sure" when it may
    mean "nothing was measured".
    """
    if verdict is None:
        return None
    prov = verdict.provenance
    clauses: list[str] = []
    total = prov.n_active_claims
    if prov.n_high_doubt_claims:
        clauses.append(
            f"{prov.n_high_doubt_claims} of {total} active claims rested on "
            f"evidence judgments the system measured as internally unsure "
            f"(judgment doubt at or above "
            f"{JUDGMENT_DOUBT_ESCALATION_THRESHOLD:.2f})"
        )
    if prov.n_doubt_unmeasured_claims:
        clauses.append(
            f"{prov.n_doubt_unmeasured_claims} of {total} carried no doubt "
            f"measurement at all"
        )
    if prov.judgment_doubt_escalated:
        clauses.append("the run spent its one doubt-triggered extra round of scrutiny")
    if not clauses:
        return None
    return "Judgment reliability: " + "; ".join(clauses) + ". " + DOUBT_SUSPICION_CAVEAT


__all__ = [
    "AdversarialSummary",
    "ClaimSummary",
    "ConfidenceScores",
    "ConvergenceSummary",
    "DOUBT_SUSPICION_CAVEAT",
    "EvidenceSummary",
    "GateTraceEntry",
    "IBECandidate",
    "InvestigationRound",
    "InvestigationStats",
    "QUESTION_TYPE_LABELS",
    "ReportData",
    "UncertaintySummary",
    "doubt_is_suspicious",
    "format_doubt",
    "judgment_reliability_line",
]
