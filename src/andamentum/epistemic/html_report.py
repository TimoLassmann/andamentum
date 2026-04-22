"""
HTML Report Generator for Epistemic System.

Layer 1 utility - framework-agnostic HTML generation.
Produces standalone HTML reports with inline CSS matching a clean academic paper style.

CRITICAL: This module NEVER truncates data. All claims, evidence, uncertainties,
and other information are rendered in full.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import html
import re


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
    evidence_refs_display: list[int] = field(default_factory=list)  # sequential numbers


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
    """Answer-level confidence scores for report rendering.

    Decoupled from confidence.py models to keep html_report at Layer 1.
    """

    # Posterior (None if not applicable)
    posterior: Optional[float] = None
    posterior_supporting: int = 0
    posterior_contradicting: int = 0
    posterior_question_type: Optional[str] = None
    terminal_state: str = "completed"


@dataclass
class ReportData:
    """Complete data for HTML report generation."""

    # Header
    research_question: str
    clarified_question: str
    investigation_date: datetime
    model_used: str
    database_name: str

    # Executive Summary (from artefact)
    direct_answer: str

    # Question type (from objective)
    question_type: Optional[str] = None

    # Verdict (one-sentence bottom line)
    verdict: str = ""

    artefact_trace: dict[str, list[str]] = field(default_factory=dict)

    # Investigation narrative (deterministic, built from entity state)
    investigation_narrative: str = ""

    # Evidence renumbered sequentially (old_id -> new_index)
    evidence_index_map: dict[str, int] = field(default_factory=dict)

    # Claims
    claims: list[ClaimSummary] = field(default_factory=list)

    # Evidence
    evidence: list[EvidenceSummary] = field(default_factory=list)

    # Uncertainties
    uncertainties: list[UncertaintySummary] = field(default_factory=list)

    # Adversarial Analysis
    adversarial: list[AdversarialSummary] = field(default_factory=list)

    # Convergence
    convergence: list[ConvergenceSummary] = field(default_factory=list)

    # Open Questions
    open_questions: list[str] = field(default_factory=list)

    # Statistics
    stats: InvestigationStats = field(default_factory=InvestigationStats)

    # Confidence scores (computed post-hoc, may be None if computation failed)
    confidence_scores: Optional[ConfidenceScores] = None


# Article-style CSS — warm, readable, centered content
_CSS_STYLES = """
* {
    box-sizing: border-box;
}

body {
    font-family: 'Source Serif 4', 'Georgia', 'Times New Roman', serif;
    background: #f9f7f4;
    color: #2b2b2b;
    line-height: 1.85;
    margin: 0;
    padding: 0;
    font-size: 19px;
    -webkit-font-smoothing: antialiased;
}

/* Centered article column */
.report-layout {
    max-width: 860px;
    margin: 0 auto;
    padding: 60px 24px 40px;
}

.main-content {
    max-width: none;
}

/* Typography */
h1 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 32px;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1.25;
    margin: 0 0 12px;
    letter-spacing: -0.3px;
}

h2 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 22px;
    font-weight: 700;
    color: #1a1a1a;
    margin: 48px 0 16px;
}

h3 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 18px;
    font-weight: 600;
    color: #2b2b2b;
    margin: 32px 0 10px;
}

p {
    margin: 0 0 16px;
}

blockquote {
    margin: 20px 0;
    padding: 0 0 0 20px;
    border-left: 2px solid #d4d0c8;
    color: #555;
    font-style: italic;
}

/* Report header */
.report-header {
    margin-bottom: 40px;
    padding-bottom: 24px;
    border-bottom: 1px solid #e8e4de;
}

.report-meta {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 13px;
    color: #999;
    margin-top: 8px;
}

.clarified-question {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 14px;
    color: #666;
    margin-top: 10px;
}

.question-type {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 13px;
    color: #999;
    margin-top: 4px;
}

/* Verdict — reads as the natural opening, not a banner */
.verdict {
    font-size: 20px;
    font-weight: 400;
    line-height: 1.65;
    margin: 0 0 32px;
    color: #1a1a1a;
}

/* Key Findings Q&A */
.key-findings-qa {
    font-family: 'Inter', system-ui, sans-serif;
    margin: 0 0 40px;
    padding: 20px 24px;
    background: #f4f1ec;
    border-radius: 6px;
    font-size: 14px;
    line-height: 1.6;
}

.key-findings-qa dl {
    margin: 0;
}

.key-findings-qa dt {
    font-weight: 600;
    font-size: 13px;
    color: #555;
    margin-top: 12px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.key-findings-qa dt:first-child {
    margin-top: 0;
}

.key-findings-qa dd {
    margin: 4px 0 0 0;
    font-size: 16px;
    color: #555;
}

/* Sections */
section + section {
    margin-top: 24px;
}

/* Executive summary */
.summary-content p {
    line-height: 1.85;
}

/* Claims: clean text blocks */
.claim {
    margin: 16px 0;
    padding: 0 0 16px;
    border-bottom: 1px solid #f0f0f0;
}

.claim:last-child {
    border-bottom: none;
}

.claim-statement {
    line-height: 1.75;
}

.claim-stage-label {
    font-size: 15px;
    color: #999;
    margin: 8px 0;
}

.claim-details {
    margin-top: 8px;
}

.claim-details summary {
    font-size: 14px;
    color: #999;
    cursor: pointer;
    font-family: 'Inter', system-ui, sans-serif;
}

.claim-details summary:hover {
    color: #555;
}

.claim-details[open] summary {
    margin-bottom: 8px;
}

.claim-detail {
    font-size: 16px;
    color: #555;
    margin-top: 6px;
    line-height: 1.7;
}

.claim-detail-label {
    font-weight: 600;
    color: #777;
}

.claim-citations {
    font-family: 'Inter', system-ui, sans-serif;
}

.cite-link {
    color: #999;
    text-decoration: none;
}

.cite-link:hover {
    color: #333;
    text-decoration: underline;
}

.cite-link sup {
    font-size: 12px;
}

.section-intro {
    color: #777;
    font-size: 17px;
    margin-bottom: 24px;
}

.claim-assumptions {
    margin-top: 8px;
    padding-left: 16px;
    border-left: 2px solid #eee;
    font-size: 13px;
    color: #666;
}

.claim-assumptions ul {
    margin: 4px 0 0;
    padding-left: 16px;
}

.claim-assumptions li {
    margin-bottom: 2px;
}

/* Counterarguments inside claim details */
.claim-counterarguments {
    margin-top: 10px;
    padding-left: 14px;
    border-left: 2px solid #e0dbd4;
}

.counterargument-item {
    padding: 6px 0;
    border-bottom: 1px solid #f0ede8;
}

.counterargument-item:last-child {
    border-bottom: none;
}

.counterargument-text {
    font-size: 15px;
    color: #555;
    line-height: 1.65;
}

.counterargument-meta {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 11px;
    color: #999;
    margin-top: 3px;
    display: flex;
    gap: 12px;
    align-items: center;
}

.counterargument-strength {
    font-weight: 500;
}

a.counterargument-source {
    color: #999;
    text-decoration: none;
}

a.counterargument-source:hover {
    color: #555;
    text-decoration: underline;
}

/* Evidence: compact */
.evidence-item {
    font-size: 17px;
    margin: 12px 0;
    padding: 12px 0 12px 40px;
    border-bottom: 1px solid #ebe7e0;
    position: relative;
}

.evidence-item:last-child {
    border-bottom: none;
}

.evidence-header {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.evidence-judgment {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    padding: 1px 6px;
    border-radius: 3px;
}

.judgment-supports { color: #1a7a3a; background: #eef7f0; }
.judgment-contradicts { color: #b91c1c; background: #fef2f2; }
.judgment-neutral { color: #666; background: #f5f5f5; }

.evidence-quality {
    font-size: 11px;
    color: #999;
}

.evidence-reasoning {
    margin: 6px 0;
    line-height: 1.75;
    color: #333;
    font-size: 17px;
}

.evidence-source {
    font-size: 12px;
    color: #888;
    word-break: break-all;
}

a.evidence-source {
    color: #666;
    text-decoration: none;
}

a.evidence-source:hover {
    color: #333;
    text-decoration: underline;
}

.evidence-limitations {
    font-size: 12px;
    color: #999;
    font-style: italic;
    margin-top: 2px;
}

.evidence-expand {
    margin-top: 4px;
}
.evidence-expand summary {
    cursor: pointer;
    color: #888;
    font-size: 12px;
}
.evidence-expand summary:hover {
    color: #333;
}
.evidence-expand div {
    white-space: pre-wrap;
    margin-top: 8px;
}

.evidence-source {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
    word-break: break-all;
}

.evidence-index {
    position: absolute;
    left: 0;
    top: 12px;
    font-size: 15px;
    font-weight: 600;
    color: #999;
    font-family: 'Inter', system-ui, sans-serif;
}

.evidence-type-tag {
    font-size: 11px;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

.evidence-limitations {
    margin-top: 4px;
    font-size: 12px;
    color: #999;
    font-style: italic;
}

/* Uncertainties: compact text */
.uncertainty-item {
    padding: 8px 0;
    line-height: 1.75;
    border-bottom: 1px solid #ebe7e0;
}

.uncertainty-item:last-child {
    border-bottom: none;
}

.uncertainty-description {
    color: #2b2b2b;
}

.uncertainty-meta {
    font-size: 12px;
    color: #888;
    margin-top: 2px;
}

/* Open questions: numbered list */
.open-questions-list {
    padding-left: 20px;
    margin: 8px 0 0;
}

.open-questions-list li {
    line-height: 1.75;
    margin-bottom: 10px;
    padding-left: 4px;
}

/* Collapsible sections */
details {
    margin-top: 8px;
}

details summary {
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: #666;
    padding: 4px 0;
}

details summary:hover {
    color: #1a1a1a;
}

/* Trace links */
.trace-link {
    font-size: 14px;
    color: #999;
    text-decoration: none;
    font-family: 'Inter', system-ui, sans-serif;
}

.trace-link:hover {
    color: #1a1a1a;
    text-decoration: underline;
}

/* Code and pre blocks */
code {
    font-family: 'SF Mono', Monaco, Consolas, monospace;
    font-size: 0.875em;
    background: #f5f5f5;
    padding: 1px 4px;
    border-radius: 2px;
}

pre {
    background: #f5f5f5;
    padding: 12px 16px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.5;
}

pre code {
    background: none;
    padding: 0;
}

/* Footer */
.report-footer {
    max-width: 960px;
    margin: 48px auto 0;
    padding-top: 16px;
    border-top: 1px solid #eee;
    font-size: 12px;
    color: #999;
}

/* Metadata section — small, unobtrusive, at the bottom */
.sidebar {
    font-family: 'Inter', system-ui, sans-serif;
    margin-top: 64px;
    padding-top: 24px;
    border-top: 1px solid #e8e4de;
    font-size: 11px;
    line-height: 1.5;
    color: #bbb;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
}

.sidebar-section { margin-bottom: 0; }

.sidebar-title {
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #ccc;
    margin: 0 0 4px;
}

.sidebar-value {
    font-size: 13px;
    font-weight: 600;
    color: #999;
}

.sidebar-label {
    font-size: 11px;
    color: #bbb;
}

.sidebar-row {
    display: flex;
    justify-content: space-between;
    padding: 1px 0;
    font-size: 11px;
}

.sidebar-row-label { color: #bbb; }
.sidebar-row-value { font-weight: 500; color: #999; }

.check-list {
    list-style: none;
    padding: 0;
    margin: 3px 0 0;
}

.check-list li {
    font-size: 10px;
    padding: 1px 0;
    display: flex;
    align-items: center;
    gap: 3px;
    color: #bbb;
}

.check-pass { color: #bbb; }
.check-fail { color: #c4a0a0; }

.check-icon {
    flex-shrink: 0;
    width: 10px;
    text-align: center;
    font-size: 9px;
}

/* Print */
@media print {
    body { padding: 0; font-size: 11pt; background: #fff; }
    .report-layout { padding: 0; }
    .sidebar { display: none; }
    section { page-break-inside: avoid; }
}

/* Responsive */
@media (max-width: 600px) {
    .report-layout { padding: 32px 16px; }
    h1 { font-size: 26px; }
    .sidebar { grid-template-columns: 1fr; }
}
"""


def _escape(text: str) -> str:
    """Escape HTML entities in text."""
    return html.escape(str(text)) if text else ""


def _markdown_to_html(text: str) -> str:
    """Convert markdown to HTML for report rendering.

    Handles: headings, bold, italic, lists, horizontal rules, links,
    and paragraphs. Escapes HTML first for safety, then applies
    markdown transformations.
    """
    if not text:
        return ""

    lines = str(text).split("\n")
    out: list[str] = []
    in_list = False
    list_type = ""  # "ul" or "ol"

    for line in lines:
        stripped = line.strip()

        # Blank line — close list if open, skip
        if not stripped:
            if in_list:
                out.append(f"</{list_type}>")
                in_list = False
            continue

        # Horizontal rule
        if re.match(r"^-{3,}$", stripped) or re.match(r"^\*{3,}$", stripped):
            if in_list:
                out.append(f"</{list_type}>")
                in_list = False
            out.append("<hr>")
            continue

        # Headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            if in_list:
                out.append(f"</{list_type}>")
                in_list = False
            level = len(heading_match.group(1))
            heading_text = _inline_markdown(html.escape(heading_match.group(2)))
            out.append(f"<h{level}>{heading_text}</h{level}>")
            continue

        # Blockquote
        if stripped.startswith("> "):
            if in_list:
                out.append(f"</{list_type}>")
                in_list = False
            quote_text = _inline_markdown(html.escape(stripped[2:]))
            out.append(f"<blockquote>{quote_text}</blockquote>")
            continue

        # Unordered list item
        ul_match = re.match(r"^[-*+]\s+(.+)$", stripped)
        if ul_match:
            if not in_list or list_type != "ul":
                if in_list:
                    out.append(f"</{list_type}>")
                out.append("<ul>")
                in_list = True
                list_type = "ul"
            item_text = _inline_markdown(html.escape(ul_match.group(1)))
            out.append(f"<li>{item_text}</li>")
            continue

        # Ordered list item
        ol_match = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if ol_match:
            if not in_list or list_type != "ol":
                if in_list:
                    out.append(f"</{list_type}>")
                out.append("<ol>")
                in_list = True
                list_type = "ol"
            item_text = _inline_markdown(html.escape(ol_match.group(1)))
            out.append(f"<li>{item_text}</li>")
            continue

        # Regular paragraph line
        if in_list:
            out.append(f"</{list_type}>")
            in_list = False
        para_text = _inline_markdown(html.escape(stripped))
        out.append(f"<p>{para_text}</p>")

    if in_list:
        out.append(f"</{list_type}>")

    return "\n".join(out)


def _inline_markdown(text: str) -> str:
    """Apply inline markdown formatting (bold, italic, links, code)."""
    # Code: `text`
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Bold: **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic: *text*
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def _stage_class(stage: str) -> str:
    """Get CSS class for claim stage."""
    return stage.lower().replace("_", "-")


def _format_datetime(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%Y-%m-%d %H:%M")


STAGE_LABELS = {
    "hypothesis": "Under investigation",
    "supported": "Supported by evidence",
    "provisional": "Provisionally established",
    "robust": "Well-established",
    "actionable": "Established — meets criteria for action",
}

QUESTION_TYPE_LABELS = {
    "verificatory": "yes/no factual question",
    "explanatory": "explanatory question (why/how)",
    "exploratory": "exploratory question (what's involved)",
    "comparative": "comparative question (which is better)",
    "predictive": "predictive question (what will happen)",
    "compositional": "analytical question (what are the parts)",
    "normative": "normative question (should we)",
}


def _claim_status_label(stage: str, adversarial_balance: Optional[float]) -> str:
    """Get plain-English claim status label."""
    stage_lower = stage.lower()
    if (
        stage_lower == "hypothesis"
        and adversarial_balance is not None
        and adversarial_balance < 0.3
    ):
        return "Challenged by counter-evidence"
    return STAGE_LABELS.get(stage_lower, stage)


def _render_counterargument(ca: AdversarialSummary) -> str:
    """Render a single counterargument item."""
    source_html = ""
    if ca.source_ref:
        source_html = f'<a href="{_escape(ca.source_ref)}" class="counterargument-source" target="_blank">{_escape(ca.source_ref)}</a>'

    return f"""
                <div class="counterargument-item">
                    <div class="counterargument-text">{_escape(ca.counterargument)}</div>
                    <div class="counterargument-meta">
                        <span class="counterargument-strength">weight: {ca.strength:.2f}</span>
                        {source_html}
                    </div>
                </div>"""


def _render_claim(
    claim: ClaimSummary,
    evidence_index_map: dict[str, int],
    counterarguments: list[AdversarialSummary] | None = None,
) -> str:
    """Render a single claim as a finding card."""
    status_label = _claim_status_label(claim.stage, claim.adversarial_balance)

    # Evidence references as superscript numbers
    evidence_refs_html = ""
    if claim.evidence_refs_display:
        ref_links = [
            f'<a href="#evidence-{idx}" class="cite-link">{idx}</a>'
            for idx in claim.evidence_refs_display
        ]
        evidence_refs_html = f'<sup class="claim-citations">{",".join(ref_links)}</sup>'

    # Build the statement with inline citations — superscript before final period
    statement_text = claim.statement.rstrip()
    if evidence_refs_html:
        if statement_text.endswith("."):
            statement_html = _escape(statement_text[:-1]) + evidence_refs_html + "."
        else:
            statement_html = _escape(statement_text) + evidence_refs_html
    else:
        statement_html = _escape(statement_text)

    # Expandable detail section
    detail_parts: list[str] = []

    if claim.scope and claim.scope.lower() not in ("general", "specific", ""):
        detail_parts.append(
            f'<div class="claim-detail"><span class="claim-detail-label">Scope:</span> {_escape(claim.scope)}</div>'
        )

    if claim.verification_summary:
        detail_parts.append(
            f'<div class="claim-detail"><span class="claim-detail-label">Verification:</span> {_escape(claim.verification_summary)}</div>'
        )

    # Adversarial story — if checked, show what happened
    if claim.adversarial_balance is not None:
        balance = claim.adversarial_balance
        if balance < 0.3:
            adv_text = f"Counter-evidence search found strong opposition (balance: {balance:.2f}). This claim was demoted after adversarial challenge."
        elif balance >= 0.8:
            adv_text = f"Counter-evidence search found no significant opposition (balance: {balance:.2f})."
        else:
            adv_text = (
                f"Counter-evidence search found mixed results (balance: {balance:.2f})."
            )
        detail_parts.append(
            f'<div class="claim-detail"><span class="claim-detail-label">Adversarial:</span> {adv_text}</div>'
        )

    # Counterargument details — show what the opposition actually said
    if counterarguments:
        ca_items = "".join(_render_counterargument(ca) for ca in counterarguments)
        detail_parts.append(f'<div class="claim-counterarguments">{ca_items}</div>')

    detail_html = ""
    if detail_parts:
        detail_html = f"""
            <details class="claim-details">
                <summary>Details</summary>
                {"".join(detail_parts)}
            </details>"""

    return f"""
        <div class="claim" id="claim-{_escape(claim.claim_id)}">
            <div class="claim-statement">{statement_html}</div>
            <div class="claim-stage-label">{_escape(status_label)}</div>
            {detail_html}
        </div>"""


def _render_evidence_item(evidence: EvidenceSummary, index: int) -> str:
    """Render a single evidence item in the bibliography."""
    # Source type badge
    type_tag = _escape(evidence.source_type)
    if evidence.provider:
        type_tag += f" via {_escape(evidence.provider)}"

    # Quality indicator
    quality_html = ""
    if evidence.quality_score is not None:
        quality_html = f'<span class="evidence-quality">quality: {evidence.quality_score:.2f}</span>'

    # Judgment badge
    judgment_html = ""
    if evidence.support_judgment:
        judgment_class = {
            "supports": "judgment-supports",
            "contradicts": "judgment-contradicts",
            "no_bearing": "judgment-neutral",
        }.get(evidence.support_judgment, "")
        judgment_html = f'<span class="evidence-judgment {judgment_class}">{_escape(evidence.support_judgment)}</span>'

    # Reasoning
    reasoning_html = ""
    if evidence.judgment_reasoning:
        reasoning_html = f'<div class="evidence-reasoning">{_escape(evidence.judgment_reasoning)}</div>'

    # Source as clickable link
    ref = evidence.source_ref
    if ref.startswith("http"):
        source_html = (
            f'<a href="{_escape(ref)}" class="evidence-source">{_escape(ref)}</a>'
        )
    else:
        source_html = f'<span class="evidence-source">{_escape(ref)}</span>'

    # Limitations
    limitations_html = ""
    if evidence.limitations:
        lim_text = "; ".join(_escape(lim) for lim in evidence.limitations)
        limitations_html = f'<div class="evidence-limitations">{lim_text}</div>'

    return f"""
        <div class="evidence-item" id="evidence-{index}">
            <div class="evidence-header">
                <span class="evidence-index">{index}.</span>
                <span class="evidence-type-tag">{type_tag}</span>
                {judgment_html}
                {quality_html}
            </div>
            {reasoning_html}
            {source_html}
            {limitations_html}
        </div>"""


def _render_evidence_bibliography(
    evidence: list[EvidenceSummary], evidence_index_map: dict[str, int]
) -> str:
    """Render evidence as a bibliography grouped by judgment."""
    supporting = [e for e in evidence if e.support_judgment == "supports"]
    contradicting = [e for e in evidence if e.support_judgment == "contradicts"]

    parts: list[str] = []
    if supporting:
        parts.append("<h3>Supporting</h3>")
        for e in supporting:
            idx = evidence_index_map.get(e.evidence_id, 0)
            parts.append(_render_evidence_item(e, idx))
    if contradicting:
        parts.append("<h3>Contradicting</h3>")
        for e in contradicting:
            idx = evidence_index_map.get(e.evidence_id, 0)
            parts.append(_render_evidence_item(e, idx))

    # Any remaining judged evidence not in supports/contradicts
    other = [
        e
        for e in evidence
        if e.support_judgment and e.support_judgment not in ("supports", "contradicts")
    ]
    if other:
        parts.append("<h3>Other</h3>")
        for e in other:
            idx = evidence_index_map.get(e.evidence_id, 0)
            parts.append(_render_evidence_item(e, idx))

    return "\n".join(parts)


def _render_uncertainty(uncertainty: UncertaintySummary) -> str:
    """Render a single uncertainty item — description only, no raw IDs."""
    return f"""
        <div class="uncertainty-item">
            <div class="uncertainty-description">{_escape(uncertainty.description)}</div>
        </div>"""


def _render_sidebar(data: ReportData) -> str:
    """Render the floating metadata sidebar."""
    sections: list[str] = []

    # 1. Posterior Confidence
    if data.confidence_scores is not None:
        scores = data.confidence_scores

        if scores.posterior is not None:
            total = scores.posterior_supporting + scores.posterior_contradicting
            if total > 0:
                evidence_line = f"{scores.posterior_supporting} claims supported / {scores.posterior_contradicting} contradicted"
            else:
                evidence_line = "No directional evidence"

            sections.append(f"""
            <div class="sidebar-section">
                <div class="sidebar-title">Posterior Confidence</div>
                <div class="sidebar-value">{scores.posterior:.2%} confident</div>
                <div class="sidebar-label">{_escape(evidence_line)}</div>
            </div>""")
        elif scores.posterior_question_type and scores.posterior_question_type not in (
            "verificatory",
            "comparative",
            "predictive",
        ):
            sections.append(f"""
            <div class="sidebar-section">
                <div class="sidebar-title">Posterior Confidence</div>
                <div class="sidebar-label">N/A ({_escape(scores.posterior_question_type)} question)</div>
            </div>""")

    # 2. Investigation Stats
    stats = data.stats
    supported = (
        stats.claims_by_stage.get("SUPPORTED", 0)
        + stats.claims_by_stage.get("ROBUST", 0)
        + stats.claims_by_stage.get("ACTIONABLE", 0)
    )
    hypothesis = stats.claims_by_stage.get("HYPOTHESIS", 0) + stats.claims_by_stage.get(
        "PROVISIONAL", 0
    )
    total_uncertainties = (
        stats.blocking_uncertainties
        + stats.non_blocking_uncertainties
        + stats.resolved_uncertainties
    )

    # Question type in sidebar
    question_type_html = ""
    if data.question_type:
        qt_label = _escape(
            QUESTION_TYPE_LABELS.get(data.question_type, data.question_type)
        )
        question_type_html = f'<div class="sidebar-row"><span class="sidebar-row-label">Type</span><span class="sidebar-row-value">{qt_label}</span></div>'

    sections.append(f"""
            <div class="sidebar-section">
                <div class="sidebar-title">Investigation</div>
                {question_type_html}
                <div class="sidebar-row"><span class="sidebar-row-label">Evidence</span><span class="sidebar-row-value">{stats.total_evidence}</span></div>
                <div class="sidebar-row"><span class="sidebar-row-label">Claims</span><span class="sidebar-row-value">{stats.total_claims}</span></div>
                <div class="sidebar-row"><span class="sidebar-row-label" style="padding-left: 8px;">supported</span><span class="sidebar-row-value">{supported}</span></div>
                <div class="sidebar-row"><span class="sidebar-row-label" style="padding-left: 8px;">hypothesis</span><span class="sidebar-row-value">{hypothesis}</span></div>
                <div class="sidebar-row"><span class="sidebar-row-label">Uncertainties</span><span class="sidebar-row-value">{total_uncertainties}</span></div>
                <div class="sidebar-row"><span class="sidebar-row-label" style="padding-left: 8px;">blocking</span><span class="sidebar-row-value">{stats.blocking_uncertainties}</span></div>
                <div class="sidebar-row"><span class="sidebar-row-label" style="padding-left: 8px;">resolved</span><span class="sidebar-row-value">{stats.resolved_uncertainties}</span></div>
            </div>""")

    return f"""
        <aside class="sidebar">
            {"".join(sections)}
        </aside>"""


def _build_key_findings_qa(data: ReportData) -> str:
    """Build the Key Findings Q&A section from deterministic data."""
    # What was studied?
    studied = (
        data.clarified_question
        if data.clarified_question != data.research_question
        else data.research_question
    )

    # What did we find?
    stats = data.stats
    supported_count = (
        stats.claims_by_stage.get("SUPPORTED", 0)
        + stats.claims_by_stage.get("PROVISIONAL", 0)
        + stats.claims_by_stage.get("ROBUST", 0)
        + stats.claims_by_stage.get("ACTIONABLE", 0)
    )
    challenged_count = sum(
        1
        for c in data.claims
        if c.stage.lower() == "hypothesis"
        and c.adversarial_balance is not None
        and c.adversarial_balance < 0.3
    )

    findings_parts: list[str] = []
    # Lead with the verdict if available
    if data.verdict:
        findings_parts.append(data.verdict)
    if supported_count > 0:
        findings_parts.append(
            f"{supported_count} claim{'s' if supported_count != 1 else ''} supported by evidence"
        )
    if challenged_count > 0:
        findings_parts.append(f"{challenged_count} challenged by counter-evidence")
    hypothesis_only = stats.claims_by_stage.get("HYPOTHESIS", 0) - challenged_count
    if hypothesis_only > 0:
        findings_parts.append(f"{hypothesis_only} under investigation")
    claims_summary = (
        ". ".join(findings_parts) if findings_parts else "No claims established"
    )

    # How confident are we?
    confidence_str = "Not assessed"
    if data.confidence_scores is not None:
        sc = data.confidence_scores
        confidence_str = (
            f"Posterior: {sc.posterior:.2%}"
            if sc.posterior is not None
            else "No posterior computed"
        )

    # How thorough was the investigation?
    thoroughness = f"{stats.total_evidence} evidence sources examined"

    return f"""
            <section class="key-findings-qa">
                <dl>
                    <dt>What was studied?</dt><dd>{_escape(studied)}</dd>
                    <dt>What did we find?</dt><dd>{_escape(claims_summary)}</dd>
                    <dt>How confident are we?</dt><dd>{_escape(confidence_str)}</dd>
                    <dt>How thorough was the investigation?</dt><dd>{_escape(thoroughness)}</dd>
                </dl>
            </section>"""


def render(data: ReportData) -> str:
    """
    Render a complete HTML report from ReportData.

    This function generates a standalone HTML file with all CSS inlined.
    Presents findings like a systematic review summary: verdict banner,
    key findings Q&A, per-claim cards, evidence bibliography, and a
    collapsible investigation process narrative.

    CRITICAL: No data is truncated - all claims, evidence, and uncertainties
    are rendered in full.

    Args:
        data: Complete report data extracted from epistemic database

    Returns:
        Complete HTML document as string
    """
    evidence_index_map = data.evidence_index_map

    # Build claim_id → counterarguments mapping for inline rendering
    adversarial_by_claim: dict[str, list[AdversarialSummary]] = {}
    for adv in data.adversarial:
        adversarial_by_claim.setdefault(adv.claim_id, []).append(adv)

    # Render sidebar
    sidebar_html = _render_sidebar(data)

    # Verdict banner
    verdict_html = ""
    if data.verdict:
        verdict_html = f'<div class="verdict">{_escape(data.verdict)}</div>'

    # Key Findings Q&A
    key_findings_html = _build_key_findings_qa(data)

    # Summary (LLM-written answer)
    # Strip sections that duplicate our structured rendering (Findings,
    # Evidence Sources, etc.) — we render those from structured data.
    summary_html = ""
    if data.direct_answer:
        summary_html = f"""
            <section id="summary">
                <h2>Summary</h2>
                <div class="summary-content">{_markdown_to_html(data.direct_answer)}</div>
            </section>"""

    # Findings (per-claim cards) — with intro text
    claims_html = ""
    if data.claims:
        claims_items = "".join(
            _render_claim(c, evidence_index_map, adversarial_by_claim.get(c.claim_id))
            for c in data.claims
        )
        n_claims = len(data.claims)
        claims_html = f"""
        <section id="claims">
            <h2>Findings</h2>
            <p class="section-intro">The investigation produced {n_claims} distinct finding{"s" if n_claims != 1 else ""}, each traced to specific evidence sources. Findings are ordered by strength of support.</p>
            {claims_items}
        </section>"""

    # Evidence bibliography (grouped, sequential numbering)
    evidence_html = ""
    if data.evidence:
        bib_content = _render_evidence_bibliography(data.evidence, evidence_index_map)
        evidence_html = f"""
        <section id="evidence">
            <h2>Sources</h2>
            {bib_content}
        </section>"""

    # Limitations (non-blocking + blocking uncertainties)
    limitations_html = ""
    if data.uncertainties:
        blocking = [
            u for u in data.uncertainties if u.is_blocking and not u.is_resolved
        ]
        non_blocking = [
            u for u in data.uncertainties if not u.is_blocking and not u.is_resolved
        ]

        uncertainty_items: list[str] = []
        if blocking:
            uncertainty_items.append("<h3>Unresolved blocking issues</h3>")
            uncertainty_items.extend(_render_uncertainty(u) for u in blocking)
        if non_blocking:
            uncertainty_items.append("<h3>Caveats</h3>")
            uncertainty_items.extend(_render_uncertainty(u) for u in non_blocking)

        if uncertainty_items:
            limitations_html = f"""
        <section id="limitations">
            <h2>Limitations</h2>
            {"".join(uncertainty_items)}
        </section>"""

    # Open questions — filter out raw markdown artifacts and empty items
    open_questions_html = ""
    if data.open_questions:
        clean_questions = []
        for q in data.open_questions:
            q = q.strip()
            # Skip raw markdown artifacts (blockquotes, headers, just punctuation)
            if not q or q.startswith(">") or q.startswith("#") or len(q) < 10:
                continue
            # Strip leading markdown formatting
            if q.startswith("**") and "**" in q[2:]:
                q = q.replace("**", "")
            clean_questions.append(q)

        if clean_questions:
            questions_items = "".join(f"<li>{_escape(q)}</li>" for q in clean_questions)
            open_questions_html = f"""
        <section id="open-questions">
            <h2>Open Questions</h2>
            <ol class="open-questions-list">
                {questions_items}
            </ol>
        </section>"""

    # Investigation process (collapsible)
    investigation_html = ""
    if data.investigation_narrative:
        investigation_html = f"""
        <details>
            <summary><h2 style="display:inline">Investigation Process</h2></summary>
            <div class="details-content">
                {_markdown_to_html(data.investigation_narrative)}
            </div>
        </details>"""

    # Question type in header
    question_type_html = ""
    if data.question_type:
        qt_label = QUESTION_TYPE_LABELS.get(data.question_type, data.question_type)
        question_type_html = (
            f'<div class="question-type">This is a {_escape(qt_label)}</div>'
        )

    # Meta line
    meta_parts = [
        _format_datetime(data.investigation_date),
        _escape(data.model_used),
        _escape(data.database_name),
    ]
    meta_line = " &middot; ".join(meta_parts)

    # Build complete HTML document
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Epistemic Report: {_escape(data.research_question[:100])}</title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap" rel="stylesheet">

    <style>
{_CSS_STYLES}
    </style>
</head>
<body>
    <div class="report-layout">
        <div class="main-content">
            <header class="report-header">
                <h1>{_escape(data.research_question)}</h1>
                <div class="report-meta">{meta_line}</div>
                {question_type_html}
            </header>

            <main>
                {verdict_html}
                {key_findings_html}
                {summary_html}
                {claims_html}
                {evidence_html}
                {limitations_html}
                {open_questions_html}
                {investigation_html}
            </main>
        </div>

        {sidebar_html}
    </div>

    <footer class="report-footer">
        <p>Generated by andamentum.epistemic</p>
    </footer>
</body>
</html>
"""
