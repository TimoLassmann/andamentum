"""Markdown renderer for ``ReviewResult``.

Ports the SECTION STRUCTURE of v1's ``whetstone/renderers/diff.py``,
adapted to v2's data shape (Findings, Edits, AuthorQuestions, summary,
document_map). Output is a single markdown string ready to write to a
``.md`` file or paste into any markdown-capable surface.

Section order (each separated by ``---``):
  1. Title + executive summary (from ``ReviewResult.summary``)
  2. Author questions (when present — most actionable)
  3. Edits (one per Edit, as ```diff``` blocks with rationale)
  4. Findings, grouped by severity (major / moderate / minor)
  5. Deterministic findings (high-confidence structural issues)
  6. Document map (so the reader can locate any referenced section_id)

Empty sections are omitted so a clean document produces a clean report.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..schemas import (
    AuthorQuestion,
    CustomEvaluation,
    Edit,
    ExpertProfile,
    ExpertReview,
    Finding,
    GuidelineEvaluation,
    PanelSynthesis,
    ReviewResult,
    SectionCard,
)
from .._watermark import (
    banner_markdown,
    metadata_markdown_comment,
)
from ._panel_layout import body_markdown

_TITLE = "# Whetstone Review"


def render_markdown(
    result: ReviewResult,
    output_path: str | Path | None = None,
    *,
    model: str | None = None,
    visible_watermark: bool = True,
) -> str:
    """Render a ReviewResult as markdown.

    Returns the markdown string. If ``output_path`` is given, also writes
    the string to that path (utf-8) — the path is created if missing.

    ``visible_watermark`` (default True) adds a top-of-file banner
    identifying this as AI-generated review content. The invisible
    HTML-comment metadata header is always written regardless of this
    flag — set it explicitly to False only when producing a derived
    artifact that should not carry a visible banner.
    """
    # Prelude (metadata header + title + optional banner) is appended
    # AFTER the review-content sections are assembled, so the
    # "looks clean" check counts review content only.
    sections: list[str] = []

    panel_mode = bool(result.expert_profiles or result.expert_reviews)

    # section_id → title map, used by the finding renderer to label
    # each per-finding location header (e.g. "Methods · s1").
    section_titles = {c.section_id: c.title for c in result.document_map}

    if result.summary.strip():
        sections.append(result.summary.strip())

    # ── Partial-coverage notice — a criterion crashed and reviewed nothing ──
    if result.failed_criteria:
        names = ", ".join(result.failed_criteria)
        sections.append(
            "> ⚠️ **Partial review.** The following "
            f"{'criterion' if len(result.failed_criteria) == 1 else 'criteria'} "
            f"failed to run and contributed no findings: {names}. "
            "Treat the coverage below as incomplete."
        )

    # ── Document map at the TOP — orientation before findings ──────
    if result.document_map and not panel_mode:
        sections.append(_render_document_map(result.document_map))

    # ── Panel-mode sections (priority order) ─────────────────────────
    if result.panel_synthesis is not None:
        sections.append(
            _render_panel_synthesis(result.panel_synthesis, result.expert_reviews)
        )

    if result.expert_reviews:
        sections.append(_render_expert_reviews(result.expert_reviews))

    if result.expert_profiles:
        sections.append(_render_expert_profiles(result.expert_profiles))

    # ── Guidelines / custom mode sections ───────────────────────────
    if result.guideline_evaluations:
        sections.append(_render_guideline_evaluations(result.guideline_evaluations))

    if result.custom_evaluations:
        sections.append(_render_custom_evaluations(result.custom_evaluations))

    # ── Standard review-mode sections ────────────────────────────────
    if result.author_questions:
        sections.append(_render_questions(result.author_questions))

    if result.edits:
        sections.append(_render_edits(result.edits))

    llm_findings = list(result.findings)
    if llm_findings:
        sections.append(
            _render_findings(
                llm_findings,
                heading="Findings",
                section_titles=section_titles,
            )
        )

    if result.deterministic_findings:
        sections.append(
            _render_findings(
                result.deterministic_findings,
                heading="Deterministic findings (structural)",
                section_titles=section_titles,
            )
        )

    # Fallback: if there's nothing but a panel, still emit the doc map.
    if result.document_map and panel_mode and not sections:
        sections.append(_render_document_map(result.document_map))

    if not sections:
        # No review content. Be explicit rather than emitting a blank file.
        sections.append("_No findings, edits, or questions — document looks clean._")

    # Prepend prelude: invisible metadata header + visible title + optional banner.
    prelude: list[str] = [metadata_markdown_comment(model=model), _TITLE]
    if visible_watermark:
        prelude.append(banner_markdown(model=model))
    output = "\n\n---\n\n".join(prelude + sections) + "\n"

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")

    return output


# ── Section renderers ──────────────────────────────────────────────────


def _render_questions(qs: Iterable[AuthorQuestion]) -> str:
    qs = list(qs)
    lines = [f"## Author questions ({len(qs)})", ""]
    for q in qs:
        lines.append(f"- **{q.question}**")
        if q.why:
            lines.append(f"  _{q.why}_")
        if q.sections_involved:
            lines.append(f"  · sections: {', '.join(q.sections_involved)}")
    return "\n".join(lines)


def _render_edits(edits: Iterable[Edit]) -> str:
    edits = list(edits)
    lines = [f"## Edits ({len(edits)})", ""]
    for e in edits:
        sev = e.severity[0].upper() + e.severity[1:]
        lines.append(f"### {e.title}  · {sev} · {e.confidence} confidence")
        lines.append("")
        lines.append(f"_{e.section_id} ({e.char_start}–{e.char_end})_")
        lines.append("")
        lines.append("```diff")
        lines.extend(f"- {ln}" for ln in e.original_text.splitlines() or [""])
        lines.extend(f"+ {ln}" for ln in e.new_text.splitlines() or [""])
        lines.append("```")
        if e.rationale:
            lines.append(f"> {e.rationale}")
        lines.append("")
    return "\n".join(lines).rstrip()


_PRIORITY_HEADINGS = {
    "must_fix": "MUST FIX",
    "should_fix": "SHOULD FIX",
    "consider": "CONSIDER",
}


def _render_findings(
    findings: Iterable[Finding],
    *,
    heading: str,
    section_titles: dict[str, str] | None = None,
) -> str:
    """Editorial-annotation layout: per-finding section header → quoted
    passage (blockquote) → comment paragraph.

    Mirrors the HTML renderer's layout so the two outputs read the same.
    The section_titles map (section_id → title) comes from the document
    map; missing ids render with a question-mark placeholder.
    """
    findings = list(findings)
    section_titles = section_titles or {}
    by_priority: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_priority[f.priority].append(f)

    counts = [
        f"{len(by_priority.get(p, []))} {label}"
        for p, label in (
            ("must_fix", "must-fix"),
            ("should_fix", "should-fix"),
            ("consider", "consider"),
        )
        if by_priority.get(p)
    ]
    counts_line = " · ".join(counts) if counts else "0 findings"
    lines = [f"## {heading} ({len(findings)})", "", f"_{counts_line}_", ""]

    for priority in ("must_fix", "should_fix", "consider"):
        group = by_priority.get(priority, [])
        if not group:
            continue
        lines.append(f"### {_PRIORITY_HEADINGS[priority]} ({len(group)})")
        lines.append("")
        for f in group:
            # Per-finding location header — pulls the section title from
            # the document map so the reader sees "Methods · s1" rather
            # than an opaque "s1".
            section_id = (
                f.quotes[0].section_id
                if f.quotes
                else (f.sections_involved[0] if f.sections_involved else "")
            )
            section_title = section_titles.get(section_id, "(no section)")
            loc = f"#### {section_title}" + (f" · `{section_id}`" if section_id else "")
            lines.append(loc)
            lines.append("")

            # Quoted passage — the thing being commented on. Verbatim
            # span from the source, wrapped in a blockquote so it
            # visually leads the comment that follows.
            for q in f.quotes[:1]:
                preview = q.text.replace("\n", " ").strip()
                lines.append(f"> {preview}")
                lines.append("")

            # Comment — title row with severity / confidence chips,
            # then body. Severity / confidence ride as bracketed
            # markers so plain-markdown readers see them as text.
            persona = f" · _{f.perspective}_" if f.perspective else ""
            chips = f"[{f.severity}] [{f.confidence} confidence]{persona}"
            lines.append(f"**{f.title}** &nbsp; {chips}")
            if f.rationale:
                lines.append("")
                lines.append(f.rationale)
            lines.append("")
    return "\n".join(lines).rstrip()


def _render_panel_synthesis(s: PanelSynthesis, reviews: list[ExpertReview]) -> str:
    """Top-of-report panel synthesis with score table + consensus/divergence."""
    return "## Panel synthesis\n\n" + body_markdown(s, reviews)


def _render_expert_reviews(reviews: Iterable[ExpertReview]) -> str:
    """Per-expert reviews. Collapsible details via blockquote indentation."""
    reviews = list(reviews)
    lines = [f"## Expert reviews ({len(reviews)})", ""]
    for r in reviews:
        lines += [
            f"### {r.expert_name} — {r.discipline}",
            "",
            f"**Overall: {r.overall_score}/10** · "
            f"Recommendation: **{r.recommendation}**",
            "",
            r.overall_assessment,
            "",
            "| Criterion | Score | Justification |",
            "| --- | --- | --- |",
            f"| Scientific rigor | {r.scientific_rigor_score}/10 | "
            f"{_oneline(r.scientific_rigor_justification)} |",
            f"| Methodology | {r.methodology_score}/10 | "
            f"{_oneline(r.methodology_justification)} |",
            f"| Novelty | {r.novelty_score}/10 | {_oneline(r.novelty_justification)} |",
            f"| Clarity | {r.clarity_score}/10 | {_oneline(r.clarity_justification)} |",
            "",
        ]
        if r.strengths:
            lines += ["**Strengths**", ""]
            lines += [f"- {item}" for item in r.strengths]
            lines += [""]
        if r.weaknesses:
            lines += ["**Weaknesses**", ""]
            lines += [f"- {item}" for item in r.weaknesses]
            lines += [""]
        if r.recommendation_justification.strip():
            lines += [
                f"_{r.recommendation_justification.strip()}_",
                "",
            ]
    return "\n".join(lines).rstrip()


def _render_expert_profiles(profiles: Iterable[ExpertProfile]) -> str:
    """Footnote-style biosketches at the bottom of the report."""
    profiles = list(profiles)
    lines = [f"## Expert biosketches ({len(profiles)})", ""]
    for p in profiles:
        lines += [
            f"### {p.name} — {p.discipline}",
            "",
            f"**Position.** {p.position}",
            "",
            f"**Education.** {p.education}",
            "",
            f"**Contributions.** {p.contributions}",
            "",
            f"**Research.** {p.research}",
            "",
        ]
    return "\n".join(lines).rstrip()


def _oneline(s: str) -> str:
    """Squash newlines + collapse whitespace so the markdown table doesn't break."""
    return " ".join(s.split())


_STATUS_HEADINGS = {
    "fail": "FAIL",
    "unclear": "UNCLEAR",
    "pass": "PASS",
}


def _render_guideline_evaluations(
    evaluations: Iterable[GuidelineEvaluation],
) -> str:
    """Group evaluations by status (fail / unclear / pass), fail first."""
    evaluations = list(evaluations)
    by_status: dict[str, list[GuidelineEvaluation]] = {
        "fail": [],
        "unclear": [],
        "pass": [],
    }
    for e in evaluations:
        by_status.setdefault(e.status, []).append(e)

    lines = [f"## Journal-guideline checks ({len(evaluations)})", ""]
    for status in ("fail", "unclear", "pass"):
        group = by_status.get(status, [])
        if not group:
            continue
        lines.append(f"### {_STATUS_HEADINGS[status]} ({len(group)})")
        lines.append("")
        for e in group:
            cat = f" · _{e.category}_" if e.category else ""
            lines.append(f"- **{e.item_name}**{cat}")
            if e.notes:
                lines.append(f"  {e.notes}")
            lines.append("")
    return "\n".join(lines).rstrip()


def _render_custom_evaluations(
    evaluations: Iterable[CustomEvaluation],
) -> str:
    """Group custom-criteria evaluations by status, fail first."""
    evaluations = list(evaluations)
    by_status: dict[str, list[CustomEvaluation]] = {
        "fail": [],
        "unclear": [],
        "pass": [],
    }
    for e in evaluations:
        by_status.setdefault(e.status, []).append(e)

    lines = [f"## Custom-criteria evaluation ({len(evaluations)})", ""]
    for status in ("fail", "unclear", "pass"):
        group = by_status.get(status, [])
        if not group:
            continue
        lines.append(f"### {_STATUS_HEADINGS[status]} ({len(group)})")
        lines.append("")
        for e in group:
            lines.append(f"- **{e.criterion}**")
            if e.notes:
                lines.append(f"  {e.notes}")
            lines.append("")
    return "\n".join(lines).rstrip()


def _render_document_map(cards: Iterable[SectionCard]) -> str:
    cards = list(cards)
    lines = [f"## Document map ({len(cards)} sections)", ""]
    for c in cards:
        gist = (c.one_line_gist or "").strip()
        gist_part = f" — {gist}" if gist else ""
        lines.append(f"- **{c.section_id}** {c.title}{gist_part}")
    return "\n".join(lines)
