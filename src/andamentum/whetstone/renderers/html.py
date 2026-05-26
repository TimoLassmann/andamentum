"""HTML renderer for ``ReviewResult`` — builds andamentum.typeset atoms.

Same approach as v1's ``whetstone/renderers/html.py``: walk the result
once, turn it into typeset atoms (heading, prose, callout, items, card),
hand the list to ``typeset.render`` for styling. No bespoke CSS — every
visual concern lives in ``andamentum.typeset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from andamentum.typeset import render as typeset_render

from .._watermark import DISCLAIMER_SHORT
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
from ._panel_layout import (
    criterion_summary,
    diverged_criteria,
    headline_line,
    reviewer_scores,
)

_TITLE = "Whetstone Review"
_SUBTITLE = "Structured feedback on a draft"
_DISCLAIMER = DISCLAIMER_SHORT  # Re-export of the shared constant.


def render_html(
    result: ReviewResult,
    output_path: str | Path | None = None,
    *,
    style: str = "article",
    model: str | None = None,
    visible_watermark: bool = True,
) -> str:
    """Render a ReviewResult as a self-contained HTML page.

    Returns the HTML string. If ``output_path`` is given, also writes
    it to that path (utf-8). The ``style`` argument is forwarded to
    ``typeset.render`` (defaults to ``"article"``, the warm-serif
    minimalistic style also used by the epistemic report; ``"report"``
    and ``"cv"`` are the other built-in options).
    """
    atoms: list[dict[str, Any]] = []

    atoms.append({"kind": "heading", "content": _TITLE, "subtitle": _SUBTITLE})
    # Single combined banner — both the "for your own drafts" scope
    # statement and the AI-provenance watermark belong in the same
    # callout. Two stacked callouts at the top of every report was
    # visual noise; one is enough.
    if visible_watermark:
        atoms.append(
            {
                "kind": "callout",
                "tone": "warning",
                "content": (
                    "**AI-generated review content.** "
                    "Whetstone is for sharpening your own drafts — not a "
                    "peer-review tool. See RESPONSIBLE_USE.md."
                ),
            }
        )
    else:
        atoms.append({"kind": "callout", "tone": "note", "content": _DISCLAIMER})
    _ = model  # reserved for future inclusion in the banner text

    panel_mode = bool(result.expert_profiles or result.expert_reviews)

    # section_id → title map (fed into the findings renderer so each
    # per-finding header reads "Methods · s1" not just "s1").
    section_titles = {c.section_id: c.title for c in result.document_map}

    if result.summary.strip():
        atoms.extend(_summary_atoms(result.summary))

    # ── Document map at the TOP — orientation before findings ──────
    if result.document_map and not panel_mode:
        atoms.extend(_document_map_atoms(result.document_map))

    # ── Panel-mode atoms (priority order) ───────────────────────────
    if result.panel_synthesis is not None:
        atoms.extend(
            _panel_synthesis_atoms(result.panel_synthesis, result.expert_reviews)
        )

    if result.expert_reviews:
        atoms.extend(_expert_reviews_atoms(result.expert_reviews))

    if result.expert_profiles:
        atoms.extend(_expert_profiles_atoms(result.expert_profiles))

    # ── Guidelines / custom mode atoms ──────────────────────────────
    if result.guideline_evaluations:
        atoms.extend(_guideline_evaluations_atoms(result.guideline_evaluations))

    if result.custom_evaluations:
        atoms.extend(_custom_evaluations_atoms(result.custom_evaluations))

    # ── Standard review-mode atoms ──────────────────────────────────
    if result.author_questions:
        atoms.extend(_questions_atoms(result.author_questions))

    if result.edits:
        atoms.extend(_edits_atoms(result.edits))

    if result.findings:
        atoms.extend(
            _findings_atoms(
                result.findings,
                heading="Findings",
                section_titles=section_titles,
            )
        )

    if result.deterministic_findings:
        atoms.extend(
            _findings_atoms(
                result.deterministic_findings,
                heading="Deterministic findings (structural)",
                section_titles=section_titles,
            )
        )

    # Count prelude atoms (heading + single combined banner) so the
    # "looks clean" message fires only when there's no review content.
    prelude_atoms = 2
    if len(atoms) <= prelude_atoms:
        atoms.append(
            {
                "kind": "callout",
                "tone": "success",
                "content": "No findings, edits, or questions — document looks clean.",
            }
        )

    html = typeset_render(atoms, style=style, title=_TITLE)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

    return html


# ── Atom builders ──────────────────────────────────────────────────────


def _summary_atoms(summary: str) -> list[dict[str, Any]]:
    """Synthesised review prose. Already markdown-ish; wrap as prose."""
    return [
        {"kind": "heading", "content": "Summary", "level": 2},
        {"kind": "prose", "content": summary},
    ]


def _questions_atoms(qs: list[AuthorQuestion]) -> list[dict[str, Any]]:
    items = []
    for q in qs:
        body_parts = [q.question]
        if q.why:
            body_parts.append(f"_{q.why}_")
        if q.sections_involved:
            body_parts.append(f"sections: {', '.join(q.sections_involved)}")
        items.append({"label": "?", "body": " — ".join(body_parts)})
    return [
        {"kind": "heading", "content": f"Author questions ({len(qs)})", "level": 2},
        {"kind": "items", "entries": items},
    ]


def _edits_atoms(edits: list[Edit]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {"kind": "heading", "content": f"Edits ({len(edits)})", "level": 2}
    ]
    for e in edits:
        details = (
            f"_Original (in section {e.section_id}, chars {e.char_start}–{e.char_end}):_\n\n"
            f"`{e.original_text}`\n\n"
            f"_Proposed:_\n\n"
            f"`{e.new_text}`"
        )
        if e.rationale:
            details += f"\n\n{e.rationale}"
        out.append(
            {
                "kind": "card",
                "content": f"**{e.title}** — {e.severity}, {e.confidence} confidence",
                "details": details,
            }
        )
    return out


_PRIORITY_HEADINGS = {
    "must_fix": "MUST FIX",
    "should_fix": "SHOULD FIX",
    "consider": "CONSIDER",
}


def _findings_atoms(
    findings: list[Finding],
    *,
    heading: str,
    section_titles: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Editorial-annotation layout: per-finding section header →
    ``tone-quote`` callout with the verbatim passage → ``tone-warning``
    callout with the comment body. Mirrors the markdown renderer.

    No collapsed ``<details>``: everything is visible at first read.
    The visual separation between the quote (serif italic, left rule,
    no background) and the comment (sans-serif, accent left bar, tinted
    background) carries the "passage → annotation" reading direction
    without literal indentation.
    """
    section_titles = section_titles or {}
    by_priority: dict[str, list[Finding]] = {
        "must_fix": [],
        "should_fix": [],
        "consider": [],
    }
    for f in findings:
        by_priority.setdefault(f.priority, []).append(f)

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

    out: list[dict[str, Any]] = [
        {
            "kind": "prose",
            "heading": f"{heading} ({len(findings)})",
            "content": f"_{counts_line}_",
        }
    ]

    for priority in ("must_fix", "should_fix", "consider"):
        group = by_priority.get(priority, [])
        if not group:
            continue
        out.append(
            {
                "kind": "prose",
                "content": f"### {_PRIORITY_HEADINGS[priority]} ({len(group)})",
            }
        )
        for f in group:
            section_id = (
                f.quotes[0].section_id
                if f.quotes
                else (f.sections_involved[0] if f.sections_involved else "")
            )
            section_title = section_titles.get(section_id, "(no section)")
            loc = f"#### {section_title}" + (
                f" &nbsp; <code>{section_id}</code>" if section_id else ""
            )
            out.append({"kind": "prose", "content": loc})

            # Verbatim passage (serif italic with left rule via
            # tone-quote — the existing typeset atom for "this is the
            # author's text, set apart from review prose").
            for q in f.quotes[:1]:
                preview = q.text.replace("\n", " ").strip()
                out.append(
                    {
                        "kind": "callout",
                        "tone": "quote",
                        "content": preview,
                    }
                )

            # Comment block. Severity / confidence ride as neutral
            # typeset-badge chips inline. tone-warning for major,
            # tone-note for everything else (keeps the visual weight
            # proportional to the severity).
            tone = "warning" if f.severity == "major" else "note"
            persona = f" · <em>{f.perspective}</em>" if f.perspective else ""
            chips = (
                f'<span class="typeset-badge">{f.severity}</span>'
                f' <span class="typeset-badge">{f.confidence} confidence</span>'
            )
            title_line = f"**{f.title}** &nbsp; {chips}{persona}"
            body = title_line + (f"\n\n{f.rationale}" if f.rationale else "")
            out.append({"kind": "callout", "tone": tone, "content": body})

    return out


def _panel_synthesis_atoms(
    s: PanelSynthesis, reviews: list[ExpertReview]
) -> list[dict[str, Any]]:
    """BLUF: recommendation + score table + consensus/divergence + factors.

    Layout follows the shared markdown body but emits typeset atoms so
    HTML output retains visual hierarchy (callout for headline, items for
    bullets, prose for the long-form synthesis).
    """
    out: list[dict[str, Any]] = [
        {"kind": "heading", "content": "Panel synthesis", "level": 2},
        {"kind": "callout", "tone": "note", "content": headline_line(s)},
    ]
    if s.recommendation_justification.strip():
        out.append({"kind": "prose", "content": s.recommendation_justification})

    rows = reviewer_scores(reviews)
    if rows:
        out.append({"kind": "heading", "content": "Reviewer scores", "level": 3})
        score_lines: list[str] = []
        for row in rows:
            per_expert = " · ".join(f"{name} {score}" for name, score in row.per_expert)
            score_lines.append(
                f"- **{row.name}:** {row.average:.1f} avg, range "
                f"{row.range_str} · {per_expert}"
            )
        out.append({"kind": "prose", "content": "\n".join(score_lines)})

    if s.consensus_strengths:
        out.append({"kind": "heading", "content": "Consensus strengths", "level": 3})
        out.append(
            {
                "kind": "items",
                "entries": [{"label": "+", "body": x} for x in s.consensus_strengths],
            }
        )
    if s.consensus_weaknesses:
        out.append({"kind": "heading", "content": "Consensus weaknesses", "level": 3})
        out.append(
            {
                "kind": "items",
                "entries": [{"label": "-", "body": x} for x in s.consensus_weaknesses],
            }
        )
    if s.divergent_opinions:
        out.append({"kind": "heading", "content": "Divergent opinions", "level": 3})
        out.append(
            {
                "kind": "items",
                "entries": [{"label": "?", "body": x} for x in s.divergent_opinions],
            }
        )

    diverged = diverged_criteria(rows)
    if diverged:
        out.append(
            {"kind": "heading", "content": "Where the panel diverged", "level": 3}
        )
        for row in diverged:
            body = criterion_summary(row.name, s).strip()
            content = f"**{row.name}** (range {row.range_str})."
            if body:
                content += f" {body}"
            out.append({"kind": "prose", "content": content})

    if s.key_decision_factors:
        out.append({"kind": "heading", "content": "Key decision factors", "level": 3})
        out.append(
            {
                "kind": "items",
                "entries": [{"label": "★", "body": x} for x in s.key_decision_factors],
            }
        )

    if s.review_summary.strip():
        out.append({"kind": "heading", "content": "Detailed synthesis", "level": 3})
        out.append({"kind": "prose", "content": s.review_summary})

    return out


def _expert_reviews_atoms(reviews: list[ExpertReview]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "kind": "heading",
            "content": f"Expert reviews ({len(reviews)})",
            "level": 2,
        }
    ]
    for r in reviews:
        details_lines: list[str] = []
        details_lines.append(
            f"**Overall {r.overall_score}/10** — {r.overall_assessment}"
        )
        details_lines.append(
            f"Scores: rigor {r.scientific_rigor_score}/10, "
            f"methodology {r.methodology_score}/10, "
            f"novelty {r.novelty_score}/10, "
            f"clarity {r.clarity_score}/10"
        )
        if r.strengths:
            details_lines.append("**Strengths.** " + "; ".join(r.strengths))
        if r.weaknesses:
            details_lines.append("**Weaknesses.** " + "; ".join(r.weaknesses))
        if r.recommendation_justification.strip():
            details_lines.append(
                f"**Why this recommendation.** {r.recommendation_justification}"
            )
        out.append(
            {
                "kind": "card",
                "content": (
                    f"**{r.expert_name}** ({r.discipline}) — "
                    f"recommendation: **{r.recommendation}** · "
                    f"overall {r.overall_score}/10"
                ),
                "details": "\n\n".join(details_lines),
            }
        )
    return out


def _expert_profiles_atoms(profiles: list[ExpertProfile]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "kind": "heading",
            "content": f"Expert biosketches ({len(profiles)})",
            "level": 2,
        }
    ]
    for p in profiles:
        details = (
            f"**Position.** {p.position}\n\n"
            f"**Education.** {p.education}\n\n"
            f"**Contributions.** {p.contributions}\n\n"
            f"**Research.** {p.research}"
        )
        out.append(
            {
                "kind": "card",
                "content": f"**{p.name}** — {p.discipline}",
                "details": details,
            }
        )
    return out


_STATUS_HEADINGS = {
    "fail": "FAIL",
    "unclear": "UNCLEAR",
    "pass": "PASS",
}

_STATUS_LABELS = {
    "fail": "x",
    "unclear": "?",
    "pass": "o",
}


def _guideline_evaluations_atoms(
    evaluations: list[GuidelineEvaluation],
) -> list[dict[str, Any]]:
    """Group evaluations by status, fail first."""
    out: list[dict[str, Any]] = [
        {
            "kind": "heading",
            "content": f"Journal-guideline checks ({len(evaluations)})",
            "level": 2,
        }
    ]
    by_status: dict[str, list[GuidelineEvaluation]] = {
        "fail": [],
        "unclear": [],
        "pass": [],
    }
    for e in evaluations:
        by_status.setdefault(e.status, []).append(e)
    for status in ("fail", "unclear", "pass"):
        group = by_status.get(status, [])
        if not group:
            continue
        out.append(
            {
                "kind": "heading",
                "content": f"{_STATUS_HEADINGS[status]} ({len(group)})",
                "level": 3,
            }
        )
        entries = []
        for e in group:
            body = f"**{e.item_name}**"
            if e.notes:
                body += f" — {e.notes}"
            if e.category:
                body += f" _({e.category})_"
            entries.append({"label": _STATUS_LABELS[status], "body": body})
        out.append({"kind": "items", "entries": entries})
    return out


def _custom_evaluations_atoms(
    evaluations: list[CustomEvaluation],
) -> list[dict[str, Any]]:
    """Group custom-criteria evaluations by status, fail first."""
    out: list[dict[str, Any]] = [
        {
            "kind": "heading",
            "content": f"Custom-criteria evaluation ({len(evaluations)})",
            "level": 2,
        }
    ]
    by_status: dict[str, list[CustomEvaluation]] = {
        "fail": [],
        "unclear": [],
        "pass": [],
    }
    for e in evaluations:
        by_status.setdefault(e.status, []).append(e)
    for status in ("fail", "unclear", "pass"):
        group = by_status.get(status, [])
        if not group:
            continue
        out.append(
            {
                "kind": "heading",
                "content": f"{_STATUS_HEADINGS[status]} ({len(group)})",
                "level": 3,
            }
        )
        entries = []
        for e in group:
            body = f"**{e.criterion}**"
            if e.notes:
                body += f" — {e.notes}"
            entries.append({"label": _STATUS_LABELS[status], "body": body})
        out.append({"kind": "items", "entries": entries})
    return out


def _document_map_atoms(cards: list[SectionCard]) -> list[dict[str, Any]]:
    items = []
    for c in cards:
        gist = (c.one_line_gist or "").strip()
        body = f"**{c.title}**" + (f" — {gist}" if gist else "")
        items.append({"label": c.section_id, "body": body})
    return [
        {"kind": "heading", "content": f"Document map ({len(cards)})", "level": 2},
        {"kind": "items", "entries": items},
    ]
