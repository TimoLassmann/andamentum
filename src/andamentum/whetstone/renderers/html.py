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

_TITLE = "Whetstone Review"
_SUBTITLE = "Structured feedback on a draft"
_DISCLAIMER = (
    "This report was generated for your own drafts. Whetstone is not a "
    "peer-review tool — do not use it on manuscripts other authors have "
    "sent you confidentially."
)


def render_html(
    result: ReviewResult,
    output_path: str | Path | None = None,
    *,
    style: str = "article",
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
    atoms.append({"kind": "callout", "tone": "note", "content": _DISCLAIMER})

    panel_mode = bool(result.expert_profiles or result.expert_reviews)

    if result.summary.strip():
        atoms.extend(_summary_atoms(result.summary))

    # ── Panel-mode atoms (priority order) ───────────────────────────
    if result.panel_synthesis is not None:
        atoms.extend(_panel_synthesis_atoms(result.panel_synthesis))

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
            _findings_atoms(result.findings, heading="Findings (LLM-investigated)")
        )

    if result.deterministic_findings:
        atoms.extend(
            _findings_atoms(
                result.deterministic_findings,
                heading="Deterministic findings (structural)",
            )
        )

    if result.document_map and not panel_mode:
        atoms.extend(_document_map_atoms(result.document_map))

    if len(atoms) <= 2:  # only heading + disclaimer
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


def _findings_atoms(findings: list[Finding], *, heading: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {"kind": "heading", "content": f"{heading} ({len(findings)})", "level": 2}
    ]
    by_priority: dict[str, list[Finding]] = {
        "must_fix": [],
        "should_fix": [],
        "consider": [],
    }
    for f in findings:
        by_priority.setdefault(f.priority, []).append(f)
    for priority in ("must_fix", "should_fix", "consider"):
        group = by_priority.get(priority, [])
        if not group:
            continue
        out.append(
            {
                "kind": "heading",
                "content": f"{_PRIORITY_HEADINGS[priority]} ({len(group)})",
                "level": 3,
            }
        )
        for f in group:
            persona = f" · _{f.perspective}_" if f.perspective else ""
            details_lines = []
            if f.rationale:
                details_lines.append(f.rationale)
            if f.sections_involved:
                details_lines.append(f"\n\nsections: {', '.join(f.sections_involved)}")
            for q in f.quotes[:3]:
                preview = q.text.replace("\n", " ")[:200]
                details_lines.append(f"\n\n> [{q.section_id}] {preview}")
            out.append(
                {
                    "kind": "card",
                    "content": (
                        f"**{f.title}** _({f.severity}, {f.confidence} confidence"
                        f"{persona})_"
                    ),
                    "details": "".join(details_lines) if details_lines else None,
                }
            )
    return out


def _panel_synthesis_atoms(s: PanelSynthesis) -> list[dict[str, Any]]:
    """Headline + recommendation card + per-criterion items."""
    out: list[dict[str, Any]] = [
        {"kind": "heading", "content": "Panel synthesis", "level": 2},
        {
            "kind": "callout",
            "tone": "note",
            "content": (
                f"**Recommendation: {s.overall_recommendation}** "
                f"(confidence: {s.confidence_level}). "
                f"Average score {s.average_overall_score:.1f}/10 "
                f"(range: {s.score_range}, n={s.number_of_experts})."
            ),
        },
    ]
    if s.recommendation_justification.strip():
        out.append({"kind": "prose", "content": s.recommendation_justification})
    if s.review_summary.strip():
        out.append({"kind": "heading", "content": "Review summary", "level": 3})
        out.append({"kind": "prose", "content": s.review_summary})
    if s.consensus_strengths:
        out.append(
            {"kind": "heading", "content": "Consensus strengths", "level": 3}
        )
        out.append(
            {
                "kind": "items",
                "entries": [{"label": "+", "body": x} for x in s.consensus_strengths],
            }
        )
    if s.consensus_weaknesses:
        out.append(
            {"kind": "heading", "content": "Consensus weaknesses", "level": 3}
        )
        out.append(
            {
                "kind": "items",
                "entries": [{"label": "-", "body": x} for x in s.consensus_weaknesses],
            }
        )
    if s.divergent_opinions:
        out.append(
            {"kind": "heading", "content": "Divergent opinions", "level": 3}
        )
        out.append(
            {
                "kind": "items",
                "entries": [{"label": "?", "body": x} for x in s.divergent_opinions],
            }
        )
    by_criterion = [
        ("Scientific rigor", s.scientific_rigor_summary),
        ("Methodology", s.methodology_summary),
        ("Novelty", s.novelty_summary),
        ("Clarity", s.clarity_summary),
    ]
    if any(body.strip() for _, body in by_criterion):
        out.append({"kind": "heading", "content": "By criterion", "level": 3})
        for label, body in by_criterion:
            if body.strip():
                out.append(
                    {
                        "kind": "card",
                        "content": f"**{label}**",
                        "details": body,
                    }
                )
    if s.key_decision_factors:
        out.append(
            {"kind": "heading", "content": "Key decision factors", "level": 3}
        )
        out.append(
            {
                "kind": "items",
                "entries": [
                    {"label": "★", "body": x} for x in s.key_decision_factors
                ],
            }
        )
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
            details_lines.append(
                "**Strengths.** " + "; ".join(r.strengths)
            )
        if r.weaknesses:
            details_lines.append(
                "**Weaknesses.** " + "; ".join(r.weaknesses)
            )
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
