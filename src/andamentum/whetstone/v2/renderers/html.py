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
    Edit,
    Finding,
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

    if result.summary.strip():
        atoms.extend(_summary_atoms(result.summary))

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

    if result.document_map:
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


def _findings_atoms(findings: list[Finding], *, heading: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {"kind": "heading", "content": f"{heading} ({len(findings)})", "level": 2}
    ]
    by_sev: dict[str, list[Finding]] = {"major": [], "moderate": [], "minor": []}
    for f in findings:
        by_sev.setdefault(f.severity, []).append(f)
    for sev in ("major", "moderate", "minor"):
        group = by_sev.get(sev, [])
        if not group:
            continue
        out.append({"kind": "heading", "content": f"{sev.title()} ({len(group)})", "level": 3})
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
                    "content": f"**{f.title}** _({f.confidence} confidence{persona})_",
                    "details": "".join(details_lines) if details_lines else None,
                }
            )
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
