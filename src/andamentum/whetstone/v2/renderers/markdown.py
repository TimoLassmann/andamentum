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
    Edit,
    Finding,
    ReviewResult,
    SectionCard,
)

_TITLE = "# Whetstone Review"


def render_markdown(
    result: ReviewResult,
    output_path: str | Path | None = None,
) -> str:
    """Render a ReviewResult as markdown.

    Returns the markdown string. If ``output_path`` is given, also writes
    the string to that path (utf-8) — the path is created if missing.
    """
    sections: list[str] = [_TITLE]

    if result.summary.strip():
        sections.append(result.summary.strip())

    if result.author_questions:
        sections.append(_render_questions(result.author_questions))

    if result.edits:
        sections.append(_render_edits(result.edits))

    llm_findings = list(result.findings)
    if llm_findings:
        sections.append(_render_findings(llm_findings, heading="Findings (LLM-investigated)"))

    if result.deterministic_findings:
        sections.append(
            _render_findings(
                result.deterministic_findings,
                heading="Deterministic findings (structural)",
            )
        )

    if result.document_map:
        sections.append(_render_document_map(result.document_map))

    if len(sections) == 1:
        # Only the title. Be explicit rather than emitting a blank file.
        sections.append("_No findings, edits, or questions — document looks clean._")

    output = "\n\n---\n\n".join(sections) + "\n"

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


def _render_findings(findings: Iterable[Finding], *, heading: str) -> str:
    by_sev: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_sev[f.severity].append(f)

    lines = [f"## {heading} ({sum(len(v) for v in by_sev.values())})", ""]
    for sev in ("major", "moderate", "minor"):
        group = by_sev.get(sev, [])
        if not group:
            continue
        lines.append(f"### {sev.title()} ({len(group)})")
        lines.append("")
        for f in group:
            persona = f" · _{f.perspective}_" if f.perspective else ""
            lines.append(f"- **{f.title}** _({f.confidence} confidence{persona})_")
            if f.rationale:
                lines.append(f"  {f.rationale}")
            if f.sections_involved:
                lines.append(f"  · sections: {', '.join(f.sections_involved)}")
            for q in f.quotes[:3]:
                preview = q.text.replace("\n", " ")[:140]
                lines.append(f"  > [{q.section_id}] {preview}")
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
