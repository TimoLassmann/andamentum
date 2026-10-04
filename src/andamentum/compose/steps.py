"""Deterministic helpers the orchestrator calls between model steps.

Workers in the dialect's sense: no graph imports, no model calls. Bundle edits
here are idempotent, so a resumed run can re-enter them safely.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from andamentum.scribe import Bundle, Section

from .evidence import stored_source_path
from .log import ActionLog, sha
from .render import strip_markers
from .schemas import (
    CheckResult,
    FigurePlan,
    Outline,
    RoundRecord,
    Roles,
    SectionPlan,
    Source,
    StopReason,
)

SUMMARY_SENTENCES = 2
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_FIGURE_LINE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\{#fig:[^}]*\}\s*$", re.MULTILINE)


def first_sentences(body: str, *, count: int = SUMMARY_SENTENCES) -> str:
    """The first *count* sentences of a body, markers removed (no model call)."""
    text = " ".join(
        line.strip()
        for line in strip_markers(_FIGURE_LINE.sub("", body)).splitlines()
        if line.strip()
    )
    return " ".join(_SENTENCE_END.split(text)[:count])


def document_hash(sections: Sequence[Section]) -> str:
    """One fingerprint of the whole document text, in section order."""
    ordered = sorted(sections, key=lambda s: s.order)
    return sha("\n".join(f"{s.title}\n{s.body}" for s in ordered))


def plan_for(outline: Outline, *, title: str) -> SectionPlan:
    """The plan for the section called *title*; a bare plan when there is none."""
    for plan in outline.sections:
        if plan.title == title:
            return plan
    return SectionPlan(
        title=title, purpose="", key_points=[], fact_ids=[], target_words=0
    )


def sync_sections(bundle: Bundle, outline: Outline) -> None:
    """Make the bundle's sections match the outline's titles, in order.

    Sections the outline dropped are removed; new titles are added as
    ``planned``; kept sections keep their text. Idempotent.
    """
    wanted = list(dict.fromkeys(plan.title for plan in outline.sections))
    existing = {s.title: s for s in bundle.sections()}
    for title, section in existing.items():
        if title not in wanted:
            bundle.remove_section(section.id)
    for title in wanted:
        if title not in existing:
            bundle.add_section(title=title, status="planned")
    bundle.reorder([bundle.section_by_title(t).id for t in wanted])


def add_figure(
    bundle: Bundle,
    section_id: str,
    *,
    figure: FigurePlan,
    sources: Sequence[Source],
) -> str | None:
    """Render the planned figure into the section. Returns a problem text or None.

    A bad chart request (unknown column, empty data) is reported, not hidden:
    the caller records it as an open question and the log keeps it.
    """
    from andamentum.figures.scribe_glue import insert_figure

    source = next((s for s in sources if s.id == figure.source_id), None)
    if source is None:
        return f"figure {figure.label}: unknown source {figure.source_id}"
    data = stored_source_path(bundle.path, source).read_text(encoding="utf-8")
    try:
        insert_figure(
            bundle,
            section_id,
            caption=figure.caption,
            label=figure.label,
            data=data,
            kind=figure.kind,
            x=figure.x,
            y=figure.y,
        )
    except (ValueError, KeyError) as exc:
        return f"figure {figure.label} could not be drawn: {exc}"
    return None


def figure_lines(body: str) -> list[str]:
    return _FIGURE_LINE.findall(body)


def keep_figures(old_body: str, new_body: str) -> str:
    """Re-append figure paragraphs a rewrite dropped, so revision never loses them."""
    missing = [line for line in figure_lines(old_body) if line not in new_body]
    if not missing:
        return new_body
    return new_body.rstrip() + "\n\n" + "\n\n".join(line.strip() for line in missing)


def write_report(
    bundle_dir: str | Path,
    *,
    title: str,
    stop_reason: StopReason,
    rounds: Sequence[RoundRecord],
    open_questions: Sequence[str],
    roles: Roles,
    usage: dict[str, int],
    calls: dict[str, int],
    outputs: Sequence[Path],
) -> Path:
    """Write ``<bundle>/report.md``: what ran, what was checked, what is left."""
    lines = [f"# Compose report: {title}", ""]
    lines += [f"Stop reason: **{stop_reason}**", ""]
    lines += ["## Outputs", ""]
    lines += [f"- {p.name}" for p in outputs] or ["- none written"]
    lines += ["", "## Open questions", ""]
    lines += [f"- {q}" for q in open_questions] or ["- none"]
    lines += ["", "## Rounds", ""]
    for record in rounds:
        verdict = "goal met" if record.goal_met else "goal not met"
        lines += [f"### Round {record.round}: {verdict}", ""]
        for check in record.checks:
            lines += _check_lines(check)
        if record.changed_sections:
            lines += ["", f"Revised: {', '.join(record.changed_sections)}"]
        lines.append("")
    lines += ["## Models and usage", ""]
    for role in ("drafter", "critic", "judge", "reader"):
        used = usage.get(role, 0)
        tokens = f"{used} tokens" if used else "tokens not reported"
        lines.append(
            f"- {role}: {getattr(roles, role)}, {calls.get(role, 0)} calls, {tokens}"
        )
    path = Path(bundle_dir) / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _check_lines(check: CheckResult) -> list[str]:
    where = f" [{check.section_id}]" if check.section_id else ""
    status = "pass" if check.passed else "FAIL"
    out = [f"- {check.name}{where}: {status}"]
    if not check.passed:
        out += [f"    - {d}" for d in check.details]
    return out


def call_counts(log: ActionLog) -> dict[str, int]:
    """Model calls per role, from the action log."""
    counts: dict[str, int] = {}
    for record in log.records():
        role = record.get("role")
        if isinstance(role, str):
            counts[role] = counts.get(role, 0) + 1
    return counts
