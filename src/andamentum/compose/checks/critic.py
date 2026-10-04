"""Critic: an independent review of the rendered draft, mapped back to sections.

The draft is rendered to a .docx inside ``<bundle>/reviews/round-NN/``, reviewed
by whetstone (checklist or review mode, per genre), and each ``major`` or
``moderate`` finding becomes a work item on the bundle section it is about.
Whetstone sectionises the .docx by its Heading 1 styles, so whetstone sections
map to bundle sections by title.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from andamentum.compose.bundle import Bundle, Section
from andamentum.whetstone import (
    Finding,
    ReviewResult,
    checklist_document,
    open_source,
    review_document,
    reviewer_profile,
)

from ..genres import Genre
from ..log import ActionLog
from ..render import shipped_view
from ..schemas import CheckResult, Fact, Source, WorkItem

ACTIONABLE = ("major", "moderate")


class CriticResult(BaseModel):
    check: CheckResult
    items: list[WorkItem]
    major: int
    moderate: int
    review_path: Path


def round_dir(bundle: Bundle, *, round_no: int) -> Path:
    return bundle.path / "reviews" / f"round-{round_no:02d}"


def _bundle_section_for(title: str, *, sections: Sequence[Section]) -> Section | None:
    """Bundle section for a whetstone section title: exact, then casefolded, then prefix."""
    wanted = title.strip().casefold()
    for s in sections:
        if s.title.strip().casefold() == wanted:
            return s
    for s in sections:
        t = s.title.strip().casefold()
        if t and (wanted.startswith(t) or t.startswith(wanted)):
            return s
    return None


def section_starts(
    text: str, *, sections: Sequence[Section]
) -> list[tuple[int, Section]]:
    """Offsets of each bundle section's heading line in the reviewed text, in order.

    Whetstone nests Heading 1 sections under the document's Title paragraph, so a
    short document comes back as ONE whetstone section and its own section ids say
    nothing about which bundle section a finding is about. The heading lines are
    still in the text, so a finding's quote offsets locate its section.
    """
    starts: list[tuple[int, Section]] = []
    cursor = 0
    for section in sorted(sections, key=lambda s: s.order):
        pattern = re.compile(
            rf"^{re.escape(section.title.strip())}[ \t]*$", re.MULTILINE
        )
        match = pattern.search(text, cursor)
        if match is None:
            continue
        starts.append((match.start(), section))
        cursor = match.end()
    return starts


def _section_at(
    offset: int, *, starts: Sequence[tuple[int, Section]]
) -> Section | None:
    found: Section | None = None
    for start, section in starts:
        if start <= offset:
            found = section
    return found


def _sections_of(
    finding: Finding,
    *,
    whetstone_titles: dict[str, str],
    sections: Sequence[Section],
    starts: Sequence[tuple[int, Section]],
) -> list[Section]:
    """Bundle sections a finding is about: by quote offset first, then by title."""
    mapped: list[Section] = []

    def add(section: Section | None) -> None:
        if section is not None and section not in mapped:
            mapped.append(section)

    for quote in finding.quotes:
        add(_section_at(quote.char_start, starts=starts))
    if mapped:
        return mapped
    ids = list(
        dict.fromkeys(
            finding.sections_involved + [q.section_id for q in finding.quotes]
        )
    )
    for wid in ids:
        title = whetstone_titles.get(wid)
        if title:
            add(_bundle_section_for(title, sections=sections))
    return mapped


def map_findings(
    findings: Sequence[Finding],
    *,
    whetstone_titles: dict[str, str],
    sections: Sequence[Section],
    text: str = "",
) -> list[WorkItem]:
    """Findings of major or moderate severity as work items.

    A finding about one or more known sections becomes one section item per
    section; a finding with no section, or whose sections do not map to a bundle
    section, becomes an outline-level item. *text* is the reviewed document text
    (for locating quotes).
    """
    starts = section_starts(text, sections=sections) if text else []
    items: list[WorkItem] = []
    for finding in findings:
        if finding.severity not in ACTIONABLE:
            continue
        reason = f"[{finding.severity}] {finding.title}: {finding.rationale}"
        if finding.fix:
            reason += f" Fix: {finding.fix}"
        mapped = _sections_of(
            finding, whetstone_titles=whetstone_titles, sections=sections, starts=starts
        )
        if not mapped:
            items.append(
                WorkItem(
                    level="outline", section_id=None, reason=reason, source="critic"
                )
            )
        for section in mapped:
            items.append(
                WorkItem(
                    level="section",
                    section_id=section.id,
                    reason=reason,
                    source="critic",
                )
            )
    return items


async def run_critic(
    bundle: Bundle,
    *,
    genre: Genre,
    model: str,
    round_no: int,
    log: ActionLog,
    facts: Sequence[Fact] = (),
    sources: Sequence[Source] = (),
) -> CriticResult:
    """Render the draft, have whetstone review it, and turn findings into work items.

    The draft is rendered exactly as it ships (citations and Sources list
    included for genres that cite), so the critic judges the delivered text.
    A finding about the Sources list has no bundle section and becomes an
    outline-level item.

    The ReviewResult JSON is saved beside the rendered draft. The brief's call
    text is not passed to whetstone: it steers drafting, and a guidelines review
    would change what the cascade measures.
    """
    directory = round_dir(bundle, round_no=round_no)
    directory.mkdir(parents=True, exist_ok=True)
    draft = directory / "draft.docx"
    with shipped_view(bundle, genre=genre, facts=facts, sources=sources) as view:
        view.bundle.render(
            draft,
            format="docx",
            text_filter=view.text_filter,
            image_root=bundle.path,
        )
        shipped_sections = view.bundle.sections()

    reviewer = reviewer_profile(genre.reviewer) if genre.reviewer else None
    document = await open_source(str(draft))
    try:
        if genre.review == "checklist":
            result: ReviewResult = await checklist_document(
                document,
                model=model,
                document_type=genre.whetstone_document_type,
                confirm_own_draft=True,
                reviewer=reviewer,
            )
        else:
            result = await review_document(
                document,
                model=model,
                document_type=genre.whetstone_document_type,
                confirm_own_draft=True,
                editor=False,
                check_novelty=False,
                flag_novelty=False,
                reviewer=reviewer,
            )
        titles = {s.id: s.title for s in document.sections}
        reviewed_text = document.text
    finally:
        document.cleanup()

    review_path = directory / "review.json"
    review_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    log.call(
        node="Check",
        role="critic",
        model=model,
        prompt=f"whetstone:{genre.review}:{genre.whetstone_document_type}",
        input_text=draft.name,
        output_text=result.model_dump_json(),
    )

    real = {s.id for s in bundle.sections()}
    items = [
        item
        if item.section_id is None or item.section_id in real
        else item.model_copy(update={"level": "outline", "section_id": None})
        for item in map_findings(
            result.findings,
            whetstone_titles=titles,
            sections=shipped_sections,
            text=reviewed_text,
        )
    ]
    major = sum(1 for f in result.findings if f.severity == "major")
    moderate = sum(1 for f in result.findings if f.severity == "moderate")
    return CriticResult(
        check=CheckResult(
            name="critic",
            passed=major == 0,
            details=list(
                dict.fromkeys(i.reason for i in items if "[major]" in i.reason)
            ),
        ),
        items=items,
        major=major,
        moderate=moderate,
        review_path=review_path,
    )
