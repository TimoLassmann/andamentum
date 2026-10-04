"""Planning: outline the document from the brief and the facts, then check it.

``make_outline`` and ``amend_outline`` are model calls; ``check_outline`` is
deterministic. A failed check gets one retry with the problems fed back, then
the outline proceeds and the remaining problems are logged as warnings.
"""

from __future__ import annotations

from ._call import call_model
from .genres import Genre
from .log import ActionLog
from .schemas import Brief, CheckResult, Fact, Outline, Source, WorkItem

MAX_OUTLINE_AMENDMENTS = 1
MAX_STATEMENT_CHARS = 300
WORD_TOLERANCE_NOTE = (
    "The target_words of all sections must add up inside the word range."
)

OUTLINE_INSTRUCTIONS = """\
You plan a document. Return an outline: an ordered list of sections. For each \
section give:
- title: the section heading.
- purpose: what this section must achieve for the reader, in one sentence.
- key_points: the specific points it makes, in order. Every item listed under \
MUST INCLUDE appears, worded the same, in the key_points of exactly the section \
where it belongs.
- fact_ids: the ids of the facts this section relies on (only ids from the \
FACTS list; none if the section needs no fact).
- target_words: a whole number; across all sections the targets sum to a value \
inside the stated word range.
- figure: leave empty unless a tabular source is listed and a chart of it \
helps the reader; then give source_id (a listed tabular source id), kind \
(bar, line or scatter), x and y (column names), caption and label (a short \
lowercase slug).

Follow the genre skeleton: use its section titles. Where the skeleton is marked \
flexible you may add, rename or drop sections that are not required, and when \
GUIDELINES are given the guidelines decide the sections. Use facts only for what \
they support.
"""

AMEND_INSTRUCTIONS = (
    OUTLINE_INSTRUCTIONS
    + """\

You are amending an existing outline, not starting again. Change only what the \
listed problems require (add, remove, rename or reorder sections, or move key \
points); keep every other section as it is. Return the full amended outline.
"""
)


def _facts_block(facts: list[Fact]) -> str:
    if not facts:
        return "FACTS: none."
    rows = [f"{f.id}: {f.statement[:MAX_STATEMENT_CHARS]}" for f in facts]
    return "FACTS:\n" + "\n".join(rows)


def _skeleton_block(genre: Genre) -> str:
    rows = [
        f"- {s.title} ({'required' if s.required else 'optional'}, "
        f"{s.words[0]}-{s.words[1]} words): {s.purpose}"
        for s in genre.sections
    ]
    flexible = "flexible" if genre.flexible_outline else "fixed"
    return f"GENRE SKELETON ({genre.name}, {flexible}):\n" + "\n".join(rows)


def _tabular_block(tabular_sources: list[Source]) -> str:
    if not tabular_sources:
        return "TABULAR SOURCES: none (do not plan figures)."
    rows = [f"- {s.id}: {s.title}" for s in tabular_sources]
    return "TABULAR SOURCES:\n" + "\n".join(rows)


def _brief_block(brief: Brief) -> str:
    lo, hi = brief.word_range
    return "\n".join(
        [
            f"TASK: {brief.task}",
            f"TITLE: {brief.title}",
            f"AUDIENCE: {brief.audience}",
            f"PURPOSE: {brief.purpose}",
            "KEY MESSAGES:\n" + "\n".join(f"- {m}" for m in brief.key_messages),
            "MUST INCLUDE:\n" + "\n".join(f"- {m}" for m in brief.must_include)
            if brief.must_include
            else "MUST INCLUDE: nothing specific",
            f"WORD RANGE: {lo}-{hi}",
            f"GUIDELINES: {brief.guidelines or 'none'}",
        ]
    )


def _context(
    brief: Brief,
    genre: Genre,
    facts: list[Fact],
    tabular_sources: list[Source],
) -> str:
    return "\n\n".join(
        [
            _brief_block(brief),
            _skeleton_block(genre),
            _facts_block(facts),
            _tabular_block(tabular_sources),
        ]
    )


def check_outline(
    outline: Outline,
    brief: Brief,
    genre: Genre,
    *,
    facts: list[Fact],
    tabular_sources: list[Source],
) -> list[CheckResult]:
    """Deterministic outline checks; one ``CheckResult`` per rule."""
    titles = [s.title.strip().casefold() for s in outline.sections]
    spec_titles = {s.title.casefold() for s in genre.sections}
    required = {s.title.casefold() for s in genre.sections if s.required}
    results: list[CheckResult] = []

    skeleton_details: list[str] = []
    if genre.flexible_outline and brief.guidelines:
        pass  # the call text may rename, add or drop sections
    elif genre.flexible_outline:
        skeleton_details += [
            f"required section missing: {t}" for t in sorted(required - set(titles))
        ]
    else:
        skeleton_details += [
            f"required section missing: {t}" for t in sorted(spec_titles - set(titles))
        ]
        skeleton_details += [
            f"section not in this genre's skeleton: {t}"
            for t in titles
            if t not in spec_titles
        ]
    results.append(
        CheckResult(
            name="outline_sections",
            passed=not skeleton_details,
            details=skeleton_details,
        )
    )

    seen: set[str] = set()
    dupes = sorted({t for t in titles if t in seen or seen.add(t)})
    results.append(
        CheckResult(
            name="outline_unique_titles",
            passed=not dupes,
            details=[f"duplicate section title: {t}" for t in dupes],
        )
    )

    points = " ".join(p for s in outline.sections for p in s.key_points).casefold()
    unassigned = [m for m in brief.must_include if m.casefold() not in points]
    results.append(
        CheckResult(
            name="outline_must_include",
            passed=not unassigned,
            details=[
                f"must-include item not in any section's key_points: {m}"
                for m in unassigned
            ],
        )
    )

    total = sum(s.target_words for s in outline.sections)
    lo, hi = brief.word_range
    results.append(
        CheckResult(
            name="outline_words",
            passed=lo <= total <= hi,
            details=[]
            if lo <= total <= hi
            else [f"target_words sum to {total}; the word range is {lo}-{hi}"],
        )
    )

    known = {f.id for f in facts}
    bad_ids = sorted(
        {fid for s in outline.sections for fid in s.fact_ids if fid not in known}
    )
    results.append(
        CheckResult(
            name="outline_fact_ids",
            passed=not bad_ids,
            details=[f"unknown fact id: {i}" for i in bad_ids],
        )
    )

    tabular_ids = {s.id for s in tabular_sources}
    figure_details: list[str] = []
    for s in outline.sections:
        if s.figure is None:
            continue
        if not genre.figures:
            figure_details.append(f"{s.title}: this genre takes no figures")
        elif s.figure.source_id not in tabular_ids:
            figure_details.append(
                f"{s.title}: figure source {s.figure.source_id} is not a tabular source"
            )
    results.append(
        CheckResult(
            name="outline_figures", passed=not figure_details, details=figure_details
        )
    )
    return results


def _problems(results: list[CheckResult]) -> list[str]:
    return [d for r in results if not r.passed for d in r.details]


async def make_outline(
    brief: Brief,
    genre: Genre,
    facts: list[Fact],
    *,
    tabular_sources: list[Source],
    model: str,
    log: ActionLog | None = None,
) -> Outline:
    """Ask for an outline, check it, retry once with the problems fed back."""
    context = _context(brief, genre, facts, tabular_sources)
    outline = await call_model(
        model=model,
        instructions=OUTLINE_INSTRUCTIONS,
        output_type=Outline,
        user_message=context,
        node="plan",
        log=log,
    )
    problems = _problems(
        check_outline(
            outline, brief, genre, facts=facts, tabular_sources=tabular_sources
        )
    )
    if not problems:
        return outline
    if log is not None:
        log.event(node="plan", event="outline_retry", problems=problems)
    retry_message = (
        context
        + "\n\nYOUR PREVIOUS OUTLINE:\n"
        + outline.model_dump_json(indent=1)
        + "\n\nIt has these problems; return a corrected outline:\n"
        + "\n".join(f"- {p}" for p in problems)
        + f"\n{WORD_TOLERANCE_NOTE}"
    )
    outline = await call_model(
        model=model,
        instructions=OUTLINE_INSTRUCTIONS,
        output_type=Outline,
        user_message=retry_message,
        node="plan",
        log=log,
    )
    remaining = _problems(
        check_outline(
            outline, brief, genre, facts=facts, tabular_sources=tabular_sources
        )
    )
    if remaining and log is not None:
        log.event(node="plan", event="outline_warnings", problems=remaining)
    return outline


async def amend_outline(
    outline: Outline,
    items: list[WorkItem],
    *,
    brief: Brief,
    genre: Genre,
    facts: list[Fact],
    tabular_sources: list[Source],
    model: str,
    log: ActionLog | None = None,
) -> Outline:
    """One amendment call. The caller allows at most ``MAX_OUTLINE_AMENDMENTS`` per run."""
    message = (
        _context(brief, genre, facts, tabular_sources)
        + "\n\nCURRENT OUTLINE:\n"
        + outline.model_dump_json(indent=1)
        + "\n\nPROBLEMS TO FIX:\n"
        + "\n".join(f"- ({i.source}) {i.reason}" for i in items)
    )
    amended = await call_model(
        model=model,
        instructions=AMEND_INSTRUCTIONS,
        output_type=Outline,
        user_message=message,
        node="plan",
        log=log,
    )
    remaining = _problems(
        check_outline(
            amended, brief, genre, facts=facts, tabular_sources=tabular_sources
        )
    )
    if remaining and log is not None:
        log.event(node="plan", event="amendment_warnings", problems=remaining)
    return amended
