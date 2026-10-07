"""REFLECT: is the sub-question answered, and if not what is missing.

One model call (``step="Reflect"``), separate from writing, over all the
findings of the sub-question so far. It returns whether the question is
answered, the best-supported answer, how sure the evidence makes it, the gaps
and the next queries. The findings are shown numbered 1..n (never ``F#``: the
engine numbers the run's findings only when every sub-question has finished).

With no findings there is nothing to judge: no call is made and the outcome
names the gap. A failed model gives ``answered=False``, no answer and a
degraded line.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from andamentum.references import Work

from .ask import MODEL_ERRORS, UNTRUSTED, Asker, BudgetExhausted, fence
from .origin import origin_key, origin_label
from .queries import MAX_QUERIES_SHOWN, fresh_queries
from .types import RawFinding, SubQuestion

# Findings put to the model at once (spread over the sources when there are more).
MAX_FINDINGS_SHOWN = 40
NOTHING_READ_GAP = "no relevant page has been read yet"

Confidence = Literal["high", "medium", "low"]


class ReflectDraft(BaseModel):
    answered: bool
    answer: str = Field(
        default="",
        description="The best-supported answer, committed to one, naming its source.",
    )
    confidence: Confidence = "low"
    gaps: list[str] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)


def reflect_instructions(*, count: int) -> str:
    return f"""\
You judge the progress of research on ONE question. You see the findings so far: \
each a claim, the quote behind it, and its source (title, site, year).
1. answered: true only when a reliable source states the specific answer, or \
several independent sources agree. One weak source (a forum, an aggregator, a \
content farm, a page that repeats the question) is not enough. Findings that \
differ: weigh them by source quality (primary, official, encyclopedic first) \
and by the date or period the question asks about; if the weight settles it, \
answered is true.
2. answer: the best-supported answer so far as a short statement that commits \
to one answer and names its source, with how sure the evidence makes you in \
words ("likely", "reported by one source"). Empty when the findings give no \
answer.
3. confidence: high, medium or low.
4. gaps: what is still missing or unsettled, specifically ("the figure for the \
year asked", "a source other than the shop's own site"). Empty when answered.
5. next_queries: up to {count} queries of 3 to 8 plain keywords aimed at the \
gaps, in different words from the queries already run. Empty when answered.
Judge only from the findings.
{UNTRUSTED}"""


@dataclass
class ReflectOutcome:
    answered: bool = False
    answer: str = ""
    confidence: Confidence = "low"
    gaps: list[str] = field(default_factory=list)
    next_queries: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    spent: str = ""


def _spread(
    findings: Sequence[RawFinding], works: Mapping[str, Work], limit: int
) -> list[RawFinding]:
    """At most *limit* findings, one origin at a time, each origin's in order;
    the result is back in the findings' own order."""
    if len(findings) <= limit:
        return list(findings)
    by_origin: dict[str, list[int]] = {}
    for index, finding in enumerate(findings):
        work = works.get(finding.work_id)
        key = origin_key(work) if work is not None else f"work:{finding.work_id}"
        by_origin.setdefault(key, []).append(index)
    chosen: list[int] = []
    depth = 0
    while len(chosen) < limit:
        for indexes in by_origin.values():
            if depth < len(indexes) and len(chosen) < limit:
                chosen.append(indexes[depth])
        depth += 1
    return [findings[i] for i in sorted(chosen)]


def findings_block(findings: Sequence[RawFinding], works: Mapping[str, Work]) -> str:
    """The findings as numbered entries: claim, quote, source."""
    lines: list[str] = []
    for number, finding in enumerate(
        _spread(findings, works, MAX_FINDINGS_SHOWN), start=1
    ):
        work = works.get(finding.work_id)
        if work is not None:
            year = f", {work.year}" if work.year else ""
            source = f"{' '.join(work.title.split())} ({origin_label(work)}{year})"
        else:
            source = "unknown source"
        mark = "" if finding.quote_verified else " [quote not found in the page]"
        lines += [
            f"[{number}] claim: {' '.join(finding.claim.split())}",
            f"    quote: {' '.join(finding.quote.split())}{mark}",
            f"    source: {source}",
        ]
    return "\n".join(lines)


async def reflect(
    subquestion: SubQuestion,
    *,
    findings: Sequence[RawFinding],
    works: Mapping[str, Work],
    done: Sequence[str],
    count: int,
    asker: Asker,
) -> ReflectOutcome:
    """Judge *subquestion* from its *findings* so far. *works* maps work id to
    work (the source shown beside each finding); *done* are the queries already
    run; *count* bounds ``next_queries``."""
    if not findings:
        return ReflectOutcome(gaps=[NOTHING_READ_GAP])
    lines = [
        f"QUESTION: {subquestion.text.strip()}",
        "",
        "FINDINGS:",
        fence(findings_block(findings, works)),
    ]
    if done:
        lines += [
            "",
            "QUERIES ALREADY RUN:",
            *(f"- {q}" for q in done[-MAX_QUERIES_SHOWN:]),
        ]
    try:
        draft = await asker.llm(
            step="Reflect",
            instructions=reflect_instructions(count=count),
            output_type=ReflectDraft,
            user_message="\n".join(lines),
        )
    except BudgetExhausted as spent:
        return ReflectOutcome(spent=spent.name)
    except MODEL_ERRORS as error:
        return ReflectOutcome(
            degraded=[
                f"reflecting on {subquestion.id} failed ({type(error).__name__}: "
                f"{error}); it counts as not answered"
            ]
        )
    return ReflectOutcome(
        answered=draft.answered,
        answer=" ".join(draft.answer.split()),
        confidence=draft.confidence,
        gaps=[g.strip() for g in draft.gaps if g.strip()],
        next_queries=fresh_queries(draft.next_queries, done=done, count=count),
    )
