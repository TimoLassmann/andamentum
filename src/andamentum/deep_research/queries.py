"""QUERIES: the next search queries for one sub-question.

Round 1 writes them with one model call (``write_queries``) and also searches
the sub-question's own text. Later rounds take the queries reflect proposed
(``fresh_queries`` removes repeats and trims to the budget); only when reflect
proposed none is the writer called again, with the gaps reflect named.

A failed or spent model never leaves a round without a query in round 1: the
sub-question's own text is searched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .ask import MODEL_ERRORS, UNTRUSTED, Asker, BudgetExhausted, fence
from .types import RawFinding, SubQuestion

# Queries already run that the writer is shown, most recent last.
MAX_QUERIES_SHOWN = 30
# Findings shown to the writer, with their claims clipped.
MAX_FINDINGS_SHOWN = 20
CLAIM_CHARS = 200
MIN_QUERIES = 2


class QueriesDraft(BaseModel):
    queries: list[str] = Field(
        description="3 to 8 plain keywords each, no quotes, operators or site: filters."
    )


def query_instructions(*, count: int) -> str:
    return f"""\
You are a search query specialist. You write the next search queries for ONE \
sub-question of a larger question.
1. Each query is 3 to 8 plain keywords. No long descriptions, no quotes, no \
operators, no site: filters.
2. Write {min(MIN_QUERIES, count)} to {count} queries. Each covers a different \
angle from the queries already run: for example the exact names and terms, the \
everyday wording, a more specific instance, a general overview, a reference \
work or official source, another time period, place or measure.
3. If gaps are named, aim the queries at them directly.
4. Stay anchored to the sub-question; keep its names, numbers, dates and \
places. Do not drift into adjacent topics.
5. Never repeat a query already run.
Prefer wording that primary, official or encyclopedic pages use over wording \
content farms repeat.
{UNTRUSTED}"""


@dataclass
class QueriesOutcome:
    queries: list[str]
    # Where the queries came from: "writer" (the model wrote some) or "own text"
    # (only the sub-question's own text).
    origin: str = "writer"
    degraded: list[str] = field(default_factory=list)
    spent: str = ""


def query_key(query: str) -> str:
    return " ".join(query.casefold().split())


def fresh_queries(
    proposed: Sequence[str], *, done: Sequence[str], count: int
) -> list[str]:
    """The queries of *proposed* not yet run (nor repeated within it), whitespace
    squashed, at most *count*, in order."""
    seen = {query_key(q) for q in done}
    fresh: list[str] = []
    for query in proposed:
        key = query_key(query)
        if not key or key in seen:
            continue
        seen.add(key)
        fresh.append(" ".join(query.split()))
        if len(fresh) >= count:
            break
    return fresh


async def write_queries(
    subquestion: SubQuestion,
    *,
    question: str,
    findings: Sequence[RawFinding],
    gaps: Sequence[str],
    done: Sequence[str],
    count: int,
    asker: Asker,
    include_own_text: bool = False,
) -> QueriesOutcome:
    """Up to *count* new keyword queries for *subquestion*, one model call
    (``step="Search"``). *findings* are shown numbered 1..n (never as ``F#``);
    *gaps* name what is still missing; *done* are the queries already run.
    With *include_own_text* the sub-question's own text comes first when it has
    not been run yet (round 1)."""
    lines = [
        f"QUESTION: {question.strip()}",
        f"SUB-QUESTION: {subquestion.text.strip()}",
    ]
    if findings:
        shown = findings[:MAX_FINDINGS_SHOWN]
        claims = "\n".join(
            f"{n}. {' '.join(f.claim.split())[:CLAIM_CHARS]}"
            for n, f in enumerate(shown, start=1)
        )
        lines += ["", "FINDINGS SO FAR:", fence(claims)]
    if gaps:
        lines += ["", "GAPS:", *(f"- {gap}" for gap in gaps)]
    if done:
        lines += ["", "ALREADY RUN:", *(f"- {q}" for q in done[-MAX_QUERIES_SHOWN:])]
    outcome = QueriesOutcome([])
    written: list[str] = []
    try:
        draft = await asker.llm(
            step="Search",
            instructions=query_instructions(count=count),
            output_type=QueriesDraft,
            user_message="\n".join(lines),
        )
        written = draft.queries
    except BudgetExhausted as spent:
        outcome.spent = spent.name
    except MODEL_ERRORS as error:
        outcome.degraded.append(
            f"query writing failed ({type(error).__name__}: {error}); "
            + (
                "the sub-question is searched by its own text"
                if include_own_text
                else "no new queries for this round"
            )
        )
    own = [subquestion.text.strip()] if include_own_text else []
    own_text = fresh_queries(own, done=done, count=1)
    model_queries = fresh_queries(written, done=[*done, *own_text], count=count)
    outcome.queries = [*own_text, *model_queries]
    outcome.origin = "writer" if model_queries else "own text"
    return outcome
