"""PLAN: the question becomes one to four sub-questions.

One model call (``step="Plan"``). A question that asks one thing stays one
sub-question, in its own words; a question with parts is split. A failed plan is
the question as its one sub-question, recorded as degraded; a spent budget is
the same plan with ``spent`` naming the bound.

``filters_for`` is the one piece of search-side planning done in code: a
metadata sub-question that states a year range or open access in words gets
those as source filters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry

from .ask import MODEL_ERRORS, Asker, BudgetExhausted
from .sources.base import SearchFilters
from .types import SubQuestion, SubQuestionKind

MIN_SUBQUESTIONS = 1
MAX_SUBQUESTIONS = 4


class PlannedSubQuestion(BaseModel):
    text: str = Field(description="One sub-question, a full sentence ending in '?'.")
    kind: SubQuestionKind = Field(
        default="topical",
        description="topical (what is known), navigational (one named page, "
        "document, product, person or place), or metadata (items by date, "
        "author, publisher or source)",
    )


class PlanDraft(BaseModel):
    subquestions: list[PlannedSubQuestion]


PLAN_INSTRUCTIONS = f"""\
You turn a question into the sub-questions a search must answer. When the \
question asks one thing, return it as one sub-question, in its own words. When \
it asks several things, split it into at most {MAX_SUBQUESTIONS} sub-questions: \
together they cover everything the question asks, and each asks one thing that \
sources can answer on their own. Do not answer them. Do not add sub-questions \
the question does not need. Keep the question's own terms, names, numbers, dates \
and places in the sub-questions that need them."""


@dataclass
class PlanOutcome:
    subquestions: list[SubQuestion]
    degraded: list[str] = field(default_factory=list)
    spent: str = ""


def single_subquestion(question: str) -> list[SubQuestion]:
    """The fallback plan: the question as its one sub-question."""
    return [SubQuestion(id="Q1", text=question.strip())]


async def plan(question: str, *, asker: Asker) -> PlanOutcome:
    """Sub-questions for *question*: the model writes them. A failed plan is
    the question as its one sub-question, recorded as degraded."""

    # async on purpose: pydantic-ai runs a sync validator in a worker thread, and
    # the order concurrent calls finish in would then depend on the threads
    async def well_formed(draft: PlanDraft) -> PlanDraft:
        texts = [s.text.strip() for s in draft.subquestions]
        if not MIN_SUBQUESTIONS <= len(texts) <= MAX_SUBQUESTIONS:
            raise ModelRetry(
                f"give {MIN_SUBQUESTIONS} to {MAX_SUBQUESTIONS} sub-questions, "
                f"not {len(texts)}"
            )
        if any(not t for t in texts):
            raise ModelRetry("every sub-question needs text")
        if len({t.casefold() for t in texts}) != len(texts):
            raise ModelRetry("the sub-questions repeat each other")
        return draft

    try:
        draft = await asker.llm(
            step="Plan",
            instructions=PLAN_INSTRUCTIONS,
            output_type=PlanDraft,
            user_message=f"QUESTION: {question.strip()}",
            validators=[well_formed],
        )
    except BudgetExhausted as spent:
        return PlanOutcome(single_subquestion(question), spent=spent.name)
    except MODEL_ERRORS as error:
        return PlanOutcome(
            single_subquestion(question),
            degraded=[
                f"planning failed ({type(error).__name__}: {error}); the question "
                "is searched as one sub-question"
            ],
        )
    return PlanOutcome(
        [
            SubQuestion(id=f"Q{n}", text=s.text.strip(), kind=s.kind)
            for n, s in enumerate(draft.subquestions, start=1)
        ]
    )


_YEAR = r"((?:19|20)\d{2})"
_SINCE = re.compile(rf"\b(since|from|after|newer than)\s+{_YEAR}\b", re.IGNORECASE)
_UNTIL = re.compile(
    rf"\b(before|until|up to|prior to|older than|through)\s+{_YEAR}\b", re.IGNORECASE
)
_RANGE = re.compile(rf"\b{_YEAR}\s*(?:-|–|to|and)\s*{_YEAR}\b", re.IGNORECASE)
_IN_YEAR = re.compile(rf"\b(?:in|during)\s+{_YEAR}\b", re.IGNORECASE)
_OPEN_ACCESS = re.compile(r"\bopen[- ]access\b", re.IGNORECASE)


def filters_for(subquestion: SubQuestion) -> SearchFilters | None:
    """Source filters a metadata sub-question states in words, read by code:
    a year range ("since 2020", "before 2015", "2018-2022", "in 2021") and
    open access. ``None`` for any other kind, or when nothing is stated."""
    if subquestion.kind != "metadata":
        return None
    text = subquestion.text
    year_from: int | None = None
    year_to: int | None = None
    if found := _RANGE.search(text):
        year_from, year_to = sorted((int(found.group(1)), int(found.group(2))))
    else:
        if found := _SINCE.search(text):
            year = int(found.group(2))
            year_from = year + 1 if found.group(1).lower() == "after" else year
        if found := _UNTIL.search(text):
            year = int(found.group(2))
            year_to = year - 1 if found.group(1).lower() == "before" else year
        if year_from is None and year_to is None and (found := _IN_YEAR.search(text)):
            year_from = year_to = int(found.group(1))
    open_access = bool(_OPEN_ACCESS.search(text))
    if year_from is None and year_to is None and not open_access:
        return None
    return SearchFilters(
        year_from=year_from, year_to=year_to, open_access_only=open_access
    )
