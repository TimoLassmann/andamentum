"""The screening arms: one generic relevance question against compiled criteria.

Both arms put their questions to every record through the engine's own
``Screener.score_candidates`` (one request per record, sequential) and are
judged by the engine's own ``combine`` rules. They differ only in the criteria
they pass:

- ``generic``: one ``must`` question, "is this record relevant to the question?"
- ``criteria``: the rubric the LLM compiles for the review's question
  (``Screener.compile``), 3 to 8 atomic questions of its own.

A scorer is the decision model (``decision``) or the LLM (``llm``); the criteria
arms compile with the LLM either way. The compile cost is part of a criteria
arm's cost. The ranking is the combined score; the kept set is ``combine``'s
keep decision (recall first: only a confident ``must`` miss or an ``exclude`` hit
drops a record).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from andamentum.core.decisions import DecisionModel

from ..ask import Asker
from ..criteria import Screener
from ..store import ResearchStore
from ..types import Budget, Candidate, Criterion, RunStats, SubQuestion
from .metrics import Cost, ScreeningScore, screening_score
from .synergy import ScreeningReview

ScreenKind = Literal["generic", "criteria"]
ScorerKind = Literal["decision", "llm"]
_KINDS: dict[str, ScreenKind] = {"generic": "generic", "criteria": "criteria"}
_SCORERS: dict[str, ScorerKind] = {"decision": "decision", "llm": "llm"}
SUBQUESTION_ID = "S1"
SOURCE_NAME = "synergy"
CALLS_PER_RECORD = 3  # a request per record, with room for the LLM scorer's retries
CALL_SLACK = 100


def parse_arm(name: str) -> tuple[ScreenKind, ScorerKind]:
    """``"criteria+decision"`` as ``("criteria", "decision")``."""
    kind, _, scorer = name.partition("+")
    if kind not in _KINDS or scorer not in _SCORERS:
        raise ValueError(
            f"{name!r} is not a screening arm: expected <generic|criteria>+<decision|llm>"
        )
    return _KINDS[kind], _SCORERS[scorer]


def review_subquestion(review: ScreeningReview) -> SubQuestion:
    """The review's own title as the one sub-question screening is asked about."""
    return SubQuestion(id=SUBQUESTION_ID, text=review.question, kind="topical")


def generic_criterion(subquestion: SubQuestion) -> Criterion:
    """The single question of the generic arm."""
    return Criterion(
        id=f"{subquestion.id}-G",
        text=f"Is this record relevant to the question: {subquestion.text.rstrip(' ?')}?",
        kind="must",
        weight=3,
        applies_to="candidate",
    )


class ScreeningOutcome(BaseModel):
    score: ScreeningScore
    criteria: list[Criterion] = Field(default_factory=list)
    cost: Cost


def _asker(
    *,
    model: Any | None,
    decision_model: DecisionModel | None,
    n_records: int,
    stats: RunStats,
) -> Asker:
    store = ResearchStore.in_memory()
    calls = n_records * CALLS_PER_RECORD + CALL_SLACK
    return Asker(
        model=model,
        store=store,
        run_id=store.start_run(
            question="screening", mode="scholarly", config={}, models={}
        ),
        budget=Budget(max_llm_calls=calls, max_decision_calls=calls),
        stats=stats,
        decision_model=decision_model,
    )


async def compile_rubric(
    review: ScreeningReview, *, model: Any
) -> tuple[list[Criterion], Cost]:
    """The LLM's rubric for the review's question, and what compiling it cost."""
    stats = RunStats()
    started = time.monotonic()
    screener = Screener(
        asker=_asker(model=model, decision_model=None, n_records=0, stats=stats)
    )
    criteria = await screener.compile(
        review_subquestion(review), question=review.question
    )
    return criteria, Cost(
        wall_s=time.monotonic() - started,
        llm_calls=stats.llm_calls,
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        decision_calls=0,
    )


class ScreeningArm:
    """One cell of the comparison: a kind of criteria and a kind of scorer."""

    def __init__(
        self,
        *,
        kind: ScreenKind,
        scorer: ScorerKind,
        model: Any | None = None,
        decision_model: DecisionModel | None = None,
    ) -> None:
        if scorer == "decision" and decision_model is None:
            raise ValueError(f"the arm {kind}+decision needs a decision model")
        if scorer == "llm" and model is None:
            raise ValueError(f"the arm {kind}+llm needs a model")
        if kind == "criteria" and model is None:
            raise ValueError("compiling criteria needs a model")
        self.kind: ScreenKind = kind
        self.scorer: ScorerKind = scorer
        self.model = model
        self.decision_model = decision_model
        self.name = f"{kind}+{scorer}"

    async def run(
        self,
        review: ScreeningReview,
        *,
        compiled: tuple[list[Criterion], Cost] | None = None,
    ) -> ScreeningOutcome:
        """Screen every record of *review*; sequential, one request per record.

        A ``criteria`` arm needs the review's *compiled* rubric (compiled once
        and shared by the arms that compare scorers on it).
        """
        subquestion = review_subquestion(review)
        if self.kind == "generic":
            criteria = [generic_criterion(subquestion)]
            carried = Cost(
                wall_s=0.0,
                llm_calls=0,
                input_tokens=0,
                output_tokens=0,
                decision_calls=0,
            )
        else:
            if compiled is None:
                raise ValueError("a criteria arm needs the compiled rubric")
            criteria, carried = compiled
        stats = RunStats()
        asker = _asker(
            model=self.model if self.scorer == "llm" else None,
            decision_model=self.decision_model if self.scorer == "decision" else None,
            n_records=len(review.records),
            stats=stats,
        )
        candidates = [
            Candidate(work=r.to_work(), sources=[SOURCE_NAME]) for r in review.records
        ]
        started = time.monotonic()
        scores = await Screener(asker=asker).score_candidates(
            subquestion, criteria, candidates
        )
        wall_s = time.monotonic() - started
        by_work = {s.work_id: s for s in scores}
        ordered = [by_work[r.record_id] for r in review.records]
        return ScreeningOutcome(
            score=screening_score(
                [s.score for s in ordered],
                [s.kept for s in ordered],
                [r.included for r in review.records],
            ),
            criteria=criteria,
            cost=Cost(
                wall_s=wall_s + carried.wall_s,
                llm_calls=stats.llm_calls + (carried.llm_calls or 0),
                input_tokens=stats.input_tokens + (carried.input_tokens or 0),
                output_tokens=stats.output_tokens + (carried.output_tokens or 0),
                decision_calls=stats.decision_calls,
            ),
        )
