"""Running a task: every item sequentially, each result recorded, none dropped.

Local models and one shared server are used one call at a time, so nothing
here is concurrent. An item that raises is recorded with its error and counted
against the arm; ``checkpoint`` is called after each item with the run so far,
so a long run that dies keeps what it finished.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..types import Criterion
from .arms import AnswerArm
from .judging import ClaimJudge, UrlChecker, check_links
from .litsearch import LitSearchItem
from .metrics import DEFAULT_KS, Cost, claim_support, retrieval_score
from .results import (
    LitSearchResultItem,
    LitSearchRun,
    RunConfig,
    ScreeningItem,
    ScreeningRun,
    WebItem,
    WebRun,
    summarise_litsearch,
    summarise_screening,
    summarise_web,
)
from .screening import ScreeningArm, compile_rubric
from .synergy import ScreeningReview, sample_records
from .web_set import WebQuestion, answer_correct

EXCERPT_CHARS = 400


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


async def run_web(
    questions: Sequence[WebQuestion],
    arm: AnswerArm,
    *,
    config: RunConfig,
    judge: ClaimJudge | None = None,
    url_checker: UrlChecker | None = None,
    checkpoint: Callable[[WebRun], None] | None = None,
) -> WebRun:
    """Ask *arm* each question; check correctness, claim support and links.

    Claim support is measured only with a *judge* and link health only with a
    *url_checker*; without them those fields stay None, and the table says n/a.
    """
    items: list[WebItem] = []
    for question in questions:
        item = WebItem(
            question_id=question.question_id,
            question=question.question,
            expected=question.expected,
        )
        try:
            output = await arm.run(question.question)
            item.correct = answer_correct(output.answer_text, question.answer)
            item.answer_excerpt = output.answer_text[:EXCERPT_CHARS]
            item.cost = output.cost
            item.stop_reason = output.stop_reason
            item.degraded = output.degraded
            if judge is not None:
                verdicts = await judge.judge(output.claims)
                item.claim_support = claim_support(
                    verdicts, n_unsourced=output.n_unsourced
                )
            if url_checker is not None:
                item.links = await check_links(output.urls, url_checker)
        except Exception as error:  # recorded and counted, never swallowed
            item.error = _error(error)
        items.append(item)
        if checkpoint is not None:
            checkpoint(
                WebRun(config=config, items=list(items), summary=summarise_web(items))
            )
    return WebRun(config=config, items=items, summary=summarise_web(items))


async def run_litsearch(
    items: Sequence[LitSearchItem],
    arm: AnswerArm,
    *,
    config: RunConfig,
    ks: Sequence[int] = DEFAULT_KS,
    checkpoint: Callable[[LitSearchRun], None] | None = None,
) -> LitSearchRun:
    """Ask *arm* each query; score its ranked works against the gold papers."""
    done: list[LitSearchResultItem] = []
    for item in items:
        result = LitSearchResultItem(
            query_id=item.query.query_id,
            query=item.query.query,
            n_gold=len(item.gold),
        )
        try:
            output = await arm.run(item.query.query)
            result.retrieval = retrieval_score(
                output.works, item.gold, ks=ks, title_contained=arm.returns_pages
            )
            result.cost = output.cost
            result.stop_reason = output.stop_reason
            result.degraded = output.degraded
        except Exception as error:
            result.error = _error(error)
        done.append(result)
        if checkpoint is not None:
            checkpoint(
                LitSearchRun(
                    config=config,
                    items=list(done),
                    summary=summarise_litsearch(done, ks=ks),
                )
            )
    return LitSearchRun(
        config=config, items=done, summary=summarise_litsearch(done, ks=ks)
    )


async def run_screening(
    reviews: Sequence[ScreeningReview],
    arms: Sequence[ScreeningArm],
    *,
    configs: dict[str, RunConfig],
    max_records: int | None,
    compile_model: Any | None,
    checkpoint: Callable[[ScreeningRun], None] | None = None,
) -> list[ScreeningRun]:
    """Screen each review with every arm, one run per arm.

    The rubric of a review is compiled once and shared by the ``criteria`` arms,
    so they differ only in the scorer. *configs* maps an arm name to its
    ``RunConfig``. A review is sampled to at most *max_records* records
    (stratified, deterministic).
    """
    done: dict[str, list[ScreeningItem]] = {arm.name: [] for arm in arms}
    for review in reviews:
        sampled = review.model_copy(
            update={
                "records": sample_records(review.records, max_records, seed=review.seed)
            }
        )
        compiled: tuple[list[Criterion], Cost] | None = None
        compile_error: str | None = None
        if compile_model is not None and any(a.kind == "criteria" for a in arms):
            try:
                compiled = await compile_rubric(sampled, model=compile_model)
            except Exception as error:
                compile_error = _error(error)
        for arm in arms:
            item = ScreeningItem(
                review_key=review.key,
                question=review.question,
                n_records=len(sampled.records),
                n_records_total=review.n_records_total,
            )
            try:
                if arm.kind == "criteria" and compiled is None:
                    raise RuntimeError(
                        f"no compiled criteria: {compile_error or 'no compile model'}"
                    )
                outcome = await arm.run(sampled, compiled=compiled)
                item.score = outcome.score
                item.criteria = outcome.criteria
                item.cost = outcome.cost
            except Exception as error:
                item.error = _error(error)
            done[arm.name].append(item)
            if checkpoint is not None:
                checkpoint(_screening_run(configs[arm.name], done[arm.name]))
    return [_screening_run(configs[arm.name], done[arm.name]) for arm in arms]


def _screening_run(config: RunConfig, items: list[ScreeningItem]) -> ScreeningRun:
    return ScreeningRun(
        config=config, items=list(items), summary=summarise_screening(items)
    )
