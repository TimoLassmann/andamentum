"""A novelty check on the v2 engine: prior work is what a passage says, not a guess.

``run_novelty_check_v2(claim, ...)`` searches the scholarly sources for work that
may already report *claim* (a ``research`` run in mode ``scholarly``), then asks
the decision model, for every passage it read, whether that passage reports the
same result, method or finding the claim presents as new. The verdict is a
function of those answers and of how healthy the search was:

- a passage the decision model says anticipates the claim (``p >= 0.5``, the
  decision-model convention) makes the claim **not novel**; the passage's own
  words are the evidence in ``similar_work``;
- nothing anticipates it **and** the search was healthy (no source or model
  failure, no spent budget, at least ``MIN_WORKS_ASSESSED`` works read) makes
  it **novel**, with confidence ``1 - max p``;
- anything else is **undetermined** (``is_novel=None``): "not found" is not
  evidence of novelty when the search could not look properly.

No text is written by a model: the assessment is built from counts and quotes,
so there is nothing to invent. The decision model is required and explicit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.models import Model

from andamentum.core.decisions import DecisionModel

from ..novelty.models import NoveltyReport, Relevance, SimilarWork
from .ask import MODEL_ERRORS, Asker, BudgetExhausted
from .graph import research
from .sources import SourceConfig
from .sources.base import SourceClient
from .store import ResearchStore
from .types import Budget, Passage, ResearchResult, RunStats

# A noul question is a yes at p >= 0.5 (core.decisions); a passage is direct
# prior work from 0.8.
ANTICIPATES_AT = 0.5
DIRECT_AT = 0.8
# "Not novel" needs one passage; "novel" needs the search to have read this
# many different works, or an empty answer would read as a finding.
MIN_WORKS_ASSESSED = 3
MAX_SIMILAR_WORKS = 5
SUMMARY_CHARS = 400
NOVELTY_BUDGET = Budget(max_rounds=2)

STEP = "novelty"
_ANTICIPATES = (
    "The passage reports the same result, method or finding that the claim "
    "presents as new, so the claim is not new."
)
_RELATED = "The passage is about the same topic or problem as the claim."
_QUESTIONS = {"anticipates": _ANTICIPATES, "related": _RELATED}


@dataclass(frozen=True)
class NoveltyResearch:
    """How a caller asks for the v2 novelty check: the decision model that
    judges the passages (opened, left open), where the sources are and where
    the research store lives (``None`` keeps it in memory)."""

    decision_model: DecisionModel
    sources: SourceConfig = SourceConfig()
    store: Path | ResearchStore | None = None
    budget: Budget = NOVELTY_BUDGET


@dataclass(frozen=True)
class _Scored:
    passage: Passage
    anticipates: float
    related: float


def _undetermined(
    claim: str, *, why: str, queries: list[str], sources: list[str]
) -> NoveltyReport:
    return NoveltyReport(
        claim=claim,
        is_novel=None,
        confidence=0.0,
        assessment=f"{why} Novelty is UNDETERMINED: manual verification required.",
        similar_work=[],
        sources=sources,
        search_queries_used=queries,
    )


def _work_url(result: ResearchResult, work_id: str) -> str:
    work = next(w for w in result.works if w.work_id == work_id)
    if work.doi:
        return f"https://doi.org/{work.doi}"
    return work.url or work.oa_url or ""


def _similar(result: ResearchResult, scored: list[_Scored]) -> list[SimilarWork]:
    """One entry per work, its best passage's words as the summary, best first."""
    titles = {w.work_id: w.title or "(untitled)" for w in result.works}
    best: dict[str, _Scored] = {}
    for item in scored:
        key = item.passage.work_id
        if key not in best or item.anticipates > best[key].anticipates:
            best[key] = item
    ranked = sorted(best.values(), key=lambda s: (-s.anticipates, -s.related))
    similar: list[SimilarWork] = []
    for item in ranked[:MAX_SIMILAR_WORKS]:
        if item.anticipates >= DIRECT_AT:
            relevance = Relevance.DIRECT
        elif item.anticipates >= ANTICIPATES_AT or item.related >= ANTICIPATES_AT:
            relevance = Relevance.PARTIAL
        else:
            continue
        similar.append(
            SimilarWork(
                title=titles[item.passage.work_id],
                url=_work_url(result, item.passage.work_id),
                relevance=relevance,
                summary=item.passage.text[:SUMMARY_CHARS],
            )
        )
    return similar


async def _score(
    asker: Asker, claim: str, result: ResearchResult
) -> tuple[list[_Scored], bool]:
    """Ask both questions of every passage of a kept work, one request each.

    Returns what was scored and whether the budget stopped it early.
    """
    titles = {w.work_id: w.title for w in result.works}
    scored: list[_Scored] = []
    for passage in result.passages:
        if passage.work_id not in titles:
            continue  # a work left out of the result (retracted) is not evidence
        try:
            p = await asker.ask_criteria(
                STEP,
                state={
                    "claim": claim,
                    "work": titles[passage.work_id],
                    "passage": passage.text,
                },
                questions=_QUESTIONS,
            )
        except BudgetExhausted:
            return scored, True
        scored.append(
            _Scored(
                passage=passage,
                anticipates=p["anticipates"],
                related=p["related"],
            )
        )
    return scored, False


def _verdict(
    claim: str,
    *,
    result: ResearchResult,
    scored: list[_Scored],
    cut_short: bool,
    queries: list[str],
) -> NoveltyReport:
    """The verdict from the scored passages and the health of the search."""
    sources = [url for w in result.works if (url := _work_url(result, w.work_id)) != ""]
    anticipating = [s for s in scored if s.anticipates >= ANTICIPATES_AT]
    if anticipating:
        top = max(s.anticipates for s in anticipating)
        works = {s.passage.work_id for s in anticipating}
        return NoveltyReport(
            claim=claim,
            is_novel=False,
            confidence=top,
            assessment=(
                f"{len(anticipating)} passage(s) from {len(works)} work(s) report "
                f"what the claim presents as new (decision model p up to {top:.2f}); "
                "their words are quoted under similar work."
            ),
            similar_work=_similar(result, scored),
            sources=sources,
            search_queries_used=queries,
        )
    works_read = len({s.passage.work_id for s in scored})
    problems: list[str] = []
    if result.degraded:
        problems.append(
            f"the search was degraded ({len(result.degraded)} failure(s): "
            f"{result.degraded[0]})"
        )
    if result.stop_reason.startswith("budget:"):
        problems.append(f"the search stopped on its budget ({result.stop_reason})")
    if cut_short:
        problems.append("the decision budget ran out before every passage was judged")
    if works_read < MIN_WORKS_ASSESSED:
        problems.append(
            f"only {works_read} work(s) were read (at least {MIN_WORKS_ASSESSED} "
            "are needed to conclude that nothing is there)"
        )
    if problems:
        return _undetermined(
            claim,
            why="No passage anticipates the claim, but " + "; ".join(problems) + ".",
            queries=queries,
            sources=sources,
        )
    top_p = max((s.anticipates for s in scored), default=0.0)
    return NoveltyReport(
        claim=claim,
        is_novel=True,
        confidence=1.0 - top_p,
        assessment=(
            f"{len(scored)} passage(s) from {works_read} work(s) were read and "
            f"none reports what the claim presents as new (highest decision model "
            f"p {top_p:.2f}). The search had no failures."
        ),
        similar_work=_similar(result, scored),
        sources=sources,
        search_queries_used=queries,
    )


async def run_novelty_check_v2(
    claim: str,
    *,
    model: str | Model,
    decision_model: DecisionModel | str,
    store: Path | ResearchStore | None = None,
    sources: SourceConfig = SourceConfig(),
    budget: Budget = NOVELTY_BUDGET,
    source_client: SourceClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> NoveltyReport:
    """Check whether *claim* is novel against the scholarly literature.

    ``model`` is the language model (a pydantic-ai id or object) that plans the search and reads the works;
    ``decision_model`` (a name, opened and closed here, or an opened handle) is
    the local decision model that judges every passage. ``store`` is the
    research store: a path opens or creates a persistent one (works, passages
    and answers are cached there), an open store is used as it is, ``None``
    keeps it in memory. A failure of the language model or decision model, or
    a search too thin to conclude from, gives ``is_novel=None``, never a
    guess; anything else raises.
    """
    if not claim.strip():
        raise ValueError("claim is empty")
    owns_store = not isinstance(store, ResearchStore)
    opened = (
        store
        if isinstance(store, ResearchStore)
        else ResearchStore(store)
        if store is not None
        else ResearchStore.in_memory()
    )
    handle = (
        await DecisionModel.open(decision_model)
        if isinstance(decision_model, str)
        else decision_model
    )
    try:
        result = await research(
            f"What published work already reports or anticipates this: {claim}",
            mode="scholarly",
            model=model,
            decision_model=handle,
            store=opened,
            budget=budget,
            sources=sources,
            source_client=source_client,
            progress=progress,
        )
        queries = result.queries
        asker = Asker(
            model=model,
            store=opened,
            run_id=result.run_id,
            budget=budget,
            stats=RunStats(decision_calls=result.stats.decision_calls),
            decision_model=handle,
        )
        try:
            scored, cut_short = await _score(asker, claim, result)
        except MODEL_ERRORS as error:
            return _undetermined(
                claim,
                why=f"The judgement of the passages failed: {error}.",
                queries=queries,
                sources=[],
            )
        return _verdict(
            claim, result=result, scored=scored, cut_short=cut_short, queries=queries
        )
    finally:
        if isinstance(decision_model, str):
            await handle.aclose()
        if owns_store:
            opened.close()
