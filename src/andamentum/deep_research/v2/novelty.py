"""A novelty check on the v2 engine: prior work is what a finding says, not a guess.

``run_novelty_check_v2(claim, ...)`` searches the scholarly sources for work that
may already report *claim* (a ``research`` run in mode ``scholarly``; the engine
uses only the language model), then asks the local decision model, for every
finding of a work that was read, whether that finding reports the same result,
method or finding the claim presents as new. The decision model judges findings
(a claim and the quote behind it), never raw passages. The verdict is a function
of those answers and of how healthy the search was:

- a finding the decision model says anticipates the claim (``p >= 0.5``, the
  decision-model convention) makes the claim **not novel**; the finding's own
  quote is the evidence in ``similar_work``;
- nothing anticipates it **and** the search was healthy (no source or model
  failure, no spent budget, at least ``MIN_WORKS_ASSESSED`` works read) makes
  it **novel**, with confidence ``1 - max p``;
- anything else is **undetermined** (``is_novel=None``): "not found" is not
  evidence of novelty when the search could not look properly.

No text is written by a model: the assessment is built from counts and quotes,
so there is nothing to invent. The decision model is required and explicit, is
called strictly one request after another, and is the only local model left in
the package (whetstone's opt-in path). A retracted work is never evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.models import Model

from andamentum.core.decisions import DecisionModel
from andamentum.references import Work

from ..novelty.models import NoveltyReport, Relevance, SimilarWork
from .ask import MODEL_ERRORS, Asker, BudgetExhausted
from .engine import research
from .sources import SourceConfig
from .sources.base import SourceClient
from .store import ResearchStore
from .types import Budget, Finding, ResearchResult, RunStats

# A noul question is a yes at p >= 0.5 (core.decisions); a finding is direct
# prior work from 0.8.
ANTICIPATES_AT = 0.5
DIRECT_AT = 0.8
# "Not novel" needs one finding; "novel" needs the search to have read this
# many different works, or an empty answer would read as a finding.
MIN_WORKS_ASSESSED = 3
MAX_SIMILAR_WORKS = 5
SUMMARY_CHARS = 400
NOVELTY_BUDGET = Budget(max_rounds=2)

STEP = "novelty"
_ANTICIPATES = (
    "The finding reports the same result, method or finding that the claim "
    "under review presents as new, so the claim is not new."
)
_RELATED = "The finding is about the same topic or problem as the claim under review."
_QUESTIONS = {"anticipates": _ANTICIPATES, "related": _RELATED}


@dataclass(frozen=True)
class NoveltyResearch:
    """How a caller asks for the v2 novelty check: the decision model that
    judges the findings (opened, left open), where the sources are and where
    the research store lives (``None`` keeps it in memory)."""

    decision_model: DecisionModel
    sources: SourceConfig = SourceConfig()
    store: Path | ResearchStore | None = None
    budget: Budget = NOVELTY_BUDGET


@dataclass(frozen=True)
class _Scored:
    finding: Finding
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


def _work_url(work: Work) -> str:
    if work.doi:
        return f"https://doi.org/{work.doi}"
    return work.url or work.oa_url or ""


def _kept(result: ResearchResult) -> dict[str, Work]:
    """The works that may be evidence: not excluded, not retracted."""
    return {
        w.work_id: w
        for w in result.works
        if w.work_id not in result.excluded and w.retracted != "retracted"
    }


def _similar(result: ResearchResult, scored: list[_Scored]) -> list[SimilarWork]:
    """One entry per work, its best finding's quote as the summary, best first."""
    kept = _kept(result)
    best: dict[str, _Scored] = {}
    for item in scored:
        key = item.finding.work_id
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
        work = kept[item.finding.work_id]
        similar.append(
            SimilarWork(
                title=work.title or "(untitled)",
                url=_work_url(work),
                relevance=relevance,
                summary=item.finding.quote[:SUMMARY_CHARS],
            )
        )
    return similar


async def _score(
    asker: Asker, claim: str, result: ResearchResult
) -> tuple[list[_Scored], bool]:
    """Ask both questions of every finding of a kept work, one request each,
    strictly one after another (the decision model is local).

    Returns what was scored and whether the budget stopped it early.
    """
    kept = _kept(result)
    scored: list[_Scored] = []
    for finding in result.findings:
        work = kept.get(finding.work_id)
        if work is None:
            continue  # a retracted or excluded work is not evidence
        try:
            p = await asker.ask_criteria(
                STEP,
                state={
                    "claim_under_review": claim,
                    "finding": finding.claim,
                    "quote": finding.quote,
                    "work": work.title,
                    "year": work.year if work.year is not None else "unknown",
                },
                questions=_QUESTIONS,
            )
        except BudgetExhausted:
            return scored, True
        scored.append(
            _Scored(finding=finding, anticipates=p["anticipates"], related=p["related"])
        )
    return scored, False


def _unverified_note(scored: list[_Scored]) -> str:
    count = sum(1 for s in scored if not s.finding.quote_verified)
    if count == 0:
        return ""
    return (
        f" {count} of the {len(scored)} judged finding(s) have a quote that "
        "could not be found in the page text."
    )


def _verdict(
    claim: str,
    *,
    result: ResearchResult,
    scored: list[_Scored],
    cut_short: bool,
    queries: list[str],
) -> NoveltyReport:
    """The verdict from the scored findings and the health of the search."""
    kept = _kept(result)
    sources = [url for w in kept.values() if (url := _work_url(w)) != ""]
    anticipating = [s for s in scored if s.anticipates >= ANTICIPATES_AT]
    if anticipating:
        top = max(s.anticipates for s in anticipating)
        works = {s.finding.work_id for s in anticipating}
        return NoveltyReport(
            claim=claim,
            is_novel=False,
            confidence=top,
            assessment=(
                f"{len(anticipating)} finding(s) from {len(works)} work(s) report "
                f"what the claim presents as new (decision model p up to {top:.2f}); "
                "their quotes are given under similar work." + _unverified_note(scored)
            ),
            similar_work=_similar(result, scored),
            sources=sources,
            search_queries_used=queries,
        )
    works_read = len({p.work_id for p in result.passages if p.work_id in kept})
    problems: list[str] = []
    if result.degraded:
        problems.append(
            f"the search was degraded ({len(result.degraded)} failure(s): "
            f"{result.degraded[0]})"
        )
    if result.stop_reason.startswith("budget:"):
        problems.append(f"the search stopped on its budget ({result.stop_reason})")
    if cut_short:
        problems.append("the decision budget ran out before every finding was judged")
    if works_read < MIN_WORKS_ASSESSED:
        problems.append(
            f"only {works_read} work(s) were read (at least {MIN_WORKS_ASSESSED} "
            "are needed to conclude that nothing is there)"
        )
    if problems:
        return _undetermined(
            claim,
            why="No finding anticipates the claim, but " + "; ".join(problems) + ".",
            queries=queries,
            sources=sources,
        )
    top_p = max((s.anticipates for s in scored), default=0.0)
    if scored:
        body = (
            f"{len(scored)} finding(s) from {works_read} work(s) were read and "
            f"none reports what the claim presents as new (highest decision model "
            f"p {top_p:.2f}). The search had no failures."
        )
    else:
        body = (
            f"{works_read} work(s) were read and no finding bore on the claim. "
            "The search had no failures."
        )
    return NoveltyReport(
        claim=claim,
        is_novel=True,
        confidence=1.0 - top_p,
        assessment=body + _unverified_note(scored),
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

    ``model`` is the language model (a pydantic-ai id or object) that plans the
    search and reads the works; ``decision_model`` (a name, opened and closed
    here, or an opened handle left open) is the local decision model that judges
    every finding, one request at a time. ``store`` is the research store: a path
    opens or creates a persistent one (works, passages and answers are cached
    there), an open store is used as it is, ``None`` keeps it in memory. A
    failure of the language model or decision model, or a search too thin to
    conclude from, gives ``is_novel=None``, never a guess; anything else raises.
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
        try:
            result = await research(
                f"What published work already reports or anticipates this: {claim}",
                mode="scholarly",
                model=model,
                store=opened,
                budget=budget,
                sources=sources,
                source_client=source_client,
                progress=progress,
            )
        except MODEL_ERRORS as error:
            return _undetermined(
                claim,
                why=f"The language model failed during the search: {error}.",
                queries=[],
                sources=[],
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
                why=f"The judgement of the findings failed: {error}.",
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
