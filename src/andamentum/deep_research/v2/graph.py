"""The v2 orchestrator: a pydantic-graph state machine over the research store.

    Plan -> Search -> Screen -> Expand -> Screen -> Read -> Coverage
        Screen -> Read            web mode, or the round's expansion is done
        Coverage -> Search        a sub-question is still open: the next round
        Coverage -> Write         covered, unresolvable, flat, or a spent budget
    Write -> Verify -> Finish

Every loop is bounded by the run's ``Budget`` (dialect L5): the rounds by
``max_rounds`` (and the stop rules in ``coverage.decide_stop``), the
Screen-Expand-Screen detour by one expansion per round, every model call by
``max_llm_calls`` / ``max_decision_calls`` (counted in ``ask.Asker``), the
clock by ``max_wall_s`` and fetching by ``max_fetch_bytes``. A spent bound is
a ``stop_reason``, never an exception.

Nodes only route and assign state; the work is in ``steps`` and the worker
modules. The store sits behind the deps (L1): every worker writes what it
learns there, so a run can be inspected, and a second run reads the first
one's cache. Every source and model failure is recorded in ``degraded``.

``research()`` is the public entry point.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import JsonValue
from pydantic_ai.models import Model
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext

from andamentum.core.decisions import DecisionModel
from andamentum.references import ReferenceHttp, Work

from . import steps
from .ask import Asker
from .coverage import PairKey, compute_coverage, decide_stop, record_coverage
from .criteria import Screener
from .identity import merge
from .read import ReadDeps, ReadTarget, read_works
from .sources import SourceClient, SourceConfig, SourceSet, build_sources
from .store import ResearchStore
from .types import Coverage as SubQuestionCoverage
from .types import (
    Budget,
    Candidate,
    CandidateScore,
    Conflict,
    Criterion,
    Finding,
    Mode,
    Passage,
    ResearchResult,
    RunStats,
    SubQuestion,
    VerifyFlag,
)
from .verify import verify_report
from .write import write_report

Pair = steps.Pair


@dataclass(frozen=True)
class ResearchDeps:
    """What a run is given: the question, the store, the sources, the run's
    models behind one ``Asker``, the budget and the clock."""

    question: str
    mode: Mode
    store: ResearchStore
    sources: SourceSet
    asker: Asker
    screener: Screener
    reader: ReadDeps
    verify_http: ReferenceHttp
    budget: Budget
    results_per_query: int
    started: float
    clock: Callable[[], float]
    progress: Callable[[str], None]


@dataclass
class ResearchState:
    """The run's evidence table as it grows. Written only by nodes."""

    subquestions: list[SubQuestion] = field(default_factory=list)
    criteria: dict[str, list[Criterion]] = field(default_factory=dict)
    round: int = 0
    round_targets: list[str] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    pending: list[Pair] = field(default_factory=list)
    screened: set[Pair] = field(default_factory=set)
    scores: dict[Pair, CandidateScore] = field(default_factory=dict)
    round_new_kept: dict[str, set[str]] = field(default_factory=dict)
    new_kept_by_round: list[int] = field(default_factory=list)
    expanded: set[str] = field(default_factory=set)
    expanded_this_round: bool = False
    read_done: set[Pair] = field(default_factory=set)
    findings: list[Finding] = field(default_factory=list)
    passages: dict[str, Passage] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)
    coverage: list[SubQuestionCoverage] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    conflict_memory: dict[PairKey, Conflict | None] = field(default_factory=dict)
    queries_done: list[str] = field(default_factory=list)
    finding_ids: Iterator[int] = field(default_factory=lambda: itertools.count(1))
    degraded: list[str] = field(default_factory=list)
    spent: str = ""
    stop_reason: str = ""
    report: str = ""
    verification: list[VerifyFlag] = field(default_factory=list)


Ctx = GraphRunContext[ResearchState, ResearchDeps]


def _spent(ctx: Ctx) -> str:
    """The bound that stops the search, or ``''``: one a worker hit, the wall
    clock, or the language-model calls left for writing the report."""
    state, deps = ctx.state, ctx.deps
    if state.spent:
        return state.spent
    if deps.clock() - deps.started >= deps.budget.max_wall_s:
        return "max_wall_s"
    left = deps.budget.max_llm_calls - deps.asker.stats.llm_calls
    if left <= steps.write_reserve(state.subquestions):
        return "max_llm_calls"
    return ""


def _note(state: ResearchState, *, degraded: list[str], spent: str = "") -> None:
    state.degraded += degraded
    if spent and not state.spent:
        state.spent = spent


def _kept(state: ResearchState) -> dict[str, tuple[float, list[str]]]:
    """Each kept work's best score and the sub-questions it was kept for."""
    kept: dict[str, tuple[float, list[str]]] = {}
    for (work_id, subquestion_id), score in state.scores.items():
        if not score.kept:
            continue
        best, ids = kept.get(work_id, (0.0, []))
        kept[work_id] = (max(best, score.score), [*ids, subquestion_id])
    return kept


def _works(state: ResearchState) -> dict[str, Work]:
    return {work_id: c.work for work_id, c in state.candidates.items()}


def _admit(ctx: Ctx, found: list[Candidate]) -> None:
    state = ctx.state
    admitted = steps.admit(found, store=ctx.deps.store, known=state.candidates)
    state.candidates = admitted.candidates
    waiting = set(state.pending) | state.screened
    state.pending += [p for p in admitted.pairs if p not in waiting]


@dataclass
class Plan(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """Sub-questions, and the screening criteria compiled for each."""

    async def run(self, ctx: Ctx) -> Search:
        state, deps = ctx.state, ctx.deps
        deps.progress("Plan: sub-questions")
        planned = await steps.plan(deps.question, asker=deps.asker)
        _note(state, degraded=planned.degraded, spent=planned.spent)
        state.subquestions = planned.subquestions
        for subquestion in state.subquestions:
            deps.store.add_run_item(
                deps.asker.run_id, "subquestion", subquestion.id, subquestion
            )
        deps.progress(f"Plan: criteria for {len(state.subquestions)} sub-questions")
        compiled = await steps.compile_all(
            state.subquestions, screener=deps.screener, question=deps.question
        )
        _note(state, degraded=compiled.degraded, spent=compiled.spent)
        state.criteria = compiled.criteria
        return Search()


@dataclass
class Search(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """One round's queries for the open sub-questions, run on every source."""

    async def run(self, ctx: Ctx) -> Screen:
        state, deps = ctx.state, ctx.deps
        state.round += 1
        state.round_new_kept = {}
        state.expanded_this_round = False
        closed = {
            c.subquestion_id for c in state.coverage if c.covered or c.unresolvable
        }
        targets = [s for s in state.subquestions if s.id not in closed]
        state.round_targets = [s.id for s in targets]
        if _spent(ctx) or not targets:
            return Screen()
        deps.progress(f"Search: round {state.round}, {len(targets)} sub-questions")
        written = await steps.write_queries(
            deps.question,
            targets=targets,
            gaps=steps.coverage_gaps(state.coverage),
            done=state.queries_done,
            per_subquestion=deps.budget.queries_per_subq,
            asker=deps.asker,
        )
        _note(state, degraded=written.degraded, spent=written.spent)
        searched = await steps.search(
            written.queries,
            sources=deps.sources.search,
            filters={s.id: steps.filters_for(s) for s in targets},
            limit=deps.results_per_query,
        )
        state.queries_done += searched.queries_run
        _note(state, degraded=searched.degraded)
        _admit(ctx, searched.candidates)
        return Screen()


@dataclass
class Screen(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """The criteria scored on every new candidate, at most a round's worth."""

    async def run(self, ctx: Ctx) -> Expand | Read:
        state, deps = ctx.state, ctx.deps
        waiting = [p for p in state.pending if p not in state.screened]
        if waiting and not _spent(ctx):
            deps.progress(f"Screen: {len(waiting)} candidates waiting")
            outcome = await steps.screen(
                waiting,
                candidates=state.candidates,
                subquestions=state.subquestions,
                criteria=state.criteria,
                screener=deps.screener,
                question=deps.question,
                limit=deps.budget.candidates_per_round,
            )
            _note(state, degraded=outcome.degraded, spent=outcome.spent)
            for score in outcome.scores:
                pair = (score.work_id, score.subquestion_id)
                before = state.scores.get(pair)
                if score.kept and (before is None or not before.kept):
                    state.round_new_kept.setdefault(score.subquestion_id, set()).add(
                        score.work_id
                    )
                state.scores[pair] = score
            state.criteria.update(outcome.criteria)
            state.screened |= set(outcome.screened)
            state.pending = [p for p in waiting if p not in state.screened]
        expands = (
            deps.mode != "web"
            and not state.expanded_this_round
            and deps.budget.expand_per_round > 0
            and bool(deps.sources.expand)
        )
        return Expand() if expands else Read()


@dataclass
class Expand(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """One hop of references and citations from the best kept works."""

    async def run(self, ctx: Ctx) -> Screen:
        state, deps = ctx.state, ctx.deps
        state.expanded_this_round = True
        seeds = steps.expansion_seeds(
            _kept(state), candidates=state.candidates, expanded=state.expanded
        )
        if not seeds or _spent(ctx):
            return Screen()
        deps.progress(f"Expand: citations of {len(seeds)} works")
        outcome = await steps.expand(
            seeds, sources=deps.sources.expand, limit=deps.budget.expand_per_round
        )
        _note(state, degraded=outcome.degraded)
        state.expanded |= set(outcome.expanded)
        _admit(ctx, merge(outcome.candidates))
        return Screen()


@dataclass
class Read(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """Kept works read into passages and findings with checked quotes."""

    async def run(self, ctx: Ctx) -> Coverage:
        state, deps = ctx.state, ctx.deps
        state.new_kept_by_round.append(
            len({w for ids in state.round_new_kept.values() for w in ids})
        )
        if _spent(ctx):
            return Coverage()
        covered = {c.subquestion_id for c in state.coverage if c.covered}
        chosen = steps.read_targets(
            _kept(state),
            done=state.read_done,
            excluded=state.excluded,
            covered=covered,
            limit=deps.budget.read_per_round,
        )
        if not chosen:
            return Coverage()
        deps.progress(f"Read: {len(chosen)} works")
        outcome = await read_works(
            [
                ReadTarget(work=state.candidates[w].work, subquestion_ids=ids)
                for w, ids in chosen
            ],
            subquestions=state.subquestions,
            criteria=state.criteria,
            deps=deps.reader,
            ids=state.finding_ids,
            skip=state.read_done,
        )
        _note(state, degraded=outcome.degraded, spent=outcome.stopped)
        for reading in outcome.readings:
            candidate = state.candidates.get(reading.work.work_id)
            if candidate is not None:
                state.candidates[reading.work.work_id] = candidate.model_copy(
                    update={"work": reading.work}
                )
        state.findings += outcome.findings
        state.passages.update({p.passage_id: p for p in outcome.passages})
        state.read_done |= outcome.done
        state.excluded.update(dict(outcome.excluded))
        return Coverage()


@dataclass
class Coverage(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """Coverage per sub-question; stop, or search again for what is open."""

    async def run(self, ctx: Ctx) -> Search | Write:
        state, deps = ctx.state, ctx.deps
        spent = _spent(ctx)
        new_kept = {
            s: len(state.round_new_kept.get(s, ())) for s in state.round_targets
        }
        state.coverage = compute_coverage(
            state.subquestions,
            criteria=state.criteria,
            findings=state.findings,
            works=_works(state),
            candidate_scores=list(state.scores.values()),
            rounds_done=state.round,
            new_kept_last_round=new_kept,
        )
        record_coverage(deps.asker, state.coverage)
        decision = decide_stop(
            state.coverage,
            round_number=state.round,
            new_kept_by_round=state.new_kept_by_round,
            budget=deps.budget,
            spent=spent or None,
        )
        covered = sum(c.covered for c in state.coverage)
        deps.progress(
            f"Coverage: round {state.round}, {covered}/{len(state.coverage)} covered, "
            f"{len(state.findings)} findings"
        )
        if not decision.stop:
            return Search()
        state.stop_reason = decision.reason
        found = await steps.conflicts(
            state.findings,
            works=_works(state),
            asker=deps.asker,
            checked=state.conflict_memory,
        )
        _note(state, degraded=found.degraded)
        state.conflicts = found.conflicts
        return Write()


@dataclass
class Write(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """The report, from the findings table only."""

    async def run(self, ctx: Ctx) -> Verify:
        state, deps = ctx.state, ctx.deps
        deps.progress(f"Write: {len(state.findings)} findings")
        report = await write_report(
            deps.question,
            subquestions=state.subquestions,
            criteria=state.criteria,
            findings=state.findings,
            conflicts=state.conflicts,
            coverage=state.coverage,
            works=_works(state),
            asker=deps.asker,
        )
        _note(state, degraded=report.degraded)
        state.report = report.markdown
        return Verify()


@dataclass
class Verify(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """Every check on the report: quotes, claims, ids, links, retractions."""

    async def run(self, ctx: Ctx) -> Finish:
        state, deps = ctx.state, ctx.deps
        deps.progress("Verify: claims, quotes and links")
        outcome = await verify_report(
            state.report,
            findings=state.findings,
            passages=state.passages,
            works=_works(state),
            asker=deps.asker,
            http=deps.verify_http,
        )
        _note(state, degraded=outcome.degraded)
        state.report = outcome.report_markdown
        state.verification = outcome.flags
        return Finish()


@dataclass
class Finish(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """The result: the evidence table, the verified report, the cost."""

    async def run(self, ctx: Ctx) -> End[ResearchResult]:
        state, deps = ctx.state, ctx.deps
        works = _works(state)
        draft = ResearchResult(
            run_id=deps.asker.run_id,
            question=deps.question,
            mode=deps.mode,
            findings=state.findings,
            report_markdown=state.report,
        )
        ordered = draft.cited_work_ids()
        by_score = sorted(_kept(state).items(), key=lambda item: (-item[1][0], item[0]))
        ordered += [w for w, _ in by_score if w not in ordered]
        result = draft.model_copy(
            update={
                "subquestions": state.subquestions,
                "criteria": state.criteria,
                "works": [
                    works[w] for w in ordered if w in works and w not in state.excluded
                ],
                "excluded": state.excluded,
                "queries": list(state.queries_done),
                "passages": list(state.passages.values()),
                "coverage": state.coverage,
                "conflicts": state.conflicts,
                "verification": state.verification,
                "stop_reason": state.stop_reason,
                "degraded": steps.unique(state.degraded),
                "stats": deps.asker.stats,
            }
        )
        deps.progress(f"Finish: {state.stop_reason}")
        return End(result)


_builder = GraphBuilder[ResearchState, ResearchDeps, Plan, ResearchResult](
    state_type=ResearchState,
    deps_type=ResearchDeps,
    input_type=Plan,
    output_type=ResearchResult,
)
_builder.add(
    _builder.edge_from(_builder.start_node).to(Plan),
    _builder.node(Plan),
    _builder.node(Search),
    _builder.node(Screen),
    _builder.node(Expand),
    _builder.node(Read),
    _builder.node(Coverage),
    _builder.node(Write),
    _builder.node(Verify),
    _builder.node(Finish),
)
graph = _builder.build()


# -- the entry point ------------------------------------------------------------


def _model_id(model: str | Model) -> str:
    return model if isinstance(model, str) else f"{model.system}:{model.model_name}"


def _run_config(budget: Budget, sources: SourceConfig) -> dict[str, JsonValue]:
    """What the run was asked to do, secrets left out."""
    return {
        "budget": budget.model_dump(mode="json"),
        "sources": sources.model_dump(
            mode="json",
            exclude={"mailto", "openalex_api_key", "semantic_scholar_api_key"},
        ),
    }


async def research(
    question: str,
    *,
    mode: Mode,
    model: str | Model,
    decision_model: str | DecisionModel | None = None,
    store: Path | ResearchStore | None = None,
    budget: Budget = Budget(),
    sources: SourceConfig = SourceConfig(),
    source_client: SourceClient | None = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ResearchResult:
    """Research *question* and return the evidence table with a verified report.

    ``mode`` picks the sources: ``web`` (SearxNG; ``sources.searxng_url`` is
    required), ``scholarly`` (``sources.scholarly``) or ``both``. ``model`` is
    the language model (a pydantic-ai model id or object); ``decision_model``
    the local decision model that screens and checks (a name, opened here and
    closed at the end, or an opened handle, left open), or ``None`` to let the
    language model answer those questions.

    ``store`` is the research store: ``None`` keeps everything in memory for
    this run; a path opens (or creates) a persistent one; an open
    ``ResearchStore`` is used as it is. ``source_client`` is the HTTP client
    the sources share (built for the store when not given; one passed must
    belong to the open store passed as ``store``). ``progress`` receives one
    line per step; ``clock`` is the clock ``budget.max_wall_s`` is measured on.

    A spent budget ends the search with ``stop_reason='budget:<name>'`` and
    still writes and verifies the report from what was found. A failing source
    or model is recorded in ``degraded``. Anything else raises: the run is
    recorded as failed in the store first.
    """
    if source_client is not None and source_client.store is not store:
        raise ValueError("source_client must belong to the open store passed as store")
    owns_store = not isinstance(store, ResearchStore)
    opened: ResearchStore = (
        store
        if isinstance(store, ResearchStore)
        else ResearchStore(store)
        if store is not None
        else ResearchStore.in_memory()
    )
    client = source_client or SourceClient(opened)
    handle: DecisionModel | None = None
    try:
        handle = (
            await DecisionModel.open(decision_model)
            if isinstance(decision_model, str)
            else decision_model
        )
        source_set = build_sources(sources, store=opened, mode=mode, client=client)
        return await _run(
            question,
            mode=mode,
            model=model,
            decision_model=handle,
            store=opened,
            budget=budget,
            config=sources,
            source_set=source_set,
            progress=progress or (lambda _line: None),
            clock=clock,
        )
    finally:
        if isinstance(decision_model, str) and handle is not None:
            await handle.aclose()
        if source_client is None:
            await client.aclose()
        if owns_store:
            opened.close()


async def _run(
    question: str,
    *,
    mode: Mode,
    model: str | Model,
    decision_model: DecisionModel | None,
    store: ResearchStore,
    budget: Budget,
    config: SourceConfig,
    source_set: SourceSet,
    progress: Callable[[str], None],
    clock: Callable[[], float],
) -> ResearchResult:
    run_id = store.start_run(
        question=question,
        mode=mode,
        config=_run_config(budget, config),
        models={
            "model": _model_id(model),
            "decision_model": decision_model.id if decision_model else None,
            "decision_digest": decision_model.digest if decision_model else None,
        },
    )
    stats = RunStats()
    asker = Asker(
        model=model,
        store=store,
        run_id=run_id,
        budget=budget,
        stats=stats,
        decision_model=decision_model,
    )
    screener = Screener(asker=asker)
    client = source_set.client
    hits, misses = store.cache_hits, store.cache_misses
    fetched, requests = client.fetched_bytes, dict(client.requests)
    started = clock()
    deps = ResearchDeps(
        question=question,
        mode=mode,
        store=store,
        sources=source_set,
        asker=asker,
        screener=screener,
        reader=ReadDeps(
            screener=screener,
            crossref=source_set.crossref,
            providers=source_set.text,
            fetched_bytes=lambda: client.fetched_bytes - fetched,
        ),
        verify_http=ReferenceHttp(
            client=client.http, cache=store.reference_cache(), mailto=config.mailto
        ),
        budget=budget,
        results_per_query=config.results_per_query,
        started=started,
        clock=clock,
        progress=progress,
    )

    def finish_stats() -> RunStats:
        stats.wall_s = clock() - started
        stats.cache_hits = store.cache_hits - hits
        stats.cache_misses = store.cache_misses - misses
        stats.fetched_bytes = client.fetched_bytes - fetched
        stats.per_source = {
            name: count - requests.get(name, 0)
            for name, count in client.requests.items()
            if count - requests.get(name, 0)
        }
        return stats

    try:
        result = await graph.run(state=ResearchState(), deps=deps, inputs=Plan())
    except BaseException as error:
        store.finish_run(
            run_id,
            stats=finish_stats().model_dump(mode="json"),
            degraded=[f"run failed: {type(error).__name__}: {error}"],
        )
        raise
    result = result.model_copy(update={"stats": finish_stats()})
    store.finish_run(
        run_id, stats=result.stats.model_dump(mode="json"), degraded=result.degraded
    )
    return result
