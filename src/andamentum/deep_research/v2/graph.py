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

What each swappable step does (the screen, refinement, ranking, passage
selection, expansion, criteria facets, and every cap) is the run's
``ResearchConfig`` (``config.py``; default ``PRESETS['baseline']``). With
``frozen`` inputs (``frozen.py``) Plan and Search come from the snapshot and
every source request is answered from the store's cache or skipped.

Nodes only route and assign state; the work is in ``steps`` and the worker
modules. The store sits behind the deps (L1): every worker writes what it
learns there, so a run can be inspected, and a second run reads the first
one's cache. Every source and model failure is recorded in ``degraded``.

``research()`` is the public entry point.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import JsonValue
from pydantic_ai.models import Model
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext

from andamentum.core.decisions import DecisionModel
from andamentum.references import ReferenceHttp, Work

from . import steps
from .ask import Asker
from .config import PRESETS, ResearchConfig, config_hash
from .decision_cache import CHECK_NAME, check_determinism
from .coverage import PairKey, compute_coverage, decide_stop, record_coverage
from .criteria import Screener
from .frozen import FrozenInputs
from .identity import merge
from .read import ReadDeps, ReadTarget, read_works, unreadable_degraded
from .sources import SourceClient, SourceConfig, SourceSet, build_sources
from .sources.base import offline_url_check
from .store import ResearchStore
from .strategies import screen_one
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
    ScreeningEntry,
    SubQuestion,
    VerifyFlag,
)
from .verify import verify_report
from .write import write_report

Pair = steps.Pair


@dataclass(frozen=True)
class ResearchDeps:
    """What a run is given: the question, the store, the sources, the run's
    models behind one ``Asker``, the configuration, the frozen inputs (if
    any), the budget and the clock."""

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
    config: ResearchConfig = PRESETS["baseline"]
    frozen: FrozenInputs | None = None
    offline_before: dict[str, int] = field(default_factory=dict)


@dataclass
class ResearchState:
    """The run's evidence table as it grows. Written only by nodes."""

    subquestions: list[SubQuestion] = field(default_factory=list)
    criteria: dict[str, list[Criterion]] = field(default_factory=dict)
    round: int = 0
    round_targets: list[str] = field(default_factory=list)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    pending: list[Pair] = field(default_factory=list)
    # Every pair the run found, in the order it found them (the screening record).
    arrivals: list[Pair] = field(default_factory=list)
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
    read_attempted: int = 0
    read_unreadable: int = 0
    finding_ids: Iterator[int] = field(default_factory=lambda: itertools.count(1))
    degraded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    spent: str = ""
    stop_reason: str = ""
    report: str = ""
    verification: list[VerifyFlag] = field(default_factory=list)


Ctx = GraphRunContext[ResearchState, ResearchDeps]


@contextmanager
def _timed(ctx: Ctx, step: str) -> Iterator[None]:
    """Count a node's seconds (on the run's clock) and the model requests made
    while it runs under *step* in the run's stats (``RunStats.step_*``)."""
    deps = ctx.deps
    deps.asker.phase = step
    began = deps.clock()
    try:
        yield
    finally:
        seconds = deps.asker.stats.step_seconds
        seconds[step] = seconds.get(step, 0.0) + (deps.clock() - began)


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


def _kept(ctx: Ctx) -> dict[str, tuple[float, list[str]]]:
    """Each kept work's ranking value (the config's rank strategy) and the
    sub-questions it was kept for."""
    return ctx.deps.config.rank_strategy.kept(
        ctx.state.scores, candidates=ctx.state.candidates
    )


def _evidence_criteria(ctx: Ctx) -> dict[str, list[Criterion]]:
    """The criteria that travel with the evidence table (the facets strategy)."""
    return ctx.deps.config.facet_strategy.evidence(ctx.state.criteria)


def _works(state: ResearchState) -> dict[str, Work]:
    return {work_id: c.work for work_id, c in state.candidates.items()}


def _admit(ctx: Ctx, found: list[Candidate]) -> None:
    state = ctx.state
    admitted = steps.admit(found, store=ctx.deps.store, known=state.candidates)
    state.candidates = admitted.candidates
    waiting = set(state.pending) | state.screened
    fresh = [p for p in admitted.pairs if p not in waiting]
    state.pending += fresh
    state.arrivals += fresh


@dataclass
class Plan(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """Sub-questions (planned, or the frozen snapshot's), and the screening
    criteria compiled for each when the configuration uses criteria."""

    async def run(self, ctx: Ctx) -> Search:
        with _timed(ctx, "Plan"):
            state, deps = ctx.state, ctx.deps
            if deps.frozen is not None:
                deps.progress("Plan: sub-questions from the frozen inputs")
                state.subquestions = list(deps.frozen.subquestions)
            else:
                deps.progress("Plan: sub-questions")
                planned = await steps.plan(deps.question, asker=deps.asker)
                _note(state, degraded=planned.degraded, spent=planned.spent)
                state.subquestions = planned.subquestions
            for subquestion in state.subquestions:
                deps.store.add_run_item(
                    deps.asker.run_id, "subquestion", subquestion.id, subquestion
                )
            if not deps.config.needs_criteria:
                state.criteria = {}
                return Search()
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
        with _timed(ctx, "Search"):
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
            if deps.frozen is not None:
                # A frozen run searches once, from the snapshot; later rounds screen
                # what is still waiting and what expansion brings.
                if state.round == 1:
                    deps.progress("Search: the frozen candidates")
                    state.queries_done += deps.frozen.all_queries()
                    _admit(
                        ctx,
                        steps.restore_licensed_text(
                            deps.frozen.candidates, store=deps.store
                        ),
                    )
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
            state.notes += searched.notes
            _admit(ctx, searched.candidates)
            return Screen()


@dataclass
class Screen(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """The criteria scored on every new candidate, at most a round's worth."""

    async def run(self, ctx: Ctx) -> Expand | Read:
        with _timed(ctx, "Screen"):
            state, deps = ctx.state, ctx.deps
            waiting = [p for p in state.pending if p not in state.screened]
            if waiting and not _spent(ctx):
                deps.progress(f"Screen: {len(waiting)} candidates waiting")
                outcome = await steps.screen(
                    # the queue strategy orders what a round's caps may take;
                    # ``pending`` below stays in arrival order
                    deps.config.queue_strategy.order(
                        waiting, candidates=state.candidates
                    ),
                    candidates=state.candidates,
                    subquestions=state.subquestions,
                    screen_one=screen_one(
                        deps.config,
                        criteria=state.criteria,
                        screener=deps.screener,
                        question=deps.question,
                    ),
                    limit=deps.budget.candidates_per_round,
                    per_subquestion=deps.config.screen_per_subquestion,
                )
                _note(state, degraded=outcome.degraded, spent=outcome.spent)
                for score in outcome.scores:
                    pair = (score.work_id, score.subquestion_id)
                    before = state.scores.get(pair)
                    if score.kept and (before is None or not before.kept):
                        state.round_new_kept.setdefault(
                            score.subquestion_id, set()
                        ).add(score.work_id)
                    state.scores[pair] = score
                # a screen without criteria (generic) leaves none behind
                state.criteria.update({k: v for k, v in outcome.criteria.items() if v})
                state.screened |= set(outcome.screened)
                state.pending = [p for p in waiting if p not in state.screened]
            expands = (
                deps.mode != "web"
                and not state.expanded_this_round
                and _expand_limit(deps) > 0
                and deps.config.expand_strategy.enabled
                and bool(deps.sources.expand)
            )
            return Expand() if expands else Read()


def _expand_limit(deps: ResearchDeps) -> int:
    """New candidates a round's expansion may bring: the budget's bound, and
    the configuration's cap within it."""
    cap = deps.config.expand_per_round
    budget = deps.budget.expand_per_round
    return budget if cap is None else min(cap, budget)


@dataclass
class Expand(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """One hop of references and citations from the best kept works."""

    async def run(self, ctx: Ctx) -> Screen:
        with _timed(ctx, "Expand"):
            state, deps = ctx.state, ctx.deps
            state.expanded_this_round = True
            seeds = deps.config.expand_strategy.seeds(
                _kept(ctx),
                candidates=state.candidates,
                expanded=state.expanded,
                config=deps.config,
            )
            if not seeds or _spent(ctx):
                return Screen()
            deps.progress(f"Expand: citations of {len(seeds)} works")
            outcome = await steps.expand(
                seeds, sources=deps.sources.expand, limit=_expand_limit(deps)
            )
            _note(state, degraded=outcome.degraded)
            state.skipped += outcome.skipped
            state.notes += outcome.notes
            state.expanded |= set(outcome.expanded)
            _admit(ctx, merge(outcome.candidates))
            return Screen()


@dataclass
class Read(BaseNode[ResearchState, ResearchDeps, ResearchResult]):
    """Kept works read into passages and findings with checked quotes."""

    async def run(self, ctx: Ctx) -> Coverage:
        with _timed(ctx, "Read"):
            state, deps = ctx.state, ctx.deps
            state.new_kept_by_round.append(
                len({w for ids in state.round_new_kept.values() for w in ids})
            )
            if _spent(ctx):
                return Coverage()
            covered = {c.subquestion_id for c in state.coverage if c.covered}
            chosen = steps.read_targets(
                _kept(ctx),
                done=state.read_done,
                excluded=state.excluded,
                covered=covered,
                limit=deps.budget.read_per_round,
                per_subquestion=deps.config.read_per_subquestion,
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
                criteria=_evidence_criteria(ctx),
                deps=deps.reader,
                ids=state.finding_ids,
                skip=state.read_done,
            )
            _note(state, degraded=outcome.degraded, spent=outcome.stopped)
            state.notes += outcome.failed_notes
            state.read_attempted += outcome.attempted
            state.read_unreadable += outcome.unreadable
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
        with _timed(ctx, "Coverage"):
            state, deps = ctx.state, ctx.deps
            spent = _spent(ctx)
            new_kept = {
                s: len(state.round_new_kept.get(s, ())) for s in state.round_targets
            }
            state.coverage = compute_coverage(
                state.subquestions,
                criteria=_evidence_criteria(ctx),
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
        with _timed(ctx, "Write"):
            state, deps = ctx.state, ctx.deps
            deps.progress(f"Write: {len(state.findings)} findings")
            report = await write_report(
                deps.question,
                subquestions=state.subquestions,
                criteria=_evidence_criteria(ctx),
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
        with _timed(ctx, "Verify"):
            state, deps = ctx.state, ctx.deps
            deps.progress("Verify: claims, quotes and links")
            outcome = await verify_report(
                state.report,
                findings=state.findings,
                passages=state.passages,
                works=_works(state),
                asker=deps.asker,
                http=deps.verify_http,
                code_first=deps.config.verify_strategy.code_first,
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
            config_name=deps.config.name,
            config_hash=config_hash(deps.config),
            frozen=deps.frozen.digest() if deps.frozen is not None else "",
            findings=state.findings,
            report_markdown=state.report,
        )
        ordered = draft.cited_work_ids()
        by_score = sorted(_kept(ctx).items(), key=lambda item: (-item[1][0], item[0]))
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
                "degraded": steps.unique(
                    state.degraded
                    + unreadable_degraded(
                        attempted=state.read_attempted,
                        unreadable=state.read_unreadable,
                    )
                ),
                "skipped": steps.unique(state.skipped + _offline_skips(deps)),
                "notes": steps.unique(state.notes),
                "stats": deps.asker.stats,
            }
        )
        result = result.model_copy(
            update={"screening": _screening_record(state, result.works)}
        )
        deps.progress(f"Finish: {state.stop_reason}")
        return End(result)


def _screening_record(
    state: ResearchState, works: Sequence[Work]
) -> list[ScreeningEntry]:
    """One entry per pair the run found, in arrival order, with the screen's
    score (``None`` for a pair never screened), its decision and the work's
    final rank among the result's works."""
    place = {w.work_id: position for position, w in enumerate(works, start=1)}
    entries: list[ScreeningEntry] = []
    for arrival, pair in enumerate(state.arrivals):
        work_id, subquestion_id = pair
        candidate = state.candidates.get(work_id)
        if candidate is None:
            continue
        score = state.scores.get(pair)
        work = candidate.work
        kept = score is not None and score.kept
        entries.append(
            ScreeningEntry(
                work_id=work_id,
                subquestion_id=subquestion_id,
                arrival=arrival,
                source_ranks=dict(candidate.source_ranks),
                p=None if score is None else score.score,
                kept=kept,
                rank=place.get(work_id) if kept else None,
                title=work.title,
                year=work.year,
                doi=work.doi,
                arxiv_id=work.arxiv_id,
                pmid=work.pmid,
                pmcid=work.pmcid,
                openalex_id=work.openalex_id,
                s2_id=work.s2_id,
                local_id=work.local_id,
                url=work.url,
            )
        )
    return entries


def _offline_skips(deps: ResearchDeps) -> list[str]:
    """What an offline (frozen) run did not fetch, per source or host."""
    misses = deps.sources.client.offline_misses
    lines: list[str] = []
    for name in sorted(misses):
        count = misses[name] - deps.offline_before.get(name, 0)
        if count > 0:
            lines.append(f"{name}: {count} request(s) not in the snapshot; not fetched")
    return lines


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


def _run_config(
    budget: Budget,
    sources: SourceConfig,
    *,
    config: ResearchConfig,
    frozen: FrozenInputs | None,
) -> dict[str, JsonValue]:
    """What the run was asked to do, secrets left out: the budget, the
    sources, the configuration and its hash, and the frozen inputs' digest."""
    return {
        "budget": budget.model_dump(mode="json"),
        "sources": sources.model_dump(
            mode="json",
            exclude={
                "mailto",
                "openalex_api_key",
                "semantic_scholar_api_key",
                "local_corpus",
            },
        ),
        "research_config": config.model_dump(mode="json"),
        "config_hash": config_hash(config),
        "frozen": (
            {"digest": frozen.digest(), "captured_at": frozen.captured_at}
            if frozen is not None
            else None
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
    config: ResearchConfig = PRESETS["baseline"],
    frozen: FrozenInputs | None = None,
    source_client: SourceClient | None = None,
    http_cache: ResearchStore | None = None,
    decision_cache: bool = True,
    shared_decisions: ResearchStore | None = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ResearchResult:
    """Research *question* and return the evidence table with a verified report.

    ``config`` is what the run does at every swappable step (``config.py``;
    the default is the ``baseline`` preset, ``PRESETS['criteria']`` the
    criteria arm); its hash is recorded on the run and the result.

    ``frozen`` replays a snapshot (``frozen.capture``): its sub-questions and
    candidates replace Plan and Search, and every source request is answered
    from the store's cache or skipped, never sent (``source_client``, when
    passed, must be offline: ``SourceClient(store, offline=True)``). The
    question and mode must be the snapshot's; pass the store it was captured
    over, or its cached texts and citations are not there to use.

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

    ``http_cache`` is a shared read-through HTTP cache (a ``ResearchStore``
    kept apart from the run's own): every source request is answered from it
    when it holds a live answer, else sent and written into it; the run's own
    store receives a copy of each entry the run used. It applies to a store
    this call opens, so it cannot be combined with an open ``store`` (open
    that one with ``shared_cache=``) or with ``frozen`` (a frozen run reads
    only its snapshot).

    ``decision_cache`` (on by default, for every preset) serves a decision
    answer from the run's store when the same model (by digest) already
    answered the same request about the same state. The model is first checked
    to be deterministic (``decision_determinism``: one fixed request sent
    twice, not counted in the run's decision calls); if its answers differ the
    cache stays off for the run, and a note and the run's stats say so. It does
    nothing without a decision model.

    ``shared_decisions`` is a second store the decision cache reads through to
    and writes to as well (an experiment's cells all screen the same works for
    the same sub-questions): an own miss is looked up there, and every answer
    goes to both stores. The determinism verdict of the model is read from it
    too. It changes what a run costs, never what it finds: a cached answer is
    the answer a request would get. Allowed with ``frozen``; ignored without a
    decision model or while the cache is off or disabled for the model.

    A spent budget ends the search with ``stop_reason='budget:<name>'`` and
    still writes and verifies the report from what was found. A failing source
    or model is recorded in ``degraded``. Anything else raises: the run is
    recorded as failed in the store first.
    """
    if source_client is not None and source_client.store is not store:
        raise ValueError("source_client must belong to the open store passed as store")
    if frozen is not None:
        if frozen.question != question or frozen.mode != mode:
            raise ValueError(
                "frozen inputs are for another question or mode: "
                f"{frozen.question!r} ({frozen.mode}), not {question!r} ({mode})"
            )
        if source_client is not None and not source_client.offline:
            raise ValueError(
                "a frozen run needs an offline source client: "
                "SourceClient(store, offline=True), or none"
            )
    if http_cache is not None and (
        frozen is not None or isinstance(store, ResearchStore)
    ):
        raise ValueError(
            "http_cache goes to a store this call opens, and not to a frozen "
            "run: open the store with shared_cache= instead"
        )
    owns_store = not isinstance(store, ResearchStore)
    opened: ResearchStore = (
        store
        if isinstance(store, ResearchStore)
        else ResearchStore(store, shared_cache=http_cache)
        if store is not None
        else ResearchStore.in_memory(shared_cache=http_cache)
    )
    client = source_client or SourceClient(
        opened, offline=frozen is not None, pdf_extraction=sources.pdf_extraction
    )
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
            sources=sources,
            config=config,
            frozen=frozen,
            source_set=source_set,
            decision_cache=decision_cache,
            shared_decisions=shared_decisions,
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
    sources: SourceConfig,
    config: ResearchConfig,
    frozen: FrozenInputs | None,
    source_set: SourceSet,
    decision_cache: bool,
    shared_decisions: ResearchStore | None,
    progress: Callable[[str], None],
    clock: Callable[[], float],
) -> ResearchResult:
    run_id = store.start_run(
        question=question,
        mode=mode,
        config=_run_config(budget, sources, config=config, frozen=frozen),
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
        model_settings=config.model_settings(),
        shared_decisions=shared_decisions,
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
            select=config.passage_strategy,
            passages_per_work=config.passages_per_work,
        ),
        verify_http=ReferenceHttp(
            client=client.http,
            cache=client.reference_cache(),
            mailto=sources.mailto,
            url_check=offline_url_check if client.offline else None,
        ),
        budget=budget,
        results_per_query=sources.results_per_query,
        started=started,
        clock=clock,
        progress=progress,
        config=config,
        frozen=frozen,
        offline_before=dict(client.offline_misses),
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
        state = ResearchState()
        if decision_model is not None and decision_cache:
            check = await check_determinism(
                decision_model, store=store, shared=shared_decisions
            )
            asker.decision_cache = check.enabled
            stats.decision_cache = check.state
            stats.decision_probe_calls = check.probe_calls
            store.add_run_item(
                run_id,
                "check",
                CHECK_NAME,
                {
                    "model": decision_model.id,
                    "digest": decision_model.digest or "",
                    "deterministic": check.deterministic,
                    "cache": "enabled" if check.enabled else "disabled",
                    "detail": check.detail,
                    "probe_calls": check.probe_calls,
                    "reused": check.reused,
                },
            )
            if not check.enabled:
                state.notes.append(f"decision cache disabled: {check.detail}")
        result = await graph.run(state=state, deps=deps, inputs=Plan())
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
