"""The v2 engine: plan, then per sub-question a loop of queries, search, choose,
read (with notes) and reflect, then one write and a code-only verify.

Plain async functions, no graph engine: this module routes and assigns state,
the step modules (``plan``, ``queries``, ``search``, ``choose``, ``reflect``,
``read``, ``write``, ``verify``, ``fusion``) compute and return values. Every
loop and fan-out is bounded by a ``Budget`` value or a named constant, and the
model answers questions; it never drives the flow.

Sub-questions run in parallel (``asyncio.gather``); one semaphore inside the
``Asker`` bounds the model calls in flight whoever makes them. A sub-question
keeps its own state (candidates, what it read, its findings); the run merges
them in sub-question order when all are done, so finding ids ``F1..`` and the
ranking do not depend on which task was faster. While a sub-question runs, its
findings are numbered 1..n locally.

A spent bound (``max_wall_s``, ``max_llm_calls``, ``max_fetch_bytes``) ends the
search with ``stop_reason='budget:<name>'``; the report is still written from
what was found. It is a result, never an exception.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.models import Model

from andamentum.references import ReferenceHttp, Work

from . import fusion, runlog
from .ask import Asker, BudgetExhausted
from .choose import choose
from .identity import merge, merge_pair
from .plan import filters_for, plan
from .queries import QueriesOutcome, fresh_queries, write_queries
from .read import ReadDeps, read_page, unreadable_degraded
from .reflect import Confidence, reflect
from .runlog import RunLogger
from .search import admit, expand, expansion_seeds, is_scholarly, search
from .sources import SourceClient, SourceConfig, SourceSet, build_sources
from .store import ResearchStore
from .types import (
    Budget,
    Candidate,
    Finding,
    LogStep,
    Mode,
    Passage,
    PageReading,
    RawFinding,
    ResearchResult,
    RunStats,
    SubQuestion,
    SubQuestionResult,
    VerifyFlag,
)
from .verify import verify_report
from .write import write_report

# Language-model calls the search leaves for writing the report.
WRITE_RESERVE = 2


def unique(items: Sequence[str]) -> list[str]:
    """*items* without repeats, in order."""
    return list(dict.fromkeys(items))


@dataclass
class _Sink:
    """Where failures and remarks are collected."""

    degraded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class _Run(_Sink):
    """What the whole run shares. Single-threaded asyncio: a plain attribute
    is safe to change as long as no ``await`` sits between read and write."""

    question: str
    mode: Mode
    store: ResearchStore
    sources: SourceSet
    asker: Asker
    log: RunLogger
    budget: Budget
    clock: Callable[[], float]
    started: float
    progress: Callable[[str], None]
    reader: ReadDeps
    results_per_query: int
    fetched_before: int
    spent: str = ""
    pages_read: int = 0
    pages_unreadable: int = 0

    @property
    def stats(self) -> RunStats:
        return self.asker.stats

    @property
    def logger(self) -> RunLogger:
        return self.log

    @contextmanager
    def timed(self, step: LogStep) -> Iterator[None]:
        """Count busy seconds under *step* on the run's clock (summed over the
        tasks that run in parallel, so the total can exceed the wall time)."""
        began = self.clock()
        try:
            yield
        finally:
            seconds = self.stats.step_seconds
            seconds[step] = seconds.get(step, 0.0) + (self.clock() - began)

    def spend(self, bound: str) -> None:
        """Remember the first bound that was hit; the search stops everywhere."""
        if bound and not self.spent:
            self.spent = bound
            self.logger.event(
                "budget_spent", f"budget spent: {bound}", step="Finish", bound=bound
            )

    def bound(self) -> str:
        """The bound that stops the search, or ``''``: one already hit, the wall
        clock, the model calls left for the report, or the bytes fetched."""
        if self.spent:
            return self.spent
        if self.clock() - self.started >= self.budget.max_wall_s:
            self.spend("max_wall_s")
        elif self.budget.max_llm_calls - self.stats.llm_calls <= WRITE_RESERVE:
            self.spend("max_llm_calls")
        elif self.fetched() >= self.budget.max_fetch_bytes:
            self.spend("max_fetch_bytes")
        return self.spent

    def fetched(self) -> int:
        return self.sources.client.fetched_bytes - self.fetched_before


@dataclass(kw_only=True)
class _Sub(_Sink):
    """One sub-question's state."""

    subquestion: SubQuestion
    candidates: dict[str, Candidate] = field(default_factory=dict)
    # Works tried for this sub-question (read, unreadable or excluded), in order.
    attempted: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    findings: list[RawFinding] = field(default_factory=list)
    passages: dict[str, Passage] = field(default_factory=dict)
    queries_done: list[str] = field(default_factory=list)
    expanded: set[str] = field(default_factory=set)
    # Scholarly works read that gave findings: how many (the seed's worth).
    seed_scores: dict[str, float] = field(default_factory=dict)
    next_queries: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    answered: bool = False
    answer: str = ""
    confidence: Confidence = "low"
    rounds: int = 0
    stop_reason: str = ""


def _record(
    run: _Run,
    sink: _Sink,
    *,
    step: LogStep,
    degraded: Sequence[str] = (),
    notes: Sequence[str] = (),
    skipped: Sequence[str] = (),
    spent: str = "",
) -> None:
    sink.degraded += degraded
    sink.notes += notes
    sink.skipped += skipped
    runlog.records(run.log, step=step, degraded=degraded, notes=notes, skipped=skipped)
    run.spend(spent)


def _works_of(sub: _Sub) -> dict[str, Work]:
    return {work_id: c.work for work_id, c in sub.candidates.items()}


def _model_id(model: str | Model) -> str:
    return model if isinstance(model, str) else f"{model.system}:{model.model_name}"


# -- one sub-question -----------------------------------------------------------


async def _queries_for(run: _Run, sub: _Sub, round_number: int) -> list[str]:
    """The queries of this round and, in the log, where they came from."""
    count = run.budget.queries_per_round
    with run.timed("Search"):
        outcome: QueriesOutcome | None = None
        if round_number == 1:
            outcome = await write_queries(
                sub.subquestion,
                question=run.question,
                findings=[],
                gaps=[],
                done=sub.queries_done,
                count=count,
                asker=run.asker,
                include_own_text=True,
            )
            queries, origin = outcome.queries, outcome.origin
        else:
            queries = fresh_queries(
                sub.next_queries, done=sub.queries_done, count=count
            )
            origin = "reflect"
            if not queries:
                outcome = await write_queries(
                    sub.subquestion,
                    question=run.question,
                    findings=sub.findings,
                    gaps=sub.gaps,
                    done=sub.queries_done,
                    count=count,
                    asker=run.asker,
                )
                queries, origin = outcome.queries, "writer"
        if outcome is not None:
            _record(
                run, sub, step="Search", degraded=outcome.degraded, spent=outcome.spent
            )
    run.logger.event(
        "queries",
        f"{sub.subquestion.id} round {round_number}: {len(queries)} queries ({origin})",
        step="Search",
        subquestion=sub.subquestion.id,
        round=round_number,
        queries=queries,
        origin=origin,
    )
    return queries


async def _search(run: _Run, sub: _Sub, queries: list[str], round_number: int) -> None:
    run.progress(f"{sub.subquestion.id} round {round_number}: searching")
    with run.timed("Search"):
        found = await search(
            queries,
            subquestion_id=sub.subquestion.id,
            sources=run.sources.search,
            filters=filters_for(sub.subquestion),
            limit=run.results_per_query,
        )
        sub.queries_done += found.queries_run
        runlog.query_hits(run.logger, found.hits)
        runlog.search_results(run.logger, found.results, round_number=round_number)
        _record(
            run,
            sub,
            step="Search",
            degraded=found.degraded,
            notes=found.notes,
            skipped=found.skipped,
        )
        _admit(
            run,
            sub,
            found.candidates,
            step="Search",
            via=f"search round {round_number}",
        )


def _admit(
    run: _Run, sub: _Sub, found: Sequence[Candidate], *, step: LogStep, via: str
) -> None:
    held = len(sub.candidates)
    admitted = admit(found, store=run.store, known=sub.candidates)
    sub.candidates = admitted.candidates
    runlog.admitted(
        run.logger, before=held, after=len(sub.candidates), step=step, via=via
    )


async def _expand(run: _Run, sub: _Sub) -> None:
    """Scholarly and both modes, from round 2: one hop of citations from the
    best works read so far that gave findings."""
    if run.mode == "web" or not run.sources.expand or run.budget.expand_per_round < 1:
        return
    seeds = expansion_seeds(
        sub.seed_scores, candidates=sub.candidates, expanded=sub.expanded
    )
    if not seeds:
        return
    run.progress(f"{sub.subquestion.id}: citations of {len(seeds)} works")
    with run.timed("Expand"):
        outcome = await expand(
            seeds,
            subquestion_id=sub.subquestion.id,
            sources=run.sources.expand,
            limit=run.budget.expand_per_round,
        )
        _record(
            run,
            sub,
            step="Expand",
            degraded=outcome.degraded,
            notes=outcome.notes,
            skipped=outcome.skipped,
        )
        sub.expanded |= set(outcome.expanded)
        found = merge(outcome.candidates)
        runlog.expanded(
            run.logger, subquestion=sub.subquestion.id, seeds=seeds, found=found
        )
        _admit(run, sub, found, step="Expand", via="citation expansion")


async def _read_one(
    run: _Run, sub: _Sub, candidate: Candidate, queries: list[str], round_number: int
) -> PageReading:
    with run.timed("Read"):
        return await read_page(
            candidate,
            subquestion=sub.subquestion,
            queries=queries,
            deps=run.reader,
            round_number=round_number,
        )


def _absorb(run: _Run, sub: _Sub, candidate: Candidate, reading: PageReading) -> None:
    """Put what a page gave into the sub-question's state."""
    work_id = candidate.work.work_id
    sub.attempted.append(work_id)
    run.pages_read += 1
    if reading.excluded:
        sub.excluded[work_id] = reading.excluded
    elif not reading.readable:
        run.pages_unreadable += 1
    sub.candidates[work_id] = candidate.model_copy(update={"work": reading.work})
    sub.passages.update({p.passage_id: p for p in reading.passages})
    sub.findings += reading.findings
    if reading.findings and is_scholarly(reading.work):
        sub.seed_scores[work_id] = float(len(reading.findings))
    _record(
        run,
        sub,
        step="Read",
        degraded=reading.degraded,
        notes=reading.failed_notes,
    )


async def _read_pages(
    run: _Run,
    sub: _Sub,
    picks: Sequence[Candidate],
    reserves: Sequence[Candidate],
    *,
    queries: list[str],
    round_number: int,
) -> None:
    """Read the picks in parallel; a pick that gave nothing (unreadable, or a
    retracted work) is replaced by the next reserve, each reserve used once."""
    batch = list(picks)
    spare = list(reserves)
    while batch:
        results = await asyncio.gather(
            *(_read_one(run, sub, c, queries, round_number) for c in batch),
            return_exceptions=True,
        )
        failed = 0
        for candidate, result in zip(batch, results, strict=True):
            if isinstance(result, BudgetExhausted):
                run.spend(result.name)
            elif isinstance(result, BaseException):
                raise result
            else:
                _absorb(run, sub, candidate, result)
                if not result.readable:
                    failed += 1
        if run.bound() or not failed or not spare:
            return
        batch, spare = spare[:failed], spare[failed:]


async def _round(run: _Run, sub: _Sub, round_number: int) -> str:
    """One round; returns the stop reason when the sub-question is done."""
    subquestion = sub.subquestion
    sub.rounds = round_number
    run.logger.event(
        "round_start",
        f"{subquestion.id} round {round_number}",
        step="Search",
        subquestion=subquestion.id,
        round=round_number,
    )
    queries = await _queries_for(run, sub, round_number)
    if bound := run.bound():
        return f"budget:{bound}"
    if queries:
        await _search(run, sub, queries, round_number)
    if round_number >= 2:
        await _expand(run, sub)
    unread = [
        c for work_id, c in sub.candidates.items() if work_id not in sub.attempted
    ]
    with run.timed("Choose"):
        choice = await choose(
            subquestion,
            question=run.question,
            candidates=unread,
            k=run.budget.reads_per_round,
            asker=run.asker,
        )
        _record(run, sub, step="Choose", degraded=choice.degraded, spent=choice.spent)
    runlog.choice(
        run.logger,
        subquestion=subquestion.id,
        round_number=round_number,
        shown=choice.shown,
        picks=choice.picks,
        reserves=choice.reserves,
        fallback=choice.fallback,
    )
    if bound := run.bound():
        return f"budget:{bound}"
    if choice.picks:
        run.progress(
            f"{subquestion.id} round {round_number}: reading {len(choice.picks)} pages"
        )
        await _read_pages(
            run,
            sub,
            choice.picks,
            choice.reserves,
            queries=queries,
            round_number=round_number,
        )
    if bound := run.bound():
        return f"budget:{bound}"
    with run.timed("Reflect"):
        judged = await reflect(
            subquestion,
            findings=sub.findings,
            works=_works_of(sub),
            done=sub.queries_done,
            count=run.budget.queries_per_round,
            asker=run.asker,
        )
        _record(run, sub, step="Reflect", degraded=judged.degraded, spent=judged.spent)
    if not judged.spent:
        sub.answered = judged.answered
        sub.answer = judged.answer
        sub.confidence = judged.confidence
        sub.gaps = judged.gaps
        sub.next_queries = judged.next_queries
        runlog.reflected(
            run.logger,
            subquestion=subquestion.id,
            round_number=round_number,
            answered=judged.answered,
            answer=judged.answer,
            confidence=judged.confidence,
            gaps=judged.gaps,
            next_queries=judged.next_queries,
        )
    if judged.spent:
        return f"budget:{judged.spent}"
    return "answered" if judged.answered else ""


async def _research_subquestion(run: _Run, sub: _Sub) -> None:
    """The loop of one sub-question: rounds until answered, the round cap, or a
    spent budget. Never raises a spent budget."""
    try:
        for round_number in range(1, run.budget.max_rounds + 1):
            if bound := run.bound():
                sub.stop_reason = f"budget:{bound}"
                break
            sub.stop_reason = await _round(run, sub, round_number)
            if sub.stop_reason:
                break
        else:
            sub.stop_reason = "rounds"
    except BudgetExhausted as spent:
        run.spend(spent.name)
        sub.stop_reason = f"budget:{spent.name}"
    run.logger.event(
        "subquestion_done",
        f"{sub.subquestion.id}: {sub.stop_reason} after {sub.rounds} rounds",
        step="Reflect",
        subquestion=sub.subquestion.id,
        stop_reason=sub.stop_reason,
        rounds=sub.rounds,
    )


# -- the run --------------------------------------------------------------------


def _run_stop_reason(subs: Sequence[_Sub]) -> str:
    if all(s.stop_reason == "answered" for s in subs):
        return "answered"
    for sub in subs:
        if sub.stop_reason.startswith("budget:"):
            return sub.stop_reason
    return "rounds"


def _log_setup(run: _Run, *, model: str, source_set: SourceSet) -> None:
    run.logger.event(
        "config",
        f"question, mode {run.mode}",
        step="Setup",
        question=run.question,
        mode=run.mode,
        budget=run.budget.model_dump(mode="json"),
    )
    run.logger.event("models", f"language model {model}", step="Setup", llm=model)
    run.logger.event(
        "sources",
        "search sources: "
        + (", ".join(s.name for s in source_set.search) or "none")
        + "; expansion: "
        + (", ".join(s.name for s in source_set.expand) or "none"),
        step="Setup",
        search=[s.name for s in source_set.search],
        expand=[s.name for s in source_set.expand],
        text=[getattr(s, "name", type(s).__name__) for s in source_set.text],
    )


async def research(
    question: str,
    *,
    mode: Mode,
    model: str | Model,
    store: Path | ResearchStore | None = None,
    budget: Budget = Budget(),
    sources: SourceConfig = SourceConfig(),
    source_client: SourceClient | None = None,
    http_cache: ResearchStore | None = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ResearchResult:
    """Research *question* and return the evidence table with a verified report.

    ``mode`` picks the sources: ``web`` (a Chrome over CDP, ``sources.browser_cdp_url``,
    or SearxNG, ``sources.searxng_url``; one of them is required), ``scholarly``
    (``sources.scholarly``) or ``both``. ``model`` is the language model (a
    pydantic-ai model id or object); it is the only model the engine uses.

    ``store`` is the research store: ``None`` keeps everything in memory for this
    run; a path opens (or creates) a persistent one; an open ``ResearchStore`` is
    used as it is. ``source_client`` is the HTTP client the sources share (built
    for the store when not given; one passed must belong to the open store passed
    as ``store``). ``http_cache`` is a shared read-through HTTP cache (a
    ``ResearchStore`` kept apart from the run's own): every source request is
    answered from it when it holds a live answer, else sent and written into it;
    the run's own store receives a copy of each entry the run used. It applies to
    a store this call opens, so it cannot be combined with an open ``store``
    (open that one with ``shared_cache=``). ``progress`` receives one line per
    step; ``clock`` is the clock ``budget.max_wall_s`` is measured on.

    Every run fills ``result.log`` (``RunLog``: what happened at each step); with
    a store it is also saved as the run item ``log/run``. A spent budget ends the
    search with ``stop_reason='budget:<name>'`` and still writes and verifies the
    report from what was found. A failing source or model is recorded in
    ``degraded``. Anything else raises: the run is recorded as failed in the
    store first, and the log so far is the error's ``run_log`` attribute.
    """
    if source_client is not None and source_client.store is not store:
        raise ValueError("source_client must belong to the open store passed as store")
    if http_cache is not None and isinstance(store, ResearchStore):
        raise ValueError(
            "http_cache goes to a store this call opens: open the store with "
            "shared_cache= instead"
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
        opened, pdf_extraction=sources.pdf_extraction
    )
    try:
        source_set = build_sources(sources, store=opened, mode=mode, client=client)
        return await _run(
            question,
            mode=mode,
            model=model,
            store=opened,
            budget=budget,
            sources=sources,
            source_set=source_set,
            progress=progress or (lambda _line: None),
            clock=clock,
        )
    finally:
        if source_client is None:
            await client.aclose()
        if owns_store:
            opened.close()


async def _run(
    question: str,
    *,
    mode: Mode,
    model: str | Model,
    store: ResearchStore,
    budget: Budget,
    sources: SourceConfig,
    source_set: SourceSet,
    progress: Callable[[str], None],
    clock: Callable[[], float],
) -> ResearchResult:
    run_id = store.start_run(
        question=question,
        mode=mode,
        config={
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
        },
        models={"model": _model_id(model)},
    )
    log = RunLogger(clock=clock, run_id=run_id)
    stats = RunStats()
    asker = Asker(
        model=model,
        store=store,
        run_id=run_id,
        budget=budget,
        stats=stats,
        log=log,
        reserve=WRITE_RESERVE,
    )
    client = source_set.client
    hits, misses = store.cache_hits, store.cache_misses
    fetched, requests = client.fetched_bytes, dict(client.requests)
    started = clock()
    run = _Run(
        question=question,
        mode=mode,
        store=store,
        sources=source_set,
        asker=asker,
        log=log,
        budget=budget,
        clock=clock,
        started=started,
        progress=progress,
        reader=ReadDeps(
            asker=asker,
            crossref=source_set.crossref,
            providers=source_set.text,
            window_chars=budget.window_chars,
            fetched_bytes=lambda: client.fetched_bytes - fetched,
        ),
        results_per_query=sources.results_per_query,
        fetched_before=fetched,
    )
    verify_http = ReferenceHttp(
        client=client.http, cache=client.reference_cache(), mailto=sources.mailto
    )

    def finish_stats() -> RunStats:
        stats.wall_s = clock() - started
        stats.cache_hits = store.cache_hits - hits
        stats.cache_misses = store.cache_misses - misses
        stats.fetched_bytes = client.fetched_bytes - fetched
        stats.pages_read = run.pages_read
        stats.pages_unreadable = run.pages_unreadable
        stats.per_source = {
            name: count - requests.get(name, 0)
            for name, count in client.requests.items()
            if count - requests.get(name, 0)
        }
        return stats

    try:
        _log_setup(run, model=_model_id(model), source_set=source_set)
        result = await _research(run, verify_http=verify_http)
    except BaseException as error:
        stats_now = finish_stats()
        log.event(
            "failed",
            f"the run failed: {type(error).__name__}: {error}",
            step="Finish",
            error=f"{type(error).__name__}: {error}",
            wall_s=round(stats_now.wall_s, 3),
        )
        store.finish_run(
            run_id,
            stats=stats_now.model_dump(mode="json"),
            degraded=[f"run failed: {type(error).__name__}: {error}"],
        )
        store.add_run_item(run_id, "log", "run", log.log)
        try:
            error.run_log = log.snapshot()  # type: ignore[attr-defined]
        except (AttributeError, TypeError):  # an exception that takes no attribute
            pass
        raise
    result = result.model_copy(update={"stats": finish_stats()})
    log.event(
        "finished",
        f"finished in {result.stats.wall_s:.1f} s: {result.stats.llm_calls} "
        f"language-model calls; {len(result.findings)} findings, "
        f"{len(result.works)} ranked works",
        step="Finish",
        stop_reason=result.stop_reason,
        wall_s=round(result.stats.wall_s, 3),
        llm_calls=result.stats.llm_calls,
        degraded=len(result.degraded),
    )
    result = result.model_copy(update={"log": log.snapshot()})
    store.finish_run(
        run_id, stats=result.stats.model_dump(mode="json"), degraded=result.degraded
    )
    store.add_run_item(run_id, "log", "run", result.log)
    return result


async def _research(run: _Run, *, verify_http: ReferenceHttp) -> ResearchResult:
    """Plan, the sub-questions' loops, write, verify, and the result."""
    asker, store = run.asker, run.store

    with run.timed("Plan"):
        run.progress("Plan: sub-questions")
        planned = await plan(run.question, asker=asker)
        _record(run, run, step="Plan", degraded=planned.degraded, spent=planned.spent)
    runlog.subquestions(run.logger, planned.subquestions)
    for subquestion in planned.subquestions:
        store.add_run_item(asker.run_id, "subquestion", subquestion.id, subquestion)

    subs = [_Sub(subquestion=s) for s in planned.subquestions]
    outcomes = await asyncio.gather(
        *(_research_subquestion(run, sub) for sub in subs), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BudgetExhausted):
            run.spend(outcome.name)
        elif isinstance(outcome, BaseException):
            raise outcome
    stop_reason = _run_stop_reason(subs)

    # Merge the sub-questions' states, in sub-question order.
    findings: list[Finding] = []
    for sub in subs:
        for raw in sub.findings:
            finding = Finding.from_raw(raw, number=len(findings) + 1)
            findings.append(finding)
            store.add_run_item(asker.run_id, "finding", finding.finding_id, finding)
    candidates: dict[str, Candidate] = {}
    excluded: dict[str, str] = {}
    passages: dict[str, Passage] = {}
    read_order: list[str] = []
    for sub in subs:
        for work_id, candidate in sub.candidates.items():
            held = candidates.get(work_id)
            candidates[work_id] = (
                merge_pair(held, candidate).model_copy(update={"work": candidate.work})
                if held is not None
                else candidate
            )
        excluded.update(sub.excluded)
        passages.update(sub.passages)
        read_order += sub.attempted
        run.degraded += sub.degraded
        run.notes += sub.notes
        run.skipped += sub.skipped
    read_order = unique(read_order)
    works = {work_id: c.work for work_id, c in candidates.items()}
    answers = {
        sub.subquestion.id: SubQuestionResult(
            subquestion_id=sub.subquestion.id,
            answered=sub.answered,
            answer=sub.answer,
            confidence=sub.confidence,
            gaps=sub.gaps,
            rounds=sub.rounds,
            stop_reason=sub.stop_reason,
        )
        for sub in subs
    }

    with run.timed("Write"):
        run.progress(f"Write: {len(findings)} findings")
        report = await write_report(
            run.question,
            subquestions=planned.subquestions,
            answers=answers,
            findings=findings,
            works=works,
            asker=asker,
        )
        _record(run, run, step="Write", degraded=report.degraded)
    draft = ResearchResult(
        run_id=asker.run_id,
        question=run.question,
        mode=run.mode,
        findings=findings,
        report_markdown=report.markdown,
    )
    cited = draft.cited_work_ids()
    runlog.report_written(
        run.logger,
        report=report.markdown,
        cited_works=len(cited),
        findings=len(findings),
    )

    with run.timed("Verify"):
        run.progress("Verify: quotes and links")
        verified = await verify_report(
            report.markdown,
            findings=findings,
            passages=passages,
            works=works,
            http=verify_http,
        )
        flags: list[VerifyFlag] = verified.flags
        _record(
            run,
            run,
            step="Verify",
            degraded=verified.degraded,
            notes=[f"not checked: {line}" for line in verified.unchecked],
        )
    runlog.flags(run.logger, flags)
    for number, flag in enumerate(flags, start=1):
        store.add_run_item(asker.run_id, "flag", f"{number}:{flag.kind}", flag)

    ranked = fusion.rank_works(
        candidates, cited=cited, chosen=read_order, excluded=set(excluded)
    )
    ranked_works = [works[w] for w in ranked if w in works]
    basis = {
        w: "cited" if w in cited else "read" if w in read_order else "fused"
        for w in ranked
    }
    runlog.final_ranking(
        run.logger,
        works=ranked_works,
        basis=basis,
        source_ranks={w: c.source_ranks for w, c in candidates.items()},
        stop_reason=stop_reason,
    )
    run.progress(f"Finish: {stop_reason}")
    run.stats.quotes_unverified = sum(not f.quote_verified for f in findings)
    unreadable = unreadable_degraded(
        attempted=run.pages_read, unreadable=run.pages_unreadable
    )
    _record(run, run, step="Finish", degraded=unreadable)
    queries = [q for sub in subs for q in sub.queries_done]
    return draft.model_copy(
        update={
            "subquestions": planned.subquestions,
            "subanswers": list(answers.values()),
            "works": ranked_works,
            "excluded": excluded,
            "queries": queries,
            "passages": list(passages.values()),
            "report_markdown": verified.report_markdown,
            "verification": flags,
            "stop_reason": stop_reason,
            "degraded": unique(run.degraded),
            "skipped": unique(run.skipped),
            "notes": unique(run.notes),
        }
    )
