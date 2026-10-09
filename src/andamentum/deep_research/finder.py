"""FIND: one sentence in, the works it could cite out, ranked (metadata only).

``find_works(sentence, *, model, paragraph=None, follow_citations=True, ...)``:

```
queries   1 call: 2 or 3 keyword queries from the sentence (and its paragraph)
search    every query on every source of ``sources.scholarly`` (Semantic Scholar,
          OpenAlex and Europe PMC by default); merged by work identity
seeds     1 call (only when following citations): the 1 to 3 best search results
          by title and abstract
follow    one hop from each seed, references and citers, one API call per list
          (``sources.expand_with``: Semantic Scholar by default), never deeper
rank      code orders every candidate (BM25 of title and abstract against the
          sentence, fused with the candidates' places in the source lists); 1
          call: the model orders the best ``RANK_SHOWN`` of them, best first
```

The result is every candidate: the model's order first, then the rest in the
code's order. No text is read: titles, abstracts and identifiers only. A failed
source is recorded in ``degraded`` (in the words ``search.search`` uses, so the
bench sees an outage) and the others go on. A model call that gives no usable
answer after its retries (no query, no seed, numbers that do not exist), fails
or meets the call budget raises ``FinderModelError`` with the log so far as its
``run_log``: there is no fallback, so a result always comes from the steps it
names. The ranking call may leave candidates out (an empty order is an answer).
Anything else raises.

The stage lists of ``FinderResult`` (``searched``, ``seeds``, ``followed``,
``shown``, ``chosen``) hold work ids of ``works``, so a lost gold work can be
traced to the step that lost it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry
from pydantic_ai.models import Model

from andamentum.references import Work

from . import fusion
from .ask import MODEL_ERRORS, UNTRUSTED, Asker, BudgetExhausted, fence
from .choose import site_of
from .errors import ResearchError
from .lexical import bm25_scores
from .queries import fresh_queries
from .runlog import RunLogger
from .search import admit, expand, search
from .sources import SourceClient, SourceConfig, build_sources
from .sources.base import ExpandingSource, Source
from .store import ResearchStore
from .types import Budget, Candidate, RunLog, RunStats

# The sources the finder searches, and those whose citation links it follows
# (Semantic Scholar answers a references or citations list in one call).
FINDER_SOURCES = SourceConfig(
    scholarly=("semantic_scholar", "openalex", "europepmc"),
    expand_with=("semantic_scholar",),
)
MIN_QUERIES = 2
MAX_QUERIES = 3
MAX_SEEDS = 3
# Search results shown to the seed choice, best code rank first.
SEEDS_SHOWN = 30
# Works asked of one references or citations call (one request each).
LINKS_PER_LIST = 100
# Candidates shown to the ranking call, best code rank first, and the most it
# may order.
RANK_SHOWN = 50
RANK_PICKS = 20
SNIPPET_CHARS = 300
# The tag of the finder's candidates (``search`` and ``expand`` tag each
# candidate with the sub-question it serves).
FINDER_TAG = "finder"
# Model calls the finder may make: queries, seeds, rank and their retries are
# one call each; the bound is generous and only stops a runaway.
FINDER_BUDGET = Budget(max_llm_calls=10)


class FinderModelError(ResearchError):
    """A finder model call gave no usable answer (``step`` names it); the
    finder has no fallback. ``run_log`` is the log up to the failure."""

    def __init__(self, step: str, error: Exception) -> None:
        super().__init__(f"{step} failed: {type(error).__name__}: {error}")
        self.step = step
        self.run_log: RunLog | None = None


class FinderQueries(BaseModel):
    queries: list[str] = Field(
        description="2 or 3 queries of 3 to 6 plain keywords each, no quotes or operators."
    )


class SeedChoice(BaseModel):
    seeds: list[int] = Field(description="1 to 3 result numbers, best first.")


class RankChoice(BaseModel):
    ranked: list[int] = Field(
        description=f"Up to {RANK_PICKS} result numbers, the most likely cited first."
    )


class FinderResult(BaseModel):
    """What ``find_works`` found: every candidate work, best first, and the
    work ids each step kept (ids of ``works``)."""

    sentence: str
    paragraph: str = ""
    follow_citations: bool
    queries: list[str] = Field(default_factory=list)
    works: list[Work] = Field(
        default_factory=list, description="Every candidate, best first."
    )
    searched: list[str] = Field(
        default_factory=list, description="Works the searches returned."
    )
    seeds: list[str] = Field(
        default_factory=list, description="Works whose citations were followed."
    )
    followed: list[str] = Field(
        default_factory=list,
        description="Works the citation step returned (some also found by search).",
    )
    shown: list[str] = Field(
        default_factory=list, description="Works shown to the ranking call, in order."
    )
    chosen: list[str] = Field(
        default_factory=list, description="The ranking call's order."
    )
    degraded: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    stats: RunStats = Field(default_factory=RunStats)
    log: RunLog | None = Field(default=None, exclude=True)


def _context(sentence: str, paragraph: str) -> list[str]:
    lines = [f"SENTENCE: {sentence.strip()}"]
    if paragraph.strip():
        lines += ["", "PARAGRAPH AROUND IT:", paragraph.strip()]
    return lines


QUERY_INSTRUCTIONS = f"""\
You write literature search queries. A writer needs the papers to cite for the \
SENTENCE (the PARAGRAPH, when given, is only context).
Write {MIN_QUERIES} or {MAX_QUERIES} queries for a scholarly search engine that \
matches words in titles and abstracts. Each query is 3 to 6 plain keywords: the \
method, task, object or finding the sentence is about, in the words a paper's \
title or abstract would use. No quotes, operators, years or author names unless \
the sentence names them. Each query takes a different angle (the exact technical \
terms, a broader name for the same thing, the specific claim)."""

SEED_INSTRUCTIONS = f"""\
You pick the search results whose reference lists and citing papers are most \
likely to hold the papers the SENTENCE should cite. You see numbered results: \
title, venue, year and the start of the abstract.
Return `seeds`: 1 to {MAX_SEEDS} result numbers, best first; the papers closest to \
what the sentence claims or describes. Numbers only.
{UNTRUSTED}"""

RANK_INSTRUCTIONS = f"""\
You rank candidate papers for a citation. You see the SENTENCE that needs a \
citation and numbered candidates: title, venue, year and the start of the \
abstract.
Return `ranked`: up to {RANK_PICKS} candidate numbers, the paper the sentence most \
likely cites first. A paper that does what the sentence describes, or reports \
what it claims, comes before one that is only on the same topic. Leave out \
candidates that do not fit. Numbers only, each once.
{UNTRUSTED}"""


def _listing(shown: Sequence[Candidate]) -> str:
    lines: list[str] = []
    for number, candidate in enumerate(shown, start=1):
        work = candidate.work
        year = f", {work.year}" if work.year else ""
        lines.append(
            f"[{number}] {' '.join(work.title.split())} ({site_of(candidate)}{year})"
        )
        snippet = " ".join((work.abstract or candidate.snippet).split())
        if snippet:
            lines.append(f"    {snippet[:SNIPPET_CHARS]}")
    return "\n".join(lines)


def code_order(candidates: Mapping[str, Candidate], *, sentence: str) -> list[str]:
    """Work ids best first by reciprocal rank fusion of two lists: BM25 of each
    work's title and abstract against *sentence*, and the candidate's own fused
    place in the source lists it came from. A tie keeps the order of
    *candidates* (a work id is random per store, so it must not decide)."""
    ids = list(candidates)
    if not ids:
        return []
    texts = [
        f"{candidates[w].work.title} {candidates[w].work.abstract or ''}" for w in ids
    ]
    lexical = dict(zip(ids, bm25_scores(sentence, texts), strict=True))
    fused = fusion.rrf_scores([lexical, fusion.fused_scores(candidates)])
    return sorted(ids, key=lambda w: -fused[w])


async def _some_query(draft: FinderQueries) -> FinderQueries:
    # async on purpose: pydantic-ai runs a sync validator in a worker thread
    if not fresh_queries(draft.queries, done=[], count=MAX_QUERIES):
        raise ModelRetry(
            f"write {MIN_QUERIES} or {MAX_QUERIES} queries of plain keywords; "
            "none was given"
        )
    return draft


async def _some_seed(draft: SeedChoice) -> SeedChoice:
    if not draft.seeds:
        raise ModelRetry(f"name 1 to {MAX_SEEDS} result numbers; none was given")
    return draft


def _numbers_exist(count: int) -> Callable[..., object]:
    # async on purpose: pydantic-ai runs a sync validator in a worker thread
    async def check(draft: BaseModel) -> BaseModel:
        numbers = getattr(draft, "seeds", None) or getattr(draft, "ranked", None) or []
        wrong = sorted({n for n in numbers if not 1 <= n <= count})
        if wrong:
            raise ModelRetry(f"there are no results numbered {wrong}; use 1 to {count}")
        return draft

    return check


def _picked(numbers: Sequence[int], shown: Sequence[str], *, most: int) -> list[str]:
    return [shown[n - 1] for n in dict.fromkeys(numbers) if 1 <= n <= len(shown)][:most]


def _failed(step: str, error: Exception, *, log: RunLogger) -> FinderModelError:
    log.event(
        "failed",
        f"{step} failed: {type(error).__name__}: {error}",
        step="Finish",
        error=f"{type(error).__name__}: {error}",
    )
    failure = FinderModelError(step, error)
    failure.run_log = log.snapshot()
    return failure


async def find_works(
    sentence: str,
    *,
    model: str | Model,
    paragraph: str | None = None,
    follow_citations: bool = True,
    sources: SourceConfig = FINDER_SOURCES,
    store: ResearchStore | None = None,
    http_cache: ResearchStore | None = None,
    source_client: SourceClient | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FinderResult:
    """The works *sentence* could cite, best first (see the module docstring).

    ``sources.scholarly`` names the sources searched and ``sources.expand_with``
    those whose citation links are followed (``None``: every one that has them);
    ``follow_citations=False`` skips the seed choice and the citation hop.
    ``store`` is an open research store (its HTTP cache answers repeated
    requests; ``None``: an in-memory one for this call, reading through
    ``http_cache`` when given). ``source_client`` must belong to ``store``.
    """
    if not sentence.strip():
        raise ValueError("find_works needs a sentence")
    if source_client is not None and source_client.store is not store:
        raise ValueError("source_client must belong to the open store passed as store")
    if http_cache is not None and store is not None:
        raise ValueError(
            "http_cache goes to a store this call opens: open the store with "
            "shared_cache= instead"
        )
    opened = store or ResearchStore.in_memory(shared_cache=http_cache)
    client = source_client or SourceClient(opened)
    try:
        source_set = build_sources(
            sources, store=opened, mode="scholarly", client=client
        )
        return await find_with(
            sentence,
            paragraph=paragraph or "",
            model=model,
            follow_citations=follow_citations,
            search_sources=source_set.search,
            expand_sources=source_set.expand,
            store=opened,
            results_per_query=sources.results_per_query,
            clock=clock,
        )
    finally:
        if source_client is None:
            await client.aclose()
        if store is None:
            opened.close()


async def find_with(
    sentence: str,
    *,
    paragraph: str,
    model: str | Model,
    follow_citations: bool,
    search_sources: Sequence[Source],
    expand_sources: Sequence[ExpandingSource],
    store: ResearchStore,
    results_per_query: int,
    clock: Callable[[], float] = time.monotonic,
) -> FinderResult:
    """``find_works`` over the given adapters (the tests pass stand-ins)."""
    run_id = store.start_run(
        question=sentence,
        mode="scholarly",
        config={"finder": {"follow_citations": follow_citations}},
        models={"model": model if isinstance(model, str) else repr(model)},
    )
    log = RunLogger(clock=clock, run_id=run_id)
    stats = RunStats()
    asker = Asker(
        model=model,
        store=store,
        run_id=run_id,
        budget=FINDER_BUDGET,
        stats=stats,
        log=log,
    )
    started = clock()
    result = FinderResult(
        sentence=sentence, paragraph=paragraph, follow_citations=follow_citations
    )
    context = _context(sentence, paragraph)

    # queries
    try:
        draft = await asker.llm(
            step="Search",
            instructions=QUERY_INSTRUCTIONS,
            output_type=FinderQueries,
            user_message="\n".join(context),
            validators=[_some_query],
        )
    except (BudgetExhausted, *MODEL_ERRORS) as error:
        raise _failed("query writing", error, log=log) from error
    # repeats (case and spacing folded) dropped, at most MAX_QUERIES; the
    # validator made sure one is left
    queries = fresh_queries(draft.queries, done=[], count=MAX_QUERIES)
    result.queries = queries
    log.event("queries", f"{len(queries)} queries", step="Search", queries=queries)

    # search
    found = await search(
        queries,
        subquestion_id=FINDER_TAG,
        sources=search_sources,
        filters=None,
        limit=results_per_query,
    )
    result.degraded += found.degraded
    result.notes += found.notes + found.skipped
    candidates = admit(found.candidates, store=store, known={}).candidates
    result.searched = list(candidates)
    log.event(
        "search_results",
        f"{len(candidates)} works from search",
        step="Search",
        hits=found.hits,
    )

    # seeds and the citation hop
    if follow_citations and candidates:
        shown = code_order(candidates, sentence=sentence)[:SEEDS_SHOWN]
        try:
            choice = await asker.llm(
                step="Expand",
                instructions=SEED_INSTRUCTIONS,
                output_type=SeedChoice,
                user_message="\n".join(
                    [
                        *context,
                        "",
                        "RESULTS:",
                        fence(_listing([candidates[w] for w in shown])),
                    ]
                ),
                validators=[_some_seed, _numbers_exist(len(shown))],
            )
        except (BudgetExhausted, *MODEL_ERRORS) as error:
            raise _failed("choosing the seeds", error, log=log) from error
        result.seeds = _picked(choice.seeds, shown, most=MAX_SEEDS)
        hop = await expand(
            [candidates[w].work for w in result.seeds],
            subquestion_id=FINDER_TAG,
            sources=expand_sources,
            limit=LINKS_PER_LIST * 2 * len(result.seeds) * max(1, len(expand_sources)),
            per_call=LINKS_PER_LIST,
        )
        result.degraded += [f"citation following failed: {m}" for m in hop.degraded]
        result.notes += hop.notes + hop.skipped
        # stored ids of the hop's works, in the hop's order (upserting twice is a no-op)
        result.followed = list(admit(hop.candidates, store=store, known={}).candidates)
        candidates = admit(hop.candidates, store=store, known=candidates).candidates
        log.event(
            "expanded",
            f"{len(result.followed)} works from {len(result.seeds)} seeds",
            step="Expand",
            seeds=[candidates[w].work.title for w in result.seeds],
        )

    # rank
    order = code_order(candidates, sentence=sentence)
    shown = order[:RANK_SHOWN]
    result.shown = shown
    chosen: list[str] = []
    if shown:
        try:
            ranking = await asker.llm(
                step="Choose",
                instructions=RANK_INSTRUCTIONS,
                output_type=RankChoice,
                user_message="\n".join(
                    [
                        *context,
                        "",
                        "CANDIDATES:",
                        fence(_listing([candidates[w] for w in shown])),
                    ]
                ),
                validators=[_numbers_exist(len(shown))],
            )
        except (BudgetExhausted, *MODEL_ERRORS) as error:
            raise _failed("ranking", error, log=log) from error
        chosen = _picked(ranking.ranked, shown, most=RANK_PICKS)
    result.chosen = chosen
    final = [*chosen, *(w for w in order if w not in set(chosen))]
    result.works = [candidates[w].work for w in final]
    log.event(
        "final_ranking",
        f"{len(final)} works ranked",
        step="Finish",
        top=[candidates[w].work.title for w in final[:RANK_PICKS]],
    )
    stats.wall_s = clock() - started
    result.stats = stats
    result.log = log.snapshot()
    return result
