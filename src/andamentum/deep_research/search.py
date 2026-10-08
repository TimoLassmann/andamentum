"""SEARCH and EXPAND: the sources' side of one sub-question's round.

``search`` runs a sub-question's queries on every source of the mode, one call
after another (pacing is per source), tags each result with the sub-question and
its place in the source's list (``source_ranks``, 1-based) and merges the
results by work identity. ``admit`` stores the works and combines them with
what the sub-question already holds. ``expand`` follows citation links one hop
from the best works already read.

Failure rules: a source that fails raises ``SourceError``; the step records why
(``"<source> search failed for <query!r>: <reason>"``: the bench detects outages
by that text), marks the run degraded and goes on with the other sources. A
dead source never reads as "nothing found".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from andamentum.references import Work

from .identity import merge, merge_pair
from .origin import registrable_domain
from .sources.base import (
    ExpandingSource,
    SearchFilters,
    Source,
    SourceError,
    WarningSource,
)
from .store import ResearchStore
from .types import Candidate

# Kept works whose references and citations are followed, per round, best first.
MAX_EXPAND_SEEDS = 3
# Results asked of one references or citations call.
EXPAND_PER_CALL = 20
# Titles kept per query and source for the run log.
HIT_TITLES = 5


def work_url(work: Work) -> str:
    """The address to show for a work: its URL, else its DOI link."""
    if work.url:
        return work.url
    return f"https://doi.org/{work.doi}" if work.doi else ""


def result_row(rank: int, candidate: Candidate) -> dict[str, object]:
    """One search result as the run log keeps it (the snippet whole, unless the
    work's licence forbids exporting it)."""
    url = work_url(candidate.work)
    return {
        "rank": rank,
        "url": url,
        "title": candidate.work.title,
        "host": registrable_domain(url),
        "snippet": candidate.snippet if candidate.work.redistributable else "",
    }


@dataclass
class SearchOutcome:
    candidates: list[Candidate]
    degraded: list[str] = field(default_factory=list)
    queries_run: list[str] = field(default_factory=list)
    # A partial answer: results, but some engines down.
    notes: list[str] = field(default_factory=list)
    # Per query and source, for the run log: the result count (or the error) and
    # the first ``HIT_TITLES`` titles with their place.
    hits: list[dict[str, object]] = field(default_factory=list)
    # Per query and source: every result (``result_row``), for the run log.
    results: list[dict[str, object]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


async def search(
    queries: Sequence[str],
    *,
    subquestion_id: str,
    sources: Sequence[Source],
    filters: SearchFilters | None,
    limit: int,
) -> SearchOutcome:
    """Every query on every source, one call after another; results tagged
    with *subquestion_id* and their position in the source's list, and merged
    by identity. A failing source is recorded and the others go on."""
    found: list[Candidate] = []
    outcome = SearchOutcome([])
    for query in queries:
        outcome.queries_run.append(query)
        for source in sources:
            try:
                results = await source.search(query, limit=limit, filters=filters)
            except SourceError as error:
                line = f"{source.name} search failed for {query!r}: {error.reason}"
                outcome.hits.append(
                    {
                        "subquestion": subquestion_id,
                        "query": query,
                        "source": source.name,
                        "count": 0,
                        "top": [],
                        "error": error.reason,
                    }
                )
                if error.kind == "offline":
                    outcome.skipped.append(line)
                    continue
                outcome.degraded.append(line)
                continue
            outcome.hits.append(
                {
                    "subquestion": subquestion_id,
                    "query": query,
                    "source": source.name,
                    "count": len(results),
                    "top": [
                        {"rank": place, "title": c.work.title}
                        for place, c in enumerate(results[:HIT_TITLES], start=1)
                    ],
                    "error": "",
                }
            )
            outcome.results.append(
                {
                    "subquestion": subquestion_id,
                    "query": query,
                    "source": source.name,
                    "results": [
                        result_row(place, c) for place, c in enumerate(results, start=1)
                    ],
                }
            )
            if isinstance(source, WarningSource):
                outcome.notes += source.take_warnings()
            found += [
                c.model_copy(
                    update={
                        "subquestion_ids": [subquestion_id],
                        "source_ranks": {source.name: position},
                    }
                )
                for position, c in enumerate(results, start=1)
            ]
    outcome.candidates = merge(found)
    return outcome


@dataclass
class Admitted:
    """The sub-question's candidates by stored work id after admission, and how
    many works were new to it."""

    candidates: dict[str, Candidate]
    new_works: int


def admit(
    found: Sequence[Candidate],
    *,
    store: ResearchStore,
    known: Mapping[str, Candidate],
) -> Admitted:
    """Store each candidate's work (merged with what the store holds) and
    combine it with the candidate the sub-question already has for that work."""
    candidates = dict(known)
    for candidate in found:
        stored = store.upsert_work(candidate.work)
        fresh = candidate.model_copy(update={"work": stored})
        before = candidates.get(stored.work_id)
        combined = merge_pair(before, fresh) if before is not None else fresh
        candidates[stored.work_id] = combined.model_copy(update={"work": stored})
    return Admitted(candidates, len(candidates) - len(known))


# -- expand -------------------------------------------------------------------


def is_scholarly(work: Work) -> bool:
    """A work a citation graph can place: it holds a scholarly identifier (a
    record of a local corpus counts, its corpus holds the links)."""
    return any(
        (
            work.doi,
            work.arxiv_id,
            work.pmid,
            work.pmcid,
            work.openalex_id,
            work.s2_id,
            work.local_id,
        )
    )


def expansion_seeds(
    scores: Mapping[str, float],
    *,
    candidates: Mapping[str, Candidate],
    expanded: set[str],
    limit: int = MAX_EXPAND_SEEDS,
) -> list[Work]:
    """Scholarly works not yet expanded, best score first (a tie keeps the order
    of *scores*: a work id is random per store), at most *limit*. *scores* maps a work id to how good a seed it is (the engine
    gives works that were read and gave findings)."""
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    seeds: list[Work] = []
    for work_id, _ in ordered:
        candidate = candidates.get(work_id)
        if candidate is None or work_id in expanded or not is_scholarly(candidate.work):
            continue
        seeds.append(candidate.work)
        if len(seeds) >= limit:
            break
    return seeds


@dataclass
class ExpandOutcome:
    candidates: list[Candidate] = field(default_factory=list)
    expanded: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


async def expand(
    seeds: Sequence[Work],
    *,
    subquestion_id: str,
    sources: Sequence[ExpandingSource],
    limit: int,
    per_call: int = EXPAND_PER_CALL,
) -> ExpandOutcome:
    """One hop back (references) and forward (citations) from each seed, until
    *limit* candidates are collected, at most *per_call* from one references or
    citations call; each new candidate is tagged with
    *subquestion_id* and its position in the hop's list (``source_ranks``). A
    source that cannot place a work is a note; a hop an offline run does not
    hold is ``skipped``; any other failure is degraded."""
    outcome = ExpandOutcome()
    for work in seeds:
        for source in sources:
            for direction in ("references", "citations"):
                room = limit - len(outcome.candidates)
                if room <= 0:
                    return outcome
                follow = (
                    source.references if direction == "references" else source.citations
                )
                try:
                    found = await follow(work, limit=min(room, per_call))
                except SourceError as error:
                    message = (
                        f"{source.name} {direction} of "
                        f"{work.title or work.work_id}: {error.reason}"
                    )
                    if error.kind == "config":
                        outcome.notes.append(message)
                    elif error.kind == "offline":
                        outcome.skipped.append(message)
                    else:
                        outcome.degraded.append(message)
                    continue
                outcome.candidates += [
                    c.model_copy(
                        update={
                            "subquestion_ids": [subquestion_id],
                            "sources": [f"{source.name}:{direction}"],
                            "source_ranks": {f"{source.name}:{direction}": position},
                        }
                    )
                    for position, c in enumerate(found[:room], start=1)
                ]
        outcome.expanded.append(work.work_id)
    return outcome
