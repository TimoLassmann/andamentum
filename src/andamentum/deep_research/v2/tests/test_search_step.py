"""SEARCH, ADMIT and EXPAND: the sources' side of one sub-question's round."""

from __future__ import annotations

from collections.abc import Sequence

from andamentum.references import LICENCE_NO_REDISTRIBUTE, Work

from ..search import (
    MAX_EXPAND_SEEDS,
    admit,
    expand,
    expansion_seeds,
    is_scholarly,
    result_row,
    search,
)
from ..sources.base import ErrorKind, SearchFilters, SourceError
from ..store import ResearchStore
from ..types import Candidate


def cand(
    title: str, *, doi: str | None = None, url: str | None = None, snippet: str = ""
) -> Candidate:
    return Candidate(
        work=Work(title=title, doi=doi, url=url),
        sources=["stub"],
        snippet=snippet,
    )


class StubSource:
    """A search source answering from a script: query -> results or an error."""

    def __init__(
        self, name: str, script: dict[str, list[Candidate] | SourceError]
    ) -> None:
        self.name = name
        self.script = script
        self.asked: list[tuple[str, SearchFilters | None]] = []
        self.warnings: list[str] = []

    async def search(
        self, query: str, *, limit: int = 20, filters: SearchFilters | None = None
    ) -> list[Candidate]:
        self.asked.append((query, filters))
        answer = self.script[query]
        if isinstance(answer, SourceError):
            raise answer
        return answer[:limit]

    def take_warnings(self) -> list[str]:
        taken, self.warnings = self.warnings, []
        return taken


async def test_every_query_runs_on_every_source_in_order() -> None:
    web = StubSource(
        "web", {"a b": [cand("One", url="https://one.example/x")], "c d": []}
    )
    papers = StubSource(
        "openalex", {"a b": [], "c d": [cand("Two", doi="10.5555/two")]}
    )
    filters = SearchFilters(year_from=2020)

    outcome = await search(
        ["a b", "c d"],
        subquestion_id="Q1",
        sources=[web, papers],
        filters=filters,
        limit=7,
    )

    assert outcome.queries_run == ["a b", "c d"]
    assert [(h["query"], h["source"]) for h in outcome.hits] == [
        ("a b", "web"),
        ("a b", "openalex"),
        ("c d", "web"),
        ("c d", "openalex"),
    ]
    assert web.asked == [("a b", filters), ("c d", filters)]
    assert [c.work.title for c in outcome.candidates] == ["One", "Two"]


async def test_results_carry_the_subquestion_and_their_place_in_each_list() -> None:
    web = StubSource(
        "web",
        {
            "q": [
                cand("First", url="https://a.example/1"),
                cand("Second", url="https://b.example/2"),
            ]
        },
    )
    outcome = await search(
        ["q"], subquestion_id="Q2", sources=[web], filters=None, limit=20
    )
    first, second = outcome.candidates
    assert first.subquestion_ids == ["Q2"] and first.source_ranks == {"web": 1}
    assert second.source_ranks == {"web": 2}


async def test_a_work_found_twice_is_one_candidate_with_its_best_place_per_list() -> (
    None
):
    same = "10.5555/same"
    web = StubSource(
        "web",
        {"a": [cand("Other", url="https://o.example/")], "b": [cand("Same", doi=same)]},
    )
    papers = StubSource("openalex", {"a": [cand("Same", doi=same)], "b": []})
    outcome = await search(
        ["a", "b"], subquestion_id="Q1", sources=[web, papers], filters=None, limit=20
    )
    (merged,) = [c for c in outcome.candidates if c.work.doi == same]
    assert merged.source_ranks == {"openalex": 1, "web": 1}
    assert len(outcome.candidates) == 2


async def test_a_failing_source_is_recorded_with_the_exact_line_and_the_others_go_on() -> (
    None
):
    web = StubSource("web", {"q": SourceError("web", "HTTP 500", kind="http")})
    papers = StubSource("openalex", {"q": [cand("Two", doi="10.5555/two")]})
    outcome = await search(
        ["q"], subquestion_id="Q1", sources=[web, papers], filters=None, limit=20
    )

    assert outcome.degraded == ["web search failed for 'q': HTTP 500"]
    assert [c.work.title for c in outcome.candidates] == ["Two"]
    failed = outcome.hits[0]
    assert (
        failed["error"] == "HTTP 500" and failed["count"] == 0 and failed["top"] == []
    )
    assert outcome.hits[1]["error"] == ""


async def test_an_offline_miss_is_skipped_not_degraded() -> None:
    web = StubSource(
        "web", {"q": SourceError("web", "not in the snapshot", kind="offline")}
    )
    outcome = await search(
        ["q"], subquestion_id="Q1", sources=[web], filters=None, limit=20
    )
    assert outcome.degraded == []
    assert outcome.skipped == ["web search failed for 'q': not in the snapshot"]


async def test_hits_list_the_first_five_titles_and_the_results_whole() -> None:
    long_snippet = "word " * 120
    results = [
        cand(f"Title {n}", url=f"https://site{n}.example/p", snippet=long_snippet)
        for n in range(1, 8)
    ]
    web = StubSource("web", {"q": results})
    outcome = await search(
        ["q"], subquestion_id="Q1", sources=[web], filters=None, limit=20
    )

    (hit,) = outcome.hits
    assert hit["count"] == 7
    assert hit["top"] == [{"rank": n, "title": f"Title {n}"} for n in range(1, 6)]
    (listed,) = outcome.results
    rows = listed["results"]
    assert isinstance(rows, list) and len(rows) == 7
    assert rows[0] == {
        "rank": 1,
        "url": "https://site1.example/p",
        "title": "Title 1",
        "host": "site1.example",
        "snippet": long_snippet,
    }


async def test_a_sources_warnings_become_notes() -> None:
    web = StubSource("web", {"q": [cand("One", url="https://one.example/x")]})
    web.warnings = ["two engines did not answer"]
    outcome = await search(
        ["q"], subquestion_id="Q1", sources=[web], filters=None, limit=20
    )
    assert outcome.notes == ["two engines did not answer"]


def test_a_result_row_never_exports_a_snippet_the_licence_forbids() -> None:
    restricted = Candidate(
        work=Work(title="Closed", doi="10.5555/c", licence=LICENCE_NO_REDISTRIBUTE),
        sources=["s2"],
        snippet="the licensed first words",
    )
    row = result_row(1, restricted)
    assert row["snippet"] == "" and row["url"] == "https://doi.org/10.5555/c"


# -- admit ----------------------------------------------------------------------


def test_admit_stores_the_work_and_merges_with_what_the_subquestion_holds() -> None:
    store = ResearchStore.in_memory()
    first = admit([cand("Apollo paper", doi="10.5555/a")], store=store, known={})
    (work_id,) = first.candidates
    assert first.new_works == 1
    assert store.get_work(work_id) is not None
    assert first.candidates[work_id].work.work_id == work_id

    again = Candidate(
        work=Work(title="Apollo paper", doi="10.5555/a"),
        sources=["europepmc"],
        source_ranks={"europepmc": 3},
        queries=["q2"],
    )
    second = admit(
        [again, cand("New", doi="10.5555/n")], store=store, known=first.candidates
    )
    assert second.new_works == 1 and len(second.candidates) == 2
    merged = second.candidates[work_id]
    assert merged.sources == ["stub", "europepmc"] and merged.source_ranks == {
        "europepmc": 3
    }


# -- expand ---------------------------------------------------------------------


def test_scholarly_means_a_scholarly_identifier() -> None:
    assert is_scholarly(Work(title="p", doi="10.5555/x"))
    assert is_scholarly(Work(title="p", openalex_id="W1"))
    assert not is_scholarly(Work(title="p", url="https://one.example/x"))


def test_seeds_are_the_best_scholarly_works_not_yet_expanded() -> None:
    works = {
        "w1": Candidate(
            work=Work(work_id="w1", title="a", doi="10.5555/a"), sources=["s"]
        ),
        "w2": Candidate(
            work=Work(work_id="w2", title="b", doi="10.5555/b"), sources=["s"]
        ),
        "w3": Candidate(
            work=Work(work_id="w3", title="c", url="https://c.example/"), sources=["s"]
        ),
        "w4": Candidate(
            work=Work(work_id="w4", title="d", doi="10.5555/d"), sources=["s"]
        ),
        "w5": Candidate(
            work=Work(work_id="w5", title="e", doi="10.5555/e"), sources=["s"]
        ),
        "w6": Candidate(
            work=Work(work_id="w6", title="f", doi="10.5555/f"), sources=["s"]
        ),
    }
    scores = {
        "w1": 1.0,
        "w2": 3.0,
        "w3": 9.0,
        "w4": 3.0,
        "w5": 2.0,
        "w6": 0.5,
        "gone": 7.0,
    }
    seeds = expansion_seeds(scores, candidates=works, expanded={"w5"})
    assert MAX_EXPAND_SEEDS == 3
    # a web page and a work the run no longer holds are no seeds; a tie keeps the order of the scores
    assert [w.work_id for w in seeds] == ["w2", "w4", "w1"]
    assert [
        w.work_id
        for w in expansion_seeds(scores, candidates=works, expanded=set(), limit=1)
    ] == ["w2"]
    flipped = dict(reversed(list(scores.items())))
    assert [
        w.work_id
        for w in expansion_seeds(flipped, candidates=works, expanded=set(), limit=2)
    ] == ["w4", "w2"]


class StubExpander:
    def __init__(
        self,
        name: str,
        refs: Sequence[Candidate] = (),
        cites: Sequence[Candidate] = (),
        fail: SourceError | None = None,
    ) -> None:
        self.name = name
        self.refs, self.cites, self.fail = list(refs), list(cites), fail

    async def search(
        self, query: str, *, limit: int = 20, filters: object = None
    ) -> list[Candidate]:
        return []

    async def references(self, work: Work, *, limit: int = 20) -> list[Candidate]:
        if self.fail:
            raise self.fail
        return self.refs[:limit]

    async def citations(self, work: Work, *, limit: int = 20) -> list[Candidate]:
        return self.cites[:limit]


async def test_expansion_tags_the_hop_and_the_place_and_stops_at_the_limit() -> None:
    seed = Work(work_id="w1", title="Seed", doi="10.5555/seed")
    source = StubExpander(
        "openalex",
        refs=[cand("R1", doi="10.5555/r1"), cand("R2", doi="10.5555/r2")],
        cites=[cand("C1", doi="10.5555/c1"), cand("C2", doi="10.5555/c2")],
    )
    outcome = await expand([seed], subquestion_id="Q2", sources=[source], limit=3)  # type: ignore[list-item]

    assert [c.work.title for c in outcome.candidates] == ["R1", "R2", "C1"]
    first, _, third = outcome.candidates
    assert first.source_ranks == {"openalex:references": 1}
    assert first.sources == ["openalex:references"] and first.subquestion_ids == ["Q2"]
    assert third.source_ranks == {"openalex:citations": 1}
    assert outcome.expanded == ["w1"]  # its last hop was cut short, not skipped


async def test_a_seed_that_finished_is_marked_expanded() -> None:
    seed = Work(work_id="w1", title="Seed", doi="10.5555/seed")
    source = StubExpander("openalex", refs=[cand("R1", doi="10.5555/r1")])
    outcome = await expand([seed], subquestion_id="Q1", sources=[source], limit=10)  # type: ignore[list-item]
    assert outcome.expanded == ["w1"]


async def test_expansion_failures_are_sorted_by_kind() -> None:
    seed = Work(work_id="w1", title="Seed", doi="10.5555/seed")
    kinds: list[tuple[ErrorKind, str]] = [
        ("config", "notes"),
        ("offline", "skipped"),
        ("http", "degraded"),
    ]
    for kind, field in kinds:
        source = StubExpander(
            "openalex", fail=SourceError("openalex", "no way", kind=kind)
        )
        outcome = await expand([seed], subquestion_id="Q1", sources=[source], limit=10)  # type: ignore[list-item]
        assert getattr(outcome, field) == ["openalex references of Seed: no way"]
