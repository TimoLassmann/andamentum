"""Round 3, freezing and reading: plan reuse with no model call, search-only
snapshots, search accounting, tolerated failures of a patient source, the
candidate-pool tables, the loss table and the report sections. Offline, over
the invented world and hand-built records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.v2.bench import cli
from andamentum.deep_research.v2.bench.arms import ArmOutput
from andamentum.deep_research.v2.bench.experiment import Experiment
from andamentum.deep_research.v2.bench.experiment_report import build_report
from andamentum.deep_research.v2.bench.freeze import (
    FreezeItem,
    InputsSetup,
    freeze_items_of,
    load_frozen,
    snapshot_dir,
)
from andamentum.deep_research.v2.bench.litsearch import (
    LitSearchItem,
    LitSearchQuery,
)
from andamentum.deep_research.v2.bench.metrics import Cost
from andamentum.deep_research.v2.bench.pool import (
    FAILED,
    HIT,
    KEPT_EXCLUDED,
    KEPT_LOW,
    LOSS_CATEGORIES,
    NOT_IN_POOL,
    NOT_SCREENED,
    SCREENED_OUT,
    UNRECORDED,
    classify_gold,
    gold_entries,
    load_pool_items,
    loss_counts,
    loss_section,
    pool_comparison,
    pool_tables,
)
from andamentum.deep_research.v2.bench.registry import RunRecord, StartRecord
from andamentum.deep_research.v2.frozen import FrozenInputs, capture
from andamentum.deep_research.v2.sources import SourceClient, SourceConfig
from andamentum.deep_research.v2.sources.base import PATIENT_GIVE_UP_AFTER
from andamentum.deep_research.v2.store import ResearchStore
from andamentum.deep_research.v2.types import (
    Candidate,
    ScreeningEntry,
    SubQuestion,
)
from andamentum.references import Work

from .engine_world import QUESTION, World, world_decisions, world_llm
from .test_bench_experiment_run import make_result

OPENALEX = SourceConfig(scholarly=("openalex",))


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


async def freeze_litsearch(
    root: Path,
    world: World,
    *,
    setup: InputsSetup = InputsSetup(),
    model: Any = "world",
    decision: Any = "world",
    sources: SourceConfig = OPENALEX,
    items: list[FreezeItem] | None = None,
    client_factory: Any = None,
):
    return await freeze_items_of(
        "litsearch",
        items or [FreezeItem("L1", QUESTION)],
        snapshots=root,
        model=world_llm().model if model == "world" else model,
        decision_model=world_decisions()[0] if decision == "world" else decision,
        decision_label="stub",
        source_config=sources,
        progress=lambda _line: None,
        prefetch_top_k=8,
        health_client=None,
        client_factory=client_factory or world.client,
        clock=lambda: 1000.0,
        setup=setup,
    )


# -- capture with a plan -------------------------------------------------------------


async def first_capture(store: ResearchStore, world: World) -> FrozenInputs:
    return await capture(
        QUESTION,
        mode="scholarly",
        model=world_llm().model,
        store=store,
        decision_model=world_decisions()[0],
        sources=OPENALEX,
        source_client=world.client(store),
    )


async def test_capture_with_a_plan_takes_the_queries_and_calls_no_model() -> None:
    first = await first_capture(ResearchStore.in_memory(), World())
    llm = world_llm()
    decision, stub = world_decisions()
    store = ResearchStore.in_memory()
    world = World()
    second = await capture(
        QUESTION,
        mode="scholarly",
        model=llm.model,
        store=store,
        decision_model=decision,
        sources=OPENALEX,
        source_client=world.client(store),
        plan=first,
    )
    assert llm.calls == [] and stub.calls == []  # not one request of any kind
    assert second.subquestions == first.subquestions
    assert second.queries == first.queries
    assert second.candidates  # the same queries were searched
    assert second.degraded == [] and second.models["decision_model"] is None
    run = store.get_run(second.run_id)
    assert run is not None and run.config["plan_from"] == first.digest()
    # no model at all is needed either
    again = ResearchStore.in_memory()
    third = await capture(
        QUESTION,
        mode="scholarly",
        model=None,
        store=again,
        sources=OPENALEX,
        source_client=World().client(again),
        plan=first,
    )
    assert third.queries == first.queries


async def test_capture_without_a_plan_still_needs_a_model_and_a_plan_must_match() -> (
    None
):
    store = ResearchStore.in_memory()
    with pytest.raises(ValueError, match="plans with a model"):
        await capture(QUESTION, mode="scholarly", model=None, store=store)
    first = await first_capture(ResearchStore.in_memory(), World())
    with pytest.raises(ValueError, match="another question or mode"):
        await capture(
            "another question entirely",
            mode="scholarly",
            model=None,
            store=store,
            plan=first,
        )


# -- freezing a profile ---------------------------------------------------------------


async def test_a_profile_that_reuses_a_plan_freezes_with_no_model_and_says_so(
    tmp_path: Path,
) -> None:
    base_root = tmp_path / "snap" / "base"
    report = await freeze_litsearch(
        base_root,
        World(),
        setup=InputsSetup(profile="base", profile_hash="h-base", keyless=True),
    )
    assert not report.failed
    llm = world_llm()
    decision, stub = world_decisions()
    wide_root = tmp_path / "snap" / "wide"
    report = await freeze_litsearch(
        wide_root,
        World(),
        setup=InputsSetup(
            profile="wide",
            profile_hash="h-wide",
            plan=base_root,
            plan_profile="base",
            keyless=True,
        ),
        model=llm.model,
        decision=decision,
    )
    assert not report.failed, report.lines()
    assert llm.calls == [] and stub.calls == []
    base = load_frozen(snapshot_dir(base_root, "litsearch", "L1"))
    wide = load_frozen(snapshot_dir(wide_root, "litsearch", "L1"))
    assert (wide.subquestions, wide.queries) == (base.subquestions, base.queries)
    meta = json.loads(
        (snapshot_dir(wide_root, "litsearch", "L1") / "meta.json").read_text()
    )
    assert meta["inputs"] == {"profile": "wide", "hash": "h-wide"}
    assert meta["plan_from"] == {"profile": "base", "digest": base.digest()}
    assert meta["keyless"] is True and meta["search_only"] is False
    assert meta["decision_model"] is None
    base_meta = json.loads(
        (snapshot_dir(base_root, "litsearch", "L1") / "meta.json").read_text()
    )
    assert base_meta["plan_from"] is None and base_meta["decision_model"] == "stub"


async def test_a_missing_plan_snapshot_fails_the_item_without_a_snapshot(
    tmp_path: Path,
) -> None:
    report = await freeze_litsearch(
        tmp_path / "wide",
        World(),
        setup=InputsSetup(plan=tmp_path / "nowhere", plan_profile="base"),
        model=None,
        decision=None,
    )
    [item] = report.items
    assert (
        item.status == "failed" and "no snapshot to take the plan from" in item.detail
    )
    assert not snapshot_dir(tmp_path / "wide", "litsearch", "L1").exists()


async def test_a_plan_directory_in_the_round_two_layout_is_read_as_it_is(
    tmp_path: Path,
) -> None:
    older = tmp_path / "old"
    await freeze_litsearch(older, World())  # <old>/litsearch/L1/, no profile folder
    report = await freeze_litsearch(
        tmp_path / "new",
        World(),
        setup=InputsSetup(plan=older, search_only=True),
        model=None,
        decision=None,
    )
    assert not report.failed


async def test_a_search_only_snapshot_stops_after_the_search_and_says_so(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    await freeze_litsearch(base, World())
    world = World()
    report = await freeze_litsearch(
        tmp_path / "only",
        world,
        setup=InputsSetup(plan=base, search_only=True),
        model=None,
        decision=None,
    )
    assert not report.failed
    hosts = {r.url.host for r in world.requests}
    assert "api.crossref.org" not in hosts  # no retraction check
    only = snapshot_dir(tmp_path / "only", "litsearch", "L1")
    meta = json.loads((only / "meta.json").read_text())
    assert meta["search_only"] is True
    assert meta["prefetch"]["attempted"] == 0 and meta["prefetched_works"] == []
    assert meta["citations"]["works"] == 0
    assert "search only" in report.items[0].detail
    # the search is there, the candidates too
    assert load_frozen(only).candidates
    # a search-only snapshot is not rewarmed
    from andamentum.deep_research.v2.bench.freeze import rewarm_items_of

    rewarm = await rewarm_items_of(
        "litsearch",
        [FreezeItem("L1", QUESTION)],
        snapshots=tmp_path / "only",
        model=None,
        source_config=OPENALEX,
        progress=lambda _line: None,
        prefetch_top_k=8,
        concurrency=1,
        client_factory=world.client,
    )
    assert (
        rewarm.items[0].status == "failed" and "search-only" in rewarm.items[0].detail
    )


async def test_the_search_accounting_is_in_the_snapshot_and_its_meta(
    tmp_path: Path,
) -> None:
    await freeze_litsearch(tmp_path, World())
    folder = snapshot_dir(tmp_path, "litsearch", "L1")
    frozen = load_frozen(folder)
    n_queries = len(frozen.all_queries())
    assert frozen.searches == {"openalex": n_queries}
    assert frozen.search_failures == {}
    assert set(frozen.result_counts["openalex"]) == set(frozen.all_queries())
    assert all(0 <= n <= 20 for n in frozen.result_counts["openalex"].values())
    assert frozen.search_s >= 0.0
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["search"]["searches"] == {"openalex": n_queries}
    assert meta["search"]["results_per_query"] == frozen.result_counts
    assert meta["sources"]["scholarly"] == ["openalex"]
    assert meta["sources"]["arxiv_query"] == "all_terms"
    assert meta["sources"]["expand_with"] is None
    # the digest names the inputs, not how the capture went
    assert (
        frozen.model_copy(update={"search_s": 99.0, "searches": {}}).digest()
        == frozen.digest()
    )


async def test_old_snapshots_keep_their_digest(tmp_path: Path) -> None:
    """A snapshot written before the accounting existed has none of its fields; its
    digest is that of the inputs alone."""
    frozen = FrozenInputs(
        question=QUESTION,
        mode="scholarly",
        captured_at=1.0,
        subquestions=[SubQuestion(id="Q1", text="one")],
        queries={"Q1": ["a query"]},
    )
    data = json.loads(frozen.model_dump_json())
    for name in (
        "searches",
        "search_failures",
        "result_counts",
        "search_s",
        "tolerated",
    ):
        del data[name]
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(data))
    loaded = FrozenInputs.load(path)
    assert loaded.searches == {} and loaded.digest() == frozen.digest()


# -- a patient source that does not answer ---------------------------------------------


class Limited:
    """A world where Semantic Scholar answers 429 to everything."""

    def __init__(self, world: World, *, openalex_status: int | None = None) -> None:
        self.world = world
        self.openalex_status = openalex_status
        self.s2_requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.semanticscholar.org":
            self.s2_requests += 1
            return httpx.Response(429, content=b"slow down")
        if request.url.host == "api.openalex.org" and self.openalex_status:
            return httpx.Response(self.openalex_status, content=b"broken")
        return self.world.handler(request)

    def factory(self, store: ResearchStore) -> SourceClient:
        async def no_sleep(_s: float) -> None:
            return None

        async def plain(body: bytes, _t: str, _u: str) -> str:
            return body.decode("utf-8")

        return SourceClient(
            store,
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
            intervals={
                n: 0.0
                for n in (
                    "openalex",
                    "crossref",
                    "europepmc",
                    "arxiv",
                    "semantic_scholar",
                    "web",
                )
            },
            sleep=no_sleep,
            extractor=plain,
            patient=("semantic_scholar",),
        )


async def test_failed_searches_of_a_patient_source_are_counted_not_fatal(
    tmp_path: Path,
) -> None:
    limited = Limited(World())
    sources = SourceConfig(
        scholarly=("openalex", "semantic_scholar"), expand_with=("openalex",)
    )
    report = await freeze_litsearch(
        tmp_path,
        World(),
        sources=sources,
        setup=InputsSetup(profile="combined", patient=("semantic_scholar",)),
        client_factory=limited.factory,
    )
    assert not report.failed, report.lines()
    folder = snapshot_dir(tmp_path, "litsearch", "L1")
    frozen = load_frozen(folder)
    n = len(frozen.all_queries())
    assert frozen.searches == {"openalex": n, "semantic_scholar": n}
    assert frozen.search_failures == {"semantic_scholar": n}  # every one failed
    assert frozen.degraded == []  # the snapshot is usable
    assert len(frozen.tolerated) == n and "semantic_scholar" in frozen.tolerated[0]
    # the dead pool was given up on after three requests that ran out of retries
    assert limited.s2_requests <= PATIENT_GIVE_UP_AFTER * 9
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["patient"] == ["semantic_scholar"]
    assert meta["search"]["search_failures"] == {"semantic_scholar": n}
    assert len(meta["search"]["tolerated_failures"]) == n
    # and the rule that judges the pool's health sees the share
    from andamentum.deep_research.v2.bench.experiment_run import failed_search_share

    assert failed_search_share(frozen) == pytest.approx(0.5)


async def test_a_failing_source_that_is_not_patient_still_ruins_the_snapshot(
    tmp_path: Path,
) -> None:
    limited = Limited(World(), openalex_status=500)
    report = await freeze_litsearch(
        tmp_path,
        World(),
        sources=SourceConfig(scholarly=("openalex",)),
        setup=InputsSetup(patient=("semantic_scholar",)),
        client_factory=limited.factory,
    )
    [item] = report.items
    assert item.status == "failed" and "degraded" in item.detail
    assert not snapshot_dir(tmp_path, "litsearch", "L1").exists()


# -- pools --------------------------------------------------------------------------


def gold_work(number: int) -> Work:
    return Work(
        title=f"An invented paper about topic {number} in kilns",
        doi=f"10.5555/gold.{number}",
        year=2021,
    )


def pool_candidate(number: int, sources: list[str], ranks: dict[str, int]) -> Candidate:
    return Candidate(
        work=Work(
            work_id=f"w{number}",
            title=f"An invented paper about topic {number} in kilns",
            doi=f"10.5555/gold.{number}",
            year=2021,
        ),
        sources=sources,
        source_ranks=ranks,
        subquestion_ids=["Q1"],
    )


def other_candidate(number: int, sources: list[str]) -> Candidate:
    return Candidate(
        work=Work(
            work_id=f"x{number}",
            title=f"Unrelated study {number}",
            url=f"https://example.org/{number}",
        ),
        sources=sources,
        source_ranks={sources[0]: number},
        subquestion_ids=["Q1"],
    )


def write_snapshot(
    root: Path,
    item_id: str,
    candidates: list[Candidate],
    *,
    result_counts: dict[str, dict[str, int]] | None = None,
    set_name: str = "litsearch",
) -> None:
    frozen = FrozenInputs(
        question=f"Which paper studies {item_id}?",
        mode="scholarly",
        captured_at=1.0,
        subquestions=[SubQuestion(id="Q1", text="one")],
        queries={"Q1": ["q one", "q two"]},
        candidates=candidates,
        result_counts=result_counts or {},
        searches={"arxiv": 2, "europepmc": 2},
        search_failures={},
        search_s=3.5,
    )
    folder = snapshot_dir(root, set_name, item_id)
    frozen.save(folder / "frozen.json")
    (folder / "meta.json").write_text(
        json.dumps(
            {
                "search": {
                    "searches": {"arxiv": 2, "europepmc": 2},
                    "search_failures": {},
                    "search_s": 3.5,
                    "tolerated_failures": [],
                },
                "http_cache": {"network_requests": {"arxiv": 5, "europepmc": 9}},
            }
        )
    )


def litsearch_item(item_id: str, golds: list[Work]) -> LitSearchItem:
    return LitSearchItem(
        query=LitSearchQuery(
            query_id=item_id,
            query=f"Which paper studies {item_id}?",
            query_set="inline_acl",
            specificity=1,
            quality=2,
            corpus_ids=list(range(len(golds))) or [0],
        ),
        gold=golds,
    )


def two_item_snapshots(root: Path) -> list[LitSearchItem]:
    write_snapshot(
        root,
        "LS0001",
        [
            pool_candidate(1, ["arxiv"], {"arxiv": 3}),
            other_candidate(1, ["europepmc"]),
            other_candidate(2, ["europepmc"]),
        ],
        result_counts={
            "arxiv": {"q one": 20, "q two": 4},
            "europepmc": {"q one": 20, "q two": 20},
        },
    )
    write_snapshot(
        root,
        "LS0002",
        [
            pool_candidate(3, ["arxiv", "europepmc"], {"arxiv": 25, "europepmc": 30}),
            other_candidate(3, ["arxiv"]),
        ],
        result_counts={
            "arxiv": {"q one": 0, "q two": 20},
            "europepmc": {"q one": 20, "q two": 20},
        },
    )
    return [
        litsearch_item("LS0001", [gold_work(1), gold_work(2)]),
        litsearch_item("LS0002", [gold_work(3)]),
    ]


def test_the_pool_tables_say_what_is_in_the_pool_and_who_found_it(
    tmp_path: Path,
) -> None:
    items = two_item_snapshots(tmp_path)
    pool, missing = load_pool_items(
        tmp_path, [*items, litsearch_item("LS0009", [gold_work(9)])]
    )
    assert missing == ["LS0009"] and [p.item_id for p in pool] == ["LS0001", "LS0002"]
    entries = [e for p in pool for e in gold_entries(p)]
    assert [(e.in_pool, e.in_search_top) for e in entries] == [
        (True, True),  # gold 1: arXiv rank 3
        (False, False),  # gold 2 is nowhere
        (True, False),  # gold 3: in the pool at ranks 25 and 30, never in a top 20
    ]
    assert entries[0].sources == frozenset({"arxiv"})
    assert entries[2].sources == frozenset({"arxiv", "europepmc"})
    text = pool_tables(pool, missing=missing)
    assert "2 items, 3 gold papers (1 item(s) without a snapshot: LS0009)" in text
    assert "| in the candidate pool | 2 (67%) |" in text
    assert "| in the top 20 of some query's list | 1 (33%) |" in text
    assert "| not in the pool | 1 (33%) |" in text
    assert "| arxiv | 2 | 1 |" in text  # found gold 1 and 3; alone: gold 1
    assert "| europepmc | 1 | 0 |" in text
    # pool sizes: arXiv holds 1 of 3 candidates in LS0001 and 2 of 2 in LS0002
    assert "| arxiv | 1.5 | 1.5 | 2 |" in text
    assert "Queries answered with fewer than 20 results" in text
    assert "| arxiv | 4 | 2 (50%) | 1 | 12.0 |" in text
    assert "| europepmc | 4 | 0 (0%) | 0 | 20.0 |" in text


def test_pool_tables_of_old_snapshots_say_the_accounting_is_absent(
    tmp_path: Path,
) -> None:
    items = [litsearch_item("LS0001", [gold_work(1)])]
    write_snapshot(tmp_path, "LS0001", [pool_candidate(1, ["arxiv"], {"arxiv": 1})])
    pool, _ = load_pool_items(tmp_path, items)
    assert "predate the search accounting" in pool_tables(pool)
    assert pool_tables([]) == "_No snapshot to read._"


def test_a_comparison_names_the_gold_papers_that_entered_and_left(
    tmp_path: Path,
) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    items = [litsearch_item("LS0001", [gold_work(1), gold_work(2)])]
    write_snapshot(
        older, "LS0001", [pool_candidate(1, ["europepmc"], {"europepmc": 1})]
    )
    write_snapshot(
        newer,
        "LS0001",
        [
            pool_candidate(2, ["semantic_scholar"], {"semantic_scholar": 2}),
            other_candidate(1, ["arxiv"]),
        ],
    )
    now, _ = load_pool_items(newer, items)
    before, _ = load_pool_items(older, items)
    text = pool_comparison(now, before, this_label="newer", against_label="older")
    assert "in the pool of older 1, of newer 1; 1 entered and 1 left" in text
    assert "1 semantic_scholar" in text
    assert "- LS0001: An invented paper about topic 2 in kilns" in text
    assert "Left the pool (older held them)" in text
    assert pool_comparison(now, [], this_label="n", against_label="o").startswith(
        "_No item"
    )


def test_the_pool_command_prints_the_tables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from .test_round3_bench import litsearch_dir

    data = litsearch_dir(tmp_path / "data", n=4)
    snapshots = tmp_path / "snap"
    from andamentum.deep_research.v2.bench.litsearch import load_litsearch

    loaded = load_litsearch(data, n=4, seed=0)
    first = loaded.items[0]
    write_snapshot(
        snapshots / "prof",
        first.query.query_id,
        [
            Candidate(
                work=Work(
                    work_id="w1",
                    title=first.gold[0].title,
                    doi=first.gold[0].doi,
                    year=2021,
                ),
                sources=["arxiv"],
                source_ranks={"arxiv": 1},
                subquestion_ids=["Q1"],
            )
        ],
    )
    code = cli.main(
        [
            "pool",
            "--snapshots",
            str(snapshots),
            "--inputs",
            "prof",
            "--dir",
            str(data),
            "--n",
            "4",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "# Candidate pool of prof" in out
    assert "3 item(s) without a snapshot" in out
    assert (
        "1 items, 1 gold papers" in out
        and "| in the candidate pool | 1 (100%) |" in out
    )
    code = cli.main(
        [
            "pool",
            "--snapshots",
            str(snapshots / "prof"),
            "--dir",
            str(data),
            "--n",
            "4",
            "--against",
            str(tmp_path / "empty"),
        ]
    )
    assert code == 0
    assert "No item has a snapshot in both roots" in capsys.readouterr().out


# -- the loss table -------------------------------------------------------------------


def entry(
    work: Work, *, p: float | None, kept: bool, rank: int | None = None, number: int = 0
) -> ScreeningEntry:
    return ScreeningEntry(
        work_id=work.work_id or f"w{number}",
        subquestion_id="Q1",
        arrival=number,
        source_ranks={"arxiv": 1},
        p=p,
        kept=kept,
        rank=rank,
        title=work.title,
        year=work.year,
        doi=work.doi,
    )


def run_of(
    item: str,
    *,
    works: list[Work],
    screening: list[ScreeningEntry],
    arm: str = "arm",
    error: str | None = None,
    result: bool = True,
) -> RunRecord:
    output = None
    if result:
        res = make_result("q", works).model_copy(update={"screening": screening})
        output = ArmOutput(
            answer_text="a", works=works, cost=Cost(wall_s=1.0), result=res
        )
    return RunRecord(
        experiment="e",
        experiment_hash="h",
        arm=arm,
        set="litsearch",
        item_id=item,
        repeat=1,
        started_at=0.0,
        output=output,
        error=error,
    )


def padding(n: int) -> list[Work]:
    return [
        Work(
            work_id=f"p{i}",
            title=f"Padding study number {i} of kilns",
            url=f"https://example.org/p{i}",
        )
        for i in range(n)
    ]


def test_every_gold_paper_of_a_run_falls_in_one_diagnosis_category() -> None:
    gold = gold_work(1)
    found = Work(work_id="g", title=gold.title, doi=gold.doi, year=2021)
    # hit: kept and ranked within 20
    assert (
        classify_gold(
            run_of(
                "L", works=[found], screening=[entry(found, p=0.9, kept=True, rank=1)]
            ),
            gold,
        )
        == HIT
    )
    # (c) kept but ranked below 20
    deep = [*padding(20), found]
    assert (
        classify_gold(
            run_of(
                "L", works=deep, screening=[entry(found, p=0.9, kept=True, rank=21)]
            ),
            gold,
        )
        == KEPT_LOW
    )
    assert (
        classify_gold(run_of("L", works=[*padding(19), found], screening=[]), gold)
        == HIT
    )  # rank 20
    # (b) screened and dropped
    assert (
        classify_gold(
            run_of("L", works=[], screening=[entry(found, p=0.2, kept=False)]), gold
        )
        == SCREENED_OUT
    )
    # (b') in the pool, never screened
    assert (
        classify_gold(
            run_of("L", works=[], screening=[entry(found, p=None, kept=False)]), gold
        )
        == NOT_SCREENED
    )
    # (a) not in the pool: another work is there, or nothing is
    other = Work(
        work_id="o",
        title="Another study entirely about something else",
        doi="10.5555/other",
    )
    assert (
        classify_gold(
            run_of("L", works=[], screening=[entry(other, p=0.9, kept=False)]), gold
        )
        == NOT_IN_POOL
    )
    assert classify_gold(run_of("L", works=[], screening=[]), gold) == NOT_IN_POOL
    # kept, yet absent from the result (a retracted work is excluded)
    assert (
        classify_gold(
            run_of("L", works=[], screening=[entry(found, p=0.9, kept=True)]), gold
        )
        == KEPT_EXCLUDED
    )
    # a run with no result, or an error
    assert (
        classify_gold(run_of("L", works=[], screening=[], result=False), gold) == FAILED
    )
    assert (
        classify_gold(run_of("L", works=[found], screening=[], error="boom"), gold)
        == FAILED
    )


def test_a_run_that_screened_but_holds_no_record_is_unrecorded_not_lost_from_the_pool() -> (
    None
):
    gold = gold_work(1)
    old = run_of("L", works=[], screening=[])
    assert old.output is not None and old.output.result is not None
    old.output.result.stats.decision_calls = 40  # it screened candidates
    assert classify_gold(old, gold) == UNRECORDED
    old.output.result.stats.decision_calls = 0
    old.output.result.stats.decision_cache_hits = 3
    assert classify_gold(old, gold) == UNRECORDED
    old.output.result.stats.decision_cache_hits = 0
    assert classify_gold(old, gold) == NOT_IN_POOL  # nothing was screened at all


def test_a_pair_screened_for_one_subquestion_and_waiting_for_another_counts_as_screened() -> (
    None
):
    gold = gold_work(1)
    found = Work(work_id="g", title=gold.title, doi=gold.doi, year=2021)
    held = [entry(found, p=0.1, kept=False), entry(found, p=None, kept=False, number=1)]
    assert classify_gold(run_of("L", works=[], screening=held), gold) == SCREENED_OUT


def test_the_loss_table_counts_per_arm_and_lists_the_gold_papers() -> None:
    golds = {"L1": [gold_work(1), gold_work(2)], "L2": [gold_work(3)]}
    w1 = Work(
        work_id="g1", title=golds["L1"][0].title, doi=golds["L1"][0].doi, year=2021
    )
    w3 = Work(
        work_id="g3", title=golds["L2"][0].title, doi=golds["L2"][0].doi, year=2021
    )
    records = [
        run_of("L1", works=[w1], screening=[entry(w1, p=0.9, kept=True, rank=1)]),
        run_of("L2", works=[], screening=[entry(w3, p=None, kept=False)]),
        run_of("L1", works=[], screening=[], arm="other"),
        run_of("L2", works=[], screening=[], arm="other", result=False),
    ]
    counts, named = loss_counts([r for r in records if r.arm == "arm"], golds)
    assert counts[HIT] == 1 and counts[NOT_IN_POOL] == 1 and counts[NOT_SCREENED] == 1
    assert sum(counts.values()) == 3
    assert named[NOT_IN_POOL] == [f"L1: {golds['L1'][1].title}"]
    text = loss_section(["arm", "other"], records, golds)
    header = "| arm | runs | " + " | ".join(LOSS_CATEGORIES) + " | gold papers |"
    assert header in text
    assert "| arm | 2 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 3 |" in text
    assert "| other | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 3 |" in text
    assert "`(b')` in the pool, never screened" in text
    assert f"- L2: {golds['L2'][0].title}" in text
    assert loss_section(["arm"], [], golds) == "_No LitSearch run to read._"


# -- the report -----------------------------------------------------------------------


def report_experiment() -> Experiment:
    return Experiment.model_validate(
        {
            "name": "r3-small",
            "registered": "2026-10-06",
            "keyless": True,
            "repeats": 1,
            "resamples": 200,
            "inputs": {
                "base": {
                    "sources": {
                        "scholarly": ["europepmc", "arxiv"],
                        "arxiv_query": "all_terms",
                    },
                    "prefetch_top_k": 8,
                },
            },
            "sets": [{"name": "litsearch", "n": 2, "seed": 0}],
            "arms": [
                {
                    "name": "a",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "m",
                    "inputs": {"litsearch": "base"},
                },
                {
                    "name": "b",
                    "system": "v2",
                    "preset": "recall",
                    "model": "m",
                    "inputs": {"litsearch": "base"},
                },
            ],
            "baselines": {"litsearch": "a"},
        }
    )


def report_records(exp: Experiment) -> list[StartRecord | RunRecord]:
    gold = gold_work(1)
    found = Work(work_id="g", title=gold.title, doi=gold.doi, year=2021)

    def record(arm: str, item: str, kept: bool) -> RunRecord:
        works = [found] if kept else []
        screening = [
            entry(found, p=0.9 if kept else None, kept=kept, rank=1 if kept else None)
        ]
        run = run_of(item, works=works, screening=screening, arm=arm)
        run.experiment, run.experiment_hash = exp.name, exp.config_hash()
        assert run.output is not None and run.output.result is not None
        run.output.result.stats.decision_calls = 10
        run.output.result.stats.decision_cache_hits = 5 if arm == "b" else 0
        run.output.result.stats.decision_shared_hits = 5 if arm == "b" else 0
        run.metrics = {
            "recall_at_20": 1.0 if kept else 0.0,
            "inputs_failed_search_share": 0.0,
        }
        return run

    start = StartRecord(
        experiment=exp.name,
        experiment_hash=exp.config_hash(),
        started_at=0.0,
        arms=[],
        keyless=True,
        inputs={"base": exp.inputs["base"].profile_hash()},
    )
    return [start, record("a", "LS0001", False), record("b", "LS0001", True)]


def test_the_report_of_an_inputs_experiment_has_the_round_three_sections(
    tmp_path: Path,
) -> None:
    exp = report_experiment()
    records = report_records(exp)
    snapshots = tmp_path / "snap"
    write_snapshot(
        snapshots / "base",
        "LS0001",
        [pool_candidate(1, ["arxiv"], {"arxiv": 2}), other_candidate(1, ["europepmc"])],
        result_counts={"arxiv": {"q one": 20, "q two": 3}},
    )
    from andamentum.deep_research.v2.bench.litsearch import load_litsearch

    from .test_round3_bench import litsearch_dir

    data = litsearch_dir(tmp_path / "data", n=3)
    first = load_litsearch(data, n=2, seed=0).items[0]
    # the snapshot of the item the experiment will draw, with its real gold paper
    write_snapshot(
        snapshots / "base",
        first.query.query_id,
        [
            Candidate(
                work=Work(
                    work_id="w1",
                    title=first.gold[0].title,
                    doi=first.gold[0].doi,
                    year=2021,
                ),
                sources=["arxiv"],
                source_ranks={"arxiv": 2},
                subquestion_ids=["Q1"],
            )
        ],
        result_counts={"arxiv": {"q one": 20, "q two": 3}},
    )
    bare = build_report(exp, records)
    assert (
        "## Inputs profiles" in bare and "Keyless: no API key was read or used." in bare
    )
    assert "### Decision answers" in bare
    assert "| b | litsearch | 1 | 10.0 | 5.0 | 5.0 | 15.0 |" in bare
    assert "| a | litsearch | 1 | 10.0 | 0.0 | 0.0 | 10.0 |" in bare
    assert "inputs_failed_search_share" in bare  # a column of the paired table
    assert (
        "## Loss table" not in bare and "## Candidate pools" not in bare
    )  # no data given
    full = build_report(exp, records, snapshots=snapshots, litsearch_dir=data)
    assert (
        "## Loss table (LitSearch)" in full and "## Candidate pools (LitSearch)" in full
    )
    assert "### Profile base" in full
    assert "Failed searches: 0 of 4 (0.0%, of which 0 tolerated" in full
    assert "network requests per item (whole freeze)" in full
    assert "| arxiv | 2 | 0 | 0.000 | 5.0 |" in full


def test_the_report_of_a_file_without_profiles_has_none_of_the_new_sections() -> None:
    from .test_bench_experiment_run import small_experiment

    report = build_report(small_experiment(), [])
    for heading in (
        "## Inputs profiles",
        "### Decision answers",
        "## Loss table",
        "## Candidate pools",
    ):
        assert heading not in report


# -- the freeze command line ------------------------------------------------------------


def write_experiment_file(path: Path) -> Path:
    from .test_round3_bench import inputs_experiment

    data = inputs_experiment()
    data["sets"] = [
        {"name": "litsearch", "n": 3, "skip": 3, "seed": 0},
        {"name": "web", "n": 2},
    ]
    for arm in data["arms"]:
        arm["inputs"] = {"litsearch": "base", "web": "old"}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def freeze_args(*argv: str):  # noqa: ANN202
    return cli.build_parser().parse_args(
        ["freeze", "litsearch", "--snapshots", "SNAP", *argv]
    )


def test_a_plain_freeze_resolves_as_it_always_did() -> None:
    plan = cli.resolve_freeze(freeze_args("--model", "ollama:x"))
    assert plan.snapshots == Path("SNAP") and plan.setup == InputsSetup()
    assert (plan.n, plan.skip, plan.seed) == (None, 0, 0)
    assert plan.prefetch_top_k == cli.PREFETCH_TOP_K == 20
    with pytest.raises(ValueError, match="--model is required"):
        cli.resolve_freeze(freeze_args())


def test_a_profile_freeze_takes_items_sources_depth_and_patience_from_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-secret")
    file = write_experiment_file(tmp_path / "x.yaml")
    plan = cli.resolve_freeze(
        freeze_args(
            "--experiment", str(file), "--inputs", "base", "--model", "ollama:x"
        )
    )
    assert plan.snapshots == Path("SNAP") / "base"
    assert (plan.n, plan.skip, plan.seed) == (3, 3, 0)
    assert plan.prefetch_top_k == 8
    assert plan.source_config.scholarly == ("europepmc", "arxiv", "semantic_scholar")
    assert plan.source_config.expand_with == ("europepmc",)
    assert plan.source_config.arxiv_query == "all_terms"
    assert plan.source_config.semantic_scholar_api_key is None  # keyless file
    assert (
        plan.setup.profile == "base" and plan.setup.keyless and plan.setup.plan is None
    )
    assert plan.setup.profile_hash == load_profile_hash(file, "base")
    # the second profile reuses the first one's queries, patiently, and needs no model
    wide = cli.resolve_freeze(
        freeze_args("--experiment", str(file), "--inputs", "wide")
    )
    assert (
        wide.setup.plan == Path("SNAP") / "base" and wide.setup.plan_profile == "base"
    )
    assert wide.setup.patient == ("semantic_scholar",)
    assert wide.source_config.arxiv_query == "any_term"


def load_profile_hash(file: Path, name: str) -> str:
    from andamentum.deep_research.v2.bench.experiment import load_experiment

    return load_experiment(file).inputs[name].profile_hash()


def test_the_registered_items_are_changed_only_by_a_search_only_freeze(
    tmp_path: Path,
) -> None:
    file = write_experiment_file(tmp_path / "x.yaml")
    base = ("--experiment", str(file), "--inputs", "base", "--model", "m")
    for flag, value in (("--n", "9"), ("--skip", "0"), ("--seed", "1")):
        with pytest.raises(
            ValueError, match="allowed only together with --search-only"
        ):
            cli.resolve_freeze(freeze_args(*base, flag, value))
    seen = cli.resolve_freeze(
        freeze_args(
            *base, "--n", "28", "--skip", "0", "--search-only", "--plan-from", "R2"
        )
    )
    assert (seen.n, seen.skip, seen.seed) == (28, 0, 0)
    assert seen.setup.search_only and seen.setup.plan == Path("R2")
    assert seen.setup.plan_profile is None


def test_plan_from_is_refused_without_search_only(tmp_path: Path) -> None:
    file = write_experiment_file(tmp_path / "x.yaml")
    with pytest.raises(ValueError, match="only together with --search-only"):
        cli.resolve_freeze(
            freeze_args(
                "--experiment",
                str(file),
                "--inputs",
                "base",
                "--model",
                "m",
                "--plan-from",
                "R2",
            )
        )


def test_a_profile_freeze_refuses_what_does_not_fit(tmp_path: Path) -> None:
    file = write_experiment_file(tmp_path / "x.yaml")
    with pytest.raises(ValueError, match="--inputs needs --experiment"):
        cli.resolve_freeze(freeze_args("--inputs", "base", "--model", "m"))
    with pytest.raises(ValueError, match="not an inputs profile"):
        cli.resolve_freeze(freeze_args("--experiment", str(file), "--inputs", "nope"))
    with pytest.raises(ValueError, match="existing profile"):
        cli.resolve_freeze(freeze_args("--experiment", str(file), "--inputs", "old"))
    with pytest.raises(ValueError, match="--model is required"):
        cli.resolve_freeze(freeze_args("--experiment", str(file), "--inputs", "base"))


def test_the_freeze_flags_parse() -> None:
    args = freeze_args(
        "--experiment",
        "x.yaml",
        "--inputs",
        "p",
        "--plan-from",
        "D",
        "--search-only",
        "--n",
        "28",
        "--skip",
        "0",
        "--keyless",
    )
    assert args.search_only and args.keyless and args.plan_from == Path("D")
    assert (args.n, args.skip, args.seed) == (28, 0, None)
