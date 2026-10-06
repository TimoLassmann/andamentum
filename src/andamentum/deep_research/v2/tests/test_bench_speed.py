"""The freeze prefetch rule, concurrency and its limits, offline.

The prefetch rule (top K per sub-question by source search rank, the rest
abstract-only), the concurrency rule (up to 4 items for an Ollama ``-cloud``
model, 1 for a local one, decision-model requests never overlapping) and the
CLI flags that carry them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic_ai.models.function import FunctionModel

from andamentum.core import fetch_gate, url_safety
from andamentum.core.decisions import DecisionModel
from andamentum.deep_research.v2.bench import cli
from andamentum.deep_research.v2.bench.concurrency import (
    MAX_CLOUD_CONCURRENCY,
    is_cloud_model,
    resolve_concurrency,
    run_bounded,
)
from andamentum.deep_research.v2.bench.experiment_report import build_report
from andamentum.deep_research.v2.bench.experiment_run import (
    DataDirs,
    run_experiment,
)
from andamentum.deep_research.v2.bench.freeze import (
    PREFETCH_TOP_K,
    FreezeItem,
    freeze_items_of,
    load_frozen,
    prefetch_targets,
    snapshot_dir,
    source_rank_score,
)
from andamentum.deep_research.v2.bench.registry import (
    RunRecord,
    StartRecord,
    read_records,
    registry_path,
)
from andamentum.deep_research.v2.frozen import FrozenInputs
from andamentum.deep_research.v2.sources import SourceClient, SourceConfig, SourceError
from andamentum.deep_research.v2.sources.base import FROZEN_TEXT_REASON
from andamentum.deep_research.v2.store import ResearchStore
from andamentum.deep_research.v2.types import Candidate, SubQuestion
from andamentum.references import Work

from .engine_world import QUESTION, SEARXNG, World, world_decisions, world_llm
from .test_bench_experiment_run import (
    ScriptedArm,
    services,
    small_experiment,
    stable_behaviour,
)

SOURCES = SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",))
CLOUD = "ollama:gemma-test:31b-cloud"


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


# -- the prefetch rule ---------------------------------------------------------------


def candidate(number: int, subquestions: list[str], ranks: dict[str, int]) -> Candidate:
    return Candidate(
        work=Work(
            work_id=f"w_{number}",
            title=f"Invented study {number}",
            url=f"https://example.org/{number}",
        ),
        subquestion_ids=subquestions,
        source_ranks=ranks,
        sources=list(ranks) or ["openalex"],
    )


def hand_built(candidates: list[Candidate]) -> FrozenInputs:
    return FrozenInputs(
        question="an invented question",
        mode="scholarly",
        captured_at=1.0,
        subquestions=[
            SubQuestion(id="S1", text="one"),
            SubQuestion(id="S2", text="two"),
        ],
        candidates=candidates,
    )


def test_the_constant_is_twenty() -> None:
    assert PREFETCH_TOP_K == 20


def test_the_rank_score_rewards_agreement_between_source_lists() -> None:
    both = candidate(1, ["S1"], {"openalex": 5, "europepmc": 5})
    one = candidate(2, ["S1"], {"openalex": 1})
    assert source_rank_score(both) > source_rank_score(one)
    assert source_rank_score(candidate(3, ["S1"], {})) == 0.0


def test_top_k_is_per_subquestion_by_source_rank_and_ties_keep_admission_order() -> (
    None
):
    frozen = hand_built(
        [
            candidate(1, ["S1"], {"openalex": 9}),
            candidate(2, ["S1"], {"openalex": 1}),
            candidate(3, ["S1"], {"openalex": 1}),  # ties with 2: admitted later
            candidate(4, ["S2"], {"openalex": 7}),
            candidate(5, ["S2"], {"openalex": 2, "europepmc": 2}),
            candidate(6, ["S1", "S2"], {"openalex": 30}),
        ]
    )
    picked = [c.work.work_id for c in prefetch_targets(frozen, top_k=2)]
    # S1: best two are 2 and 3 (w_6 is 30th, w_1 9th... ranks 1,1 beat 9 and 30);
    # S2: 5 (two lists) beats 4; w_2 and w_3 are S1's, each work once
    assert picked == ["w_2", "w_3", "w_5", "w_4"]
    assert prefetch_targets(frozen, top_k=1)[0].work.work_id == "w_2"
    every = prefetch_targets(frozen, top_k=None)
    assert sorted(c.work.work_id for c in every) == [f"w_{n}" for n in range(1, 7)]


def freeze_args(snapshots: Path, world: World, **extra) -> dict:  # noqa: ANN003
    return {
        "snapshots": snapshots,
        "model": world_llm().model,
        "decision_model": world_decisions()[0],
        "decision_label": "stub",
        "source_config": SOURCES,
        "progress": lambda _line: None,
        "prefetch_top_k": 1,
        "health_client": httpx.AsyncClient(
            transport=httpx.MockTransport(world.handler)
        ),
        "client_factory": world.client,
        "clock": lambda: 1000.0,
        **extra,
    }


async def test_only_the_top_ranked_texts_are_fetched_and_the_rest_are_recorded(
    tmp_path: Path,
) -> None:
    world = World()
    report = await freeze_items_of(
        "web", [FreezeItem("W01", QUESTION)], **freeze_args(tmp_path, world)
    )
    assert not report.failed
    folder = snapshot_dir(tmp_path, "web", "W01")
    frozen = load_frozen(folder)
    meta = json.loads((folder / "meta.json").read_text())
    targets = prefetch_targets(frozen, top_k=1)
    keys = {c.work.work_id for c in frozen.candidates}
    assert meta["prefetch_top_k"] == 1
    assert "best 1 per sub-question" in meta["prefetch_rule"]
    assert set(meta["prefetched_works"]) == {c.work.work_id for c in targets}
    assert set(meta["prefetched_works"]) | set(meta["not_prefetched_works"]) == keys
    assert not set(meta["prefetched_works"]) & set(meta["not_prefetched_works"])
    assert meta["prefetch"]["attempted"] == len(targets)
    assert meta["not_prefetched_works"], "the world has more candidates than targets"
    assert FROZEN_TEXT_REASON in meta["not_prefetched_reading"]
    # the cache agrees: a frozen client finds the prefetched text and refuses the rest
    with ResearchStore(folder / "store.db") as store:
        client = SourceClient(store, offline=True)
        by_id = {c.work.work_id: c for c in frozen.candidates}
        for work_id in meta["not_prefetched_works"]:
            url = by_id[work_id].work.url
            assert url
            with pytest.raises(SourceError) as raised:
                await client.fetch_document("web", url)
            assert raised.value.reason == FROZEN_TEXT_REASON
        served = 0
        for work_id in meta["prefetched_works"]:
            url = by_id[work_id].work.url
            if url:
                served += bool((await client.fetch_document("web", url)).text)
        assert served >= 1
    assert meta["extractor_counts"] is not None and "extractors" in meta
    assert meta["pdf_extraction"] == "docling"  # SourceConfig's library default


# -- concurrency: the rule ------------------------------------------------------------


def test_cloud_models_are_the_ollama_cloud_tags() -> None:
    assert is_cloud_model("ollama:gemma4:31b-cloud")
    assert not is_cloud_model("ollama:gemma4:31b-nvfp4")
    assert not is_cloud_model("openai:gpt-5.4-nano")
    assert not is_cloud_model("gemma4:31b-cloud")  # no provider prefix: not Ollama's


def test_concurrency_is_capped_and_refused_for_local_models() -> None:
    assert MAX_CLOUD_CONCURRENCY == 4
    assert resolve_concurrency(1, models=["ollama:gemma4:31b-nvfp4"]) == 1
    assert resolve_concurrency(4, models=[CLOUD, None]) == 4
    assert resolve_concurrency(4, models=[None]) == 4  # decision model only
    with pytest.raises(ValueError, match="above the cap of 4"):
        resolve_concurrency(5, models=[CLOUD])
    with pytest.raises(ValueError, match="at least 1"):
        resolve_concurrency(0, models=[CLOUD])
    with pytest.raises(ValueError, match="refused for local model.*gemma4:31b-nvfp4"):
        resolve_concurrency(2, models=[CLOUD, "ollama:gemma4:31b-nvfp4"])


async def test_run_bounded_limits_the_jobs_in_flight_and_keeps_job_order() -> None:
    running = 0
    peak = 0

    def job(n: int):  # noqa: ANN202
        async def run() -> int:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01 * (5 - n % 5))
            running -= 1
            return n

        return run

    assert await run_bounded([job(n) for n in range(10)], concurrency=3) == list(
        range(10)
    )
    assert peak == 3
    running = peak = 0
    await run_bounded([job(n) for n in range(4)], concurrency=1)
    assert peak == 1


async def test_a_failing_job_stops_the_ones_after_it() -> None:
    started: list[int] = []

    def job(n: int):  # noqa: ANN202
        async def run() -> int:
            started.append(n)
            await asyncio.sleep(0.001)
            if n == 2:
                raise RuntimeError("down")
            return n

        return run

    with pytest.raises(RuntimeError, match="down"):
        await run_bounded([job(n) for n in range(8)], concurrency=2)
    assert 7 not in started and len(started) < 8


# -- concurrency: freeze --------------------------------------------------------------


class Tracking(httpx.AsyncBaseTransport):
    """Wraps a transport and counts the requests in flight at once."""

    def __init__(
        self, inner: httpx.AsyncBaseTransport, items_begun: list[object]
    ) -> None:
        self.inner = inner
        self.items_begun = items_begun
        self.in_flight = 0
        self.peak = 0
        self.most_items_begun = 0
        self.requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.requests += 1
        self.peak = max(self.peak, self.in_flight)
        self.most_items_begun = max(self.most_items_begun, len(self.items_begun))
        try:
            await asyncio.sleep(0.005)  # a real request takes time: others may run
            return await self.inner.handle_async_request(request)
        finally:
            self.in_flight -= 1


def cloud_world_model() -> FunctionModel:
    llm = world_llm()
    return FunctionModel(llm._answer, model_name="gemma-test-cloud")  # noqa: SLF001


async def test_concurrent_items_never_overlap_decision_requests(
    tmp_path: Path,
) -> None:
    world = World()
    begun: list[object] = []
    _handle, stub = world_decisions()
    tracking = Tracking(stub.transport(), begun)
    decisions = DecisionModel(
        stub.name,
        digest=stub.digest,
        base_url="http://stub.invalid",
        transport=tracking,
    )

    def factory(store: ResearchStore) -> SourceClient:
        begun.append(store)
        return world.client(store)

    items = [FreezeItem(f"W0{n}", QUESTION) for n in (1, 2, 3, 4)]
    report = await freeze_items_of(
        "web",
        items,
        **freeze_args(
            tmp_path,
            world,
            model=cloud_world_model(),
            decision_model=decisions,
            client_factory=factory,
            concurrency=4,
            prefetch_top_k=20,
        ),
    )
    assert not report.failed, report.lines()
    assert report.concurrency == 4
    assert [i.item_id for i in report.items] == ["W01", "W02", "W03", "W04"]
    assert tracking.requests > 0
    assert tracking.most_items_begun >= 2  # the items really were in flight together
    assert tracking.peak == 1  # and their decision requests never overlapped
    assert "concurrency 4" in "\n".join(report.lines())
    meta = json.loads((snapshot_dir(tmp_path, "web", "W01") / "meta.json").read_text())
    assert meta["concurrency"] == 4


async def test_the_decision_lock_serialises_requests_of_separate_handles() -> None:
    _h, stub = world_decisions()
    tracking = Tracking(stub.transport(), [])
    handles = [
        DecisionModel(
            stub.name,
            digest=stub.digest,
            base_url="http://stub.invalid",
            transport=tracking,
        )
        for _ in range(3)
    ]
    from andamentum.core.decisions import Question

    async def ask(handle: DecisionModel) -> None:
        await handle.decide(
            "an invented state",
            {"q": Question(type="noul", instructions="Is this about widgets?")},
        )

    await asyncio.gather(*(ask(h) for h in handles for _ in range(2)))
    assert tracking.requests == 6 and tracking.peak == 1


async def test_concurrency_is_refused_for_a_local_model_before_anything_is_captured(
    tmp_path: Path,
) -> None:
    world = World()
    with pytest.raises(ValueError, match="refused for local model"):
        await freeze_items_of(
            "web",
            [FreezeItem("W01", QUESTION)],
            **freeze_args(tmp_path, world, concurrency=2),  # a function model: local
        )
    assert world.requests == []  # not even the SearxNG probe
    assert not (tmp_path / "web").exists()


def test_the_cli_refuses_concurrency_for_a_local_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "freeze",
            "web",
            "--snapshots",
            str(tmp_path),
            "--model",
            "ollama:gemma4:31b-nvfp4",
            "--concurrency",
            "2",
        ]
    )
    assert code == 1
    assert "refused for local model" in capsys.readouterr().err
    code = cli.main(
        [
            "freeze",
            "web",
            "--snapshots",
            str(tmp_path),
            "--model",
            CLOUD,
            "--concurrency",
            "5",
        ]
    )
    assert code == 1


def test_the_cli_defaults_are_the_bench_settings() -> None:
    args = cli.build_parser().parse_args(
        ["freeze", "web", "--snapshots", "s", "--model", "m"]
    )
    assert args.concurrency == 1
    assert args.prefetch_top_k == PREFETCH_TOP_K
    assert args.pdf_extraction == "fast"
    config = cli._source_config("http://x")  # noqa: SLF001
    assert config.pdf_extraction == "fast"
    run = cli.build_parser().parse_args(
        ["experiment", "run", "f.yaml", "--out-dir", "o", "--concurrency", "3"]
    )
    assert run.concurrency == 3 and run.pdf_extraction == "fast"


# -- concurrency: experiment runs --------------------------------------------------------


def cloud_experiment() -> object:
    return small_experiment(
        repeats=2,
        arms=[
            {
                "name": "base",
                "system": "v2",
                "preset": "baseline",
                "model": CLOUD,
                "frozen": False,
            },
            {
                "name": "other",
                "system": "v2",
                "preset": "baseline",
                "model": CLOUD,
                "frozen": False,
            },
        ],
    )


class OverlapArm(ScriptedArm):
    """Counts how many of its runs are in flight at once."""

    def __init__(self, name: str) -> None:
        super().__init__(name, stable_behaviour)
        self.in_flight = 0
        self.peak = 0

    async def run(self, question, *, frozen=None, store=None):  # noqa: ANN001, ANN201
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return await super().run(question, frozen=frozen, store=store)
        finally:
            self.in_flight -= 1


async def test_an_experiment_runs_cells_concurrently_and_states_it(
    tmp_path: Path,
) -> None:
    exp = cloud_experiment()
    base, other = OverlapArm("base"), OverlapArm("other")
    arms: dict[str, ScriptedArm] = {"base": base, "other": other}
    written = await run_experiment(
        exp,  # type: ignore[arg-type]
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=services(arms),
        concurrency=4,
    )
    assert len(written) == 16  # 4 items x 2 repeats x 2 arms
    assert all(r.error is None for r in written)
    assert base.peak == 4 and other.peak == 4
    records = list(read_records(registry_path(tmp_path)))
    [start] = [r for r in records if isinstance(r, StartRecord)]
    assert start.concurrency == 4
    assert len([r for r in records if isinstance(r, RunRecord)]) == 16
    report = build_report(exp, records)  # type: ignore[arg-type]
    assert "Concurrency used: 4 cell(s) of one arm at a time" in report
    assert "decision-model requests are always one at a time" in report


async def test_an_experiment_defaults_to_one_cell_at_a_time(tmp_path: Path) -> None:
    base, other = OverlapArm("base"), OverlapArm("other")
    arms: dict[str, ScriptedArm] = {"base": base, "other": other}
    await run_experiment(
        cloud_experiment(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=services(arms),
    )
    assert base.peak == 1 and other.peak == 1
    [start] = [
        r for r in read_records(registry_path(tmp_path)) if isinstance(r, StartRecord)
    ]
    assert start.concurrency == 1


async def test_an_experiment_with_a_local_arm_refuses_concurrency(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="refused for local model.*m"):
        await run_experiment(
            small_experiment(),  # arms use the model "m"
            out_dir=tmp_path,
            dirs=DataDirs(),
            services=services({}),
            concurrency=2,
        )
    assert not registry_path(tmp_path).exists()  # nothing was started


# -- one warming rule, citations included ---------------------------------------------


def test_citation_warming_covers_the_scholarly_works_of_the_prefetch_set() -> None:
    from andamentum.deep_research.v2.bench.freeze import citation_targets

    def make(number: int, work: Work, ranks: dict[str, int]) -> Candidate:
        return candidate(number, ["S1"], ranks).model_copy(update={"work": work})

    ranked = [
        make(
            1, Work(work_id="w_s", title="Scholarly", doi="10.1000/s1"), {"openalex": 1}
        ),
        make(
            2,
            Work(work_id="w_w", title="Page", url="https://example.org/p"),
            {"web": 2},
        ),
        make(3, Work(work_id="w_x", title="Far", doi="10.1000/x"), {"openalex": 40}),
    ]
    frozen = hand_built(ranked)
    chosen = [c.work.work_id for c in citation_targets(frozen, top_k=2)]
    assert chosen == ["w_s"]  # the page is not scholarly, w_x is outside the top two
    every = [c.work.work_id for c in citation_targets(frozen, top_k=None)]
    assert every == ["w_s", "w_x"]


async def test_a_rate_limited_hop_is_counted_apart_from_other_failures() -> None:
    from andamentum.deep_research.v2.bench.freeze import warm_citations

    class Hops:
        name = "stub"

        async def references(self, work: Work, *, limit: int):  # noqa: ANN202
            raise SourceError("stub", "HTTP 429: budget", kind="rate_limited")

        async def citations(self, work: Work, *, limit: int):  # noqa: ANN202
            raise SourceError("stub", "HTTP 500", kind="http")

    one = candidate(1, ["S1"], {"openalex": 1})
    report = await warm_citations([one], sources=[Hops()])  # type: ignore[list-item]
    assert report.hops_failed == 2 and report.rate_limited == 1
    assert report.hops_cached == 0 and len(report.errors) == 2


# -- spacing and in-flight requests ----------------------------------------------------


async def test_requests_to_one_source_overlap_in_flight_with_their_starts_spaced() -> (
    None
):
    store = ResearchStore.in_memory()
    client = SourceClient(store, offline=True, intervals={"crossref": 0.05})
    starts: list[float] = []
    now = asyncio.get_running_loop().time
    running = 0
    peak = 0

    async def one() -> None:
        nonlocal running, peak
        async with client.paced("crossref"):
            starts.append(now())
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.2)
            running -= 1

    await asyncio.gather(*(one() for _ in range(3)))
    assert peak == 3  # Crossref's polite pool: three in flight
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert all(gap >= 0.045 for gap in gaps), gaps


async def test_a_source_not_listed_takes_one_request_at_a_time() -> None:
    store = ResearchStore.in_memory()
    client = SourceClient(store, offline=True, intervals={"arxiv": 0.0})
    running = 0
    peak = 0

    async def one() -> None:
        nonlocal running, peak
        async with client.paced("arxiv"):
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1

    await asyncio.gather(*(one() for _ in range(4)))
    assert peak == 1


async def test_a_rewarm_completes_the_cache_and_leaves_the_inputs_alone(
    tmp_path: Path,
) -> None:
    from andamentum.deep_research.v2.bench.freeze import (
        rewarm_items_of,
        snapshot_incomplete,
    )

    world = World()
    items = [FreezeItem("LS1", QUESTION)]
    report = await freeze_items_of(
        "litsearch", items, **freeze_args(tmp_path, world, prefetch_top_k=5)
    )
    assert not report.failed
    folder = snapshot_dir(tmp_path, "litsearch", "LS1")
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["incomplete"] is False and not snapshot_incomplete(folder)
    digest_before = load_frozen(folder).digest()
    frozen_bytes = (folder / "frozen.json").read_bytes()
    # an incomplete snapshot is reported as such, and kept, by a second freeze
    meta_path.write_text(json.dumps({**meta, "incomplete": True}))
    again = await freeze_items_of(
        "litsearch", items, **freeze_args(tmp_path, world, prefetch_top_k=5)
    )
    assert again.incomplete and not again.failed
    assert [i.status for i in again.items] == ["incomplete"]
    assert (folder / "frozen.json").read_bytes() == frozen_bytes
    # the rewarm asks again, completes it and does not touch frozen.json
    done = await rewarm_items_of(
        "litsearch",
        items,
        snapshots=tmp_path,
        model=world_llm().model,
        source_config=SOURCES,
        progress=lambda _line: None,
        prefetch_top_k=5,
        concurrency=1,
        client_factory=world.client,
    )
    assert [i.status for i in done.items] == ["frozen"], done.lines()
    assert not snapshot_incomplete(folder)
    assert (folder / "frozen.json").read_bytes() == frozen_bytes
    assert load_frozen(folder).digest() == digest_before
    assert json.loads(meta_path.read_text())["digest"] == digest_before
    # nothing to rewarm is a failure, never a search
    missing = await rewarm_items_of(
        "litsearch",
        [FreezeItem("LS9", QUESTION)],
        snapshots=tmp_path,
        model=world_llm().model,
        source_config=SOURCES,
        progress=lambda _line: None,
        prefetch_top_k=5,
        concurrency=1,
        client_factory=world.client,
    )
    assert [i.status for i in missing.items] == ["failed"]
