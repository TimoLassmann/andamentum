"""Freezing and frozen experiments end to end, offline, over the invented world.

Real engine, store, sources and capture; the network is a mock transport, the
language model a scripted one and the decision model a stub (``engine_world``).
The web question set is replaced by one question the world can answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.v2.bench import experiment_run, freeze
from andamentum.deep_research.v2.bench.arms import V2Arm
from andamentum.deep_research.v2.bench.experiment import Experiment
from andamentum.deep_research.v2.bench.experiment_report import (
    Cells,
    build_report,
    item_stability,
)
from andamentum.deep_research.v2.bench.experiment_run import (
    DataDirs,
    Services,
    preset_of,
    run_experiment,
)
from andamentum.deep_research.v2.bench.freeze import (
    FreezeItem,
    SnapshotMissing,
    freeze_items_of,
    load_frozen,
    plant_into_store,
    planted_frozen,
    prefetch_targets,
    snapshot_dir,
)
from andamentum.deep_research.v2.bench.health import SearxDown
from andamentum.deep_research.v2.bench.planted import planted_source
from andamentum.deep_research.v2.bench.registry import read_records, registry_path
from andamentum.deep_research.v2.bench.web_set import AnswerSpec, WebQuestion
from andamentum.deep_research.v2.frozen import capture
from andamentum.deep_research.v2.sources import SourceConfig, build_sources
from andamentum.deep_research.v2.store import ResearchStore, cache_key

from .engine_world import QUESTION, SEARXNG, World, world_decisions, world_llm

SOURCES = SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",))
WORLD_QUESTION = WebQuestion(
    question_id="W01",
    question=QUESTION,
    answer=AnswerSpec(all_of=[["latency"]]),
    source_url="https://example.org/world",
    expected="latency",
)


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()
    monkeypatch.setattr(experiment_run, "WEB_QUESTIONS", (WORLD_QUESTION,))
    monkeypatch.setattr(freeze, "WEB_QUESTIONS", (WORLD_QUESTION,))


def health_client(world: World) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(world.handler))


async def freeze_world(
    snapshots: Path, world: World, *, items: list[FreezeItem] | None = None
) -> freeze.FreezeReport:
    return await freeze_items_of(
        "web",
        items or [FreezeItem("W01", QUESTION)],
        snapshots=snapshots,
        model=world_llm().model,
        decision_model=world_decisions()[0],
        decision_label="stub",
        source_config=SOURCES,
        progress=lambda _line: None,
        prefetch_top_k=None,
        health_client=health_client(world),
        client_factory=world.client,
        clock=lambda: 1000.0,
    )


# -- freezing --------------------------------------------------------------------


async def test_freeze_writes_the_snapshot_and_warms_the_cache(tmp_path: Path) -> None:
    world = World()
    report = await freeze_world(tmp_path, world)
    assert not report.failed
    assert report.health is not None and report.health.ok
    [item] = report.items
    assert item.status == "frozen" and "texts cached" in item.detail
    folder = snapshot_dir(tmp_path, "web", "W01")
    for name in ("frozen.json", "store.db", "meta.json"):
        assert (folder / name).exists()
    assert not (folder.with_name("W01.partial")).exists()
    frozen = load_frozen(folder)
    assert frozen.question == QUESTION and frozen.mode == "web"
    assert frozen.candidates and not frozen.degraded
    # the page texts of the candidates are in the snapshot's cache
    with ResearchStore(folder / "store.db") as store:
        cached = [
            c
            for c in frozen.candidates
            if c.work.url and store.cache_get(cache_key("web:text", c.work.url))
        ]
    assert cached


async def test_a_second_freeze_keeps_what_exists(tmp_path: Path) -> None:
    world = World()
    await freeze_world(tmp_path, world)
    folder = snapshot_dir(tmp_path, "web", "W01")
    before = (folder / "frozen.json").read_bytes()
    requests = len(world.requests)
    again = await freeze_world(tmp_path, world)
    assert [i.status for i in again.items] == ["kept"]
    assert (folder / "frozen.json").read_bytes() == before
    assert len(world.requests) == requests + 1  # only the health probe


async def test_web_freezing_is_refused_during_a_searxng_outage(tmp_path: Path) -> None:
    world = World(searxng_down=True)
    with pytest.raises(SearxDown, match="not usable"):
        await freeze_world(tmp_path, world)
    assert not (tmp_path / "web").exists()  # nothing was captured


async def test_a_degraded_capture_is_a_failed_item_and_writes_no_snapshot(
    tmp_path: Path,
) -> None:
    world = World()

    def broken_client(store: ResearchStore):  # noqa: ANN202
        client = world.client(store)
        world.searxng_down = True  # the engines die after the health check
        return client

    report = await freeze_items_of(
        "web",
        [FreezeItem("W01", QUESTION)],
        snapshots=tmp_path,
        model=world_llm().model,
        decision_model=world_decisions()[0],
        decision_label="stub",
        source_config=SOURCES,
        progress=lambda _line: None,
        prefetch_top_k=None,
        health_client=health_client(World()),
        client_factory=broken_client,
        clock=lambda: 1000.0,
    )
    assert report.failed
    [item] = report.items
    assert item.status == "failed" and "degraded" in item.detail
    assert not snapshot_dir(tmp_path, "web", "W01").exists()
    assert not (snapshot_dir(tmp_path, "web", "W01").with_name("W01.partial")).exists()


async def test_the_set_chooses_what_can_be_frozen() -> None:
    with pytest.raises(ValueError, match="freeze litsearch or web"):
        freeze.freeze_items("synergy", data_dir=None, n=None, seed=0)
    with pytest.raises(ValueError, match="--dir"):
        freeze.freeze_items("litsearch", data_dir=None, n=None, seed=0)


async def test_missing_snapshot_is_named(tmp_path: Path) -> None:
    with pytest.raises(SnapshotMissing, match="bench freeze"):
        load_frozen(tmp_path / "nowhere")


async def test_prefetch_targets_with_no_limit_are_every_candidate(
    tmp_path: Path,
) -> None:
    await freeze_world(tmp_path, World())
    frozen = load_frozen(snapshot_dir(tmp_path, "web", "W01"))
    everything = prefetch_targets(frozen, top_k=None)
    one = prefetch_targets(frozen, top_k=1)
    assert len(one) <= len(frozen.subquestions) <= len(everything)
    urls = [c.work.url or c.work.title for c in everything]
    assert len(urls) == len(set(urls))  # each work once
    assert {c.work.work_id for c in everything} == {
        c.work.work_id for c in frozen.candidates
    }


async def test_freeze_records_the_rules_that_decide_what_is_cached(
    tmp_path: Path,
) -> None:
    await freeze_world(tmp_path, World())
    meta = json.loads((snapshot_dir(tmp_path, "web", "W01") / "meta.json").read_text())
    assert meta["prefetch_rule"] == "every candidate"
    assert meta["prefetch"]["attempted"] == meta["n_candidates"]
    assert "citations_rule" in meta and "citations" in meta


async def test_citation_warming_makes_the_hops_a_frozen_run_asks_for(
    tmp_path: Path,
) -> None:
    store = ResearchStore.in_memory()
    world = World()
    frozen = await capture(
        QUESTION,
        mode="scholarly",
        model=world_llm().model,
        store=store,
        decision_model=world_decisions()[0],
        sources=SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",)),
        source_client=world.client(store),
    )
    targets = freeze.citation_targets(frozen, top_k=25)
    assert targets and all(freeze.is_scholarly(c.work) for c in targets)
    client = world.client(store)
    sources = build_sources(
        SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",)),
        store=store,
        mode="scholarly",
        client=client,
    )
    report = await freeze.warm_citations(targets, sources=sources.expand)
    assert report.works == len(targets) and report.hops_cached > 0


# -- planting --------------------------------------------------------------------


async def test_the_planted_snapshot_adds_the_false_page_first(tmp_path: Path) -> None:
    await freeze_world(tmp_path, World())
    base = load_frozen(snapshot_dir(tmp_path, "web", "W01"))
    source = planted_source(WORLD_QUESTION)
    planted = planted_frozen(base, source)
    first = planted.candidates[0]
    assert first.work.url == source.url
    assert first.sources == ["planted"]
    assert first.subquestion_ids == [s.id for s in base.subquestions]
    assert first.source_ranks == {"planted": 1}
    assert planted.candidates[1:] == base.candidates  # nothing else changed
    assert planted.digest() != base.digest()
    with pytest.raises(ValueError, match="already among"):
        planted_frozen(planted, source)


async def test_planting_into_a_store_makes_the_page_text_fetchable_offline(
    tmp_path: Path,
) -> None:
    source = planted_source(WORLD_QUESTION)
    path = tmp_path / "store.db"
    plant_into_store(path, source)
    with ResearchStore(path) as store:
        hit = store.cache_get(cache_key("web:text", source.url))
    assert hit is not None and source.statement in hit.body.decode()


# -- a frozen experiment ---------------------------------------------------------


def experiment() -> Experiment:
    return Experiment.model_validate(
        {
            "name": "world",
            "registered": "2026-10-06",
            "repeats": 2,
            "resamples": 100,
            "sets": [
                {"name": "web", "n": 1},
                {"name": "web_planted", "n": 1},
            ],
            "arms": [
                {
                    "name": "baseline",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "stub",
                    "decision_model": "stub-decisions",
                    "frozen": True,
                    "sets": ["web", "web_planted"],
                },
            ],
            "baselines": {"web": "baseline", "web_planted": "baseline"},
        }
    )


def world_services(world: World) -> Services:
    handle, _stub = world_decisions()

    def build(spec, mode, decision):  # noqa: ANN001, ANN202
        return V2Arm(
            mode=mode,
            model=world_llm().model,
            decision_model=decision,
            sources=SOURCES,
            config=preset_of(spec),
            name=spec.name,
        )

    return Services(
        decisions={"stub-decisions": handle}, source_config=SOURCES, answer_arm=build
    )


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_a_frozen_experiment_replays_the_snapshot_offline_and_repeats_agree(
    tmp_path: Path,
) -> None:
    snapshots = tmp_path / "snap"
    await freeze_world(snapshots, World())
    store_file = snapshot_dir(snapshots, "web", "W01") / "store.db"
    before = digest_of(store_file)
    exp = experiment()
    out = tmp_path / "out"
    requests_before = None
    world = World()
    written = await run_experiment(
        exp,
        out_dir=out,
        dirs=DataDirs(snapshots=snapshots),
        services=world_services(world),
    )
    assert requests_before is None and world.requests == []  # no HTTP: offline
    assert len(written) == 4
    assert all(r.error is None for r in written), [r.error for r in written]
    assert digest_of(store_file) == before  # the snapshot itself is never written
    # every cell's working copy is its own and is gone once the cell is done
    assert not list((out / "work").rglob("store.db"))
    plain = [r for r in written if r.set == "web"]
    planted = [r for r in written if r.set == "web_planted"]
    assert all(r.frozen and r.config_hash for r in written)
    assert all(r.output and r.output.result for r in written)
    plain_digest = {
        r.output.result.frozen for r in plain if r.output and r.output.result
    }
    planted_digest = {
        r.output.result.frozen for r in planted if r.output and r.output.result
    }
    assert len(plain_digest) == 1 and len(planted_digest) == 1
    assert plain_digest != planted_digest  # the false page changed the inputs
    assert all(r.planted_outcome for r in planted)
    # the same frozen input twice: the same kept set and the same findings
    stability = item_stability(plain, live=False)
    assert stability["kept_jaccard"] == 1.0
    assert stability["findings_reproduced"] == 1.0
    report = build_report(exp, list(read_records(registry_path(out))))
    assert "## Planted false source" in report
    assert "| baseline | 2 |" in report
    cells = Cells(plain + planted)
    assert cells.n_errors("baseline", "web") == 0


async def test_a_frozen_run_of_an_unfrozen_set_is_named_before_anything_runs(
    tmp_path: Path,
) -> None:
    with pytest.raises(SnapshotMissing, match="web/W01"):
        await run_experiment(
            experiment(),
            out_dir=tmp_path / "out",
            dirs=DataDirs(snapshots=tmp_path / "empty"),
            services=world_services(World()),
        )
