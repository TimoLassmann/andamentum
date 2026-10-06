"""Frozen inputs: capture once, replay with no planning, no search and no network.

The capture and the replays run the real engine over the invented world of
``engine_world.py``; the replay's source client is offline, and every real
transport is made to fail the test if anything tries to reach it.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety

from .. import research
from ..config import PRESETS
from ..frozen import FrozenInputs, capture
from ..sources import SourceConfig
from ..sources.base import OFFLINE_REASON, SourceClient, SourceError
from ..store import ResearchStore, cache_key
from ..types import Budget, ResearchResult
from .engine_world import (
    QUESTION,
    RETRACTED_DOI,
    SEARXNG,
    World,
    world_decisions,
    world_llm,
)
from .llm_stub import ScriptedLLM

SOURCES = SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",))


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any request that reaches a real transport fails the test."""

    async def refuse(self: object, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a network request was made: {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)


async def _capture(
    store: ResearchStore, world: World, *, decision: bool = True
) -> FrozenInputs:
    return await capture(
        QUESTION,
        mode="scholarly",
        model=world_llm().model,
        store=store,
        decision_model=world_decisions()[0] if decision else None,
        sources=SOURCES,
        source_client=world.client(store),
    )


async def _replay(
    store: ResearchStore,
    frozen: FrozenInputs,
    *,
    preset: str = "baseline",
    decision: bool = True,
) -> tuple[ResearchResult, ScriptedLLM, World]:
    world = World()
    llm = world_llm()
    result = await research(
        QUESTION,
        mode="scholarly",
        model=llm.model,
        decision_model=world_decisions()[0] if decision else None,
        store=store,
        sources=SOURCES,
        config=PRESETS[preset],
        frozen=frozen,
        source_client=world.client(store, offline=True),
    )
    return result, llm, world


# -- capture --------------------------------------------------------------------------


async def test_capture_records_the_plan_the_queries_and_the_ranked_candidates(
    tmp_path: Path,
) -> None:
    store = ResearchStore.in_memory()
    frozen = await _capture(store, World())
    assert frozen.question == QUESTION and frozen.mode == "scholarly"
    assert [s.id for s in frozen.subquestions] == ["Q1", "Q2"]
    assert frozen.degraded == [] and frozen.retraction_unchecked == []
    assert set(frozen.queries) == {"Q1", "Q2"} and all(frozen.queries.values())
    assert frozen.candidates and all(c.work.work_id for c in frozen.candidates)
    assert all(c.source_ranks.get("openalex", 0) >= 1 for c in frozen.candidates)
    assert {c.work.doi for c in frozen.candidates_for("Q2")} == {
        "10.5555/saltlog-d",
        "10.5555/saltlog-e",
    }
    assert frozen.models["model_settings"] == {
        "temperature": PRESETS["baseline"].temperature,
        "seed": PRESETS["baseline"].seed,
    }
    run = store.get_run(frozen.run_id)
    assert run is not None and run.config["frozen_capture"] is True

    path = tmp_path / "snap" / "frozen.json"
    frozen.save(path)
    again = FrozenInputs.load(path)
    assert again == frozen and again.digest() == frozen.digest()

    path.write_text(path.read_text().replace('"version": 1', '"version": 99'))
    with pytest.raises(ValueError, match="version"):
        FrozenInputs.load(path)


# -- replay ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["baseline", "criteria"])
async def test_a_frozen_run_plans_nothing_searches_nothing_and_makes_no_request(
    preset: str, no_network: None
) -> None:
    store = ResearchStore.in_memory()
    frozen = await _capture(store, World())
    result, llm, world = await _replay(store, frozen, preset=preset)

    assert world.requests == []  # the world was never asked anything
    asked = [instructions for instructions, _ in llm.calls]
    assert not [i for i in asked if i.startswith("You split a research question")]
    assert not [i for i in asked if i.startswith("You write search queries")]
    assert result.subquestions == frozen.subquestions
    assert result.queries == frozen.all_queries()
    assert result.frozen == frozen.digest()
    assert result.config_name == preset
    assert result.findings  # read from the abstracts the snapshot's records carry
    assert result.degraded == []
    # citations were never fetched at capture time: skipped, never fetched now
    assert any("not in the snapshot" in line for line in result.skipped)
    # the retraction status came from the capture's checks
    retracted = [w for w in store.iter_works() if w.doi == RETRACTED_DOI]
    assert retracted and retracted[0].work_id in result.excluded
    record = store.get_run(result.run_id)
    assert record is not None
    recorded = record.config["frozen"]
    assert isinstance(recorded, dict) and recorded["digest"] == frozen.digest()


async def test_a_frozen_run_uses_the_citations_and_texts_the_store_holds(
    no_network: None,
) -> None:
    """A store warmed by a live run answers the replay's expansion offline:
    the follow-up study, found only by citation, is screened and read."""
    store = ResearchStore.in_memory()
    world = World()
    frozen = await _capture(store, world)
    await research(
        QUESTION,
        mode="scholarly",
        model=world_llm().model,
        decision_model=world_decisions()[0],
        store=store,
        sources=SOURCES,
        source_client=world.client(store),
    )
    result, _, replay_world = await _replay(store, frozen)
    assert replay_world.requests == []
    assert "10.5555/apollo-f" in {w.doi for w in result.works}


async def test_two_frozen_runs_of_one_arm_give_the_same_evidence() -> None:
    store = ResearchStore.in_memory()
    frozen = await _capture(store, World())
    first, _, _ = await _replay(store, frozen)
    second, _, _ = await _replay(store, frozen)
    assert [w.doi for w in first.works] == [w.doi for w in second.works]
    assert [(f.claim, f.quote) for f in first.findings] == [
        (f.claim, f.quote) for f in second.findings
    ]


async def test_a_frozen_run_refuses_another_question_or_a_live_client() -> None:
    store = ResearchStore.in_memory()
    world = World()
    frozen = await _capture(store, world)
    with pytest.raises(ValueError, match="another question"):
        await research(
            "Another question?",
            mode="scholarly",
            model=world_llm().model,
            store=store,
            sources=SOURCES,
            frozen=frozen,
        )
    with pytest.raises(ValueError, match="offline source client"):
        await research(
            QUESTION,
            mode="scholarly",
            model=world_llm().model,
            store=store,
            sources=SOURCES,
            frozen=frozen,
            source_client=world.client(store),
            budget=Budget(max_rounds=1),
        )


# -- the offline client ----------------------------------------------------------------


async def test_an_offline_client_answers_from_the_cache_only() -> None:
    now = [1000.0]
    store = ResearchStore.in_memory(clock=lambda: now[0])
    url = "https://api.openalex.org/works"
    store.cache_put(
        cache_key("openalex", url, {"q": "a"}),
        source="openalex",
        url=url,
        status=200,
        body=b'{"results": []}',
        ttl_s=10,
    )
    now[0] += 3600  # long expired: an offline run reads it anyway
    client = SourceClient(store, offline=True)
    hit = await client.get("openalex", url, params={"q": "a"}, ttl_s=10)
    assert hit.from_cache and hit.body == b'{"results": []}'
    with pytest.raises(SourceError) as raised:
        await client.get("openalex", url, params={"q": "b"}, ttl_s=10)
    assert raised.value.kind == "offline" and raised.value.reason == OFFLINE_REASON
    with pytest.raises(SourceError) as fetched:
        await client.fetch_document("web", "https://pages.example/a")
    assert fetched.value.kind == "offline"
    with pytest.raises(httpx.ConnectError):
        await client.http.get("https://api.crossref.org/works/10.5555/x")
    assert client.offline_misses == {"openalex": 1, "web": 1, "api.crossref.org": 1}
    assert client.requests == {}
    await client.aclose()


def test_an_offline_client_cannot_wrap_a_live_one() -> None:
    with pytest.raises(ValueError):
        SourceClient(
            ResearchStore.in_memory(), client=httpx.AsyncClient(), offline=True
        )


async def test_an_offline_link_check_makes_no_dns_lookup_and_flags_no_dead_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification of a frozen run must not depend on the machine's DNS."""
    import socket

    from andamentum.references import ReferenceHttp

    from ..sources.base import offline_url_check
    from ..verify import check_links

    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("a DNS lookup was made in an offline run")

    monkeypatch.undo()  # drop the autouse stub of the safe-URL check
    monkeypatch.setattr(socket, "gethostbyname", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    client = SourceClient(store=ResearchStore.in_memory(), offline=True)
    http = ReferenceHttp(
        client=client.http,
        cache=client.reference_cache(),
        url_check=offline_url_check,
    )
    flags, unchecked = await check_links(
        "See https://planted-source.invalid/page and https://example.org/a.", http=http
    )
    assert [f for f in flags if f.kind == "dead_link"] == []
    assert len(unchecked) == 2
