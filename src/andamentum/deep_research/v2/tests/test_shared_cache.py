"""The shared read-through HTTP cache, the cache merge, one limiter per source.

Offline: the invented world answers through a mock transport, the language
model is scripted and the decision model a stub. Nothing here waits on a
real rate limit.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.v2 import cli as v2_cli
from andamentum.deep_research.v2.bench import cli as bench_cli
from andamentum.deep_research.v2.sources import SourceClient, SourceConfig, SourceError
from andamentum.deep_research.v2.sources.base import process_pacing
from andamentum.deep_research.v2.store import (
    TTL_SEARCH_S,
    ResearchStore,
    cache_key,
)
from andamentum.references import LICENCE_NO_REDISTRIBUTE

from .engine_world import QUESTION, SEARXNG, World, world_llm

SOURCES = SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",))


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


def put(
    store: ResearchStore,
    name: str,
    body: bytes = b"answer",
    *,
    source: str = "openalex",
    ttl_s: int = TTL_SEARCH_S,
    licence: str = "",
) -> str:
    key = cache_key(source, f"https://example.org/{name}")
    store.cache_put(
        key,
        source=source,
        url=f"https://example.org/{name}",
        status=200,
        body=body,
        ttl_s=ttl_s,
        licence=licence,
    )
    return key


# -- the store: read-through and write-through ---------------------------------------


def test_a_miss_reads_through_and_the_item_store_keeps_a_copy() -> None:
    clock = Clock()
    shared = ResearchStore.in_memory(clock=clock)
    key = put(shared, "a", b"held by the shared cache")
    clock.now += 100
    own = ResearchStore.in_memory(clock=clock, shared_cache=shared)
    hit = own.cache_get(key)
    assert hit is not None and hit.body == b"held by the shared cache"
    assert (own.cache_hits, own.cache_misses, own.shared_hits) == (1, 0, 1)
    # the copy is the entry as it was fetched (its TTL keeps counting from then)
    assert hit.fetched_at == 1_000.0
    # and it is in the item store itself: readable with no shared cache at all
    alone = own.cache_get(key, any_age=True)
    assert alone is not None and alone.fetched_at == 1_000.0
    own.cache_get(key)  # now an own hit
    assert (own.cache_hits, own.shared_hits) == (3, 1)


def test_a_miss_in_both_counts_one_miss_and_a_put_writes_both() -> None:
    shared = ResearchStore.in_memory()
    own = ResearchStore.in_memory(shared_cache=shared)
    key = cache_key("openalex", "https://example.org/new")
    assert own.cache_get(key) is None
    assert (own.cache_hits, own.cache_misses) == (0, 1)
    own.cache_put(key, source="openalex", url="u", status=200, body=b"fresh", ttl_s=60)
    assert shared.cache_get(key) is not None  # the next item finds it
    assert own.cache_get(key) is not None


def test_the_shared_ttl_is_respected_and_an_expired_row_is_a_miss() -> None:
    clock = Clock()
    shared = ResearchStore.in_memory(clock=clock)
    key = put(shared, "short", ttl_s=10)
    own = ResearchStore.in_memory(clock=clock, shared_cache=shared)
    clock.now += 11
    assert own.cache_get(key) is None
    assert own.cache_misses == 1 and own.shared_hits == 0


def test_an_offline_read_never_consults_the_shared_cache() -> None:
    shared = ResearchStore.in_memory()
    key = put(shared, "only-shared")
    own = ResearchStore.in_memory(shared_cache=shared)
    assert own.cache_get(key, any_age=True) is None
    assert own.shared_hits == 0


def test_a_store_cannot_be_its_own_shared_cache() -> None:
    store = ResearchStore.in_memory()
    with pytest.raises(ValueError, match="own shared cache"):
        store._open(":memory:", lambda: 0.0, store)


def test_a_no_redistribute_entry_is_cached_locally_and_never_exported() -> None:
    shared = ResearchStore.in_memory()
    own = ResearchStore.in_memory(shared_cache=shared)
    key = put(
        own,
        "s2",
        b"abstract",
        source="semantic_scholar",
        licence=LICENCE_NO_REDISTRIBUTE,
    )
    other = ResearchStore.in_memory(shared_cache=shared)
    hit = other.cache_get(key)
    assert hit is not None and hit.licence == LICENCE_NO_REDISTRIBUTE
    for store in (shared, own, other):
        assert store.cache_get(key) is not None  # usable inside
        assert list(store.iter_cache()) == []  # left out of every export
        assert len(list(store.iter_cache(exportable_only=False))) == 1


# -- the merge of existing stores ----------------------------------------------------


def test_the_cache_option_is_wired_in_the_bench_and_the_command_line() -> None:
    run = bench_cli.build_parser().parse_args(
        ["experiment", "run", "e.yaml", "--out-dir", "o", "--cache", "c.db"]
    )
    assert run.cache == Path("c.db")
    one = v2_cli.build_parser().parse_args(
        ["q", "--mode", "web", "--model", "m", "--cache", "c.db"]
    )
    assert one.cache == "c.db"


# -- one limiter per source, shared by every client of the process -------------------


def timing_client(
    starts: list[float], gap: float, store: ResearchStore
) -> SourceClient:
    now = asyncio.get_running_loop().time

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(now())
        return httpx.Response(200, json={"u": str(request.url)})

    return SourceClient(
        store,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        intervals={"arxiv": gap, "openalex": 0.0},
    )


async def test_clients_of_separate_items_share_one_limiter_per_source() -> None:
    starts: list[float] = []
    clients = [timing_client(starts, 0.05, ResearchStore.in_memory()) for _ in range(3)]

    async def call(client: SourceClient, n: int) -> None:
        await client.get(
            "arxiv", "https://export.arxiv.org/api/query", params={"n": n}, ttl_s=60
        )

    await asyncio.gather(
        *(
            call(client, i * 10 + j)
            for i, client in enumerate(clients)
            for j in range(2)
        )
    )
    starts.sort()
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert len(starts) == 6 and all(gap >= 0.045 for gap in gaps), gaps
    assert process_pacing() is process_pacing()


async def test_a_source_that_said_its_budget_is_spent_is_not_asked_by_any_item() -> (
    None
):
    sent: list[str] = []

    def spent(request: httpx.Request) -> httpx.Response:
        sent.append("a")
        return httpx.Response(
            200,
            json={},
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "3600"},
        )

    def other(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        sent.append("b")
        return httpx.Response(200, json={})

    first = SourceClient(
        ResearchStore.in_memory(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(spent)),
        intervals={"openalex": 0.0},
    )
    second = SourceClient(
        ResearchStore.in_memory(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(other)),
        intervals={"openalex": 0.0},
    )
    await first.get(
        "openalex", "https://api.openalex.org/works", params={"q": 1}, ttl_s=60
    )
    with pytest.raises(SourceError) as raised:
        await second.get(
            "openalex", "https://api.openalex.org/works", params={"q": 2}, ttl_s=60
        )
    assert raised.value.kind == "rate_limited" and sent == ["a"]
    process_pacing().exhausted_until.clear()


async def test_a_client_with_an_injected_clock_keeps_its_own_pacing() -> None:
    async def no_sleep(_s: float) -> None:
        return None

    store = ResearchStore.in_memory()
    plain = SourceClient(store)
    faked = SourceClient(store, clock=lambda: 0.0, sleep=no_sleep)
    assert plain._locks is process_pacing().locks
    assert faked._locks is not process_pacing().locks


def test_research_refuses_a_cache_it_cannot_apply() -> None:
    from andamentum.deep_research.v2.engine import research

    cache = ResearchStore.in_memory()
    with pytest.raises(ValueError, match="http_cache"):
        asyncio.run(
            research(
                "q",
                mode="web",
                model="ollama:stub",
                store=ResearchStore.in_memory(),
                http_cache=cache,
            )
        )


async def test_a_live_run_in_a_fresh_store_reads_the_shared_cache() -> None:
    from andamentum.deep_research.v2.engine import research

    world = World()
    cache = ResearchStore.in_memory()
    api = {"api.openalex.org", "api.crossref.org"}

    async def run() -> int:
        store = ResearchStore.in_memory(shared_cache=cache)
        await research(
            QUESTION,
            mode="scholarly",
            model=world_llm().model,
            store=store,
            sources=SourceConfig(scholarly=("openalex",)),
            source_client=world.client(store),
        )
        return store.shared_hits

    assert await run() == 0
    sent = [r for r in world.requests if r.url.host in api]
    assert sent
    assert await run() > 0
    # work ids are random and break ties between works, so a fresh store may
    # expand from a different seed (one more citation request); the rest is cached
    again = [r for r in world.requests if r.url.host in api][len(sent) :]
    assert len(again) < len(sent) / 2, [str(r.url) for r in again]


async def test_a_cached_crossref_answer_is_neither_spaced_nor_counted() -> None:
    from andamentum.deep_research.v2.sources import CrossrefSource

    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        return httpx.Response(200, json={"message": {"DOI": "10.5555/x"}})

    store = ResearchStore.in_memory()
    client = SourceClient(
        store,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        intervals={"crossref": 0.0},
    )
    source = CrossrefSource(client=client)
    assert await source.retraction_status("10.5555/x") == "no"
    assert client.requests["crossref"] == 1 and len(sent) == 1
    for _ in range(5):
        assert await source.retraction_status("10.5555/x") == "no"
    assert client.requests["crossref"] == 1 and len(sent) == 1
