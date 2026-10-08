"""SearxNG answers 200 with no results when its upstream engines are suspended.

That must never read as "no results": it raises, is retried with backoff when
the engines were rate-limited, throttled, cached only on request, and recorded
by the web source. No network: an httpx MockTransport behind a SourceClient.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from andamentum.deep_research.searxng_search import (
    SEARXNG_MIN_INTERVAL_S,
    SEARXNG_RETRIES,
    SEARXNG_RETRY_DELAYS_S,
    SearchEnginesBlocked,
    parse_unresponsive,
    searxng_query,
)
from andamentum.deep_research.sources import SourceError, WebSource
from andamentum.deep_research.sources.base import DEFAULT_INTERVAL_S

from .source_stub import SourceKit

ENDPOINT = "http://searx.test/search"
BLOCKED = {
    "results": [],
    "unresponsive_engines": [
        ["brave", "Suspended: too many requests"],
        ["duckduckgo", "CAPTCHA"],
    ],
}
TIMEOUT = {"results": [], "unresponsive_engines": [["google", "timeout"]]}
PARTIAL = {
    "results": [{"url": "https://a.test/x", "title": "A", "content": "c"}],
    "unresponsive_engines": [["brave", "Suspended: too many requests"]],
}
HEALTHY = {"results": [{"url": "https://a.test/x", "title": "A", "content": "c"}]}


def kit_for(replies: list[dict], *, spaced: bool = False) -> SourceKit:
    """A kit whose SearxNG answers *replies* in turn (the last one repeats)."""
    count = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        count["n"] += 1
        return httpx.Response(200, json=replies[min(count["n"], len(replies)) - 1])

    intervals = {"web": SEARXNG_MIN_INTERVAL_S} if spaced else None
    return SourceKit(handler, intervals=intervals)


def searches(kit: SourceKit) -> list[httpx.Request]:
    return [r for r in kit.requests if r.url.path == "/search"]


def web(kit: SourceKit) -> WebSource:
    return WebSource(searxng_url="http://searx.test", client=kit.client)


def test_parse_unresponsive_reads_engine_reason_pairs() -> None:
    found = parse_unresponsive([["brave", "CAPTCHA"], ["bing"], "odd"])
    assert [(e.engine, e.reason) for e in found] == [
        ("brave", "CAPTCHA"),
        ("bing", ""),
        ("odd", ""),
    ]
    assert parse_unresponsive(None) == ()


async def test_blocked_raises_typed_error_with_the_engines_and_never_returns_empty() -> (
    None
):
    kit = kit_for([BLOCKED])
    with pytest.raises(SearchEnginesBlocked) as caught:
        await searxng_query(kit.client, ENDPOINT, {"q": "apollo", "format": "json"})
    assert [(e.engine, e.reason) for e in caught.value.engines] == [
        ("brave", "Suspended: too many requests"),
        ("duckduckgo", "CAPTCHA"),
    ]
    assert caught.value.attempts == SEARXNG_RETRIES == len(searches(kit))


async def test_blocked_is_a_search_failure_of_the_web_source() -> None:
    kit = kit_for([BLOCKED])
    with pytest.raises(SourceError) as caught:
        await web(kit).search("apollo", limit=5)
    assert caught.value.kind == "rate_limited"
    assert "brave (Suspended: too many requests)" in caught.value.reason


async def test_retry_sequence_backs_off_5_15_then_raises_after_three_attempts() -> None:
    kit = kit_for([BLOCKED])
    with pytest.raises(SearchEnginesBlocked):
        await searxng_query(kit.client, ENDPOINT, {"q": "apollo"})
    assert len(searches(kit)) == 3
    assert SEARXNG_RETRY_DELAYS_S == (5.0, 15.0, 45.0)
    # the backoff waits are the first two delays; spacing is covered below
    assert kit.time.sleeps == [5.0, 15.0]


async def test_a_blocked_query_that_recovers_returns_its_results() -> None:
    kit = kit_for([BLOCKED, HEALTHY])
    page = await searxng_query(kit.client, ENDPOINT, {"q": "apollo"})
    assert [r["url"] for r in page.results] == ["https://a.test/x"]
    assert len(searches(kit)) == 2 and 5.0 in kit.time.sleeps


async def test_a_timeout_is_not_retried() -> None:
    kit = kit_for([TIMEOUT])
    with pytest.raises(SearchEnginesBlocked) as caught:
        await searxng_query(kit.client, ENDPOINT, {"q": "apollo"})
    assert len(searches(kit)) == 1 and caught.value.attempts == 1
    assert not caught.value.retryable
    assert kit.time.sleeps == []


async def test_partial_answer_returns_results_and_a_warning() -> None:
    kit = kit_for([PARTIAL])
    source = web(kit)
    found = await source.search("apollo", limit=5)
    assert [c.work.url for c in found] == ["https://a.test/x"]
    [warning] = source.take_warnings()
    assert "brave (Suspended: too many requests)" in warning
    assert source.take_warnings() == []

    page = await searxng_query(kit.client, ENDPOINT, {"q": "apollo"})
    assert page.warning and "brave" in page.warning


async def test_queries_are_spaced_by_the_shared_interval() -> None:
    assert DEFAULT_INTERVAL_S["web"] == SEARXNG_MIN_INTERVAL_S == 1.5
    kit = kit_for([HEALTHY], spaced=True)
    for q in ("one", "two", "three"):
        await searxng_query(kit.client, ENDPOINT, {"q": q})
    assert kit.time.sleeps == [SEARXNG_MIN_INTERVAL_S, SEARXNG_MIN_INTERVAL_S]


async def test_a_repeated_search_is_served_from_the_store_cache() -> None:
    kit = kit_for([HEALTHY])
    await searxng_query(kit.client, ENDPOINT, {"q": "apollo"})
    page = await searxng_query(kit.client, ENDPOINT, {"q": "apollo"})
    assert len(searches(kit)) == 1 and page.from_cache
    assert kit.store.cache_hits == 1
    [row] = list(kit.store.iter_cache())
    assert row.ttl_s == 7 * 86_400


async def test_a_blocked_answer_is_not_cached() -> None:
    kit = kit_for([BLOCKED])
    with pytest.raises(SearchEnginesBlocked):
        await searxng_query(kit.client, ENDPOINT, {"q": "apollo"})
    assert list(kit.store.iter_cache()) == []


async def test_cache_false_reads_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    kit = kit_for([HEALTHY])
    for _ in range(2):
        await searxng_query(kit.client, ENDPOINT, {"q": "apollo"}, cache=False)
    assert len(searches(kit)) == 2  # asked twice: no cache
    assert list(kit.store.iter_cache()) == []
    assert list(tmp_path.iterdir()) == []


def test_the_web_source_needs_a_store_or_a_client() -> None:
    with pytest.raises(SourceError) as caught:
        WebSource(searxng_url="http://searx.test")
    assert caught.value.kind == "config"


async def test_concurrent_queries_share_one_gate() -> None:
    kit = kit_for([HEALTHY], spaced=True)
    await asyncio.gather(
        *(searxng_query(kit.client, ENDPOINT, {"q": q}) for q in ("a", "b", "c"))
    )
    # the first starts at once; the other two wait for their turn at the gate
    assert len(kit.time.sleeps) == 2
    assert sum(kit.time.sleeps) == 2 * SEARXNG_MIN_INTERVAL_S
