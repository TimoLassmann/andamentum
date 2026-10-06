"""SearxNG answers 200 with no results when its upstream engines are suspended.

That must never read as "no results": it raises, is retried with backoff when
the engines were rate-limited, throttled, cached only on request, and recorded
by both the legacy backend and the v2 web source. No network: MockTransport.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from andamentum.deep_research.backends import HttpxSearchBackend
from andamentum.deep_research.run_searches import run_searches
from andamentum.deep_research.searxng_search import (
    SEARXNG_MIN_INTERVAL_S,
    SEARXNG_RETRIES,
    SEARXNG_RETRY_DELAYS_S,
    SearchEnginesBlocked,
    parse_unresponsive,
)
from andamentum.deep_research.synthesize_report import _stamp_degradation
from andamentum.deep_research.v2.store import ResearchStore

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


class Time:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def backend(replies, *, cache: ResearchStore | None = None):
    """A backend whose SearxNG answers *replies* in turn (the last one repeats)."""
    seen: list[httpx.Request] = []
    fake = Time()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = replies[min(len(seen), len(replies)) - 1]
        return httpx.Response(200, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    made = HttpxSearchBackend(
        searxng_url="http://searx.test",
        http_client=client,
        cache=cache,
        clock=fake.clock,
        sleep=fake.sleep,
    )
    return made, seen, fake


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
    searx, seen, _ = backend([BLOCKED])
    with pytest.raises(SearchEnginesBlocked) as caught:
        await searx.search("apollo")
    assert [(e.engine, e.reason) for e in caught.value.engines] == [
        ("brave", "Suspended: too many requests"),
        ("duckduckgo", "CAPTCHA"),
    ]
    assert caught.value.attempts == SEARXNG_RETRIES == len(seen)


async def test_blocked_is_a_search_failure_and_degrades_the_run() -> None:
    searx, _, _ = backend([BLOCKED])
    [outcome] = await run_searches(["apollo"], backend=searx, max_results=5)
    assert outcome.results == []
    assert outcome.error and "brave (Suspended: too many requests)" in outcome.error

    from andamentum.deep_research.models import EvidenceReport

    report = _stamp_degradation(
        EvidenceReport(
            evidence_summary="x",
            key_findings=["none"],
            sources=["none"],
            total_searches_performed=1,
            total_pages_fetched=0,
            iterations_required=1,
        ),
        search_attempts=1,
        search_failures=1,
        fetch_attempts=0,
        fetch_failures=0,
        summary_attempts=0,
        summary_failures=0,
        threshold=0.5,
    )
    assert report.degraded and "1/1 searches failed" in report.degraded_reason


async def test_retry_sequence_backs_off_5_15_then_raises_after_three_attempts() -> None:
    searx, seen, fake = backend([BLOCKED])
    with pytest.raises(SearchEnginesBlocked):
        await searx.search("apollo")
    assert len(seen) == 3
    assert SEARXNG_RETRY_DELAYS_S == (5.0, 15.0, 45.0)
    # the backoff waits are the first two delays; the throttle's own waits are
    # the 1.5 s spacing and are covered by test_queries_are_spaced
    backoffs = [s for s in fake.sleeps if s >= SEARXNG_RETRY_DELAYS_S[0]]
    assert backoffs == [5.0, 15.0]


async def test_a_blocked_query_that_recovers_returns_its_results() -> None:
    searx, seen, fake = backend([BLOCKED, HEALTHY])
    results = await searx.search("apollo")
    assert [r.url for r in results] == ["https://a.test/x"]
    assert len(seen) == 2 and 5.0 in fake.sleeps


async def test_a_timeout_is_not_retried() -> None:
    searx, seen, fake = backend([TIMEOUT])
    with pytest.raises(SearchEnginesBlocked) as caught:
        await searx.search("apollo")
    assert len(seen) == 1 and caught.value.attempts == 1
    assert not caught.value.retryable
    assert fake.sleeps == []


async def test_partial_answer_returns_results_and_a_warning() -> None:
    searx, seen, _ = backend([PARTIAL])
    results = await searx.search("apollo")
    assert [r.url for r in results] == ["https://a.test/x"]
    assert len(seen) == 1
    [warning] = results.warnings  # type: ignore[attr-defined]
    assert "brave (Suspended: too many requests)" in warning

    [outcome] = await run_searches(["apollo"], backend=searx, max_results=5)
    assert outcome.error is None and outcome.warnings == [warning]


async def test_queries_are_spaced_by_the_shared_interval() -> None:
    searx, _, fake = backend([HEALTHY])
    await searx.search("one")
    await searx.search("two")
    await searx.search("three")
    assert fake.sleeps == [SEARXNG_MIN_INTERVAL_S, SEARXNG_MIN_INTERVAL_S]
    assert SEARXNG_MIN_INTERVAL_S == 1.5


async def test_a_repeated_search_is_served_from_an_explicit_cache() -> None:
    store = ResearchStore.in_memory()
    searx, seen, _ = backend([HEALTHY], cache=store)
    await searx.search("apollo")
    await searx.search("apollo")
    assert len(seen) == 1
    assert store.cache_hits == 1
    [row] = list(store.iter_cache())
    assert row.ttl_s == 7 * 86_400


async def test_a_blocked_answer_is_not_cached() -> None:
    store = ResearchStore.in_memory()
    searx, _, _ = backend([BLOCKED], cache=store)
    with pytest.raises(SearchEnginesBlocked):
        await searx.search("apollo")
    assert list(store.iter_cache()) == []


async def test_without_a_store_nothing_is_cached_or_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    searx, seen, _ = backend([HEALTHY])
    await searx.search("apollo")
    await searx.search("apollo")
    assert len(seen) == 2  # asked twice: no cache
    assert list(tmp_path.iterdir()) == []


async def test_concurrent_queries_share_one_gate() -> None:
    searx, _, fake = backend([HEALTHY])
    await asyncio.gather(*(searx.search(q) for q in ("a", "b", "c")))
    # the first starts at once; the other two wait for their turn at the gate
    assert len(fake.sleeps) == 2 and sum(fake.sleeps) == 2 * SEARXNG_MIN_INTERVAL_S
