"""R1 reliability: search failures are loud, the breaker belongs to one backend.

No network: httpx.MockTransport stands in for SearXNG.
"""

from __future__ import annotations

import httpx
import pytest

from andamentum.deep_research import circuit_breaker as circuit_breaker_module
from andamentum.deep_research.backends import HttpxSearchBackend, SearchBackendError
from andamentum.deep_research.circuit_breaker import CircuitBreaker, CircuitOpenError
from andamentum.deep_research.run_searches import run_searches


def _backend(handler, *, breaker: CircuitBreaker | None = None) -> HttpxSearchBackend:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpxSearchBackend(
        searxng_url="http://searx.test", http_client=client, breaker=breaker
    )


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"results": [{"url": "https://a.test/x", "title": "A", "content": "c"}]},
    )


async def test_http_error_raises_instead_of_returning_empty() -> None:
    backend = _backend(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(SearchBackendError, match="500"):
        await backend.search("q")


async def test_transport_error_raises() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(SearchBackendError, match="connection refused"):
        await _backend(refuse).search("q")


async def test_unreadable_reply_raises() -> None:
    backend = _backend(lambda request: httpx.Response(200, text="<html>not json"))
    with pytest.raises(SearchBackendError):
        await backend.search("q")


async def test_empty_result_list_is_not_an_error() -> None:
    backend = _backend(lambda request: httpx.Response(200, json={"results": []}))
    assert await backend.search("q") == []


async def test_open_breaker_raises_circuit_open_error() -> None:
    breaker = CircuitBreaker(name="searxng", failure_threshold=2)
    backend = _backend(lambda request: httpx.Response(503), breaker=breaker)
    for _ in range(2):
        with pytest.raises(SearchBackendError):
            await backend.search("q")
    with pytest.raises(CircuitOpenError):
        await backend.search("q")


async def test_breaker_is_per_backend_instance() -> None:
    """One run's failure streak must not make the next run skip its searches."""
    failing = _backend(lambda request: httpx.Response(503))
    for _ in range(5):
        with pytest.raises(SearchBackendError):
            await failing.search("q")
    with pytest.raises(CircuitOpenError):
        await failing.search("q")

    fresh = _backend(_ok)
    results = await fresh.search("q")
    assert [r.url for r in results] == ["https://a.test/x"]


def test_no_process_global_breaker_remains() -> None:
    assert not hasattr(circuit_breaker_module, "get_searxng_breaker")
    assert not hasattr(circuit_breaker_module, "reset_searxng_breaker")


async def test_run_searches_records_a_failed_search_as_an_error() -> None:
    backend = _backend(lambda request: httpx.Response(500))
    outcomes = await run_searches(["q1", "q2"], backend=backend, max_results=5)
    assert [o.query for o in outcomes] == ["q1", "q2"]
    assert all(o.error and "500" in o.error for o in outcomes)
    assert all(o.results == [] for o in outcomes)
