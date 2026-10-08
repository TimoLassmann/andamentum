"""Test kit for the source adapters: recorded fixtures served by a mock transport.

No test here touches the network. ``SourceKit`` wraps an ``httpx.MockTransport``
whose handler records every request, a fake clock and sleep (so spacing and
retry are observable without waiting), and a ``SourceClient`` over both.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx

from andamentum.deep_research.sources.base import SourceClient
from andamentum.deep_research.store import ResearchStore

FIXTURES = Path(__file__).parent / "fixtures"

Handler = Callable[[httpx.Request], httpx.Response]


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def ok(name: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(200, content=fixture(name), headers=headers or {})


class FakeTime:
    """A clock that only moves when a test sleeps."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class SourceKit:
    """A store, a mock transport and a client; ``requests`` lists what was sent."""

    def __init__(
        self,
        handler: Handler,
        *,
        intervals: dict[str, float] | None = None,
        retries: int = 2,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler
        self.time = FakeTime()
        self.store = ResearchStore.in_memory()
        self.http = httpx.AsyncClient(
            transport=httpx.MockTransport(self._record), follow_redirects=False
        )
        self.client = SourceClient(
            self.store,
            client=self.http,
            intervals={
                k: 0.0
                for k in (
                    "openalex",
                    "crossref",
                    "europepmc",
                    "arxiv",
                    "semantic_scholar",
                    "web",
                )
            }
            | (intervals or {}),
            clock=self.time.clock,
            sleep=self.time.sleep,
            retries=retries,
        )

    def _record(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return self._handler(request)

    def param(self, index: int, name: str) -> str | None:
        return self.requests[index].url.params.get(name)
