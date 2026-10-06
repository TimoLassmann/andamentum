"""The one place that asks SearxNG a question and reads the answer.

SearxNG meta-searches upstream engines (Brave, DuckDuckGo, Google and others)
and those suspend it under load ("too many requests", a CAPTCHA). It then
answers HTTP 200 with no results and names the engines in
``unresponsive_engines``, a list of ``[engine, reason]`` pairs. That reply
says nothing about the web, so it must never read as "no results".

Every SearxNG query of the package goes through :func:`searxng_query`, the legacy
``HttpxSearchBackend`` and the v2 web source alike:

- empty results with an unresponsive engine raise :class:`SearchEnginesBlocked`
  (never cached);
- when every engine named a rate-limit or CAPTCHA reason, the query is retried
  after ``SEARXNG_RETRY_DELAYS_S`` (5 s, 15 s, 45 s), ``SEARXNG_RETRIES``
  attempts in all, then the error is raised;
- some results with an unresponsive engine are returned with a warning
  (:attr:`SearxPage.warning`);
- queries are spaced ``SEARXNG_MIN_INTERVAL_S`` apart by the shared client's
  per-source limiter (``DEFAULT_INTERVAL_S["web"]``), a process-wide gate.

This module imports no other part of the package at run time, so the v2 client
can take its constants without a cycle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import JsonValue

if TYPE_CHECKING:
    from .v2.sources.base import SourceClient

SEARXNG_SOURCE = "web"  # the spacing key and cache source name of SearxNG queries
SEARXNG_MIN_INTERVAL_S = 1.5  # one query per this many seconds, per process
SEARXNG_RETRIES = 3  # attempts at a blocked query, the first included
SEARXNG_RETRY_DELAYS_S: tuple[float, ...] = (5.0, 15.0, 45.0)  # waits between attempts
SEARXNG_CACHE_TTL_S = 7 * 86_400  # a search answer is kept a week

# Substrings (lower case) of an engine's reason that mean "asked too often or
# challenged", where waiting can help. A timeout or a parse error is not one.
_BLOCK_MARKERS = (
    "too many",
    "captcha",
    "rate limit",
    "rate-limit",
    "suspended",
    "access denied",
    "429",
    "blocked",
)


@dataclass(frozen=True)
class UnresponsiveEngine:
    """One engine SearxNG could not get an answer from, and the reason it gave."""

    engine: str
    reason: str

    def __str__(self) -> str:
        return f"{self.engine} ({self.reason or '?'})"

    @property
    def rate_limited(self) -> bool:
        reason = self.reason.lower()
        return any(marker in reason for marker in _BLOCK_MARKERS)


class SearchEnginesBlocked(RuntimeError):
    """SearxNG returned no results because its upstream engines did not answer.

    ``engines`` lists every ``(engine, reason)`` it named; ``attempts`` is how
    many times the query was asked.
    """

    def __init__(
        self, engines: tuple[UnresponsiveEngine, ...], *, attempts: int = 1
    ) -> None:
        self.engines = engines
        self.attempts = attempts
        names = ", ".join(str(e) for e in engines)
        super().__init__(
            f"no results and the engines did not answer: {names}"
            + (f" (after {attempts} attempts)" if attempts > 1 else "")
        )

    @property
    def retryable(self) -> bool:
        """Every engine named a rate-limit or CAPTCHA reason: waiting may help."""
        return bool(self.engines) and all(e.rate_limited for e in self.engines)


@dataclass
class SearxPage:
    """One SearxNG answer: its results, and the engines that did not answer."""

    results: list[dict[str, JsonValue]] = field(default_factory=list)
    unresponsive: tuple[UnresponsiveEngine, ...] = ()
    from_cache: bool = False

    @property
    def warning(self) -> str:
        """Set when some results came back but some engines did not answer."""
        if self.results and self.unresponsive:
            return "partial search: engines did not answer: " + ", ".join(
                str(e) for e in self.unresponsive
            )
        return ""


def parse_unresponsive(raw: object) -> tuple[UnresponsiveEngine, ...]:
    """The ``unresponsive_engines`` of a SearxNG reply as engines and reasons."""
    found: list[UnresponsiveEngine] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, list) and entry:
                reason = str(entry[1]) if len(entry) > 1 else ""
                found.append(UnresponsiveEngine(str(entry[0]), reason))
            else:
                found.append(UnresponsiveEngine(str(entry), ""))
    return tuple(found)


def parse_page(body: bytes, *, from_cache: bool = False) -> SearxPage:
    """Read a reply body. A body that is not a JSON object raises ``ValueError``."""
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("SearxNG reply is not a JSON object")
    raw = data.get("results")
    results: list[dict[str, JsonValue]] = (
        [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    )
    return SearxPage(
        results=results,
        unresponsive=parse_unresponsive(data.get("unresponsive_engines")),
        from_cache=from_cache,
    )


def reject_unanswered(body: bytes) -> None:
    """Raise :class:`SearchEnginesBlocked` for a reply with no results while
    engines were unresponsive. Used as the client's ``validate`` hook, so the
    reply is not cached. A body that is not usable JSON is left to the caller."""
    try:
        page = parse_page(body)
    except ValueError:
        return
    if not page.results and page.unresponsive:
        raise SearchEnginesBlocked(page.unresponsive)


async def searxng_query(
    client: SourceClient,
    endpoint: str,
    params: Mapping[str, str | int],
    *,
    cache: bool = True,
) -> SearxPage:
    """Ask SearxNG at *endpoint* (``.../search``) and return the page.

    Raises :class:`SearchEnginesBlocked` when it had no results because engines
    did not answer (after the retries, when the reasons were rate limits), and
    the client's ``SourceError`` for a transport or HTTP failure. ``cache=False``
    reads and writes no cache entry.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            result = await client.get(
                SEARXNG_SOURCE,
                endpoint,
                params=params,
                ttl_s=SEARXNG_CACHE_TTL_S,
                validate=reject_unanswered,
                cache=cache,
            )
        except SearchEnginesBlocked as blocked:
            if not blocked.retryable or attempts >= SEARXNG_RETRIES:
                raise SearchEnginesBlocked(
                    blocked.engines, attempts=attempts
                ) from blocked
            await client.pause(SEARXNG_RETRY_DELAYS_S[attempts - 1])
            continue
        try:
            return parse_page(result.body, from_cache=result.from_cache)
        except ValueError as exc:
            from .v2.sources.base import SourceError

            raise SourceError(
                SEARXNG_SOURCE,
                f"the response is not a SearxNG JSON reply ({exc})",
                kind="parse",
                url=endpoint,
            ) from exc
