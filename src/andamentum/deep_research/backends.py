"""Search backend protocol and default implementation.

SearchBackend is the injection point: application code can inject a
Playwright-based backend, while standalone users get HttpxSearchBackend.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

from andamentum.core.fetch_gate import (
    PaywallBlocked,
    RobotsBlocked,
    check_fetch_allowed,
    user_agent_for,
)

from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .models import SearchResult, FetchedPage
from .searxng_search import SearchEnginesBlocked, searxng_query

if TYPE_CHECKING:
    from .v2.sources.base import SourceClient
    from .v2.store import ResearchStore

logger = logging.getLogger(__name__)

_USER_AGENT = user_agent_for("research")


class SearchBackendError(RuntimeError):
    """A search could not be completed (transport, HTTP or malformed reply).

    Raised instead of returning ``[]`` so the caller can tell "the search
    failed" from "the search found nothing" and record it.
    """


class SearchResults(list[SearchResult]):
    """A search's results, with ``warnings``: set when SearxNG answered with
    results while some of its engines did not (a partial answer)."""

    def __init__(self, results: list[SearchResult], warnings: list[str]) -> None:
        super().__init__(results)
        self.warnings = warnings


# ── Protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class SearchBackend(Protocol):
    """Protocol for search + fetch operations.

    Implement this to provide a custom backend (e.g. Playwright, Selenium,
    or a paid search API).
    """

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Execute a web search and return structured results.

        Raises on failure; an empty list means the search ran and found nothing.
        """
        ...

    async def fetch_page(self, url: str) -> FetchedPage:
        """Fetch and extract text content from a URL."""
        ...


# ── Default Implementation ──────────────────────────────────────────────


class HttpxSearchBackend:
    """Default backend: SearXNG JSON API + httpx page fetch.

    This is a lightweight, standalone backend that requires no browser.
    For richer page extraction (JavaScript rendering), inject a
    Playwright-based backend from Layer 4.
    """

    def __init__(
        self,
        searxng_url: str = "http://127.0.0.1:4070",
        http_client: httpx.AsyncClient | None = None,
        *,
        tdm_allowed_hosts: frozenset[str] = frozenset(),
        breaker: CircuitBreaker | None = None,
        cache: "ResearchStore | None" = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        extract: Callable[[bytes, str, str], Awaitable[str]] | None = None,
        browser_cdp_url: str | None = None,
    ):
        """``cache`` is an explicit store for SearxNG search answers (kept
        seven days, size-capped by the store). There is no default: without
        one, nothing is read from or written to any cache. ``clock`` and
        ``sleep`` time the shared query spacing and the blocked-engine
        backoff, injectable so a test does not wait. ``extract`` turns a
        fetched body ``(data, content_type, url)`` into text (default:
        ``content_extractor.extract_content``); it raises ``ExtractionError``
        on failure. ``browser_cdp_url`` (a Chrome's debugging address) makes
        the search go through that browser instead of SearxNG
        (``browser_search``)."""
        self.searxng_url = searxng_url
        # One breaker per backend instance; ``run_research`` builds a backend
        # per run, so a failure streak never leaks into the next run.
        self._breaker = breaker or CircuitBreaker(name="searxng")
        self._owns_client = http_client is None
        # follow_redirects is off at the client level: page fetches drive the
        # redirect chain through fetch_with_safe_redirects so each hop is
        # SSRF-re-validated. (A redirect chased blindly by httpx could land on
        # a private/loopback/cloud-metadata address.)
        self._http = http_client or httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        )
        self._tdm_allowed_hosts = tdm_allowed_hosts
        self._cache = cache
        self._clock = clock
        self._sleep = sleep
        self._extract = extract
        self._browser = browser_cdp_url
        self._searx_client: SourceClient | None = None

    def _client(self) -> "SourceClient":
        """The shared SearxNG client over this backend's HTTP client: the
        process-wide query spacing, the retry of a blocked query and the cache
        all live there (``searxng_search``). Built on first use."""
        if self._searx_client is None:
            from .v2.sources.base import SourceClient
            from .v2.store import ResearchStore

            self._searx_client = SourceClient(
                self._cache if self._cache is not None else ResearchStore.in_memory(),
                client=self._http,
                clock=self._clock,
                sleep=self._sleep,
                retries=0,  # an HTTP failure is one failure, as it always was
            )
        return self._searx_client

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search via SearXNG JSON API.

        Raises :class:`SearchBackendError` on a transport or HTTP failure or
        an unreadable reply, and :class:`CircuitOpenError` while this
        backend's breaker is open. Never returns ``[]`` for an error.
        :class:`SearchEnginesBlocked` is raised when SearxNG had no results
        because its upstream engines did not answer (after backoff retries).
        A partial answer is returned as :class:`SearchResults` with
        ``warnings``.
        """
        if not self._breaker.allow_request():
            raise CircuitOpenError(self._breaker.name)

        try:
            if self._browser:
                from .browser_search import browser_query

                page = await browser_query(
                    self._client(),
                    self._browser,
                    query,
                    cache=self._cache is not None,
                )
            else:
                page = await searxng_query(
                    self._client(),
                    f"{self.searxng_url}/search",
                    {"q": query, "format": "json"},
                    cache=self._cache is not None,
                )
        except SearchEnginesBlocked:
            self._breaker.record_failure()
            logger.error("Search blocked for '%s': engines did not answer", query[:50])
            raise
        except Exception as e:
            self._breaker.record_failure()
            logger.error(f"Search failed for '{query[:50]}...': {e}")
            raise SearchBackendError(
                f"search failed for {query[:50]!r}: {type(e).__name__}: {e}"
            ) from e

        self._breaker.record_success()
        warnings = [f"{query!r}: {page.warning}"] if page.warning else []
        if warnings:
            logger.warning("Search for '%s': %s", query[:50], page.warning)

        raw = page.results
        results: list[SearchResult] = []
        for i, item in enumerate(raw[:max_results]):
            url = str(item.get("url") or "")
            if not url:
                continue
            results.append(
                SearchResult(
                    link_id=i,
                    title=str(item.get("title", "No title")).strip(),
                    url=url,
                    snippet=str(item.get("content", item.get("snippet", ""))).strip(),
                    domain=urlparse(url).netloc or "unknown",
                    relevance_score=0.8,
                )
            )
        return SearchResults(results, warnings)

    async def fetch_page(self, url: str) -> FetchedPage:
        """Fetch page content and extract markdown via content-type routing."""
        from andamentum.core.url_safety import (
            ResponseTooLarge,
            SsrfBlocked,
            fetch_with_safe_redirects,
        )

        from .text_utils import is_safe_url
        from .content_extractor import extract_content, ExtractionError

        is_safe, reason = is_safe_url(url)
        if not is_safe:
            raise RuntimeError(f"URL blocked by SSRF protection: {reason}")

        try:
            await check_fetch_allowed(
                url,
                user_agent=_USER_AGENT,
                tdm_allowed_hosts=self._tdm_allowed_hosts,
                client=self._http,
            )
        except (PaywallBlocked, RobotsBlocked) as exc:
            raise RuntimeError(f"fetch refused by gate: {exc}") from exc

        # Follow redirects manually, re-validating every hop against the SSRF
        # checks — a 3xx to a private/cloud-metadata host is blocked here.
        try:
            resp = await fetch_with_safe_redirects(self._http, url)
        except SsrfBlocked as exc:
            raise RuntimeError(f"URL blocked by SSRF protection: {exc}") from exc
        except ResponseTooLarge as exc:
            raise RuntimeError(f"Response too large for {url}: {exc}") from exc
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        # extract_content is async (it delegates to harvest, which already
        # offloads CPU-bound docling work via asyncio.to_thread internally).
        try:
            markdown = await (self._extract or extract_content)(
                resp.content, content_type, str(resp.url)
            )
        except ExtractionError as e:
            raise RuntimeError(f"Content extraction failed for {url}: {e}") from e

        if len(markdown.strip()) < 50:
            raise RuntimeError(f"Fetched page has insufficient content from {url}")

        markdown_full_len = len(markdown)
        markdown_capped = markdown[:50000]
        return FetchedPage(
            url=url,
            title=urlparse(url).netloc or url,
            content=markdown_capped,
            word_count=len(markdown_capped.split()),
            relevance_score=0.8,
            is_relevant=True,
            original_length=markdown_full_len,
            truncated=markdown_full_len > 50000,
        )

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> "HttpxSearchBackend":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
