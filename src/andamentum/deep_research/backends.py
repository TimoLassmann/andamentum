"""Search backend protocol and default implementation.

SearchBackend is the injection point: application code can inject a
Playwright-based backend, while standalone users get HttpxSearchBackend.
"""

import logging
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

from .models import SearchResult, FetchedPage

logger = logging.getLogger(__name__)


# ── Protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class SearchBackend(Protocol):
    """Protocol for search + fetch operations.

    Implement this to provide a custom backend (e.g. Playwright, Selenium,
    or a paid search API).
    """

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Execute a web search and return structured results."""
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
    ):
        self.searxng_url = searxng_url
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=30.0, follow_redirects=True
        )

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search via SearXNG JSON API."""
        from .circuit_breaker import get_searxng_breaker

        breaker = get_searxng_breaker()
        if not breaker.allow_request():
            logger.warning(
                f"Circuit breaker open, skipping search for '{query[:50]}...'"
            )
            return []

        try:
            resp = await self._http.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            breaker.record_failure()
            logger.error(f"Search failed for '{query[:50]}...': {e}")
            return []

        breaker.record_success()

        raw = data.get("results", [])
        results: list[SearchResult] = []
        for i, item in enumerate(raw[:max_results]):
            url = item.get("url", "")
            if not url:
                continue
            results.append(
                SearchResult(
                    link_id=i,
                    title=item.get("title", "No title").strip(),
                    url=url,
                    snippet=item.get("content", item.get("snippet", "")).strip(),
                    domain=urlparse(url).netloc or "unknown",
                    relevance_score=0.8,
                )
            )
        return results

    async def fetch_page(self, url: str) -> FetchedPage:
        """Fetch page content and extract markdown via content-type routing."""
        from .text_utils import is_safe_url
        from .content_extractor import extract_content, ExtractionError

        is_safe, reason = is_safe_url(url)
        if not is_safe:
            raise RuntimeError(f"URL blocked by SSRF protection: {reason}")

        resp = await self._http.get(url)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        # extract_content is async (it delegates to harvest, which already
        # offloads CPU-bound docling work via asyncio.to_thread internally).
        try:
            markdown = await extract_content(
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
