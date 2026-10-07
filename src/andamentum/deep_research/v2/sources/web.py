"""The open web: SearxNG search and a safe fetch.

Search asks a SearxNG instance (its JSON API; the address is an explicit
argument, there is no default) through the store's cache. A SearxNG that
answers with no results while engines were unresponsive is an error, not an
empty result: the answer says nothing about the web, and it is not cached.

Fetching a page goes through ``SourceClient.fetch_document``: the SSRF check
on every hop, robots.txt and the paywalled-publisher list
(``tdm_allowed_hosts`` is the caller's attestation of a licence), a size cap,
and harvest's extractor (trafilatura or docling for HTML, docling for PDF),
cached as text. ``backends.HttpxSearchBackend`` does the same fetch but caps
the text at 50,000 characters and keeps no cache; v2 reads whole pages and
passages, and a search that must raise and be cached, so it makes its calls
through the shared client and reuses the gate, the SSRF layer and the extractor.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlparse

from pydantic import JsonValue

from andamentum.deep_research.browser_search import browser_query
from andamentum.deep_research.searxng_search import (
    SearchEnginesBlocked,
    SearxPage,
    searxng_query,
)
from andamentum.references import Work
from andamentum.references.works import canonical_url

from ..store import ResearchStore
from ..types import Candidate
from ._util import as_mapping, squash, year_of_date
from .base import (
    DEFAULT_RESULTS_PER_QUERY,
    FetchedText,
    SearchFilters,
    SourceClient,
    SourceError,
    resolve_client,
)

NAME = "web"
MAX_PAGES = 3  # SearxNG pages of ~10-20 results; three cover any sensible limit
SNIPPET_CHARS = 1000


class WebSource:
    """Web search (SearxNG, or a Chrome when ``browser_cdp_url`` is given:
    ``deep_research.browser_search``, one page of results) and page fetch."""

    name = NAME

    def __init__(
        self,
        *,
        searxng_url: str | None = None,
        browser_cdp_url: str | None = None,
        store: ResearchStore | None = None,
        client: SourceClient | None = None,
        tdm_allowed_hosts: frozenset[str] = frozenset(),
    ) -> None:
        if not searxng_url and not browser_cdp_url:
            raise SourceError(
                NAME,
                "web search needs a SearxNG instance (searxng_url) or a Chrome "
                "(browser_cdp_url)",
                kind="config",
            )
        self._client = resolve_client(NAME, store=store, client=client)
        self._browser = browser_cdp_url
        self._endpoint = f"{(searxng_url or '').rstrip('/')}/search"
        self._tdm = tdm_allowed_hosts
        self._warnings: list[str] = []

    def take_warnings(self) -> list[str]:
        """The partial-search warnings since the last call (and forget them)."""
        taken, self._warnings = self._warnings, []
        return taken

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """Pages SearxNG returns for *query*, in its rank order, at most *limit*.

        ``filters`` do not apply to the open web and are ignored.
        """
        found: list[Candidate] = []
        seen: set[str] = set()
        for page in range(1, MAX_PAGES + 1):
            try:
                answer = await self._ask(query, page=page)
            except SearchEnginesBlocked as blocked:
                raise SourceError(
                    NAME,
                    str(blocked),
                    kind="rate_limited" if blocked.retryable else "http",
                ) from blocked
            if answer.warning:
                note = f"web search for {query!r}: {answer.warning}"
                if note not in self._warnings:
                    self._warnings.append(note)
            if not answer.results:
                break
            for item in answer.results:
                candidate = _candidate(as_mapping(item), query, rank=len(found))
                if candidate is None:
                    continue
                key = canonical_url(candidate.work.url) or candidate.work.url or ""
                if key not in seen:
                    seen.add(key)
                    found.append(candidate)
            if len(found) >= limit or self._browser:
                break
        return found[:limit]

    async def _ask(self, query: str, *, page: int) -> SearxPage:
        if self._browser:
            return await browser_query(self._client, self._browser, query)
        return await searxng_query(
            self._client,
            self._endpoint,
            {"q": query, "format": "json", "pageno": page},
        )

    async def fetch_text(self, work: Work) -> FetchedText | None:
        """The text of the page at the work's URL, or None when it has no URL."""
        if not work.url:
            return None
        return await self._client.fetch_document(
            NAME, work.url, tdm_allowed_hosts=self._tdm
        )


def _candidate(
    item: Mapping[str, JsonValue], query: str, *, rank: int
) -> Candidate | None:
    url = squash(item.get("url"))
    title = squash(item.get("title"))
    if not url or not title:
        return None
    host = (urlparse(url).hostname or "").removeprefix("www.")
    snippet = squash(item.get("content") or item.get("snippet"))[:SNIPPET_CHARS]
    try:
        work = Work(
            url=url,
            title=title,
            year=year_of_date(item.get("publishedDate")),
            venue=host,
            access="metadata",
        )
    except ValueError:
        return None  # a URL the identity layer cannot read is not a source we can cite
    return Candidate(
        work=work, sources=[NAME], snippet=snippet, queries=[query], rank=rank
    )
