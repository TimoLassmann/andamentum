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

import json
from collections.abc import Mapping
from urllib.parse import urlparse

from pydantic import JsonValue

from andamentum.references import Work
from andamentum.references.works import canonical_url

from ..store import TTL_SEARCH_S, ResearchStore
from ..types import Candidate
from ._util import as_list, as_mapping, expect_mapping, squash, year_of_date
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
    """SearxNG search and page fetch."""

    name = NAME

    def __init__(
        self,
        *,
        searxng_url: str | None = None,
        store: ResearchStore | None = None,
        client: SourceClient | None = None,
        tdm_allowed_hosts: frozenset[str] = frozenset(),
    ) -> None:
        if not searxng_url:
            raise SourceError(
                NAME,
                "web search needs the address of a SearxNG instance (searxng_url)",
                kind="config",
            )
        self._client = resolve_client(NAME, store=store, client=client)
        self._endpoint = f"{searxng_url.rstrip('/')}/search"
        self._tdm = tdm_allowed_hosts

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
            body = await self._client.get_json(
                NAME,
                self._endpoint,
                params={"q": query, "format": "json", "pageno": page},
                ttl_s=TTL_SEARCH_S,
                validate=_reject_unanswered,
            )
            results = as_list(
                expect_mapping(NAME, body, what="the search").get("results")
            )
            if not results:
                break
            for item in results:
                candidate = _candidate(as_mapping(item), query, rank=len(found))
                if candidate is None:
                    continue
                key = canonical_url(candidate.work.url) or candidate.work.url or ""
                if key not in seen:
                    seen.add(key)
                    found.append(candidate)
            if len(found) >= limit:
                break
        return found[:limit]

    async def fetch_text(self, work: Work) -> FetchedText | None:
        """The text of the page at the work's URL, or None when it has no URL."""
        if not work.url:
            return None
        return await self._client.fetch_document(
            NAME, work.url, tdm_allowed_hosts=self._tdm
        )


def _reject_unanswered(body: bytes) -> None:
    """No results while engines failed says nothing about the web: raise, do not cache."""
    try:
        data = json.loads(body)
    except ValueError:
        return  # get_json reports a body that is not JSON
    if not isinstance(data, dict) or data.get("results"):
        return
    down = as_list(data.get("unresponsive_engines"))
    if down:
        names = ", ".join(
            f"{as_list(e)[0]} ({as_list(e)[1] if len(as_list(e)) > 1 else '?'})"
            if isinstance(e, list) and e
            else str(e)
            for e in down
        )
        raise SourceError(
            NAME, f"no results and the engines did not answer: {names}", kind="http"
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
