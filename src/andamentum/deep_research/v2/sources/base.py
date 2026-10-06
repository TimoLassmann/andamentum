"""What every source shares: the protocol, the error, the configuration, and the
one HTTP client all of them go through.

A source finds works (``search``), may give the text of one (``fetch_text``),
and may follow its citation links (``references`` / ``citations``). The three
are separate protocols, so an adapter implements what its API offers and the
engine asks ``isinstance(source, TextSource)`` instead of probing for methods.

Every request goes through :class:`SourceClient`: it reads the store's cache
first, spaces network requests per source, retries the transient failures,
reads the rate and cost headers a server sends, and raises
:class:`SourceError` for anything that is not an answer. An adapter never
returns an empty list to hide a failure.
"""

from __future__ import annotations

import asyncio
import json
import time
import weakref
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlencode, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from andamentum.core.fetch_gate import (
    FetchGateError,
    check_fetch_allowed,
    user_agent_for,
)
from andamentum.core.url_safety import (
    ResponseTooLarge,
    SsrfBlocked,
    fetch_with_safe_redirects,
)
from andamentum.deep_research.content_extractor import (
    ExtractionError,
    extract_content,
)
from andamentum.references import Work

from ..store import (
    TTL_FULL_TEXT_S,
    ReferenceCache,
    ResearchStore,
    cache_key,
)
from ..types import Candidate

SourceName = Literal[
    "openalex", "europepmc", "arxiv", "semantic_scholar", "crossref", "web"
]
ErrorKind = Literal[
    "http",  # the server answered with an error status
    "transport",  # no answer: timeout, connection or DNS failure
    "rate_limited",  # 429, or a budget the server says is spent
    "refused",  # our own gate refused the fetch: SSRF, robots.txt, paywall list
    "parse",  # an answer that is not the shape the API documents
    "config",  # the source cannot run as configured
    "offline",  # an offline run: the answer is not in the store's cache
]

# Smallest gap between two network requests to one source, in seconds.
# OpenAlex and Crossref allow bursts; arXiv's API terms ask for one request in
# three seconds on a single connection; Semantic Scholar's key allows one a
# second. Cached answers cost no network request and are never spaced.
DEFAULT_INTERVAL_S: dict[str, float] = {
    "openalex": 0.1,
    "crossref": 0.1,
    "europepmc": 0.2,
    "arxiv": 3.0,
    "semantic_scholar": 1.0,
    "web": 0.0,
}
# arxiv.org's robots.txt asks for a 15 second crawl delay on its pages and
# PDFs (the API host, export.arxiv.org, disallows everything to robots), so a
# PDF fetch is spaced by this, a separate clock from the API's.
ARXIV_PDF_INTERVAL_S = 15.0
FETCH_INTERVAL_S = 0.5  # between fetches of pages from the open web

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
DEFAULT_RETRIES = 2
MAX_RETRY_WAIT_S = 30.0  # a longer Retry-After or reset is an error, not a sleep
DEFAULT_TIMEOUT_S = 60.0  # OpenAlex's semantic search takes about ten seconds
DEFAULT_RESULTS_PER_QUERY = 20
MIN_FULL_TEXT_CHARS = 200  # shorter is an error page or a stub, not a paper

Extractor = Callable[[bytes, str, str], Awaitable[str]]


class SourceError(RuntimeError):
    """A source could not answer. The engine records it and marks the run degraded."""

    def __init__(
        self,
        source: str,
        reason: str,
        *,
        kind: ErrorKind = "http",
        status: int | None = None,
        url: str | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        self.source = source
        self.reason = reason
        self.kind: ErrorKind = kind
        self.status = status
        self.url = url
        self.retry_after_s = retry_after_s
        super().__init__(f"{source}: {reason}")


class SourceConfig(BaseModel):
    """How the sources are set up. Every secret and address is an explicit field.

    ``scholarly`` names the sources a scholarly run searches; Crossref is not
    among them because it verifies rather than discovers. ``searxng_url`` is
    required for web mode. ``openalex_search`` picks the OpenAlex endpoint:
    both cost credits (about ten per call), so ``keyword`` is the default and
    ``semantic`` or ``both`` is a choice.
    """

    model_config = ConfigDict(frozen=True)

    mailto: str | None = None
    openalex_api_key: str | None = None
    semantic_scholar_api_key: str | None = None
    searxng_url: str | None = None
    tdm_allowed_hosts: frozenset[str] = frozenset()
    scholarly: tuple[
        Literal["openalex", "europepmc", "arxiv", "semantic_scholar"], ...
    ] = ("openalex", "europepmc", "arxiv")
    openalex_search: Literal["keyword", "semantic", "both"] = "keyword"
    results_per_query: int = Field(default=DEFAULT_RESULTS_PER_QUERY, ge=1, le=200)
    arxiv_pdf_interval_s: float = Field(default=ARXIV_PDF_INTERVAL_S, ge=0.0)


class SearchFilters(BaseModel):
    """What a search may be narrowed by. A source applies those it can and
    ignores the rest (arXiv has no open-access filter, for one)."""

    model_config = ConfigDict(frozen=True)

    year_from: int | None = None
    year_to: int | None = None
    open_access_only: bool = False


class FetchedText(BaseModel):
    """The readable text of one work and where it came from."""

    source: str
    url: str
    text: str = Field(min_length=1)
    access: Literal["full_text", "abstract"] = "full_text"
    content_type: str = ""


class RateInfo(BaseModel):
    """What a server last said about its limits. All optional: servers differ."""

    remaining: float | None = None
    limit: float | None = None
    reset_s: float | None = None
    cost_usd: float | None = None
    credits_used: float | None = None
    retry_after_s: float | None = None


class HttpResult(BaseModel):
    status: int
    body: bytes
    from_cache: bool = False

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


@runtime_checkable
class Source(Protocol):
    """Finds works. ``search`` raises :class:`SourceError`; it never returns
    ``[]`` for a failure."""

    name: str

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]: ...


@runtime_checkable
class TextSource(Source, Protocol):
    """A source that can give a work's text. ``None`` means it holds none for
    this work (not an error); a failed attempt raises."""

    async def fetch_text(self, work: Work) -> FetchedText | None: ...


@runtime_checkable
class ExpandingSource(Source, Protocol):
    """A source that follows citation links one hop. A work the source cannot
    place raises :class:`SourceError`."""

    async def references(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]: ...

    async def citations(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]: ...


class _OfflineTransport(httpx.AsyncBaseTransport):
    """A transport that sends nothing: every request fails as a connection
    error, counted by host. An offline client's ``http`` uses it, so a caller
    that makes its own request (Crossref through ``references``, the link
    checks of verification) reaches no network either."""

    def __init__(self, refused: Counter[str]) -> None:
        self._refused = refused

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host or "?"
        self._refused[host] += 1
        raise httpx.ConnectError(
            f"offline run: no network request is made ({host})", request=request
        )


OFFLINE_REASON = "not in the snapshot (offline run); not fetched"


def offline_url_check(_url: str) -> tuple[bool, str]:
    """Accept every URL for the safety pre-check of an offline run.

    The real check resolves the host through DNS before any request, which is a
    network call. An offline client's transport already refuses every request,
    so nothing is sent; the probe then ends as an error ("unchecked"), not a
    dead link that depends on the machine's DNS.
    """
    return True, ""


class SourceClient:
    """The one HTTP client the sources share: cache, spacing, retry, headers.

    ``get`` is for API calls to a fixed host: a redirect is an error. ``fetch_document``
    is for a URL taken from the open web or from a record: it passes the SSRF
    check on every hop, the robots.txt and paywall gate, a size cap and
    the text extractor.

    ``clock`` and ``sleep`` are injected so a test runs without waiting.

    ``offline=True`` makes a client that never touches the network: answers
    come from the store's cache only (expired entries included), a miss
    raises :class:`SourceError` with ``kind='offline'`` and is counted in
    ``offline_misses`` (by source, or by host for a request made through
    ``http``). A frozen run uses one, so every arm sees the same inputs.
    """

    def __init__(
        self,
        store: ResearchStore,
        *,
        client: httpx.AsyncClient | None = None,
        intervals: Mapping[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        retries: int = DEFAULT_RETRIES,
        max_wait_s: float = MAX_RETRY_WAIT_S,
        extractor: Extractor = extract_content,
        max_fetch_bytes: int | None = None,
        user_agent: str | None = None,
        offline: bool = False,
    ) -> None:
        if offline and client is not None:
            raise ValueError("an offline SourceClient makes its own http client")
        self.store = store
        self.user_agent = user_agent or user_agent_for("research")
        self.offline = offline
        self.offline_misses: Counter[str] = Counter()
        self._owns_client = client is None
        self.http = client or httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT_S,
            follow_redirects=False,
            headers={"User-Agent": self.user_agent},
            transport=_OfflineTransport(self.offline_misses) if offline else None,
        )
        self._intervals = {**DEFAULT_INTERVAL_S, **(intervals or {})}
        self._clock = clock
        self._sleep = sleep
        self._retries = retries
        self._max_wait_s = max_wait_s
        self._extractor = extractor
        self.max_fetch_bytes = max_fetch_bytes
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}
        self._exhausted_until: dict[str, float] = {}
        self.rate_info: dict[str, RateInfo] = {}
        self.requests: Counter[str] = Counter()
        self.spent_usd: defaultdict[str, float] = defaultdict(float)
        self.fetched_bytes = 0

    @classmethod
    def for_store(cls, store: ResearchStore) -> SourceClient:
        """The client shared by every adapter built from *store* alone.

        One client per store keeps the spacing between requests to a source
        honest when several adapters (or several tool calls of a long-lived
        server) share it. The first call creates it; it lives as long as the store.
        """
        client = _SHARED.get(store)
        if client is None:
            client = cls(store)
            _SHARED[store] = client
        return client

    @asynccontextmanager
    async def paced(self, source: str) -> AsyncIterator[None]:
        """Hold *source*'s turn and wait out its spacing, for a caller that makes its
        own request (Crossref goes through ``references``' client)."""
        async with self._lock(source):
            await self._space(source)
            self.requests[source] += 1
            try:
                yield
            finally:
                self._last_request[source] = self._clock()

    def reference_cache(self) -> ReferenceCache:
        """The store as the ``references`` cache, serving any age when offline."""
        return self.store.reference_cache(any_age=self.offline)

    def _offline_miss(self, source: str, url: str) -> SourceError:
        self.offline_misses[source] += 1
        return SourceError(source, OFFLINE_REASON, kind="offline", url=url)

    def set_interval(self, name: str, seconds: float) -> None:
        """Set the smallest gap between network requests under spacing key *name*."""
        self._intervals[name] = seconds

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    async def __aenter__(self) -> SourceClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # -- API calls -------------------------------------------------------

    async def get(
        self,
        source: str,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        auth_params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        ttl_s: int,
        licence: str = "",
        validate: Callable[[bytes], None] | None = None,
    ) -> HttpResult:
        """GET *url*, from the cache when it holds a live answer.

        ``params`` and ``url`` make the cache key. ``auth_params`` (an API key,
        a contact mail) and ``headers`` are sent but never keyed or stored, so
        a key is not written into the cache and one answer serves every caller.
        Only a 200 is cached, and only after ``validate`` (which raises
        :class:`SourceError`) accepted it: some APIs answer 200 with an error
        body, and that must not be remembered for a week. Anything else raises.
        """
        key = cache_key(source, url, {k: str(v) for k, v in (params or {}).items()})
        cached = self.store.cache_get(key, any_age=self.offline)
        if cached is not None:
            return HttpResult(status=cached.status, body=cached.body, from_cache=True)
        if self.offline:
            raise self._offline_miss(source, _audit_url(url, params))
        sent = {**(params or {}), **(auth_params or {})}
        async with self._lock(source):
            result = await self._request(
                source, url, sent, {**(headers or {})}, interval=source
            )
        if validate is not None:
            validate(result.body)
        self.store.cache_put(
            key,
            source=source,
            url=_audit_url(url, params),
            status=result.status,
            body=result.body,
            ttl_s=ttl_s,
            licence=licence,
        )
        return result

    async def get_json(
        self,
        source: str,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        auth_params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        ttl_s: int,
        licence: str = "",
        validate: Callable[[bytes], None] | None = None,
    ) -> JsonValue:
        """:meth:`get` with the body read as JSON; a body that is not raises a ``parse`` error."""
        result = await self.get(
            source,
            url,
            params=params,
            auth_params=auth_params,
            headers=headers,
            ttl_s=ttl_s,
            licence=licence,
            validate=validate,
        )
        try:
            return json.loads(result.body)
        except ValueError as exc:
            raise SourceError(
                source, f"the response is not JSON ({exc})", kind="parse", url=url
            ) from exc

    async def _request(
        self,
        source: str,
        url: str,
        params: Mapping[str, str | int],
        headers: Mapping[str, str],
        *,
        interval: str,
    ) -> HttpResult:
        self._refuse_if_exhausted(source, url)
        reason = ""
        for attempt in range(self._retries + 1):
            await self._space(interval)
            self.requests[source] += 1
            try:
                response = await self.http.get(
                    url, params=dict(params), headers=dict(headers)
                )
            except httpx.HTTPError as exc:
                self._last_request[interval] = self._clock()
                reason = f"{type(exc).__name__}: {exc}"
                if attempt < self._retries:
                    await self._sleep(min(2.0**attempt, self._max_wait_s))
                    continue
                raise SourceError(source, reason, kind="transport", url=url) from exc
            self._last_request[interval] = self._clock()
            info = self._read_rate_headers(source, response)
            if response.status_code == 200:
                self.fetched_bytes += len(response.content)
                return HttpResult(status=200, body=response.content)
            reason = _http_reason(response)
            if response.status_code in RETRY_STATUSES and attempt < self._retries:
                wait = (
                    info.retry_after_s
                    if info.retry_after_s is not None
                    else 2.0**attempt
                )
                if wait <= self._max_wait_s:
                    await self._sleep(wait)
                    continue
            raise SourceError(
                source,
                reason,
                kind="rate_limited" if response.status_code == 429 else "http",
                status=response.status_code,
                url=url,
                retry_after_s=info.retry_after_s,
            )
        raise SourceError(source, reason or "no attempt made", url=url)  # unreachable

    def _lock(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    async def _space(self, name: str) -> None:
        gap = self._intervals.get(
            name, FETCH_INTERVAL_S if name.startswith("fetch:") else 0.0
        )
        last = self._last_request.get(name)
        if gap > 0 and last is not None:
            wait = last + gap - self._clock()
            if wait > 0:
                await self._sleep(wait)

    def _read_rate_headers(self, source: str, response: httpx.Response) -> RateInfo:
        h = response.headers
        info = RateInfo(
            remaining=_number(h.get("x-ratelimit-remaining")),
            limit=_number(h.get("x-ratelimit-limit")),
            reset_s=_number(h.get("x-ratelimit-reset")),
            cost_usd=_number(h.get("x-ratelimit-cost-usd")),
            credits_used=_number(h.get("x-ratelimit-credits-used")),
            retry_after_s=_number(h.get("retry-after")),
        )
        self.rate_info[source] = info
        if info.cost_usd:
            self.spent_usd[source] += info.cost_usd
        if info.remaining is not None and info.remaining <= 0 and info.reset_s:
            self._exhausted_until[source] = self._clock() + info.reset_s
        else:
            self._exhausted_until.pop(source, None)
        return info

    def _refuse_if_exhausted(self, source: str, url: str) -> None:
        """A server that said its budget is spent is not asked again until it resets."""
        until = self._exhausted_until.get(source)
        if until is None:
            return
        left = until - self._clock()
        if left <= 0:
            del self._exhausted_until[source]
        elif left > self._max_wait_s:
            raise SourceError(
                source,
                f"the request budget is spent; it resets in {left:.0f} s "
                "(an API key raises it)",
                kind="rate_limited",
                url=url,
                retry_after_s=left,
            )

    # -- documents -------------------------------------------------------

    async def fetch_document(
        self,
        source: str,
        url: str,
        *,
        tdm_allowed_hosts: frozenset[str] = frozenset(),
        interval: str | None = None,
        max_bytes: int | None = None,
        ttl_s: int = TTL_FULL_TEXT_S,
    ) -> FetchedText:
        """The text of the page or PDF at *url*, extracted once and cached.

        Every hop passes the SSRF check; robots.txt and the paywalled-publisher
        list can refuse the fetch (``kind='refused'``); the body is size
        capped; a page with almost no text is an error, not a paper.
        """
        cap = max_bytes if max_bytes is not None else self.max_fetch_bytes
        key = cache_key(f"{source}:text", url)
        cached = self.store.cache_get(key, any_age=self.offline)
        if cached is not None:
            return FetchedText(
                source=source,
                url=url,
                text=cached.body.decode("utf-8"),
                content_type="text/markdown",
            )
        if self.offline:
            raise self._offline_miss(source, url)
        gap = interval or f"fetch:{urlparse(url).netloc.lower()}"
        async with self._lock(gap):
            try:
                await check_fetch_allowed(
                    url,
                    user_agent=self.user_agent,
                    tdm_allowed_hosts=tdm_allowed_hosts,
                    client=self.http,
                )
            except FetchGateError as exc:
                raise SourceError(
                    source, f"fetch refused: {exc}", kind="refused", url=url
                ) from exc
            await self._space(gap)
            self.requests[source] += 1
            try:
                response = await fetch_with_safe_redirects(
                    self.http, url, max_bytes=cap
                )
            except SsrfBlocked as exc:
                raise SourceError(source, str(exc), kind="refused", url=url) from exc
            except ResponseTooLarge as exc:
                raise SourceError(source, str(exc), kind="refused", url=url) from exc
            except httpx.HTTPError as exc:
                raise SourceError(
                    source,
                    f"{type(exc).__name__}: {exc}",
                    kind="transport",
                    url=url,
                ) from exc
            finally:
                self._last_request[gap] = self._clock()
        if response.status_code >= 400:
            raise SourceError(
                source,
                _http_reason(response),
                kind="rate_limited" if response.status_code == 429 else "http",
                status=response.status_code,
                url=url,
            )
        self.fetched_bytes += len(response.content)
        content_type = response.headers.get("content-type", "")
        try:
            text = await self._extractor(
                response.content, content_type, str(response.url)
            )
        except ExtractionError as exc:
            raise SourceError(
                source, f"text extraction failed: {exc}", kind="parse", url=url
            ) from exc
        return self.keep_text(source, url, text, content_type=content_type, ttl_s=ttl_s)

    def keep_text(
        self,
        source: str,
        url: str,
        text: str,
        *,
        content_type: str = "text/markdown",
        ttl_s: int = TTL_FULL_TEXT_S,
    ) -> FetchedText:
        """Cache extracted *text* for *url* and return it; too little text raises."""
        ensure_text(source, url, text)
        self.store.cache_put(
            cache_key(f"{source}:text", url),
            source=f"{source}:text",
            url=url,
            status=200,
            body=text.encode("utf-8"),
            ttl_s=ttl_s,
        )
        return FetchedText(source=source, url=url, text=text, content_type=content_type)


def ensure_text(source: str, url: str, text: str) -> None:
    """Raise a ``parse`` error when *text* is too short to be a paper or a page."""
    if len(text.strip()) < MIN_FULL_TEXT_CHARS:
        raise SourceError(
            source,
            f"only {len(text.strip())} characters of text at {url}",
            kind="parse",
            url=url,
        )


_SHARED: weakref.WeakKeyDictionary[ResearchStore, SourceClient] = (
    weakref.WeakKeyDictionary()
)


def resolve_client(
    source: str, *, store: ResearchStore | None, client: SourceClient | None
) -> SourceClient:
    """The client an adapter uses: the one passed, else the shared one of *store*."""
    if client is not None:
        return client
    if store is not None:
        return SourceClient.for_store(store)
    raise SourceError(
        source,
        "pass a store or a SourceClient: HTTP goes through the cache",
        kind="config",
    )


def _audit_url(url: str, params: Mapping[str, str | int] | None) -> str:
    """The URL as stored with a cache row: no secrets, since only keyed params are in it."""
    return (
        f"{url}?{urlencode(sorted((k, str(v)) for k, v in params.items()))}"
        if params
        else url
    )


def _http_reason(response: httpx.Response) -> str:
    detail = response.text.strip().replace("\n", " ")[:200]
    return f"HTTP {response.status_code}" + (f": {detail}" if detail else "")


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
