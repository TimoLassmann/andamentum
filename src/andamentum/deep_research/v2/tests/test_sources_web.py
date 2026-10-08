"""Web source: SearxNG search and safe fetch, and build_sources."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.v2.sources import browser_fetch
from andamentum.deep_research.v2.sources.browser_fetch import BrowserPage
from andamentum.deep_research.v2.sources import (
    ArxivSource,
    OpenAlexSource,
    SemanticScholarSource,
    SourceConfig,
    SourceError,
    WebSource,
    build_sources,
)
from andamentum.deep_research.v2.sources.base import (
    ExpandingSource,
    Source,
    TextSource,
)
from andamentum.references import Work

from .source_stub import SourceKit, ok

SEARX = "http://127.0.0.1:4070"


def results(request: httpx.Request) -> httpx.Response:
    if request.url.path != "/search":
        return httpx.Response(404)
    if request.url.params.get("pageno", "1") != "1":
        return ok("searxng_empty.json")
    return ok("searxng_results.json")


def web(kit: SourceKit, url: str | None = SEARX) -> WebSource:
    return WebSource(searxng_url=url, client=kit.client)


async def test_search_returns_unique_pages_in_rank_order() -> None:
    kit = SourceKit(results)
    found = await web(kit).search("apollo lattice", limit=10)
    request = kit.requests[0]
    assert request.url.params["q"] == "apollo lattice"
    assert request.url.params["format"] == "json"
    assert [c.work.title for c in found] == [
        "Apollo lattice overview",
        "SaltLog documentation",
    ]
    first = found[0]
    assert (
        first.work.url == "https://example.org/apollo"
    )  # the work holds the canonical URL
    assert first.work.year == 2023 and first.work.venue == "example.org"
    assert first.work.access == "metadata"
    assert first.snippet.startswith("An overview of the Apollo lattice")
    assert first.sources == ["web"] and first.queries == ["apollo lattice"]
    assert [c.rank for c in found] == [0, 1]
    assert found[1].work.year is None


async def test_search_asks_for_more_pages_only_while_short_of_the_limit() -> None:
    kit = SourceKit(results)
    await web(kit).search("apollo lattice", limit=1)
    assert len(kit.requests) == 1
    await web(kit).search("apollo lattice", limit=5)
    # page 1 is in the cache from the first search; only page 2 is asked for
    assert [r.url.params["pageno"] for r in kit.requests[1:]] == ["2"]


async def test_no_results_with_dead_engines_is_an_error_and_is_not_cached() -> None:
    kit = SourceKit(lambda _r: ok("searxng_unanswered.json"))
    with pytest.raises(SourceError) as caught:
        await web(kit).search("apollo", limit=5)
    assert "google (timeout)" in caught.value.reason
    assert "bing (HTTP error 429)" in caught.value.reason
    assert list(kit.store.iter_cache()) == []


async def test_no_results_from_healthy_engines_is_an_honest_empty_list() -> None:
    kit = SourceKit(lambda _r: ok("searxng_empty.json"))
    assert await web(kit).search("zzzz qqqq", limit=5) == []


async def test_a_searxng_that_refuses_json_raises() -> None:
    kit = SourceKit(lambda _r: httpx.Response(403, text="forbidden"))
    with pytest.raises(SourceError) as caught:
        await web(kit).search("apollo", limit=5)
    assert caught.value.status == 403


def test_web_search_needs_an_address() -> None:
    kit = SourceKit(results)
    with pytest.raises(SourceError) as caught:
        web(kit, url=None)
    assert caught.value.kind == "config"
    assert (
        WebSource(searxng_url=SEARX + "/", client=kit.client)._endpoint
        == f"{SEARX}/search"
    )


async def test_a_page_is_fetched_through_the_gate_and_returned_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    fetch_gate._robots_cache_clear()
    body = ("Apollo lattices are sampled sparsely in practice. " * 8).encode()
    kit = SourceKit(
        lambda _r: httpx.Response(
            200, content=body, headers={"content-type": "text/plain"}
        )
    )
    reader = web(kit)
    fetched = await reader.fetch_text(
        Work(
            url="https://docs.example.net/page", title="An invented documentation page"
        )
    )
    assert (
        fetched is not None
        and fetched.text == body.decode()
        and fetched.source == "web"
    )
    assert (
        await reader.fetch_text(
            Work(arxiv_id="2101.00001", title="A work with no web page")
        )
        is None
    )


# -- build_sources -----------------------------------------------------------


def names(sources: object) -> list[str]:
    return [s.name for s in sources]  # type: ignore[attr-defined]


def test_scholarly_mode_searches_the_scholarly_sources_only() -> None:
    kit = SourceKit(results)
    built = build_sources(
        SourceConfig(), store=kit.store, mode="scholarly", client=kit.client
    )
    assert names(built.search) == ["openalex", "europepmc", "arxiv"]
    assert names(built.text) == [
        "europepmc",
        "arxiv",
        "openalex",
    ]  # the order to try reading in
    assert names(built.expand) == ["openalex", "europepmc"]  # arXiv has no links
    assert built.crossref.name == "crossref"
    assert built.client is kit.client


def test_web_mode_needs_an_address_and_searches_the_web_alone() -> None:
    kit = SourceKit(results)
    with pytest.raises(SourceError, match="SearxNG"):
        build_sources(SourceConfig(), store=kit.store, mode="web", client=kit.client)
    built = build_sources(
        SourceConfig(searxng_url=SEARX), store=kit.store, mode="web", client=kit.client
    )
    assert names(built.search) == ["web"]
    assert names(built.text)[-1] == "web"


def test_both_mode_searches_everything_with_the_web_last() -> None:
    kit = SourceKit(results)
    config = SourceConfig(searxng_url=SEARX, scholarly=("openalex", "semantic_scholar"))
    built = build_sources(config, store=kit.store, mode="both", client=kit.client)
    assert names(built.search) == ["openalex", "semantic_scholar", "web"]
    assert names(built.expand) == ["openalex", "semantic_scholar"]
    assert all(isinstance(s, Source) for s in built.search)


def test_the_config_reaches_the_adapters() -> None:
    kit = SourceKit(results)
    config = SourceConfig(
        openalex_api_key="OA",
        mailto="me@example.org",
        openalex_search="both",
        semantic_scholar_api_key="S2",
        scholarly=("openalex", "arxiv", "semantic_scholar"),
        arxiv_pdf_interval_s=5.0,
    )
    built = build_sources(config, store=kit.store, mode="scholarly", client=kit.client)
    openalex, arxiv, s2 = built.search
    assert isinstance(openalex, OpenAlexSource) and openalex._auth == {
        "api_key": "OA",
        "mailto": "me@example.org",
    }
    assert openalex._mode == "both"
    assert isinstance(arxiv, ArxivSource)
    assert kit.client._intervals["arxiv:pdf"] == 5.0
    assert isinstance(s2, SemanticScholarSource) and s2._headers == {"x-api-key": "S2"}


def test_the_protocols_tell_what_an_adapter_can_do() -> None:
    kit = SourceKit(results)
    built = build_sources(
        SourceConfig(), store=kit.store, mode="scholarly", client=kit.client
    )
    arxiv = next(s for s in built.search if s.name == "arxiv")
    openalex = next(s for s in built.search if s.name == "openalex")
    assert isinstance(arxiv, TextSource) and not isinstance(arxiv, ExpandingSource)
    assert isinstance(openalex, TextSource) and isinstance(openalex, ExpandingSource)


# -- engines that did not answer ------------------------------------------------

_RATE_LIMITED = b'{"results": [], "unresponsive_engines": [["brave", "Suspended: too many requests"], ["duckduckgo", "CAPTCHA"]]}'
_PARTIAL = (
    b'{"results": [{"url": "https://example.org/a", "title": "A page", "content": "x"}],'
    b' "unresponsive_engines": [["brave", "Suspended: too many requests"]]}'
)


async def test_rate_limited_engines_are_retried_with_backoff_then_raise() -> None:
    kit = SourceKit(lambda _r: httpx.Response(200, content=_RATE_LIMITED))
    with pytest.raises(SourceError) as caught:
        await web(kit).search("apollo", limit=5)
    assert caught.value.kind == "rate_limited"
    assert "brave (Suspended: too many requests)" in caught.value.reason
    assert len(kit.requests) == 3
    assert kit.time.sleeps == [5.0, 15.0]
    assert list(kit.store.iter_cache()) == []


async def test_a_partial_answer_is_returned_with_a_warning() -> None:
    kit = SourceKit(lambda _r: httpx.Response(200, content=_PARTIAL))
    source = web(kit)
    found = await source.search("apollo", limit=1)
    assert [c.work.title for c in found] == ["A page"]
    [warning] = source.take_warnings()
    assert "brave (Suspended: too many requests)" in warning
    assert source.take_warnings() == []


async def test_web_queries_are_spaced_by_the_shared_interval() -> None:
    kit = SourceKit(results, intervals={"web": 1.5})
    reader = web(kit)
    await reader.search("one", limit=1)
    await reader.search("two", limit=1)
    assert kit.time.sleeps == [1.5]


# -- the Chrome fallback of fetch_text --------------------------------------------

CDP = "http://127.0.0.1:9222"
PAGE = "https://docs.example.net/apollo"
PAGE_TEXT = "Apollo lattices are sampled sparsely in practice. " * 8


class FakeBrowser:
    """Stands in for ``browser_page_text``: counts the pages it was asked for."""

    def __init__(
        self, *, text: str = PAGE_TEXT, error: SourceError | None = None
    ) -> None:
        self.text = text
        self.error = error
        self.urls: list[str] = []
        self.running = 0
        self.peak = 0
        self.hold: asyncio.Event | None = None

    async def __call__(self, cdp_url: str, url: str) -> BrowserPage:
        assert cdp_url == CDP
        self.urls.append(url)
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            if self.hold is not None:
                await self.hold.wait()
            if self.error is not None:
                raise self.error
            return BrowserPage(href=url, title="Apollo", text=self.text)
        finally:
            self.running -= 1


def page_handler(status: int, *, body: bytes = b"blocked"):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            status, content=body, headers={"content-type": "text/plain"}
        )

    return handle


def chrome_web(
    kit: SourceKit,
    browser: FakeBrowser,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cdp: str | None = CDP,
) -> WebSource:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    fetch_gate._robots_cache_clear()
    monkeypatch.setattr(browser_fetch, "browser_page_text", browser)
    return WebSource(searxng_url=SEARX, browser_cdp_url=cdp, client=kit.client)


def apollo_page(url: str = PAGE) -> Work:
    return Work(url=url, title="An invented documentation page")


async def test_a_plain_fetch_that_works_never_opens_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(200, body=PAGE_TEXT.encode()))
    browser = FakeBrowser()
    fetched = await chrome_web(kit, browser, monkeypatch).fetch_text(apollo_page())
    assert fetched is not None and fetched.text == PAGE_TEXT
    assert browser.urls == [] and "chrome-cdp" not in kit.client.extractions.values()


async def test_a_403_is_read_through_the_browser_and_cached_for_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(403))
    browser = FakeBrowser()
    source = chrome_web(kit, browser, monkeypatch)
    before = kit.client.fetched_bytes
    fetched = await source.fetch_text(apollo_page())
    assert fetched is not None and fetched.text == PAGE_TEXT and fetched.source == "web"
    assert "chrome-cdp" in fetched.content_type
    assert browser.urls == [PAGE]
    assert kit.client.extractions[PAGE] == "chrome-cdp"
    assert kit.client.fetched_bytes == before + len(PAGE_TEXT.encode())
    # replay: the text is in the cache under the web text key, no browser, no request
    requests = len(kit.requests)
    again = await source.fetch_text(apollo_page())
    assert again is not None and again.text == PAGE_TEXT
    assert browser.urls == [PAGE] and len(kit.requests) == requests


@pytest.mark.parametrize(
    ("status", "body"), [(404, b"gone"), (500, b"oops"), (403, b"blocked")]
)
async def test_http_errors_fall_back_to_the_browser(
    monkeypatch: pytest.MonkeyPatch, status: int, body: bytes
) -> None:
    kit = SourceKit(page_handler(status, body=body), retries=0)
    browser = FakeBrowser()
    fetched = await chrome_web(kit, browser, monkeypatch).fetch_text(apollo_page())
    assert fetched is not None and browser.urls == [PAGE]


async def test_a_page_with_too_little_text_falls_back_to_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(200, body=b"Enable JavaScript to view this page."))
    browser = FakeBrowser()
    fetched = await chrome_web(kit, browser, monkeypatch).fetch_text(apollo_page())
    assert fetched is not None and fetched.text == PAGE_TEXT and browser.urls == [PAGE]


async def test_a_bot_check_in_the_browser_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(403))
    browser = FakeBrowser(
        error=SourceError("web", "bot check", kind="http", status=403)
    )
    with pytest.raises(SourceError) as caught:
        await chrome_web(kit, browser, monkeypatch).fetch_text(apollo_page())
    assert caught.value.reason == "bot check" and caught.value.status == 403
    assert list(kit.store.iter_cache()) == []  # nothing was kept


async def test_an_unsafe_final_address_is_refused_and_nothing_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(403))
    browser = FakeBrowser(
        error=SourceError("web", "ended on a refused address", kind="refused")
    )
    with pytest.raises(SourceError) as caught:
        await chrome_web(kit, browser, monkeypatch).fetch_text(apollo_page())
    assert caught.value.kind == "refused"
    assert list(kit.store.iter_cache()) == []


async def test_a_browser_transport_failure_keeps_the_plain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(403))
    browser = FakeBrowser(error=SourceError("web", "no Chrome", kind="transport"))
    with pytest.raises(SourceError) as caught:
        await chrome_web(kit, browser, monkeypatch).fetch_text(apollo_page())
    assert caught.value.status == 403 and caught.value.kind == "http"


async def test_a_pdf_never_goes_to_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(403))
    browser = FakeBrowser()
    with pytest.raises(SourceError) as caught:
        await chrome_web(kit, browser, monkeypatch).fetch_text(
            apollo_page("https://docs.example.net/Paper.PDF")
        )
    assert caught.value.status == 403 and browser.urls == []


async def test_no_browser_is_configured_so_the_plain_error_stands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(403))
    browser = FakeBrowser()
    source = chrome_web(kit, browser, monkeypatch, cdp=None)
    with pytest.raises(SourceError) as caught:
        await source.fetch_text(apollo_page())
    assert caught.value.status == 403 and browser.urls == []


async def test_a_rate_limit_and_a_gate_refusal_never_go_to_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(429), retries=0)
    browser = FakeBrowser()
    with pytest.raises(SourceError) as limited:
        await chrome_web(kit, browser, monkeypatch).fetch_text(apollo_page())
    assert limited.value.kind == "rate_limited" and browser.urls == []

    kit = SourceKit(page_handler(200, body=PAGE_TEXT.encode()))
    with pytest.raises(SourceError) as refused:  # a paywalled publisher is refused
        await chrome_web(kit, browser, monkeypatch).fetch_text(
            apollo_page("https://www.nature.com/articles/invented")
        )
    assert refused.value.kind == "refused" and browser.urls == []


async def test_an_offline_client_never_opens_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from andamentum.deep_research.v2.sources.base import SourceClient
    from andamentum.deep_research.v2.store import ResearchStore

    browser = FakeBrowser()
    monkeypatch.setattr(browser_fetch, "browser_page_text", browser)
    source = WebSource(
        browser_cdp_url=CDP,
        client=SourceClient(ResearchStore.in_memory(), offline=True),
    )
    with pytest.raises(SourceError) as caught:
        await source.fetch_text(apollo_page())
    assert caught.value.kind == "offline" and browser.urls == []


async def test_at_most_three_tabs_are_open_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(403))
    browser = FakeBrowser()
    browser.hold = asyncio.Event()
    source = chrome_web(kit, browser, monkeypatch)
    pages = [apollo_page(f"https://docs.example.net/p{i}") for i in range(6)]
    tasks = [asyncio.ensure_future(source.fetch_text(w)) for w in pages]
    for _ in range(200):
        await asyncio.sleep(0)
        if browser.running == 3:
            break
    assert browser.running == 3
    browser.hold.set()
    results = await asyncio.gather(*tasks)
    assert all(r is not None for r in results)
    assert browser.peak == 3 and len(browser.urls) == 6


async def test_a_plain_fetch_that_hangs_is_cut_and_the_browser_is_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from andamentum.deep_research.v2.sources import web

    kit = SourceKit(page_handler(200))
    browser = FakeBrowser()
    source = chrome_web(kit, browser, monkeypatch)

    async def hang(*_args: object, **_kwargs: object) -> object:
        await asyncio.sleep(60)

    monkeypatch.setattr(kit.client, "fetch_document", hang)
    monkeypatch.setattr(web, "PLAIN_FETCH_TIMEOUT_S", 0.01)
    fetched = await source.fetch_text(apollo_page())
    assert fetched is not None and fetched.text == PAGE_TEXT
    assert browser.urls == [PAGE]


async def test_a_hanging_plain_fetch_without_a_browser_is_a_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from andamentum.deep_research.v2.sources import web

    kit = SourceKit(page_handler(200))
    source = chrome_web(kit, FakeBrowser(), monkeypatch, cdp=None)

    async def hang(*_args: object, **_kwargs: object) -> object:
        await asyncio.sleep(60)

    monkeypatch.setattr(kit.client, "fetch_document", hang)
    monkeypatch.setattr(web, "PLAIN_FETCH_TIMEOUT_S", 0.01)
    with pytest.raises(SourceError) as caught:
        await source.fetch_text(apollo_page())
    assert (
        caught.value.kind == "transport" and "no answer within" in caught.value.reason
    )


async def test_a_host_that_showed_a_bot_check_is_not_asked_again_in_the_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = SourceKit(page_handler(403))
    browser = FakeBrowser(
        error=SourceError("web", "bot check", kind="http", status=403)
    )
    source = chrome_web(kit, browser, monkeypatch)
    for path in ("one", "two"):
        with pytest.raises(SourceError) as caught:
            await source.fetch_text(apollo_page(f"https://www.docs.example.net/{path}"))
        assert caught.value.status == 403
    assert len(browser.urls) == 1  # the second page of the host was not tried
    with pytest.raises(SourceError):
        await source.fetch_text(apollo_page("https://elsewhere.example.org/a"))
    assert len(browser.urls) == 2  # another host is still tried
