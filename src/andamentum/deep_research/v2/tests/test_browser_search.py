"""Web search through a browser: reading the page, the engines, cache and errors.

No browser and no network: the page reading is invented JSON, and the step
that would open a tab in Chrome is replaced by a script keyed by the engine.
"""

from __future__ import annotations

import base64
import json
from urllib.parse import quote

import pytest

from andamentum.deep_research import browser_search
from andamentum.deep_research.browser_search import (
    BING,
    DUCKDUCKGO,
    browser_query,
    read_page,
    unwrap,
)
from andamentum.deep_research.searxng_search import SearchEnginesBlocked

from ..sources.base import SourceClient, SourceError
from ..sources.web import WebSource
from ..store import ResearchStore


def reading(state: str, rows: list[tuple[str, str]]) -> str:
    return json.dumps(
        {
            "state": state,
            "rows": [
                {"title": title, "href": href, "snippet": f"About {title}."}
                for title, href in rows
            ],
        }
    )


FERRY = "https://example.org/apollo-ferry"
DDG_FERRY = "https://duckduckgo.com/l/?uddg=" + quote(FERRY, safe="") + "&rut=x"
BING_FERRY = (
    "https://www.bing.com/ck/a?!&&p=abc&u=a1"
    + base64.urlsafe_b64encode(FERRY.encode()).decode().rstrip("=")
    + "&ntb=1"
)


class FakeChrome:
    """Answers each engine's page from a script and counts the pages opened."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.opened: list[str] = []

    async def __call__(self, cdp_url: str, *, url: str, script: str) -> str:
        engine = "duckduckgo" if url.startswith(DUCKDUCKGO.url_prefix) else "bing"
        self.opened.append(engine)
        return self.pages[engine]


def client(*, offline: bool = False) -> SourceClient:
    async def no_wait(_seconds: float) -> None:
        return None

    store = ResearchStore.in_memory()
    if offline:
        return SourceClient(store, offline=True)
    return SourceClient(store, clock=lambda: 0.0, sleep=no_wait)


def test_engine_links_are_unwrapped_to_their_target() -> None:
    assert unwrap(DDG_FERRY) == FERRY
    assert unwrap(BING_FERRY) == FERRY
    assert unwrap(FERRY) == FERRY


def test_a_page_reading_is_results_empty_or_a_bot_check() -> None:
    state, results = read_page(reading("results", [("Apollo Ferry", DDG_FERRY)]))
    assert state == "results"
    assert results == [
        {"url": FERRY, "title": "Apollo Ferry", "content": "About Apollo Ferry."}
    ]
    assert read_page(reading("empty", []))[0] == "empty"
    assert read_page(reading("blocked", []))[0] == "blocked"
    # rows whose links lead nowhere usable are not results
    assert read_page(reading("results", [("Ad", "javascript:void(0)")]))[0] == "blocked"
    with pytest.raises(ValueError):
        read_page('{"rows": []}')


async def test_duckduckgo_answers_first_and_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome = FakeChrome({"duckduckgo": reading("results", [("Apollo", DDG_FERRY)])})
    monkeypatch.setattr(browser_search, "_evaluate_in_new_tab", chrome)
    shared = client()
    first = await browser_query(shared, "http://127.0.0.1:9222", "apollo ferry")
    again = await browser_query(shared, "http://127.0.0.1:9222", "apollo ferry")
    assert [r["url"] for r in first.results] == [FERRY]
    assert first.warning == "" and not first.from_cache
    assert again.from_cache and again.results == first.results
    assert chrome.opened == ["duckduckgo"]


async def test_a_bot_check_on_duckduckgo_asks_bing_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome = FakeChrome(
        {
            "duckduckgo": reading("blocked", []),
            "bing": reading("results", [("Apollo", BING_FERRY)]),
        }
    )
    monkeypatch.setattr(browser_search, "_evaluate_in_new_tab", chrome)
    page = await browser_query(client(), "http://127.0.0.1:9222", "apollo ferry")
    assert chrome.opened == ["duckduckgo", "bing"]
    assert [r["url"] for r in page.results] == [FERRY]
    assert "duckduckgo" in page.warning


async def test_a_bot_check_everywhere_raises_and_is_not_an_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome = FakeChrome(
        {"duckduckgo": reading("blocked", []), "bing": reading("blocked", [])}
    )
    monkeypatch.setattr(browser_search, "_evaluate_in_new_tab", chrome)
    with pytest.raises(SearchEnginesBlocked) as caught:
        await browser_query(client(), "http://127.0.0.1:9222", "apollo ferry")
    assert caught.value.retryable
    assert [e.engine for e in caught.value.engines] == [DUCKDUCKGO.name, BING.name]


async def test_no_results_is_an_answer_and_bing_is_not_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome = FakeChrome({"duckduckgo": reading("empty", [])})
    monkeypatch.setattr(browser_search, "_evaluate_in_new_tab", chrome)
    page = await browser_query(client(), "http://127.0.0.1:9222", "zzqx moonfish")
    assert page.results == [] and chrome.opened == ["duckduckgo"]


async def test_an_offline_client_answers_from_the_cache_only() -> None:
    with pytest.raises(SourceError) as caught:
        await browser_query(client(offline=True), "http://127.0.0.1:9222", "apollo")
    assert caught.value.kind == "offline"


async def test_no_chrome_is_a_transport_error_that_names_the_fix() -> None:
    with pytest.raises(SourceError, match="remote-debugging-port") as caught:
        await browser_query(client(), "http://127.0.0.1:9", "apollo ferry")
    assert caught.value.kind == "transport"


async def test_the_web_source_searches_through_the_browser_one_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chrome = FakeChrome({"duckduckgo": reading("results", [("Apollo", DDG_FERRY)])})
    monkeypatch.setattr(browser_search, "_evaluate_in_new_tab", chrome)
    source = WebSource(browser_cdp_url="http://127.0.0.1:9222", client=client())
    found = await source.search("apollo ferry", limit=20)
    assert [c.work.url for c in found] == [FERRY]
    assert chrome.opened == ["duckduckgo"]


def test_the_web_source_needs_searxng_or_a_browser() -> None:
    with pytest.raises(SourceError, match="browser_cdp_url"):
        WebSource(client=client())
