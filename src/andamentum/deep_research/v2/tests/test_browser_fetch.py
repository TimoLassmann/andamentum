"""Reading a page through Chrome over CDP: the tab session, the settle wait, the
bot-check and SSRF guards. The Chrome is a stub: no browser, no socket, no
network."""

from __future__ import annotations

import json

import httpx
import pytest
import websockets

from andamentum.core import url_safety

from ..sources import browser_fetch
from ..sources.base import SourceError
from ..sources.browser_fetch import BrowserPage, browser_page_text, looks_like_bot_check

CDP = "http://127.0.0.1:9222"
URL = "https://docs.example.net/apollo"
BODY = "Apollo lattices are sampled sparsely in practice. " * 12


class FakeChrome:
    """What the stub Chrome does: tabs, a socket per tab, a page to show."""

    def __init__(
        self,
        *,
        text: str = BODY,
        title: str = "Apollo",
        href: str = URL,
        lengths: list[int] | None = None,
        reading: object = None,
        socket_error: Exception | None = None,
    ) -> None:
        self.page = {"href": href, "title": title, "text": text}
        self.reading = json.dumps(self.page) if reading is None else reading
        self.lengths = list(lengths or [len(text), len(text)])
        self.socket_error = socket_error
        self.methods: list[str] = []
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.navigated: list[str] = []

    # -- the HTTP side (/json/new, /json/close) --

    def http(self, **_kwargs: object) -> FakeHttp:
        return FakeHttp(self)

    # -- the websocket side --

    def connect(self, ws_url: str, **_kwargs: object) -> FakeSocket:
        if self.socket_error is not None:
            raise self.socket_error
        return FakeSocket(self)


class FakeResponse:
    def __init__(self, data: object) -> None:
        self._data = data

    def json(self) -> object:
        return self._data


class FakeHttp:
    def __init__(self, chrome: FakeChrome) -> None:
        self.chrome = chrome

    async def __aenter__(self) -> FakeHttp:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def put(self, url: str) -> FakeResponse:
        tab = f"tab{len(self.chrome.opened) + 1}"
        self.chrome.opened.append(tab)
        return FakeResponse({"id": tab, "webSocketDebuggerUrl": f"ws://stub/{tab}"})

    async def get(self, url: str) -> FakeResponse:
        self.chrome.closed.append(url.rsplit("/", 1)[-1])
        return FakeResponse("Target is closing")


class FakeSocket:
    def __init__(self, chrome: FakeChrome) -> None:
        self.chrome = chrome
        self.inbox: list[str] = []

    async def __aenter__(self) -> FakeSocket:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        method, params = message["method"], message["params"]
        self.chrome.methods.append(method)
        value: object = None
        if method == "Page.navigate":
            self.chrome.navigated.append(params["url"])
        if method == "Runtime.evaluate":
            if params["expression"] == browser_fetch._LENGTH_SCRIPT:
                lengths = self.chrome.lengths
                value = lengths.pop(0) if len(lengths) > 1 else lengths[0]
            else:
                value = self.chrome.reading
        result = {"result": {"value": value}} if method == "Runtime.evaluate" else {}
        self.inbox.append(json.dumps({"id": message["id"], "result": result}))
        if method == "Page.navigate":
            self.inbox.append(json.dumps({"method": "Page.loadEventFired"}))

    async def recv(self) -> str:
        return self.inbox.pop(0)


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeChrome) -> FakeChrome:
    monkeypatch.setattr(browser_fetch.httpx, "AsyncClient", fake.http)
    monkeypatch.setattr(websockets, "connect", fake.connect)
    monkeypatch.setattr(browser_fetch, "POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        url_safety,
        "is_safe_url",
        lambda url, **_kw: (
            (False, "private address") if "internal.example" in url else (True, "")
        ),
    )
    return fake


async def test_the_page_text_is_read_after_the_load_event_and_the_tab_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install(monkeypatch, FakeChrome(lengths=[40, 400, len(BODY), len(BODY)]))
    page = await browser_page_text(CDP, URL)
    assert page == BrowserPage(href=URL, title="Apollo", text=BODY)
    assert fake.navigated == [URL]
    # enable, navigate, then the length polled until it stopped growing, then the read
    assert fake.methods[:2] == ["Page.enable", "Page.navigate"]
    assert fake.methods[2:] == ["Runtime.evaluate"] * 5
    assert fake.opened == ["tab1"] and fake.closed == ["tab1"]


async def test_the_wait_for_the_text_to_stop_growing_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    growing = list(range(1, 10_000))
    fake = install(monkeypatch, FakeChrome(lengths=growing))
    monkeypatch.setattr(browser_fetch, "SETTLE_MAX_S", 0.0)
    page = await browser_page_text(CDP, URL)
    assert page.text == BODY
    assert fake.methods.count("Runtime.evaluate") == 2  # one length, one read


async def test_a_short_bot_check_page_is_a_403_not_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install(
        monkeypatch,
        FakeChrome(
            text="Checking your browser before accessing.", title="Just a moment..."
        ),
    )
    with pytest.raises(SourceError) as caught:
        await browser_page_text(CDP, URL)
    assert caught.value.kind == "http" and caught.value.status == 403
    assert caught.value.reason == "bot check"
    assert fake.closed == ["tab1"]


def test_bot_check_wording_in_a_long_page_is_just_a_page() -> None:
    long_page = BrowserPage(href=URL, title="Notes", text="We solved a captcha. " * 200)
    assert not looks_like_bot_check(long_page)
    for marker in browser_fetch.BOT_CHECK_MARKERS:
        short = BrowserPage(href=URL, title="", text=f"Please {marker.upper()} now.")
        assert looks_like_bot_check(short), marker
    assert not looks_like_bot_check(
        BrowserPage(href=URL, title="Apollo", text="Short.")
    )


async def test_a_final_address_that_fails_the_ssrf_check_discards_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install(monkeypatch, FakeChrome(href="http://internal.example/admin"))
    with pytest.raises(SourceError) as caught:
        await browser_page_text(CDP, URL)
    assert caught.value.kind == "refused" and "private address" in caught.value.reason
    assert fake.closed == ["tab1"]


async def test_no_chrome_is_a_transport_error_that_names_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**_kwargs: object) -> object:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(browser_fetch.httpx, "AsyncClient", refuse)
    with pytest.raises(SourceError, match="remote-debugging-port") as caught:
        await browser_page_text(CDP, URL)
    assert caught.value.kind == "transport"


async def test_a_socket_failure_is_a_transport_error_and_the_tab_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install(monkeypatch, FakeChrome(socket_error=OSError("connection reset")))
    with pytest.raises(SourceError) as caught:
        await browser_page_text(CDP, URL)
    assert (
        caught.value.kind == "transport" and "connection reset" in caught.value.reason
    )
    assert caught.value.source == "web"
    assert fake.closed == ["tab1"]


async def test_a_reading_that_is_not_the_scripts_json_is_a_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install(monkeypatch, FakeChrome(reading="not json"))
    with pytest.raises(SourceError) as caught:
        await browser_page_text(CDP, URL)
    assert caught.value.kind == "parse" and fake.closed == ["tab1"]
