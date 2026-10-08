"""Read a page through the Chrome that searches (Chrome DevTools Protocol).

The fallback of ``WebSource.fetch_text``: when the plain fetch fails (HTTP 403
from a site that blocks scripts, a page that needs JavaScript to show its
text), a real Chrome opens the page in a new tab, waits for the load event,
waits until the visible text stops growing, and returns it. The same CDP
mechanics as ``deep_research.browser_search._evaluate_in_new_tab`` (a tab from
``/json/new``, a websocket to it, the tab always closed); here the tab is asked
more than once, so it has its own session.

Two guards are part of the contract:

- the FINAL address the tab ended on (after redirects and scripts) must pass the
  SSRF check, else the text is discarded (``kind='refused'``);
- a short page that is a bot-check interstitial is not text
  (``kind='http'``, status 403, reason ``bot check``).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from andamentum.core import url_safety

from .base import SourceError

SOURCE = "web"
# The content type of a text read through the Chrome: how the reader tells it
# from a plainly fetched one (a cache hit on replay reports the default).
CONTENT_TYPE = "text/markdown; via=chrome-cdp"
# The whole read of one page, tab open to text. A page that has not fired its
# load event after LOAD_WAIT_S (a bot-check page that keeps loading, a script that
# never ends) is read as it stands: it may already show its text, or be the
# interstitial that is told apart below.
PAGE_LOAD_TIMEOUT_S = 15.0
LOAD_WAIT_S = 5.0
# The text is read again every POLL_INTERVAL_S until its length stops growing,
# at most SETTLE_MAX_S after the load event.
POLL_INTERVAL_S = 0.25
SETTLE_MAX_S = 3.0
# A page whose text is shorter than this and says one of BOT_CHECK_MARKERS is
# an interstitial, not the page.
BOT_CHECK_MAX_CHARS = 2000
BOT_CHECK_MARKERS = (
    "just a moment",
    "verify you are human",
    "attention required",
    "access denied",
    "enable javascript and cookies",
    "captcha",
)

_LENGTH_SCRIPT = "document.body ? document.body.innerText.length : 0"
_READ_SCRIPT = (
    "JSON.stringify({href: location.href, title: document.title, "
    "text: document.body ? document.body.innerText : ''})"
)


@dataclass(frozen=True)
class BrowserPage:
    """What the tab showed: where it ended, its title and its visible text."""

    href: str
    title: str
    text: str


def looks_like_bot_check(page: BrowserPage) -> bool:
    """A short page whose title or text carries bot-check wording."""
    if len(page.text.strip()) >= BOT_CHECK_MAX_CHARS:
        return False
    shown = f"{page.title}\n{page.text}".casefold()
    return any(marker in shown for marker in BOT_CHECK_MARKERS)


async def browser_page_text(cdp_url: str, url: str) -> BrowserPage:
    """The visible text of *url* as the Chrome at *cdp_url* shows it.

    Raises ``SourceError``: ``transport`` when Chrome does not answer or the
    page does not load, ``parse`` when the tab gave no reading, ``refused``
    when the tab ended on an address the SSRF check refuses, ``http`` (403,
    reason ``bot check``) for an interstitial.
    """
    page = await _read_in_new_tab(cdp_url, url)
    safe, why = await asyncio.to_thread(url_safety.is_safe_url, page.href)
    if not safe:
        raise SourceError(
            SOURCE,
            f"the page ended on a refused address {page.href}: {why}",
            kind="refused",
            url=url,
        )
    if looks_like_bot_check(page):
        raise SourceError(SOURCE, "bot check", kind="http", status=403, url=url)
    return page


async def _read_in_new_tab(cdp_url: str, url: str) -> BrowserPage:
    import websockets

    base = cdp_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            tab = (await http.put(f"{base}/json/new?about:blank")).json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SourceError(
            SOURCE,
            f"no Chrome answers at {cdp_url} ({type(exc).__name__}); start Chrome "
            "with --remote-debugging-port and its own --user-data-dir",
            kind="transport",
            url=cdp_url,
        ) from exc
    ids = itertools.count(1)
    try:
        async with websockets.connect(
            tab["webSocketDebuggerUrl"], max_size=None
        ) as socket:
            loaded = asyncio.Event()

            async def call(method: str, **params: object) -> dict[str, object]:
                wanted = next(ids)
                await socket.send(
                    json.dumps({"id": wanted, "method": method, "params": params})
                )
                while True:
                    message = json.loads(await socket.recv())
                    if message.get("method") == "Page.loadEventFired":
                        loaded.set()
                    if message.get("id") == wanted:
                        if "error" in message:
                            raise RuntimeError(f"{method}: {message['error']}")
                        return message.get("result") or {}

            async def evaluate(script: str) -> object:
                answer = await call(
                    "Runtime.evaluate", expression=script, returnByValue=True
                )
                result = answer.get("result")
                return result.get("value") if isinstance(result, dict) else None

            async with asyncio.timeout(PAGE_LOAD_TIMEOUT_S):
                await call("Page.enable")
                await call("Page.navigate", url=url)
                try:
                    async with asyncio.timeout(LOAD_WAIT_S):
                        while not loaded.is_set():
                            message = json.loads(await socket.recv())
                            if message.get("method") == "Page.loadEventFired":
                                loaded.set()
                except TimeoutError:
                    pass  # read what the tab shows
                await _settle(evaluate)
                raw = await evaluate(_READ_SCRIPT)
    except (OSError, TimeoutError, RuntimeError, websockets.WebSocketException) as exc:
        raise SourceError(
            SOURCE,
            f"the browser could not read {url}: {type(exc).__name__}: {exc}",
            kind="transport",
            url=url,
        ) from exc
    finally:
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                await http.get(f"{base}/json/close/{tab['id']}")
        except httpx.HTTPError:
            pass  # a tab left open costs nothing; the reading stands
    return _page_of(raw, url)


async def _settle(evaluate: Callable[[str], Awaitable[object]]) -> None:
    """Wait until the visible text's length has stopped growing (two equal
    readings in a row), at most ``SETTLE_MAX_S``."""
    deadline = time.monotonic() + SETTLE_MAX_S
    previous = -1
    while True:
        value = await evaluate(_LENGTH_SCRIPT)
        length = value if isinstance(value, int) else 0
        if length == previous or time.monotonic() >= deadline:
            return
        previous = length
        await asyncio.sleep(POLL_INTERVAL_S)


def _page_of(raw: object, url: str) -> BrowserPage:
    try:
        data = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise SourceError(
            SOURCE, f"the page {url} gave no reading", kind="parse", url=url
        )
    return BrowserPage(
        href=str(data.get("href") or url),
        title=str(data.get("title") or ""),
        text=data["text"],
    )
