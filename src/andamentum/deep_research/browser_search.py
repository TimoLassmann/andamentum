"""Web search through a real Chrome, over the Chrome DevTools Protocol (CDP).

The alternative to SearxNG when its upstream engines suspend it: the isolated
Chrome (``deep_research.browser``, a podman container started for the run) opens
the search page of an engine, and one JavaScript expression reads the result list.
No language model is involved and nothing is clicked: a search page has a
fixed structure, so reading it is code.

Every browser search of the package goes through :func:`browser_query`, which
returns the same :class:`SearxPage` as ``searxng_query``, so the web source
reads the results of either unchanged:

- DuckDuckGo's HTML page is asked first; when it shows a bot check instead of
  results, Bing is asked, and the page names DuckDuckGo as unresponsive (its
  ``warning`` says so);
- a page with neither a result list nor the engine's own "no results" note is
  a bot check, never an empty result; when every engine shows one,
  :class:`SearchEnginesBlocked` is raised (``retryable``), as for SearxNG;
- queries are spaced by the shared client's limiter under the same key as
  SearxNG (``SEARXNG_SOURCE``), and an answer is cached for
  ``SEARXNG_CACHE_TTL_S`` in the client's store (keyed by the engine's URL);
- an offline client answers from the cache only;
- one page of results (about ten) per query: the engines' further pages need
  a form or a session, and one page is what a person reads.

Each page is opened in a new browser context and disposed after reading. ``websockets`` is
the only library used (already installed with the package).
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qs, quote_plus, urlparse

from pydantic import JsonValue

from .browser.cdp import CdpError, isolated_page
from .searxng_search import (
    SEARXNG_CACHE_TTL_S,
    SEARXNG_SOURCE,
    SearchEnginesBlocked,
    SearxPage,
    UnresponsiveEngine,
)

if TYPE_CHECKING:
    from .sources.base import SourceClient

PAGE_LOAD_TIMEOUT_S = 30.0  # a search page that has not loaded by then is a failure
BOT_CHECK = "captcha or bot check: no result list on the page"

PageState = Literal["results", "empty", "blocked"]


@dataclass(frozen=True)
class Engine:
    """A search engine's HTML page and the script that reads it."""

    name: str
    url_prefix: str
    script: str

    def url(self, query: str) -> str:
        return self.url_prefix + quote_plus(query)


def _reader(*, rows: str, link: str, snippet: str, empty: str, skip: str = "") -> str:
    """A script that returns ``{state, rows: [{title, href, snippet}]}`` as JSON.
    ``innerText`` is empty for a result a headless Chrome has not laid out (Bing's
    lower rows), so ``textContent`` stands in."""
    keep = f"!r.matches({json.dumps(skip)})" if skip else "true"
    return f"""JSON.stringify((() => {{
  const rows = [...document.querySelectorAll({json.dumps(rows)})].filter(r => {keep})
    .map(r => {{
      const a = r.querySelector({json.dumps(link)});
      const s = r.querySelector({json.dumps(snippet)});
      return a ? {{title: a.innerText || a.textContent, href: a.href,
        snippet: s ? (s.innerText || s.textContent) : ''}} : null;
    }}).filter(Boolean);
  const state = rows.length ? 'results'
    : (document.querySelector({json.dumps(empty)}) ? 'empty' : 'blocked');
  return {{state, rows}};
}})())"""


DUCKDUCKGO = Engine(
    name="duckduckgo",
    url_prefix="https://html.duckduckgo.com/html/?q=",
    script=_reader(
        rows=".result",
        link="a.result__a",
        snippet=".result__snippet",
        empty=".no-results",
        skip=".result--ad",
    ),
)
BING = Engine(
    name="bing",
    url_prefix="https://www.bing.com/search?q=",
    script=_reader(
        rows="li.b_algo", link="h2 a", snippet=".b_caption p, p", empty=".b_no"
    ),
)
ENGINES: tuple[Engine, ...] = (DUCKDUCKGO, BING)


def unwrap(href: str) -> str:
    """The target of an engine's redirect link (DuckDuckGo ``/l/?uddg=``, Bing
    ``/ck/a?...&u=a1<base64url>``); any other link as it is."""
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    host = parsed.hostname or ""
    if host.endswith("duckduckgo.com") and query.get("uddg"):
        return query["uddg"][0]
    if host.endswith("bing.com") and parsed.path.startswith("/ck/"):
        encoded = (query.get("u") or [""])[0]
        if encoded.startswith("a1"):
            data = encoded[2:]
            try:
                target = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
                return target.decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return href
    return href


def read_page(raw: str) -> tuple[PageState, list[dict[str, JsonValue]]]:
    """The state and the results (``url``, ``title``, ``content``) of a script's
    answer. Rows without a usable link are left out; an answer that is not the
    script's JSON raises ``ValueError``."""
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("state") not in (
        "results",
        "empty",
        "blocked",
    ):
        raise ValueError("not a search page reading")
    results: list[dict[str, JsonValue]] = []
    for row in data.get("rows") or []:
        if not isinstance(row, dict):
            continue
        url = unwrap(str(row.get("href") or ""))
        if not url.startswith(("http://", "https://")):
            continue
        results.append(
            {
                "url": url,
                "title": str(row.get("title") or "").strip(),
                "content": str(row.get("snippet") or "").strip(),
            }
        )
    state: PageState = data["state"]
    if state == "results" and not results:
        state = "blocked"
    return state, results


async def _evaluate_in_new_tab(
    cdp_url: str, *, url: str, script: str, source: str = SEARXNG_SOURCE
) -> str:
    """Open *url* in a new tab of a new browser context of the Chrome at
    *cdp_url* (cookies and storage die with it, downloads are denied), run
    *script* once the page has loaded, close the tab and return the script's
    string value. A failure is a ``SourceError`` named *source*."""
    import websockets

    from .sources.base import SourceError

    try:
        async with isolated_page(cdp_url) as page:
            async with asyncio.timeout(PAGE_LOAD_TIMEOUT_S):
                await page.call("Page.enable")
                await page.call("Page.navigate", url=url)
                await page.wait_loaded()
                value = await page.evaluate(script)
    except CdpError as exc:
        raise SourceError(
            source,
            f"the browser search page {url} failed: {exc}",
            kind="transport",
            url=url,
        ) from exc
    except (OSError, TimeoutError, websockets.WebSocketException) as exc:
        raise SourceError(
            source,
            f"the browser search page {url} failed: {type(exc).__name__}: {exc}",
            kind="transport",
            url=url,
        ) from exc
    if not isinstance(value, str):
        raise SourceError(
            source,
            f"the search page {url} gave no reading",
            kind="parse",
            url=url,
        )
    return value


async def browser_query(
    client: SourceClient,
    cdp_url: str,
    query: str,
    *,
    cache: bool = True,
) -> SearxPage:
    """Search *query* in the Chrome at *cdp_url* (its address; the isolated
    Chrome's session provides it) and return the page.

    Raises :class:`SearchEnginesBlocked` when every engine showed a bot check,
    and ``SourceError`` when Chrome does not answer, a page fails to load or
    its reading is not usable (``kind='offline'`` for an offline client's miss).
    """
    from .sources.base import SourceError
    from .store import cache_key

    unresponsive: list[UnresponsiveEngine] = []
    use_cache = cache or client.offline
    for engine in ENGINES:
        url = engine.url(query)
        key = cache_key(SEARXNG_SOURCE, url, {})
        cached = (
            client.store.cache_get(key, any_age=client.offline) if use_cache else None
        )
        if cached is not None:
            _state, results = read_page(cached.body.decode("utf-8"))
            return SearxPage(
                results=results, unresponsive=tuple(unresponsive), from_cache=True
            )
        if client.offline:
            raise SourceError(
                SEARXNG_SOURCE, "not in the cache (offline)", kind="offline", url=url
            )
        async with client.paced(SEARXNG_SOURCE):
            raw = await _evaluate_in_new_tab(cdp_url, url=url, script=engine.script)
        try:
            state, results = read_page(raw)
        except ValueError as exc:
            raise SourceError(
                SEARXNG_SOURCE,
                f"the {engine.name} page for {query!r} gave no reading ({exc})",
                kind="parse",
                url=url,
            ) from exc
        if state == "blocked":
            unresponsive.append(UnresponsiveEngine(engine.name, BOT_CHECK))
            continue
        if use_cache:
            client.store.cache_put(
                key,
                source=SEARXNG_SOURCE,
                url=url,
                status=200,
                body=raw.encode("utf-8"),
                ttl_s=SEARXNG_CACHE_TTL_S,
            )
        return SearxPage(results=results, unresponsive=tuple(unresponsive))
    raise SearchEnginesBlocked(tuple(unresponsive))
