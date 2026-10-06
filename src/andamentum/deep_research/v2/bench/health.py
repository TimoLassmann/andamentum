"""Is the SearxNG instance answering? Checked before anything web is frozen or run live.

A search that returns nothing while the engines behind it are unresponsive says
nothing about the web; a snapshot taken then would freeze an empty answer and
every arm would be measured against it. ``check_searxng`` asks one real query
and reports what came back, so the caller can refuse.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel

from andamentum.deep_research.searxng_search import parse_unresponsive

PROBE_QUERY = "Mount Kilimanjaro height"
PROBE_TIMEOUT_S = 30.0


class SearxHealth(BaseModel):
    """What one probe query to SearxNG showed."""

    url: str
    ok: bool
    n_results: int = 0
    unresponsive: list[str] = []
    detail: str = ""

    def explain(self) -> str:
        if self.ok:
            return f"SearxNG at {self.url} answered with {self.n_results} results"
        return f"SearxNG at {self.url} is not usable: {self.detail}"


class SearxDown(RuntimeError):
    """SearxNG is not answering usefully; web work must not start."""


def _engine_names(raw: object) -> list[str]:
    return [str(engine) for engine in parse_unresponsive(raw)]


async def check_searxng(
    url: str, *, client: httpx.AsyncClient | None = None
) -> SearxHealth:
    """One probe search. Never raises for an unreachable or broken server."""
    endpoint = f"{url.rstrip('/')}/search?" + urlencode(
        {"q": PROBE_QUERY, "format": "json", "pageno": 1}
    )
    owned = client is None
    http = client or httpx.AsyncClient(timeout=PROBE_TIMEOUT_S)
    try:
        try:
            response = await http.get(endpoint)
        except httpx.HTTPError as error:
            return SearxHealth(
                url=url, ok=False, detail=f"{type(error).__name__}: {error}"
            )
    finally:
        if owned:
            await http.aclose()
    if response.status_code != 200:
        return SearxHealth(url=url, ok=False, detail=f"HTTP {response.status_code}")
    try:
        data = json.loads(response.content)
    except ValueError:
        return SearxHealth(url=url, ok=False, detail="the reply is not JSON")
    if not isinstance(data, dict):
        return SearxHealth(url=url, ok=False, detail="the reply is not a JSON object")
    results = data.get("results")
    n = len(results) if isinstance(results, list) else 0
    down = _engine_names(data.get("unresponsive_engines"))
    if n == 0:
        reason = "no results"
        if down:
            reason += "; engines did not answer: " + ", ".join(down)
        return SearxHealth(url=url, ok=False, unresponsive=down, detail=reason)
    return SearxHealth(url=url, ok=True, n_results=n, unresponsive=down)


async def require_searxng(
    url: str | None, *, client: httpx.AsyncClient | None = None
) -> SearxHealth:
    """The health of SearxNG, or ``SearxDown`` when it cannot be used."""
    if not url:
        raise SearxDown("web work needs --searxng-url")
    health = await check_searxng(url, client=client)
    if not health.ok:
        raise SearxDown(health.explain())
    return health
