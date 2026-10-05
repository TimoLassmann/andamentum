"""Read-only client for Zotero's local API.

Zotero 7 serves the user's library at ``http://127.0.0.1:23119/api/`` when the
setting "Allow other applications on this computer to communicate with Zotero"
is on. This client only reads: there is no method that writes, and the harness
never writes to Zotero (a ``--zotero-write`` flag, if it ever exists, would be
a separate, explicit piece of code).

The whole library is read once (pages of 100 items with their CSL JSON) and
kept for the life of the object; ranking against a query is local. Notes,
attachments and annotations are left out. A library over ``max_items`` raises
rather than being cut short. Each work's ``uri`` is the item URI Zotero uses in
its citation fields. It comes from the CSL ``id`` when that is an item URI, else
from the entry's ``library`` (a synced user or group library) and key, else
from its links. A local library that was never synced has none: ``uri`` is then
None and the citation carries only the item data, which Zotero treats as an
embedded item. A URI is never made up.
"""

from __future__ import annotations

import httpx

from .models import LiteratureError, Work
from .search import rank

ZOTERO_API = "http://127.0.0.1:23119/api/"
_URI_PREFIX = "http://zotero.org/"
_SKIP = frozenset({"attachment", "note", "annotation"})
_PAGE = 100


class ZoteroLibrary:
    """The user's Zotero library, read through the local API."""

    def __init__(
        self,
        *,
        base_url: str = ZOTERO_API,
        client: httpx.AsyncClient | None = None,
        max_items: int = 5000,
    ) -> None:
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=20.0)
        self.max_items = max_items
        self._works: list[Work] | None = None

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def works(self) -> list[Work]:
        """Every bibliographic item in the library."""
        if self._works is None:
            self._works = await self._read()
        return self._works

    async def search(self, query: str, *, limit: int) -> list[Work]:
        return rank(await self.works(), query, limit=limit)

    async def _read(self) -> list[Work]:
        works: list[Work] = []
        start = 0
        while True:
            page = await self._page(start)
            for entry in page:
                work = _to_work(entry)
                if work is not None:
                    works.append(work)
            start += _PAGE
            if len(page) < _PAGE:
                return works
            if start >= self.max_items:
                raise LiteratureError(
                    f"the Zotero library has more than {self.max_items} items; "
                    "use --library with an export of the relevant collection"
                )

    async def _page(self, start: int) -> list[dict[str, object]]:
        url = f"{self.base_url}users/0/items"
        params = {
            "format": "json",
            "include": "data,csljson",
            "limit": str(_PAGE),
            "start": str(start),
        }
        try:
            response = await self.client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise LiteratureError(
                f"Zotero is not reachable at {self.base_url} ({type(exc).__name__}); "
                "start Zotero and enable 'Allow other applications on this "
                "computer to communicate with Zotero'"
            ) from exc
        if response.status_code == 403:
            raise LiteratureError(
                "Zotero refused the request; enable 'Allow other applications on "
                "this computer to communicate with Zotero' in its settings"
            )
        if response.status_code != 200:
            raise LiteratureError(f"Zotero answered {response.status_code} for {url}")
        try:
            body = response.json()
        except ValueError as exc:
            raise LiteratureError(
                f"Zotero sent something that is not JSON: {exc}"
            ) from exc
        if not isinstance(body, list):
            raise LiteratureError("Zotero did not answer with a list of items")
        return [e for e in body if isinstance(e, dict)]


def _to_work(entry: dict[str, object]) -> Work | None:
    data = entry.get("data")
    if isinstance(data, dict) and data.get("itemType") in _SKIP:
        return None
    csl = entry.get("csljson")
    if not isinstance(csl, dict):
        raise LiteratureError(
            f"Zotero item {entry.get('key', '?')!r} came without CSL JSON"
        )
    item = {str(k): v for k, v in csl.items()}
    uri = _item_uri(entry, item)
    if not (item.get("title") or item.get("DOI")):
        return None
    return Work(csl=item, source="zotero", uri=uri)


def _item_uri(entry: dict[str, object], csl: dict[str, object]) -> str | None:
    """The item's real Zotero URI, or None when the API does not reveal one."""
    identifier = str(csl.get("id", ""))
    if identifier.startswith(_URI_PREFIX):
        return identifier
    key = str(entry.get("key", ""))
    library = entry.get("library")
    if key and isinstance(library, dict):
        number = library.get("id")
        if isinstance(number, int) and number > 0:
            kind = "groups" if library.get("type") == "group" else "users"
            return f"{_URI_PREFIX}{kind}/{number}/items/{key}"
    links = entry.get("links")
    if isinstance(links, dict):
        for name in ("self", "alternate"):
            link = links.get(name)
            href = link.get("href") if isinstance(link, dict) else None
            if isinstance(href, str) and href.startswith(_URI_PREFIX):
                return href
    return None
