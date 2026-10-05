"""Open the literature sources a verb was asked to use, and close them again.

One place builds the ``Sources`` and the ``ReferenceHttp`` that ``write``,
``find-citation`` and ``insert-citation`` share, so the three read the same
library, the same Zotero and the same project cache. Verifying a work always
asks Crossref, so there is no offline mode here; a caller that must not use
the network does not ask for literature at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from andamentum.references import ReferenceHttp

from ..kernel.store import ANSWER_MAX_AGE, HttpCache, ProjectCache
from . import Sources, ZoteroLibrary


@asynccontextmanager
async def open_literature(
    *,
    library: Path | None,
    zotero: bool,
    cache_dir: Path | None,
    mailto: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    url_check: Callable[[str], tuple[bool, str]] | None = None,
) -> AsyncIterator[tuple[Sources, ReferenceHttp]]:
    """``(sources, http)``: the user's library file, Zotero (read only) when
    asked, and Crossref through *http*, cached in *cache_dir* when given."""
    client = httpx.AsyncClient(transport=transport, timeout=20.0)
    cache = (
        ProjectCache(HttpCache(cache_dir), max_age=ANSWER_MAX_AGE)
        if cache_dir is not None
        else None
    )
    http = (
        ReferenceHttp(client=client, cache=cache, mailto=mailto, url_check=url_check)
        if url_check is not None
        else ReferenceHttp(client=client, cache=cache, mailto=mailto)
    )
    reader = ZoteroLibrary() if zotero else None
    try:
        yield Sources(library=library, zotero=reader, crossref=http), http
    finally:
        await client.aclose()
        if reader is not None:
            await reader.aclose()
