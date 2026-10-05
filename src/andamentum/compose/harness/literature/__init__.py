"""Find works to cite and verify them.

``find_works(query, sources=..., limit=...)`` looks, in order, in the user's
library file (CSL JSON or BibTeX), in Zotero's local API (read only) and in
Crossref, and returns candidate :class:`Work` objects. ``resolve(work, http=...)``
turns one into a :class:`VerifiedWork` or an :class:`Unverified`; only a
verified work is ever cited. See ``harness/CLAUDE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from andamentum.references import ReferenceHttp

from ..kernel.csl import work_key
from .crossref import fetch_work, search_crossref
from .library import read_library
from .models import LiteratureError, Unverified, VerifiedWork, Work
from .resolve import resolve
from .search import rank
from .zotero import ZOTERO_API, ZoteroLibrary


@dataclass
class Sources:
    """Where to look. A source that is set is used; one left None is not."""

    library: Path | None = None
    zotero: ZoteroLibrary | None = None
    crossref: ReferenceHttp | None = None


async def find_works(query: str, *, sources: Sources, limit: int = 10) -> list[Work]:
    """Candidate works for *query* (a title, a topic or a claim), best first.

    The user's library comes first, then Zotero, then Crossref; Crossref is
    asked only when they give fewer than *limit* works. A work seen twice
    (same DOI or title) is returned once, from the earliest source. A source
    that was asked for and cannot answer raises ``LiteratureError``.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    found: dict[str, Work] = {}

    def add(works: list[Work]) -> None:
        for work in works:
            found.setdefault(work_key(work.csl), work)

    if sources.library is not None:
        add(rank(read_library(sources.library), query, limit=limit))
    if sources.zotero is not None and len(found) < limit:
        add(await sources.zotero.search(query, limit=limit))
    if sources.crossref is not None and len(found) < limit:
        # Crossref ranks by its own relevance and always returns something; a
        # record that shares fewer than two of the query's words is not evidence.
        add(
            rank(
                await search_crossref(query, http=sources.crossref, limit=limit),
                query,
                limit=limit,
            )
        )
    return list(found.values())[:limit]


__all__ = [
    "ZOTERO_API",
    "LiteratureError",
    "Sources",
    "Unverified",
    "VerifiedWork",
    "Work",
    "ZoteroLibrary",
    "fetch_work",
    "find_works",
    "read_library",
    "resolve",
    "search_crossref",
]
