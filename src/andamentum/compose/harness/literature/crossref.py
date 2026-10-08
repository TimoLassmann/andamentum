"""Crossref search and fetch as ``Work`` values.

The by-DOI fetch, the search and the Crossref-to-CSL mapping live in
``andamentum.references.crossref``; this module wraps their CSL JSON in a
``Work`` and maps ``CrossrefError`` to ``LiteratureError``.
"""

from __future__ import annotations

from andamentum.references import CrossrefError, ReferenceHttp
from andamentum.references import crossref as references_crossref

from .models import LiteratureError, Work


async def search_crossref(query: str, *, http: ReferenceHttp, limit: int) -> list[Work]:
    """The works Crossref ranks best for *query* (a title, a citation or a claim).

    The query text is sent; nothing else. A failed request raises
    ``LiteratureError`` (never an empty list).
    """
    try:
        found = await references_crossref.search_crossref(query, http=http, limit=limit)
    except CrossrefError as exc:
        raise LiteratureError(str(exc)) from exc
    return [Work(csl=csl, source="crossref") for csl in found]


async def fetch_work(doi: str, *, http: ReferenceHttp) -> Work:
    """The Crossref record of *doi* as a work (it is not yet verified).

    Raises ``LiteratureError`` when Crossref does not know the DOI or cannot be
    reached; a DOI that does not resolve is never turned into a made-up work.
    """
    try:
        csl = await references_crossref.fetch_work(doi, http=http)
    except CrossrefError as exc:
        raise LiteratureError(str(exc)) from exc
    return Work(csl=csl, source="crossref")
