"""Crossref: DOI metadata and retraction status, over ``references``.

There is no Crossref client here. ``references.ReferenceHttp`` makes the calls
(one retry, the polite ``mailto``, a cache) and ``references.lookup_retraction``
reads the update notices; this adapter fits them to the ``Source`` protocol,
shares the store's cache and spaces the requests. Crossref is for verifying a
work and finding one by its citation, not for discovery at scale, so the
engine does not list it among the scholarly sources it searches by default.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from pydantic import JsonValue

from andamentum.references import (
    ReferenceHttp,
    Work,
    WorkRetraction,
    lookup_retraction,
)
from andamentum.references.works import normalise_doi

from ..store import ResearchStore
from ..types import Candidate
from ._util import as_int, as_list, as_mapping, snippet_of, squash
from .base import (
    DEFAULT_RESULTS_PER_QUERY,
    SearchFilters,
    SourceClient,
    SourceError,
    resolve_client,
)

NAME = "crossref"
MAX_ROWS = 100
SELECT = "DOI,title,author,issued,published,container-title,type,abstract,URL"
_TAGS = re.compile(r"<[^>]+>")
_ABSTRACT_LABEL = re.compile(r"^\s*abstract\b[:.\s]*", re.IGNORECASE)
_DATE_FIELDS = ("issued", "published")


class CrossrefSource:
    """Search, DOI lookup and retraction status from Crossref."""

    name = NAME

    def __init__(
        self,
        *,
        store: ResearchStore | None = None,
        client: SourceClient | None = None,
        http: ReferenceHttp | None = None,
        mailto: str | None = None,
    ) -> None:
        self._client = resolve_client(NAME, store=store, client=client)
        self._http = http or ReferenceHttp(
            client=self._client.http,
            cache=self._client.store.reference_cache(),
            mailto=mailto,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """Works Crossref ranks best for *query*, at most *limit*."""
        params = {"query": query, "rows": str(min(limit, MAX_ROWS)), "select": SELECT}
        if flt := _filter(filters):
            params["filter"] = flt
        body = await self._call("/works", params)
        if body is None:
            raise SourceError(NAME, "HTTP 404 for the works search", status=404)
        message = as_mapping(body.get("message"))
        found: list[Candidate] = []
        for item in as_list(message.get("items")):
            work = work_from_item(as_mapping(item))
            if work is not None:
                found.append(
                    Candidate(
                        work=work,
                        sources=[NAME],
                        snippet=snippet_of(work),
                        queries=[query],
                        rank=len(found),
                    )
                )
        return found[:limit]

    async def get_work(self, doi: str) -> Candidate | None:
        """The record for *doi*, or None when Crossref does not know it."""
        clean = normalise_doi(doi)
        if clean is None:
            raise SourceError(NAME, f"not a DOI: {doi!r}", kind="config")
        body = await self._call(f"/works/{quote(clean, safe='/')}", None)
        if body is None:
            return None
        work = work_from_item(as_mapping(body.get("message")))
        if work is None:
            return None
        return Candidate(work=work, sources=[NAME], snippet=snippet_of(work))

    async def retraction_status(self, doi: str) -> WorkRetraction:
        """Whether Crossref records a retraction, concern or correction for *doi*.

        ``unknown`` when Crossref does not know the DOI; a failed call raises,
        because "could not check" must never read as "not retracted".
        """
        clean = normalise_doi(doi)
        if clean is None:
            raise SourceError(NAME, f"not a DOI: {doi!r}", kind="config")
        async with self._client.paced(NAME):
            status = await lookup_retraction(clean, http=self._http)
        if status.status == "service_error":
            raise SourceError(NAME, status.detail or "retraction lookup failed")
        if status.status == "not_found":
            return "unknown"
        kinds = {n.kind for n in status.notices}
        if "retraction" in kinds:
            return "retracted"
        if "expression_of_concern" in kinds:
            return "concern"
        if "correction" in kinds:
            return "correction"
        return "no"

    async def _call(
        self, path: str, params: dict[str, str] | None
    ) -> dict[str, JsonValue] | None:
        """One cached Crossref GET; None for a 404, which a lookup reads as "unknown"."""
        async with self._client.paced(NAME):
            result = await self._http.crossref(path, params)
        if isinstance(result, str):
            raise SourceError(NAME, result, url=path)
        status, body = result
        return None if status == 404 else body


def work_from_item(item: dict[str, JsonValue] | object) -> Work | None:
    """A ``Work`` from a Crossref work object; None without a DOI and title."""
    data = as_mapping(item)
    doi = normalise_doi(str(data.get("DOI") or ""))
    titles = as_list(data.get("title"))
    title = squash(titles[0]) if titles else ""
    if doi is None or not title:
        return None
    venues = as_list(data.get("container-title"))
    abstract = _abstract(data.get("abstract"))
    return Work(
        doi=doi,
        title=title,
        authors=_authors(data),
        year=_year(data),
        venue=squash(venues[0]) if venues else "",
        abstract=abstract,
        access="abstract" if abstract else "metadata",
    )


def _authors(data: dict[str, JsonValue] | object) -> list[str]:
    names: list[str] = []
    for a in as_list(as_mapping(data).get("author")):
        person = as_mapping(a)
        name = squash(
            f"{person.get('given', '')} {person.get('family', '')}"
        ) or squash(person.get("name"))
        if name:
            names.append(name)
    return names


def _year(data: dict[str, JsonValue] | object) -> int | None:
    for field in _DATE_FIELDS:
        parts = as_list(as_mapping(as_mapping(data).get(field)).get("date-parts"))
        first = as_list(parts[0]) if parts else []
        if first and (year := as_int(first[0])):
            return year
    return None


def _abstract(value: object) -> str:
    """Crossref abstracts are JATS fragments (``<jats:p>``); this keeps the words."""
    if not isinstance(value, str):
        return ""
    return _ABSTRACT_LABEL.sub("", squash(_TAGS.sub(" ", value)))


def _filter(filters: SearchFilters | None) -> str:
    if filters is None:
        return ""
    parts: list[str] = []
    if filters.year_from is not None:
        parts.append(f"from-pub-date:{filters.year_from}")
    if filters.year_to is not None:
        parts.append(f"until-pub-date:{filters.year_to}")
    return ",".join(parts)
