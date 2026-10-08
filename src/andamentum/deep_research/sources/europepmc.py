"""Europe PMC: biomedical abstracts, citation links and open-access full text.

It indexes PubMed, PubMed Central, preprints and more, so PubMed needs no
adapter of its own. ``search`` uses ``resultType=core`` (abstracts included)
with cursor paging. Full text is the JATS XML of the open-access subset
(``/{PMCID}/fullTextXML``, 404 when the article is not in it), read by
``jats.py``. References and citations come from ``/{source}/{id}/references``
and ``/citations`` and carry a source and id (and sometimes a DOI), not the
abstract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from pydantic import JsonValue

from andamentum.references import Work
from andamentum.references.works import (
    normalise_doi,
    normalise_pmcid,
    normalise_pmid,
)

from ..store import TTL_FULL_TEXT_S, TTL_METADATA_S, TTL_SEARCH_S, ResearchStore
from ..types import Candidate
from ._util import (
    as_int,
    as_list,
    as_mapping,
    expect_mapping,
    snippet_of,
    squash,
    year_of_date,
)
from .base import (
    DEFAULT_RESULTS_PER_QUERY,
    FROZEN_TEXT_REASON,
    FetchedText,
    SearchFilters,
    SourceClient,
    SourceError,
    ensure_text,
    resolve_client,
)
from .jats import JatsError, jats_to_text

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
NAME = "europepmc"
MAX_PAGE_SIZE = 1000
_TAGS = re.compile(r"<[^>]+>")


class EuropePmcSource:
    """Search, references, citations and open-access full text from Europe PMC."""

    name = NAME

    def __init__(
        self,
        *,
        store: ResearchStore | None = None,
        client: SourceClient | None = None,
    ) -> None:
        self._client = resolve_client(NAME, store=store, client=client)

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """Records matching *query* (Europe PMC query syntax, plain words work), at most *limit*."""
        full_query = _with_filters(query, filters)
        found: list[Candidate] = []
        cursor = "*"
        while len(found) < limit:
            body = await self._get(
                f"{API}/search",
                {
                    "query": full_query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": min(limit - len(found), MAX_PAGE_SIZE),
                    "cursorMark": cursor,
                },
                ttl_s=TTL_SEARCH_S,
            )
            results = as_list(as_mapping(body.get("resultList")).get("result"))
            for item in results:
                work = work_from_core(as_mapping(item))
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
            following = str(body.get("nextCursorMark") or "")
            if not results or not following or following == cursor:
                break
            cursor = following
        return found[:limit]

    async def references(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]:
        """The works *work* cites, as Europe PMC lists them (title, authors, year, ids)."""
        return await self._links(
            work, "references", "referenceList", "reference", limit
        )

    async def citations(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]:
        """Works that cite *work*."""
        return await self._links(work, "citations", "citationList", "citation", limit)

    async def _links(
        self, work: Work, endpoint: str, list_key: str, item_key: str, limit: int
    ) -> list[Candidate]:
        source, record_id = await self._locate(work)
        body = await self._get(
            f"{API}/{source}/{record_id}/{endpoint}",
            {"format": "json", "page": 1, "pageSize": min(limit, MAX_PAGE_SIZE)},
            ttl_s=TTL_METADATA_S,
        )
        found: list[Candidate] = []
        for item in as_list(as_mapping(body.get(list_key)).get(item_key)):
            linked = work_from_link(as_mapping(item))
            if linked is not None:
                found.append(
                    Candidate(work=linked, sources=[NAME], snippet="", rank=len(found))
                )
        return found[:limit]

    async def _locate(self, work: Work) -> tuple[str, str]:
        """The ``(source, id)`` Europe PMC knows *work* by, found by DOI when needed."""
        if work.pmid:
            return "MED", work.pmid
        if work.pmcid:
            return "PMC", work.pmcid
        if work.doi:
            body = await self._get(
                f"{API}/search",
                {
                    "query": f'DOI:"{work.doi}"',
                    "format": "json",
                    "resultType": "lite",
                    "pageSize": 1,
                },
                ttl_s=TTL_METADATA_S,
            )
            for item in as_list(as_mapping(body.get("resultList")).get("result")):
                data = as_mapping(item)
                if data.get("source") and data.get("id"):
                    return str(data["source"]), str(data["id"])
        raise SourceError(
            NAME,
            "the work is not in Europe PMC (no PMID, PMCID or a DOI it knows)",
            kind="config",
        )

    async def fetch_text(self, work: Work) -> FetchedText | None:
        """The open-access full text, or None when the article is not in the OA subset."""
        if not work.pmcid:
            return None
        url = f"{API}/{work.pmcid}/fullTextXML"
        try:
            result = await self._client.get(NAME, url, ttl_s=TTL_FULL_TEXT_S)
        except SourceError as exc:
            if exc.status in (404, 403):  # not open access, or withheld
                return None
            if exc.kind == "offline":  # the same reason whichever source holds the text
                raise SourceError(
                    NAME, FROZEN_TEXT_REASON, kind="offline", url=url
                ) from exc
            raise
        try:
            text = jats_to_text(result.body)
        except JatsError as exc:
            raise SourceError(
                NAME, f"unreadable full text: {exc}", kind="parse", url=url
            ) from exc
        ensure_text(NAME, url, text)
        self._client.extractions[url] = "jats"
        return FetchedText(
            source=NAME, url=url, text=text, content_type="application/xml"
        )

    async def _get(
        self, url: str, params: dict[str, str | int], *, ttl_s: int
    ) -> dict[str, JsonValue]:
        body = await self._client.get_json(
            NAME, url, params=params, ttl_s=ttl_s, validate=_reject_error_body
        )
        return dict(expect_mapping(NAME, body, what=url))


def _reject_error_body(body: bytes) -> None:
    """Europe PMC reports some failures as a 200 with ``errCode``; none is cached."""
    try:
        data = json.loads(body)
    except ValueError:
        return  # get_json reports a body that is not JSON
    if isinstance(data, dict) and "errCode" in data:
        raise SourceError(
            NAME, f"{data.get('errCode')}: {data.get('errMsg', '')}", kind="http"
        )


def work_from_core(data: dict[str, JsonValue] | object) -> Work | None:
    """A ``Work`` from a ``resultType=core`` record; None when it has no title."""
    item = as_mapping(data)
    title = _title(item.get("title"))
    if not title:
        return None
    pmcid = normalise_pmcid(_s(item.get("pmcid")))
    abstract = _clean(item.get("abstractText"))
    open_access = item.get("isOpenAccess") == "Y"
    journal = as_mapping(as_mapping(item.get("journalInfo")).get("journal"))
    pub_types = [
        str(t) for t in as_list(as_mapping(item.get("pubTypeList")).get("pubType"))
    ]
    return Work(
        pmid=normalise_pmid(_s(item.get("pmid"))),
        pmcid=pmcid,
        doi=normalise_doi(_s(item.get("doi"))),
        title=title,
        authors=_authors(item),
        year=year_of_date(item.get("pubYear")),
        venue=squash(journal.get("title") or item.get("journalTitle")),
        abstract=abstract,
        oa_url=f"https://europepmc.org/articles/{pmcid}"
        if open_access and pmcid
        else None,
        access="abstract" if abstract else "metadata",
        retracted="retracted" if "Retracted Publication" in pub_types else "unknown",
    )


def work_from_link(data: dict[str, JsonValue] | object) -> Work | None:
    """A ``Work`` from a reference or citation entry (no abstract); None without a title."""
    item = as_mapping(data)
    title = _title(item.get("title"))
    if not title:
        return None
    source, record_id = _s(item.get("source")), _s(item.get("id"))
    return Work(
        pmid=normalise_pmid(record_id) if source == "MED" else None,
        pmcid=normalise_pmcid(record_id) if source == "PMC" else None,
        doi=normalise_doi(_s(item.get("doi"))),
        title=title,
        authors=_names(_s(item.get("authorString"))),
        year=as_int(item.get("pubYear")),
        venue=squash(item.get("journalAbbreviation")),
    )


def _with_filters(query: str, filters: SearchFilters | None) -> str:
    if filters is None:
        return query
    parts = [f"({query})"]
    if filters.year_from is not None or filters.year_to is not None:
        low = filters.year_from if filters.year_from is not None else 1000
        high = filters.year_to if filters.year_to is not None else 9999
        parts.append(f"PUB_YEAR:[{low} TO {high}]")
    if filters.open_access_only:
        parts.append("OPEN_ACCESS:y")
    return " AND ".join(parts)


def _authors(item: Mapping[str, JsonValue]) -> list[str]:
    listed = [
        name
        for a in as_list(as_mapping(item.get("authorList")).get("author"))
        if (name := squash(as_mapping(a).get("fullName")))
    ]
    return listed or _names(_s(item.get("authorString")))


def _names(author_string: str) -> list[str]:
    return [n.strip() for n in author_string.rstrip(".").split(",") if n.strip()]


def _clean(value: object) -> str:
    """Text with markup (Europe PMC leaves ``<h4>`` and ``<i>`` in titles and abstracts) removed."""
    return squash(_TAGS.sub(" ", value)) if isinstance(value, str) else ""


def _title(value: object) -> str:
    """A title as the other sources print it: Europe PMC ends them with a full stop."""
    return _clean(value).removesuffix(".")


def _s(value: object) -> str:
    return str(value).strip() if isinstance(value, (str, int)) else ""
