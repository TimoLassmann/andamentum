"""Semantic Scholar (optional): search plus references and citations.

The Graph API works without a key, but keyless callers share one pool and are
often answered 429; a free key (``x-api-key``) gives one request a second. The
key is an explicit argument, sent as a header and never cached or keyed.

Licence: the API's terms forbid redistributing its data. Every work from here
carries ``licence='no_redistribute'`` and every cache row of its answers is
tagged the same, so ``ResearchStore.iter_cache`` never exports it and
``Work.to_csl`` omits its abstract.

A paper whose publisher withheld its reference list comes back with
``data: null`` and a notice; that is "no references available", an empty list,
not an error.
"""

from __future__ import annotations

from urllib.parse import quote

from pydantic import JsonValue

from andamentum.references import Work
from andamentum.references.works import (
    LICENCE_NO_REDISTRIBUTE,
    normalise_arxiv_id,
    normalise_doi,
    normalise_pmcid,
    normalise_pmid,
    normalise_s2_id,
)

from ..store import TTL_METADATA_S, TTL_SEARCH_S, ResearchStore
from ..types import Candidate
from ._util import as_list, as_mapping, expect_mapping, snippet_of, squash
from .base import (
    DEFAULT_RESULTS_PER_QUERY,
    FetchedText,
    SearchFilters,
    SourceClient,
    SourceError,
    resolve_client,
)

API = "https://api.semanticscholar.org/graph/v1"
NAME = "semantic_scholar"
FIELDS = ",".join(
    (
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "year",
        "venue",
        "authors",
        "openAccessPdf",
        "isOpenAccess",
    )
)
MAX_PAGE = 100  # the search endpoint's limit per page
MAX_LINKS = 1000  # the references / citations endpoints' limit per page


class SemanticScholarSource:
    """Search, references and citations from Semantic Scholar."""

    name = NAME

    def __init__(
        self,
        *,
        store: ResearchStore | None = None,
        client: SourceClient | None = None,
        api_key: str | None = None,
        tdm_allowed_hosts: frozenset[str] = frozenset(),
    ) -> None:
        self._client = resolve_client(NAME, store=store, client=client)
        self._headers = {"x-api-key": api_key} if api_key else {}
        self._tdm = tdm_allowed_hosts

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """Papers matching *query*, best first, at most *limit*."""
        found: list[Candidate] = []
        offset = 0
        while len(found) < limit:
            params: dict[str, str | int] = {
                "query": query,
                "fields": FIELDS,
                "limit": min(limit - len(found), MAX_PAGE),
                "offset": offset,
            }
            if year := _years(filters):
                params["year"] = year
            if filters is not None and filters.open_access_only:
                params["openAccessPdf"] = ""
            body = await self._get(f"{API}/paper/search", params, ttl_s=TTL_SEARCH_S)
            page = as_list(body.get("data"))
            for item in page:
                work = work_from_paper(as_mapping(item))
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
            if not page or body.get("next") is None:
                break
            offset = int(str(body["next"]))
        return found[:limit]

    async def references(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]:
        """The papers *work* cites."""
        return await self._links(work, "references", "citedPaper", limit)

    async def citations(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]:
        """The papers that cite *work*."""
        return await self._links(work, "citations", "citingPaper", limit)

    async def _links(
        self, work: Work, endpoint: str, key: str, limit: int
    ) -> list[Candidate]:
        paper = _paper_id(work)
        body = await self._get(
            f"{API}/paper/{quote(paper, safe=':')}/{endpoint}",
            {"fields": FIELDS, "limit": min(limit, MAX_LINKS)},
            ttl_s=TTL_METADATA_S,
        )
        found: list[Candidate] = []
        for item in as_list(body.get("data")):  # None when the publisher withheld it
            linked = work_from_paper(as_mapping(as_mapping(item).get(key)))
            if linked is not None:
                found.append(
                    Candidate(
                        work=linked,
                        sources=[NAME],
                        snippet=snippet_of(linked),
                        rank=len(found),
                    )
                )
        return found[:limit]

    async def fetch_text(self, work: Work) -> FetchedText | None:
        """The text at the open-access PDF Semantic Scholar names, or None."""
        if not work.oa_url:
            return None
        return await self._client.fetch_document(
            NAME, work.oa_url, tdm_allowed_hosts=self._tdm
        )

    async def _get(
        self, url: str, params: dict[str, str | int], *, ttl_s: int
    ) -> dict[str, JsonValue]:
        body = await self._client.get_json(
            NAME,
            url,
            params=params,
            headers=self._headers,
            ttl_s=ttl_s,
            licence=LICENCE_NO_REDISTRIBUTE,
        )
        return dict(expect_mapping(NAME, body, what=url))


def work_from_paper(data: dict[str, JsonValue] | object) -> Work | None:
    """A ``Work`` from a Graph API paper object; None without a title or paper id."""
    paper = as_mapping(data)
    title = squash(paper.get("title"))
    s2_id = normalise_s2_id(str(paper.get("paperId") or ""))
    if not title or s2_id is None:
        return None
    external = as_mapping(paper.get("externalIds"))
    pdf = as_mapping(paper.get("openAccessPdf"))
    abstract = squash(paper.get("abstract"))
    year = paper.get("year")
    return Work(
        s2_id=s2_id,
        doi=normalise_doi(_s(external.get("DOI"))),
        arxiv_id=normalise_arxiv_id(_s(external.get("ArXiv"))),
        pmid=normalise_pmid(_s(external.get("PubMed"))),
        pmcid=_pmcid(external.get("PubMedCentral")),
        title=title,
        authors=[
            name
            for a in as_list(paper.get("authors"))
            if (name := squash(as_mapping(a).get("name")))
        ],
        year=year if isinstance(year, int) and not isinstance(year, bool) else None,
        venue=squash(paper.get("venue")),
        abstract=abstract,
        oa_url=_s(pdf.get("url")) or None,
        access="abstract" if abstract else "metadata",
        licence=LICENCE_NO_REDISTRIBUTE,
    )


def _paper_id(work: Work) -> str:
    """The id form the API accepts, from the strongest identifier the work holds."""
    if work.s2_id:
        return work.s2_id
    if work.doi:
        return f"DOI:{work.doi}"
    if work.arxiv_id:
        return f"ARXIV:{work.arxiv_id}"
    if work.pmid:
        return f"PMID:{work.pmid}"
    if work.pmcid:
        return f"PMCID:{work.pmcid}"
    raise SourceError(
        NAME,
        "cannot place the work in Semantic Scholar: no id, DOI, arXiv id, PMID or PMCID",
        kind="config",
    )


def _years(filters: SearchFilters | None) -> str:
    if filters is None or (filters.year_from is None and filters.year_to is None):
        return ""
    low = "" if filters.year_from is None else str(filters.year_from)
    high = "" if filters.year_to is None else str(filters.year_to)
    return low if low == high else f"{low}-{high}"


def _pmcid(value: object) -> str | None:
    text = _s(value)
    if not text:
        return None
    return normalise_pmcid(text if text.upper().startswith("PMC") else f"PMC{text}")


def _s(value: object) -> str:
    return str(value).strip() if isinstance(value, (str, int)) else ""
