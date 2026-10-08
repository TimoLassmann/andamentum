"""OpenAlex: the general scholarly backbone, every discipline, CC0 metadata.

Search is keyword (``search.title_abstract_keywords``, stemmed, boolean
operators allowed) or semantic (``search.semantic``, embeddings; it cannot be
combined with another search and is slow and sometimes times out, so a failure
is an error the engine records). References are the work's ``referenced_works``
fetched in batches by id; citations are the ``cites:`` filter.

Since 2026 OpenAlex meters by credits, not by request: a search costs 10
credits, a filtered list 1, a lookup of one work by id nothing, and a keyless
caller has 1000 a day (a free key raises it tenfold). The cost and the budget
are read from the ``x-ratelimit-*`` response headers by ``SourceClient``;
when the budget is spent the next call raises ``rate_limited`` instead of
sleeping until the reset. Answers are cached, so a repeated query costs nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal
from urllib.parse import quote

from pydantic import JsonValue

from andamentum.references import Work
from andamentum.references.works import (
    canonical_url,
    normalise_doi,
    normalise_openalex_id,
    normalise_pmcid,
    normalise_pmid,
)

from ..store import TTL_METADATA_S, TTL_SEARCH_S, ResearchStore
from ..types import Candidate
from ._util import (
    as_list,
    as_mapping,
    expect_mapping,
    snippet_of,
    squash,
    year_of_date,
)
from .base import (
    DEFAULT_RESULTS_PER_QUERY,
    FetchedText,
    SearchFilters,
    SourceClient,
    SourceError,
    resolve_client,
)

API = "https://api.openalex.org"
NAME = "openalex"
MAX_PER_PAGE = 100
SEMANTIC_MAX = 50  # the largest page the semantic endpoint was observed to answer
ID_BATCH = 50  # ids per ``openalex:W1|W2`` filter, to keep the URL short
SELECT = ",".join(
    (
        "id",
        "doi",
        "title",
        "publication_year",
        "primary_location",
        "open_access",
        "type",
        "ids",
        "is_retracted",
        "authorships",
        "abstract_inverted_index",
    )
)


class OpenAlexSource:
    """Search, references, citations and open-access text from OpenAlex."""

    name = NAME

    def __init__(
        self,
        *,
        store: ResearchStore | None = None,
        client: SourceClient | None = None,
        api_key: str | None = None,
        mailto: str | None = None,
        mode: Literal["keyword", "semantic", "both"] = "keyword",
        tdm_allowed_hosts: frozenset[str] = frozenset(),
    ) -> None:
        self._client = resolve_client(NAME, store=store, client=client)
        self._auth = {
            **({"api_key": api_key} if api_key else {}),
            **({"mailto": mailto} if mailto else {}),
        }
        self._mode = mode
        self._tdm = tdm_allowed_hosts
        self._warnings: list[str] = []

    def take_warnings(self) -> list[str]:
        """The records read since the last call that lost a field (and forget them)."""
        taken, self._warnings = self._warnings, []
        return taken

    # -- search ------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """Works matching *query*, best first, at most *limit*.

        In ``both`` mode the semantic hits follow the keyword hits and the
        ones already found are not repeated; a failure of either raises.
        """
        found: list[Candidate] = []
        if self._mode in ("keyword", "both"):
            found += await self._keyword(query, limit=limit, filters=filters)
        if self._mode in ("semantic", "both"):
            seen = {c.work.openalex_id for c in found}
            found += [
                c
                for c in await self.search_semantic(query, limit=limit, filters=filters)
                if c.work.openalex_id not in seen
            ]
        return _ranked(found)

    async def _keyword(
        self, query: str, *, limit: int, filters: SearchFilters | None
    ) -> list[Candidate]:
        base: dict[str, str | int] = {
            "search.title_abstract_keywords": keyword_query(query),
            "select": SELECT,
        }
        if flt := _filter(filters):
            base["filter"] = flt
        return await self._paged(base, limit=limit, ttl_s=TTL_SEARCH_S, query=query)

    async def search_semantic(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """Works near *query* in embedding space (one page, at most ``SEMANTIC_MAX``)."""
        params: dict[str, str | int] = {
            "search.semantic": query,
            "select": SELECT,
            "per_page": min(limit, SEMANTIC_MAX),
        }
        if flt := _filter(filters):
            params["filter"] = flt
        body = await self._get(f"{API}/works", params, ttl_s=TTL_SEARCH_S)
        return self._candidates(body, query=query)

    async def _paged(
        self,
        base: dict[str, str | int],
        *,
        limit: int,
        ttl_s: int,
        query: str = "",
    ) -> list[Candidate]:
        """Cursor-page a list endpoint until *limit* works or no more."""
        found: list[Candidate] = []
        cursor = "*"
        while len(found) < limit:
            params = {
                **base,
                "per_page": min(limit - len(found), MAX_PER_PAGE),
                "cursor": cursor,
            }
            body = await self._get(f"{API}/works", params, ttl_s=ttl_s)
            found += self._candidates(body, query=query, offset=len(found))
            cursor = str(as_mapping(body.get("meta")).get("next_cursor") or "")
            if not cursor or not as_list(body.get("results")):
                break
        return found[:limit]

    # -- links ---------------------------------------------------------------

    async def references(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]:
        """The works *work* cites, as far as OpenAlex has them."""
        oa_id = await self._openalex_id(work)
        body = await self._get(
            f"{API}/works/{oa_id}", {"select": "referenced_works"}, ttl_s=TTL_METADATA_S
        )
        ids = [
            short
            for ref in as_list(body.get("referenced_works"))
            if (short := normalise_openalex_id(str(ref)))
        ][:limit]
        found: list[Candidate] = []
        for batch in _batches(ids, ID_BATCH):
            page = await self._get(
                f"{API}/works",
                {
                    "filter": "openalex:" + "|".join(batch),
                    "select": SELECT,
                    "per_page": len(batch),
                },
                ttl_s=TTL_METADATA_S,
            )
            found += self._candidates(page, offset=len(found))
        return found

    async def citations(
        self, work: Work, *, limit: int = DEFAULT_RESULTS_PER_QUERY
    ) -> list[Candidate]:
        """Works that cite *work*, most cited first."""
        oa_id = await self._openalex_id(work)
        return await self._paged(
            {
                "filter": f"cites:{oa_id}",
                "sort": "cited_by_count:desc",
                "select": SELECT,
            },
            limit=limit,
            ttl_s=TTL_METADATA_S,
        )

    async def _openalex_id(self, work: Work) -> str:
        """The OpenAlex id of *work*: held, else looked up by an identifier (free of credits)."""
        if work.openalex_id:
            return work.openalex_id
        for prefix, value in (
            ("doi", work.doi),
            ("pmid", work.pmid),
            ("pmcid", work.pmcid),
        ):
            if value:
                body = await self._get(
                    f"{API}/works/{prefix}:{quote(value, safe='/:')}",
                    {"select": "id"},
                    ttl_s=TTL_METADATA_S,
                )
                found = normalise_openalex_id(str(body.get("id", "")))
                if found:
                    return found
        raise SourceError(
            NAME,
            "cannot place the work in OpenAlex: no OpenAlex id, DOI, PMID or PMCID",
            kind="config",
        )

    # -- text ----------------------------------------------------------------

    async def fetch_text(self, work: Work) -> FetchedText | None:
        """The text at the work's open-access location, or None when it has none."""
        if not work.oa_url:
            return None
        return await self._client.fetch_document(
            NAME, work.oa_url, tdm_allowed_hosts=self._tdm
        )

    # -- reading the API -------------------------------------------------------

    async def _get(
        self, url: str, params: dict[str, str | int], *, ttl_s: int
    ) -> dict[str, JsonValue]:
        body = await self._client.get_json(
            NAME, url, params=params, auth_params=self._auth, ttl_s=ttl_s
        )
        return dict(expect_mapping(NAME, body, what=url))

    def _candidates(
        self, body: dict[str, JsonValue], *, query: str = "", offset: int = 0
    ) -> list[Candidate]:
        found: list[Candidate] = []
        for item in as_list(body.get("results")):
            work = work_from_item(as_mapping(item), notes=self._warnings)
            if work is None:
                continue
            found.append(
                Candidate(
                    work=work,
                    sources=[NAME],
                    snippet=snippet_of(work),
                    queries=[query] if query else [],
                    rank=offset + len(found),
                )
            )
        return found


def work_from_item(
    item: dict[str, JsonValue] | object, *, notes: list[str]
) -> Work | None:
    """A ``Work`` from one OpenAlex work object; None when it has no title.

    A landing page that is not an http(s) URL (OpenAlex holds some, such as a
    repository handle written after a colon) is left out and one line naming it
    is added to *notes*; the rest of the record is kept.
    """
    data = as_mapping(item)
    title = squash(data.get("title"))
    if not title:
        return None
    ids = as_mapping(data.get("ids"))
    location = as_mapping(data.get("primary_location"))
    source = as_mapping(location.get("source"))
    access = as_mapping(data.get("open_access"))
    abstract = abstract_from_index(data.get("abstract_inverted_index"))
    landing = _text(location.get("landing_page_url"))
    if landing and canonical_url(landing) is None:
        note = f"openalex: {title!r} has a landing page that is not a URL ({landing!r}); left out"
        if note not in notes:
            notes.append(note)
        landing = ""
    return Work(
        openalex_id=normalise_openalex_id(str(data.get("id", ""))),
        doi=normalise_doi(_text(data.get("doi")) or _text(ids.get("doi"))),
        pmid=normalise_pmid(_pmid(ids.get("pmid"))),
        pmcid=normalise_pmcid(_pmcid(ids.get("pmcid"))),
        url=landing or None,
        title=title,
        authors=[
            name
            for a in as_list(data.get("authorships"))
            if (
                name := squash(
                    as_mapping(as_mapping(a).get("author")).get("display_name")
                )
            )
        ],
        year=year_of_date(data.get("publication_year")),
        venue=squash(source.get("display_name")),
        abstract=abstract,
        oa_url=_text(access.get("oa_url")) or None,
        access="abstract" if abstract else "metadata",
        retracted="retracted" if data.get("is_retracted") is True else "unknown",
    )


def abstract_from_index(index: object) -> str:
    """OpenAlex ships abstracts as ``{word: [positions]}``; this puts them back in order."""
    positions: dict[int, str] = {}
    for word, places in as_mapping(index).items():
        for place in as_list(places):
            if isinstance(place, int) and not isinstance(place, bool):
                positions[place] = word
    return " ".join(positions[i] for i in sorted(positions))


_WILDCARDS = re.compile(r"[?*]")


def keyword_query(query: str) -> str:
    """*query* without the wildcard characters ``?`` and ``*``: OpenAlex reads
    them as wildcards and refuses them in its default (stemmed) search with
    HTTP 400, and a question ends with ``?``."""
    return " ".join(_WILDCARDS.sub(" ", query).split())


def _filter(filters: SearchFilters | None) -> str:
    if filters is None:
        return ""
    parts: list[str] = []
    if filters.year_from is not None:
        parts.append(f"from_publication_date:{filters.year_from}-01-01")
    if filters.year_to is not None:
        parts.append(f"to_publication_date:{filters.year_to}-12-31")
    if filters.open_access_only:
        parts.append("open_access.is_oa:true")
    return ",".join(parts)


def _ranked(found: list[Candidate]) -> list[Candidate]:
    return [c.model_copy(update={"rank": i}) for i, c in enumerate(found)]


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _pmid(value: object) -> str | None:
    """OpenAlex gives a PMID as a pubmed.ncbi.nlm.nih.gov URL."""
    text = _text(value)
    tail = text.rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.isdigit() else None


def _pmcid(value: object) -> str | None:
    """OpenAlex gives a PMCID as a ncbi.nlm.nih.gov/pmc/articles/PMC... URL."""
    found = re.search(r"PMC\d+", _text(value), re.IGNORECASE)
    return found.group(0) if found else None
