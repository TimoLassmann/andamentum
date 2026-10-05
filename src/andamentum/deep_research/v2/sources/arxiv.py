"""arXiv: preprints, with the full text from the PDF.

The Atom API (``export.arxiv.org/api/query``) is asked for one request in
three seconds on a single connection, which ``SourceClient`` enforces. It has
no semantic search, no open-access filter and no citation links. A query is
turned into an AND of its significant words over all fields, because arXiv's
default treats a long phrase as a phrase; a query that already uses field
syntax (``ti:``, ``abs:``, ``au:``, ``cat:``, ``all:``) is sent as written.

The PDF is fetched from ``arxiv.org/pdf/<id>`` (the API host's robots.txt
disallows robots; arxiv.org's allows ``/pdf`` with a 15 second crawl delay,
which is the spacing used) and read by harvest through the shared extractor.
"""

from __future__ import annotations

import re

from lxml import etree  # type: ignore[attr-defined]  # lxml ships no stubs

from andamentum.references import Work
from andamentum.references.works import normalise_arxiv_id, normalise_doi

from ..store import TTL_SEARCH_S, ResearchStore
from ..types import Candidate
from ._util import parse_xml, snippet_of, squash, year_of_date
from .base import (
    ARXIV_PDF_INTERVAL_S,
    DEFAULT_RESULTS_PER_QUERY,
    FetchedText,
    SearchFilters,
    SourceClient,
    SourceError,
    resolve_client,
)

API = "https://export.arxiv.org/api/query"
NAME = "arxiv"
PDF_INTERVAL_KEY = "arxiv:pdf"
MAX_RESULTS = (
    200  # arXiv allows 2000 per request; a research round never needs that many
)
MAX_TERMS = 8
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_FIELD_SYNTAX = re.compile(r"\b(?:ti|abs|au|cat|all|co|jr|rn|id):", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOP = frozenset(
    "the and for with from that this these those into onto over under about "
    "between among how what which when where why does did are was were been "
    "can could should would will not but its their there than then also such "
    "using use used via per".split()
)


class ArxivSource:
    """Search and PDF full text from arXiv."""

    name = NAME

    def __init__(
        self,
        *,
        store: ResearchStore | None = None,
        client: SourceClient | None = None,
        pdf_interval_s: float = ARXIV_PDF_INTERVAL_S,
    ) -> None:
        self._client = resolve_client(NAME, store=store, client=client)
        self._client.set_interval(PDF_INTERVAL_KEY, pdf_interval_s)

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_RESULTS_PER_QUERY,
        filters: SearchFilters | None = None,
    ) -> list[Candidate]:
        """Preprints matching *query*, best first. A year filter is applied after the
        fetch (the API filters by submission date only in its own syntax)."""
        search_query = to_search_query(query)
        if not search_query:
            raise SourceError(
                NAME, f"no searchable words in the query {query!r}", kind="config"
            )
        result = await self._client.get(
            NAME,
            API,
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": min(limit, MAX_RESULTS),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            ttl_s=TTL_SEARCH_S,
            validate=_check_feed,  # an error entry in a 200 feed is not cached
        )
        found: list[Candidate] = []
        for work in works_from_feed(result.body):
            if not _in_years(work, filters):
                continue
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

    async def fetch_text(self, work: Work) -> FetchedText | None:
        """The text of the PDF, or None when *work* has no arXiv id."""
        if not work.arxiv_id:
            return None
        return await self._client.fetch_document(
            NAME,
            f"https://arxiv.org/pdf/{work.arxiv_id}",
            interval=PDF_INTERVAL_KEY,
        )


def to_search_query(query: str) -> str:
    """An arXiv ``search_query``: the query as written when it uses field syntax, else
    ``all:`` over each significant word, joined by AND."""
    text = query.strip()
    if _FIELD_SYNTAX.search(text):
        return text
    terms: list[str] = []
    for word in _WORD.findall(text):
        lowered = word.lower()
        if len(lowered) >= 3 and lowered not in _STOP and lowered not in terms:
            terms.append(lowered)
    return " AND ".join(f"all:{t}" for t in terms[:MAX_TERMS])


def works_from_feed(xml: bytes) -> list[Work]:
    """The works in an Atom feed. A feed the API answers with an error entry raises."""
    try:
        root = parse_xml(xml)
    except etree.XMLSyntaxError as exc:
        raise SourceError(NAME, f"the feed is not XML: {exc}", kind="parse") from exc
    works: list[Work] = []
    for entry in root.iterfind(f"{_ATOM}entry"):
        entry_id = squash(entry.findtext(f"{_ATOM}id"))
        if "/api/errors" in entry_id:
            raise SourceError(
                NAME,
                squash(entry.findtext(f"{_ATOM}summary"))
                or "the API reported an error",
                kind="http",
            )
        work = _work(entry, entry_id)
        if work is not None:
            works.append(work)
    return works


def _check_feed(xml: bytes) -> None:
    works_from_feed(xml)


def _work(entry: etree._Element, entry_id: str) -> Work | None:
    arxiv_id = normalise_arxiv_id(entry_id)
    title = squash(entry.findtext(f"{_ATOM}title"))
    if arxiv_id is None or not title:
        return None
    return Work(
        arxiv_id=arxiv_id,
        doi=normalise_doi(entry.findtext(f"{_ARXIV}doi")),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        title=title,
        authors=[
            name
            for a in entry.iterfind(f"{_ATOM}author")
            if (name := squash(a.findtext(f"{_ATOM}name")))
        ],
        year=year_of_date(entry.findtext(f"{_ATOM}published")),
        venue=squash(entry.findtext(f"{_ARXIV}journal_ref")) or "arXiv",
        abstract=squash(entry.findtext(f"{_ATOM}summary")),
        oa_url=f"https://arxiv.org/pdf/{arxiv_id}",
        access="abstract",
    )


def _in_years(work: Work, filters: SearchFilters | None) -> bool:
    if filters is None or work.year is None:
        return True
    if filters.year_from is not None and work.year < filters.year_from:
        return False
    return filters.year_to is None or work.year <= filters.year_to
