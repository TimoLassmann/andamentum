"""PubMed records: efetch XML in, typed records out.

:func:`parse_pubmed_xml` is a pure function from an efetch ``retmode=xml``
response to :class:`PubMedRecord` values; :func:`fetch_pubmed` is the efetch
call. Searching (esearch), rate limiting and where an API key comes from
belong to the caller: the key is an explicit argument here, and this package
reads no environment variables.

Titles and abstract text are read with ``itertext()``, so mixed content
(``<i>``, ``<sub>``) is kept whole rather than cut at the first child.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, Field

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedRecord(BaseModel):
    """One PubMed article."""

    pmid: str = ""
    title: str = ""
    abstract: str = ""  # sections joined by spaces, "LABEL: text" when labelled
    abstract_sections: list[tuple[str, str]] = Field(default_factory=list)
    """``(label, text)`` per AbstractText element; the label is "" if unlabelled."""
    authors: list[str] = Field(default_factory=list)  # "Surname Initials"
    journal: str = ""
    year: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    pmcid: str = ""
    mesh_terms: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)


def parse_pubmed_xml(xml_text: str) -> list[PubMedRecord]:
    """The articles in an efetch XML response.

    Raises ``ValueError`` when *xml_text* is not XML. An article that cannot
    be read is not skipped silently: the exception propagates.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"not PubMed XML: {exc}") from exc
    return [_parse_article(el) for el in root.findall(".//PubmedArticle")]


def _text(el: ET.Element | None) -> str:
    """All text under *el*, mixed content included."""
    return "".join(el.itertext()).strip() if el is not None else ""


def _parse_article(article_el: ET.Element) -> PubMedRecord:
    medline = article_el.find(".//MedlineCitation")
    article = medline.find(".//Article") if medline is not None else None

    sections: list[tuple[str, str]] = []
    authors: list[str] = []
    publication_types: list[str] = []
    if article is not None:
        for abs_el in article.findall(".//Abstract/AbstractText"):
            sections.append((abs_el.get("Label", ""), _text(abs_el)))
        for author_el in article.findall(".//AuthorList/Author"):
            last = author_el.findtext("LastName", "")
            initials = author_el.findtext("Initials", "")
            if last:
                authors.append(f"{last} {initials}".strip())
        for pt_el in article.findall(".//PublicationTypeList/PublicationType"):
            if pt_el.text:
                publication_types.append(pt_el.text)
    abstract = " ".join(
        f"{label}: {text}" if label else text for label, text in sections
    )

    def article_id(kind: str) -> str:
        for id_el in article_el.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if id_el.get("IdType") == kind:
                return (id_el.text or "").strip()
        return ""

    doi = article_id("doi")
    if not doi and article is not None:
        for id_el in article.findall(".//ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = (id_el.text or "").strip()
                break

    def under_article(path: str) -> str:
        return _text(article.find(path)) if article is not None else ""

    mesh_terms: list[str] = []
    if medline is not None:
        for mesh_el in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
            if mesh_el.text:
                mesh_terms.append(mesh_el.text)

    return PubMedRecord(
        pmid=_text(medline.find(".//PMID")) if medline is not None else "",
        title=under_article(".//ArticleTitle"),
        abstract=abstract,
        abstract_sections=sections,
        authors=authors,
        journal=under_article(".//Journal/Title"),
        year=under_article(".//Journal/JournalIssue/PubDate/Year"),
        volume=under_article(".//Journal/JournalIssue/Volume"),
        issue=under_article(".//Journal/JournalIssue/Issue"),
        pages=under_article(".//Pagination/MedlinePgn"),
        doi=doi,
        pmcid=article_id("pmc"),
        mesh_terms=mesh_terms,
        publication_types=publication_types,
    )


async def fetch_pubmed(
    pmids: list[str],
    *,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
) -> list[PubMedRecord]:
    """The records for *pmids* (efetch, ``retmode=xml``).

    *api_key* raises NCBI's rate limit from 3 to 10 requests per second; it is
    an explicit argument because this package reads no environment variables.
    An HTTP error raises ``httpx.HTTPStatusError``; an empty *pmids* returns
    ``[]`` without a request. A client passed in is not closed.
    """
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    owned = client is None
    http = client or httpx.AsyncClient(timeout=30.0)
    try:
        response = await http.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        response.raise_for_status()
        return parse_pubmed_xml(response.text)
    finally:
        if owned:
            await http.aclose()
