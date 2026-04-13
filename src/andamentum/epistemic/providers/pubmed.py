"""PubMed Evidence Provider.

Searches NCBI PubMed via E-utilities for biomedical literature.
Returns structured article metadata with PMID, DOI, MeSH terms,
and publication type for quality scoring.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
Rate limit: 3/s without API key, 10/s with NCBI_API_KEY.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Publication type → quality score mapping
_PUB_TYPE_QUALITY: dict[str, float] = {
    "Meta-Analysis": 0.95,
    "Systematic Review": 0.90,
    "Randomized Controlled Trial": 0.85,
    "Clinical Trial, Phase III": 0.85,
    "Clinical Trial, Phase IV": 0.85,
    "Clinical Trial": 0.75,
    "Comparative Study": 0.70,
    "Observational Study": 0.65,
    "Review": 0.60,
    "Case Reports": 0.45,
    "Editorial": 0.30,
    "Letter": 0.25,
}


class PubMedProvider:
    """Evidence provider using NCBI PubMed E-utilities."""

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.api_key = os.getenv("NCBI_API_KEY")

    async def check_health(self) -> "CheckResult":
        """Test PubMed E-utilities reachability."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            params: dict[str, Any] = {
                "db": "pubmed",
                "term": "test",
                "retmax": "1",
                "retmode": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{EUTILS_BASE}/esearch.fcgi", params=params
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    return CheckResult(
                        name="PubMedProvider",
                        status="pass",
                        message=f"API reachable ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="PubMedProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="PubMedProvider", status="fail", message=str(e), elapsed_ms=elapsed
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search PubMed and fetch article metadata."""
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Search for PMIDs
                search_params: dict[str, Any] = {
                    "db": "pubmed",
                    "term": query,
                    "retmax": str(self.max_results),
                    "retmode": "json",
                    "sort": "relevance",
                }
                if self.api_key:
                    search_params["api_key"] = self.api_key

                search_resp = await client.get(
                    f"{EUTILS_BASE}/esearch.fcgi", params=search_params
                )
                if search_resp.status_code != 200:
                    return []

                search_data = search_resp.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])
                if not pmids:
                    return []

                # Step 2: Fetch article details
                fetch_params: dict[str, Any] = {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                }
                if self.api_key:
                    fetch_params["api_key"] = self.api_key

                fetch_resp = await client.get(
                    f"{EUTILS_BASE}/efetch.fcgi", params=fetch_params
                )
                if fetch_resp.status_code != 200:
                    return []

                gathered = self._parse_articles(fetch_resp.text)

        except Exception as e:
            logger.warning(f"PubMed query failed for '{query}': {e}")

        return gathered

    def _parse_articles(self, xml_text: str) -> list[GatheredEvidence]:
        """Parse PubMed XML response into GatheredEvidence items."""
        gathered: list[GatheredEvidence] = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        for article_el in root.findall(".//PubmedArticle"):
            try:
                gathered.append(self._parse_single_article(article_el))
            except Exception as e:
                logger.debug(f"Failed to parse article: {e}")
                continue

        return gathered

    def _parse_single_article(self, article_el: Any) -> GatheredEvidence:
        """Parse a single PubmedArticle XML element."""
        medline = article_el.find(".//MedlineCitation")
        article = medline.find(".//Article") if medline else None

        # PMID
        pmid_el = medline.find(".//PMID") if medline else None
        pmid = pmid_el.text if pmid_el is not None else ""

        # Title
        title_el = article.find(".//ArticleTitle") if article else None
        title = title_el.text if title_el is not None else ""

        # Abstract
        abstract_parts = []
        if article is not None:
            for abs_text in article.findall(".//Abstract/AbstractText"):
                label = abs_text.get("Label", "")
                text = abs_text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        if article is not None:
            for author_el in article.findall(".//AuthorList/Author"):
                last = author_el.findtext("LastName", "")
                initials = author_el.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials}".strip())

        # Journal
        journal_el = article.find(".//Journal/Title") if article else None
        journal = journal_el.text if journal_el is not None else ""

        # Year
        year_el = (
            article.find(".//Journal/JournalIssue/PubDate/Year") if article else None
        )
        year = year_el.text if year_el is not None else ""

        # DOI
        doi = ""
        for id_el in article_el.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "doi":
                doi = id_el.text or ""
                break

        # PMC ID
        pmcid = ""
        for id_el in article_el.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if id_el.get("IdType") == "pmc":
                pmcid = id_el.text or ""
                break

        # MeSH terms
        mesh_terms = []
        if medline is not None:
            for mesh_el in medline.findall(
                ".//MeshHeadingList/MeshHeading/DescriptorName"
            ):
                if mesh_el.text:
                    mesh_terms.append(mesh_el.text)

        # Publication types
        pub_types = []
        if article is not None:
            for pt_el in article.findall(".//PublicationTypeList/PublicationType"):
                if pt_el.text:
                    pub_types.append(pt_el.text)

        # Build human-readable content
        content_parts = []
        if title:
            content_parts.append(title)
        if authors:
            content_parts.append(f"Authors: {', '.join(authors[:5])}")
        if journal and year:
            content_parts.append(f"{journal}, {year}")
        if abstract:
            content_parts.append(f"\n{abstract}")

        # Quality scoring based on publication type
        quality = 0.6  # Default: peer-reviewed article
        for pt in pub_types:
            if pt in _PUB_TYPE_QUALITY:
                quality = max(quality, _PUB_TYPE_QUALITY[pt])
                break

        # Build identifiers
        identifiers: dict[str, str] = {}
        if pmid:
            identifiers["pmid"] = pmid
        if doi:
            identifiers["doi"] = doi
        if pmcid:
            identifiers["pmcid"] = pmcid

        source_ref = f"PMID:{pmid}" if pmid else (f"doi:{doi}" if doi else title)

        return GatheredEvidence(
            content="\n".join(content_parts),
            source_ref=source_ref,
            source_type="pubmed",
            evidence_kind="literature",
            identifiers=identifiers,
            structured_data={
                "title": title,
                "authors": authors[:10],
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "mesh_terms": mesh_terms,
                "publication_types": pub_types,
            },
            quality_score=quality,
            quality_metadata={"publication_types": pub_types, "journal": journal},
            limitations=["Abstract only; full text may contain more detail"]
            if not pmcid
            else [],
        )
