"""Cochrane Library Evidence Provider.

Searches for Cochrane systematic reviews and meta-analyses via PubMed
E-utilities. Each review synthesizes findings from multiple randomized
controlled trials on a specific clinical question.

API: PubMed E-utilities with journal filter "Cochrane Database Syst Rev"
No authentication required.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class CochraneProvider:
    """Evidence provider for Cochrane systematic reviews via PubMed E-utilities."""

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test PubMed E-utilities reachability with Cochrane journal filter."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            params: dict[str, Any] = {
                "db": "pubmed",
                "term": '"Cochrane Database Syst Rev"[Journal]',
                "retmax": "1",
                "retmode": "json",
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{EUTILS_BASE}/esearch.fcgi", params=params
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    data = response.json()
                    count = int(data.get("esearchresult", {}).get("count", "0"))
                    if count > 0:
                        return CheckResult(
                            name="CochraneProvider",
                            status="pass",
                            message=f"API reachable ({elapsed:.0f}ms)",
                            elapsed_ms=elapsed,
                        )
                    return CheckResult(
                        name="CochraneProvider",
                        status="fail",
                        message="No Cochrane reviews found in PubMed",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="CochraneProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="CochraneProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search PubMed for Cochrane systematic reviews matching the query."""
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Search for PMIDs filtered to Cochrane reviews
                search_params: dict[str, Any] = {
                    "db": "pubmed",
                    "term": f'{query} AND "Cochrane Database Syst Rev"[Journal]',
                    "retmax": str(self.max_results),
                    "retmode": "json",
                    "sort": "relevance",
                }

                search_resp = await client.get(
                    f"{EUTILS_BASE}/esearch.fcgi", params=search_params
                )
                if search_resp.status_code != 200:
                    return []

                search_data = search_resp.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])
                if not pmids:
                    return []

                # Step 2: Fetch article details as XML
                fetch_params: dict[str, Any] = {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "rettype": "abstract",
                    "retmode": "xml",
                }

                fetch_resp = await client.get(
                    f"{EUTILS_BASE}/efetch.fcgi", params=fetch_params
                )
                if fetch_resp.status_code != 200:
                    return []

                gathered = self._parse_articles(fetch_resp.text)

        except Exception as e:
            logger.warning(f"CochraneProvider query failed for '{query}': {e}")

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
                result = self._parse_single_article(article_el)
                if result is not None:
                    gathered.append(result)
            except Exception as e:
                logger.debug(f"Failed to parse Cochrane article: {e}")
                continue

        return gathered

    def _parse_single_article(self, article_el: Any) -> GatheredEvidence | None:
        """Parse a single PubmedArticle XML element into GatheredEvidence."""
        medline = article_el.find(".//MedlineCitation")
        article = medline.find(".//Article") if medline is not None else None

        # PMID
        pmid_el = medline.find(".//PMID") if medline is not None else None
        pmid = pmid_el.text if pmid_el is not None else ""

        # Title
        title_el = article.find(".//ArticleTitle") if article is not None else None
        title = title_el.text if title_el is not None else ""

        if not title:
            return None

        # Authors
        authors: list[str] = []
        if article is not None:
            for author_el in article.findall(".//AuthorList/Author"):
                last = author_el.findtext("LastName", "")
                initials = author_el.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials}".strip())

        # Structured abstract sections
        abstract_sections: dict[str, str] = {}
        content_parts: list[str] = [title]
        if authors:
            if len(authors) > 5:
                content_parts.append(
                    f"Authors: {', '.join(authors[:5])} (et al, {len(authors)} authors total)"
                )
            else:
                content_parts.append(f"Authors: {', '.join(authors)}")

        if article is not None:
            for abstract_text in article.findall(".//Abstract/AbstractText"):
                label = abstract_text.get("Label", "")
                text = abstract_text.text or ""
                if label:
                    abstract_sections[label] = text
                    content_parts.append(f"{label}: {text}")
                else:
                    content_parts.append(text)

        # DOI — check both locations (MedlineCitation and PubmedData)
        doi = ""
        if article is not None:
            for id_el in article.findall(".//ArticleIdList/ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text or ""
                    break
        if not doi:
            for id_el in article_el.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if id_el.get("IdType") == "doi":
                    doi = id_el.text or ""
                    break

        # Publication date
        pub_date = ""
        if article is not None:
            year_el = article.find(".//Journal/JournalIssue/PubDate/Year")
            if year_el is not None and year_el.text:
                pub_date = year_el.text

        # Build identifiers
        identifiers: dict[str, str] = {}
        if pmid:
            identifiers["pmid"] = pmid
        if doi:
            identifiers["doi"] = doi

        # Source reference
        source_ref = f"doi:{doi}" if doi else f"PMID:{pmid}"

        return GatheredEvidence(
            content="\n".join(content_parts),
            source_ref=source_ref,
            source_type="cochrane",
            evidence_kind="systematic_review",
            identifiers=identifiers,
            structured_data={
                "title": title,
                "authors": authors,
                "pub_date": pub_date,
                "abstract_sections": abstract_sections,
            },
            quality_score=None,
            quality_metadata={
                "journal": "Cochrane Database Syst Rev",
                "publication_types": ["Systematic Review"],
            },
            limitations=[],
        )
