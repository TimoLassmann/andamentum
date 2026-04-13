"""bioRxiv/medRxiv Evidence Provider.

Searches the Cold Spring Harbor Laboratory API for preprints.
Returns preprint metadata with DOI, category, version, and publication status.

API docs: https://api.biorxiv.org/
No authentication required.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

BIORXIV_API = "https://api.biorxiv.org"


class BioRxivProvider:
    """Evidence provider using bioRxiv/medRxiv preprint API."""

    def __init__(self, max_results: int = 10, server: str = "biorxiv"):
        self.max_results = max_results
        self.server = server  # "biorxiv" or "medrxiv"

    async def check_health(self) -> "CheckResult":
        """Test bioRxiv API reachability."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            # Use a recent date range to test the API
            today = datetime.now().strftime("%Y-%m-%d")
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            url = f"{BIORXIV_API}/details/{self.server}/{week_ago}/{today}/0/1"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    return CheckResult(
                        name="BioRxivProvider",
                        status="pass",
                        message=f"API reachable ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="BioRxivProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="BioRxivProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search bioRxiv for preprints.

        Note: The bioRxiv API is date-range based, not keyword-search based.
        For keyword search, we use the NCBI content API endpoint which indexes
        bioRxiv content.
        """
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Use the content API for keyword search via PubMed-style interface
                # This searches bioRxiv content indexed by NCBI
                params = {
                    "db": "pubmed",
                    "term": f"{query} AND (biorxiv[filter] OR medrxiv[filter])",
                    "retmax": str(self.max_results),
                    "retmode": "json",
                    "sort": "relevance",
                }

                search_resp = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params=params,
                )
                if search_resp.status_code != 200:
                    # Fallback: use date-range API for recent preprints
                    return await self._gather_recent(client, query)

                search_data = search_resp.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])

                if not pmids:
                    return await self._gather_recent(client, query)

                # Fetch details from PubMed (bioRxiv preprints are indexed there)
                fetch_resp = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                    params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
                )
                if fetch_resp.status_code != 200:
                    return []

                data = fetch_resp.json()
                results = data.get("result", {})

                for pmid in pmids:
                    article = results.get(pmid, {})
                    if not isinstance(article, dict):
                        continue

                    title = article.get("title", "")
                    source = article.get("source", "")
                    authors_list = article.get("authors", [])
                    authors = [
                        a.get("name", "") for a in authors_list if isinstance(a, dict)
                    ]
                    pubdate = article.get("pubdate", "")
                    doi = ""
                    for aid in article.get("articleids", []):
                        if isinstance(aid, dict) and aid.get("idtype") == "doi":
                            doi = aid.get("value", "")
                            break

                    if not title:
                        continue

                    content_parts = [title]
                    if authors:
                        content_parts.append(f"Authors: {', '.join(authors[:5])}")
                    if source:
                        content_parts.append(f"Source: {source}")

                    identifiers: dict[str, str] = {"pmid": pmid}
                    if doi:
                        identifiers["doi"] = doi

                    gathered.append(
                        GatheredEvidence(
                            content="\n".join(content_parts),
                            source_ref=f"doi:{doi}" if doi else f"PMID:{pmid}",
                            source_type=self.server,
                            evidence_kind="preprint",
                            identifiers=identifiers,
                            structured_data={
                                "title": title,
                                "authors": authors[:10],
                                "source": source,
                                "pubdate": pubdate,
                                "server": self.server,
                            },
                            quality_score=0.5,  # Preprint: not peer-reviewed
                            quality_metadata={
                                "peer_reviewed": False,
                                "server": self.server,
                            },
                            limitations=["Preprint — not peer-reviewed"],
                        )
                    )

        except Exception as e:
            logger.warning(f"bioRxiv query failed for '{query}': {e}")

        return gathered[: self.max_results]

    async def _gather_recent(self, client: Any, query: str) -> list[GatheredEvidence]:
        """Fallback: get recent preprints from the date-range API."""
        today = datetime.now().strftime("%Y-%m-%d")
        month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        url = f"{BIORXIV_API}/details/{self.server}/{month_ago}/{today}/0/{self.max_results}"

        try:
            response = await client.get(url)
            if response.status_code != 200:
                return []

            data = response.json()
            collection = data.get("collection", [])
            gathered: list[GatheredEvidence] = []

            query_lower = query.lower()
            for item in collection:
                title = item.get("title", "")
                abstract = item.get("abstract", "")

                # Basic relevance filter
                text = f"{title} {abstract}".lower()
                if not any(term in text for term in query_lower.split()[:3]):
                    continue

                doi = item.get("doi", "")
                authors = item.get("authors", "")
                category = item.get("category", "")
                version = item.get("version", "")
                date = item.get("date", "")
                published = item.get("published", "")

                content_parts = [title]
                if authors:
                    content_parts.append(f"Authors: {authors}")
                if abstract:
                    content_parts.append(f"\n{abstract[:500]}")

                identifiers: dict[str, str] = {}
                if doi:
                    identifiers["doi"] = doi

                gathered.append(
                    GatheredEvidence(
                        content="\n".join(content_parts),
                        source_ref=f"doi:{doi}" if doi else title,
                        source_type=self.server,
                        evidence_kind="preprint",
                        identifiers=identifiers,
                        structured_data={
                            "title": title,
                            "authors": authors,
                            "category": category,
                            "posted_date": date,
                            "version": version,
                            "published_doi": published or None,
                            "server": self.server,
                        },
                        quality_score=0.55 if published else 0.5,
                        quality_metadata={
                            "peer_reviewed": bool(published),
                            "version": version,
                        },
                        limitations=["Preprint — not peer-reviewed"]
                        if not published
                        else [],
                    )
                )

            return gathered[: self.max_results]
        except Exception as e:
            logger.debug(f"bioRxiv date-range fallback failed: {e}")
            return []
