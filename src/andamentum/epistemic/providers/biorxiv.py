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
from typing import TYPE_CHECKING

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

        bioRxiv's own API is date-range based, not keyword-search based,
        so keyword search goes through NCBI E-utilities filtered to
        ``biorxiv[filter] OR medrxiv[filter]``. When the filter returns no
        PMIDs, return an empty list — bioRxiv genuinely has no
        keyword-searchable hit for the query through this index.
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
                    return []

                search_data = search_resp.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])

                if not pmids:
                    return []

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
                        if len(authors) > 5:
                            content_parts.append(
                                f"Authors: {', '.join(authors[:5])} (et al, {len(authors)} authors total)"
                            )
                        else:
                            content_parts.append(f"Authors: {', '.join(authors)}")
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
                                "authors": authors,
                                "source": source,
                                "pubdate": pubdate,
                                "server": self.server,
                            },
                            quality_score=None,
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
