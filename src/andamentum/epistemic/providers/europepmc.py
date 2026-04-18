"""Europe PMC Evidence Provider.

Searches the Europe PMC REST API for biomedical and life sciences literature.
Returns full abstracts, author data, and citation counts across PubMed, PMC,
preprints, and patents.

API docs: https://europepmc.org/RestfulWebService
No authentication required.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class EuropePMCProvider:
    """Evidence provider using the Europe PMC REST API."""

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test Europe PMC API reachability."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{EUROPEPMC_API}/search",
                    params={
                        "query": "test",
                        "resultType": "lite",
                        "format": "json",
                        "pageSize": 1,
                    },
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    return CheckResult(
                        name="EuropePMCProvider",
                        status="pass",
                        message=f"API reachable ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="EuropePMCProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="EuropePMCProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search Europe PMC for biomedical literature."""
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{EUROPEPMC_API}/search",
                    params={
                        "query": query,
                        "resultType": "core",
                        "format": "json",
                        "pageSize": self.max_results,
                        "cursorMark": "*",
                    },
                )
                if response.status_code != 200:
                    logger.warning(
                        "EuropePMC query failed for '%s': HTTP %d",
                        query,
                        response.status_code,
                    )
                    return gathered

                data = response.json()
                results = (
                    data.get("resultList", {}).get("result", [])
                )

                for item in results:
                    title = item.get("title", "")
                    if not title:
                        continue

                    authors = item.get("authorString", "")
                    abstract_raw = item.get("abstractText", "")
                    abstract = _HTML_TAG_RE.sub("", abstract_raw) if abstract_raw else ""

                    doi = item.get("doi", "")
                    pmid = item.get("pmid", "")
                    pmcid = item.get("pmcid", "")
                    source = item.get("source", "")
                    journal = item.get("journalTitle", "")
                    pub_year = item.get("pubYear", "")
                    cited_by_count = item.get("citedByCount", 0)
                    is_open_access = item.get("isOpenAccess", "N")
                    pub_type_list = item.get("pubTypeList", {})
                    pub_types: list[str] = pub_type_list.get("pubType", []) if isinstance(pub_type_list, dict) else []

                    # Build content
                    content_parts = [title]
                    if authors:
                        content_parts.append(f"Authors: {authors}")
                    if abstract:
                        content_parts.append(f"\n{abstract}")

                    # Source ref
                    if doi:
                        source_ref = f"doi:{doi}"
                    elif pmid:
                        source_ref = f"PMID:{pmid}"
                    else:
                        source_ref = title

                    # Evidence kind
                    evidence_kind = "preprint" if source == "PPR" else "literature"

                    # Identifiers
                    identifiers: dict[str, str] = {}
                    if pmid:
                        identifiers["pmid"] = str(pmid)
                    if doi:
                        identifiers["doi"] = doi
                    if pmcid:
                        identifiers["pmcid"] = pmcid

                    # Limitations
                    limitations: list[str] = (
                        ["Preprint \u2014 not peer-reviewed"] if source == "PPR" else []
                    )

                    gathered.append(
                        GatheredEvidence(
                            content="\n".join(content_parts),
                            source_ref=source_ref,
                            source_type="europepmc",
                            evidence_kind=evidence_kind,
                            identifiers=identifiers,
                            structured_data={
                                "title": title,
                                "authors": authors,
                                "journal": journal,
                                "pub_year": pub_year,
                                "cited_by_count": cited_by_count,
                                "is_open_access": is_open_access,
                                "pub_types": pub_types,
                                "source": source,
                            },
                            quality_score=None,
                            quality_metadata={
                                "journal": journal,
                                "cited_by_count": cited_by_count,
                                "is_open_access": is_open_access,
                                "pub_types": pub_types,
                            },
                            limitations=limitations,
                        )
                    )

        except Exception as e:
            logger.warning(f"EuropePMC query failed for '{query}': {e}")

        return gathered
