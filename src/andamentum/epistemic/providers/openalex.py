"""OpenAlex Evidence Provider.

Implements EvidenceProvider protocol. Wraps quality.py's search_literature()
to return GatheredEvidence items with quality pre-populated.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence
from ..quality import search_literature, QualityScore, score_source


class OpenAlexProvider:
    """Evidence provider using OpenAlex for literature search.

    Quality comes FOR FREE with search results — no second API call needed.
    """

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test OpenAlex API reachability."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.openalex.org/works",
                    params={"filter": "title.search:test", "per_page": 1},
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    return CheckResult(
                        name="OpenAlexProvider",
                        status="pass",
                        message=f"API reachable ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="OpenAlexProvider", status="fail", message=f"HTTP {response.status_code}", elapsed_ms=elapsed
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(name="OpenAlexProvider", status="fail", message=str(e), elapsed_ms=elapsed)

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search OpenAlex for literature matching a query.

        Args:
            query: Natural language query

        Returns:
            List of GatheredEvidence with quality pre-populated
        """
        results = await search_literature(query, max_results=self.max_results)
        gathered: list[GatheredEvidence] = []

        for r in results:
            if not r.title and not r.abstract:
                continue

            content_parts: list[str] = []
            if r.title:
                content_parts.append(r.title)
            if r.authors:
                content_parts.append(f"Authors: {', '.join(r.authors[:5])}")
            if r.abstract:
                content_parts.append(f"\n{r.abstract}")

            content = "\n".join(content_parts)

            source_ref = f"doi:{r.doi}" if r.doi else r.title
            if r.pmid:
                source_ref += f" (PMID:{r.pmid})"

            # Build structured identifiers
            ids: dict[str, str] = {}
            if r.doi:
                ids["doi"] = r.doi
            if r.pmid:
                ids["pmid"] = r.pmid

            gathered.append(
                GatheredEvidence(
                    content=content,
                    source_ref=source_ref,
                    source_type="openalex",
                    evidence_kind="literature",
                    identifiers=ids,
                    structured_data={
                        "title": r.title or "",
                        "authors": r.authors[:10] if r.authors else [],
                        "year": r.year if hasattr(r, "year") else None,
                        "journal": getattr(r, "journal", None),
                        "citation_count": getattr(r, "citation_count", None),
                        "is_open_access": getattr(r, "is_open_access", None),
                    },
                    quality_score=r.quality.score if r.quality else None,
                    quality_metadata=asdict(r.quality) if r.quality else {},
                    limitations=["Abstract only; full text may contain more detail"],
                )
            )

        return gathered


class OpenAlexQualityScorer:
    """QualityScorer implementation using OpenAlex.

    Implements the QualityScorer protocol from epistemic.operations.
    Looks up DOI/PMID via OpenAlex API and returns quality score.
    """

    async def score(self, source_ref: str, source_type: str) -> "QualityScore | None":
        """Score a source's quality via OpenAlex.

        Extracts DOI/PMID from source_ref if present.

        Args:
            source_ref: Source reference string (may contain DOI/PMID)
            source_type: Source type (for logging context)

        Returns:
            QualityScore from OpenAlex, or None if no DOI/PMID or lookup fails
        """
        doi = None
        pmid = None

        # Try to extract DOI from source_ref
        if "doi:" in source_ref.lower() or "10." in source_ref:
            for part in source_ref.split():
                cleaned = part.removeprefix("doi:").removeprefix("https://doi.org/")
                if cleaned.startswith("10."):
                    doi = cleaned
                    break

        # Try to extract PMID
        if "pmid:" in source_ref.lower() or "PMID:" in source_ref:
            for part in source_ref.split():
                cleaned = part.removeprefix("pmid:").removeprefix("PMID:").strip("()")
                if cleaned.isdigit():
                    pmid = cleaned
                    break

        return await score_source(
            doi=doi,
            pmid=pmid,
            source_ref=source_ref,
            source_type=source_type,
        )
