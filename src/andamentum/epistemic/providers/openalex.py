"""OpenAlex Evidence Provider.

Implements EvidenceProvider protocol. Wraps quality.py's search_literature()
to return GatheredEvidence items with quality pre-populated.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..operations.identifier_extraction import Identifiers
    from ..preflight import CheckResult

from ..operations import GatheredEvidence
from ..quality import search_literature, QualityScore, score_source


class OpenAlexProvider:
    """Evidence provider using OpenAlex for literature search.

    Quality comes FOR FREE with search results — no second API call needed.
    """

    # ── Description-driven-dispatch contract (Phase 1) ───────────────────
    # The dispatch agent reads ``description`` and ``query_examples`` at
    # runtime to decide triage and construct queries. The legacy
    # formulator path also reads ``description`` and ``query_guidance``
    # via the registry shim in ``providers/__init__.py``. Phase 1 keeps
    # both consumers happy by colocating all data on the provider class.

    description = (
        "The default general-purpose academic literature search for any scholarly "
        "question that does not specifically concern human medicine, drug compounds, "
        "or clinical trials. Use this provider whenever the question is about "
        "scientific research, scholarly work, or academic publications in general. "
        "Good default choice for any research question, especially broad or "
        "cross-disciplinary ones. Example queries: 'what do we know about the "
        "Permian-Triassic mass extinction', 'research on transformer attention "
        "mechanisms', 'academic papers about the origin of the Indo-European "
        "languages', 'scholarly work on population genetics and genetic drift'."
    )

    query_guidance = (
        "The query goes to OpenAlex `/works` as the `search` parameter — "
        'full-text relevance ranking. Phrase quoting ("...") and implicit '
        "AND between tokens are supported.\n"
        "\n"
        "Query styles that all work:\n"
        "- Plain bag of terms: metformin HbA1c diabetes\n"
        '- Phrase-anchored: "GLP-1 receptor agonist" obesity\n'
        "- Topic plus study type: meta-analysis aspirin cardiovascular prevention\n"
        "- Author plus topic: Hinton backpropagation\n"
        "- Cross-disciplinary topic: transformer attention mechanism\n"
        "- Multi-domain: gravitational wave detection LIGO\n"
        "\n"
        "OpenAlex does NOT support PubMed-style [MeSH] field tags. OpenAlex is "
        "the strongest pick for non-biomedical scholarly questions (physics, "
        "history, economics, social sciences) and broad cross-disciplinary "
        "searches; for tight biomedical questions, PubMed and Europe PMC "
        "return less noise. The `site:` operator does not work."
    )

    # (claim, native-query) pairs. None = abstain. Used by the
    # description-driven dispatch agent as in-context teaching.
    # Native queries follow this provider's ``query_guidance`` literally
    # — no paraphrasing.
    query_examples: list[tuple[str, str | None]] = [
        (
            "what do we know about the Permian-Triassic mass extinction",
            "Permian-Triassic mass extinction",
        ),
        (
            "research on transformer attention mechanisms in language models",
            "transformer attention mechanism language model",
        ),
        (
            "scholarly work on gravitational wave detection at LIGO",
            '"gravitational wave" detection LIGO',
        ),
        (
            "what has Geoffrey Hinton published on backpropagation",
            "Hinton backpropagation",
        ),
        (
            "academic literature on monetary policy and inflation expectations",
            '"monetary policy" inflation expectations',
        ),
        (
            "scholarly work on population genetics and genetic drift",
            "population genetics genetic drift",
        ),
        # Out-of-domain — non-scholarly current event
        (
            "what was today's closing price of the S&P 500",
            None,
        ),
        # Out-of-domain — technical how-to
        (
            "how do I install Python 3.12 on macOS",
            None,
        ),
    ]

    output_kind = "assertion_evidence"
    independence_group = "general_scholarly"
    provider_contract_version = 1

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
                    name="OpenAlexProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="OpenAlexProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

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
                if len(r.authors) > 5:
                    content_parts.append(
                        f"Authors: {', '.join(r.authors[:5])} (et al, {len(r.authors)} authors total)"
                    )
                else:
                    content_parts.append(f"Authors: {', '.join(r.authors)}")
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
                        "authors": r.authors if r.authors else [],
                        "year": getattr(r, "year", None),
                        "journal": getattr(r, "journal", None),
                        "citation_count": getattr(r, "citation_count", None),
                        "is_open_access": getattr(r, "is_open_access", None),
                    },
                    quality_score=None,
                    quality_metadata=asdict(r.quality) if r.quality else {},
                    limitations=["Abstract only; full text may contain more detail"],
                )
            )

        return gathered


class OpenAlexQualityScorer:
    """QualityScorer implementation using OpenAlex.

    Implements the QualityScorer protocol from andamentum.epistemic.operations.
    Looks up DOI/PMID via OpenAlex API and returns quality score.

    Phase 3 of the efficiency plan: identifiers are now extracted
    upstream by ``operations.identifier_extraction.extract_identifiers``
    and passed in directly. The scorer no longer does its own primitive
    string-based extraction (which missed DOIs in URL paths, content
    bodies, and non-standard formats); the upstream regex-based
    extractor handles those cases.
    """

    async def score(
        self,
        identifiers: "Identifiers",
        source_ref: str,
        source_type: str,
    ) -> "QualityScore | None":
        """Score a source's quality via OpenAlex.

        Args:
            identifiers: Pre-extracted DOI / PMID / arXiv identifiers.
                When all are None, returns None without calling the API.
            source_ref: Source reference string (kept for logging context).
            source_type: Source type (kept for logging context).

        Returns:
            QualityScore from OpenAlex, or None if no identifier resolved
            or lookup failed.
        """
        return await score_source(
            doi=identifiers.doi,
            pmid=identifiers.pmid,
            source_ref=source_ref,
            source_type=source_type,
        )
