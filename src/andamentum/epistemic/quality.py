"""Quality scoring for evidence sources via OpenAlex.

Framework-agnostic (Layer 1). Uses httpx for API calls.

Dual purpose:
1. Quality scoring: Score any evidence with a DOI/PMID
2. Literature search: Search OpenAlex and return papers WITH quality pre-scored

Scoring logic is deterministic (no LLM):
- Citation component: log-normalized citation count
- Journal component: DOAJ status and known journals
- Retraction: hard floor (score = 0.0 if retracted)
- Age-normalization: citations per year since publication
- Composite: weighted average, clamped [0.05, 1.0]
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualityScore:
    """Source quality assessment from OpenAlex.

    Attributes:
        score: Composite quality score 0.0-1.0
        cited_by_count: Number of citations
        is_retracted: Whether the work has been retracted
        publication_year: Year of publication
        journal_name: Name of the journal/source
        oa_status: Open access status
        source: Which scorer produced this ("openalex")
        raw_metadata: Full OpenAlex response for traceability
    """

    score: float
    cited_by_count: int = 0
    is_retracted: bool = False
    publication_year: Optional[int] = None
    journal_name: Optional[str] = None
    oa_status: Optional[str] = None
    source: str = "openalex"
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiteratureResult:
    """A literature search result with pre-scored quality.

    Attributes:
        title: Paper title
        abstract: Paper abstract (may be empty)
        doi: DOI identifier
        pmid: PubMed ID (if available)
        authors: List of author names
        quality: Pre-computed quality score (no second API call needed)
    """

    title: str
    abstract: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    quality: Optional[QualityScore] = None


# ══════════════════════════════════════════════════════════════════════════════
# SCORING LOGIC (deterministic, no LLM)
# ══════════════════════════════════════════════════════════════════════════════


def _citation_component(cited_by_count: int, years_since_pub: float) -> float:
    """Compute citation component of quality score.

    Uses age-normalized citations: citations per year.
    Saturates at ~3000 citations/year (log10(3001) ≈ 3.48).
    """
    if years_since_pub < 1.0:
        years_since_pub = 1.0  # Avoid division by zero for recent papers

    citations_per_year = cited_by_count / years_since_pub
    return min(1.0, math.log10(citations_per_year + 1) / 3.5)


def _journal_component(primary_location: Optional[dict[str, Any]]) -> float:
    """Compute journal component of quality score.

    1.0 if in DOAJ (Directory of Open Access Journals — curated list)
    0.5 if known journal with name
    0.0 if unknown source
    """
    if not primary_location:
        return 0.0

    source = primary_location.get("source") or {}
    if not source:
        return 0.0

    is_in_doaj = source.get("is_in_doaj", False)
    if is_in_doaj:
        return 1.0

    display_name = source.get("display_name", "")
    if display_name:
        return 0.5

    return 0.0


def compute_quality_score(work: dict[str, Any]) -> QualityScore:
    """Compute quality score from an OpenAlex work object.

    Args:
        work: OpenAlex work response dict

    Returns:
        QualityScore with all components computed
    """
    cited_by_count = work.get("cited_by_count", 0) or 0
    is_retracted = work.get("is_retracted", False) or False
    publication_year = work.get("publication_year")

    # Extract journal info
    primary_location = work.get("primary_location") or {}
    source_info = primary_location.get("source") or {}
    journal_name = source_info.get("display_name")
    oa_status = work.get("open_access", {}).get("oa_status")

    # Hard floor: retracted papers get 0.0
    if is_retracted:
        return QualityScore(
            score=0.0,
            cited_by_count=cited_by_count,
            is_retracted=True,
            publication_year=publication_year,
            journal_name=journal_name,
            oa_status=oa_status,
            source="openalex",
            raw_metadata=work,
        )

    # Compute components
    years_since_pub = 1.0
    if publication_year:
        years_since_pub = max(1.0, datetime.now().year - publication_year)

    citation_score = _citation_component(cited_by_count, years_since_pub)
    journal_score = _journal_component(primary_location)

    # Composite: 60% citations, 40% journal
    composite = 0.6 * citation_score + 0.4 * journal_score

    # Clamp to [0.05, 1.0]
    score = max(0.05, min(1.0, composite))

    return QualityScore(
        score=score,
        cited_by_count=cited_by_count,
        is_retracted=False,
        publication_year=publication_year,
        journal_name=journal_name,
        oa_status=oa_status,
        source="openalex",
        raw_metadata=work,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OPENALEX API CLIENT
# ══════════════════════════════════════════════════════════════════════════════


async def _get_openalex_client() -> Any:
    """Create an httpx AsyncClient for OpenAlex with polite pool headers."""
    import os
    import httpx

    email = os.environ.get("OPENALEX_EMAIL", "")
    headers = {"User-Agent": f"mosaic-epistemic/0.1 (mailto:{email})" if email else "mosaic-epistemic/0.1"}
    return httpx.AsyncClient(
        base_url="https://api.openalex.org",
        headers=headers,
        timeout=30.0,
    )


async def score_source(
    doi: Optional[str] = None,
    pmid: Optional[str] = None,
    source_ref: str = "",
    source_type: str = "unknown",
) -> Optional[QualityScore]:
    """Score a source's quality via OpenAlex lookup.

    Tries DOI first, then PMID. Returns None if no identifier is available
    or if the API lookup fails (quality assessment deferred to agent).

    Args:
        doi: DOI identifier (e.g., "10.1038/s41586-020-2012-7")
        pmid: PubMed ID (e.g., "35486828")
        source_ref: Source reference (for logging context)
        source_type: Source type (for logging context)

    Returns:
        QualityScore from OpenAlex, or None if lookup not possible/failed
    """
    if not doi and not pmid:
        return None

    try:
        client = await _get_openalex_client()
        async with client:
            # Try DOI first
            if doi:
                clean_doi = doi.removeprefix("doi:").removeprefix("https://doi.org/")
                response = await client.get(f"/works/doi:{clean_doi}")
                if response.status_code == 200:
                    work = response.json()
                    return compute_quality_score(work)

            # Try PMID
            if pmid:
                clean_pmid = pmid.removeprefix("pmid:").removeprefix("PMID:")
                response = await client.get(f"/works/pmid:{clean_pmid}")
                if response.status_code == 200:
                    work = response.json()
                    return compute_quality_score(work)

        # API didn't find it
        logger.debug(f"OpenAlex lookup failed for doi={doi}, pmid={pmid}")
        return None

    except Exception as e:
        logger.warning(f"OpenAlex API error: {e}")
        return None


async def search_literature(
    query: str,
    max_results: int = 10,
) -> list[LiteratureResult]:
    """Search OpenAlex for literature matching a query.

    Returns papers WITH quality pre-scored — no second API call needed.

    Args:
        query: Search query (natural language or structured)
        max_results: Maximum number of results to return

    Returns:
        List of LiteratureResult with pre-computed quality scores
    """
    try:
        client = await _get_openalex_client()
        async with client:
            response = await client.get(
                "/works",
                params={
                    "search": query,
                    "per_page": max_results,
                    "select": "id,doi,title,display_name,publication_year,cited_by_count,"
                              "is_retracted,open_access,primary_location,authorships,ids,"
                              "abstract_inverted_index",
                },
            )

            if response.status_code != 200:
                logger.warning(f"OpenAlex search failed: {response.status_code}")
                return []

            data = response.json()
            results: list[LiteratureResult] = []

            for work in data.get("results", []):
                # Reconstruct abstract from inverted index
                abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

                # Extract DOI and PMID
                doi = work.get("doi", "").removeprefix("https://doi.org/") if work.get("doi") else None
                ids = work.get("ids", {}) or {}
                pmid = ids.get("pmid", "").removeprefix("https://pubmed.ncbi.nlm.nih.gov/").rstrip("/") if ids.get("pmid") else None

                # Extract authors
                authorships = work.get("authorships", []) or []
                authors = [
                    a.get("author", {}).get("display_name", "")
                    for a in authorships[:10]  # Limit to first 10 authors
                    if a.get("author", {}).get("display_name")
                ]

                # Compute quality (no second API call)
                quality = compute_quality_score(work)

                title = work.get("title") or work.get("display_name") or ""
                results.append(LiteratureResult(
                    title=title,
                    abstract=abstract,
                    doi=doi,
                    pmid=pmid,
                    authors=authors,
                    quality=quality,
                ))

            return results

    except Exception as e:
        logger.warning(f"OpenAlex search error: {e}")
        return []


def _reconstruct_abstract(inverted_index: Optional[dict[str, list[int]]]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format.

    OpenAlex stores abstracts as inverted indices: {"word": [position1, position2, ...]}.
    """
    if not inverted_index:
        return ""

    # Build word list sorted by position
    words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))

    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)
