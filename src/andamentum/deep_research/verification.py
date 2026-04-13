"""Source verification for deep research results.

Simple, deterministic verification that cited sources were actually accessed.
"""

from typing import TypedDict
from urllib.parse import urlparse


class VerificationResult(TypedDict):
    """Results of source verification."""

    total_cited: int
    verified_count: int
    verified: list[str]
    unverified: list[str]
    accessed_not_cited: list[str]
    verification_rate: float


def normalize_url(url: str) -> str:
    """Normalize URL for comparison.

    Simple normalization:
    - Strip whitespace
    - Convert to lowercase
    - Remove trailing slashes
    - Ensure http/https consistency (treat as equivalent)

    Args:
        url: Raw URL string

    Returns:
        Normalized URL string

    Examples:
        >>> normalize_url("HTTPS://Example.com/Page/")
        'https://example.com/page'
        >>> normalize_url("http://example.com/page")
        'https://example.com/page'
    """
    if not url:
        return ""

    # Strip and lowercase
    url = url.strip().lower()

    # Parse to handle protocol normalization
    parsed = urlparse(url)

    # Force https (treat http/https as equivalent)
    scheme = "https"

    # Remove trailing slashes from path
    path = parsed.path.rstrip("/")

    # Rebuild URL
    normalized = f"{scheme}://{parsed.netloc}{path}"

    # Add query string if present (preserve parameters)
    if parsed.query:
        normalized += f"?{parsed.query}"

    return normalized


def verify_sources(
    cited_sources: list[str], searched_urls: set[str], fetched_urls: set[str]
) -> VerificationResult:
    """Verify that cited sources were actually accessed during research.

    Checks if each cited source matches:
    1. A URL that appeared in search results (searched_urls)
    2. A URL that was actually fetched and read (fetched_urls)

    Args:
        cited_sources: List of source URLs from final report
        searched_urls: Set of URLs from search results
        fetched_urls: Set of URLs that were successfully fetched

    Returns:
        VerificationResult with detailed breakdown
    """
    # Normalize all URLs for comparison
    normalized_cited = {normalize_url(url) for url in cited_sources}
    normalized_searched = {normalize_url(url) for url in searched_urls}
    normalized_fetched = {normalize_url(url) for url in fetched_urls}

    # All accessed URLs (searched OR fetched)
    all_accessed = normalized_searched | normalized_fetched

    # Categorize citations
    verified = normalized_cited & all_accessed  # In report AND accessed
    unverified = normalized_cited - all_accessed  # In report but NEVER accessed
    accessed_not_cited = all_accessed - normalized_cited  # Accessed but NOT cited

    # Calculate rate
    total = len(normalized_cited)
    verified_count = len(verified)
    rate = verified_count / total if total > 0 else 0.0

    return VerificationResult(
        total_cited=total,
        verified_count=verified_count,
        verified=sorted(list(verified)),
        unverified=sorted(list(unverified)),
        accessed_not_cited=sorted(list(accessed_not_cited)),
        verification_rate=rate,
    )
