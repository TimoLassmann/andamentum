"""Content extraction — delegates to ``andamentum.harvest`` internally.

This module used to own the trafilatura/Docling extraction logic directly.
That logic now lives in ``andamentum.harvest`` (which adds metadata-driven
routing and a race-and-score path for ambiguous HTML). These wrappers
preserve the deep_research public API so existing callers keep working,
while routing all real work through the shared harvest pipeline — no more
duplicated extraction code in two places.

Public API (unchanged shape, but now async):
    extract_html(html, url)            — HTML string → markdown
    extract_pdf(data, source_name)     — PDF bytes → markdown
    extract_content(data, ct, url)     — Router by content-type

    ExtractionError — re-exported from harvest for backward compat.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

# Re-export so callers that catch deep_research.ExtractionError keep working.
from andamentum.harvest import (
    ExtractionError as _HarvestExtractionError,
)
from andamentum.harvest import (
    HarvestError,
    extract_from_bytes,
)

logger = logging.getLogger(__name__)


class ExtractionError(_HarvestExtractionError):
    """Backward-compat alias for ``harvest.ExtractionError``.

    Subclassed (not aliased) so existing ``except ExtractionError`` blocks
    in deep_research callers continue to work, AND code that raises
    ``ExtractionError(...)`` (with the old single-string constructor) still
    constructs cleanly via the parent's signature.
    """


async def extract_html(html: str, url: str) -> str:
    """Extract article content from web page HTML and return clean markdown.

    Delegates to ``harvest.extract_from_bytes`` which sniffs page metadata
    (og:type / JSON-LD @type) and either dispatches to a single backend or
    races trafilatura + Docling and picks the higher-scoring output.

    Args:
        html: Raw HTML string.
        url: Page URL (used for link resolution and diagnostics).

    Returns:
        Clean markdown string.

    Raises:
        ExtractionError: If no extractable content is found.
    """
    try:
        return (
            await extract_from_bytes(
                html.encode("utf-8"), format="html", source_url=url
            )
        ).strip()
    except HarvestError as exc:
        raise ExtractionError(str(exc)) from exc


async def extract_pdf(data: bytes, source_name: str = "document.pdf") -> str:
    """Extract text from PDF bytes and return clean markdown.

    Delegates to ``harvest.extract_from_bytes`` which uses Docling.

    Args:
        data: Raw PDF bytes.
        source_name: Filename hint for diagnostics.

    Returns:
        Markdown string.

    Raises:
        ExtractionError: If PDF cannot be parsed or produces no content.
    """
    if not data:
        raise ExtractionError(f"Empty PDF data for {source_name}")
    try:
        return (
            await extract_from_bytes(data, format="pdf", source_url=source_name)
        ).strip()
    except HarvestError as exc:
        raise ExtractionError(
            f"PDF extraction failed for {source_name}: {exc}"
        ) from exc


def _parse_charset(content_type: str) -> str:
    """Extract charset from content-type header, default to utf-8."""
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip().strip('"')
    return "utf-8"


def _filename_from_url(url: str) -> str:
    """Extract filename from URL path, or return a default."""
    path = urlparse(url).path
    if path and "/" in path:
        name = path.rsplit("/", 1)[-1]
        if name:
            return name
    return "document"


async def extract_content(data: bytes, content_type: str, url: str) -> str:
    """Route raw bytes to the right extractor based on content-type.

    Inspects the content-type header and delegates:
        - HTML  → harvest (trafilatura/Docling race or metadata-routed)
        - PDF   → harvest (Docling)
        - Other → UTF-8 text decode

    Args:
        data: Raw response bytes.
        content_type: Value of the Content-Type HTTP header.
        url: The URL the content was fetched from.

    Returns:
        Extracted content as a string.

    Raises:
        ExtractionError: If HTML/PDF extraction fails.
    """
    ct_lower = content_type.lower()

    if "html" in ct_lower:
        charset = _parse_charset(content_type)
        html = data.decode(charset, errors="replace")
        return await extract_html(html, url)

    if "pdf" in ct_lower:
        source_name = _filename_from_url(url)
        return await extract_pdf(data, source_name)

    # Fallback: decode as UTF-8 text
    return data.decode("utf-8", errors="replace")
