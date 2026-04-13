"""Content extraction — route raw bytes to the right tool based on content type.

Public API:
    extract_html(html, url)     — Web page HTML → clean markdown via trafilatura
    extract_pdf(data, name)     — PDF bytes → markdown via Docling standard pipeline
    extract_content(data, ct, url) — Router: inspect content-type, delegate above

    ExtractionError             — Raised when extraction produces no usable content
"""

import logging
import unicodedata

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when content extraction fails or produces no usable content."""


def extract_html(html: str, url: str) -> str:
    """Extract article content from web page HTML and return clean markdown.

    Uses trafilatura for intelligent article extraction — strips navigation,
    sidebars, footers, and other page chrome while preserving the main content.

    Args:
        html: Raw HTML string.
        url: Page URL (used by trafilatura for link resolution).

    Returns:
        Clean markdown string of the article content.

    Raises:
        ExtractionError: If no extractable article content is found.
    """
    import trafilatura

    # Suppress noisy trafilatura warnings (e.g., "missing link attribute" on every anchor)
    logging.getLogger("trafilatura").setLevel(logging.ERROR)

    text = trafilatura.extract(
        html,
        url=url,
        include_links=True,
        include_comments=False,
        include_images=False,
        include_tables=True,
        output_format="markdown",
        favor_recall=True,
    )

    if not text:
        # Fallback: favor_precision focuses on main content block
        text = trafilatura.extract(
            html,
            url=url,
            include_links=False,
            include_comments=False,
            include_images=False,
            include_tables=True,
            favor_precision=True,
        )

    if not text:
        raise ExtractionError(f"No extractable article content found in HTML from {url}")

    return text.strip()


def extract_pdf(data: bytes, source_name: str = "document.pdf") -> str:
    """Extract text from PDF bytes and return clean markdown.

    Uses Docling's full standard pipeline with OCR and table structure
    recognition. Handles both digital-native and scanned PDFs.

    Args:
        data: Raw PDF bytes.
        source_name: Filename hint for Docling (used in logging/errors).

    Returns:
        Markdown string of the PDF content.

    Raises:
        ExtractionError: If PDF cannot be parsed or produces no content.
    """
    from io import BytesIO

    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if not data:
        raise ExtractionError(f"Empty PDF data for {source_name}")

    try:
        options = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )
        stream = DocumentStream(name=source_name, stream=BytesIO(data))
        result = converter.convert(stream)
        markdown = result.document.export_to_markdown()
    except Exception as e:
        raise ExtractionError(f"PDF extraction failed for {source_name}: {e}") from e

    if not markdown.strip():
        raise ExtractionError(f"No extractable text content in PDF {source_name}")

    return unicodedata.normalize("NFKC", markdown).strip()


def _parse_charset(content_type: str) -> str:
    """Extract charset from content-type header, default to utf-8."""
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip().strip('"')
    return "utf-8"


def _filename_from_url(url: str) -> str:
    """Extract filename from URL path, or return a default."""
    from urllib.parse import urlparse

    path = urlparse(url).path
    if path and "/" in path:
        name = path.rsplit("/", 1)[-1]
        if name:
            return name
    return "document"


def extract_content(data: bytes, content_type: str, url: str) -> str:
    """Route raw bytes to the right extractor based on content-type.

    This is the main entry point for the fetch pipeline. Inspects the
    content-type header and delegates:
        - HTML  → extract_html (trafilatura)
        - PDF   → extract_pdf  (Docling)
        - Other → UTF-8 text decode

    Args:
        data: Raw response bytes.
        content_type: Value of the Content-Type HTTP header.
        url: The URL the content was fetched from.

    Returns:
        Extracted content as a string (markdown for HTML/PDF, plain text otherwise).

    Raises:
        ExtractionError: If HTML/PDF extraction fails.
    """
    ct_lower = content_type.lower()

    if "html" in ct_lower:
        charset = _parse_charset(content_type)
        html = data.decode(charset, errors="replace")
        return extract_html(html, url)

    if "pdf" in ct_lower:
        source_name = _filename_from_url(url)
        return extract_pdf(data, source_name)

    # Fallback: decode as UTF-8 text
    return data.decode("utf-8", errors="replace")
