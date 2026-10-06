"""Turning a fetched body into text: which extractor, and a fast one for PDFs.

``pdf_extraction`` names how a PDF becomes text:

- ``"docling"`` (the library default): ``deep_research.content_extractor``, which
  runs harvest's docling pipeline. Layout-aware and slow (seconds to minutes a
  paper, gigabytes of memory on a long one).
- ``"fast"``: the PDF's own text layer, read with pypdfium2 (already a
  dependency, the library docling itself sits on). Milliseconds a paper, no
  layout analysis, no OCR: a scanned PDF has no text layer and is refused
  (``ExtractionError``), so the reader falls back to the abstract instead of
  waiting for an OCR pass. Headings and tables come out as plain lines.

Only PDFs differ; HTML and other bodies go through ``extract_content`` either way.
``extraction_method`` names the extractor a body will go through, so the client
can record it per fetched URL.
"""

from __future__ import annotations

import asyncio
import re
import statistics
from typing import Literal, get_args

from andamentum.deep_research.content_extractor import (
    ExtractionError,
    extract_content,
)

PdfExtraction = Literal["docling", "fast"]
PDF_EXTRACTIONS: tuple[str, ...] = get_args(PdfExtraction)
DEFAULT_PDF_EXTRACTION: PdfExtraction = "docling"

# A text layer with fewer characters than this per page on average is a scan
# (or an image-only PDF): nothing a reader can use.
MIN_CHARS_PER_PAGE = 100

_SENTENCE_END = re.compile(r"[.!?:;]['\")\]]?$")
_HYPHEN_BREAK = re.compile(r"[A-Za-z]{2}-$")
# pdfium joins a line-end hyphen itself and leaves U+0002 where it was; control
# characters are not text.
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SHORT_LINE = 0.6  # of the page's median line length: a paragraph's last line


def extraction_method(content_type: str, *, pdf_extraction: PdfExtraction) -> str:
    """The extractor a body of *content_type* goes through (recorded per URL)."""
    lowered = content_type.lower()
    if "pdf" in lowered:
        return "pypdfium2" if pdf_extraction == "fast" else "docling"
    if "html" in lowered:
        return "harvest-html"
    return "text"


def _page_paragraphs(raw: str) -> list[str]:
    """One page's text-layer lines as paragraphs.

    A text layer has one line per visual line. A line that ends a sentence and
    is clearly shorter than the page's lines ends a paragraph; a trailing
    hyphen joins the next line's word; every other break is a space.
    """
    lines = [
        _CONTROL.sub("", line).strip() for line in raw.replace("\r", "\n").split("\n")
    ]
    lines = [line for line in lines if line]
    if not lines:
        return []
    typical = statistics.median(len(line) for line in lines)
    paragraphs: list[str] = []
    current = ""
    for line in lines:
        if current:
            if _HYPHEN_BREAK.search(current):
                current = current[:-1] + line
            else:
                current = f"{current} {line}"
        else:
            current = line
        if len(line) < _SHORT_LINE * typical and _SENTENCE_END.search(line):
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return paragraphs


def pdf_text_layer(data: bytes, *, name: str = "document.pdf") -> str:
    """The text layer of a PDF as paragraphs separated by blank lines.

    Raises ``ExtractionError`` for a PDF pypdfium2 cannot open, one with no
    pages, and one whose text layer is too thin to be text (a scan).
    """
    import pypdfium2 as pdfium

    if not data:
        raise ExtractionError(f"Empty PDF data for {name}")
    try:
        document = pdfium.PdfDocument(data)
    except Exception as exc:  # pypdfium2 raises PdfiumError and bare RuntimeError
        raise ExtractionError(f"PDF text layer unreadable for {name}: {exc}") from exc
    try:
        pages: list[str] = []
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            try:
                pages.append(textpage.get_text_bounded())
            finally:
                textpage.close()
                page.close()
    finally:
        document.close()
    if not pages:
        raise ExtractionError(f"PDF {name} has no pages")
    total = sum(len(page.strip()) for page in pages)
    if total < MIN_CHARS_PER_PAGE * len(pages):
        raise ExtractionError(
            f"PDF {name} has no usable text layer ({total} characters over "
            f"{len(pages)} pages); fast extraction does not OCR"
        )
    paragraphs = [p for page in pages for p in _page_paragraphs(page)]
    return "\n\n".join(paragraphs).strip()


async def extract_fast_pdf(
    data: bytes, content_type: str, url: str, *, name: str | None = None
) -> str:
    """:func:`pdf_text_layer` off the event loop (a long PDF takes a moment)."""
    del content_type, url
    return await asyncio.to_thread(pdf_text_layer, data, name=name or "document.pdf")


async def extract_body(
    data: bytes, content_type: str, url: str, *, pdf_extraction: PdfExtraction
) -> str:
    """The text of a fetched body: the fast text layer for a PDF when asked,
    ``extract_content`` for everything else."""
    if pdf_extraction == "fast" and "pdf" in content_type.lower():
        return await extract_fast_pdf(
            data, content_type, url, name=url.rsplit("/", 1)[-1] or None
        )
    return await extract_content(data, content_type, url)
