"""PDF / HTML / DOCX / PPTX → markdown via Docling.

Docling preserves layout and emits a typed `DoclingDocument` tree which
exports cleanly to markdown with `##` headings. Slower than trafilatura
on HTML (~3-8s cold-start vs ~50ms) but understands non-article layouts
(homepages, link cards, list views) that trafilatura collapses to soup.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Literal

from ..errors import ExtractionError

DoclingFormat = Literal["pdf", "html", "docx", "pptx"]


async def extract(data: bytes, source_url: str, fmt: DoclingFormat = "html") -> str:
    """Run Docling on raw bytes; return markdown.

    `fmt` tells Docling which input backend to use. The orchestrator passes
    in the format detected upstream so we don't need to re-sniff here.
    """
    return await asyncio.to_thread(_extract_sync, data, source_url, fmt)


def _extract_sync(data: bytes, source_url: str, fmt: DoclingFormat) -> str:
    # Imports are deferred so this module is cheap to import at app startup.
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.document_converter import DocumentConverter

    fmt_map = {
        "pdf": InputFormat.PDF,
        "html": InputFormat.HTML,
        "docx": InputFormat.DOCX,
        "pptx": InputFormat.PPTX,
    }
    if fmt not in fmt_map:
        raise ExtractionError(
            f"docling backend does not support format {fmt!r}",
            attempted=["docling"],
        )

    converter = DocumentConverter(allowed_formats=[fmt_map[fmt]])
    # Give docling a stable filename hint so it picks the right backend.
    name = f"document.{fmt}"
    stream = DocumentStream(name=name, stream=BytesIO(data))
    try:
        result = converter.convert(stream)
    except Exception as exc:  # docling raises a bag of types
        raise ExtractionError(
            f"docling conversion failed: {exc}",
            attempted=["docling"],
            diagnostics={"docling": f"{type(exc).__name__}: {exc}"},
        ) from exc

    md = result.document.export_to_markdown()
    if not md or not md.strip():
        raise ExtractionError(
            "docling returned empty markdown",
            attempted=["docling"],
            diagnostics={"docling": "empty output"},
        )
    return md
