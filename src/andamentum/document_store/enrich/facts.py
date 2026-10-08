"""Facts enricher: size and shape of the text."""

from __future__ import annotations

from andamentum.chunker.structural import find_headings

from ..store import Store
from ._base import EnrichResult, load, write_key

KEY = "facts"


async def enrich_facts(store: Store, doc_id: str) -> EnrichResult:
    """Write chars, words, lines, markdown headings and media type under ``"facts"``.

    ``lines`` counts newline-separated lines (0 for empty text). No model.
    """
    doc = await load(store, doc_id)
    text = doc.text
    value = {
        "chars": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()),
        "headings": len(find_headings(text)),
        "media_type": doc.media_type,
        "source": "andamentum.document_store.enrich.facts",
    }
    return await write_key(store, doc, key=KEY, value=value)
