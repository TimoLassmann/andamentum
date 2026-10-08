"""Identifiers enricher: DOI, PMID, arXiv, PMCID, ISBN and URLs found in the text."""

from __future__ import annotations

from andamentum.references import extract_identifiers

from ..store import Store
from ._base import EnrichResult, load, write_key

KEY = "identifiers"
SOURCE = "andamentum.references.extract_identifiers"


async def enrich_identifiers(store: Store, doc_id: str) -> EnrichResult:
    """Write the non-empty identifier fields under ``metadata["identifiers"]``.

    No model, no network. ``urls`` is a list; ``source`` names the extractor.
    """
    doc = await load(store, doc_id)
    found = extract_identifiers(doc.text)
    value: dict[str, object] = {}
    for name in ("doi", "pmid", "arxiv", "pmcid", "isbn"):
        field = getattr(found, name)
        if field:
            value[name] = field
    if found.urls:
        value["urls"] = list(found.urls)
    value["source"] = SOURCE
    return await write_key(store, doc, key=KEY, value=value)
