"""Extract enricher: title, projects and people by an LLM (opt-in)."""

from __future__ import annotations

from ..extraction import extract_document_metadata
from ..store import Store
from ._base import EnrichResult, load, write_key

KEY = "extracted"


async def enrich_extract(store: Store, doc_id: str, *, model: str) -> EnrichResult:
    """Write ``title``, ``projects``, ``people`` and ``model`` under ``"extracted"``.

    ``model`` is required: unlike ``extract_document_metadata`` there is no
    model-free default, because a document marked extracted with empty fields
    would be a silent wrong answer. An unreachable model raises
    ``ExtractionUnavailable``.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("enrich_extract needs a model id")
    doc = await load(store, doc_id)
    fields = await extract_document_metadata(doc.text, model=model)
    value = fields.model_dump(mode="json")
    value["model"] = model
    return await write_key(store, doc, key=KEY, value=value)
