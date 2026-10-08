"""Any source (PDF, DOCX, HTML, URL, ...) into the store, through ``andamentum.harvest``.

This module is the ONLY place ``harvest`` is imported, and it imports it
lazily, so a caller that already has text never pulls in Docling. ``ingest``
converts one source to markdown, keeps the original bytes beside the text,
and stores both with ``Store.put``. It calls no model besides harvest's own
converter; indexing happens later, in ``Store.sync``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .store import Store

#: Media types recorded for the original bytes, by harvest's detected format.
_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "html": "text/html",
    "markdown": "text/markdown",
    "plain": "text/plain",
}


async def ingest(
    store: Store,
    source: str | Path,
    *,
    key: str | None = None,
    metadata: dict[str, Any] | None = None,
    mode: Literal["fast", "accurate"] = "fast",
) -> str:
    """Convert ``source`` (a path or URL) to markdown and store it; returns the doc id.

    The original bytes are kept (``Store.original``). ``metadata["source"]``
    records where the text came from: ``{"location": ..., "format": ...,
    "media_type": ...}``; any other metadata is the caller's. A conversion
    error is raised; nothing is stored then.
    """
    from andamentum.harvest import extract_from_bytes
    from andamentum.harvest.fetch import resolve

    if isinstance(source, str) and "://" not in source:
        source = Path(source).expanduser()
    fetched = await resolve(source)
    text = await extract_from_bytes(
        fetched.data, format=fetched.format, source_url=fetched.source_url, mode=mode
    )
    meta = dict(metadata or {})
    if "source" in meta:
        raise ValueError("metadata['source'] is set by ingest; use another key")
    meta["source"] = {
        "location": fetched.source_url,
        "format": str(fetched.format),
        "media_type": _MEDIA_TYPES.get(str(fetched.format), "application/octet-stream"),
    }
    return await store.put(
        text, key=key, media_type="text/markdown", metadata=meta, original=fetched.data
    )


__all__ = ["ingest"]
