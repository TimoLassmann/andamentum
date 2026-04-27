"""Epistemic's embedding surface — re-exports from ``andamentum.core.embeddings``.

The HTTP plumbing now lives in core so every sub-module shares one
implementation. This module preserves the names epistemic callers and tests
already use (``embed_texts``, ``embed_documents``, ``_chunk_text``,
``_MAX_EMBED_CHARS``, ``_OVERLAP_CHARS``) so import paths and
``mock.patch("andamentum.epistemic.embeddings.embed_texts", ...)`` keep
working unchanged.
"""

from __future__ import annotations

from andamentum.core.embeddings import (
    DEFAULT_MAX_EMBED_CHARS as _MAX_EMBED_CHARS,
    DEFAULT_OLLAMA_URL as DEFAULT_BASE_URL,
    DEFAULT_OVERLAP_CHARS as _OVERLAP_CHARS,
    chunk_text as _chunk_text,
    embed_documents,
    embed_texts,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "embed_documents",
    "embed_texts",
    # Underscore-prefixed names retained for the few internal callsites
    # (passage_extraction.py, evidence_gathering.py) that imported them.
    "_MAX_EMBED_CHARS",
    "_OVERLAP_CHARS",
    "_chunk_text",
]
