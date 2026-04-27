"""Chunker's embedding surface — re-exports from ``andamentum.core.embeddings``.

The chunker only needs three names (``EmbeddingFn``, ``make_ollama_embedder``,
``cosine_similarity``); they all live in core now so every sub-module shares
one HTTP client implementation. Kept as a re-export shim so existing
``from .embeddings import ...`` callers don't change.
"""

from __future__ import annotations

from andamentum.core.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_URL,
    EmbeddingFn,
    cosine_similarity,
    make_ollama_embedder,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_OLLAMA_URL",
    "EmbeddingFn",
    "cosine_similarity",
    "make_ollama_embedder",
]
