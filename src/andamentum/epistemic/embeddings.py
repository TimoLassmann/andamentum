"""Embedding client for evidence clustering.

Uses an HTTP-compatible embedding endpoint. Long texts are chunked
before embedding (aligned with document-store's 500-token chunk size).

Architecture: Layer 1 (framework-agnostic, async HTTP only)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"

# Chunk size aligned with document-store (500 tokens × 4 chars/token).
_MAX_EMBED_CHARS = 2000
_OVERLAP_CHARS = 200


def _chunk_text(text: str, max_chars: int = _MAX_EMBED_CHARS, overlap: int = _OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    if not text or len(text) <= max_chars:
        return [text or ""]
    chunks: list[str] = []
    stride = max_chars - overlap
    for start in range(0, len(text), stride):
        chunk = text[start : start + max_chars]
        if chunk.strip():
            chunks.append(chunk)
        if start + max_chars >= len(text):
            break
    return chunks or [text[:max_chars]]


async def embed_texts(
    texts: list[str],
    *,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
) -> list[list[float]]:
    """Embed short texts using Ollama (one embedding per text).

    Texts are truncated to _MAX_EMBED_CHARS as a safety net.
    For long documents, use embed_documents() instead.

    Raises:
        ImportError: If httpx is not installed
        RuntimeError: If Ollama is unreachable or returns an error
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            embeddings = []
            for text in texts:
                resp = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": text[:_MAX_EMBED_CHARS]},
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            return embeddings
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. "
            f"Embedding is required for assertion clustering. "
            f"Start Ollama with: ollama serve"
        ) from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Ollama embedding failed (HTTP {e.response.status_code}). "
            f"Ensure model '{model}' is pulled: ollama pull {model}"
        ) from e


async def embed_documents(
    texts: list[str],
    *,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
) -> list[list[list[float]]]:
    """Embed long documents by chunking, returning per-document chunk embeddings.

    Each document is split into overlapping chunks of ~2000 chars, each chunk
    is embedded independently.  Callers use the chunk embeddings for max-sim
    pairwise comparison (best cosine between any chunk pair of two documents).

    Returns:
        List of length len(texts), where each element is a list of chunk
        embeddings for that document.
    """
    doc_chunks: list[list[str]] = [_chunk_text(t) for t in texts]

    # Flatten for a single batch call
    all_chunks: list[str] = [chunk for chunks in doc_chunks for chunk in chunks]
    all_embeddings = await embed_texts(all_chunks, model=model, base_url=base_url)

    # Unflatten back to per-document
    result: list[list[list[float]]] = []
    idx = 0
    for chunks in doc_chunks:
        result.append(all_embeddings[idx : idx + len(chunks)])
        idx += len(chunks)
    return result

