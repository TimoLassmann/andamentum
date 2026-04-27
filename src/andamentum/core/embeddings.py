"""Shared Ollama embedding client.

Single home for the HTTP plumbing every sub-module needs:
- ``make_ollama_embedder`` — long-lived client + semaphore, for code paths
  that issue many small embed calls (chunker's semantic split).
- ``embed_texts`` — one-shot fresh-client batch embed with a char-cap safety
  net, for code paths that embed a small batch then move on.
- ``embed_documents`` — embed long documents by chunking, returning
  per-document chunk embeddings (max-sim pairwise comparison).
- ``cosine_similarity`` — with the 1e-8 epsilon every callsite uses.
- ``chunk_text`` — overlapping-window splitter aligned with document_store's
  500-token convention (≈2000 chars).

Sub-modules (chunker, epistemic, ...) re-export the names they need from
here so callers don't have to know which module owns the embedder.

Architecture: shared infrastructure, lazy httpx import, no andamentum deps.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import httpx
import numpy as np

# Public type alias: any caller can supply their own embedder by matching this.
EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"

# Aligned with document_store's 500-token / 4-chars-per-token chunking.
DEFAULT_MAX_EMBED_CHARS = 2000
DEFAULT_OVERLAP_CHARS = 200


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for empty input."""
    if not a or not b:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-8
    return float(np.dot(av, bv) / denom)


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_EMBED_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping char-windows for embedding."""
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


def make_ollama_embedder(
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    max_concurrent: int = 8,
    timeout: float = 60.0,
) -> EmbeddingFn:
    """Build a long-lived embedder bound to one Ollama HTTP client.

    The returned coroutine accepts a list of strings and returns one vector
    per string. Concurrency is bounded so we don't overwhelm Ollama on long
    documents. The client lives for the life of the embedder — pair this
    with code paths that embed many batches before tear-down.
    """
    sem = asyncio.Semaphore(max_concurrent)
    client = httpx.AsyncClient(timeout=timeout)

    async def _one(text: str) -> list[float]:
        async with sem:
            r = await client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            r.raise_for_status()
            return r.json()["embedding"]

    async def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.gather(*(_one(t) for t in texts))

    return embed


async def embed_texts(
    texts: list[str],
    *,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    max_chars: int = DEFAULT_MAX_EMBED_CHARS,
    timeout: float = 30.0,
) -> list[list[float]]:
    """One-shot embed a batch of short texts via a fresh Ollama client.

    Each text is truncated to ``max_chars`` as a safety net. For long
    documents use ``embed_documents``.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            embeddings: list[list[float]] = []
            for text in texts:
                resp = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": text[:max_chars]},
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            return embeddings
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. "
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
    base_url: str = DEFAULT_OLLAMA_URL,
) -> list[list[list[float]]]:
    """Embed long documents by chunking, returning per-document chunk embeddings.

    Each document is split into overlapping chunks of ``DEFAULT_MAX_EMBED_CHARS``;
    chunks are embedded as a single flat batch; results are reshaped back to
    one list of chunk-vectors per input document. Callers use the chunk
    embeddings for max-sim pairwise comparison (best cosine between any
    chunk pair of two documents).
    """
    doc_chunks: list[list[str]] = [chunk_text(t) for t in texts]
    flat: list[str] = [c for chunks in doc_chunks for c in chunks]
    flat_embeds = await embed_texts(flat, model=model, base_url=base_url)

    out: list[list[list[float]]] = []
    idx = 0
    for chunks in doc_chunks:
        out.append(flat_embeds[idx : idx + len(chunks)])
        idx += len(chunks)
    return out
