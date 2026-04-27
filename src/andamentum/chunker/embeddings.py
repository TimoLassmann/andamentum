"""Minimal embedding client for the chunker.

Lives here (not in core) so the chunker stays self-contained — it only
needs embeddings for one specific job (splitting oversized sections via
cosine similarity) and pulling in `document_store.EmbeddingService` would
violate the layering rule. ~30 lines, no dependencies beyond httpx.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import httpx
import numpy as np

# Public type: any caller can supply their own embedder by matching this.
EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]

DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def make_ollama_embedder(
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    max_concurrent: int = 8,
) -> EmbeddingFn:
    """Build an EmbeddingFn that talks to a local Ollama server.

    The returned coroutine accepts a list of strings and returns a list of
    vectors of equal length. Concurrency is bounded so we don't overwhelm
    Ollama on long documents.
    """
    sem = asyncio.Semaphore(max_concurrent)
    client = httpx.AsyncClient(timeout=60.0)

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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0 for the empty case."""
    if not a or not b:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-8
    return float(np.dot(av, bv) / denom)
