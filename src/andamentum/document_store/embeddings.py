"""Embedding service using Ollama.

Standalone embedding service for document-store package.
Handles model-specific formatting (e.g., embeddinggemma prefixes)
and token truncation.
"""

import asyncio
from typing import List, Literal, Optional

import httpx
import numpy as np


class EmbeddingService:
    """Ollama-compatible embedding service for RAG."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        max_concurrent: int = 10,
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _format_text(
        self,
        text: str,
        text_type: Literal["query", "document"] = "document",
        title: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Format text for embedding (pass-through — no local model prefixes)."""
        return text

    async def embed_batch(
        self,
        texts: List[str],
        text_type: Literal["query", "document"] = "document",
    ) -> List[List[float]]:
        """Embed many texts in a single Ollama ``/api/embed`` request.

        One HTTP round-trip and one model batch for the whole list, rather than
        one request per text. The returned vectors are in the same order as
        ``texts``. Returns ``[]`` for an empty input.
        """
        if not texts:
            return []
        formatted = [self._format_text(t, text_type) for t in texts]
        async with self._semaphore:
            response = await self.client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": formatted},
            )
            response.raise_for_status()
            return response.json()["embeddings"]

    async def embed_text(
        self,
        text: str,
        text_type: Literal["query", "document"] = "document",
        title: Optional[str] = None,
    ) -> List[float]:
        """Get embedding for a single text (one ``/api/embed`` call)."""
        embeddings = await self.embed_batch([text], text_type)
        return embeddings[0]

    async def embed_texts(
        self,
        texts: List[str],
        text_type: Literal["query", "document"] = "document",
        title: Optional[str] = None,
    ) -> List[List[float]]:
        """Get embeddings for multiple texts (batched into one ``/api/embed``)."""
        return await self.embed_batch(list(texts), text_type)

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        return float(dot_product / norm_product)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
