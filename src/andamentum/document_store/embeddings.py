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

    async def embed_text(
        self,
        text: str,
        text_type: Literal["query", "document"] = "document",
        title: Optional[str] = None,
    ) -> List[float]:
        """Get embedding for a single text."""
        formatted_text = self._format_text(text, text_type, title)
        async with self._semaphore:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": formatted_text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_texts(
        self,
        texts: List[str],
        text_type: Literal["query", "document"] = "document",
        title: Optional[str] = None,
    ) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        tasks = [self.embed_text(text, text_type, title) for text in texts]
        return await asyncio.gather(*tasks)

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        return float(dot_product / norm_product)

    def mean_knn_distance(
        self, embedding: List[float], archive: List[List[float]], k: int = 10
    ) -> float:
        """Calculate mean distance to k nearest neighbors."""
        if not archive:
            return 1.0

        similarities = [self.cosine_similarity(embedding, vec) for vec in archive]
        similarities_sorted = sorted(similarities, reverse=True)[
            : min(k, len(similarities))
        ]
        distances = [1 - sim for sim in similarities_sorted]
        return sum(distances) / len(distances) if distances else 1.0

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
