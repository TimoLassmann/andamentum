"""Embedding generation wrapper.

Standalone embedding functions for the document-store package.

Usage:
    from document_store.rag.embeddings import generate_embedding, generate_embeddings

    # For search queries
    embedding = await generate_embedding("machine learning", text_type="query")

    # For documents to be searched
    embeddings = await generate_embeddings(["text 1", "text 2"], text_type="document")
"""

from typing import List, Literal, Optional
from .._defaults import DEFAULT_EMBEDDING_MODEL


async def generate_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    text_type: Literal["query", "document"] = "query",
    title: Optional[str] = None,
) -> List[float]:
    """Generate embedding for text (standalone function).

    Args:
        text: Text to embed
        model: Embedding model name
        text_type: "query" for search queries, "document" for content to be searched
        title: Optional document title (only used for documents with embeddinggemma)

    Returns:
        768-dimensional embedding vector
    """
    from ..embeddings import EmbeddingService

    service = EmbeddingService(model=model)
    try:
        return await service.embed_text(text, text_type=text_type, title=title)
    finally:
        await service.close()


async def generate_embeddings(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    text_type: Literal["query", "document"] = "document",
    title: Optional[str] = None,
) -> List[List[float]]:
    """Generate embeddings for multiple texts (batch).

    Args:
        texts: List of texts to embed
        model: Embedding model name
        text_type: "query" for search queries, "document" for content to be searched
        title: Optional document title (only used for documents with embeddinggemma)

    Returns:
        List of 768-dimensional embedding vectors
    """
    from ..embeddings import EmbeddingService

    service = EmbeddingService(model=model)
    try:
        return await service.embed_texts(texts, text_type=text_type, title=title)
    finally:
        await service.close()
