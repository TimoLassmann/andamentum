"""Single-call embedding wrapper.

Used by search.py to embed a query string when no pre-computed embedding
is supplied. Creates and tears down an EmbeddingService per call.
"""

from typing import List, Literal, Optional


async def generate_embedding(
    text: str,
    *,
    model: str,
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
