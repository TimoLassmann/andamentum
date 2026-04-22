"""RAG internals - chunk storage, embedding search, and reranking.

For document management, use DocumentStore:

    from andamentum.document_store import DocumentStore

    store = DocumentStore.for_database("brain")
    await store.initialize()
    doc_id = await store.register_document("title", content="...")
    results = await store.search("query")

This module provides internal components for:
- Chunk and embedding storage (sqlite-vec)
- Vector similarity search
- Cross-encoder reranking
"""

from .database import (
    delete_chunks_for_document,
    get_document_stats,
    search_chunks,
    store_chunk_for_document,
)
from .embeddings import (
    generate_embedding,
    generate_embeddings,
)
from .search import (
    SearchConfig,
    SearchResult,
    semantic_search,
)

__all__ = [
    # Database operations
    "search_chunks",
    "get_document_stats",
    "store_chunk_for_document",
    "delete_chunks_for_document",
    # Search
    "semantic_search",
    "SearchConfig",
    "SearchResult",
    # Embeddings
    "generate_embedding",
    "generate_embeddings",
]
