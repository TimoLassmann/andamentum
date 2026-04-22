"""andamentum.document_store — Personal knowledge base with 4-signal search.

A storage and search library for a personal "second brain." Stores documents
with automatic chunking, embedding, and LLM metadata extraction. Searches
using 4-signal Reciprocal Rank Fusion (FTS5 keyword, chunk embeddings,
doc embeddings, DHP temporal clustering).

Public API (10 functions — all you need):

    from andamentum.document_store import (
        ingest,            # Store content with auto-chunking and metadata extraction
        search,            # Natural language search with LLM query planning
        find_by_metadata,  # Structured query by exact metadata fields
        update_metadata,   # Update metadata on a document (with change history)
        delete,            # Soft-delete a document (can be restored)
        restore,           # Restore a soft-deleted document
        purge,             # Permanently remove old soft-deleted documents
        list_deleted,      # List soft-deleted documents (trash view)
        repair,            # Fix incomplete ingestions after crashes
        find_duplicates,   # Detect near-duplicate documents via embeddings
    )

Quick start:

    doc_id = await ingest("brain", "I think MAP-Elites could work for antibody optimization")
    results = await search("brain", "What have I captured about MAP-Elites?")
    await update_metadata("brain", doc_id, {"record_type": "idea", "status": "exploring"})
    tasks = await find_by_metadata("brain", {"record_type": "task", "status": "todo"})
    await delete("brain", doc_id)
    report = await repair("brain")
    dupes = await find_duplicates("brain")

Architecture:
- Named databases: ~/.local/share/document-store/{name}.db (override with DOCUMENT_STORE_DIR env var)
- SQLite + sqlite-vec + FTS5 (no external services except Ollama for embeddings/LLM)
- Two-phase ingest: document registered immediately (FTS5 searchable),
  chunks + embeddings stored in background (repairable if interrupted)

Requires:
- Ollama running locally with embeddinggemma:latest (for embeddings)
- pydantic-ai for metadata extraction and query planning (installed as part of andamentum)
"""

# Public API
from .public import (
    DuplicateGroup,
    MetadataFilterValue,
    RepairReport,
    SearchResult,
    delete,
    find_by_metadata,
    find_duplicates,
    ingest,
    list_deleted,
    purge,
    repair,
    restore,
    search,
    update_metadata,
)

# Low-level API for power users
from .api import DocumentStore
from .chunking import Chunk, chunk_markdown
from .extraction import extract_chunk_metadata, extract_document_metadata
from .lifecycle import (
    database_exists,
    delete_database,
    get_databases_dir,
    get_db_path,
    list_databases,
)
from .metadata_models import (
    ChunkLLMFields,
    ChunkMetadataFields,
    DocumentLLMFields,
    DocumentMetadataFields,
)
from .models import (
    Document,
    DocumentMetadata,
    DocumentType,
    ReembedResult,
    UpdateResult,
)
from .search import (
    MultiDatabaseSearchResult,
    SearchResultMetadata,
    UnifiedSearchResult,
    search_multi_database,
    search_unified,
)

__all__ = [
    # Public API
    "ingest",
    "search",
    "find_by_metadata",
    "update_metadata",
    "delete",
    "restore",
    "purge",
    "list_deleted",
    "repair",
    "find_duplicates",
    "SearchResult",
    "RepairReport",
    "DuplicateGroup",
    "MetadataFilterValue",
    # Low-level API
    "DocumentStore",
    # Chunking
    "Chunk",
    "chunk_markdown",
    # Metadata models
    "DocumentMetadataFields",
    "DocumentLLMFields",
    "ChunkMetadataFields",
    "ChunkLLMFields",
    # Metadata extraction
    "extract_document_metadata",
    "extract_chunk_metadata",
    # Data models
    "Document",
    "DocumentMetadata",
    "DocumentType",
    "ReembedResult",
    "UpdateResult",
    # Search result models
    "SearchResultMetadata",
    "UnifiedSearchResult",
    "MultiDatabaseSearchResult",
    # Search functions
    "search_unified",
    "search_multi_database",
    # Database management
    "get_db_path",
    "get_databases_dir",
    "list_databases",
    "database_exists",
    "delete_database",
]
