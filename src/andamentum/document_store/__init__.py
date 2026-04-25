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

# === Functions you can wrap as agent tools ===
# `DocumentStore` is a class — wrap its methods (`add`, `read`, `search`,
# `delete`, `find_by_metadata`, `list_documents`, …) as tools.
# The 10 module-level functions below (ingest/search/etc.) are an alternative
# higher-level API that doesn't require holding a DocumentStore instance.
from .api import DocumentStore
from .chunking import chunk_markdown
from .extraction import extract_chunk_metadata, extract_document_metadata
from .lifecycle import (
    database_exists,
    delete_database,
    get_databases_dir,
    get_db_path,
    list_databases,
)
from .public import (
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
from .search import search_multi_database, search_unified

# === Result/data types (returned by the above; not tools themselves) ===
from .chunking import Chunk
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
from .public import (
    DuplicateGroup,
    MetadataFilterValue,
    RepairReport,
    SearchResult,
)
from .search import (
    MultiDatabaseSearchResult,
    SearchResultMetadata,
    UnifiedSearchResult,
)

__all__ = [
    # Functions / callables
    "DocumentStore",
    "chunk_markdown",
    "database_exists",
    "delete",
    "delete_database",
    "extract_chunk_metadata",
    "extract_document_metadata",
    "find_by_metadata",
    "find_duplicates",
    "get_databases_dir",
    "get_db_path",
    "ingest",
    "list_databases",
    "list_deleted",
    "purge",
    "repair",
    "restore",
    "search",
    "search_multi_database",
    "search_unified",
    "update_metadata",
    # Data types
    "Chunk",
    "ChunkLLMFields",
    "ChunkMetadataFields",
    "Document",
    "DocumentLLMFields",
    "DocumentMetadata",
    "DocumentMetadataFields",
    "DocumentType",
    "DuplicateGroup",
    "MetadataFilterValue",
    "MultiDatabaseSearchResult",
    "ReembedResult",
    "RepairReport",
    "SearchResult",
    "SearchResultMetadata",
    "UnifiedSearchResult",
    "UpdateResult",
]
