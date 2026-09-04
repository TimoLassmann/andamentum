"""andamentum.document_store — Personal knowledge base with 4-signal search.

A storage and search library for a personal "second brain." Stores documents
with automatic chunking and embedding — no LLM. Searches
using 4-signal Reciprocal Rank Fusion (FTS5 keyword, chunk embeddings,
doc embeddings, DHP temporal clustering).

Public API:

    from andamentum.document_store import (
        ingest,            # Store content with auto-chunking + embedding (no LLM)
        search,            # Natural language search with LLM query planning
        get_chunks,        # Every chunk of one document, in order (no LLM)
        get_chunk,         # Resolve one chunk id back to its passage (no LLM)
        find_by_metadata,  # Structured query by metadata fields (exact or set-membership)
        describe_metadata, # Discover the metadata schema (fields + value distributions)
        update_metadata,   # Update metadata on a document (with change history)
        delete,            # Soft-delete a document (can be restored)
        restore,           # Restore a soft-deleted document
        purge,             # Permanently remove old soft-deleted documents
        list_deleted,      # List soft-deleted documents (trash view)
        repair,            # Fix incomplete ingestions after crashes
        find_duplicates,   # Detect near-duplicate documents via embeddings
    )

Two-stage retrieval. ``search`` ranks DOCUMENTS by a Reciprocal Rank Fusion
sum — an ordering, not a calibrated relevance, and not comparable between
queries. When a caller needs to know which PASSAGE to read, or how many
results are actually relevant rather than merely highest-ranked, it takes a
second, deliberate step:

    from andamentum.document_store import get_chunks, rerank, select_relevant

    hits = [c for r in await search(...) for c in await get_chunks(db, doc_id=r.doc_id)]
    scores = await rerank(question, [h.content for h in hits], reranker_model=...)
    best = select_relevant(list(zip(hits, scores, strict=True)),
                           min_relevance=..., buffer=5)

``search`` never does this for you. ``rerank`` needs the optional ``rerank``
extra and loads a 0.6B cross-encoder into this process; making it a default
would charge every caller that memory and that load time, and would change
what ``search`` returns depending on which extras happen to be installed.

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
- SQLite + sqlite-vec + FTS5 (no external services except Ollama for embeddings)
- Two-phase ingest: document registered immediately (FTS5 searchable),
  chunks + embeddings stored in background (repairable if interrupted)

Requires:
- Ollama running locally with embeddinggemma:latest (for embeddings)
- pydantic-ai for query planning in search() (installed as part of andamentum).
  Ingest needs no LLM; the store does not author metadata.
"""

# === Functions you can wrap as agent tools ===
# `DocumentStore` is a class — wrap its methods (`add`, `read`, `search`,
# `delete`, `find_by_metadata`, `list_documents`, …) as tools.
# The 10 module-level functions below (ingest/search/etc.) are an alternative
# higher-level API that doesn't require holding a DocumentStore instance.
from .api import DocumentStore
from .extraction import extract_document_metadata
from .lifecycle import (
    database_exists,
    delete_database,
    get_databases_dir,
    get_db_path,
    list_databases,
)
from .public import (
    delete,
    describe_metadata,
    find_by_metadata,
    find_duplicates,
    get_chunk,
    get_chunks,
    ingest,
    ingest_source,
    list_deleted,
    pending_status,
    process_pending,
    purge,
    reembed_document,
    repair,
    restore,
    retry_failed,
    search,
    update_metadata,
)
from .reranking import (
    SUGGESTED_RERANKER_MODEL,
    rerank,
    reset_reranker_cache,
    select_relevant,
)
from .search import search_multi_database, search_unified

# === Result/data types (returned by the above; not tools themselves) ===
from .metadata_models import (
    ChunkMetadataFields,
    DocumentLLMFields,
    DocumentMetadataFields,
)
from .models import (
    Document,
    DocumentMetadata,
    ReembedResult,
    UpdateResult,
)
from .chunk_hits import ChunkHit, chunk_id_for, parse_chunk_id
from .public import (
    ConvertFn,
    DuplicateGroup,
    FieldProfile,
    MetadataFilterValue,
    MetadataScalar,
    PendingStatus,
    ProcessMode,
    ProcessReport,
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
    "database_exists",
    "delete",
    "delete_database",
    "describe_metadata",
    "extract_document_metadata",
    "find_by_metadata",
    "find_duplicates",
    "get_chunk",
    "get_chunks",
    "get_databases_dir",
    "get_db_path",
    "ingest",
    "ingest_source",
    "list_databases",
    "list_deleted",
    "pending_status",
    "process_pending",
    "purge",
    "reembed_document",
    "repair",
    "rerank",
    "reset_reranker_cache",
    "restore",
    "retry_failed",
    "search",
    "select_relevant",
    "search_multi_database",
    "search_unified",
    "update_metadata",
    # Data types
    "SUGGESTED_RERANKER_MODEL",
    "ChunkHit",
    "ChunkMetadataFields",
    "ConvertFn",
    "chunk_id_for",
    "Document",
    "DocumentLLMFields",
    "DocumentMetadata",
    "DocumentMetadataFields",
    "DuplicateGroup",
    "FieldProfile",
    "MetadataFilterValue",
    "MetadataScalar",
    "MultiDatabaseSearchResult",
    "PendingStatus",
    "ProcessMode",
    "ProcessReport",
    "ReembedResult",
    "RepairReport",
    "SearchResult",
    "SearchResultMetadata",
    "UnifiedSearchResult",
    "UpdateResult",
    "parse_chunk_id",
]
