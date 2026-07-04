"""Public API for the document store.

Core functions:
    ingest(database, content) → doc_id
    search(database, query) → list[SearchResult]
    find_by_metadata(database, filters) → list[SearchResult]
    describe_metadata(database) → dict[str, FieldProfile]
    update_metadata(database, doc_id, metadata) → bool
    delete(database, doc_id) → bool
    repair(database) → RepairReport
    find_duplicates(database) → list[DuplicateGroup]

Read side — three uniform functions for humans and agents alike:
    search()             unstructured recall over content (NL + RRF ranking)
    find_by_metadata()   structured query — exact match or set-membership
    describe_metadata()  discover the schema-less metadata vocabulary

The store stays domain-agnostic: consumers define their own metadata
vocabulary and build their own tools on top of these primitives (e.g. a task
layer wrapping find_by_metadata), rather than the store growing a function per
consumer.

Usage:
    from andamentum.document_store import ingest, search, find_by_metadata, describe_metadata

    doc_id = await ingest("brain", "I think MAP-Elites could work for antibody optimization")
    results = await search("brain", "What decisions did I make about GROVE last month?")

    # Discover what's queryable, then filter — no prior knowledge of the schema:
    schema = await describe_metadata("brain", filters={"record_type": "task"})
    open_tasks = await find_by_metadata("brain", {
        "record_type": "task",
        "status": ["todo", "in_progress", "blocked"],  # set-membership
    })
    await update_metadata("brain", doc_id, {"status": "done"})
    await delete("brain", doc_id)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional


from andamentum.chunker import extract_units
from andamentum.core.embeddings import make_ollama_embedder

from .api import DocumentStore
from .chunker_adapter import units_to_chunks
from .extraction import extract_chunk_metadata, extract_document_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A single scalar metadata value (exact match; None matches SQL NULL).
MetadataScalar = str | int | bool | None

#: Allowed value types for :func:`find_by_metadata` filter predicates.
#: All conditions are AND-ed. A scalar matches by equality; a list/tuple/set
#: matches by set-membership (``field IN (...)``).
MetadataFilterValue = MetadataScalar | list[MetadataScalar]

# Module-level caches
_stores: dict[str, DocumentStore] = {}
_preflight_done: set[str] = set()

#: Max chunks processed concurrently during phase-2 ingest. Each unit of work
#: is one embedding call + one LLM metadata-extraction call; bounding it keeps
#: the embedding/LLM backend from being flooded while still collapsing the
#: previously-serial per-chunk round-trips into a handful of waves.
_INGEST_CONCURRENCY = 5


async def _get_store(database: str) -> DocumentStore:
    """Get or create an initialized DocumentStore."""
    if database not in _stores:
        store = DocumentStore.for_database(database)
        await store.initialize()
        _stores[database] = store
    return _stores[database]


async def _preflight(database: str, model: str, embedding_model: str) -> None:
    """Test that embedding service and LLM are reachable. Raises on failure.

    Called once per (database, model, embedding_model) combination.
    """
    cache_key = f"{database}:{model}:{embedding_model}"
    if cache_key in _preflight_done:
        return

    from .embeddings import EmbeddingService

    embed_svc = EmbeddingService(model=embedding_model)
    try:
        await embed_svc.embed_text("preflight", text_type="query")
    except Exception as e:
        raise RuntimeError(
            f"Embedding service unavailable (model={embedding_model}). Is Ollama running? Error: {e}"
        ) from e
    finally:
        await embed_svc.close()

    try:
        from pydantic_ai import Agent

        agent: Agent[None, str] = Agent(model, output_type=str)
        await agent.run("Reply with exactly: ok")
    except ImportError:
        raise RuntimeError(
            "pydantic-ai not installed. Install with: pip install andamentum"
        )
    except Exception as e:
        raise RuntimeError(f"LLM model '{model}' unavailable. Error: {e}") from e

    _preflight_done.add(cache_key)
    logger.info(
        f"Preflight passed: database={database}, model={model}, embeddings={embedding_model}"
    )

    # Run repair on first access — fix any incomplete ingestions from previous crashes
    store = await _get_store(database)
    report = await _repair_incomplete(store, model, embedding_model)
    if report.documents_repaired > 0:
        logger.info(
            f"Auto-repair: fixed {report.documents_repaired} incomplete documents in '{database}'"
        )
    if report.documents_failed > 0:
        logger.warning(
            f"Auto-repair: {report.documents_failed} documents could not be repaired in '{database}'"
        )


# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A search result from the knowledge base."""

    doc_id: str
    title: str
    snippet: str
    score: float
    metadata: dict = field(default_factory=dict)
    match_type: str = ""
    warning: str = ""


@dataclass
class DuplicateGroup:
    """A group of documents that are near-duplicates based on embedding similarity."""

    doc_ids: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    similarity: float = 0.0


@dataclass
class FieldProfile:
    """Profile of one metadata field across a database (or a filtered subset).

    Returned by :func:`describe_metadata`. ``values`` is populated only for
    low-cardinality fields (``distinct <= max_values``); for high-cardinality
    fields (ids, titles, free text) it is None so the output stays bounded —
    ``present_in`` and ``distinct`` are always available.
    """

    present_in: int
    """Number of documents that carry this field."""

    distinct: int
    """Number of distinct values the field takes."""

    values: dict[str, int] | None = None
    """value -> occurrence count, or None when distinct > max_values."""


@dataclass
class RepairReport:
    """Report from a repair() run."""

    documents_scanned: int = 0
    documents_incomplete: int = 0
    documents_repaired: int = 0
    documents_failed: int = 0
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------


async def ingest(
    database: str,
    content: str,
    title: str | None = None,
    source: str = "manual",
    metadata: dict | None = None,
    *,
    model: str,
    embedding_model: str,
) -> str:
    """Add content to the knowledge base. Returns doc_id.

    Two-phase design:
    - Phase 1 (atomic): Register document in DB. FTS5 indexes immediately.
      The document is keyword-searchable the moment this returns.
    - Phase 2 (can fail): Chunk, embed, extract metadata, store chunks.
      If this fails, repair() can re-run it later.

    Args:
        database: Database name (e.g., "brain")
        content: Text content (markdown or plain text)
        title: Optional title. If None, LLM generates one.
        source: Where content came from (manual, slack, claude_code, zotero, voice)
        metadata: Optional dict merged with LLM-extracted metadata. Caller values win.
        model: LLM model for metadata extraction.
        embedding_model: Embedding model.

    Returns:
        Document ID (UUID string)

    Raises:
        RuntimeError: If embedding service or LLM model is unavailable.
    """
    store = await _get_store(database)
    await _preflight(database, model, embedding_model)

    # --- Phase 1: Register document (atomic, FTS5 immediate) ---
    doc_meta = await extract_document_metadata(content, model=model)
    doc_meta.source = source
    if title:
        doc_meta.title = title
    elif not doc_meta.title:
        first_line = next(
            (
                line.strip().lstrip("#").strip()
                for line in content.split("\n")
                if line.strip()
            ),
            "Untitled",
        )
        doc_meta.title = first_line

    doc_meta_dict = doc_meta.model_dump(
        mode="json"
    )  # datetime → ISO string for JSON storage
    if metadata:
        doc_meta_dict.update(metadata)

    doc_id = await store.register_document(
        title=doc_meta.title,
        content=content,
        metadata=doc_meta_dict,
    )

    # --- Phase 2: Chunk, embed, store (can be repaired if interrupted) ---
    await _run_phase2(store, doc_id, content, doc_meta.title, model, embedding_model)

    logger.info(f"Ingested '{doc_meta.title}' into {database}: doc_id={doc_id}")
    return doc_id


async def _run_phase2(
    store: DocumentStore,
    doc_id: str,
    content: str,
    title: str,
    model: str,
    embedding_model: str,
) -> None:
    """Phase 2 of ingestion: chunk, embed, extract metadata, store.

    Separated so repair() can re-run this for incomplete documents.
    Idempotent: deletes existing chunks before re-storing.

    Chunking uses ``andamentum.chunker.extract_units`` with
    ``target_max_chars=4000`` — paragraph-of-paragraphs sizing tuned for the
    chunk-level metadata extractor's ``topics`` / ``has_decision`` /
    ``has_action_item`` fields, which were validated on ~2k char chunks.
    """
    from .chunker_adapter import Chunk
    from .embeddings import EmbeddingService

    # Delete any existing chunks (idempotent for repair)
    await store.delete_chunks(doc_id)

    # Stage-2 semantic split (when needed) re-uses the same Ollama embedder
    # the chunk-level loop uses below.
    embedder = make_ollama_embedder(model=embedding_model)
    chunking = await extract_units(
        content,
        target_min_chars=1500,
        target_max_chars=4000,
        embedding_fn=embedder,
    )
    chunks = units_to_chunks(content, chunking.units)
    if not chunks:
        chunks = [
            Chunk(
                text=content,
                section_path="",
                chunk_index=0,
                start_char=0,
                end_char=len(content),
            )
        ]

    # Phase-2 work runs concurrently instead of one chunk at a time:
    #   * all chunk embeddings in a single batched /api/embed call,
    #   * per-chunk LLM metadata extraction, bounded by _INGEST_CONCURRENCY,
    #   * the doc-level embedding,
    # then ordered writes. Previously each chunk was embedded, extracted, and
    # written serially — the per-chunk LLM extraction dominated ingest latency.
    embed_svc = EmbeddingService(model=embedding_model)
    try:
        sem = asyncio.Semaphore(_INGEST_CONCURRENCY)

        async def _extract(chunk):  # type: ignore[no-untyped-def]
            """Extract chunk metadata (no DB writes), bounded by the semaphore."""
            async with sem:
                chunk_meta = await extract_chunk_metadata(chunk.text, model=model)
            chunk_meta.parent_doc_id = doc_id
            chunk_meta.section_path = chunk.section_path
            chunk_meta.chunk_index = chunk.chunk_index
            return chunk_meta

        async def _embed_doc_level() -> None:
            """Doc-level embedding — skip gracefully if too large for the model."""
            try:
                doc_emb = await embed_svc.embed_text(
                    content, text_type="document", title=title
                )
                await store.store_doc_embedding(doc_id, doc_emb)
            except Exception:
                logger.info(
                    f"Doc-level embedding skipped for '{title}' (content too large for embedding model)"
                )

        chunk_embeddings, chunk_metas, _ = await asyncio.gather(
            embed_svc.embed_batch([c.text for c in chunks], text_type="document"),
            asyncio.gather(*(_extract(c) for c in chunks)),
            _embed_doc_level(),
        )

        # Ordered, sequential writes (sync sqlite — single writer).
        for chunk, chunk_emb, chunk_meta in zip(chunks, chunk_embeddings, chunk_metas):
            await store.store_chunk(
                doc_id,
                chunk.text,
                chunk_emb,
                metadata=chunk_meta.model_dump(mode="json"),
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
    finally:
        await embed_svc.close()


# ---------------------------------------------------------------------------
# repair()
# ---------------------------------------------------------------------------


async def repair(
    database: str,
    *,
    model: str,
    embedding_model: str,
) -> RepairReport:
    """Scan database for incomplete ingestions and re-run phase 2.

    A document is incomplete if ANY of these are true:
    - Has no chunks
    - Has chunks but some are missing embeddings (chunk_embeddings)

    (The document-level embedding is optional — skipped for over-large content —
    so its absence does not mark a document incomplete.)

    For each incomplete document, the entire phase 2 is re-run:
    delete all existing chunks/embeddings → re-chunk → re-embed → re-store.

    This is safe to run at any time — it's idempotent and only touches
    documents that are genuinely incomplete.

    Note: repair also runs automatically on the first ingest()/search() call
    for each database, as part of the preflight check.

    Args:
        database: Database name
        model: LLM model for metadata extraction
        embedding_model: Embedding model

    Returns:
        RepairReport with counts and any failures

    Raises:
        RuntimeError: If embedding service or LLM is unavailable.
    """
    store = await _get_store(database)
    await _preflight(database, model, embedding_model)

    return await _repair_incomplete(store, model, embedding_model)


async def _repair_incomplete(
    store: DocumentStore,
    model: str,
    embedding_model: str,
) -> RepairReport:
    """Internal: scan for incomplete documents and re-run phase 2.

    Used by both repair() (explicit) and _preflight() (automatic on first access).
    """
    report = RepairReport()

    from .database import get_async_connection

    async with get_async_connection(str(store.db_path)) as db:
        async with db.execute(
            "SELECT doc_uuid, dc_title, markdown_content FROM documents WHERE markdown_content IS NOT NULL AND deleted_at IS NULL"
        ) as cursor:
            all_docs = list(await cursor.fetchall())

    report.documents_scanned = len(all_docs)

    for doc_uuid, title, content in all_docs:
        if not content:
            continue

        incomplete = await _is_incomplete(store, doc_uuid)
        if not incomplete:
            continue

        report.documents_incomplete += 1
        logger.info(f"Repairing '{title}' ({doc_uuid}): {incomplete}")

        try:
            await _run_phase2(
                store, doc_uuid, content, title or "Untitled", model, embedding_model
            )
            report.documents_repaired += 1
            logger.info(f"Repaired '{title}' ({doc_uuid})")
        except Exception as e:
            report.documents_failed += 1
            msg = f"Failed to repair '{title}' ({doc_uuid}): {e}"
            report.failures.append(msg)
            logger.warning(msg)

    return report


async def _is_incomplete(store: DocumentStore, doc_uuid: str) -> str:
    """Check if a document's phase 2 is incomplete. Returns reason string, or empty if complete."""
    from .database import get_async_connection

    async with get_async_connection(str(store.db_path)) as db:
        # Check: has chunks?
        async with db.execute(
            """
            SELECT COUNT(*) FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.doc_uuid = ?
            """,
            (doc_uuid,),
        ) as cursor:
            row = await cursor.fetchone()
            chunk_count = row[0] if row else 0

        if chunk_count == 0:
            return "no chunks"

        # Doc-level embedding is optional (skipped for large documents).
        # Don't treat its absence as incomplete.

    # Check: all chunks have embeddings? (requires sync connection for vec0)
    from pathlib import Path

    from .connection import get_connection

    with get_connection(Path(str(store.db_path))) as conn:
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM chunks c
            JOIN documents d ON c.document_id = d.id
            LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
            WHERE d.doc_uuid = ? AND ce.chunk_id IS NULL
            """,
            (doc_uuid,),
        )
        row = cursor.fetchone()
        missing_embeddings = row[0] if row else 0

    if missing_embeddings > 0:
        return f"{missing_embeddings} chunks missing embeddings"

    return ""


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


async def search(
    database: str,
    query: str,
    limit: int = 10,
    *,
    model: str,
    embedding_model: str,
) -> list[SearchResult]:
    """Search the knowledge base with natural language.

    Parses the query into a search plan (semantic query + optional metadata filter)
    using the LLM, then executes against the database.

    Args:
        database: Database name (e.g., "brain")
        query: Natural language query
        limit: Max results
        model: LLM model for query planning.
        embedding_model: Embedding model.

    Returns:
        List of SearchResult objects sorted by relevance.

    Raises:
        RuntimeError: If LLM or embedding service is unavailable.
    """
    store = await _get_store(database)
    await _preflight(database, model, embedding_model)

    # 1. Query planning — LLM decomposes into semantic query + optional filter
    from .query_planner import plan_search

    plan = await plan_search(query, model=model)
    logger.info(
        f"Search plan: semantic_query='{plan.semantic_query}', "
        f"filter={plan.filter.field if plan.filter else 'none'}, "
        f"needs_search={plan.needs_semantic_search}"
    )

    # 2. Apply metadata filter → set of matching doc_uuids
    doc_uuids: Optional[set[str]] = None
    warning = ""

    if plan.filter is not None:
        from .database import find_doc_uuids_by_filters

        filter_dicts = [plan.filter.model_dump()]
        doc_uuids = await find_doc_uuids_by_filters(str(store.db_path), filter_dicts)

        if not doc_uuids:
            logger.warning(
                f"Filter produced no results, searching without filter: {plan.filter.model_dump()}"
            )
            doc_uuids = None
            warning = "Filter produced no results, showing unfiltered results."

    # 3. Run search
    query_embedding: Optional[list[float]] = None
    if plan.needs_semantic_search and plan.semantic_query.strip():
        from .embeddings import EmbeddingService

        embed_svc = EmbeddingService(model=embedding_model)
        try:
            query_embedding = await embed_svc.embed_text(
                plan.semantic_query, text_type="query"
            )
        finally:
            await embed_svc.close()

    search_query = plan.semantic_query.strip() if plan.semantic_query.strip() else query

    from .search import search_unified

    fetch_limit = limit * 3 if doc_uuids is not None else limit

    raw_results = await search_unified(
        db_path=str(store.db_path),
        query=search_query,
        limit=fetch_limit,
        query_embedding=query_embedding,
        doc_uuids=doc_uuids,
    )

    # 4. Pure metadata query with no search results — return filtered docs directly
    if not plan.needs_semantic_search and doc_uuids is not None and not raw_results:
        from .database import get_documents_metadata

        ids = list(doc_uuids)[:limit]
        metas = await get_documents_metadata(str(store.db_path), ids)
        results: list[SearchResult] = []
        for uuid in ids:
            meta = metas.get(uuid)
            if meta:
                results.append(
                    SearchResult(
                        doc_id=uuid,
                        title=meta.title,
                        snippet="",
                        score=1.0,
                        metadata=meta.metadata,
                        match_type="metadata_filter",
                        warning=warning,
                    )
                )
        return results

    # 5. Enrich results with document metadata (batched — one query for titles/
    #    metadata, one more for the content fallback, instead of N per result).
    from .database import get_documents_content, get_documents_metadata

    top = raw_results[:limit]
    metas = await get_documents_metadata(str(store.db_path), [r.doc_id for r in top])

    # Only results with no chunk snippet need the full-content fallback read.
    need_content = [r.doc_id for r in top if not r.snippet and r.doc_id in metas]
    contents = await get_documents_content(str(store.db_path), need_content)

    results = []
    for r in top:
        meta = metas.get(r.doc_id)
        title = meta.title if meta else ""
        doc_metadata = meta.metadata if meta else {}

        snippet = r.snippet
        if not snippet and meta:
            # No chunk snippet — return full document content
            snippet = contents.get(r.doc_id, "")

        results.append(
            SearchResult(
                doc_id=r.doc_id,
                title=title,
                snippet=snippet,
                score=r.score,
                metadata=doc_metadata,
                match_type=r.tier,
                warning=warning,
            )
        )

    return results


# ---------------------------------------------------------------------------
# find_by_metadata()
# ---------------------------------------------------------------------------


async def find_by_metadata(
    database: str,
    filters: Mapping[str, MetadataFilterValue],
    limit: int = 100,
    *,
    include_content: bool = True,
) -> list[SearchResult]:
    """Find documents by metadata field values.

    Structured query for when the upstream agent knows exactly what to look for.
    Uses SQLite JSON functions on the metadata column.

    This is the complement to search(): search() uses natural language + embeddings,
    find_by_metadata() uses field matching. Agents use this for structured
    workflows (find all tasks for a goal, find all open delegations, etc.).

    Each filter value matches in one of three ways (all conditions AND-ed):
      * scalar (str / int / bool) → exact equality
      * ``None`` → SQL NULL
      * list / tuple / set → set-membership (``field IN (...)``); an empty
        collection matches nothing

    Args:
        database: Database name (e.g., "brain")
        filters: Mapping of {field_name: predicate} to match (see above).
        limit: Maximum results to return
        include_content: If True (default), each result's ``snippet`` holds the
            document's full content (one read per match). Set False for cheap
            overviews / counts — ``snippet`` is left empty and no content is
            read, so large result sets stay fast.

    Returns:
        List of SearchResult objects (score=1.0, match_type="metadata").

    Examples:
        # All to-do tasks (exact match)
        await find_by_metadata("brain", {"record_type": "task", "status": "todo"})

        # All "open" tasks in one query (set-membership)
        await find_by_metadata("brain", {
            "record_type": "task",
            "status": ["todo", "in_progress", "blocked"],
        })

        # All tasks for a specific goal
        await find_by_metadata("brain", {"goal_id": "abc-123", "record_type": "task"})

        # Cheap overview — metadata only, no content reads
        rows = await find_by_metadata(
            "brain", {"record_type": "task"}, limit=10_000, include_content=False,
        )
    """
    from .database import find_by_metadata as _find_by_metadata

    store = await _get_store(database)
    docs = await _find_by_metadata(str(store.db_path), filters, limit)

    # Batch the content reads — one query for all matches instead of one per row.
    contents: dict[str, str] = {}
    if include_content:
        from .database import get_documents_content

        contents = await get_documents_content(
            str(store.db_path), [m.doc_id for m in docs]
        )

    results: list[SearchResult] = []
    for meta in docs:
        content = contents.get(meta.doc_id, "") if include_content else ""

        results.append(
            SearchResult(
                doc_id=meta.doc_id,
                title=meta.title,
                snippet=content,
                score=1.0,
                metadata=meta.metadata,
                match_type="metadata",
            )
        )

    return results


# ---------------------------------------------------------------------------
# describe_metadata()
# ---------------------------------------------------------------------------


async def describe_metadata(
    database: str,
    *,
    filters: Mapping[str, MetadataFilterValue] | None = None,
    max_values: int = 25,
) -> dict[str, FieldProfile]:
    """Discover the metadata schema actually present in a database.

    The store is schema-less — metadata is arbitrary JSON, and consumers define
    their own vocabulary (``record_type``, ``status``, …). This function reports
    that vocabulary so a caller (human or agent) can fill ``find_by_metadata``
    filters without prior knowledge of which fields or values exist.

    For each top-level metadata field it returns a :class:`FieldProfile`:
    how many documents carry it, how many distinct values it takes, and — when
    the field is low-cardinality (a closed set like ``status``) — the
    value→count breakdown. High-cardinality fields (ids, titles, free text)
    report counts only, so the output stays bounded.

    Pass ``filters`` to scope the profile to a subset and drill in:

        await describe_metadata("brain")
        #   -> {"record_type": FieldProfile(present_in=62, distinct=3,
        #                                    values={"task": 42, "idea": 13, ...}),
        #       "title":       FieldProfile(present_in=62, distinct=62, values=None)}

        await describe_metadata("brain", filters={"record_type": "task"})
        #   -> {"status":   FieldProfile(present_in=42, distinct=4,
        #                                values={"todo": 20, "in_progress": 5, ...}),
        #       "due_date": FieldProfile(present_in=38, distinct=38, values=None)}

    Internal fields (keys starting with ``_``, e.g. ``_history``) are excluded.

    Args:
        database: Database name (e.g., "brain")
        filters: Optional subset to profile (same matching semantics as
            :func:`find_by_metadata`). None profiles the whole database.
        max_values: A field's per-value breakdown is included only when it has
            at most this many distinct values; above it, ``values`` is None.

    Returns:
        Mapping of ``field_name -> FieldProfile``.
    """
    from .database import describe_metadata as _describe_metadata

    store = await _get_store(database)
    raw = await _describe_metadata(str(store.db_path), filters)

    profiles: dict[str, FieldProfile] = {}
    for field_name, value_counts in raw.items():
        distinct = len(value_counts)
        profiles[field_name] = FieldProfile(
            present_in=sum(value_counts.values()),
            distinct=distinct,
            values=dict(value_counts) if distinct <= max_values else None,
        )
    return profiles


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


async def delete(database: str, doc_id: str) -> bool:
    """Delete a document and all its chunks from the knowledge base.

    Args:
        database: Database name (e.g., "brain")
        doc_id: Document UUID to delete

    Returns:
        True if deleted, False if not found
    """
    store = await _get_store(database)
    return await store.delete(doc_id)


# ---------------------------------------------------------------------------
# update_metadata()
# ---------------------------------------------------------------------------


async def update_metadata(
    database: str,
    doc_id: str,
    metadata: dict,
    merge: bool = True,
) -> bool:
    """Update metadata on a document. Changes are recorded in _history.

    This is how higher layers manage structured workflows — tasks, goals,
    decisions — by writing status and relationship fields into the metadata.

    Every change is recorded in a _history list within the metadata, capturing
    what changed, when, and what the previous values were. History is capped
    at 50 entries.

    Args:
        database: Database name (e.g., "brain")
        doc_id: Document UUID
        metadata: Dict of fields to update. Merged with existing metadata by default.
        merge: If True, merge with existing metadata (caller values win).
            If False, replace metadata entirely.

    Returns:
        True if document was found and updated, False if not found.
    """
    from datetime import datetime, timezone

    store = await _get_store(database)

    # Read current metadata to compute diff for history
    if merge:
        doc = await store.read(doc_id)
        if doc and doc.metadata and doc.metadata.metadata:
            current = doc.metadata.metadata
            # Compute what actually changed (exclude _history from diff)
            changed_fields = {}
            for key, new_val in metadata.items():
                if key.startswith("_"):
                    continue
                old_val = current.get(key)
                if old_val != new_val:
                    changed_fields[key] = old_val

            if changed_fields:
                history = current.get("_history", [])
                history.append(
                    {
                        "changed_at": datetime.now(timezone.utc).isoformat(),
                        "old_values": changed_fields,
                    }
                )
                # Cap at 50 entries
                metadata["_history"] = history[-50:]

    result = await store.update(doc_id, metadata=metadata, merge_metadata=merge)
    return result.success


# ---------------------------------------------------------------------------
# restore() / purge() / list_deleted()
# ---------------------------------------------------------------------------


async def restore(database: str, doc_id: str) -> bool:
    """Restore a soft-deleted document. Returns True if found and restored."""
    store = await _get_store(database)
    return await store.restore(doc_id)


async def purge(database: str, older_than_days: int = 30) -> int:
    """Permanently delete soft-deleted documents older than N days. Returns count purged."""
    from .database import purge_deleted

    store = await _get_store(database)
    return await purge_deleted(str(store.db_path), older_than_days)


async def list_deleted(database: str, limit: int = 50) -> list[SearchResult]:
    """List soft-deleted documents (for trash view / undo UI)."""
    from .database import list_deleted_documents

    store = await _get_store(database)
    results = await list_deleted_documents(str(store.db_path), limit)
    return [
        SearchResult(
            doc_id=r.doc_id,
            title=r.title,
            snippet="",
            score=0.0,
            metadata=r.metadata,
        )
        for r in results
    ]


# ---------------------------------------------------------------------------
# find_duplicates()
# ---------------------------------------------------------------------------


async def find_duplicates(
    database: str,
    threshold: float = 0.92,
) -> list[DuplicateGroup]:
    """Find groups of near-duplicate documents using embedding similarity.

    Compares doc-level embeddings (cosine similarity) across all documents.
    Documents without embeddings are skipped.

    Args:
        database: Database name (e.g., "brain")
        threshold: Cosine similarity threshold for considering documents
            as duplicates. Default 0.92 (very similar content).

    Returns:
        List of DuplicateGroup, each containing 2+ documents that are
        near-duplicates. Empty list if no duplicates found.
    """
    import numpy as np

    from .database import load_doc_embeddings

    store = await _get_store(database)

    # Load all non-deleted documents with a doc-level embedding (from vec0).
    rows = await load_doc_embeddings(str(store.db_path), include_deleted=False)

    if len(rows) < 2:
        return []

    # Build a set of derived_from relationships to skip
    derived_pairs: set[tuple[str, str]] = set()
    doc_metadata: dict[str, dict] = {}
    for doc_uuid, _title, _embedding, meta, _created in rows:
        doc_metadata[doc_uuid] = meta
        if derived_from := meta.get("derived_from"):
            derived_pairs.add((doc_uuid, derived_from))
            derived_pairs.add((derived_from, doc_uuid))

    # L2-normalize for cosine similarity
    docs: list[tuple[str, str, np.ndarray]] = []
    for doc_uuid, title, embedding_list, _meta, _created in rows:
        embedding = np.array(embedding_list, dtype=np.float64)
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm
        docs.append((doc_uuid, title, embedding))

    # Cosine similarity matrix (all vectors L2-normalized, so dot product = cosine)
    n = len(docs)
    matrix = np.stack([d[2] for d in docs])
    similarity = matrix @ matrix.T

    # Greedy grouping above threshold
    grouped: set[int] = set()
    groups: list[DuplicateGroup] = []

    for i in range(n):
        if i in grouped:
            continue

        group_indices = [i]
        for j in range(i + 1, n):
            if j in grouped:
                continue
            if similarity[i, j] >= threshold:
                # Skip pairs where one is derived from the other
                id_i, id_j = docs[i][0], docs[j][0]
                if (id_i, id_j) in derived_pairs:
                    continue
                group_indices.append(j)
                grouped.add(j)

        if len(group_indices) > 1:
            grouped.add(i)
            sims = [
                float(similarity[a, b])
                for a in group_indices
                for b in group_indices
                if a < b
            ]
            avg_sim = sum(sims) / len(sims) if sims else 0.0

            groups.append(
                DuplicateGroup(
                    doc_ids=[docs[idx][0] for idx in group_indices],
                    titles=[docs[idx][1] for idx in group_indices],
                    similarity=round(avg_sim, 4),
                )
            )

    return groups
