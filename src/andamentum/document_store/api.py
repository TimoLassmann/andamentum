"""DocumentStore API - Main interface for document management.

Architecture: Named Database System
- All databases stored in ~/.local/share/document-store/{name}.db (override with DOCUMENT_STORE_DIR)
- Users create databases explicitly before attaching to sessions

Pure storage + search layer. Chunking and embedding happen upstream.
The store accepts pre-chunked text with pre-computed embeddings.

FTS5 is auto-synced via database triggers on INSERT/UPDATE/DELETE.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

if TYPE_CHECKING:
    from .cluster_models import ClusterDetail, ClusterSummary, ReclusterResult
    from .dhp import DHPConfig


from .database import (
    delete_document_record,
    get_document_metadata,
    list_documents_by_type,
    register_document,
    store_doc_embedding,
    update_document_content,
    update_document_metadata,
)
from .lifecycle import get_db_path, init_database_metadata, is_ephemeral_name
from .models import (
    Document,
    DocumentMetadata,
    DocumentType,
    ReembedResult,
    UpdateResult,
)
from .search import search_unified, UnifiedSearchResult

logger = logging.getLogger(__name__)


class DocumentStore:
    """Unified document management with single-tier indexing.

    Pure storage + search layer. Callers handle chunking and embedding upstream,
    then store results via register_document() + store_chunk().

    Search uses 4-signal RRF fusion:
    1. FTS5 keyword search (auto-triggered on document INSERT/UPDATE)
    2. Chunk-level semantic search (via store_chunk embeddings)
    3. Document-level semantic search (via store_doc_embedding)
    4. DHP temporal cluster scoring

    Usage:
        store = DocumentStore.for_database("brain")
        await store.initialize()

        # Register document (FTS5 auto-indexed)
        doc_id = await store.register_document("My thought", content="...")

        # Store chunks with embeddings
        chunk_id = await store.store_chunk(doc_id, chunk_text, embedding, metadata={})

        # Store doc-level embedding
        await store.store_doc_embedding(doc_id, doc_embedding)

        # Search
        results = await store.search("query", query_embedding=emb)
    """

    def __init__(
        self,
        database_name: str,
        db_dir: Optional[str | Path] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize DocumentStore for a named database.

        Args:
            database_name: Name of the database (e.g., "brain", "research")
            embedding_model: Embedding model for vectors (used by reembed_all)
            db_dir: Custom directory for database files. When provided, databases
                are written here instead of the default directory.

        Examples:
            DocumentStore("brain")
            DocumentStore("research", db_dir="./output")
        """
        self.database_name = database_name
        self.embedding_model = embedding_model

        if db_dir is not None:
            self.is_ephemeral = False
            db_dir_path = Path(db_dir)
            db_dir_path.mkdir(parents=True, exist_ok=True)
            self.db_path = db_dir_path / f"{database_name}.db"
        else:
            self.is_ephemeral = is_ephemeral_name(database_name)
            self.db_path = get_db_path(database_name, ephemeral=self.is_ephemeral)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize database tables and metadata.

        Creates the database if it doesn't exist.
        Creates full schema (documents, chunks, embeddings, FTS5, clusters).
        """
        from .schema import init_all_tables

        init_all_tables(self.db_path)
        init_database_metadata(str(self.db_path), self.database_name)

    @classmethod
    def for_database(
        cls,
        database_name: str,
        db_dir: Optional[str | Path] = None,
    ) -> "DocumentStore":
        """Create DocumentStore instance for a named database.

        Args:
            database_name: Database name (e.g., "brain", "research")
            db_dir: Custom directory for database files.

        Returns:
            DocumentStore configured for the named database
        """
        kwargs: dict[str, Any] = {"database_name": database_name}
        if db_dir is not None:
            kwargs["db_dir"] = db_dir
        return cls(**kwargs)

    # -------------------------------------------------------------------------
    # Document lifecycle
    # -------------------------------------------------------------------------

    async def register_document(
        self,
        title: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Register a new document. Content is stored and FTS5-indexed automatically.

        Idempotent: if a non-deleted document with the same content hash already
        exists, returns the existing doc_id instead of creating a duplicate.

        Args:
            title: Document title
            content: Full document content (markdown)
            metadata: Optional metadata dict (use DocumentMetadataFields for structure)

        Returns:
            Document ID (UUID string) — existing ID if content is a duplicate
        """
        import hashlib

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Dedup at the storage layer: if content already exists, return existing ID
        from .database import get_async_connection

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                "SELECT doc_uuid FROM documents WHERE file_hash = ? AND deleted_at IS NULL LIMIT 1",
                (content_hash,),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    logger.info(
                        f"Dedup: content already exists as {row[0]}, skipping insert"
                    )
                    return row[0]

        doc_id = str(uuid.uuid4())

        await register_document(
            db_path=str(self.db_path),
            doc_id=doc_id,
            title=title,
            content=content,
            metadata=metadata or {},
        )

        return doc_id

    async def add(
        self,
        file_path: str,
        content: Optional[str] = None,
        title: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a document (backward-compatible wrapper around register_document).

        For new code, prefer register_document() which has a cleaner API.

        Args:
            file_path: Path to document file or identifier
            content: Document content (required — conversion removed from this layer)
            title: Document title (defaults to filename stem)
            document_type: Ignored (kept for backward compatibility)
            metadata: Additional metadata

        Returns:
            Document ID (UUID)
        """
        if content is None:
            raise ValueError(
                "Content must be provided. Document conversion has been moved upstream. "
                "Use andamentum-convert to convert documents to markdown before calling add()."
            )

        if not title:
            title = Path(file_path).stem

        doc_id = str(uuid.uuid4())

        await register_document(
            db_path=str(self.db_path),
            doc_id=doc_id,
            title=title,
            content=content,
            metadata=metadata or {},
            document_type=document_type,
            file_path=file_path,
        )

        return doc_id

    # -------------------------------------------------------------------------
    # Chunk storage
    # -------------------------------------------------------------------------

    async def store_chunk(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
        chunk_index: int = 0,
        start_char: int = 0,
        end_char: int = 0,
    ) -> int:
        """Store a chunk with its embedding for an existing document.

        This is the atomic storage operation. The store doesn't know or care
        how chunking was done — it just stores text + embedding + metadata.

        Args:
            doc_id: Document UUID (from register_document)
            text: Chunk text content
            embedding: Embedding vector (768-dim)
            metadata: Optional chunk metadata (use ChunkMetadataFields for structure)
            chunk_index: Position within document (0-based)
            start_char: Start character offset in original document
            end_char: End character offset in original document

        Returns:
            Chunk integer ID
        """
        from .chunks import store_chunk_for_document

        return store_chunk_for_document(
            doc_uuid=doc_id,
            chunk_text=text,
            embedding=embedding,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
            db_path=Path(self.db_path),
        )

    async def delete_chunks(self, doc_id: str) -> int:
        """Delete all chunks and chunk embeddings for a document.

        Args:
            doc_id: Document UUID

        Returns:
            Number of chunks deleted
        """
        from .chunks import delete_chunks_for_document

        return delete_chunks_for_document(
            doc_uuid=doc_id,
            db_path=Path(self.db_path),
        )

    async def store_doc_embedding(self, doc_id: str, embedding: list[float]) -> None:
        """Store a document-level embedding.

        Args:
            doc_id: Document UUID
            embedding: Embedding vector (768-dim)
        """
        await store_doc_embedding(str(self.db_path), doc_id, embedding)

    # -------------------------------------------------------------------------
    # Read / Update / Delete
    # -------------------------------------------------------------------------

    async def read(self, doc_id: str) -> Optional[Document]:
        """Read a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document with metadata and content, or None if not found
        """
        doc_metadata = await get_document_metadata(str(self.db_path), doc_id)
        if not doc_metadata:
            return None

        from .database import get_async_connection

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                "SELECT markdown_content FROM documents WHERE doc_uuid = ? AND deleted_at IS NULL",
                (doc_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row is not None and row[0] is not None:
                    return Document(
                        metadata=doc_metadata,
                        content=row[0],
                        raw_file_path=None,
                    )

        return None

    async def update(
        self,
        doc_id: str,
        new_content: Optional[str] = None,
        metadata: Optional[dict] = None,
        merge_metadata: bool = True,
    ) -> UpdateResult:
        """Update document content and/or metadata.

        When content is updated, FTS5 trigger fires automatically.
        Caller should delete_chunks() + re-store if content changed and
        chunk-level search is needed.

        Args:
            doc_id: Document identifier
            new_content: New document content (optional)
            metadata: Metadata dict to update (optional)
            merge_metadata: If True, merge with existing. If False, replace.

        Returns:
            UpdateResult with operation details
        """
        if new_content is None and metadata is None:
            return UpdateResult(
                success=False,
                doc_id=doc_id,
                previous_hash="",
                new_hash="",
                reindexed=False,
                metadata_updated=False,
                message="Nothing to update: provide new_content or metadata",
            )

        doc_metadata = await get_document_metadata(str(self.db_path), doc_id)
        if not doc_metadata:
            return UpdateResult(
                success=False,
                doc_id=doc_id,
                previous_hash="",
                new_hash="",
                reindexed=False,
                metadata_updated=False,
                message=f"Document {doc_id} not found",
            )

        previous_hash = doc_metadata.content_hash
        new_hash = previous_hash
        metadata_updated = False

        if new_content is not None:
            previous_hash, new_hash = await update_document_content(
                str(self.db_path), doc_id, new_content
            )

        if metadata is not None:
            await update_document_metadata(
                str(self.db_path), doc_id, metadata, merge=merge_metadata
            )
            metadata_updated = True

        return UpdateResult(
            success=True,
            doc_id=doc_id,
            previous_hash=previous_hash,
            new_hash=new_hash,
            reindexed=False,
            metadata_updated=metadata_updated,
            message=f"Document {doc_id} updated successfully",
        )

    async def delete(self, doc_id: str) -> bool:
        """Soft-delete a document (sets deleted_at, excluded from search/listing).

        Use restore() to undo. The document and its chunks remain in the database
        but are excluded from all search and listing operations.

        Args:
            doc_id: Document identifier

        Returns:
            True if soft-deleted, False if not found
        """
        from .database import soft_delete_document

        return await soft_delete_document(str(self.db_path), doc_id)

    async def exists_by_hash(self, file_hash: str) -> bool:
        """Check if a non-deleted document with the given content hash exists.

        Uses the indexed file_hash column for fast lookups.

        Args:
            file_hash: SHA-256 hex digest of the content.

        Returns:
            True if a document with this hash exists and is not soft-deleted.
        """
        from .database import get_async_connection

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                "SELECT 1 FROM documents WHERE file_hash = ? AND deleted_at IS NULL LIMIT 1",
                (file_hash,),
            ) as cursor:
                return await cursor.fetchone() is not None

    async def restore(self, doc_id: str) -> bool:
        """Restore a soft-deleted document.

        Args:
            doc_id: Document identifier

        Returns:
            True if restored, False if not found or not deleted
        """
        from .database import restore_document

        return await restore_document(str(self.db_path), doc_id)

    async def hard_delete(self, doc_id: str) -> bool:
        """Permanently delete a document and all its chunks. Cannot be undone.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        doc_metadata = await get_document_metadata(str(self.db_path), doc_id)
        if not doc_metadata:
            return False

        await self.delete_chunks(doc_id)
        return await delete_document_record(str(self.db_path), doc_id)

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        query_embedding: Optional[list[float]] = None,
    ) -> list["UnifiedSearchResult"]:
        """Search across all documents with 4-signal RRF fusion.

        Fuses FTS5 keyword, chunk-level semantic, doc-level semantic,
        and DHP temporal cluster scoring.

        Args:
            query: Search query
            limit: Maximum results to return
            query_embedding: Optional pre-computed query embedding.
                If provided, enables semantic search signals.
                If None, only FTS5 keyword search runs.

        Returns:
            List of UnifiedSearchResult objects sorted by relevance
        """
        return await search_unified(
            db_path=str(self.db_path),
            query=query,
            limit=limit,
            query_embedding=query_embedding,
        )

    # -------------------------------------------------------------------------
    # List / Find / Stats
    # -------------------------------------------------------------------------

    async def list_documents(
        self, document_type: Optional[DocumentType] = None
    ) -> list[DocumentMetadata]:
        """List all documents, optionally filtered by type."""
        return await list_documents_by_type(str(self.db_path), document_type)

    async def find_by_metadata(
        self,
        filters: Mapping[str, Any],
        limit: int = 100,
    ) -> list[DocumentMetadata]:
        """Find documents by metadata field values.

        Uses SQLite JSON functions to query the metadata column.

        Args:
            filters: Dict of {field_name: expected_value} to match
            limit: Maximum results to return

        Returns:
            List of matching DocumentMetadata objects
        """
        from .database import find_by_metadata

        return await find_by_metadata(str(self.db_path), filters, limit)

    async def get_stats(self) -> dict:
        """Get statistics about this database."""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]

            return {
                "database_name": self.database_name,
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "db_path": str(self.db_path),
            }
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Bulk operations
    # -------------------------------------------------------------------------

    async def reembed_all(
        self,
        embedding_model: Optional[str] = None,
        batch_size: int = 50,
    ) -> ReembedResult:
        """Backfill document-level embeddings for all documents missing them.

        Args:
            embedding_model: Embedding model to use (defaults to store's configured model)
            batch_size: Number of documents to process per batch

        Returns:
            ReembedResult with counts and duration
        """
        from .database import get_async_connection
        from .embeddings import EmbeddingService

        model = embedding_model or self.embedding_model
        start_time = time.monotonic()
        n_embedded = 0
        n_skipped = 0
        n_failed = 0

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                """
                SELECT doc_uuid, dc_title, markdown_content
                FROM documents
                WHERE doc_embedding IS NULL AND markdown_content IS NOT NULL
                ORDER BY created_date ASC
                """
            ) as cursor:
                docs_to_embed: list[Any] = list(await cursor.fetchall())

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM documents WHERE doc_embedding IS NOT NULL"
            ) as cursor:
                row = await cursor.fetchone()
                n_skipped = row[0] if row else 0

        if not docs_to_embed:
            duration = time.monotonic() - start_time
            return ReembedResult(
                n_embedded=0,
                n_skipped=n_skipped,
                n_failed=0,
                duration_seconds=round(duration, 2),
            )

        logger.info(
            f"Re-embedding {len(docs_to_embed)} documents (skipping {n_skipped} with existing embeddings)"
        )

        if not model:
            raise ValueError("embedding_model is required for reembed_all")
        embedding_service = EmbeddingService(model=model)
        try:
            for i in range(0, len(docs_to_embed), batch_size):
                batch = docs_to_embed[i : i + batch_size]
                for doc_uuid, title, content in batch:
                    try:
                        doc_embedding = await embedding_service.embed_text(
                            content, text_type="document", title=title
                        )
                        await store_doc_embedding(
                            str(self.db_path), doc_uuid, doc_embedding
                        )
                        n_embedded += 1
                        if n_embedded % 10 == 0:
                            logger.info(
                                f"Re-embedded {n_embedded}/{len(docs_to_embed)} documents"
                            )
                    except Exception as e:
                        n_failed += 1
                        logger.warning(
                            f"Failed to embed document '{title}' ({doc_uuid}): {e}"
                        )
        finally:
            await embedding_service.close()

        duration = time.monotonic() - start_time
        logger.info(
            f"Re-embedding complete: {n_embedded} embedded, {n_skipped} skipped, {n_failed} failed in {duration:.1f}s"
        )
        return ReembedResult(
            n_embedded=n_embedded,
            n_skipped=n_skipped,
            n_failed=n_failed,
            duration_seconds=round(duration, 2),
        )

    # -------------------------------------------------------------------------
    # DHP Temporal Clustering
    # -------------------------------------------------------------------------

    async def recluster(
        self, config: Optional["DHPConfig"] = None
    ) -> "ReclusterResult":
        """Run full offline DHP re-clustering on all documents.

        Reads stored embeddings and timestamps, runs the complete DHP algorithm
        from scratch, and writes new cluster assignments. Never re-embeds.

        Args:
            config: DHP configuration. Uses defaults if None.

        Returns:
            ReclusterResult with cluster count, doc count, and duration.
        """
        from .cluster_models import ReclusterResult
        from .connection import get_connection
        from .dhp import (
            DHPConfig,
            cluster_state_to_dict,
            recluster as dhp_recluster,
            timestamp_to_hours,
        )

        if config is None:
            config = DHPConfig()

        start_time = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()

        from .database import get_async_connection

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                """
                SELECT doc_uuid, doc_embedding, created_date
                FROM documents
                WHERE doc_embedding IS NOT NULL
                ORDER BY created_date ASC
                """
            ) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            duration = time.monotonic() - start_time
            return ReclusterResult(
                n_clusters=0,
                n_documents=0,
                duration_seconds=round(duration, 2),
                config=config.to_dict(),
            )

        embeddings_and_times: list[tuple[str, np.ndarray, float]] = []
        for row in rows:
            doc_uuid = row[0]
            embedding = np.array(json.loads(row[1]), dtype=np.float64)
            created = datetime.fromisoformat(row[2])
            t_hours = timestamp_to_hours(created.timestamp())
            embeddings_and_times.append((doc_uuid, embedding, t_hours))

        assignments, cluster_states = dhp_recluster(embeddings_and_times, config)

        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM clusters")
            cursor.execute("UPDATE documents SET cluster_id = NULL")

            cluster_id_map: dict[int, int] = {}
            for internal_id, cstate in cluster_states.items():
                state_dict = cluster_state_to_dict(cstate)
                cursor.execute(
                    """
                    INSERT INTO clusters (centroid, decay_rate, kernel_params, doc_count, created_at, last_active_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        json.dumps(state_dict["centroid"]),
                        float(np.mean(cstate.kernel_weights)),
                        json.dumps(
                            {
                                "weights": state_dict["kernel_weights"],
                                "doc_times": state_dict["doc_times"],
                            }
                        ),
                        cstate.doc_count,
                        datetime.fromtimestamp(
                            cstate.created_at * 3600, tz=timezone.utc
                        ).isoformat(),
                        datetime.fromtimestamp(
                            cstate.last_active_at * 3600, tz=timezone.utc
                        ).isoformat(),
                    ),
                )
                cluster_id_map[internal_id] = cursor.lastrowid  # type: ignore[assignment]

            for doc_uuid, internal_cid in assignments.items():
                db_cid = cluster_id_map.get(internal_cid)
                if db_cid is not None:
                    cursor.execute(
                        "UPDATE documents SET cluster_id = ? WHERE doc_uuid = ?",
                        (db_cid, doc_uuid),
                    )

            completed_at = datetime.now(timezone.utc).isoformat()
            duration = time.monotonic() - start_time
            cursor.execute(
                """
                INSERT INTO cluster_runs (config, doc_count, cluster_count, started_at, completed_at, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    json.dumps(config.to_dict()),
                    len(assignments),
                    len(cluster_states),
                    started_at,
                    completed_at,
                    round(duration, 2),
                ),
            )

            conn.commit()

        # Invalidate cluster cache so next search picks up new clusters
        from .search import _invalidate_cluster_cache

        _invalidate_cluster_cache(str(self.db_path))

        duration = time.monotonic() - start_time
        logger.info(
            f"Re-clustering complete: {len(cluster_states)} clusters from {len(assignments)} docs in {duration:.1f}s"
        )

        return ReclusterResult(
            n_clusters=len(cluster_states),
            n_documents=len(assignments),
            duration_seconds=round(duration, 2),
            config=config.to_dict(),
        )

    async def list_clusters(
        self, sort_by: str = "last_active_at", include_docs: bool = False
    ) -> list:
        """List all clusters with summary information."""
        from .cluster_models import ClusterDetail, ClusterInfo
        from .database import get_async_connection

        valid_sorts = {"last_active_at", "doc_count", "decay_rate", "created_at"}
        if sort_by not in valid_sorts:
            sort_by = "last_active_at"

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                f"SELECT id, doc_count, decay_rate, created_at, last_active_at FROM clusters ORDER BY {sort_by} DESC"
            ) as cursor:
                rows = await cursor.fetchall()

        results = []
        for row in rows:
            cluster_id, doc_count, decay_rate, created_at, last_active_at = row
            if include_docs:
                docs = await self._get_cluster_docs(cluster_id)
                centroid, kernel_params = await self._get_cluster_params(cluster_id)
                results.append(
                    ClusterDetail(
                        cluster_id=cluster_id,
                        doc_count=doc_count,
                        decay_rate=decay_rate,
                        created_at=created_at,
                        last_active_at=last_active_at,
                        documents=docs,
                        kernel_params=kernel_params,
                        centroid=centroid,
                    )
                )
            else:
                results.append(
                    ClusterInfo(
                        cluster_id=cluster_id,
                        doc_count=doc_count,
                        decay_rate=decay_rate,
                        created_at=created_at,
                        last_active_at=last_active_at,
                    )
                )
        return results

    async def get_cluster(
        self, cluster_id: int, include_docs: bool = True
    ) -> Optional["ClusterDetail"]:
        """Get detailed information about a specific cluster."""
        from .cluster_models import ClusterDetail
        from .database import get_async_connection

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                "SELECT id, doc_count, decay_rate, created_at, last_active_at FROM clusters WHERE id = ?",
                (cluster_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            return None

        docs = await self._get_cluster_docs(cluster_id) if include_docs else []
        centroid, kernel_params = await self._get_cluster_params(cluster_id)

        return ClusterDetail(
            cluster_id=row[0],
            doc_count=row[1],
            decay_rate=row[2],
            created_at=row[3],
            last_active_at=row[4],
            documents=docs,
            kernel_params=kernel_params,
            centroid=centroid,
        )

    async def cluster_summary(self) -> "ClusterSummary":
        """Get high-level summary of clustering state."""
        from .cluster_models import ClusterSummary
        from .database import get_async_connection

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute("SELECT COUNT(*) FROM clusters") as c:
                _row = await c.fetchone()
                n_clusters = _row[0] if _row else 0
            async with db.execute(
                "SELECT COUNT(*) FROM documents WHERE cluster_id IS NOT NULL"
            ) as c:
                _row = await c.fetchone()
                n_clustered = _row[0] if _row else 0
            async with db.execute(
                "SELECT COUNT(*) FROM documents WHERE cluster_id IS NULL"
            ) as c:
                _row = await c.fetchone()
                n_unclustered = _row[0] if _row else 0
            async with db.execute(
                "SELECT config, completed_at FROM cluster_runs ORDER BY id DESC LIMIT 1"
            ) as c:
                run_row = await c.fetchone()

        last_config = json.loads(run_row[0]) if run_row else None
        last_run_at = run_row[1] if run_row else None

        return ClusterSummary(
            n_clusters=n_clusters,
            n_clustered_docs=n_clustered,
            n_unclustered_docs=n_unclustered,
            last_run_config=last_config,
            last_run_at=last_run_at,
        )

    async def _get_cluster_docs(self, cluster_id: int) -> list[DocumentMetadata]:
        """Get documents assigned to a cluster."""
        from .database import get_async_connection

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                "SELECT doc_uuid FROM documents WHERE cluster_id = ? ORDER BY created_date DESC",
                (cluster_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        docs = []
        for row in rows:
            meta = await get_document_metadata(str(self.db_path), row[0])
            if meta:
                docs.append(meta)
        return docs

    async def _get_cluster_params(self, cluster_id: int) -> tuple[list[float], dict]:
        """Get cluster centroid and kernel parameters."""
        from .database import get_async_connection

        async with get_async_connection(str(self.db_path)) as db:
            async with db.execute(
                "SELECT centroid, kernel_params FROM clusters WHERE id = ?",
                (cluster_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            return [], {}
        return json.loads(row[0]), json.loads(row[1])
