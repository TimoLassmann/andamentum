"""Unified search across all documents using RRF fusion.

Fuses four signals in parallel using Reciprocal Rank Fusion (RRF, k=60):
1. FTS5 keyword search (all documents)
2. Chunk-level semantic search (RAG embeddings)
3. Document-level semantic search (doc_embeddings vec0)
4. DHP temporal cluster scoring
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .dhp import ClusterState, DHPConfig

import aiosqlite

from andamentum.core.embeddings import OllamaEmbedder

from .chunk_hits import ChunkHit, chunk_id_for
from .chunks_search import RRF_K, SearchConfig
from .database import get_async_connection
from .fts import extract_snippet
from .fts_query import prepare_fts_query
from .hybrid_search import multi_strategy_search

logger = logging.getLogger(__name__)


@dataclass
class SearchResultMetadata:
    """Metadata about how a search result was matched.

    Captures information about the search strategy used and
    any entity/tag matches for transparency in results.
    """

    match_type: str = ""
    entity_matches: list[str] = field(default_factory=list)
    tag_matches: list[str] = field(default_factory=list)


@dataclass
class UnifiedSearchResult:
    """A single search result from unified search.

    For chunk-based signals ("chunks"), snippet contains the matching chunk text.
    For other signals, snippet is empty — call DocumentStore.read(doc_id) for full content.

    ``chunks`` carries every chunk of this document that the chunk signal
    actually matched, best-scoring first, each with the id needed to resolve it
    again (see :mod:`.chunk_hits`). It is EMPTY for a document that reached the
    fused list only through a document-level signal — keyword, document
    embedding, or temporal cluster — because those signals never looked at a
    chunk and have no passage to report. An empty list therefore means "this
    signal found no passage", never "this document has no chunks"; to enumerate
    a document's chunks regardless of how it was matched, call
    :func:`~.public.get_chunks`.
    """

    doc_id: str
    score: float
    tier: str  # "fts5", "chunks", "doc_semantic", or "cluster"
    snippet: str = ""
    metadata: SearchResultMetadata = field(default_factory=SearchResultMetadata)
    chunks: list[ChunkHit] = field(default_factory=list)


@dataclass
class MultiDatabaseSearchResult(UnifiedSearchResult):
    """Search result from multi-database search.

    Extends UnifiedSearchResult with database name for
    results spanning multiple databases.
    """

    database_name: str = ""


@dataclass
class _ClusterCacheEntry:
    """Cached DHP cluster state for a database."""

    cluster_states: dict[int, "ClusterState"]
    config: "DHPConfig"
    timestamp: float


_cluster_cache: dict[str, _ClusterCacheEntry] = {}
_CLUSTER_CACHE_TTL = 300  # 5 minutes


def _load_cluster_state(db_path: str) -> tuple[dict[int, "ClusterState"], "DHPConfig"]:
    """Load cluster states, using cache if fresh.

    Returns:
        Tuple of (cluster_states dict, DHPConfig).
        Empty dict and default config if no clusters exist.
    """
    import json
    import sqlite3

    import numpy as np

    now = time.time()
    entry = _cluster_cache.get(db_path)
    if entry is not None and (now - entry.timestamp) < _CLUSTER_CACHE_TTL:
        return entry.cluster_states, entry.config

    from .dhp import ClusterState, DHPConfig

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM clusters").fetchone()
        if not row or row["cnt"] == 0:
            _cluster_cache[db_path] = _ClusterCacheEntry(
                cluster_states={}, config=DHPConfig(), timestamp=now
            )
            return {}, DHPConfig()

        rows = conn.execute(
            "SELECT id, centroid, kernel_params, doc_count, created_at, last_active_at FROM clusters"
        ).fetchall()

        cluster_states: dict[int, ClusterState] = {}
        for r in rows:
            params = json.loads(r["kernel_params"])
            centroid = np.array(json.loads(r["centroid"]), dtype=np.float64)
            kernel_weights = np.array(params.get("weights", []), dtype=np.float64)
            doc_times = params.get("doc_times", [])

            cluster_states[r["id"]] = ClusterState(
                cluster_id=r["id"],
                centroid=centroid,
                kernel_weights=kernel_weights,
                doc_times=doc_times,
                doc_count=r["doc_count"],
                created_at=0.0,
                last_active_at=0.0,
            )

        config_row = conn.execute(
            "SELECT config FROM cluster_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        config = (
            DHPConfig.from_dict(json.loads(config_row["config"]))
            if config_row
            else DHPConfig()
        )

        _cluster_cache[db_path] = _ClusterCacheEntry(
            cluster_states=cluster_states, config=config, timestamp=now
        )
        return cluster_states, config
    finally:
        conn.close()


def _invalidate_cluster_cache(db_path: str | None = None) -> None:
    """Invalidate cluster cache.

    Args:
        db_path: Specific database to invalidate. If None, clears all entries.
    """
    if db_path is None:
        _cluster_cache.clear()
    else:
        _cluster_cache.pop(db_path, None)


def _get_production_search_config() -> SearchConfig:
    """Get production-quality search configuration.

    BM25 hybrid search (50/50 semantic/keyword balance). RRF fusion across
    the four top-level signals provides ranking; no per-result reranker.
    """
    return SearchConfig(
        include_bm25=True,
        bm25_weight=0.5,
    )


async def search_fts5(
    db_path: str, query: str, limit: int = 10
) -> list[tuple[str, float]]:
    """Search fully-indexed documents using FTS5.

    Restricted to ``ingest_status='complete'`` so the RRF fusion below compares
    like with like. A document still in the deferred queue can only ever appear
    in this one signal — it has no chunks, no doc embedding and no cluster — so
    letting it into the fusion would have it scored as if three signals had
    *rejected* it rather than never having seen it. Those documents are
    retrieved separately by :func:`search_pending_keyword`.

    Args:
        db_path: Path to SQLite database
        query: Search query
        limit: Maximum results

    Returns:
        List of (doc_uuid, score) tuples, score normalized to 0-1
    """
    match_query = prepare_fts_query(query)
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute(
            """
            SELECT d.doc_uuid, rank
            FROM documents_fts fts
            JOIN documents d ON fts.rowid = d.id
            WHERE documents_fts MATCH ? AND d.deleted_at IS NULL
              AND d.ingest_status = 'complete'
            ORDER BY rank
            LIMIT ?
            """,
            (match_query, limit),
        ) as cursor:
            rows = await cursor.fetchall()

            # FTS5 rank is negative; convert to positive, higher = better
            results = [(row["doc_uuid"], abs(row["rank"])) for row in rows]

            if results:
                max_score = max(score for _, score in results)
                if max_score > 0:
                    results = [(doc_id, score / max_score) for doc_id, score in results]

            return results


def _search_doc_embeddings(
    db_path: str, query_embedding: list[float], limit: int = 20
) -> list[tuple[str, float]]:
    """Search document-level embeddings using vec0 cosine distance.

    Uses the sync get_connection() which loads sqlite-vec extension.
    Searches ALL documents regardless of tier.

    Args:
        db_path: Path to SQLite database
        query_embedding: Query embedding vector (768-dim)
        limit: Maximum results to return

    Returns:
        List of (doc_uuid, score) tuples, where score is 0-1 (higher is better)
    """
    from .connection import get_connection

    query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)

    try:
        with get_connection(Path(db_path)) as conn:
            # Check if doc_embeddings table exists and has data
            try:
                row = conn.execute("SELECT COUNT(*) FROM doc_embeddings").fetchone()
                if row is None or row[0] == 0:
                    return []
            except Exception:
                return []

            cursor = conn.execute(
                """
                SELECT d.doc_uuid, vec_distance_cosine(de.embedding, ?) as distance
                FROM doc_embeddings de
                JOIN documents d ON de.doc_id = d.id
                WHERE d.deleted_at IS NULL
                ORDER BY distance ASC
                LIMIT ?
                """,
                (query_bytes, limit),
            )
            rows = cursor.fetchall()

            if not rows:
                return []

            # Convert cosine distance to similarity score (0-1, higher is better)
            # vec_distance_cosine returns distance in [0, 2], where 0 = identical
            results = []
            for row in rows:
                doc_uuid = row[0] if isinstance(row, tuple) else row["doc_uuid"]
                distance = row[1] if isinstance(row, tuple) else row["distance"]
                score = max(0.0, 1.0 - distance)  # Convert distance to similarity
                results.append((doc_uuid, score))

            return results
    except Exception as e:
        logger.debug(f"Document-level embedding search failed: {type(e).__name__}: {e}")
        return []


#: Fraction of a result page reserved for keyword matches in documents whose
#: enrichment is still queued — one slot in five, at least one. The reservation
#: is only spent when such matches exist, so a database with an empty queue
#: returns exactly what it returned before this tail existed.
PENDING_TAIL_DIVISOR = 5


def _pending_quota(limit: int) -> int:
    """Slots reserved at the tail of a page for not-yet-enriched documents."""
    return max(1, limit // PENDING_TAIL_DIVISOR)


async def search_pending_keyword(
    db_path: str, query: str, limit: int = 2
) -> list[tuple[str, str, str]]:
    """Keyword-match documents that are still in the deferred ingestion queue.

    The counterpart to :func:`search_fts5`: that one sees only ``complete``
    documents, this one sees only the rest. Together they cover the corpus
    exactly once.

    ``markdown_content != ''`` is the load-bearing predicate, and it is a
    *content* test rather than a status test on purpose. A ``pending_source``
    row is a queue entry that happens to live in the ``documents`` table: its
    only indexed text is a title derived from the filename, so on a
    title-shaped query BM25 ranks it top while it has nothing to show — a
    contentless hit dressed as a real one. Testing content rather than status
    drops those, keeps ``pending_enrich`` documents (which hold real markdown),
    and also drops a ``failed`` conversion that left the row empty while
    keeping a ``failed`` enrichment that did not.

    Returns:
        List of (doc_uuid, ingest_status, markdown_content) tuples, best match
        first. The status is returned so the caller can label the result with
        the store's own vocabulary rather than inventing a parallel one.
    """
    match_query = prepare_fts_query(query)
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT d.doc_uuid, d.ingest_status, d.markdown_content
            FROM documents_fts fts
            JOIN documents d ON fts.rowid = d.id
            WHERE documents_fts MATCH ? AND d.deleted_at IS NULL
              AND d.ingest_status != 'complete'
              AND d.markdown_content != ''
            ORDER BY rank
            LIMIT ?
            """,
            (match_query, limit),
        ) as cursor:
            rows = await cursor.fetchall()

    return [
        (row["doc_uuid"], row["ingest_status"], row["markdown_content"] or "")
        for row in rows
    ]


async def _run_pending_tail(
    db_path: str,
    query: str,
    limit: int,
    doc_uuids: Optional[set[str]],
) -> list[UnifiedSearchResult]:
    """Build the labelled tail of not-yet-enriched keyword matches.

    Failure here is logged and yields an empty tail rather than aborting the
    search: the fused results are the answer, this is the addendum.
    """
    try:
        rows = await search_pending_keyword(db_path, query, _pending_quota(limit))
    except Exception as e:
        logger.warning(
            f"Pending-document keyword tail FAILED — documents awaiting "
            f"enrichment are missing from this search: {type(e).__name__}: {e}"
        )
        return []

    results: list[UnifiedSearchResult] = []
    for doc_uuid, status, content in rows:
        if doc_uuids is not None and doc_uuid not in doc_uuids:
            continue
        results.append(
            UnifiedSearchResult(
                doc_id=doc_uuid,
                score=0.0,  # assigned by _append_pending_tail, below the fused floor
                # The store's own queue vocabulary, carried through
                # UnifiedSearchResult.tier -> SearchResult.match_type. No new
                # field, and no second name for a state that already has one.
                tier=status,
                # Bounded here so the caller never falls back to the full
                # markdown: these documents have no chunks, so the fallback in
                # public.search() would return an entire converted PDF as the
                # "snippet".
                snippet=extract_snippet(content, query, max_length=500),
                metadata=SearchResultMetadata(match_type=status),
            )
        )
    return results


def _append_pending_tail(
    fused: list[UnifiedSearchResult],
    tail: list[UnifiedSearchResult],
    limit: int,
) -> list[UnifiedSearchResult]:
    """Append ``tail`` below ``fused``, honouring ``limit``.

    Tail scores are pushed strictly below the lowest fused score so the page
    stays sorted by score. Appending without doing that would leave a caller
    who re-sorts by score — a reasonable thing to do — interleaving keyword-only
    matches among fully-indexed ones, which is precisely what this design
    avoids. RRF scores and normalized BM25 scores are not comparable
    quantities; the ordering is the claim, not the magnitude.
    """
    if not tail:
        return fused[:limit]

    kept_tail = tail[: _pending_quota(limit)]
    head = fused[: max(0, limit - len(kept_tail))]

    floor = min((r.score for r in head), default=0.0)
    for rank, result in enumerate(kept_tail):
        # Strictly decreasing, strictly below the floor, never negative.
        result.score = floor * (0.9 ** (rank + 1)) if floor > 0 else 0.0

    return head + kept_tail


async def _run_fts5_signal(
    db_path: str, query: str, limit: int
) -> Optional[tuple[str, list[UnifiedSearchResult]]]:
    """Signal 1: FTS5 keyword search.

    A failure here does not abort the search — the other signals still answer —
    but it is logged at WARNING, not DEBUG. This signal failing silently is
    exactly how a malformed MATCH query (``fts5: syntax error``) hid for so long:
    keyword ranking vanished from the fusion and the caller saw only slightly
    worse results, with the cause invisible at default log level.
    """
    try:
        fts5_results = await search_fts5(db_path, query, limit * 2)
        if fts5_results:
            logger.debug(f"FTS5 search returned {len(fts5_results)} results")
            return (
                "fts5",
                [
                    UnifiedSearchResult(doc_id=doc_id, score=score, tier="fts5")
                    for doc_id, score in fts5_results
                ],
            )
    except Exception as e:
        logger.warning(
            f"FTS5 keyword signal FAILED and is missing from this search's "
            f"fusion — results are degraded: {type(e).__name__}: {e}"
        )
    return None


async def _run_chunk_search(
    db_path: str, query: str, query_embedding: Optional[list[float]], limit: int
) -> Optional[tuple[str, list[UnifiedSearchResult]]]:
    """Signal 2: Chunk-level semantic search (hybrid BM25 + embeddings)."""
    try:
        config = _get_production_search_config()

        features = []
        if config.include_bm25:
            features.append(f"BM25({config.bm25_weight:.0%})")
        if query_embedding:
            features.append("semantic")
        logger.info(
            f"Chunk search config: {' + '.join(features) if features else 'keyword-only'}"
        )

        start_time = time.perf_counter()
        rag_results = await asyncio.to_thread(
            multi_strategy_search,
            query=query,
            query_embedding=query_embedding,
            limit=limit * 2,
            db_path=Path(db_path),
            config=config,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Chunk search completed: {len(rag_results)} results in {elapsed_ms:.0f}ms"
        )

        if rag_results:
            chunk_results: list[UnifiedSearchResult] = []
            for r in rag_results:
                metadata = SearchResultMetadata(
                    match_type=r.match_type,
                    entity_matches=r.metadata.get("matched_entities", [])
                    if r.metadata
                    else [],
                    tag_matches=r.metadata.get("matched_tags", [])
                    if r.metadata
                    else [],
                )
                # A chunk that came back without an index cannot be named,
                # so it ships as a snippet only rather than as a resolvable
                # ChunkHit carrying an id that would not resolve.
                hits: list[ChunkHit] = []
                if r.chunk_index is not None:
                    hits.append(
                        ChunkHit(
                            chunk_id=chunk_id_for(
                                doc_id=r.doc_id, chunk_index=r.chunk_index
                            ),
                            doc_id=r.doc_id,
                            chunk_index=r.chunk_index,
                            content=r.content,
                            start_char=r.start_char,
                            end_char=r.end_char,
                            score=r.similarity_score,
                            token_count=r.token_count,
                            metadata=dict(r.metadata) if r.metadata else {},
                        )
                    )
                chunk_results.append(
                    UnifiedSearchResult(
                        doc_id=r.doc_id,
                        score=r.similarity_score,
                        tier="chunks",
                        snippet=r.content,
                        metadata=metadata,
                        chunks=hits,
                    )
                )
            logger.debug(f"Chunk search returned {len(chunk_results)} results")
            return ("chunks", chunk_results)
    except Exception as e:
        logger.warning(f"Chunk search failed: {type(e).__name__}: {e}")
    return None


async def _run_doc_embedding_search(
    db_path: str, query_embedding: Optional[list[float]], limit: int
) -> Optional[tuple[str, list[UnifiedSearchResult]]]:
    """Signal 3: Document-level embedding search."""
    if query_embedding is None:
        return None
    try:
        doc_level_results = await asyncio.to_thread(
            _search_doc_embeddings, db_path, query_embedding, limit * 2
        )
        if doc_level_results:
            logger.debug(
                f"Document-level semantic search returned {len(doc_level_results)} results"
            )
            return (
                "doc_semantic",
                [
                    UnifiedSearchResult(
                        doc_id=doc_id,
                        score=score,
                        tier="doc_semantic",
                        metadata=SearchResultMetadata(match_type="doc_embedding"),
                    )
                    for doc_id, score in doc_level_results
                ],
            )
    except Exception as e:
        logger.warning(
            f"Document-level semantic signal FAILED and is missing from this "
            f"search's fusion — results are degraded: {type(e).__name__}: {e}"
        )
    return None


async def _run_cluster_search(
    db_path: str, query_embedding: Optional[list[float]], limit: int
) -> Optional[tuple[str, list[UnifiedSearchResult]]]:
    """Signal 4: DHP temporal cluster scoring."""
    if query_embedding is None:
        return None
    try:
        cluster_results = await asyncio.to_thread(
            _search_via_clusters, db_path, query_embedding, limit * 2
        )
        if cluster_results:
            logger.debug(
                f"Cluster-boosted search returned {len(cluster_results)} results"
            )
            return (
                "cluster",
                [
                    UnifiedSearchResult(
                        doc_id=doc_id,
                        score=score,
                        tier="cluster",
                        metadata=SearchResultMetadata(match_type="cluster_temporal"),
                    )
                    for doc_id, score in cluster_results
                ],
            )
    except Exception as e:
        # A database with no clusters yet is NOT an error — _search_via_clusters
        # returns an empty list for that. Reaching here means something actually
        # broke, so it is a degraded search, not a skip.
        logger.warning(
            f"DHP cluster signal FAILED and is missing from this search's "
            f"fusion — results are degraded: {type(e).__name__}: {e}"
        )
    return None


async def search_unified(
    db_path: str,
    query: str,
    limit: int = 10,
    query_embedding: Optional[list[float]] = None,
    doc_uuids: Optional[set[str]] = None,
    embedding_model: Optional[str] = None,
) -> list[UnifiedSearchResult]:
    """Unified search across all documents with RRF fusion.

    Runs four signals over fully-indexed documents and fuses them with
    Reciprocal Rank Fusion (RRF, k=60):
    1. FTS5 keyword search
    2. Chunk-level semantic search (hybrid BM25 + embeddings)
    3. Document-level semantic search (doc_embeddings vec0)
    4. DHP temporal cluster scoring

    Documents still awaiting enrichment are deliberately *not* fused: they can
    only ever appear in signal 1, so summing reciprocal ranks across four
    signals reads their missing index as three rejections and buries them. They
    are retrieved separately and appended as a labelled tail below the fused
    results, within ``limit`` — see :func:`_append_pending_tail`. A database
    with an empty ingestion queue is unaffected: the tail is empty and the page
    is exactly the fused list, ranked exactly as before.

    Args:
        db_path: Path to SQLite database
        query: Search query
        limit: Maximum results to return
        query_embedding: Optional pre-computed embedding (generated if not provided)
        doc_uuids: Optional set of document UUIDs to restrict results to.
            When provided, only results matching these doc_ids are returned.
            Used by the query planner for metadata pre-filtering.
        embedding_model: Model name for embedding generation (required when query_embedding is None)

    Returns:
        List of UnifiedSearchResult objects sorted by score.
    """
    all_results: list[tuple[str, list[UnifiedSearchResult]]] = []

    # Generate embedding first — gates signals 2-4
    if query_embedding is None and embedding_model is not None:
        try:
            embed_svc = OllamaEmbedder(model=embedding_model)
            try:
                query_embedding = await embed_svc.embed_text(query)
            finally:
                await embed_svc.close()
            logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")
        except Exception as e:
            logger.warning(f"Embedding generation failed: {type(e).__name__}: {e}")

    # Run all 4 signals in parallel
    signal_results = await asyncio.gather(
        _run_fts5_signal(db_path, query, limit),
        _run_chunk_search(db_path, query, query_embedding, limit),
        _run_doc_embedding_search(db_path, query_embedding, limit),
        _run_cluster_search(db_path, query_embedding, limit),
    )

    for result in signal_results:
        if result is not None:
            all_results.append(result)

    # Keyword matches among documents still in the deferred queue. Fetched
    # after the signals rather than alongside them: it is one indexed FTS
    # lookup, and keeping it out of the gather keeps that gather homogeneous.
    pending_tail = await _run_pending_tail(db_path, query, limit, doc_uuids)

    # No results from any signal
    if not all_results:
        return _append_pending_tail([], pending_tail, limit)

    # If only one signal, return directly (with optional doc_uuids filter)
    if len(all_results) == 1:
        _, results = all_results[0]
        if doc_uuids is not None:
            results = [r for r in results if r.doc_id in doc_uuids]
        return _append_pending_tail(results, pending_tail, limit)

    # RRF fusion across all signals (k=60, industry standard)
    doc_lookup: dict[str, UnifiedSearchResult] = {}
    rrf_scores: dict[str, float] = {}
    # Chunk hits accumulate per document instead of collapsing with it. The
    # chunk signal emits one result per matching CHUNK, so a document with
    # three good passages arrives here three times; fusion is document-level
    # and keeps one of those rows, which would silently discard two real
    # passages. Keyed by chunk id so the same chunk arriving from two signals
    # is one hit, keeping the better score.
    chunks_by_doc: dict[str, dict[str, ChunkHit]] = {}

    for _, signal_results_list in all_results:
        for rank, result in enumerate(signal_results_list):
            # Skip results not in the filtered set
            if doc_uuids is not None and result.doc_id not in doc_uuids:
                continue
            rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0.0) + 1.0 / (
                rank + RRF_K
            )
            if result.chunks:
                seen = chunks_by_doc.setdefault(result.doc_id, {})
                for hit in result.chunks:
                    previous = seen.get(hit.chunk_id)
                    if previous is None or (hit.score or 0.0) > (previous.score or 0.0):
                        seen[hit.chunk_id] = hit
            existing = doc_lookup.get(result.doc_id)
            if existing is None or _tier_priority(result.tier) > _tier_priority(
                existing.tier
            ):
                doc_lookup[result.doc_id] = result

    # Sort by RRF score
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final results with fused scores
    final_results: list[UnifiedSearchResult] = []
    for doc_id, score in fused_results[:limit]:
        original = doc_lookup[doc_id]
        hits = sorted(
            chunks_by_doc.get(doc_id, {}).values(),
            key=lambda hit: hit.score or 0.0,
            reverse=True,
        )
        final_results.append(
            UnifiedSearchResult(
                doc_id=doc_id,
                score=score,
                tier=original.tier,
                snippet=original.snippet,
                metadata=original.metadata,
                chunks=hits,
            )
        )

    return _append_pending_tail(final_results, pending_tail, limit)


def _tier_priority(tier: str) -> int:
    """Return priority for signal selection in RRF fusion.

    When the same document appears in multiple signals, the one with
    the highest priority is used for metadata. Richer metadata wins.
    """
    priorities = {"chunks": 3, "cluster": 2, "doc_semantic": 2, "fts5": 1}
    return priorities.get(tier, 0)


def _search_via_clusters(
    db_path: str, query_embedding: list[float], limit: int = 20
) -> list[tuple[str, float]]:
    """Score documents via DHP temporal cluster relevance.

    Uses cached cluster state (5-minute TTL) to avoid per-query JSON parsing.

    Args:
        db_path: Path to SQLite database
        query_embedding: Query embedding vector (768-dim)
        limit: Maximum results to return

    Returns:
        List of (doc_uuid, score) tuples, or empty list if no clusters exist.
    """
    import sqlite3

    import numpy as np

    try:
        cluster_states, config = _load_cluster_state(db_path)
        if not cluster_states:
            return []

        from .dhp import score_clusters_for_query, timestamp_to_hours

        current_time = timestamp_to_hours(time.time())
        query_emb = np.array(query_embedding, dtype=np.float64)
        scored = score_clusters_for_query(
            query_emb, current_time, cluster_states, config
        )

        if not scored:
            return []

        # Get documents from top clusters (take top 3 clusters max)
        top_cluster_ids = [cid for cid, _ in scored[:3]]
        placeholders = ",".join("?" * len(top_cluster_ids))

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            doc_rows = conn.execute(
                f"SELECT doc_uuid, cluster_id FROM documents WHERE cluster_id IN ({placeholders}) AND deleted_at IS NULL LIMIT ?",
                (*top_cluster_ids, limit),
            ).fetchall()
        finally:
            conn.close()

        cluster_score_map = {cid: score for cid, score in scored}
        results = []
        for dr in doc_rows:
            cscore = cluster_score_map.get(dr["cluster_id"], 0.0)
            results.append((dr["doc_uuid"], cscore))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    except Exception as e:
        logger.debug(f"Cluster search failed: {e}")
        return []


async def search_multi_database(
    query: str, database_names: list[str], limit: int = 10
) -> list[MultiDatabaseSearchResult]:
    """Search across multiple named databases with RRF fusion.

    Enhanced with production-quality features inherited from search_unified():
    - Rich metadata in results (match type, tags)

    Args:
        query: Search query
        database_names: List of database names to search (e.g., ["research", "papers"])
        limit: Max results to return

    Returns:
        List of MultiDatabaseSearchResult objects sorted by relevance.
    """
    from .lifecycle import get_db_path

    all_results: list[MultiDatabaseSearchResult] = []

    for db_name in database_names:
        db_path = get_db_path(db_name)

        if not db_path.exists():
            continue

        # Search this database (returns UnifiedSearchResult objects)
        results = await search_unified(
            db_path=str(db_path),
            query=query,
            limit=limit * 2,  # Over-fetch for RRF
        )

        # Convert to MultiDatabaseSearchResult with database name
        for result in results:
            all_results.append(
                MultiDatabaseSearchResult(
                    doc_id=result.doc_id,
                    score=result.score,
                    tier=result.tier,
                    metadata=result.metadata,
                    database_name=db_name,
                )
            )

    if not all_results:
        return []

    # RRF fusion across all databases (k=60)
    # Key is (doc_id, db_name) to keep same doc from different DBs separate
    rrf_data: dict[tuple[str, str], MultiDatabaseSearchResult] = {}
    rrf_scores: dict[tuple[str, str], float] = {}

    for rank, result in enumerate(
        sorted(all_results, key=lambda x: x.score, reverse=True)
    ):
        key = (result.doc_id, result.database_name)
        if key not in rrf_data:
            rrf_data[key] = result
            rrf_scores[key] = 0.0
        rrf_scores[key] += 1.0 / (rank + RRF_K)

    # Sort by final RRF score and build results
    sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results: list[MultiDatabaseSearchResult] = []
    for key, score in sorted_keys[:limit]:
        original = rrf_data[key]
        final_results.append(
            MultiDatabaseSearchResult(
                doc_id=original.doc_id,
                score=score,
                tier=original.tier,
                metadata=original.metadata,
                database_name=original.database_name,
            )
        )

    return final_results
