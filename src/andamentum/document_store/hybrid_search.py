"""Multi-strategy hybrid search combining semantic, keyword, and tag-based search."""

from __future__ import annotations
from typing import List, Optional, Literal
from pathlib import Path

from .connection import get_connection, DEFAULT_DB_PATH
from .rag.search import SearchResult, SearchConfig, semantic_search, reciprocal_rank_fusion
from .fts import fts_search
import json


def multi_strategy_search(
    query: str,
    query_embedding: Optional[List[float]] = None,
    strategy: Literal["auto", "semantic", "keyword", "hybrid", "tag"] = "auto",
    limit: int = 10,
    tag_filter: Optional[List[str]] = None,
    config: Optional[SearchConfig] = None,
    db_path: Optional[Path] = None,
) -> List[SearchResult]:
    """Unified search interface supporting multiple strategies.

    Automatically selects best strategy or allows manual override.
    Combines semantic (vector), keyword (FTS5), and tag-based search.

    Args:
        query: Search query string
        query_embedding: Query embedding vector (768-dim, required for semantic search)
        strategy: Search strategy to use:
            - "auto": Automatically select best strategy (default)
            - "semantic": Pure vector similarity search
            - "keyword": Pure FTS5 keyword search
            - "hybrid": Combine vector + keyword (50/50 or custom weights)
            - "tag": Tag-based search (filter by tags, then rank by relevance)
        limit: Maximum number of results
        tag_filter: Optional list of tag names to filter by (works with all strategies)
        config: Optional search configuration (for hybrid mode)
        db_path: Path to database file (uses default if None)

    Returns:
        List of SearchResult objects, ordered by relevance

    Strategy Selection Logic (when strategy="auto"):
        - Query has quotes ("...") → keyword (phrase search)
        - Query has boolean operators (AND/OR/NOT) → keyword
        - Query has wildcards (*) → keyword
        - Query is very short (<= 3 words) AND has embedding → hybrid
        - Query is long (> 10 words) AND has embedding → semantic
        - Default → hybrid (if embedding available) or keyword (if not)
    """
    if config is None:
        config = SearchConfig()

    # Auto strategy selection
    if strategy == "auto":
        strategy = _select_strategy(query, query_embedding)

    # Execute selected strategy
    if strategy == "semantic":
        if query_embedding is None:
            raise ValueError("query_embedding required for semantic search")
        results = semantic_search(query, query_embedding, limit * 2, config, db_path)

    elif strategy == "keyword":
        results = fts_search(query, limit * 2, db_path)

    elif strategy == "hybrid":
        if query_embedding is None:
            # Fall back to keyword if no embedding
            results = fts_search(query, limit * 2, db_path)
        else:
            results = semantic_search(query, query_embedding, limit * 2, config, db_path)

    elif strategy == "tag":
        if not tag_filter:
            raise ValueError("tag_filter required for tag strategy")
        # Get all documents with matching tags, then rank by relevance
        results = _tag_based_search(query, query_embedding, tag_filter, limit * 2, db_path)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply tag filter if provided (works with all strategies)
    if tag_filter and strategy != "tag":
        results = _filter_by_tags(results, tag_filter, db_path)

    # Return top results
    return results[:limit]


def _select_strategy(
    query: str,
    query_embedding: Optional[List[float]]
) -> Literal["semantic", "keyword", "hybrid"]:
    """Automatically select best search strategy based on query characteristics."""
    # Check for FTS5-specific syntax
    if '"' in query:  # Phrase search
        return "keyword"
    if any(op in query.upper() for op in [' AND ', ' OR ', ' NOT ']):  # Boolean
        return "keyword"
    if '*' in query:  # Wildcard
        return "keyword"

    # If no embedding available, use keyword
    if query_embedding is None:
        return "keyword"

    # Query length analysis
    word_count = len(query.split())

    if word_count <= 3:
        return "hybrid"
    elif word_count > 10:
        return "semantic"
    else:
        return "hybrid"


def _tag_based_search(
    query: str,
    query_embedding: Optional[List[float]],
    tags: List[str],
    limit: int,
    db_path: Optional[Path]
) -> List[SearchResult]:
    """Tag-filtered search using RRF fusion of semantic + keyword strategies."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Stage 1: Get document IDs matching tags (OR logic)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        placeholders = ','.join('?' * len(tags))
        cursor.execute(f"""
            SELECT DISTINCT d.id
            FROM documents d
            JOIN document_tags dt ON d.id = dt.document_id
            JOIN tags t ON dt.tag_id = t.id
            WHERE t.name IN ({placeholders})
        """, tags)

        doc_ids = [row[0] for row in cursor.fetchall()]

        if not doc_ids:
            return []

    # Stage 2: Run semantic search on tag-filtered chunks
    semantic_results = []
    if query_embedding:
        doc_placeholders = ','.join('?' * len(doc_ids))
        query_blob = json.dumps(query_embedding)

        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT
                    c.content, c.start_char, c.end_char,
                    c.metadata, c.token_count,
                    d.doc_uuid as doc_id, d.file_path,
                    d.dc_title, d.dc_format, d.dc_creator, d.dc_subject,
                    vec_distance_cosine(ce.embedding, ?) as distance
                FROM chunk_embeddings ce
                JOIN chunks c ON ce.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE d.id IN ({doc_placeholders})
                ORDER BY distance ASC
                LIMIT ?
            """, (query_blob, *doc_ids, limit * 2))

            rows = cursor.fetchall()

            for row in rows:
                similarity = 1.0 - row['distance']
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                metadata['matched_tags'] = tags
                dc_subject = json.loads(row['dc_subject']) if row['dc_subject'] else None

                semantic_results.append(SearchResult(
                    content=row['content'],
                    file_path=row['file_path'],
                    doc_id=row['doc_id'],
                    similarity_score=similarity,
                    match_type="semantic",
                    start_char=row['start_char'],
                    end_char=row['end_char'],
                    token_count=row['token_count'],
                    dc_title=row['dc_title'],
                    dc_format=row['dc_format'],
                    dc_creator=row['dc_creator'],
                    dc_subject=dc_subject,
                    para_type=metadata.get('para_type'),
                    metadata=metadata
                ))

    # Stage 3: Run keyword search on tag-filtered chunks
    keyword_results = []
    doc_placeholders = ','.join('?' * len(doc_ids))

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT
                c.content, c.start_char, c.end_char,
                c.metadata, c.token_count,
                d.doc_uuid as doc_id, d.file_path,
                d.dc_title, d.dc_format, d.dc_creator, d.dc_subject,
                cf.rank as bm25_score
            FROM chunks_fts cf
            JOIN chunks c ON cf.rowid = c.rowid
            JOIN documents d ON c.document_id = d.id
            WHERE chunks_fts MATCH ?
              AND d.id IN ({doc_placeholders})
            ORDER BY bm25_score
            LIMIT ?
        """, (query, *doc_ids, limit * 2))

        rows = cursor.fetchall()

        for row in rows:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            metadata['matched_tags'] = tags
            dc_subject = json.loads(row['dc_subject']) if row['dc_subject'] else None

            keyword_results.append(SearchResult(
                content=row['content'],
                file_path=row['file_path'],
                doc_id=row['doc_id'],
                similarity_score=abs(row['bm25_score']),
                match_type="keyword",
                start_char=row['start_char'],
                end_char=row['end_char'],
                token_count=row['token_count'],
                dc_title=row['dc_title'],
                dc_format=row['dc_format'],
                dc_creator=row['dc_creator'],
                dc_subject=dc_subject,
                para_type=metadata.get('para_type'),
                metadata=metadata
            ))

    # Stage 4: Fuse with RRF
    if not semantic_results and not keyword_results:
        return []
    elif not semantic_results:
        for result in keyword_results:
            result.match_type = "tag+keyword"
        return keyword_results[:limit]
    elif not keyword_results:
        for result in semantic_results:
            result.match_type = "tag+semantic"
        return semantic_results[:limit]
    else:
        fused_results = reciprocal_rank_fusion(
            result_lists=[semantic_results, keyword_results],
            k=60
        )

        for result in fused_results:
            result.match_type = "tag+rrf_ensemble"
            if 'matched_tags' not in result.metadata:
                result.metadata['matched_tags'] = tags

        return fused_results[:limit]


def _filter_by_tags(
    results: List[SearchResult],
    tags: List[str],
    db_path: Optional[Path]
) -> List[SearchResult]:
    """Filter search results to only include documents with matching tags."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        file_paths = list(set(r.file_path for r in results))

        if not file_paths:
            return []

        placeholders = ','.join('?' * len(tags))
        file_placeholders = ','.join('?' * len(file_paths))

        cursor.execute(f"""
            SELECT DISTINCT d.file_path
            FROM documents d
            JOIN document_tags dt ON d.id = dt.document_id
            JOIN tags t ON dt.tag_id = t.id
            WHERE t.name IN ({placeholders})
            AND d.file_path IN ({file_placeholders})
        """, tags + file_paths)

        tagged_files = {row[0] for row in cursor.fetchall()}

    return [r for r in results if r.file_path in tagged_files]
