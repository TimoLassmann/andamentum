"""Semantic and hybrid search functionality.

Combines vector similarity search with BM25 keyword scoring.

Usage:
    from andamentum.document_store.rag.search import semantic_search, SearchConfig

    results = semantic_search(
        query="machine learning",
        query_embedding=[0.1, 0.2, ...],  # 768-dim vector
        limit=10,
        config=SearchConfig(include_bm25=True)
    )
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from rank_bm25 import BM25Okapi

from .database import search_chunks, get_connection, DEFAULT_DB_PATH


@dataclass
class SearchConfig:
    """Configuration for semantic search and re-ranking."""

    include_bm25: bool = True  # Enable BM25 keyword scoring
    bm25_weight: float = 0.5  # 50% keyword, 50% vector (balanced hybrid)
    min_similarity: float = 0.0  # Minimum similarity threshold (0-1)

    # Cross-encoder re-ranking configuration
    enable_reranking: bool = (
        True  # Enable cross-encoder re-ranking (production quality)
    )
    reranking_model: str = (
        "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Cross-encoder model name
    )
    reranking_top_k: Optional[int] = (
        20  # Re-rank top 20 results for optimal speed/quality balance
    )


@dataclass
class SearchResult:
    """A search result with file context and metadata."""

    content: str  # Chunk content
    file_path: str  # Relative file path
    doc_id: str  # Document UUID (for DocumentStore.read() integration)
    similarity_score: float  # Combined score (0-1)
    match_type: str  # "semantic", "keyword", or "hybrid"

    # Chunk position
    start_char: int
    end_char: int
    token_count: Optional[int] = None

    # Document metadata
    dc_title: Optional[str] = None
    dc_format: Optional[str] = None
    dc_creator: Optional[str] = None
    dc_subject: Optional[List[str]] = None
    para_type: Optional[str] = None  # Paragraph type from Docling chunker

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range using min-max normalization.

    Args:
        scores: List of raw scores

    Returns:
        List of normalized scores (0-1)
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    # Avoid division by zero
    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]], k: int = 60
) -> List[SearchResult]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF is a parameter-free, score-independent fusion method that combines
    rankings from multiple search strategies. Industry standard with k=60.

    Formula: score = 1 / (rank + k)

    Deduplication operates at the chunk level using (doc_id, start_char) tuple
    to preserve chunk-level diversity within the same document.

    Args:
        result_lists: List of result lists from different strategies
        k: RRF constant (default 60, industry standard)

    Returns:
        Fused and re-ranked results

    References:
        Cormack et al. "Reciprocal Rank Fusion"
    """
    from typing import Tuple

    scores: Dict[Tuple[str, int], float] = {}
    results_by_chunk: Dict[Tuple[str, int], SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (rank + k)
            chunk_key = (result.doc_id, result.start_char)
            scores[chunk_key] = scores.get(chunk_key, 0.0) + rrf_score
            if chunk_key not in results_by_chunk:
                # Update match_type to indicate RRF ensemble fusion
                results_by_chunk[chunk_key] = SearchResult(
                    content=result.content,
                    file_path=result.file_path,
                    doc_id=result.doc_id,
                    similarity_score=result.similarity_score,
                    match_type="rrf_ensemble",
                    start_char=result.start_char,
                    end_char=result.end_char,
                    token_count=result.token_count,
                    dc_title=result.dc_title,
                    dc_format=result.dc_format,
                    dc_creator=result.dc_creator,
                    dc_subject=result.dc_subject,
                    para_type=result.para_type,
                )

    sorted_chunks = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [results_by_chunk[chunk_key] for chunk_key in sorted_chunks]


def semantic_search(
    query: str,
    query_embedding: List[float],
    limit: int = 10,
    config: Optional[SearchConfig] = None,
    db_path: Optional[Path] = None,
) -> List[SearchResult]:
    """Search using vector similarity + optional BM25 keyword scoring.

    This is the core search logic combining:
    - Vector similarity (from database via cosine distance)
    - BM25 keyword scoring (from rank_bm25 library)
    - Weighted fusion of normalized scores

    Args:
        query: Search query string (for BM25)
        query_embedding: Query embedding vector (768-dim)
        limit: Maximum number of results
        config: Optional search configuration
        db_path: Path to database file (uses default if None)

    Returns:
        List of SearchResult objects, sorted by relevance
    """
    if config is None:
        config = SearchConfig()

    # 1. Vector similarity search
    # Get 3x limit to have enough candidates for BM25 re-ranking
    # Reduced from 10x — 3x provides sufficient candidate diversity with lower cost
    vector_limit = limit * 3 if config.include_bm25 else limit

    vector_results = search_chunks(
        query_embedding=query_embedding, limit=vector_limit, db_path=db_path
    )

    if not vector_results:
        return []

    # 2. If BM25 enabled, combine with keyword scoring
    if config.include_bm25 and len(vector_results) > 0:
        # Tokenize corpus (all chunk contents)
        corpus = [result["content"].lower().split() for result in vector_results]

        # Build BM25 index
        bm25 = BM25Okapi(corpus)

        # Score query
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Normalize both score types to 0-1 range
        vector_scores = [
            1 - result["distance"] for result in vector_results
        ]  # Convert distance to similarity
        vector_normalized = normalize_scores(vector_scores)
        bm25_normalized = normalize_scores(
            list(bm25_scores)
        )  # Convert numpy array to list

        # Combine scores: weighted sum
        for i, result in enumerate(vector_results):
            combined_score = (1 - config.bm25_weight) * vector_normalized[
                i
            ] + config.bm25_weight * bm25_normalized[i]
            result["similarity_score"] = combined_score
            result["match_type"] = "hybrid"

        # Re-sort by combined score (descending)
        vector_results.sort(key=lambda x: x["similarity_score"], reverse=True)

    else:
        # Pure vector search - convert distance to similarity
        for result in vector_results:
            result["similarity_score"] = 1 - result["distance"]
            result["match_type"] = "semantic"

    # 3. Filter by minimum similarity threshold
    filtered_results = [
        r for r in vector_results if r["similarity_score"] >= config.min_similarity
    ]

    # 4. Convert to SearchResult objects
    search_results = []
    for result in filtered_results[:limit]:
        search_results.append(
            SearchResult(
                content=result["content"],
                file_path=result["file_path"],
                doc_id=result["doc_id"],
                similarity_score=result["similarity_score"],
                match_type=result["match_type"],
                start_char=result["start_char"],
                end_char=result["end_char"],
                token_count=result.get("token_count"),
                dc_title=result.get("dc_title"),
                dc_format=result.get("dc_format"),
                dc_creator=result.get("dc_creator"),
                dc_subject=result.get("dc_subject"),
                metadata=result.get("metadata", {}),
            )
        )

    # 5. Optional cross-encoder re-ranking for improved relevance
    if config.enable_reranking and search_results:
        from .reranking import rerank_results, RerankingConfig

        rerank_config = RerankingConfig(model_name=config.reranking_model)
        search_results = rerank_results(
            query=query,
            results=search_results,
            top_k=config.reranking_top_k,
            config=rerank_config,
        )

    return search_results


def _sanitize_fts5_query(query: str) -> str:
    """Sanitize query for FTS5 MATCH by removing/escaping special characters.

    FTS5 special characters that cause syntax errors:
    - () : grouping operators
    - : : column prefix
    - " : phrase delimiter
    - * : prefix wildcard
    - ? : invalid character
    - - : NOT operator (e.g., "semantic-based" → "semantic NOT based")
    - . : can cause column errors
    - / : path separator causing syntax errors
    - % : wildcard character causing syntax errors

    For natural language queries, we want simple word matching, not operators.

    Args:
        query: Raw search query

    Returns:
        Sanitized query safe for FTS5 MATCH
    """
    import re

    # Replace hyphens with spaces (avoid NOT operator interpretation)
    query = query.replace("-", " ")

    # Remove FTS5 special characters: ():"*?.,/%
    query = re.sub(r'[():"*?.,/%]', " ", query)

    # Normalize whitespace
    query = " ".join(query.split())

    return query


def fts_search(
    query: str, limit: int = 10, db_path: Optional[Path] = None
) -> List[SearchResult]:
    """Chunk-level full-text search using SQLite FTS5.

    Fast keyword search on chunk content using Porter stemming and BM25 ranking.
    This provides accurate TF-IDF statistics at chunk granularity, aligning with
    Anthropic's contextual retrieval approach.

    Args:
        query: Search query (natural language - special chars auto-sanitized)
        limit: Maximum number of results
        db_path: Path to database file (uses default if None)

    Returns:
        List of SearchResult objects (chunk-level), ordered by BM25 relevance

    Examples:
        >>> # Simple keyword search
        >>> results = fts_search("machine learning")

        >>> # Natural language queries (auto-sanitized)
        >>> results = fts_search("What is semantic-based clustering?")
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Sanitize query for FTS5 (removes operators, special chars)
    sanitized_query = _sanitize_fts5_query(query)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Chunk-level FTS5 search with BM25 ranking on full corpus
        cursor.execute(
            """
            SELECT
                c.id as chunk_id,
                c.content,
                c.start_char,
                c.end_char,
                c.token_count,
                c.metadata,
                d.doc_uuid as doc_id,
                d.file_path,
                d.dc_title,
                d.dc_format,
                d.dc_creator,
                d.dc_subject,
                fts.rank
            FROM chunks_fts fts
            JOIN chunks c ON fts.rowid = c.id
            JOIN documents d ON c.document_id = d.id
            WHERE chunks_fts MATCH ? AND d.deleted_at IS NULL
            ORDER BY fts.rank
            LIMIT ?
        """,
            (sanitized_query, limit),
        )

        rows = cursor.fetchall()

        # Convert to SearchResult objects
        results = []
        for row in rows:
            (
                chunk_id,
                content,
                start_char,
                end_char,
                token_count,
                metadata_json,
                doc_id,
                file_path,
                dc_title,
                dc_format,
                dc_creator,
                dc_subject,
                rank,
            ) = row

            # Parse metadata
            metadata = {}
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except Exception:
                    pass

            # Parse dc_subject
            dc_subject_list = None
            if dc_subject:
                try:
                    dc_subject_list = json.loads(dc_subject)
                except Exception:
                    dc_subject_list = (
                        dc_subject.split(",") if isinstance(dc_subject, str) else None
                    )

            # Convert rank (negative value) to positive similarity score (0-1)
            # FTS5 rank is negative, more negative = better match
            # Normalize to 0-1 range (1 = best match)
            similarity_score = min(1.0, abs(rank) / 10.0)  # Heuristic normalization

            # Add FTS rank to metadata
            metadata["fts_rank"] = rank

            # Extract para_type from metadata if available
            para_type = metadata.get("para_type")

            results.append(
                SearchResult(
                    content=content,
                    file_path=file_path,
                    doc_id=doc_id,
                    similarity_score=similarity_score,
                    match_type="keyword",
                    start_char=start_char,
                    end_char=end_char,
                    token_count=token_count,
                    dc_title=dc_title,
                    dc_format=dc_format,
                    dc_creator=dc_creator,
                    dc_subject=dc_subject_list,
                    para_type=para_type,
                    metadata=metadata,
                )
            )

        return results


def _extract_snippet(content: str, query: str, max_length: int = 500) -> str:
    """Extract snippet around first occurrence of query terms.

    Args:
        content: Full document content
        query: Search query
        max_length: Maximum snippet length

    Returns:
        Snippet with query context
    """
    if not content:
        return ""

    # Simple tokenization - split query into terms
    query_terms = [
        term.strip('"').lower()
        for term in query.split()
        if term.strip('"').lower() not in {"and", "or", "not"}
    ]

    if not query_terms:
        # No valid query terms, return beginning
        return content[:max_length] + ("..." if len(content) > max_length else "")

    # Find first occurrence of any query term
    content_lower = content.lower()
    first_match_pos = -1

    for term in query_terms:
        pos = content_lower.find(term.rstrip("*"))  # Handle wildcard
        if pos != -1 and (first_match_pos == -1 or pos < first_match_pos):
            first_match_pos = pos

    if first_match_pos == -1:
        # No match found (shouldn't happen with FTS5), return beginning
        return content[:max_length] + ("..." if len(content) > max_length else "")

    # Extract snippet centered on match
    snippet_start = max(0, first_match_pos - max_length // 3)
    snippet_end = min(len(content), snippet_start + max_length)

    snippet = content[snippet_start:snippet_end]

    # Add ellipsis if truncated
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(content):
        snippet = snippet + "..."

    return snippet


def ensemble_search(
    query: str,
    query_embedding: List[float],
    limit: int = 10,
    semantic_limit: int = 50,
    keyword_limit: int = 50,
    enable_reranking: bool = False,
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    db_path: Optional[Path] = None,
) -> List[SearchResult]:
    """True ensemble search with independent strategies, RRF fusion, and optional re-ranking.

    Implements Anthropic's contextual retrieval approach:
    1. Run semantic search (vector similarity) independently
    2. Run keyword search (chunk-level FTS5 BM25) independently
    3. Fuse results using Reciprocal Rank Fusion (RRF, k=60)
    4. Optionally re-rank top results with cross-encoder (if enable_reranking=True)

    This provides:
    - Accurate chunk-level BM25 statistics (full corpus, not candidate subset)
    - Score-independent fusion that combines ranking signals
    - Diversity from multiple retrieval strategies
    - Optional cross-encoder re-ranking for improved top-k precision

    Args:
        query: Search query string
        query_embedding: Query embedding vector (768-dim)
        limit: Maximum number of results to return after fusion (and re-ranking if enabled)
        semantic_limit: Number of results from semantic search (default: 50)
        keyword_limit: Number of results from keyword search (default: 50)
        enable_reranking: Enable cross-encoder re-ranking (default: False)
        reranking_model: Cross-encoder model name (default: ms-marco-MiniLM-L-12-v2)
        db_path: Path to database file (uses default if None)

    Returns:
        List of SearchResult objects, fused by RRF (and re-ranked if enabled)

    Example:
        >>> from andamentum.document_store.rag.embeddings import get_embedding
        >>> embedding = get_embedding("machine learning transformers")
        >>> # Without re-ranking (default)
        >>> results = ensemble_search(
        ...     query="machine learning transformers",
        ...     query_embedding=embedding,
        ...     limit=10
        ... )
        >>> # With cross-encoder re-ranking
        >>> results = ensemble_search(
        ...     query="machine learning transformers",
        ...     query_embedding=embedding,
        ...     limit=10,
        ...     enable_reranking=True
        ... )
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Strategy 1: Semantic search (vector similarity)
    # Returns chunks ranked by cosine similarity
    semantic_results = []
    try:
        raw_semantic = search_chunks(
            query_embedding, limit=semantic_limit, db_path=db_path
        )
        for result in raw_semantic:
            # Convert to SearchResult
            similarity = 1.0 - result["distance"]  # cosine distance → similarity

            # Parse metadata
            metadata = result.get("metadata", {})
            para_type = metadata.get("para_type")

            semantic_results.append(
                SearchResult(
                    content=result["content"],
                    file_path=result["file_path"],
                    doc_id=result["doc_id"],
                    similarity_score=similarity,
                    match_type="semantic",
                    start_char=result["start_char"],
                    end_char=result["end_char"],
                    token_count=result.get("token_count"),
                    dc_title=result.get("dc_title"),
                    dc_format=result.get("dc_format"),
                    dc_creator=result.get("dc_creator"),
                    dc_subject=result.get("dc_subject"),
                    para_type=para_type,
                    metadata=metadata,
                )
            )
    except Exception as e:
        print(f"⚠️  Semantic search failed: {e}")

    # Strategy 2: Keyword search (chunk-level FTS5 BM25)
    # Returns chunks ranked by BM25 relevance on full corpus
    keyword_results = []
    try:
        keyword_results = fts_search(query, limit=keyword_limit, db_path=db_path)
    except Exception as e:
        print(f"⚠️  Keyword search failed: {e}")

    # Fusion: Reciprocal Rank Fusion (RRF, k=60)
    # Combines rankings without relying on score calibration
    if not semantic_results and not keyword_results:
        return []
    elif not semantic_results:
        fused_results = keyword_results
    elif not keyword_results:
        fused_results = semantic_results
    else:
        # Both strategies returned results - fuse with RRF
        fused_results = reciprocal_rank_fusion(
            result_lists=[semantic_results, keyword_results],
            k=60,  # Industry standard
        )

    # Optional: Cross-encoder re-ranking
    # Re-rank fused results for improved top-k precision
    if enable_reranking and fused_results:
        from .reranking import rerank_results, RerankingConfig

        # Re-rank with cross-encoder
        # Note: We retrieve limit * 2 from fusion, then re-rank to get best top-k
        candidates = (
            fused_results[: limit * 2] if len(fused_results) > limit else fused_results
        )

        config = RerankingConfig(model_name=reranking_model)
        fused_results = rerank_results(
            query=query, results=candidates, top_k=limit, config=config
        )

    return fused_results[:limit]
