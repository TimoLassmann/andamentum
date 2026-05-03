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

from rank_bm25 import BM25Okapi

from .database import search_chunks

RRF_K = 60  # Reciprocal Rank Fusion smoothing constant; industry standard


@dataclass
class SearchConfig:
    """Configuration for chunk-level semantic + BM25 hybrid search."""

    include_bm25: bool = True  # Enable BM25 keyword scoring
    bm25_weight: float = 0.5  # 50% keyword, 50% vector (balanced hybrid)
    min_similarity: float = 0.0  # Minimum similarity threshold (0-1)


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
    result_lists: List[List[SearchResult]], k: int = RRF_K
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

    return search_results
