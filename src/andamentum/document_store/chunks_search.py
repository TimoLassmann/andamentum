"""Chunk-level semantic + BM25 hybrid search and RRF fusion primitives.

Used by the top-level four-signal search (see ``search.py``) and the
multi-strategy router (``hybrid_search.py``). All functions here operate
on the chunks + chunk_embeddings tables (see ``chunks.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from .chunks import search_chunks

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
    """Normalize scores to 0-1 range using min-max normalization."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]], k: int = RRF_K
) -> List[SearchResult]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Score-independent fusion: score = 1 / (rank + k). Industry standard k=60.
    Deduplicates at chunk level using (doc_id, start_char) — preserves
    chunk-level diversity within the same document.

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
    """Vector similarity search with optional BM25 keyword fusion.

    Vector similarity comes from sqlite-vec cosine distance; BM25 from
    ``rank_bm25``. When BM25 is enabled the two are min-max normalised and
    combined with ``config.bm25_weight``.

    Args:
        query: Search query string (used for BM25)
        query_embedding: Query embedding vector (768-dim)
        limit: Maximum number of results
        config: Optional search configuration
        db_path: Path to database file (uses default if None)

    Returns:
        List of SearchResult objects, sorted by relevance
    """
    if config is None:
        config = SearchConfig()

    # Over-fetch 3x when BM25 will re-score; otherwise fetch exactly limit.
    vector_limit = limit * 3 if config.include_bm25 else limit

    vector_results = search_chunks(
        query_embedding=query_embedding, limit=vector_limit, db_path=db_path
    )

    if not vector_results:
        return []

    if config.include_bm25 and len(vector_results) > 0:
        corpus = [result["content"].lower().split() for result in vector_results]
        bm25 = BM25Okapi(corpus)

        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        vector_scores = [1 - result["distance"] for result in vector_results]
        vector_normalized = normalize_scores(vector_scores)
        bm25_normalized = normalize_scores(list(bm25_scores))

        for i, result in enumerate(vector_results):
            combined_score = (1 - config.bm25_weight) * vector_normalized[
                i
            ] + config.bm25_weight * bm25_normalized[i]
            result["similarity_score"] = combined_score
            result["match_type"] = "hybrid"

        vector_results.sort(key=lambda x: x["similarity_score"], reverse=True)

    else:
        for result in vector_results:
            result["similarity_score"] = 1 - result["distance"]
            result["match_type"] = "semantic"

    filtered_results = [
        r for r in vector_results if r["similarity_score"] >= config.min_similarity
    ]

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
