"""Cross-encoder re-ranking for search results.

Framework-agnostic module for re-ranking search results using cross-encoder models.

Cross-encoders encode query and document together, providing more accurate relevance
scores than bi-encoders at the cost of higher computational overhead. Best used for
re-ranking top-k results after initial retrieval.

Usage:
    from andamentum.document_store.rag.reranking import rerank_results
    from andamentum.document_store.rag.search import SearchResult

    results = [...]  # Initial search results
    reranked = rerank_results(
        query="machine learning transformers",
        results=results,
        top_k=10
    )
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RerankingConfig:
    """Configuration for cross-encoder re-ranking."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Fast, accurate model for ranking
    batch_size: int = 32  # Batch size for scoring (adjust based on memory)
    max_length: int = 512  # Maximum sequence length (query + document)
    device: Optional[str] = None  # Device for inference (None = auto-detect)


# Global model instance for lazy loading
_model_instance: Optional[object] = None
_current_model_name: Optional[str] = None


def _get_model(config: RerankingConfig):
    """Get or load cross-encoder model (lazy loading).

    Args:
        config: Reranking configuration

    Returns:
        Loaded CrossEncoder model instance
    """
    global _model_instance, _current_model_name

    # Lazy load model on first use
    if _model_instance is None or _current_model_name != config.model_name:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for re-ranking. "
                "Install with: pip install sentence-transformers"
            )

        # Load model
        _model_instance = CrossEncoder(
            config.model_name, max_length=config.max_length, device=config.device
        )
        _current_model_name = config.model_name

    return _model_instance


def score_pairs(query: str, documents: List[str], config: Optional[RerankingConfig] = None) -> List[float]:
    """Score query-document pairs using cross-encoder.

    Args:
        query: Query string
        documents: List of document strings to score
        config: Reranking configuration (uses defaults if None)

    Returns:
        List of relevance scores (higher = more relevant)

    Example:
        >>> scores = score_pairs(
        ...     query="machine learning",
        ...     documents=["ML is a subset of AI", "Python is a programming language"]
        ... )
        >>> scores[0] > scores[1]  # First document more relevant
        True
    """
    if config is None:
        config = RerankingConfig()

    if not documents:
        return []

    # Get model
    model = _get_model(config)

    # Prepare pairs for scoring
    pairs = [[query, doc] for doc in documents]

    # Score in batches for efficiency
    scores = model.predict(pairs, batch_size=config.batch_size, show_progress_bar=False)  # type: ignore

    # Convert to list of floats
    return scores.tolist()  # type: ignore


def rerank_results(
    query: str, results: List, top_k: Optional[int] = None, config: Optional[RerankingConfig] = None
) -> List:
    """Re-rank search results using cross-encoder model.

    Takes initial search results (from ensemble_search or other retrieval methods)
    and re-ranks them using a cross-encoder model for improved relevance.

    Args:
        query: Search query string
        results: List of SearchResult objects to re-rank
        top_k: Number of top results to return (None = return all, re-ranked)
        config: Reranking configuration (uses defaults if None)

    Returns:
        Re-ranked list of SearchResult objects (top_k results if specified)

    Example:
        >>> from andamentum.document_store.rag.search import ensemble_search
        >>> results = ensemble_search("machine learning", embedding, limit=50)
        >>> reranked = rerank_results("machine learning", results, top_k=10)
        >>> # Top 10 results are now re-ranked by cross-encoder
    """
    if config is None:
        config = RerankingConfig()

    if not results:
        return []

    # Extract documents (content) from results
    documents = [r.content for r in results]

    # Score all pairs
    scores = score_pairs(query, documents, config)

    # Pair results with scores
    scored_results = list(zip(results, scores))

    # Sort by score (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Take top_k if specified
    if top_k is not None:
        scored_results = scored_results[:top_k]

    # Extract just the results (drop scores)
    reranked_results = [r for r, _ in scored_results]

    return reranked_results


def rerank_with_scores(
    query: str, results: List, config: Optional[RerankingConfig] = None
) -> List[Tuple[object, float]]:
    """Re-rank results and return with cross-encoder scores.

    Useful for debugging or when you need to see the actual scores.

    Args:
        query: Search query string
        results: List of SearchResult objects to re-rank
        config: Reranking configuration (uses defaults if None)

    Returns:
        List of (SearchResult, score) tuples, sorted by score (descending)

    Example:
        >>> reranked = rerank_with_scores("machine learning", results)
        >>> for result, score in reranked[:5]:
        ...     print(f"{score:.3f}: {result.content[:50]}")
    """
    if config is None:
        config = RerankingConfig()

    if not results:
        return []

    # Extract documents
    documents = [r.content for r in results]

    # Score all pairs
    scores = score_pairs(query, documents, config)

    # Pair results with scores
    scored_results = list(zip(results, scores))

    # Sort by score (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)

    return scored_results


__all__ = ["RerankingConfig", "score_pairs", "rerank_results", "rerank_with_scores"]
