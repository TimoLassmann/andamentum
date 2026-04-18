"""Semantic evidence-provider routing via example-query matching.

Ranks registered evidence providers by comparing the research question
against each provider's EXAMPLE QUERIES using embedding cosine similarity.
Each provider is scored by its best-matching example (max-sim), which
solves the short-vs-long embedding mismatch that occurs when comparing
a terse 10-word claim against a 200-word provider description.

Public surface
--------------
- :class:`ProviderScore` — per-provider ranking result (name, score, matched example).
- :func:`rank_providers` — return all registered providers ranked by similarity.
- :func:`select_providers` — return a top-K shortlist with web_search appended.

Caching
-------
Example-query embeddings are cached at module level keyed by
``(embedding_model, provider_name, example_index)``. Examples are static
during a process lifetime, so each is embedded at most once per model.
Only the query embedding is computed per call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .embeddings import embed_texts
from .providers import PROVIDER_DESCRIPTIONS, PROVIDER_EXAMPLES, PROVIDER_REGISTRY
from .similarity import cosine_similarity

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.15
WEB_SEARCH_FALLBACK = "web_search"


@dataclass(frozen=True)
class ProviderScore:
    """One provider's ranking result.

    Attributes:
        name: Provider identifier (matches keys in ``PROVIDER_REGISTRY``).
        score: Cosine similarity between the query embedding and the
            best-matching example query for this provider.
        description: The provider description (for display/logging).
        matched_example: The specific example query that produced the
            best match (for debugging/transparency).
    """

    name: str
    score: float
    description: str
    matched_example: str = ""


# Module-level cache of example-query embeddings.
# Key: (embedding_model, provider_name, example_index). Value: embedding vector.
_EXAMPLE_EMBEDDING_CACHE: dict[tuple[str, str, int], list[float]] = {}


def _get_provider_examples() -> dict[str, list[str]]:
    """Return example queries for each registered provider.

    Providers without examples or without a registry entry are skipped.
    """
    result: dict[str, list[str]] = {}
    for name in sorted(PROVIDER_REGISTRY):
        examples = PROVIDER_EXAMPLES.get(name, [])
        if not examples:
            logger.warning(
                "Provider %r has no example queries in PROVIDER_EXAMPLES; "
                "excluding from semantic routing",
                name,
            )
            continue
        result[name] = examples
    return result


async def _embed_examples_cached(
    embedding_model: str,
) -> dict[str, list[list[float]]]:
    """Return per-provider example embeddings, computing and caching missing ones.

    Returns a dict mapping provider name to a list of embedding vectors,
    one per example query.
    """
    provider_examples = _get_provider_examples()

    # Collect all missing (provider, index) pairs
    missing: list[tuple[str, int, str]] = []  # (provider, index, text)
    for name, examples in provider_examples.items():
        for i, example in enumerate(examples):
            if (embedding_model, name, i) not in _EXAMPLE_EMBEDDING_CACHE:
                missing.append((name, i, example))

    if missing:
        logger.debug(
            "Embedding %d provider example queries with model %r",
            len(missing),
            embedding_model,
        )
        missing_texts = [text for _, _, text in missing]
        new_embeddings = await embed_texts(missing_texts, model=embedding_model)
        for (name, idx, _), vector in zip(missing, new_embeddings):
            _EXAMPLE_EMBEDDING_CACHE[(embedding_model, name, idx)] = vector

    # Assemble result
    result: dict[str, list[list[float]]] = {}
    for name, examples in provider_examples.items():
        result[name] = [
            _EXAMPLE_EMBEDDING_CACHE[(embedding_model, name, i)]
            for i in range(len(examples))
        ]
    return result


async def rank_providers(
    question: str,
    *,
    embedding_model: str,
) -> list[ProviderScore]:
    """Rank all registered providers by semantic similarity to ``question``.

    For each provider, computes cosine similarity between the question
    and EACH of the provider's example queries, then scores the provider
    by its best match (max-sim). This handles short inputs well because
    example queries are at the same granularity as typical user inputs.

    Args:
        question: Research question or claim text.
        embedding_model: Ollama embedding model name.

    Returns:
        All registered providers sorted by descending best-match score.
    """
    provider_examples = _get_provider_examples()
    if not provider_examples:
        logger.warning("No providers with example queries; rank_providers returning empty list")
        return []

    example_embeddings = await _embed_examples_cached(embedding_model)
    query_embedding = (await embed_texts([question], model=embedding_model))[0]

    scores: list[ProviderScore] = []
    for name, example_vecs in example_embeddings.items():
        examples = provider_examples[name]
        description = PROVIDER_DESCRIPTIONS.get(name, "")

        # Score by best-matching example (max-sim)
        best_score = -1.0
        best_example = ""
        for example_text, example_vec in zip(examples, example_vecs):
            sim = cosine_similarity(query_embedding, example_vec)
            if sim > best_score:
                best_score = sim
                best_example = example_text

        scores.append(
            ProviderScore(
                name=name,
                score=best_score,
                description=description,
                matched_example=best_example,
            )
        )

    scores.sort(key=lambda s: s.score, reverse=True)
    return scores


async def select_providers(
    question: str,
    *,
    embedding_model: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[str]:
    """Return a top-K provider shortlist for an evidence-gathering plan.

    Semantics:

    1. Rank all registered providers by max-sim against example queries.
    2. Keep providers whose score is at or above ``min_score``.
    3. Take at most the top ``top_k`` of those.
    4. Append ``web_search`` as the universal fallback.
    5. Deduplicate while preserving order.

    Args:
        question: Clarified research question or claim text.
        embedding_model: Ollama embedding model name.
        top_k: Maximum semantic matches (excluding web_search fallback).
        min_score: Minimum cosine similarity for a provider to be selected.
            Defaults to ``0.15``, calibrated from the 200-query benchmark.

    Returns:
        Ordered list of provider names. Always ends with ``"web_search"``.
    """
    ranked = await rank_providers(question, embedding_model=embedding_model)

    above_threshold = [s for s in ranked if s.score >= min_score]
    selected = [s.name for s in above_threshold[:top_k]]

    logger.info(
        "Semantic routing for query %.80r selected %s (top_k=%d, min_score=%.2f, "
        "best_matches=%s)",
        question,
        selected,
        top_k,
        min_score,
        [f"{s.name}={s.score:.3f}({s.matched_example[:40]})" for s in ranked],
    )

    if WEB_SEARCH_FALLBACK not in selected:
        selected.append(WEB_SEARCH_FALLBACK)

    return selected


def _clear_cache() -> None:
    """Clear the example-embedding cache. Test helper."""
    _EXAMPLE_EMBEDDING_CACHE.clear()


__all__ = [
    "ProviderScore",
    "DEFAULT_TOP_K",
    "DEFAULT_MIN_SCORE",
    "WEB_SEARCH_FALLBACK",
    "rank_providers",
    "select_providers",
]
