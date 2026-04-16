"""Semantic evidence-provider routing.

Ranks registered evidence providers by embedding cosine similarity between
the research question and each provider's description. Replaces the
deprecated keyword-based ``DOMAIN_PROVIDER_MAP`` in ``routing.py``.

The router reads provider metadata at call time from
``andamentum.epistemic.providers.PROVIDER_REGISTRY`` and
``PROVIDER_DESCRIPTIONS`` — no network, no external config, no runtime
discovery. Providers are known at import time because they are Python
classes inside this package.

Public surface
--------------
- :class:`ProviderScore` — per-provider ranking result (name, score, description).
- :func:`rank_providers` — return all registered providers ranked by similarity.
- :func:`select_providers` — return a top-K shortlist with web_search appended
  as the universal fallback.

Caching
-------
Provider description embeddings are cached at module level keyed by
``(embedding_model, provider_name)``. Descriptions are static during a process
lifetime, so each provider is embedded at most once per embedding model.
Only the query embedding is computed per call.

Usage
-----
.. code-block:: python

    from andamentum.epistemic.provider_routing import select_providers

    providers = await select_providers(
        question="What is the clinical significance of BRCA1 c.5266dupC?",
        embedding_model="embeddinggemma:latest",
    )
    # ['pubmed', 'monarch', 'open_targets', 'web_search']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .embeddings import embed_texts
from .providers import PROVIDER_DESCRIPTIONS, PROVIDER_REGISTRY
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
            provider-description embedding. Range ``[-1.0, 1.0]`` in theory;
            in practice Ollama embeddings are non-negative so ``[0.0, 1.0]``.
        description: The provider description that was embedded.
    """

    name: str
    score: float
    description: str


# Module-level cache of provider description embeddings.
# Key: (embedding_model, provider_name). Value: embedding vector.
_PROVIDER_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}


def _get_provider_catalogue() -> list[tuple[str, str]]:
    """Snapshot the registered providers and their descriptions.

    Reads the current state of ``PROVIDER_REGISTRY`` and
    ``PROVIDER_DESCRIPTIONS``. Providers registered at import time via
    ``register_provider()`` show up automatically. A provider without a
    description is skipped with a warning — an undescribed provider cannot
    be routed to semantically.

    Returns:
        List of ``(provider_name, description)`` pairs sorted by name for
        deterministic ordering.
    """
    catalogue: list[tuple[str, str]] = []
    for name in sorted(PROVIDER_REGISTRY):
        description = PROVIDER_DESCRIPTIONS.get(name, "")
        if not description:
            logger.warning(
                "Provider %r has no description in PROVIDER_DESCRIPTIONS; "
                "excluding from semantic routing",
                name,
            )
            continue
        catalogue.append((name, description))
    return catalogue


async def _embed_providers_cached(
    embedding_model: str,
) -> list[tuple[str, list[float]]]:
    """Return provider embeddings, computing and caching any missing entries.

    Only providers whose ``(embedding_model, name)`` key is not yet in the
    cache are embedded. On subsequent calls with the same model, this is a
    pure cache read — zero network cost.
    """
    catalogue = _get_provider_catalogue()

    missing: list[tuple[str, str]] = []
    for name, description in catalogue:
        if (embedding_model, name) not in _PROVIDER_EMBEDDING_CACHE:
            missing.append((name, description))

    if missing:
        logger.debug(
            "Embedding %d provider descriptions with model %r",
            len(missing),
            embedding_model,
        )
        missing_texts = [desc for _, desc in missing]
        new_embeddings = await embed_texts(missing_texts, model=embedding_model)
        for (name, _), vector in zip(missing, new_embeddings):
            _PROVIDER_EMBEDDING_CACHE[(embedding_model, name)] = vector

    return [
        (name, _PROVIDER_EMBEDDING_CACHE[(embedding_model, name)])
        for name, _ in catalogue
    ]


async def rank_providers(
    question: str,
    *,
    embedding_model: str,
) -> list[ProviderScore]:
    """Rank all registered providers by semantic similarity to ``question``.

    Args:
        question: Research question or clarified query text.
        embedding_model: Ollama embedding model name (e.g.
            ``"embeddinggemma:latest"``). Must match the model used elsewhere
            in the pipeline for consistency.

    Returns:
        All registered providers sorted by descending cosine similarity. The
        caller is responsible for applying any top-K or score-gate policy.

    Raises:
        RuntimeError: If the embedding backend is unreachable. The router
            intentionally does not silently fall back to keyword matching —
            semantic routing is a hard requirement, not a degradable feature.
    """
    catalogue = _get_provider_catalogue()
    if not catalogue:
        logger.warning("No registered providers with descriptions; rank_providers returning empty list")
        return []

    provider_embeddings = await _embed_providers_cached(embedding_model)
    query_embedding = (await embed_texts([question], model=embedding_model))[0]

    description_by_name = dict(catalogue)
    scores: list[ProviderScore] = []
    for name, vector in provider_embeddings:
        score = cosine_similarity(query_embedding, vector)
        scores.append(
            ProviderScore(
                name=name,
                score=score,
                description=description_by_name[name],
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

    1. Rank all registered providers by cosine similarity to ``question``.
    2. Keep providers whose score is at or above ``min_score``.
    3. Take at most the top ``top_k`` of those.
    4. Append ``web_search`` as the universal fallback (always present,
       even when semantic matches pass the gate).
    5. Deduplicate while preserving order.

    If no provider clears the score gate, the result is
    ``["web_search"]`` — a sensible fallback for off-domain questions.

    Args:
        question: Clarified research question.
        embedding_model: Ollama embedding model name. Required. No default.
        top_k: Maximum number of semantically matched providers to include,
            excluding the always-appended ``web_search``.
        min_score: Minimum cosine similarity for a provider to be
            considered a match. Defaults to ``0.15``, calibrated from the
            200-query benchmark: general-academic queries against OpenAlex
            cluster around μ=0.19 σ=0.05 under embeddinggemma, so 0.15 sits
            just below μ−σ and lets general queries clear the gate while
            still filtering genuinely off-topic matches.

    Returns:
        Ordered list of provider names. Always ends with ``"web_search"``.
    """
    ranked = await rank_providers(question, embedding_model=embedding_model)

    above_threshold = [s for s in ranked if s.score >= min_score]
    selected = [s.name for s in above_threshold[:top_k]]

    logger.info(
        "Semantic routing for query %.80r selected %s (top_k=%d, min_score=%.2f, "
        "raw_scores=%s)",
        question,
        selected,
        top_k,
        min_score,
        [f"{s.name}={s.score:.3f}" for s in ranked],
    )

    # Always include web_search as universal fallback, appended last.
    if WEB_SEARCH_FALLBACK not in selected:
        selected.append(WEB_SEARCH_FALLBACK)

    return selected


def _clear_cache() -> None:
    """Clear the provider-embedding cache. Test helper."""
    _PROVIDER_EMBEDDING_CACHE.clear()


__all__ = [
    "ProviderScore",
    "DEFAULT_TOP_K",
    "DEFAULT_MIN_SCORE",
    "WEB_SEARCH_FALLBACK",
    "rank_providers",
    "select_providers",
]
