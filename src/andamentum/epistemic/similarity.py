"""Shared similarity utility for embedding, comparing, and grouping text items.

Provides embed_and_group() for deterministic threshold-based clustering,
validate_groups() for LLM-assisted refinement of large clusters,
assess_clustering() for diagnostic quality metrics, and medoid() for
representative selection.

Used by assertion clustering, uncertainty grouping, and caveat dedup.
Evidence dedup uses HDBSCAN separately (see dedup.py).

Architecture: Layer 1 (framework-agnostic, async)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class AgentRunner(Protocol):
    """Minimal protocol for running epistemic agents."""

    async def run(self, agent_name: str, **kwargs: Any) -> Any: ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors.

    Returns 0.0 if either vector has zero magnitude.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def group_by_similarity(
    embeddings: list[list[float]],
    threshold: float,
) -> list[list[int]]:
    """Group embeddings by cosine similarity using single-linkage clustering.

    Single-linkage means items form transitive groups: if sim(A,B) >= threshold
    and sim(B,C) >= threshold, then A, B, and C are all in the same group —
    even if sim(A,C) < threshold.

    Implemented via Union-Find for correctness and clarity.

    Args:
        embeddings: List of embedding vectors (one per text item).
        threshold: Cosine similarity threshold. Pairs at or above this
            are linked together.

    Returns:
        List of groups, each a list of indices into the input.
        Every index appears in exactly one group.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Link all pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                union(i, j)

    # Extract groups
    groups_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups_map.setdefault(root, []).append(i)

    return list(groups_map.values())


@dataclass
class GroupQuality:
    """Diagnostic quality for a single cluster."""

    group_id: int
    size: int
    mean_silhouette: float
    mean_intra_sim: float  # Mean pairwise cosine similarity within this group.


@dataclass
class ClusteringQuality:
    """Diagnostic quality report for a clustering result.

    Returned by assess_clustering(). Useful for logging and monitoring
    but not for automatic correction — the LLM validation pass
    (validate_groups) handles that.
    """

    silhouette: float  # Mean silhouette score (-1 to +1). Higher = better.
    interpretation: str  # "strong", "reasonable", "weak", or "no_structure"
    groups: list[GroupQuality] = field(default_factory=list)
    computable: bool = True  # False when metrics can't be computed.


def assess_clustering(
    embeddings: list[list[float]],
    groups: list[list[int]],
) -> ClusteringQuality:
    """Assess clustering quality using silhouette analysis (diagnostic only).

    Provides overall and per-group quality scores. Useful for logging
    and monitoring — does not modify groups. The LLM validation pass
    (validate_groups) is the mechanism for correcting bad groupings.

    Args:
        embeddings: Embedding vectors (same as passed to group_by_similarity).
        groups: Groups from group_by_similarity (each a list of indices).

    Returns:
        ClusteringQuality with overall silhouette and per-group breakdowns.
    """
    import numpy as np
    from sklearn.metrics import silhouette_score, silhouette_samples

    n = len(embeddings)
    n_clusters = len(groups)

    # Edge cases: metrics require 2 <= n_clusters <= n_samples - 1
    if n < 2 or n_clusters < 2 or n_clusters >= n:
        return ClusteringQuality(
            silhouette=0.0,
            interpretation="not_applicable",
            computable=False,
        )

    X = np.array(embeddings)

    # Build per-item label array from groups
    labels = np.full(n, -1, dtype=int)
    for group_id, group in enumerate(groups):
        for idx in group:
            labels[idx] = group_id

    sil_mean = float(silhouette_score(X, labels, metric="cosine"))
    sil_per_item = silhouette_samples(X, labels, metric="cosine")

    if sil_mean >= 0.7:
        interpretation = "strong"
    elif sil_mean >= 0.5:
        interpretation = "reasonable"
    elif sil_mean >= 0.25:
        interpretation = "weak"
    else:
        interpretation = "no_structure"

    # Per-group diagnostics
    group_qualities: list[GroupQuality] = []
    for group_id, group in enumerate(groups):
        group_sils = [float(sil_per_item[i]) for i in group]

        if len(group) > 1:
            intra_sims = [
                cosine_similarity(embeddings[group[a]], embeddings[group[b]])
                for a in range(len(group))
                for b in range(a + 1, len(group))
            ]
            mean_intra = sum(intra_sims) / len(intra_sims)
        else:
            mean_intra = 1.0

        group_qualities.append(GroupQuality(
            group_id=group_id,
            size=len(group),
            mean_silhouette=float(np.mean(group_sils)),
            mean_intra_sim=mean_intra,
        ))

    return ClusteringQuality(
        silhouette=sil_mean,
        interpretation=interpretation,
        groups=group_qualities,
    )


def medoid(
    embeddings: list[list[float]],
    group: list[int],
) -> int:
    """Find the medoid (most central item) of a group.

    The medoid is the item with the highest mean cosine similarity
    to all other items in the group. For a singleton, returns that item.

    Args:
        embeddings: Full embedding list (not just group members).
        group: Indices of items in this group.

    Returns:
        Index of the medoid item.
    """
    if len(group) == 1:
        return group[0]

    best_idx = group[0]
    best_mean_sim = -2.0

    for i in group:
        sims = [cosine_similarity(embeddings[i], embeddings[j]) for j in group if j != i]
        mean_sim = sum(sims) / len(sims)
        if mean_sim > best_mean_sim:
            best_mean_sim = mean_sim
            best_idx = i

    return best_idx


async def embed_and_group(
    texts: list[str],
    threshold: float,
    *,
    embedding_model: str,
) -> list[list[int]]:
    """Embed texts and group by cosine similarity.

    Thin wrapper: embeds via Ollama, then groups with single-linkage.

    Args:
        texts: Text strings to embed and group.
        threshold: Cosine similarity threshold for grouping.

    Returns:
        List of groups, each a list of indices into the input texts.

    Raises:
        RuntimeError: If the embedding model is unavailable.
    """
    from .embeddings import embed_texts

    if len(texts) == 0:
        return []
    if len(texts) == 1:
        return [[0]]

    embeddings = await embed_texts(texts, model=embedding_model)
    return group_by_similarity(embeddings, threshold)


async def validate_groups(
    texts: list[str],
    groups: list[list[int]],
    runner: AgentRunner,
    min_group_size: int = 3,
) -> list[list[int]]:
    """Validate groups using LLM, splitting incorrectly merged items.

    For each group with >= min_group_size members, asks an LLM whether
    all items truly express the same idea. Groups that contain distinct
    sub-themes are split into separate groups.

    Groups smaller than min_group_size pass through unchanged.

    Args:
        texts: Original text strings (same list passed to embed_and_group).
        groups: Groups from embed_and_group (each a list of indices).
        runner: Agent runner implementing async run(agent_name, **kwargs).
        min_group_size: Minimum group size to trigger LLM validation.

    Returns:
        Refined list of groups. May contain more groups than input
        if the LLM splits any clusters.

    Raises:
        RuntimeError: If the LLM is unavailable.
    """
    refined: list[list[int]] = []

    for group in groups:
        if len(group) < min_group_size:
            refined.append(group)
            continue

        # Build numbered item list for LLM
        group_texts = [texts[i] for i in group]
        items_formatted = "\n".join(
            f"Item {j + 1}: {text}" for j, text in enumerate(group_texts)
        )

        from .agents.output_models import ValidateGroupOutput

        result: ValidateGroupOutput = await runner.run(
            "epistemic_validate_group",
            items=items_formatted,
            item_count=str(len(group)),
        )

        # Map 1-based LLM item numbers back to original indices
        for subgroup_numbers in result.subgroups:
            original_indices = []
            for item_num in subgroup_numbers:
                idx_in_group = item_num - 1  # 1-based → 0-based within group
                if 0 <= idx_in_group < len(group):
                    original_indices.append(group[idx_in_group])
            if original_indices:
                refined.append(original_indices)

    return refined
