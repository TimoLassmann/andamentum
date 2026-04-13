"""Evidence deduplication via semantic clustering.

Standalone module: takes document texts, clusters by semantic similarity
using HDBSCAN, returns clusters with representative documents.

No epistemic domain knowledge. No thresholds to tune.
HDBSCAN discovers cluster count from data structure.

Architecture: Layer 1 (framework-agnostic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances

from .embeddings import embed_documents

logger = logging.getLogger(__name__)


@dataclass
class EvidenceCluster:
    """A cluster of semantically similar evidence documents.

    Attributes:
        medoid_index: Index of the most central document (what the cluster is about)
        representative_indices: Medoid, plus best-quality member if different (added by operations layer)
        member_indices: All document indices in this cluster
        count: Total documents in cluster (= corroboration count)
    """

    medoid_index: int
    representative_indices: list[int] = field(default_factory=list)
    member_indices: list[int] = field(default_factory=list)
    count: int = 1


async def deduplicate_evidence(
    texts: list[str],
    min_cluster_size: int = 2,
    *,
    embedding_model: str,
) -> list[EvidenceCluster]:
    """Cluster documents by semantic similarity using HDBSCAN.

    Args:
        texts: Document texts to cluster
        min_cluster_size: Minimum documents to form a cluster (HDBSCAN param).
            Default 2 means even two similar documents cluster together.

    Returns:
        List of EvidenceCluster, one per distinct finding plus singletons.
        Every input index appears in exactly one cluster.
    """
    n = len(texts)
    if n == 0:
        return []
    if n == 1:
        return [
            EvidenceCluster(
                medoid_index=0, representative_indices=[0], member_indices=[0], count=1
            )
        ]

    # Chunk each document and embed all chunks via Ollama.
    # Long documents (e.g. raw web pages) are split into ~2000-char chunks
    # so that each chunk fits within the embedding model's context window.
    doc_embeddings = await embed_documents(texts, model=embedding_model)

    # Build max-sim distance matrix: for each pair of documents, the distance
    # is 1 - max(cosine_similarity(chunk_a, chunk_b)) over all chunk pairs.
    # This detects shared content even when documents are mostly different.
    all_chunk_embeddings = [np.array(chunks) for chunks in doc_embeddings]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # cosine_distances returns a matrix; we want the min distance
            # (= max similarity) between any chunk pair
            chunk_dists = cosine_distances(
                all_chunk_embeddings[i], all_chunk_embeddings[j]
            )
            min_dist = float(chunk_dists.min())
            dist_matrix[i, j] = min_dist
            dist_matrix[j, i] = min_dist

    # Mean-pooled embedding per document for medoid selection within clusters
    emb_matrix = np.array([np.mean(chunks, axis=0) for chunks in all_chunk_embeddings])

    # Cluster with HDBSCAN — no epsilon, no k, discovers structure from data
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="precomputed",
        copy=True,  # type: ignore[arg-type]
    )
    labels = clusterer.fit_predict(dist_matrix)

    # Build clusters from labels
    # HDBSCAN labels: -1 = noise (no cluster), 0+ = cluster ID
    cluster_map: dict[int, list[int]] = {}
    noise_indices: list[int] = []

    for idx, label in enumerate(labels):
        if label == -1:
            noise_indices.append(idx)
        else:
            cluster_map.setdefault(int(label), []).append(idx)

    result: list[EvidenceCluster] = []

    # Process real clusters
    for cluster_label, member_indices in sorted(cluster_map.items()):
        cluster = _build_cluster(emb_matrix, member_indices)
        result.append(cluster)

    # Noise points become singleton clusters (truly unique evidence)
    for idx in noise_indices:
        result.append(
            EvidenceCluster(
                medoid_index=idx,
                representative_indices=[idx],
                member_indices=[idx],
                count=1,
            )
        )

    return result


def _build_cluster(
    emb_matrix: np.ndarray, member_indices: list[int]
) -> EvidenceCluster:
    """Build an EvidenceCluster with medoid as the sole representative.

    The medoid (most central document) captures what the cluster is about.
    The best-quality member is added later by select_top_k_evidence() in
    operations.py — keeping quality logic out of the geometry module.

    Args:
        emb_matrix: Full embedding matrix (N x dim)
        member_indices: Indices of documents in this cluster

    Returns:
        EvidenceCluster with medoid as representative
    """
    if len(member_indices) == 1:
        idx = member_indices[0]
        return EvidenceCluster(
            medoid_index=idx,
            representative_indices=[idx],
            member_indices=member_indices,
            count=1,
        )

    # Compute cluster centroid
    members = np.array(member_indices)
    cluster_embeddings = emb_matrix[members]
    centroid = cluster_embeddings.mean(axis=0)

    # Find medoid (closest to centroid)
    distances_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    medoid_local = int(np.argmin(distances_to_centroid))
    medoid_index = member_indices[medoid_local]

    return EvidenceCluster(
        medoid_index=medoid_index,
        representative_indices=[medoid_index],
        member_indices=member_indices,
        count=len(member_indices),
    )
