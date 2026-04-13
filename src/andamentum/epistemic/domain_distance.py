"""Domain distance calculator for the epistemic system.

Calculates domain distance between evidence pairs and clusters evidence by domain.

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import Dict, List, Tuple, Optional
import uuid

from .primitives import (
    DomainClassification,
    DomainCluster,
    DOMAIN_DIMENSION_WEIGHTS,
)


def calculate_dimension_distance(value1: str, value2: str) -> float:
    """Calculate distance for a single dimension.

    Binary distance: 0 if same, 1 if different.
    This captures epistemic independence - different error modes.

    Args:
        value1: First value
        value2: Second value

    Returns:
        Distance: 0.0 (same) or 1.0 (different)
    """
    return 0.0 if value1 == value2 else 1.0


def calculate_domain_distance(
    classification1: DomainClassification,
    classification2: DomainClassification,
) -> float:
    """Calculate domain distance between two evidence classifications.

    Uses weighted sum of dimension distances.

    Distance interpretation:
    - < 0.2: Same domain (not independent)
    - 0.2-0.5: Related domains (partially independent)
    - > 0.5: Different domains (independent)

    Args:
        classification1: First classification
        classification2: Second classification

    Returns:
        Domain distance from 0.0 to 1.0
    """
    vec1 = classification1.dimension_vector
    vec2 = classification2.dimension_vector

    weighted_sum = 0.0
    total_weight = 0.0

    for dimension, weight in DOMAIN_DIMENSION_WEIGHTS.items():
        distance = calculate_dimension_distance(vec1[dimension], vec2[dimension])
        weighted_sum += distance * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5

    return weighted_sum / total_weight


def compute_pairwise_distances(
    classifications: List[DomainClassification],
) -> Dict[Tuple[str, str], float]:
    """Compute distance matrix for all pairs of evidence.

    Args:
        classifications: List of domain classifications

    Returns:
        Dict mapping (evidence_id_1, evidence_id_2) to distance
    """
    distances: Dict[Tuple[str, str], float] = {}

    for i, c1 in enumerate(classifications):
        for c2 in classifications[i + 1 :]:
            distance = calculate_domain_distance(c1, c2)
            # Store both directions for easy lookup
            distances[(c1.evidence_id, c2.evidence_id)] = distance
            distances[(c2.evidence_id, c1.evidence_id)] = distance

    return distances


def cluster_by_domain(
    classifications: List[DomainClassification],
    distance_threshold: float = 0.3,
) -> List[DomainCluster]:
    """Cluster evidence into domain groups.

    Simple single-linkage clustering: evidence items with distance < threshold
    are in the same cluster.

    Args:
        classifications: List of domain classifications
        distance_threshold: Maximum distance for same cluster (default 0.3)

    Returns:
        List of DomainCluster objects
    """
    if not classifications:
        return []

    # Compute pairwise distances
    distances = compute_pairwise_distances(classifications)

    # Initialize each evidence as its own cluster
    evidence_to_cluster: Dict[str, int] = {}
    clusters: Dict[int, List[str]] = {}

    for i, c in enumerate(classifications):
        evidence_to_cluster[c.evidence_id] = i
        clusters[i] = [c.evidence_id]

    # Merge clusters when distance < threshold
    for (e1, e2), distance in distances.items():
        if distance < distance_threshold:
            c1 = evidence_to_cluster[e1]
            c2 = evidence_to_cluster[e2]

            if c1 != c2:
                # Merge smaller cluster into larger
                if len(clusters[c1]) >= len(clusters[c2]):
                    for e in clusters[c2]:
                        clusters[c1].append(e)
                        evidence_to_cluster[e] = c1
                    del clusters[c2]
                else:
                    for e in clusters[c1]:
                        clusters[c2].append(e)
                        evidence_to_cluster[e] = c2
                    del clusters[c1]

    # Build DomainCluster objects
    classification_map = {c.evidence_id: c for c in classifications}
    result = []

    for cluster_id, evidence_ids in clusters.items():
        # Get representative classification (first in cluster)
        representative = classification_map.get(evidence_ids[0])

        # Generate cluster label
        if representative:
            from .domain_classifier import get_domain_label

            label = get_domain_label(representative)
        else:
            label = f"cluster_{cluster_id}"

        result.append(
            DomainCluster(
                cluster_id=str(uuid.uuid4()),
                evidence_ids=list(set(evidence_ids)),  # Deduplicate
                representative_classification=representative,
                cluster_size=len(set(evidence_ids)),
                average_evidence_quality=0.0,  # Will be set by caller
                cluster_label=label,
            )
        )

    return result


def calculate_inter_cluster_distances(
    clusters: List[DomainCluster],
) -> Tuple[float, float]:
    """Calculate average and minimum distance between clusters.

    Args:
        clusters: List of domain clusters

    Returns:
        Tuple of (average_distance, min_distance)
    """
    if len(clusters) < 2:
        return 0.0, 0.0

    distances = []
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i + 1 :]:
            if c1.representative_classification and c2.representative_classification:
                d = calculate_domain_distance(
                    c1.representative_classification,
                    c2.representative_classification,
                )
                distances.append(d)

    if not distances:
        return 0.0, 0.0

    return sum(distances) / len(distances), min(distances)


def find_most_distant_clusters(
    clusters: List[DomainCluster],
    top_n: int = 3,
) -> List[Tuple[DomainCluster, DomainCluster, float]]:
    """Find the most distant cluster pairs.

    Useful for identifying the most independent evidence streams.

    Args:
        clusters: List of domain clusters
        top_n: Number of top pairs to return

    Returns:
        List of (cluster1, cluster2, distance) tuples
    """
    if len(clusters) < 2:
        return []

    pairs = []
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i + 1 :]:
            if c1.representative_classification and c2.representative_classification:
                d = calculate_domain_distance(
                    c1.representative_classification,
                    c2.representative_classification,
                )
                pairs.append((c1, c2, d))

    # Sort by distance descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def interpret_distance(distance: float) -> str:
    """Interpret a domain distance value.

    Args:
        distance: Domain distance from 0.0 to 1.0

    Returns:
        Human-readable interpretation
    """
    if distance < 0.2:
        return "same domain (not independent)"
    elif distance < 0.35:
        return "closely related domains (limited independence)"
    elif distance < 0.5:
        return "related domains (partial independence)"
    elif distance < 0.7:
        return "different domains (meaningful independence)"
    else:
        return "very different domains (strong independence)"
