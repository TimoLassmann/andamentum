"""Convergence detector for the epistemic system.

Detects cross-domain convergence - when evidence from epistemically independent
domains (different error modes) converges on the same conclusion.

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import uuid

from .primitives import (
    DomainClassification,
    DomainCluster,
    ConvergentEvidence,
    DOMAIN_DIMENSION_WEIGHTS,
)

from .domain_classifier import (
    classify_evidence_domain,
    get_domain_label,
)

from .domain_distance import (
    cluster_by_domain,
    calculate_inter_cluster_distances,
)
from .thresholds import (
    CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW,
    CONVERGENCE_INTRA_DIVERSITY_THRESHOLD,
    CONVERGENCE_STRONG_THRESHOLD,
)


# Thresholds for convergence detection
CONVERGENCE_THRESHOLDS = {
    "min_independent_domains": 2,  # Need at least 2 independent domains
    "min_inter_domain_distance": 0.3,  # Minimum distance for independence
    "strong_convergence_threshold": 3,  # 3+ domains = strong convergence
    "cluster_distance_threshold": 0.3,  # Max distance for same cluster
}


def detect_convergence(
    evidence_items: List[Dict[str, Any]],
    claim_id: str,
    objective_id: str,
    evidence_qualities: Optional[Dict[str, float]] = None,
    precomputed_classifications: Optional[List[DomainClassification]] = None,
    pairwise_independence: Optional[Dict[Tuple[str, str], bool]] = None,
) -> ConvergentEvidence:
    """Detect cross-domain convergence in evidence.

    This is the main entry point for convergence detection.

    Args:
        evidence_items: List of evidence dicts with at least 'evidence_id' and 'content'
        claim_id: ID of the claim this evidence supports
        objective_id: ID of the objective
        evidence_qualities: Optional dict of evidence_id -> quality score (0-1)
        precomputed_classifications: Optional list of DomainClassification from the LLM
            agent (epistemic_classify_evidence_domain). When provided, these replace
            the deterministic classify_evidence_domain heuristic. Must have exactly one
            entry per evidence item in the same order.
        pairwise_independence: Optional dict mapping (eid_a, eid_b) -> bool from the
            epistemic_check_pairwise_independence agent. When provided, within-cluster
            pairs judged independent signal hidden methodological diversity and boost
            the independence score.

    Returns:
        ConvergentEvidence with full convergence analysis
    """
    if not evidence_items:
        return _empty_convergence(claim_id, objective_id)

    # Step 1: Use precomputed classifications if provided, else classify internally.
    # Precomputed classifications come from the LLM agent (epistemic_classify_evidence_domain),
    # which is more accurate than the deterministic classify_evidence_domain heuristic.
    if precomputed_classifications is not None:
        # Validate alignment: one classification per evidence item, in order
        if len(precomputed_classifications) != len(evidence_items):
            raise ValueError(
                f"precomputed_classifications length ({len(precomputed_classifications)}) "
                f"does not match evidence_items length ({len(evidence_items)})"
            )
        classifications = list(precomputed_classifications)
    else:
        classifications = []
        for item in evidence_items:
            evidence_id = item.get("evidence_id", str(uuid.uuid4()))
            text = (
                item.get("content", "")
                or item.get("text", "")
                or item.get("summary", "")
            )
            metadata = {
                k: v
                for k, v in item.items()
                if k not in ["evidence_id", "content", "text"]
            }
            classification = classify_evidence_domain(
                evidence_id=evidence_id,
                claim_id=claim_id,
                evidence_text=text,
                evidence_metadata=metadata,
            )
            classifications.append(classification)

    # Step 2: Cluster by domain
    clusters = cluster_by_domain(
        classifications,
        distance_threshold=CONVERGENCE_THRESHOLDS["cluster_distance_threshold"],
    )

    # Step 3: Set quality scores on clusters
    if evidence_qualities:
        for cluster in clusters:
            qualities = [
                evidence_qualities.get(eid, 0.5) for eid in cluster.evidence_ids
            ]
            cluster.average_evidence_quality = (
                sum(qualities) / len(qualities) if qualities else 0.5
            )

    # Step 4: Calculate inter-cluster distances
    avg_distance, min_distance = calculate_inter_cluster_distances(clusters)

    # Step 5: Run independence checks (now augmented by agent's pairwise judgments)
    independence_checks = _check_independence(
        clusters, min_distance, pairwise_independence
    )

    # Step 6: Calculate independence score
    independence_score = _calculate_independence_score(
        clusters, avg_distance, independence_checks
    )

    # Step 7: Detect convergence
    convergence_detected, strength = _detect_convergence_signal(
        clusters, independence_score, independence_checks
    )

    # Step 8: Determine verdict
    verdict = _determine_verdict(
        clusters, convergence_detected, strength, independence_checks
    )

    # Step 9: Generate explanation
    explanation = _generate_explanation(
        clusters, verdict, independence_score, convergence_detected
    )

    # Step 10: Find missing domains and strongest per domain
    missing_domains = _find_missing_domains(classifications)
    strongest_per_domain = _find_strongest_per_domain(
        classifications, evidence_qualities
    )

    return ConvergentEvidence(
        evidence_id=str(uuid.uuid4()),
        claim_id=claim_id,
        objective_id=objective_id,
        evidence_classifications=classifications,
        total_evidence_count=len(evidence_items),
        domain_clusters=clusters,
        num_independent_domains=len(clusters),
        average_inter_domain_distance=avg_distance,
        min_inter_domain_distance=min_distance,
        independence_checks=independence_checks,
        independence_score=independence_score,
        convergence_detected=convergence_detected,
        convergence_strength=strength,
        convergence_justification=_generate_justification(
            clusters, independence_checks
        ),
        verdict=verdict,
        confidence=_calculate_confidence(independence_score, len(clusters)),
        explanation=explanation,
        missing_domains=missing_domains,
        strongest_per_domain=strongest_per_domain,
        created_at=datetime.utcnow(),
    )


def _empty_convergence(claim_id: str, objective_id: str) -> ConvergentEvidence:
    """Create empty convergence result for no evidence."""
    return ConvergentEvidence(
        evidence_id=str(uuid.uuid4()),
        claim_id=claim_id,
        objective_id=objective_id,
        evidence_classifications=[],
        total_evidence_count=0,
        domain_clusters=[],
        num_independent_domains=0,
        average_inter_domain_distance=0.0,
        min_inter_domain_distance=0.0,
        independence_checks={},
        independence_score=0.0,
        convergence_detected=False,
        convergence_strength=0.0,
        convergence_justification="No evidence provided",
        verdict="NO_EVIDENCE",
        confidence=0.0,
        explanation="Cannot assess convergence without evidence.",
        missing_domains=list(DOMAIN_DIMENSION_WEIGHTS.keys()),
        strongest_per_domain={},
        created_at=datetime.utcnow(),
    )


def _check_independence(
    clusters: List[DomainCluster],
    min_distance: float,
    pairwise_independence: Optional[Dict[Tuple[str, str], bool]] = None,
) -> Dict[str, bool]:
    """Check various independence criteria."""
    checks = {}

    # Check 1: Multiple clusters exist
    checks["multiple_clusters"] = (
        len(clusters) >= CONVERGENCE_THRESHOLDS["min_independent_domains"]
    )

    # Check 2: Minimum distance between clusters
    checks["sufficient_distance"] = (
        min_distance >= CONVERGENCE_THRESHOLDS["min_inter_domain_distance"]
    )

    # Check 3: Method diversity (different method types)
    method_types = set()
    for cluster in clusters:
        if cluster.representative_classification:
            method_types.add(cluster.representative_classification.method_type)
    checks["method_diversity"] = len(method_types) >= 2

    # Check 4: Data source diversity
    data_sources = set()
    for cluster in clusters:
        if cluster.representative_classification:
            data_sources.add(cluster.representative_classification.data_source)
    checks["data_source_diversity"] = len(data_sources) >= 2

    # Check 5: Agent-judged within-cluster independence
    # If the LLM finds that items WITHIN the same domain cluster are actually
    # independent (different labs, different methodologies despite same domain),
    # the cluster has hidden diversity — a positive signal beyond structural checks.
    if pairwise_independence:
        intra_cluster_pairs = []
        for cluster in clusters:
            eids = cluster.evidence_ids
            for i in range(len(eids)):
                for j in range(i + 1, len(eids)):
                    pair_key = (eids[i], eids[j])
                    rev_key = (eids[j], eids[i])
                    if pair_key in pairwise_independence:
                        intra_cluster_pairs.append(pairwise_independence[pair_key])
                    elif rev_key in pairwise_independence:
                        intra_cluster_pairs.append(pairwise_independence[rev_key])
        if intra_cluster_pairs:
            independent_count = sum(1 for v in intra_cluster_pairs if v)
            ratio = independent_count / len(intra_cluster_pairs)
            checks["intra_cluster_diversity"] = (
                ratio >= CONVERGENCE_INTRA_DIVERSITY_THRESHOLD
            )
        else:
            checks["intra_cluster_diversity"] = False
    else:
        checks["intra_cluster_diversity"] = False

    return checks


def _calculate_independence_score(
    clusters: List[DomainCluster],
    avg_distance: float,
    independence_checks: Optional[Dict[str, bool]] = None,
) -> float:
    """Calculate overall independence score (0-1)."""
    if len(clusters) < 2:
        return 0.0

    # Components:
    # 1. Number of clusters (more = better, diminishing returns)
    cluster_score = min(1.0, (len(clusters) - 1) / 3)  # Max at 4 clusters

    # 2. Average inter-cluster distance
    distance_score = min(1.0, avg_distance / 0.7)  # Max at 0.7 distance

    # 3. Quality of cluster representatives
    quality_sum = sum(
        c.representative_classification.classification_confidence
        for c in clusters
        if c.representative_classification
    )
    quality_score = quality_sum / len(clusters) if clusters else 0.5

    # 4. Bonus for agent-judged intra-cluster diversity (if available)
    bonus = 0.0
    if independence_checks and independence_checks.get("intra_cluster_diversity"):
        bonus = 0.1  # small but visible boost

    base = 0.4 * cluster_score + 0.4 * distance_score + 0.2 * quality_score
    return min(1.0, base + bonus)


def _detect_convergence_signal(
    clusters: List[DomainCluster],
    independence_score: float,
    independence_checks: Dict[str, bool],
) -> tuple[bool, float]:
    """Detect if convergence is present and its strength."""
    # Need at least 2 independent domains
    if len(clusters) < 2:
        return False, 0.0

    # Check key independence criteria
    key_checks = ["multiple_clusters", "sufficient_distance"]
    if not all(independence_checks.get(c, False) for c in key_checks):
        return False, 0.0

    # Convergence detected - calculate strength
    # Strong convergence: 3+ domains with high independence score
    if len(clusters) >= CONVERGENCE_THRESHOLDS["strong_convergence_threshold"]:
        strength = min(1.0, 0.7 + independence_score * 0.3)
    else:
        strength = min(0.8, 0.4 + independence_score * 0.4)

    return True, strength


def _determine_verdict(
    clusters: List[DomainCluster],
    convergence_detected: bool,
    strength: float,
    independence_checks: Dict[str, bool],
) -> str:
    """Determine the convergence verdict."""
    if len(clusters) == 0:
        return "NO_EVIDENCE"

    if len(clusters) == 1:
        return "SINGLE_DOMAIN"

    if not convergence_detected:
        # Check if evidence conflicts
        # (For now, assume non-convergent means partial)
        return "PARTIAL"

    if strength >= CONVERGENCE_STRONG_THRESHOLD:
        return "CONVERGENT"
    else:
        return "PARTIAL"


def _generate_explanation(
    clusters: List[DomainCluster],
    verdict: str,
    independence_score: float,
    convergence_detected: bool,
) -> str:
    """Generate human-readable explanation."""
    if verdict == "NO_EVIDENCE":
        return "No evidence was provided to assess convergence."

    if verdict == "SINGLE_DOMAIN":
        if clusters and clusters[0].cluster_label:
            return (
                f"All evidence comes from a single domain ({clusters[0].cluster_label}). "
                "Cross-domain convergence cannot be assessed."
            )
        return "All evidence comes from a single domain. Cross-domain convergence cannot be assessed."

    # Build domain summary
    domain_labels = [c.cluster_label for c in clusters if c.cluster_label]

    if verdict == "CONVERGENT":
        return (
            f"Evidence from {len(clusters)} independent domains converges on this claim. "
            f"Domains: {', '.join(domain_labels)}. "
            f"Independence score: {independence_score:.2f}. "
            "This cross-domain convergence provides strong epistemic support."
        )

    if verdict == "PARTIAL":
        return (
            f"Evidence from {len(clusters)} domains partially supports this claim. "
            f"Domains: {', '.join(domain_labels)}. "
            f"Independence score: {independence_score:.2f}. "
            "More independent evidence would strengthen the case."
        )

    return f"Convergence assessment: {verdict}"


def _generate_justification(
    clusters: List[DomainCluster],
    independence_checks: Dict[str, bool],
) -> str:
    """Generate justification for the convergence assessment."""
    parts = []

    if independence_checks.get("multiple_clusters"):
        parts.append(f"Evidence spans {len(clusters)} distinct domain clusters")
    else:
        parts.append("Evidence concentrated in single domain")

    if independence_checks.get("sufficient_distance"):
        parts.append("domains have sufficient methodological distance")
    else:
        parts.append("domains are closely related")

    if independence_checks.get("method_diversity"):
        parts.append("multiple research methods represented")

    if independence_checks.get("data_source_diversity"):
        parts.append("multiple data sources represented")

    if independence_checks.get("intra_cluster_diversity"):
        parts.append("agent finds within-cluster items methodologically diverse")

    return "; ".join(parts) + "."


def _find_missing_domains(
    classifications: List[DomainClassification],
) -> List[str]:
    """Find domains not represented in the evidence."""
    # Track which values are present for each dimension
    present_methods = {c.method_type.value for c in classifications}
    present_sources = {c.data_source.value for c in classifications}

    missing = []

    # Check method types
    all_methods = {"experimental", "observational", "computational", "theoretical"}
    missing_methods = all_methods - present_methods
    if missing_methods:
        missing.append(f"methods: {', '.join(missing_methods)}")

    # Check data sources
    all_sources = {"primary", "secondary", "synthetic", "meta"}
    missing_sources = all_sources - present_sources
    if missing_sources:
        missing.append(f"sources: {', '.join(missing_sources)}")

    return missing


def _find_strongest_per_domain(
    classifications: List[DomainClassification],
    evidence_qualities: Optional[Dict[str, float]],
) -> Dict[str, str]:
    """Find strongest evidence for each domain cluster."""
    if not evidence_qualities:
        return {}

    # Group by domain label
    domain_groups: Dict[str, List[str]] = {}
    for c in classifications:
        label = get_domain_label(c)
        if label not in domain_groups:
            domain_groups[label] = []
        domain_groups[label].append(c.evidence_id)

    # Find best in each group
    strongest = {}
    for label, evidence_ids in domain_groups.items():
        best_id = max(evidence_ids, key=lambda eid: evidence_qualities.get(eid, 0.5))
        strongest[label] = best_id

    return strongest


def _calculate_confidence(independence_score: float, num_clusters: int) -> float:
    """Calculate confidence in the convergence assessment."""
    if num_clusters == 0:
        return 0.0

    # Base confidence from independence score
    base = independence_score * 0.7

    # Boost for more clusters (diminishing returns)
    cluster_boost = min(0.3, (num_clusters - 1) * 0.1)

    return min(1.0, base + cluster_boost)


def assess_convergence_quality(convergence: ConvergentEvidence) -> Dict[str, Any]:
    """Assess the quality of a convergence result.

    Useful for debugging and understanding convergence assessments.

    Args:
        convergence: The convergence result to assess

    Returns:
        Dict with quality metrics and suggestions
    """
    quality = {
        "overall_quality": "unknown",
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
    }

    # Assess strengths
    if convergence.num_independent_domains >= 3:
        quality["strengths"].append("Strong domain diversity (3+ independent domains)")
    elif convergence.num_independent_domains >= 2:
        quality["strengths"].append("Adequate domain diversity (2 independent domains)")

    if convergence.independence_score >= 0.7:
        quality["strengths"].append("High independence score")

    if convergence.independence_checks.get("method_diversity"):
        quality["strengths"].append("Multiple research methods represented")

    # Assess weaknesses
    if convergence.num_independent_domains < 2:
        quality["weaknesses"].append("Insufficient domain diversity")
        quality["suggestions"].append("Seek evidence from additional research methods")

    if (
        convergence.average_inter_domain_distance
        < CONVERGENCE_INTER_DOMAIN_DISTANCE_LOW
    ):
        quality["weaknesses"].append("Domains are closely related (shared error modes)")
        quality["suggestions"].append(
            "Seek evidence from more methodologically distinct domains"
        )

    if convergence.missing_domains:
        quality["weaknesses"].append(f"Missing domains: {convergence.missing_domains}")

    # Overall quality
    if len(quality["weaknesses"]) == 0 and len(quality["strengths"]) >= 2:
        quality["overall_quality"] = "high"
    elif len(quality["weaknesses"]) <= 1:
        quality["overall_quality"] = "medium"
    else:
        quality["overall_quality"] = "low"

    return quality
