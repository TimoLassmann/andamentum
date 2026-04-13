"""Dirichlet-Hawkes Process (DHP) temporal clustering algorithm.

Implementation of Du et al., "Dirichlet-Hawkes Processes with Applications to
Clustering Continuous-Time Document Streams" (KDD 2015).

This module is PURE computation — it takes embeddings and timestamps as input
and returns cluster assignments. No database I/O. Database integration is
handled by the DocumentStore layer.

All times are in HOURS since Unix epoch (float). This gives good numerical
range for the kernel bandwidths which are specified in hours.

Key equations referenced:
- Eq. 10: New cluster probability (base intensity lambda_0)
- Eq. 11: Existing cluster probability (Hawkes intensity x content similarity)
- Eq. 13: Gaussian RBF kernel mixture
- Eq. 14: Hawkes intensity for a cluster
- Eq. 17: Particle weight update
- Eq. 20: Kernel parameter estimation via weighted average from prior
- Eq. 22: Active interval truncation for efficient computation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DHPConfig:
    """Configuration for the Dirichlet-Hawkes Process algorithm.

    Attributes:
        lambda_0: Base intensity — controls the probability of creating a new
            cluster. Higher values produce more clusters. Typical range: 0.001-0.1.
        kernel_bandwidths: Gaussian RBF kernel bandwidths in hours. Each bandwidth
            captures a different temporal scale. The default covers 30 minutes to
            1 week, suitable for research note-taking and paper ingestion.
        n_particles: Number of SMC particles. More particles give better estimates
            of the posterior but cost proportionally more. 8 is a good default.
        max_gibbs_iter: Maximum Gibbs sampling iterations per document assignment.
            Used in the full offline re-clustering.
        max_metropolis_iter: Maximum Metropolis-Hastings iterations for sampling
            missing timestamps.
        epsilon: Active interval truncation tolerance. Kernel contributions below
            this threshold are ignored. Controls the speed/accuracy tradeoff in
            Hawkes intensity computation.
        similarity_threshold: Minimum cosine similarity between a document embedding
            and a cluster centroid for the document to be considered a member.
            Below this, the content contribution to cluster assignment is zero.
        seed: Random seed for reproducible re-clustering. If None, non-deterministic.
    """

    lambda_0: float = 0.01
    kernel_bandwidths: list[float] = field(default_factory=lambda: [
        0.5, 1.0, 8.0, 12.0, 24.0, 48.0, 96.0, 168.0
    ])  # Hours: 30min, 1h, 8h, 12h, 1d, 2d, 4d, 1wk
    n_particles: int = 8
    max_gibbs_iter: int = 100
    max_metropolis_iter: int = 50
    epsilon: float = 1e-4
    similarity_threshold: float = 0.3
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize to dict for storage in cluster_runs audit table."""
        return {
            "lambda_0": self.lambda_0,
            "kernel_bandwidths": self.kernel_bandwidths,
            "n_particles": self.n_particles,
            "max_gibbs_iter": self.max_gibbs_iter,
            "max_metropolis_iter": self.max_metropolis_iter,
            "epsilon": self.epsilon,
            "similarity_threshold": self.similarity_threshold,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DHPConfig:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Internal state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClusterState:
    """Internal state of a single cluster during DHP inference.

    Attributes:
        cluster_id: Unique integer identifier for this cluster.
        centroid: Running average of document embeddings assigned to this cluster.
            768-dimensional float vector.
        kernel_weights: Weights for the Gaussian RBF kernel mixture, one per
            bandwidth in DHPConfig.kernel_bandwidths. Sum to 1.
        doc_times: Timestamps (hours since epoch) of all documents in this cluster,
            sorted ascending.
        doc_count: Number of documents assigned to this cluster.
        created_at: Timestamp (hours since epoch) when cluster was created.
        last_active_at: Timestamp of the most recent document assigned.
    """

    cluster_id: int
    centroid: np.ndarray
    kernel_weights: np.ndarray
    doc_times: list[float] = field(default_factory=list)
    doc_count: int = 0
    created_at: float = 0.0
    last_active_at: float = 0.0

    def copy(self) -> ClusterState:
        """Deep copy for particle branching."""
        return ClusterState(
            cluster_id=self.cluster_id,
            centroid=self.centroid.copy(),
            kernel_weights=self.kernel_weights.copy(),
            doc_times=list(self.doc_times),
            doc_count=self.doc_count,
            created_at=self.created_at,
            last_active_at=self.last_active_at,
        )


@dataclass
class Particle:
    """A single SMC particle representing one possible clustering state.

    The SMC (Sequential Monte Carlo) framework maintains multiple particles,
    each representing a different hypothesis about how documents are assigned
    to clusters. Particle weights reflect how well each hypothesis explains
    the observed data.

    Attributes:
        cluster_states: Mapping from cluster_id to the cluster's internal state.
        assignments: Mapping from doc_uuid to the cluster_id assigned in this particle.
        weight: Log-space particle weight. Higher weight means this particle
            better explains the observed data.
        next_cluster_id: Counter for generating new cluster IDs.
    """

    cluster_states: dict[int, ClusterState] = field(default_factory=dict)
    assignments: dict[str, int] = field(default_factory=dict)
    weight: float = 0.0  # Log-space
    next_cluster_id: int = 0

    def copy(self) -> Particle:
        """Deep copy for resampling."""
        return Particle(
            cluster_states={k: v.copy() for k, v in self.cluster_states.items()},
            assignments=dict(self.assignments),
            weight=self.weight,
            next_cluster_id=self.next_cluster_id,
        )


# ---------------------------------------------------------------------------
# Core mathematical functions
# ---------------------------------------------------------------------------

def _gaussian_rbf_kernel(delta: float, bandwidth: float) -> float:
    """Gaussian RBF kernel value.

    Computes kappa(tau, delta) = exp(-(delta - tau)^2 / (2 * tau^2)) / (sqrt(2*pi) * tau)

    where tau is the bandwidth parameter (used as both center and standard deviation).

    Args:
        delta: Time difference (hours). Must be >= 0.
        bandwidth: Kernel bandwidth tau (hours). Also used as std dev sigma = tau.

    Returns:
        Kernel value. Always >= 0.
    """
    if bandwidth <= 0.0:
        return 0.0
    sigma = bandwidth
    exponent = -((delta - bandwidth) ** 2) / (2.0 * sigma * sigma)
    # Clamp exponent to avoid underflow
    if exponent < -500.0:
        return 0.0
    normalizer = math.sqrt(2.0 * math.pi) * sigma
    return math.exp(exponent) / normalizer


def _active_interval_lower_bound(t: float, max_bandwidth: float, epsilon: float) -> float:
    """Compute the active interval lower bound (Eq. 22).

    Only documents with timestamps >= t_u need to be considered in the Hawkes
    intensity sum, because documents before t_u contribute less than epsilon
    to the intensity.

    t_u = t - (tau_m + sqrt(-2 * sigma_m^2 * ln(0.5 * epsilon * sqrt(2 * pi * sigma_m^2))))

    where tau_m = max bandwidth, sigma_m = max bandwidth.

    Args:
        t: Current time (hours since epoch).
        max_bandwidth: The largest kernel bandwidth tau_m.
        epsilon: Truncation tolerance.

    Returns:
        Lower bound timestamp t_u. Documents before this can be skipped.
    """
    if max_bandwidth <= 0.0:
        return t
    sigma_m = max_bandwidth
    sigma_m_sq = sigma_m * sigma_m

    # Argument to log: 0.5 * epsilon * sqrt(2 * pi * sigma_m^2)
    log_arg = 0.5 * epsilon * math.sqrt(2.0 * math.pi * sigma_m_sq)
    if log_arg <= 0.0 or log_arg >= 1.0:
        # If epsilon is too large, fall back to considering all documents
        return -math.inf

    inner = -2.0 * sigma_m_sq * math.log(log_arg)
    if inner < 0.0:
        return t - max_bandwidth
    offset = max_bandwidth + math.sqrt(inner)
    return t - offset


def hawkes_intensity(cluster: ClusterState, t: float, config: DHPConfig) -> float:
    """Compute Hawkes intensity lambda_k(t) for a cluster at time t (Eq. 14).

    lambda_k(t) = sum over documents i in cluster k of:
        alpha^T * kappa(tau, t - t_i)

    where alpha are the kernel weights and kappa is the Gaussian RBF kernel
    with bandwidths tau.

    Uses active interval optimization (Eq. 22): only sums over documents
    within [t_u, t] where kernel contribution exceeds epsilon.

    Args:
        cluster: The cluster state containing doc_times and kernel_weights.
        t: Current time (hours since epoch).
        config: DHP configuration (bandwidths, epsilon).

    Returns:
        Hawkes intensity value. Always >= 0.
    """
    if cluster.doc_count == 0 or len(cluster.doc_times) == 0:
        return 0.0

    bandwidths = config.kernel_bandwidths
    n_kernels = len(bandwidths)
    weights = cluster.kernel_weights

    if len(weights) != n_kernels:
        # Fallback: uniform weights if mismatch
        weights = np.ones(n_kernels) / n_kernels

    max_bw = max(bandwidths)
    t_u = _active_interval_lower_bound(t, max_bw, config.epsilon)

    total_intensity = 0.0

    for t_i in cluster.doc_times:
        if t_i > t:
            # Future documents don't contribute
            break
        if t_i < t_u:
            # Outside active interval — contribution below epsilon
            continue

        delta = t - t_i
        # Sum weighted kernel values across all bandwidths
        kernel_sum = 0.0
        for l in range(n_kernels):
            kernel_sum += weights[l] * _gaussian_rbf_kernel(delta, bandwidths[l])

        total_intensity += kernel_sum

    return total_intensity


def content_similarity(doc_embedding: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine similarity between a document embedding and a cluster centroid.

    Handles zero vectors gracefully by returning 0.0.

    Args:
        doc_embedding: Document embedding vector.
        centroid: Cluster centroid vector.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero.
    """
    norm_doc = np.linalg.norm(doc_embedding)
    norm_cent = np.linalg.norm(centroid)

    if norm_doc < 1e-10 or norm_cent < 1e-10:
        return 0.0

    return float(np.dot(doc_embedding, centroid) / (norm_doc * norm_cent))


def compute_assignment_probabilities(
    doc_embedding: np.ndarray,
    doc_time: float,
    particle: Particle,
    config: DHPConfig,
) -> tuple[list[int], np.ndarray]:
    """Compute assignment probabilities for a document across all clusters (Eq. 10-11).

    For each existing cluster k:
        score_k = lambda_k(t) * max(cosine_sim(embedding, centroid_k), 0)  (Eq. 11)

    For a new cluster:
        score_new = lambda_0  (Eq. 10)

    Scores are normalized to a probability distribution.

    Args:
        doc_embedding: Document embedding vector.
        doc_time: Document timestamp (hours since epoch).
        particle: Current particle state.
        config: DHP configuration.

    Returns:
        Tuple of (cluster_ids, probabilities) where cluster_ids[-1] is the
        sentinel for "new cluster" (particle.next_cluster_id).
    """
    cluster_ids: list[int] = []
    scores: list[float] = []

    for cid, cstate in particle.cluster_states.items():
        intensity = hawkes_intensity(cstate, doc_time, config)
        sim = content_similarity(doc_embedding, cstate.centroid)

        # Apply similarity threshold: below threshold, content contribution is zero
        if sim < config.similarity_threshold:
            sim = 0.0
        else:
            # Shift similarity to be non-negative (cosine sim can be negative)
            sim = max(sim, 0.0)

        score = intensity * sim
        cluster_ids.append(cid)
        scores.append(score)

    # New cluster option (Eq. 10)
    new_cluster_id = particle.next_cluster_id
    cluster_ids.append(new_cluster_id)
    scores.append(config.lambda_0)

    # Convert to probability distribution
    scores_arr = np.array(scores, dtype=np.float64)
    total = scores_arr.sum()

    if total <= 0.0:
        # All scores zero — assign to new cluster with certainty
        probs = np.zeros(len(scores_arr))
        probs[-1] = 1.0
    else:
        probs = scores_arr / total

    return cluster_ids, probs


def update_cluster_centroid(cluster: ClusterState, new_embedding: np.ndarray, learning_rate: float = 0.1) -> None:
    """Update cluster centroid with a running weighted average.

    centroid_new = (1 - lr) * centroid_old + lr * new_embedding

    The centroid is then L2-normalized to maintain unit length, which ensures
    cosine similarity comparisons remain meaningful.

    Args:
        cluster: Cluster state to update in place.
        new_embedding: New document embedding to incorporate.
        learning_rate: Weight for the new embedding. Default 0.1 gives
            a smooth update that doesn't overreact to individual documents.
    """
    if cluster.doc_count <= 1:
        # First document: centroid IS the embedding
        cluster.centroid = new_embedding.copy()
    else:
        cluster.centroid = (1.0 - learning_rate) * cluster.centroid + learning_rate * new_embedding

    # Normalize to unit length
    norm = np.linalg.norm(cluster.centroid)
    if norm > 1e-10:
        cluster.centroid /= norm


def update_kernel_params(cluster: ClusterState, config: DHPConfig) -> None:
    """Estimate triggering kernel parameters using weighted average from prior (Eq. 20).

    Uses the observed inter-arrival times within the cluster to update the
    kernel weights. The update is a blend between the prior (uniform weights)
    and the empirical contribution of each kernel bandwidth to explaining
    the observed inter-arrival times.

    Args:
        cluster: Cluster state to update in place.
        config: DHP configuration containing bandwidths.
    """
    n_kernels = len(config.kernel_bandwidths)

    if cluster.doc_count < 2 or len(cluster.doc_times) < 2:
        # Not enough data — keep uniform prior
        cluster.kernel_weights = np.ones(n_kernels) / n_kernels
        return

    # Compute empirical kernel contributions from inter-arrival times
    empirical_weights = np.zeros(n_kernels, dtype=np.float64)
    n_pairs = 0

    times = sorted(cluster.doc_times)
    for i in range(1, len(times)):
        delta = times[i] - times[i - 1]
        if delta <= 0.0:
            continue

        for l in range(n_kernels):
            empirical_weights[l] += _gaussian_rbf_kernel(delta, config.kernel_bandwidths[l])
        n_pairs += 1

    if n_pairs > 0 and empirical_weights.sum() > 0:
        empirical_weights /= empirical_weights.sum()

        # Blend with uniform prior (Eq. 20 weighted average)
        # As we see more data, trust the empirical estimate more
        prior_weight = 1.0 / (1.0 + n_pairs)
        uniform_prior = np.ones(n_kernels) / n_kernels
        cluster.kernel_weights = prior_weight * uniform_prior + (1.0 - prior_weight) * empirical_weights
    else:
        cluster.kernel_weights = np.ones(n_kernels) / n_kernels

    # Ensure normalization
    total = cluster.kernel_weights.sum()
    if total > 0:
        cluster.kernel_weights /= total


def effective_sample_size(particles: list[Particle]) -> float:
    """Compute effective sample size (ESS) from particle weights.

    ESS = 1 / sum(w_i^2) where w_i are normalized weights.

    An ESS of N (number of particles) means all particles are equally
    weighted (maximum diversity). An ESS of 1 means one particle dominates.

    Args:
        particles: List of particles with log-space weights.

    Returns:
        Effective sample size. Range: [1, len(particles)].
    """
    if not particles:
        return 0.0

    # Convert from log space to normalized probabilities
    log_weights = np.array([p.weight for p in particles])
    max_log_w = log_weights.max()

    # Subtract max for numerical stability (log-sum-exp trick)
    shifted = log_weights - max_log_w
    weights = np.exp(shifted)
    total = weights.sum()

    if total <= 0.0:
        return 1.0

    normalized = weights / total
    sum_sq = np.sum(normalized ** 2)

    if sum_sq <= 0.0:
        return float(len(particles))

    return float(1.0 / sum_sq)


def resample_particles(particles: list[Particle], rng: np.random.Generator) -> list[Particle]:
    """Systematic resampling of particles when ESS drops below threshold.

    Systematic resampling is preferred over multinomial because it has lower
    variance: it guarantees that high-weight particles are represented
    proportionally, while still maintaining randomness.

    After resampling, all particle weights are reset to uniform (log 0 = 0).

    Args:
        particles: List of particles with log-space weights.
        rng: Numpy random generator for reproducibility.

    Returns:
        New list of particles after resampling. Same length as input.
    """
    n = len(particles)
    if n == 0:
        return []

    # Normalize weights to probabilities
    log_weights = np.array([p.weight for p in particles])
    max_log_w = log_weights.max()
    shifted = log_weights - max_log_w
    weights = np.exp(shifted)
    total = weights.sum()

    if total <= 0.0:
        # All weights zero — uniform resampling
        weights = np.ones(n) / n
    else:
        weights = weights / total

    # Systematic resampling
    cumsum = np.cumsum(weights)
    u = rng.uniform(0.0, 1.0 / n)
    positions = u + np.arange(n) / n

    indices = np.zeros(n, dtype=int)
    j = 0
    for i in range(n):
        while cumsum[j] < positions[i] and j < n - 1:
            j += 1
        indices[i] = j

    # Create new particles from selected indices
    new_particles = []
    for idx in indices:
        p = particles[idx].copy()
        p.weight = 0.0  # Reset to uniform in log space
        new_particles.append(p)

    return new_particles


# ---------------------------------------------------------------------------
# Online assignment (Algorithm 1 from paper)
# ---------------------------------------------------------------------------

def _create_new_cluster(
    particle: Particle,
    doc_embedding: np.ndarray,
    doc_time: float,
    config: DHPConfig,
) -> int:
    """Create a new cluster in a particle and return its ID."""
    cid = particle.next_cluster_id
    n_kernels = len(config.kernel_bandwidths)

    cluster = ClusterState(
        cluster_id=cid,
        centroid=doc_embedding.copy(),
        kernel_weights=np.ones(n_kernels) / n_kernels,  # Uniform prior
        doc_times=[doc_time],
        doc_count=1,
        created_at=doc_time,
        last_active_at=doc_time,
    )
    # Normalize centroid
    norm = np.linalg.norm(cluster.centroid)
    if norm > 1e-10:
        cluster.centroid /= norm

    particle.cluster_states[cid] = cluster
    particle.next_cluster_id = cid + 1
    return cid


def assign_document_online(
    doc_uuid: str,
    doc_embedding: np.ndarray,
    doc_time: float,
    particles: list[Particle],
    config: DHPConfig,
    rng: np.random.Generator,
) -> tuple[list[Particle], int]:
    """One step of online SMC (Algorithm 1). Assigns a document to a cluster.

    For each particle:
    1. Compute assignment probabilities across existing clusters + new cluster
    2. Sample a cluster assignment from the categorical distribution
    3. Update the assigned cluster's centroid and kernel parameters
    4. Update particle weight

    After all particles are updated, resample if ESS drops below N/2.
    The final assignment is determined by majority vote across particles.

    Args:
        doc_uuid: Unique document identifier.
        doc_embedding: Document embedding vector (768-dim).
        doc_time: Document timestamp (hours since epoch).
        particles: List of SMC particles representing current clustering hypotheses.
        config: DHP configuration.
        rng: Numpy random generator.

    Returns:
        Tuple of (updated_particles, chosen_cluster_id) where chosen_cluster_id
        is the majority-vote assignment across all particles.
    """
    for particle in particles:
        cluster_ids, probs = compute_assignment_probabilities(
            doc_embedding, doc_time, particle, config
        )

        # Sample assignment
        chosen_idx = rng.choice(len(cluster_ids), p=probs)
        chosen_cid = cluster_ids[chosen_idx]

        # Check if this is a new cluster
        new_cluster_id = particle.next_cluster_id
        if chosen_cid == new_cluster_id:
            chosen_cid = _create_new_cluster(particle, doc_embedding, doc_time, config)
        else:
            # Update existing cluster
            cstate = particle.cluster_states[chosen_cid]
            cstate.doc_times.append(doc_time)
            cstate.doc_count += 1
            cstate.last_active_at = doc_time
            update_cluster_centroid(cstate, doc_embedding)
            update_kernel_params(cstate, config)

        particle.assignments[doc_uuid] = chosen_cid

        # Update particle weight (Eq. 17): log w += log P(assignment)
        chosen_prob = probs[chosen_idx]
        if chosen_prob > 0:
            particle.weight += math.log(chosen_prob)
        else:
            particle.weight += -100.0  # Very low weight for impossible assignment

    # Resample if ESS drops below N/2
    ess = effective_sample_size(particles)
    n = len(particles)
    if ess < n / 2.0:
        particles = resample_particles(particles, rng)

    # Majority vote for final assignment
    vote_counts: dict[int, int] = {}
    for p in particles:
        cid = p.assignments.get(doc_uuid, -1)
        vote_counts[cid] = vote_counts.get(cid, 0) + 1

    chosen_cluster_id = max(vote_counts, key=lambda k: vote_counts[k])

    return particles, chosen_cluster_id


# ---------------------------------------------------------------------------
# Offline re-clustering
# ---------------------------------------------------------------------------

def recluster(
    embeddings_and_times: list[tuple[str, np.ndarray, float]],
    config: DHPConfig,
) -> tuple[dict[str, int], dict[int, ClusterState]]:
    """Full offline DHP re-clustering.

    Processes all documents in temporal order through the complete SMC
    framework. This is the same algorithm as online assignment but run
    from scratch on the full dataset.

    The result is deterministic when config.seed is set.

    Args:
        embeddings_and_times: List of (doc_uuid, 768-dim embedding, timestamp_hours)
            MUST be sorted by timestamp ascending. Embeddings should be numpy
            arrays. Timestamps are in hours since Unix epoch.
        config: DHP configuration controlling cluster formation behavior.

    Returns:
        Tuple of (assignments, cluster_states) where:
        - assignments: {doc_uuid: cluster_id} mapping for every document
        - cluster_states: {cluster_id: ClusterState} with final cluster parameters

    Raises:
        ValueError: If embeddings_and_times is empty.
    """
    if not embeddings_and_times:
        return {}, {}

    # Initialize RNG
    rng = np.random.default_rng(config.seed)

    # Initialize particles
    particles: list[Particle] = []
    for _ in range(config.n_particles):
        particles.append(Particle(
            cluster_states={},
            assignments={},
            weight=0.0,
            next_cluster_id=0,
        ))

    # Process each document in temporal order
    for doc_uuid, embedding, timestamp in embeddings_and_times:
        emb = np.asarray(embedding, dtype=np.float64)
        particles, _ = assign_document_online(
            doc_uuid=doc_uuid,
            doc_embedding=emb,
            doc_time=timestamp,
            particles=particles,
            config=config,
            rng=rng,
        )

    # Select best particle (highest weight) for final assignments
    best_particle = max(particles, key=lambda p: p.weight)

    return dict(best_particle.assignments), dict(best_particle.cluster_states)


# ---------------------------------------------------------------------------
# Cluster scoring for search
# ---------------------------------------------------------------------------

def score_clusters_for_query(
    query_embedding: np.ndarray,
    current_time: float,
    cluster_states: dict[int, ClusterState],
    config: DHPConfig,
) -> list[tuple[int, float]]:
    """Score clusters for a search query.

    Each cluster is scored by combining content relevance (cosine similarity
    between query embedding and cluster centroid) with temporal recency
    (current Hawkes intensity of the cluster).

    score_k = cosine(query, centroid_k) * lambda_k(now)

    Clusters with recent activity AND relevant content score highest.

    Args:
        query_embedding: Query embedding vector.
        current_time: Current time (hours since epoch) for Hawkes intensity.
        cluster_states: Mapping from cluster_id to ClusterState.
        config: DHP configuration.

    Returns:
        List of (cluster_id, score) tuples sorted by score descending.
        Only clusters with positive scores are included.
    """
    query_emb = np.asarray(query_embedding, dtype=np.float64)
    results: list[tuple[int, float]] = []

    for cid, cstate in cluster_states.items():
        sim = content_similarity(query_emb, cstate.centroid)
        if sim <= 0.0:
            continue

        intensity = hawkes_intensity(cstate, current_time, config)
        if intensity <= 0.0:
            # Cluster has no temporal activity — still include with reduced score
            # Use a small baseline so content-only matches aren't completely lost
            intensity = config.lambda_0 * 0.01

        score = sim * intensity
        if score > 0.0:
            results.append((cid, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def timestamp_to_hours(unix_timestamp: float) -> float:
    """Convert Unix timestamp (seconds) to hours since epoch.

    Args:
        unix_timestamp: Standard Unix timestamp in seconds.

    Returns:
        Hours since epoch (float).
    """
    return unix_timestamp / 3600.0


def hours_to_timestamp(hours: float) -> float:
    """Convert hours since epoch back to Unix timestamp (seconds).

    Args:
        hours: Hours since Unix epoch.

    Returns:
        Standard Unix timestamp in seconds.
    """
    return hours * 3600.0


def cluster_state_to_dict(cluster: ClusterState) -> dict:
    """Serialize a ClusterState to a JSON-compatible dict.

    Used for storing cluster state in the database.

    Args:
        cluster: Cluster state to serialize.

    Returns:
        Dict with all cluster fields as JSON-compatible types.
    """
    return {
        "cluster_id": cluster.cluster_id,
        "centroid": cluster.centroid.tolist(),
        "kernel_weights": cluster.kernel_weights.tolist(),
        "doc_times": cluster.doc_times,
        "doc_count": cluster.doc_count,
        "created_at": cluster.created_at,
        "last_active_at": cluster.last_active_at,
    }


def cluster_state_from_dict(d: dict) -> ClusterState:
    """Deserialize a ClusterState from a JSON-compatible dict.

    Args:
        d: Dict produced by cluster_state_to_dict().

    Returns:
        ClusterState instance.
    """
    return ClusterState(
        cluster_id=d["cluster_id"],
        centroid=np.array(d["centroid"], dtype=np.float64),
        kernel_weights=np.array(d["kernel_weights"], dtype=np.float64),
        doc_times=d["doc_times"],
        doc_count=d["doc_count"],
        created_at=d["created_at"],
        last_active_at=d["last_active_at"],
    )
