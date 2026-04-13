"""Data models for DHP temporal clustering.

Pydantic models and dataclasses for cluster state, results, and summaries.
These are the public-facing types; internal algorithm state lives in dhp.py.
"""

from dataclasses import dataclass, field
from typing import Optional

from .models import DocumentMetadata


@dataclass
class ClusterInfo:
    """Summary information about a single cluster."""

    cluster_id: int
    doc_count: int
    decay_rate: float
    created_at: str
    last_active_at: str


@dataclass
class ClusterDetail(ClusterInfo):
    """Full cluster details including documents and internal parameters."""

    documents: list[DocumentMetadata] = field(default_factory=list)
    kernel_params: dict = field(default_factory=dict)
    centroid: list[float] = field(default_factory=list)


@dataclass
class ReclusterResult:
    """Result of a full offline DHP re-clustering run."""

    n_clusters: int
    n_documents: int
    duration_seconds: float
    config: dict  # Serialized DHPConfig


@dataclass
class ClusterSummary:
    """High-level summary of clustering state for a database."""

    n_clusters: int
    n_clustered_docs: int
    n_unclustered_docs: int
    last_run_config: Optional[dict]
    last_run_at: Optional[str]
