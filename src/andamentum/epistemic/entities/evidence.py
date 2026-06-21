"""Evidence Entity - Interpreted observation from a source.

Evidence represents what has been observed, NOT claims about meaning.
Evidence does NOT make claims - it supports or challenges them.

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import Field, model_validator

from ..judgment_signal import (
    distribution_confidence,
    distribution_entropy,
    distribution_is_one_hot,
)
from .base import EpistemicEntity


class Evidence(EpistemicEntity):
    """Interpreted observation from a source.

    State fields for pattern matching:
    - extracted: Has content been extracted from source?
    - verified: Has source been verified as accessible?
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "evidence_id" in data and "entity_id" not in data:
            data["entity_id"] = data.pop("evidence_id")
        return data

    entity_type: str = "evidence"  # type: ignore[assignment]

    # Core fields
    source_type: str = Field(
        default="unknown",
        description="paper, dataset, note, conversation, web, prior_claim",
    )
    source_ref: str = Field(
        default="", description="URL, DOI, file path, or document ID"
    )
    extracted_content: str = Field(
        default="", description="What is relevant from the source"
    )
    experimental_context: Optional[str] = Field(
        default=None, description="Conditions/context of the observation"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Known caveats and limitations"
    )

    # Traceability for prior claims
    depends_on_claim_id: Optional[str] = Field(
        default=None,
        description="If source_type='prior_claim', the claim ID this derives from",
    )

    # Multi-seed-claim mode: which sub-investigation's queries fetched this.
    # Set at PlanEvidence time (before Claims exist); read at MultiSeedClaim
    # time to link this Evidence to the specific Claim minted for this
    # sub-investigation. Each Claim then has its OWN evidence subset, which
    # avoids the support_judgment-collision problem (single scalar field;
    # can't represent "supports claim A, contradicts claim B" simultaneously).
    sub_investigation_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the sub-investigation whose queries fetched this evidence "
            "(matches one of objective.decomposition.sub_investigations[i].id). "
            "Used in multi-seed-claim mode to link evidence per-claim."
        ),
    )

    # Quality scoring
    quality_score: Optional[float] = Field(
        default=None, description="Source quality 0.0-1.0 from OpenAlex or heuristic"
    )
    quality_metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Raw quality assessment data for traceability"
    )

    # State fields for pattern matching
    extracted: bool = Field(
        default=False, description="Has content been extracted from source?"
    )
    verified: bool = Field(
        default=False, description="Has source been verified as accessible?"
    )

    # TMS: Invalidation tracking
    invalidated: bool = Field(
        default=False, description="Whether this evidence has been invalidated"
    )
    invalidation_reason: Optional[str] = Field(
        default=None, description="Why invalidated"
    )
    invalidation_cascaded: bool = Field(
        default=False, description="Whether cascade processing is complete"
    )

    # Provenance
    created_by: str = Field(
        default="system", description="Executor or human who registered this"
    )

    # Evidence-claim judgment (set inline by ProposeClaimsOperation / ExtractEvidenceOperation)
    support_judgment: Optional[str] = Field(
        default=None,
        description='LLM judgment: "supports", "contradicts", or "no_bearing"',
    )
    judgment_reasoning: Optional[str] = Field(
        default=None,
        description="One-sentence explanation of the support judgment",
    )
    judgment_distribution: Optional[list[float]] = Field(
        default=None,
        description=(
            "Verbalized belief distribution from the evidence-claim judge "
            "(Tier 0), normalised and ordered by judgment_signal."
            "JUDGMENT_CLASSES = [supports, contradicts, no_bearing]. "
            "support_judgment is its argmax; confidence/entropy/one-hot are "
            "derived via the properties below. None for evidence judged "
            "outside the verbalized-distribution path (e.g. adversarial "
            "counter-evidence) or not yet judged."
        ),
    )

    # Clustering / deduplication state
    cluster_status: str = Field(
        default="unclustered",
        description='Dedup status: "unclustered", "representative", "corroborative", or "deferred"',
    )
    cluster_id: Optional[str] = Field(
        default=None,
        description="ID of the cluster this evidence belongs to",
    )
    corroboration_count: int = Field(
        default=1,
        description="Number of sources confirming this finding (cluster size)",
    )
    corroborating_sources: list[str] = Field(
        default_factory=list,
        description="Source URLs/refs of corroborating evidence in the same cluster",
    )

    def _extra_metadata(self) -> dict[str, Any]:
        """Add evidence-specific metadata for filtering."""
        meta: dict[str, Any] = {
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "extracted": self.extracted,
            "verified": self.verified,
            "invalidated": self.invalidated,
            "invalidation_cascaded": self.invalidation_cascaded,
            "depends_on_claim_id": self.depends_on_claim_id,
            "sub_investigation_id": self.sub_investigation_id,
            "created_by": self.created_by,
            "support_judgment": self.support_judgment,
            "judgment_reasoning": self.judgment_reasoning,
        }
        if self.judgment_distribution is not None:
            meta["judgment_distribution"] = self.judgment_distribution
        if self.invalidation_reason is not None:
            meta["invalidation_reason"] = self.invalidation_reason
        if self.quality_score is not None:
            meta["quality_score"] = self.quality_score
        if self.quality_metadata is not None:
            meta["quality_metadata"] = self.quality_metadata
        meta["cluster_status"] = self.cluster_status
        if self.cluster_id:
            meta["cluster_id"] = self.cluster_id
        meta["corroboration_count"] = self.corroboration_count
        if self.corroborating_sources:
            meta["corroborating_sources"] = self.corroborating_sources
        return meta

    @property
    def evidence_id(self) -> str:
        """Backward-compatible alias for entity_id."""
        return self.entity_id

    @property
    def judgment_confidence(self) -> Optional[float]:
        """Belief mass the judge placed on its verdict, in [0, 1] (Tier 0).
        None when no verbalized distribution was captured."""
        if self.judgment_distribution is None:
            return None
        return distribution_confidence(self.judgment_distribution)

    @property
    def judgment_entropy(self) -> Optional[float]:
        """Normalised entropy of the judge's belief distribution, in [0, 1]
        (higher = less sure — the validated wrong-answer signal). None when
        no verbalized distribution was captured."""
        if self.judgment_distribution is None:
            return None
        return distribution_entropy(self.judgment_distribution)

    @property
    def judgment_one_hot(self) -> Optional[bool]:
        """True if the judge's distribution is degenerate (top class ≥
        threshold), meaning its entropy is uninformative. None when no
        verbalized distribution was captured."""
        if self.judgment_distribution is None:
            return None
        return distribution_is_one_hot(self.judgment_distribution)

    @classmethod
    def from_metadata(
        cls,
        meta: dict[str, Any],
        content: str = "",
        limitations: Optional[list[str]] = None,
    ) -> "Evidence":
        """Reconstruct Evidence from metadata dict (legacy API)."""
        if limitations is not None:
            meta = dict(meta)
            meta["limitations"] = limitations
        return cls._from_metadata(content, meta)

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> "Evidence":
        """Reconstruct from metadata (legacy support)."""
        return cls(
            entity_id=metadata.get("evidence_id", ""),
            objective_id=metadata.get("objective_id", ""),
            source_type=metadata.get("source_type", "unknown"),
            source_ref=metadata.get("source_ref", ""),
            extracted_content=content,
            experimental_context=metadata.get("experimental_context"),
            limitations=metadata.get("limitations", []),
            depends_on_claim_id=metadata.get("depends_on_claim_id"),
            sub_investigation_id=metadata.get("sub_investigation_id"),
            quality_score=metadata.get("quality_score"),
            quality_metadata=metadata.get("quality_metadata"),
            extracted=metadata.get("extracted", False),
            verified=metadata.get("verified", False),
            invalidated=metadata.get("invalidated", False),
            invalidation_reason=metadata.get("invalidation_reason"),
            invalidation_cascaded=metadata.get("invalidation_cascaded", False),
            created_by=metadata.get("created_by", "system"),
            support_judgment=metadata.get("support_judgment"),
            judgment_reasoning=metadata.get("judgment_reasoning"),
            judgment_distribution=metadata.get("judgment_distribution"),
            cluster_status=metadata.get("cluster_status", "unclustered"),
            cluster_id=metadata.get("cluster_id"),
            corroboration_count=metadata.get("corroboration_count", 1),
            corroborating_sources=metadata.get("corroborating_sources", []),
            created_at=datetime.fromisoformat(
                metadata.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                metadata.get("updated_at", datetime.now().isoformat())
            ),
        )
