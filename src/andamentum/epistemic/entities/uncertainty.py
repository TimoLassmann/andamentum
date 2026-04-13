"""Uncertainty Entity - First-class uncertainty that blocks or qualifies claims.

From the philosophy document:
- Uncertainty must never be hidden in prose
- Linked to affected claims
- Blocks promotion (if blocking type)
- Can be resolved or intentionally left open

If the system cannot list its open uncertainties, it is not doing research.

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import Field, model_validator

from .base import EpistemicEntity


class UncertaintyType(str, Enum):
    """Types of epistemic uncertainty.

    Two categories:
    1. BLOCKING uncertainties - prevent claim promotion
    2. NON-BLOCKING uncertainties - recorded as caveats
    """

    # === BLOCKING UNCERTAINTIES (prevent promotion) ===
    UNKNOWN = "unknown"                            # Genuinely missing critical information
    CONTRADICTION = "contradiction"                # Sources genuinely disagree on CORE claim
    COMPUTATIONAL_DISAGREEMENT = "computational_disagreement"  # Dual execution results disagree
    STRONG_COUNTEREVIDENCE = "strong_counterevidence"          # Adversarial search found strong counterarguments
    LOGICAL_INCONSISTENCY = "logical_inconsistency"            # Claim contradicts itself or established facts
    PHYSICAL_IMPLAUSIBILITY = "physical_implausibility"        # Claim violates conservation laws, causality
    MISSING_PREMISE = "missing_premise"                        # Claim cannot be derived without unstated assumptions

    # === NON-BLOCKING UNCERTAINTIES (caveats) ===
    EVIDENCE_GAP = "evidence_gap"                  # Insufficient evidence (not fatal)
    ASSUMPTION = "assumption"                      # We assume X without proof
    RISK = "risk"                                  # X could go wrong
    WEAK_CONVERGENCE = "weak_convergence"          # Evidence sources show weak independence
    DEFINITIONAL_VARIATION = "definitional_variation"  # Depends on how terms are defined
    SCOPE_DIFFERENCE = "scope_difference"          # Different sources apply to different contexts
    METHODOLOGICAL_VARIATION = "methodological_variation"  # Different methods yield different specifics
    PERSPECTIVAL = "perspectival"                  # Valid different viewpoints on same fact
    GRANULARITY_DIFFERENCE = "granularity_difference"  # True at one level, nuanced at finer level


class UncertaintyScope(str, Enum):
    """Scope of an uncertainty's impact."""

    CLAIM = "claim"        # Affects specific claims only (most common)
    OBJECTIVE = "objective"  # Affects entire objective
    GLOBAL = "global"      # Affects entire project (rare)


# Which types block promotion
BLOCKING_TYPES = {
    UncertaintyType.UNKNOWN,
    UncertaintyType.CONTRADICTION,
    UncertaintyType.COMPUTATIONAL_DISAGREEMENT,
    UncertaintyType.STRONG_COUNTEREVIDENCE,
    UncertaintyType.LOGICAL_INCONSISTENCY,
    UncertaintyType.PHYSICAL_IMPLAUSIBILITY,
    UncertaintyType.MISSING_PREMISE,
}


class Uncertainty(EpistemicEntity):
    """First-class uncertainty that blocks or qualifies claims.

    State fields for pattern matching:
    - resolution: How this was resolved (None if unresolved)
    - is_blocking: Denormalized from uncertainty_type for pattern matching
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "uncertainty_id" in data and "entity_id" not in data:
            data["entity_id"] = data.pop("uncertainty_id")
        return data

    entity_type: str = "uncertainty"  # type: ignore[assignment]

    # Core fields
    uncertainty_type: UncertaintyType = Field(
        default=UncertaintyType.UNKNOWN,
        description="Type of uncertainty"
    )
    description: str = Field(description="What is uncertain")
    affected_claim_ids: list[str] = Field(
        default_factory=list,
        description="Claims this affects"
    )
    scope: UncertaintyScope = Field(
        default=UncertaintyScope.CLAIM,
        description="Scope of impact"
    )

    # State fields
    resolution: Optional[str] = Field(
        default=None,
        description="How this was resolved, if at all"
    )
    resolved_at: Optional[datetime] = Field(default=None)

    # Denormalized for pattern matching (set on save)
    is_blocking: bool = Field(
        default=True,
        description="Whether this blocks promotion (computed from type)"
    )

    # Resolution chain tracking
    spawned_from_id: Optional[str] = Field(
        default=None,
        description="Parent uncertainty ID if created during resolution of another uncertainty"
    )

    # Provenance
    created_by: str = Field(default="system")

    def model_post_init(self, __context: Any) -> None:
        """Update denormalized fields after initialization."""
        self.is_blocking = self.uncertainty_type in BLOCKING_TYPES

    @property
    def is_resolved(self) -> bool:
        """Check if this uncertainty has been resolved."""
        return self.resolved_at is not None

    def resolve(self, resolution: str) -> None:
        """Mark this uncertainty as resolved."""
        self.resolution = resolution
        self.resolved_at = datetime.now()
        self.touch()

    def _extra_metadata(self) -> dict[str, Any]:
        """Add uncertainty-specific metadata for filtering."""
        meta: dict[str, Any] = {
            "uncertainty_type": self.uncertainty_type.value,
            "affected_claim_ids": self.affected_claim_ids,
            "scope": self.scope.value,
            "resolution": self.resolution,
            "is_resolved": self.is_resolved,
            "is_blocking": self.is_blocking,
            "spawned_from_id": self.spawned_from_id,
            "created_by": self.created_by,
        }
        return meta

    @property
    def uncertainty_id(self) -> str:
        """Backward-compatible alias for entity_id."""
        return self.entity_id

    @classmethod
    def from_metadata(cls, meta: dict[str, Any], description: str = "") -> "Uncertainty":
        """Reconstruct Uncertainty from metadata dict (legacy API)."""
        return cls._from_metadata(description, meta)

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> "Uncertainty":
        """Reconstruct from metadata (legacy support)."""
        resolved_at_str = metadata.get("resolved_at")
        resolved_at = datetime.fromisoformat(resolved_at_str) if resolved_at_str else None

        return cls(
            entity_id=metadata.get("uncertainty_id", ""),
            objective_id=metadata.get("objective_id", ""),
            uncertainty_type=UncertaintyType(metadata.get("uncertainty_type", "unknown")),
            description=content or metadata.get("description", ""),
            affected_claim_ids=metadata.get("affected_claim_ids", []),
            scope=UncertaintyScope(metadata.get("scope", "claim")),
            resolution=metadata.get("resolution"),
            resolved_at=resolved_at,
            spawned_from_id=metadata.get("spawned_from_id"),
            created_by=metadata.get("created_by", "system"),
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.now().isoformat())),
        )
