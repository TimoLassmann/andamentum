"""Snapshot Entity - Immutable epistemic state for artefact generation.

A snapshot freezes the current state of claims, uncertainties, and evidence
at a point in time. Artefacts are ALWAYS compiled from snapshots.

This ensures artefacts are traceable to a specific epistemic state.

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import Field, model_validator

from .base import EpistemicEntity
from .claim import ClaimStage
from .verdict import EpistemicVerdict


class Snapshot(EpistemicEntity):
    """Immutable epistemic state for artefact generation.

    State fields for pattern matching:
    - snapshot_type: "checkpoint" or "final"
    - artefact_id: ID of generated artefact (None until compiled)
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "snapshot_id" in data and "entity_id" not in data:
            data["entity_id"] = data.pop("snapshot_id")
        return data

    entity_type: str = "snapshot"  # type: ignore[assignment]

    # Core fields - IDs of entities included in snapshot
    claim_ids: list[str] = Field(
        default_factory=list, description="Claims included at their current stages"
    )
    uncertainty_ids: list[str] = Field(
        default_factory=list,
        description="Active (unresolved) uncertainties at freeze time",
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="All evidence referenced by included claims"
    )

    # Filtering criteria used when creating snapshot
    minimum_claim_stage: ClaimStage = Field(
        default=ClaimStage.SUPPORTED,
        description="Minimum stage for claims to be included",
    )

    # State fields for pattern matching
    snapshot_type: str = Field(
        default="checkpoint",
        description="'checkpoint' (interim) or 'final' (ready for artefact)",
    )
    artefact_id: Optional[str] = Field(
        default=None, description="ID of artefact generated from this snapshot"
    )
    frozen: bool = Field(
        default=True, description="Snapshots are immutable once created"
    )

    # The run's single decision record, stamped by FreezeSnapshotOperation
    # from EpistemicGraphState.verdict before the snapshot is frozen. This
    # is the auditable freeze half of the artefact duality: the live route
    # is PipelineResult.verdict, the persisted route is
    # objective.snapshot_id → snapshot.verdict. Both synthesis operations
    # RENDER this object and raise when it is absent — a missing verdict
    # at synthesis is a routing bug, not a reason to omit the headline.
    #
    # Replaces the previous ``combined_verdict: dict[str, Any]``, which was
    # written only for decomposed runs and was an untyped boundary
    # CLAUDE.md forbids. Round-trips through ``to_document``'s full-JSON
    # path; it was never a metadata filter key.
    verdict: Optional[EpistemicVerdict] = Field(
        default=None,
        description=(
            "The typed EpistemicVerdict this snapshot froze. None for "
            "snapshots written before the verdict object existed, or for "
            "snapshots created outside the graph."
        ),
    )

    @property
    def snapshot_id(self) -> str:
        """Backward-compatible alias for entity_id."""
        return self.entity_id

    def _extra_metadata(self) -> dict[str, Any]:
        """Add snapshot-specific metadata for filtering."""
        meta: dict[str, Any] = {
            "claim_ids": self.claim_ids,
            "uncertainty_ids": self.uncertainty_ids,
            "evidence_ids": self.evidence_ids,
            "minimum_claim_stage": self.minimum_claim_stage.value,
            "claim_count": len(self.claim_ids),
            "snapshot_type": self.snapshot_type,
            "artefact_id": self.artefact_id,
            "frozen": self.frozen,
        }
        return meta

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> "Snapshot":
        """Reconstruct from metadata (legacy support)."""
        min_stage = metadata.get("minimum_claim_stage", "supported")
        if isinstance(min_stage, str):
            min_stage = ClaimStage(min_stage)

        return cls(
            entity_id=metadata.get("snapshot_id", ""),
            objective_id=metadata.get("objective_id", ""),
            claim_ids=metadata.get("claim_ids", []),
            uncertainty_ids=metadata.get("uncertainty_ids", []),
            evidence_ids=metadata.get("evidence_ids", []),
            minimum_claim_stage=min_stage,
            snapshot_type=metadata.get("snapshot_type", "checkpoint"),
            artefact_id=metadata.get("artefact_id"),
            frozen=metadata.get("frozen", True),
            created_at=datetime.fromisoformat(
                metadata.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                metadata.get("updated_at", datetime.now().isoformat())
            ),
        )
