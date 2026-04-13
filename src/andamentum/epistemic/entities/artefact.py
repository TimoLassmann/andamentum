"""Artefact Entity - Human-facing output compiled from epistemic state.

Artefacts are COMPILED VIEWS of epistemic state - they:
- May simplify or omit uncertainty
- Must NEVER invent beliefs
- Are always traceable to a snapshot

If an artefact is wrong, the epistemic plane must be corrected
and the artefact regenerated.

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from typing import Any
from pydantic import Field, model_validator

from .base import EpistemicEntity


class Artefact(EpistemicEntity):
    """Human-facing output compiled from epistemic state.

    State fields for pattern matching:
    - artefact_type: Type of output (summary, report, etc.)
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "artefact_id" in data and "entity_id" not in data:
            data["entity_id"] = data.pop("artefact_id")
        return data

    entity_type: str = "artefact"  # type: ignore[assignment]

    # Core fields
    snapshot_id: str = Field(description="Snapshot this was compiled from")
    artefact_type: str = Field(
        default="summary",
        description="Type: grant_rationale, manuscript, slides, memo, summary, report, analysis"
    )
    audience_profile: str = Field(
        default="general",
        description="Target audience"
    )
    content: str = Field(
        default="",
        description="The generated content (markdown), including quality signals"
    )
    content_body: str = Field(
        default="",
        description="Content without quality signals (confidence headers, methodology stats). "
        "Use this for benchmarking or downstream interpretation where pre-computed "
        "quality labels would bias the consumer."
    )

    # Traceability - maps paragraph IDs to claim IDs
    trace: dict[str, list[str]] = Field(
        default_factory=dict,
        description="paragraph_id -> claim_ids mapping for traceability"
    )

    @property
    def artefact_id(self) -> str:
        """Backward-compatible alias for entity_id."""
        return self.entity_id

    def _extra_metadata(self) -> dict[str, Any]:
        """Add artefact-specific metadata for filtering."""
        meta: dict[str, Any] = {
            "snapshot_id": self.snapshot_id,
            "artefact_type": self.artefact_type,
            "audience_profile": self.audience_profile,
            "trace": self.trace,
            "content_length": len(self.content),
        }
        if self.content_body:
            meta["content_body"] = self.content_body
        return meta

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> "Artefact":
        """Reconstruct from metadata (legacy support)."""
        return cls(
            entity_id=metadata.get("artefact_id", ""),
            objective_id=metadata.get("objective_id", ""),
            snapshot_id=metadata.get("snapshot_id", ""),
            artefact_type=metadata.get("artefact_type", "summary"),
            audience_profile=metadata.get("audience_profile", "general"),
            content=content,
            content_body=metadata.get("content_body", ""),
            trace=metadata.get("trace", {}),
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.now().isoformat())),
        )
