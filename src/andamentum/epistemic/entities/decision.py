"""Decision Entity - Record of a commitment that changes system behavior.

Decisions must:
- Reference claims and stages
- Be justified
- Be reversible (with audit trail)

Examples:
- "Proceed with experimental validation"
- "Treat Claim C17 as actionable for grant A"

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import Field, model_validator

from .base import EpistemicEntity


class Decision(EpistemicEntity):
    """Record of a commitment that changes system behavior.

    State fields for pattern matching:
    - reversed_at: If set, decision has been reversed
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "decision_id" in data and "entity_id" not in data:
            data["entity_id"] = data.pop("decision_id")
        return data

    entity_type: str = "decision"  # type: ignore[assignment]

    # Core fields
    statement: str = Field(description="What was decided")
    justification: str = Field(description="Why this decision was made")
    claim_ids: list[str] = Field(
        default_factory=list,
        description="Claims this decision references"
    )

    # State fields
    reversible: bool = Field(default=True, description="Whether this can be reversed")
    reversed_at: Optional[datetime] = Field(default=None, description="When reversed (None if active)")
    reversal_reason: Optional[str] = Field(default=None, description="Why it was reversed")

    # Provenance
    created_by: str = Field(default="system")

    @property
    def decision_id(self) -> str:
        """Backward-compatible alias for entity_id."""
        return self.entity_id

    @property
    def is_reversed(self) -> bool:
        """Check if this decision has been reversed."""
        return self.reversed_at is not None

    def reverse(self, reason: str) -> None:
        """Reverse this decision with a reason."""
        if not self.reversible:
            raise ValueError("This decision is marked as irreversible")
        self.reversed_at = datetime.now()
        self.reversal_reason = reason
        self.touch()

    def _extra_metadata(self) -> dict[str, Any]:
        """Add decision-specific metadata for filtering."""
        return {
            "statement": self.statement,
            "claim_ids": self.claim_ids,
            "reversible": self.reversible,
            "is_reversed": self.is_reversed,
            "created_by": self.created_by,
        }

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> "Decision":
        """Reconstruct from metadata (legacy support)."""
        reversed_at_str = metadata.get("reversed_at")
        reversed_at = datetime.fromisoformat(reversed_at_str) if reversed_at_str else None

        return cls(
            entity_id=metadata.get("decision_id", ""),
            objective_id=metadata.get("objective_id", ""),
            statement=metadata.get("statement", content),
            justification=metadata.get("justification", ""),
            claim_ids=metadata.get("claim_ids", []),
            reversible=metadata.get("reversible", True),
            reversed_at=reversed_at,
            reversal_reason=metadata.get("reversal_reason"),
            created_by=metadata.get("created_by", "system"),
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.now().isoformat())),
        )
