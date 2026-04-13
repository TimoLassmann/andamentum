"""Objective Entity - Top-level research objective.

The objective defines what we're trying to learn or produce.
All epistemic work is scoped to an objective.

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import Field

from .base import EpistemicEntity


class Objective(EpistemicEntity):
    """Top-level research objective with phase tracking.

    State fields for pattern matching:
    - phase: Current workflow phase
    - claims_proposed: Whether claims have been proposed
    - snapshot_id: ID of the current snapshot (if frozen)
    - artefact_id: ID of the generated artefact (if complete)
    """

    entity_type: str = "objective"  # type: ignore[assignment]

    # Core fields
    description: str = Field(description="What are we trying to learn/produce")
    goal_context: Optional[str] = Field(
        default=None,
        description="Higher-level goal for alignment (goal grounding)"
    )
    artefact_specs: list[str] = Field(
        default_factory=list,
        description="Expected deliverables (summary, report, etc.)"
    )

    # Clarification results (populated by ClarifyQuestionOperation)
    clarified_question: Optional[str] = Field(
        default=None, description="Clarified version of the research question"
    )
    key_terms: list[str] = Field(
        default_factory=list, description="Key terms identified during clarification"
    )

    # Question type classification (set by ClassifyQuestionOperation, immutable after)
    question_type: Optional[str] = Field(
        default=None,
        description="Epistemic question type for verification routing (set once by classify_question)",
    )

    # Phase tracking - pattern matching field
    # Phase values (in order):
    #   "new" -> "clarified" -> "analyzed" -> "planned" ->
    #   "claims_proposed" -> "claims_done" -> "complete"
    phase: str = Field(default="new", description="Current workflow phase")

    # State fields for pattern matching
    claims_proposed: bool = Field(default=False, description="Whether claims have been proposed")
    snapshot_id: Optional[str] = Field(default=None, description="ID of current snapshot")
    artefact_id: Optional[str] = Field(default=None, description="ID of generated artefact")

    # Buffered remaining concerns from uncertainty resolution.
    # Collected during the resolution round and batch-deduped before
    # creating new uncertainty entities. Each entry is a dict with:
    #   text: str, parent_id: str, affected_claim_ids: list[str], depth: int
    pending_concerns: list[dict] = Field(
        default_factory=list,
        description="Buffered remaining concerns awaiting batch dedup",
    )

    # Status (separate from phase - can pause/abandon at any phase)
    status: str = Field(default="active", description="active, paused, completed, abandoned")

    @property
    def pending_concerns_count(self) -> int:
        """Number of buffered concerns — used by pattern matching."""
        return len(self.pending_concerns)

    def _extra_metadata(self) -> dict[str, Any]:
        """Add objective-specific metadata for filtering."""
        meta: dict[str, Any] = {
            "phase": self.phase,
            "claims_proposed": self.claims_proposed,
            "snapshot_id": self.snapshot_id,
            "artefact_id": self.artefact_id,
            "status": self.status,
            "pending_concerns_count": len(self.pending_concerns),
        }
        if self.question_type:
            meta["question_type"] = self.question_type
        if self.clarified_question:
            meta["clarified_question"] = self.clarified_question
        if self.key_terms:
            meta["key_terms"] = self.key_terms
        return meta

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> "Objective":
        """Reconstruct from metadata (legacy support)."""
        return cls(
            entity_id=metadata.get("objective_id", ""),
            objective_id=metadata.get("objective_id", ""),  # Self-referential for objectives
            description=content or metadata.get("description", ""),
            goal_context=metadata.get("goal_context"),
            artefact_specs=metadata.get("artefact_specs", []),
            clarified_question=metadata.get("clarified_question"),
            key_terms=metadata.get("key_terms", []),
            question_type=metadata.get("question_type"),
            phase=metadata.get("phase", "new"),
            claims_proposed=metadata.get("claims_proposed", False),
            snapshot_id=metadata.get("snapshot_id"),
            artefact_id=metadata.get("artefact_id"),
            status=metadata.get("status", "active"),
            created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(metadata.get("updated_at", datetime.now().isoformat())),
        )
