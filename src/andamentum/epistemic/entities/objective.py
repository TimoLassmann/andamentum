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
        default=None, description="Higher-level goal for alignment (goal grounding)"
    )
    artefact_specs: list[str] = Field(
        default_factory=list,
        description="Expected deliverables (summary, report, etc.)",
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

    # Claim verification mode. When set, the pipeline skips propose_claims
    # and creates a single Claim entity from this exact statement. Used for
    # benchmarks (e.g. SciFact) where the claim is known and the task is to
    # verify it, not to discover new claims from evidence.
    claim_to_verify: Optional[str] = Field(
        default=None,
        description="Seed claim for verification mode (skips propose_claims)",
    )

    # State fields for pattern matching
    claims_proposed: bool = Field(
        default=False, description="Whether claims have been proposed"
    )
    snapshot_id: Optional[str] = Field(
        default=None, description="ID of current snapshot"
    )
    artefact_id: Optional[str] = Field(
        default=None, description="ID of generated artefact"
    )

    # Buffered remaining concerns from uncertainty resolution.
    # Collected during the resolution round and batch-deduped before
    # creating new uncertainty entities. Each entry is a dict with:
    #   text: str, parent_id: str, affected_claim_ids: list[str], depth: int
    pending_concerns: list[dict] = Field(
        default_factory=list,
        description="Buffered remaining concerns awaiting batch dedup",
    )

    # Status (separate from phase - can pause/abandon at any phase)
    status: str = Field(
        default="active", description="active, paused, completed, abandoned"
    )

    # ── Top-down decomposition fields ─────────────────────────────────
    #
    # These support the unified inquiry architecture (Phase 2 of the
    # decomposition + reflection refactor). On the parent objective:
    # decomposition + sub_objective_ids + combination_rule are populated
    # after DecomposeQuestion runs and SpawnSubObjectives spawns children.
    # On a sub-objective: parent_objective_id + sub_investigation_id are
    # set so the run can traverse upward.
    #
    # All fields default empty so existing databases load cleanly. The
    # decomposition is stored as a serialized dict (model_dump) rather
    # than a typed QuestionDecomposition to avoid coupling entities/ to
    # agents/output_models.py — operations reconstruct the typed form
    # when needed.

    parent_objective_id: Optional[str] = Field(
        default=None,
        description=(
            "On a sub-objective spawned from a decomposition: the parent "
            "objective's entity_id. None for root objectives."
        ),
    )
    sub_investigation_id: Optional[str] = Field(
        default=None,
        description=(
            "On a sub-objective: the 'A'/'B'/'C' tag from the parent's "
            "decomposition. Lets the parent collect verdicts back into "
            "the right slots when combining results."
        ),
    )
    decomposition: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "On a parent objective: the serialized QuestionDecomposition "
            "produced by DecomposeQuestionOperation. Operations that "
            "consume this reconstruct the typed form via "
            "QuestionDecomposition(**objective.decomposition)."
        ),
    )
    sub_objective_ids: list[str] = Field(
        default_factory=list,
        description=(
            "On a parent objective: the entity_ids of sub-objectives "
            "spawned from the decomposition, in decomposition order."
        ),
    )
    combination_rule: Optional[str] = Field(
        default=None,
        description=(
            "On a parent objective: the combination rule from the "
            "decomposition (AND / OR / WEIGHTED_AND / UNION). Used by "
            "the future Combine step to aggregate sub-investigation "
            "verdicts into the question's final answer."
        ),
    )

    # ── Phase 4: reflection state ─────────────────────────────────────
    #
    # On a parent objective only. Reflection is the corrective loop
    # ReflectOnGapsOperation runs after children are scored and combined:
    # if a load-bearing gap remains, it appends new sub-investigations to
    # ``decomposition`` for the orchestrator to spawn. The cap is
    # enforced by the orchestrator (default 1 round) so reflection
    # remains corrective rather than search-like.

    reflection_rounds: int = Field(
        default=0,
        description=(
            "On a parent objective: number of reflection rounds completed "
            "so far. Bumped by ReflectOnGapsOperation each time it adds "
            "new sub-investigations. Compared against the orchestrator's "
            "cap to decide whether to reflect again."
        ),
    )
    reflection_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "On a parent objective: per-round audit trail of reflection "
            "decisions. Each entry: {round, sufficient, gap_description, "
            "added_count, rationale}. Append-only."
        ),
    )

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
        if self.parent_objective_id:
            meta["parent_objective_id"] = self.parent_objective_id
        if self.sub_investigation_id:
            meta["sub_investigation_id"] = self.sub_investigation_id
        if self.decomposition is not None:
            meta["decomposition"] = self.decomposition
        if self.sub_objective_ids:
            meta["sub_objective_ids"] = self.sub_objective_ids
        if self.combination_rule:
            meta["combination_rule"] = self.combination_rule
        if self.reflection_rounds:
            meta["reflection_rounds"] = self.reflection_rounds
        if self.reflection_history:
            meta["reflection_history"] = self.reflection_history
        return meta

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> "Objective":
        """Reconstruct from metadata (legacy support)."""
        return cls(
            entity_id=metadata.get("objective_id", ""),
            objective_id=metadata.get(
                "objective_id", ""
            ),  # Self-referential for objectives
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
            parent_objective_id=metadata.get("parent_objective_id"),
            sub_investigation_id=metadata.get("sub_investigation_id"),
            decomposition=metadata.get("decomposition"),
            sub_objective_ids=metadata.get("sub_objective_ids", []),
            combination_rule=metadata.get("combination_rule"),
            reflection_rounds=metadata.get("reflection_rounds", 0),
            reflection_history=metadata.get("reflection_history", []),
            created_at=datetime.fromisoformat(
                metadata.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                metadata.get("updated_at", datetime.now().isoformat())
            ),
        )
