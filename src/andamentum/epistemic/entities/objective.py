"""Objective Entity - Top-level research objective.

The objective defines what we're trying to learn or produce.
All epistemic work is scoped to an objective.

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import Field, model_validator

from .base import EpistemicEntity
from .decomposition import Decomposition


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
    # Multi-seed-claim mode (v0.3): when the user asks for decomposition
    # (--decompose CLI flag), DecomposeQuestionOperation populates
    # ``decomposition`` and ``combination_rule`` on this Objective.
    # The graph's CreateClaims node then routes to MultiSeedClaim, which
    # mints N Claims (one per sub-investigation) on the SAME Objective —
    # no child Objectives are spawned. The per-sub-investigation tag
    # lives on the minted Claim's ``sub_investigation_id`` field, not
    # here.
    #
    # Phase 6 of the Move-3 plan: this field used to be a raw
    # ``dict[str, Any]`` to avoid coupling entities/ to
    # agents/output_models.py. It's now a typed ``Decomposition``
    # model (defined in entities/decomposition.py) so consumers
    # access fields by name rather than via ``dict.get(...)``.
    # Bug C from the post-audit-2 fix queue was the divergent-lookup
    # failure mode this typing prevents.

    decomposition: Optional[Decomposition] = Field(
        default=None,
        description=(
            "Top-down decomposition produced by DecomposeQuestionOperation. "
            "After CombineClaimVerdicts runs, ``combined_verdict`` is "
            "populated in-place; FreezeSnapshot then promotes that onto "
            "Snapshot.combined_verdict."
        ),
    )

    @model_validator(mode="after")
    def _check_seed_modes_exclusive(self) -> "Objective":
        """Refuse Objectives that try to be both single-seed and multi-seed.

        ``CreateClaims`` (graph/nodes.py) branches on ``claim_to_verify``
        first, then ``decomposition``, then falls through to
        ``ProposeClaims``. Setting both fields silently picks single-seed
        and discards the decomposition — a footgun for programmatic
        consumers (the SciFact harness hit it). Refuse loudly at
        construction time so the precedence rule is documented in the
        error rather than buried in node code.
        """
        if self.claim_to_verify and self.decomposition is not None:
            raise ValueError(
                "Objective cannot have both claim_to_verify and "
                "decomposition set — they are mutually exclusive seed "
                "modes. CreateClaims branches on claim_to_verify first, "
                "so decomposition would be silently ignored. Pick one: "
                "single-seed (claim_to_verify) for known-claim "
                "verification, or multi-seed (decomposition) for "
                "decomposed research."
            )
        return self

    @property
    def pending_concerns_count(self) -> int:
        """Number of buffered concerns — used by pattern matching."""
        return len(self.pending_concerns)

    def is_verification_task(self) -> bool:
        """True when this Objective is verifying specific claim(s).

        Two cases qualify:

        * ``claim_to_verify`` set: single-seed mode (``SeedClaimOperation``
          mints one claim from the user's text).
        * ``decomposition`` set: multi-seed mode
          (``MultiSeedClaimOperation`` mints N claims, one per
          sub-investigation).

        Both are binary verification by construction — each minted
        claim has a defined yes/no answer regardless of how the
        Objective itself was classified by the LLM. Routing and
        posterior eligibility key off this method to force the
        verificatory profile in these modes, preventing classifier
        misclassifications (e.g. "explanatory" on a SciFact-style
        declarative claim) from cascading into wrong routing.
        """
        if self.claim_to_verify:
            return True
        if self.decomposition is not None:
            if self.decomposition.sub_investigations:
                return True
        return False

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
        if self.decomposition is not None:
            meta["decomposition"] = self.decomposition.model_dump()
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
            decomposition=Decomposition.from_dict(metadata.get("decomposition")),
            created_at=datetime.fromisoformat(
                metadata.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                metadata.get("updated_at", datetime.now().isoformat())
            ),
        )
