"""Claim Entity - Scoped proposition with stage tracking.

A claim is the core unit of the epistemic system. Progress is
measured by CLAIM PROMOTION, not text volume.

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import Field, model_validator

from .base import EpistemicEntity


class ClaimStage(str, Enum):
    """Claim lifecycle stages with increasing confidence requirements.

    Claims advance through these stages via explicit promotion.
    Each stage has deterministic gate requirements (see gates.py).
    """

    HYPOTHESIS = "hypothesis"  # Initial proposal, no evidence required
    SUPPORTED = "supported"  # >=1 evidence + uncertainties listed + skeptic review
    PROVISIONAL = "provisional"  # >=2 evidence + skeptic review passed
    ROBUST = "robust"  # Independent evidence lines + counterevidence addressed
    ACTIONABLE = "actionable"  # Decision criteria met + "what would change mind"


class Claim(EpistemicEntity):
    """Scoped proposition with stage tracking and degeneracy detection.

    State fields for pattern matching:
    - stage: Current lifecycle stage
    - scrutiny_verdict: Result of skeptic review (None, "pass", "fail", "needs_resolution")
    - adversarial_checked, convergence_checked, deductive_checked, computational_checked:
      Verification track completion flags
    - evidence_count: Denormalized count for pattern filtering
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "claim_id" in data and "entity_id" not in data:
            data["entity_id"] = data.pop("claim_id")
        return data

    entity_type: str = "claim"  # type: ignore[assignment]

    # Core fields
    statement: str = Field(description="The claim in controlled language")
    scope: str = Field(
        default="General", description="Under what conditions this claim applies"
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Explicit assumptions"
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="Links to supporting evidence"
    )
    uncertainty_ids: list[str] = Field(
        default_factory=list, description="Links to associated uncertainties"
    )

    # Stage tracking
    stage: ClaimStage = Field(default=ClaimStage.HYPOTHESIS)
    promotion_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Record of stage transitions: [{from, to, timestamp, justification}]",
    )

    # Versioning for append-only demotion
    supersedes_id: Optional[str] = Field(
        default=None, description="ID of claim this version supersedes"
    )
    superseded_by_id: Optional[str] = Field(
        default=None, description="ID of claim that superseded this one"
    )

    # Degeneracy detection (Lakatos)
    # DEGEN_001: modification_count > 3 triggers review
    # DEGEN_003: 3+ modifications in 24h window blocks promotion
    modification_count: int = Field(
        default=0, description="Number of times claim has been modified"
    )
    modification_timestamps: list[str] = Field(
        default_factory=list, description="ISO timestamps of each modification"
    )

    # State fields for pattern matching
    scrutiny_verdict: Optional[str] = Field(
        default=None,
        description="Result of scrutiny: None, 'pass', 'fail', or 'needs_resolution'",
    )

    # Adversarial balance (computed by adversarial_balance.py)
    adversarial_balance: Optional[float] = Field(
        default=None,
        description="Adversarial balance score 0.0-1.0 from adversarial search (supporting / total weight)",
    )

    # Verification track state
    adversarial_checked: bool = Field(
        default=False, description="Adversarial search complete"
    )
    convergence_checked: bool = Field(
        default=False, description="Cross-domain convergence assessed"
    )
    deductive_checked: bool = Field(
        default=False, description="Deductive validation complete"
    )
    computational_checked: bool = Field(
        default=False, description="Computational verification complete (if applicable)"
    )
    contrastive_checked: bool = Field(
        default=False, description="Contrastive evaluation completed"
    )
    consistency_checked: bool = Field(
        default=False, description="Cross-claim consistency checked"
    )
    routing_applied: bool = Field(
        default=False,
        description="Whether routing defaults have been applied for this claim",
    )
    saturated: bool = Field(
        default=False,
        description="Whether investigation has stopped producing new information",
    )

    # Denormalized for pattern matching (updated on save via model_post_init)
    evidence_count: int = Field(
        default=0, description="Count of evidence_ids for pattern filtering"
    )

    # Confidence score (set by PromoteClaimOperation after successful promotion)
    confidence_score: Optional[float] = Field(
        default=None,
        description="Numeric confidence 0.0-1.0 derived from stage + evidence quality",
    )

    # Investigation cycle tracking (Peirce inquiry cycling)
    investigation_count: int = Field(
        default=0,
        description="Number of investigation cycles triggered by scrutiny doubt",
    )
    abandoned: bool = Field(
        default=False,
        description="Claim abandoned after exhausting investigation attempts",
    )

    # TMS: Revalidation flag
    needs_revalidation: bool = Field(
        default=False, description="Re-check current stage gate after evidence change"
    )

    # Operation completion flags
    argument_analyzed: bool = Field(
        default=False, description="Argument structure analysis complete"
    )
    predictions_generated: bool = Field(
        default=False, description="Testable predictions generated from claim"
    )
    decision_recorded: bool = Field(
        default=False, description="Decision recorded for this actionable claim"
    )

    # Prediction storage
    predictions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Generated testable predictions: [{statement, type, time_horizon, specificity}]",
    )

    def model_post_init(self, __context: Any) -> None:
        """Update denormalized fields after initialization."""
        self.evidence_count = len(self.evidence_ids)

    def record_modification(self) -> None:
        """Record a modification for degeneracy detection."""
        self.modification_count += 1
        self.modification_timestamps.append(datetime.now().isoformat())
        self.touch()

    def record_promotion(
        self, from_stage: ClaimStage, to_stage: ClaimStage, justification: str
    ) -> None:
        """Record a stage promotion in history."""
        self.promotion_history.append(
            {
                "from": from_stage.value,
                "to": to_stage.value,
                "timestamp": datetime.now().isoformat(),
                "justification": justification,
            }
        )
        self.stage = to_stage
        self.touch()

    def record_demotion(self, target_stage: ClaimStage, justification: str) -> None:
        """Record a stage demotion with full state cleanup.

        Single source of truth for all demotion side effects:
        - Stage regression
        - Modification tracking (Lakatos degeneracy detection)
        - Promotion history
        - Verification flag reset for Peirce cycling

        Both DemoteClaimOperation and RevalidateClaimOperation call this
        to ensure consistent behavior regardless of demotion cause.
        """
        old_stage = self.stage
        self.stage = target_stage

        # Lakatos: track the regression
        self.record_modification()

        # Record in promotion history (demotions are recorded too)
        self.promotion_history.append(
            {
                "from": old_stage.value,
                "to": target_stage.value,
                "timestamp": datetime.now().isoformat(),
                "justification": justification,
            }
        )

        # Peirce cycling: reset verification state so the claim can be
        # re-evaluated from its new stage. All verification tracks and
        # routing must re-run.
        if target_stage in [ClaimStage.HYPOTHESIS, ClaimStage.SUPPORTED]:
            self.scrutiny_verdict = None
            self.adversarial_checked = False
            self.convergence_checked = False
            self.deductive_checked = False
            self.computational_checked = False
            self.contrastive_checked = False
            self.consistency_checked = False
            self.routing_applied = False
            self.saturated = False

    def _extra_metadata(self) -> dict[str, Any]:
        """Add claim-specific metadata for filtering."""
        meta: dict[str, Any] = {
            "statement": self.statement,
            "scope": self.scope,
            "assumptions": self.assumptions,
            "claim_stage": self.stage.value,
            "stage": self.stage.value,  # Alias for pattern matching
            "evidence_ids": self.evidence_ids,
            "uncertainty_ids": self.uncertainty_ids,
            "evidence_count": self.evidence_count,
            "scrutiny_verdict": self.scrutiny_verdict,
            "adversarial_checked": self.adversarial_checked,
            "convergence_checked": self.convergence_checked,
            "deductive_checked": self.deductive_checked,
            "computational_checked": self.computational_checked,
            "contrastive_checked": self.contrastive_checked,
            "consistency_checked": self.consistency_checked,
            "routing_applied": self.routing_applied,
            "saturated": self.saturated,
            "argument_analyzed": self.argument_analyzed,
            "predictions_generated": self.predictions_generated,
            "decision_recorded": self.decision_recorded,
            "predictions": self.predictions,
            "supersedes_id": self.supersedes_id,
            "superseded_by_id": self.superseded_by_id,
            "modification_count": self.modification_count,
            "investigation_count": self.investigation_count,
            "abandoned": self.abandoned,
            "needs_revalidation": self.needs_revalidation,
        }
        if self.confidence_score is not None:
            meta["confidence_score"] = self.confidence_score
        if self.adversarial_balance is not None:
            meta["adversarial_balance"] = self.adversarial_balance
        return meta

    @property
    def claim_id(self) -> str:
        """Backward-compatible alias for entity_id."""
        return self.entity_id

    @classmethod
    def from_metadata(
        cls, meta: dict[str, Any], statement_override: Optional[str] = None
    ) -> "Claim":
        """Reconstruct Claim from metadata dict (legacy API)."""
        content = statement_override or meta.get("statement", "")
        return cls._from_metadata(content, meta)

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> "Claim":
        """Reconstruct from metadata (legacy support)."""
        # Handle stage field (may be claim_stage in legacy)
        stage_value = metadata.get("stage") or metadata.get("claim_stage", "hypothesis")
        stage = ClaimStage(stage_value)

        return cls(
            entity_id=metadata.get("claim_id", ""),
            objective_id=metadata.get("objective_id", ""),
            statement=metadata.get("statement", content),
            scope=metadata.get("scope", "General"),
            assumptions=metadata.get("assumptions", []),
            evidence_ids=metadata.get("evidence_ids", []),
            uncertainty_ids=metadata.get("uncertainty_ids", []),
            stage=stage,
            promotion_history=metadata.get("promotion_history", []),
            supersedes_id=metadata.get("supersedes_id"),
            superseded_by_id=metadata.get("superseded_by_id"),
            modification_count=metadata.get("modification_count", 0),
            modification_timestamps=metadata.get("modification_timestamps", []),
            investigation_count=metadata.get("investigation_count", 0),
            abandoned=metadata.get("abandoned", False),
            needs_revalidation=metadata.get("needs_revalidation", False),
            scrutiny_verdict=metadata.get("scrutiny_verdict"),
            adversarial_checked=metadata.get("adversarial_checked", False),
            convergence_checked=metadata.get("convergence_checked", False),
            deductive_checked=metadata.get("deductive_checked", False),
            computational_checked=metadata.get("computational_checked", False),
            contrastive_checked=metadata.get("contrastive_checked", False),
            consistency_checked=metadata.get("consistency_checked", False),
            routing_applied=metadata.get("routing_applied", False),
            saturated=metadata.get("saturated", False),
            confidence_score=metadata.get("confidence_score"),
            adversarial_balance=metadata.get("adversarial_balance"),
            argument_analyzed=metadata.get("argument_analyzed", False),
            predictions_generated=metadata.get("predictions_generated", False),
            decision_recorded=metadata.get("decision_recorded", False),
            predictions=metadata.get("predictions", []),
            created_at=datetime.fromisoformat(
                metadata.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                metadata.get("updated_at", datetime.now().isoformat())
            ),
        )
