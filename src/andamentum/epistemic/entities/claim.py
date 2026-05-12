"""Claim Entity - Scoped proposition with stage tracking.

A claim is the core unit of the epistemic system. Progress is
measured by CLAIM PROMOTION, not text volume.

Architecture: Layer 1 (framework-agnostic)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator

from .base import EpistemicEntity
from .prediction import Prediction


class CandidateRecord(BaseModel):
    """One candidate verdict accumulated through the IBE pipeline.

    Populated incrementally by the four IBE operations:
      - EnumerateCandidatesOperation writes id, verdict, description
      - ScoreLovelinessOperation fills loveliness + loveliness_reasoning
      - ScoreLikelinessOperation fills likeliness + likeliness_reasoning
      - SelectBestExplanationOperation marks chosen / runner_up and
        writes the gap fields on the chosen record

    Persisted as part of Claim.integration_candidates so the full
    abductive deliberation trace survives in the database.
    """

    candidate_id: str = Field(
        description="Stable id assigned by enumeration: 'A', 'B', ..."
    )
    verdict: str = Field(
        description=(
            "One of supports / contradicts / insufficient / "
            "supports_refined / contradicts_refined."
        )
    )
    description: str = Field(description="1-2 sentences from the enumeration step.")
    loveliness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    loveliness_reasoning: Optional[str] = Field(default=None)
    likeliness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    likeliness_reasoning: Optional[str] = Field(default=None)
    chosen: bool = Field(default=False)
    runner_up: bool = Field(default=False)
    gap_loveliness: Optional[float] = Field(
        default=None,
        description="Set on the chosen candidate: chosen.loveliness - runner_up.loveliness.",
    )
    gap_likeliness: Optional[float] = Field(
        default=None,
        description="Set on the chosen candidate: chosen.likeliness - runner_up.likeliness.",
    )


class PromotionHistoryEntry(BaseModel):
    """Typed entry in ``Claim.promotion_history``.

    Replaces the previous untyped ``list[dict[str, Any]]`` shape, which
    masked a real bug — ``trace_renderers.py`` was reading
    ``"from_stage"`` / ``"to_stage"`` keys that no writer produced.
    """

    from_stage: "ClaimStage"
    to_stage: "ClaimStage"
    timestamp: datetime
    justification: str


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
    promotion_history: list[PromotionHistoryEntry] = Field(
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
    scrutiny_fingerprint: Optional[str] = Field(
        default=None,
        description=(
            "SHA-256 of the (claim, linked-evidence-set) inputs from the last "
            "scrutiny pass. Identical fingerprint means scrutiny would re-derive "
            "the same verdict, so the operation short-circuits without minting "
            "fresh Uncertainty entities."
        ),
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
    convergence_verdict: Optional[str] = Field(
        default=None,
        description=(
            "Outcome of convergence assessment when convergence_checked is True: "
            "'CONVERGENT', 'WEAKLY_CONVERGENT', 'DIVERGENT', 'PARTIAL', "
            "'SINGLE_DOMAIN', or 'NO_EVIDENCE'. Read by the graph as a "
            "termination signal — a CONVERGENT verdict with no remaining "
            "blocking uncertainties skips the resolve-uncertainties cycle."
        ),
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
    # Prior-round evidence-gap analysis memory. Each entry is one intent
    # ("we need a methodologically-independent angle on the mechanism")
    # that ``epistemic_investigate_claim`` proposed in a previous round.
    # Passed back into the agent on the next round as ``previous_intents``
    # so it can deliberately propose a different angle rather than
    # reshuffling the same lexicon. The agent never sees query strings —
    # only intents — because the routing layer (description-driven
    # dispatch) shapes provider-specific queries from each intent.
    investigation_intents: list[str] = Field(
        default_factory=list,
        description=(
            "Natural-language descriptions of evidence-gap angles proposed "
            "in prior investigation rounds. Used as agent memory to avoid "
            "repeating the same angle round after round."
        ),
    )
    abandoned: bool = Field(
        default=False,
        description="Claim abandoned after exhausting investigation attempts",
    )

    # ── Scrutinise ↔ Resolve oscillation termination ──────────────────
    #
    # Set when the Scrutinise/Resolve cycle cap fires for this claim
    # (graph/nodes.py:SCRUTINY_RESOLVE_CYCLE_CAP). Cycle-capped claims
    # are NOT promoted to SUPPORTED — the inquiry didn't converge, so
    # forcing a verdict via the IBE chain would misrepresent the
    # epistemic state. compute_posterior detects this flag on any
    # active claim and emits terminal_state="oscillation_detected"
    # with posterior=0.5, distinguishing "system reached a verdict"
    # from "system gave up after bounded inquiry".
    #
    # This is downstream of the runtime cap in graph state — the cap
    # bounds work; this field communicates the consequence honestly.
    cycle_capped: bool = Field(
        default=False,
        description=(
            "True if this claim hit the Scrutinise ↔ Resolve cycle cap. "
            "Cycle-capped claims terminate at HYPOTHESIS without IBE; "
            "compute_posterior surfaces them via "
            "terminal_state='oscillation_detected'."
        ),
    )

    # Multi-seed-claim mode: which sub-investigation this claim is the seed for.
    # Set when MultiSeedClaimOperation mints the claim from the parent's
    # decomposition; matches Evidence.sub_investigation_id for linkage.
    sub_investigation_id: Optional[str] = Field(
        default=None,
        description=(
            "When this claim was minted from the parent's decomposition "
            "as the seed for sub-investigation X, this is X. Matches "
            "Evidence.sub_investigation_id for per-claim evidence linkage."
        ),
    )
    persistent_concerns: list[str] = Field(
        default_factory=list,
        description=(
            "Forensic snapshot: entity_ids of the blocking uncertainties "
            "active on this claim at the moment the cycle cap first "
            "fired. Captured once and not overwritten on subsequent "
            "cap firings — the first-firing concerns are the diagnostic "
            "input for deciding whether cluster-dedup or claim "
            "reformulation is the right architectural follow-up."
        ),
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

    # Prediction storage. Phase 6 of the Move-3 plan: previously
    # ``list[dict[str, Any]]``; now a typed ``list[Prediction]`` so
    # consumers (gates.py, render, audit) access fields by attribute
    # rather than via ``dict.get(...)``.
    predictions: list[Prediction] = Field(
        default_factory=list,
        description=(
            "Generated testable predictions, one per call to "
            "GeneratePredictionOperation."
        ),
    )

    # Abductive integration (Peirce + Kahneman).
    # Written by IntegrateEvidenceOperation on the normal path; also written by
    # PromoteAsRefutedOperation when a HYPOTHESIS claim is refute-promoted
    # (contradicts + mechanical confidence, integration LLM skipped). Both
    # writers target the same entity, so the shared write is P5-compatible.
    integrated_assessment: Optional[str] = Field(
        default=None,
        description="Holistic evidence verdict: 'supports', 'contradicts', 'insufficient'",
    )
    integrated_confidence: Optional[float] = Field(
        default=None,
        description="Confidence from abductive integration 0.0-1.0",
    )
    integrated_reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning chain from integration assessment",
    )

    # IBE deliberation trace. Populated by the 4-stage abductive
    # integration pipeline (enumerate → score loveliness → score
    # likeliness → select). Empty for claims that haven't reached
    # integration yet, and for refute-promoted claims (which set
    # integrated_assessment directly without going through IBE).
    integration_candidates: list[CandidateRecord] = Field(
        default_factory=list,
        description=(
            "Full per-candidate IBE record: verdict, description, "
            "loveliness, likeliness, score reasonings, chosen / runner_up "
            "flags, and gap-to-runner-up on the chosen candidate."
        ),
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
        """Record a stage transition in history.

        Canonical writer for ``promotion_history``. Demotions are
        transitions too — ``record_demotion`` delegates here for the
        history append and stage assignment, then layers on its own
        Lakatos / verification-flag cleanup.
        """
        self.promotion_history.append(
            PromotionHistoryEntry(
                from_stage=from_stage,
                to_stage=to_stage,
                timestamp=datetime.now(),
                justification=justification,
            )
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

        # Lakatos: track the regression. Stage assignment and history
        # append are delegated to record_promotion (the canonical writer).
        self.record_modification()
        self.record_promotion(old_stage, target_stage, justification)

        # Peirce cycling: reset verification state so the claim can be
        # re-evaluated from its new stage. All verification tracks and
        # routing must re-run.
        if target_stage in [ClaimStage.HYPOTHESIS, ClaimStage.SUPPORTED]:
            self.scrutiny_verdict = None
            self.adversarial_checked = False
            self.convergence_checked = False
            self.convergence_verdict = None
            self.deductive_checked = False
            self.computational_checked = False
            self.contrastive_checked = False
            self.consistency_checked = False
            self.integrated_assessment = None
            self.integrated_confidence = None
            self.integrated_reasoning = None
            self.integration_candidates = []

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
            "scrutiny_fingerprint": self.scrutiny_fingerprint,
            "adversarial_checked": self.adversarial_checked,
            "convergence_checked": self.convergence_checked,
            "convergence_verdict": self.convergence_verdict,
            "deductive_checked": self.deductive_checked,
            "computational_checked": self.computational_checked,
            "contrastive_checked": self.contrastive_checked,
            "consistency_checked": self.consistency_checked,
            "argument_analyzed": self.argument_analyzed,
            "predictions_generated": self.predictions_generated,
            "decision_recorded": self.decision_recorded,
            "predictions": [p.model_dump() for p in self.predictions],
            "supersedes_id": self.supersedes_id,
            "superseded_by_id": self.superseded_by_id,
            "modification_count": self.modification_count,
            "investigation_count": self.investigation_count,
            "investigation_intents": self.investigation_intents,
            "abandoned": self.abandoned,
            "cycle_capped": self.cycle_capped,
            "persistent_concerns": self.persistent_concerns,
            "sub_investigation_id": self.sub_investigation_id,
            "integrated_assessment": self.integrated_assessment,
            "integrated_confidence": self.integrated_confidence,
            "integration_candidates": [
                c.model_dump() for c in self.integration_candidates
            ],
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
            investigation_intents=metadata.get("investigation_intents", []),
            abandoned=metadata.get("abandoned", False),
            cycle_capped=metadata.get("cycle_capped", False),
            persistent_concerns=metadata.get("persistent_concerns", []),
            sub_investigation_id=metadata.get("sub_investigation_id"),
            scrutiny_verdict=metadata.get("scrutiny_verdict"),
            scrutiny_fingerprint=metadata.get("scrutiny_fingerprint"),
            adversarial_checked=metadata.get("adversarial_checked", False),
            convergence_checked=metadata.get("convergence_checked", False),
            convergence_verdict=metadata.get("convergence_verdict"),
            deductive_checked=metadata.get("deductive_checked", False),
            computational_checked=metadata.get("computational_checked", False),
            contrastive_checked=metadata.get("contrastive_checked", False),
            consistency_checked=metadata.get("consistency_checked", False),
            confidence_score=metadata.get("confidence_score"),
            adversarial_balance=metadata.get("adversarial_balance"),
            argument_analyzed=metadata.get("argument_analyzed", False),
            predictions_generated=metadata.get("predictions_generated", False),
            decision_recorded=metadata.get("decision_recorded", False),
            predictions=[
                Prediction.from_dict(p) for p in metadata.get("predictions", [])
            ],
            integrated_assessment=metadata.get("integrated_assessment"),
            integrated_confidence=metadata.get("integrated_confidence"),
            integrated_reasoning=metadata.get("integrated_reasoning"),
            integration_candidates=[
                CandidateRecord(**c) for c in metadata.get("integration_candidates", [])
            ],
            created_at=datetime.fromisoformat(
                metadata.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                metadata.get("updated_at", datetime.now().isoformat())
            ),
        )
