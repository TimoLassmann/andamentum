"""Reasoning Trace - Data structures for epistemic reasoning visualization.

This module provides data structures for collecting and representing the
reasoning trace of an epistemic investigation. The trace shows HOW the
system arrived at its conclusions, linking outcomes to sources and decisions.

Architecture: Layer 1 (Libraries) - framework-agnostic, no model calls
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .primitives import Claim, Evidence, Uncertainty


@dataclass
class TraceStep:
    """Single step in the reasoning trace.

    Represents one operation in the epistemic pipeline with its inputs,
    outputs, timing, and success status.
    """

    timestamp: datetime
    operation: (
        str  # PLAN, COLLECT, EXTRACT, PROPOSE, SCRUTINISE, PROMOTE, FREEZE, COMPILE
    )
    description: str  # Human-readable description of what was done
    inputs: List[str] = field(default_factory=list)  # Brief input summary
    outputs: List[str] = field(
        default_factory=list
    )  # What was produced (IDs, summaries)
    duration_ms: Optional[int] = None  # How long it took
    success: bool = True
    error: Optional[str] = None  # Error message if failed

    @property
    def operation_display(self) -> str:
        """Human-friendly operation name."""
        mapping = {
            "plan_task": "PLAN",
            "collect_evidence": "COLLECT",
            "extract_evidence": "EXTRACT",
            "propose_claims": "PROPOSE",
            "scrutinise_claim": "SCRUTINISE",
            "promote_claim": "PROMOTE",
            "freeze_snapshot": "FREEZE",
            "synthesize_report": "SYNTHESIZE",
        }
        return mapping.get(self.operation.lower(), self.operation.upper())


@dataclass
class ClaimLineage:
    """Lineage for a single claim.

    Shows the evidence supporting a claim, the uncertainties challenging it,
    and the promotion path it has taken through stages.
    """

    claim_id: str
    statement: str
    scope: str
    stage: str  # Current stage (hypothesis, supported, etc.)
    assumptions: List[str] = field(default_factory=list)

    # Supporting evidence
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {evidence_id, source_type, source_ref, extracted_content_preview}

    # Challenging uncertainties
    uncertainties: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {uncertainty_id, uncertainty_type, description}

    # Promotion history
    promotion_path: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {from_stage, to_stage, justification, timestamp}

    @classmethod
    def from_claim(
        cls,
        claim: "Claim",
        evidence_list: List["Evidence"],
        uncertainty_list: List["Uncertainty"],
    ) -> "ClaimLineage":
        """Build ClaimLineage from a Claim and related objects.

        Args:
            claim: The claim to build lineage for
            evidence_list: All evidence items (will be filtered to claim.evidence_ids)
            uncertainty_list: All uncertainties (will be filtered to those affecting this claim)

        Returns:
            ClaimLineage with populated evidence and uncertainty links
        """
        # Filter evidence to those supporting this claim
        supporting = []
        evidence_by_id = {e.evidence_id: e for e in evidence_list}
        for eid in claim.evidence_ids:
            if eid in evidence_by_id:
                e = evidence_by_id[eid]
                supporting.append(
                    {
                        "evidence_id": e.evidence_id,
                        "source_type": e.source_type,
                        "source_ref": e.source_ref,
                        "extracted_content_preview": e.extracted_content[:200]
                        if e.extracted_content
                        else "",
                    }
                )

        # Filter uncertainties to those affecting this claim
        affecting = []
        for u in uncertainty_list:
            if claim.claim_id in u.affected_claim_ids:
                affecting.append(
                    {
                        "uncertainty_id": u.uncertainty_id,
                        "uncertainty_type": u.uncertainty_type.value
                        if hasattr(u.uncertainty_type, "value")
                        else str(u.uncertainty_type),
                        "description": u.description,
                        "is_resolved": u.is_resolved,
                    }
                )

        return cls(
            claim_id=claim.claim_id,
            statement=claim.statement,
            scope=claim.scope,
            stage=claim.stage.value
            if hasattr(claim.stage, "value")
            else str(claim.stage),
            assumptions=list(claim.assumptions),
            supporting_evidence=supporting,
            uncertainties=affecting,
            promotion_path=list(claim.promotion_history)
            if claim.promotion_history
            else [],
        )


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for visualization.

    Collects all data needed to render trace visualizations:
    - Timeline: chronological steps
    - Flow: pipeline DAG structure
    - Claims: per-claim evidence/uncertainty lineage
    """

    question: str  # Original research question
    objective_id: str

    # Chronological operation steps
    steps: List[TraceStep] = field(default_factory=list)

    # Per-claim lineage (for claims view)
    claim_lineages: List[ClaimLineage] = field(default_factory=list)

    # Summary metrics
    total_duration_ms: int = 0
    evidence_count: int = 0
    claim_count: int = 0
    uncertainty_count: int = 0

    # Timing boundaries
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def add_step(self, step: TraceStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)
        if step.duration_ms:
            self.total_duration_ms += step.duration_ms

        # Update timing boundaries
        if self.started_at is None or step.timestamp < self.started_at:
            self.started_at = step.timestamp

    def finalize(
        self,
        claims: List["Claim"],
        evidence: List["Evidence"],
        uncertainties: List["Uncertainty"],
    ) -> None:
        """Finalize the trace with claim lineages and metrics.

        Call this after all steps have been added to build claim lineages
        and calculate final metrics.

        Args:
            claims: All claims for the objective
            evidence: All evidence for the objective
            uncertainties: All uncertainties for the objective
        """
        self.claim_count = len(claims)
        self.evidence_count = len(evidence)
        self.uncertainty_count = len(uncertainties)

        # Build claim lineages
        self.claim_lineages = [
            ClaimLineage.from_claim(c, evidence, uncertainties) for c in claims
        ]

        # Set completion time to last step timestamp
        if self.steps:
            last_step = max(self.steps, key=lambda s: s.timestamp)
            self.completed_at = last_step.timestamp

    def get_steps_by_operation(self, operation: str) -> List[TraceStep]:
        """Get all steps of a specific operation type."""
        return [s for s in self.steps if s.operation.lower() == operation.lower()]

    def get_grouped_steps(self) -> Dict[str, List[TraceStep]]:
        """Group steps by operation type for flow visualization.

        Returns steps grouped by operation, maintaining chronological order
        within each group.
        """
        groups: Dict[str, List[TraceStep]] = {}
        for step in self.steps:
            op = step.operation_display
            if op not in groups:
                groups[op] = []
            groups[op].append(step)
        return groups

    @property
    def success_rate(self) -> float:
        """Percentage of successful steps."""
        if not self.steps:
            return 0.0
        successful = sum(1 for s in self.steps if s.success)
        return (successful / len(self.steps)) * 100

    @property
    def failed_steps(self) -> List[TraceStep]:
        """Get all failed steps."""
        return [s for s in self.steps if not s.success]
