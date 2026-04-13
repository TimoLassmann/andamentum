"""Typed result models for epistemic CLI handlers.

All handler functions return typed Pydantic models instead of Dict[str, Any].
This ensures type safety at component boundaries and enables IDE support.

Architecture: Layer 4 (Application)

Per TYPING_CONVENTIONS.md: Never use Dict[str, Any] for data passed between components.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from .primitives import (
    Claim,
    Evidence,
    Uncertainty,
    Decision,
    AdversarialEvidence,
    ComputationalEvidence,
    ConvergentEvidence,
    TemporalEvidence,
    DeductiveEvidence,
    EpistemicEvent,
)
from .trace import ReasoningTrace


class BaseResult(BaseModel):
    """Base result with success/error handling."""

    success: bool = Field(description="Whether the operation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class InitResult(BaseResult):
    """Result from handle_init."""

    objective_id: Optional[str] = Field(default=None, description="Created objective ID")
    database_name: Optional[str] = Field(default=None, description="Database name")
    description: Optional[str] = Field(default=None, description="Objective description")


class RunStats(BaseModel):
    """Statistics from an orchestrator run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workitems_executed_this_run: int = Field(default=0)
    total_workitems: int = Field(default=0)
    workitems_by_status: Dict[str, int] = Field(default_factory=dict)
    claims_by_stage: Dict[str, int] = Field(default_factory=dict)
    evidence_count: int = Field(default=0)
    uncertainty_count: int = Field(default=0)
    synthesis: Optional[Dict[str, Any]] = Field(default=None, description="Synthesis results")
    reasoning_trace: Optional[ReasoningTrace] = Field(default=None)


class RunResult(BaseResult):
    """Result from handle_run."""

    objective_id: Optional[str] = Field(default=None)
    stats: Optional[RunStats] = Field(default=None)


class ObjectiveStats(BaseModel):
    """Statistics for an objective from workitem_manager.get_objective_stats()."""

    objective_id: str
    evidence_count: int = Field(default=0)
    claims_by_stage: Dict[str, int] = Field(default_factory=dict)
    uncertainties_unresolved: int = Field(default=0)
    uncertainties_resolved: int = Field(default=0)
    decisions_active: int = Field(default=0)
    decisions_reversed: int = Field(default=0)
    workitems_queued: int = Field(default=0)
    workitems_done: int = Field(default=0)
    workitems_failed: int = Field(default=0)
    snapshots: int = Field(default=0)
    artefacts: int = Field(default=0)


class StatusResult(BaseResult):
    """Result from handle_status."""

    objective_id: Optional[str] = Field(default=None)
    stats: Optional[ObjectiveStats] = Field(default=None, description="Objective statistics")


class DebateResult(BaseResult):
    """Result from handle_debate."""

    objective_id: Optional[str] = Field(default=None)
    total_claims: int = Field(default=0)
    supported: int = Field(default=0, description="Claims with SUPPORTED verdict")
    contested: int = Field(default=0, description="Claims with CONTESTED verdict")
    challenged: int = Field(default=0, description="Claims with CHALLENGED verdict")
    refuted: int = Field(default=0, description="Claims with REFUTED verdict")
    overall_balance: float = Field(default=0.0, description="Overall adversarial balance (0-1)")


class ClaimsResult(BaseResult):
    """Result from handle_claims."""

    objective_id: Optional[str] = Field(default=None)
    claims: List[Claim] = Field(default_factory=list)
    count: int = Field(default=0)


class VerificationEvidence(BaseModel):
    """Verification evidence grouped by type."""

    adversarial: List[AdversarialEvidence] = Field(default_factory=list)
    computational: List[ComputationalEvidence] = Field(default_factory=list)
    convergent: List[ConvergentEvidence] = Field(default_factory=list)
    temporal: List[TemporalEvidence] = Field(default_factory=list)
    deductive: List[DeductiveEvidence] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Total count across all types."""
        return (
            len(self.adversarial)
            + len(self.computational)
            + len(self.convergent)
            + len(self.temporal)
            + len(self.deductive)
        )


class EvidenceResult(BaseResult):
    """Result from handle_evidence."""

    objective_id: Optional[str] = Field(default=None)
    evidence: List[Evidence] = Field(default_factory=list)
    count: int = Field(default=0)
    verification_evidence: Optional[VerificationEvidence] = Field(default=None)
    verification_count: int = Field(default=0)


class UncertaintiesResult(BaseResult):
    """Result from handle_uncertainties."""

    objective_id: Optional[str] = Field(default=None)
    uncertainties: List[Uncertainty] = Field(default_factory=list)
    count: int = Field(default=0)


class DecisionsResult(BaseResult):
    """Result from handle_decisions."""

    objective_id: Optional[str] = Field(default=None)
    decisions: List[Decision] = Field(default_factory=list)
    count: int = Field(default=0)


class LogResult(BaseResult):
    """Result from handle_log."""

    events: List[EpistemicEvent] = Field(default_factory=list, description="Log events")
    count: int = Field(default=0)


class AskResult(BaseResult):
    """Result from handle_ask - the primary research interface."""

    question: str = Field(default="", description="The original research question")
    project_name: str = Field(default="", description="Project/database name")
    claims: List[Claim] = Field(default_factory=list, description="Research claims found")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence")
    uncertainties: List[Uncertainty] = Field(default_factory=list, description="Open uncertainties")
    stats: Optional[RunStats] = Field(default=None, description="Execution statistics")
    kept: bool = Field(default=False, description="Whether project was kept")
    artefact_content: Optional[str] = Field(default=None, description="Full artefact markdown — the canonical output")


class CleanupResult(BaseResult):
    """Result from handle_cleanup."""

    deleted: int = Field(default=0, description="Number of databases deleted")
    freed_bytes: int = Field(default=0, description="Bytes freed")
    dry_run: bool = Field(default=False, description="Whether this was a dry run")

    @property
    def freed_mb(self) -> float:
        """Freed space in megabytes."""
        return self.freed_bytes / (1024 * 1024)


class ReportResult(BaseResult):
    """Result from handle_report."""

    database_name: str = Field(default="", description="Database name")
    output_path: str = Field(default="", description="Path to generated HTML file")
    claims_count: int = Field(default=0, description="Number of claims in report")
    evidence_count: int = Field(default=0, description="Number of evidence items in report")
    uncertainties_count: int = Field(default=0, description="Number of uncertainties in report")
