"""Typed result of ``ask`` / ``verify``.

The functions return a typed Pydantic model instead of ``Dict[str, Any]``;
the command prints the same model as JSON with ``--json``.

Architecture: Layer 4 (Application)
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .entities.verdict import EpistemicVerdict
from .primitives import Claim, Evidence, Uncertainty
from .trace import ReasoningTrace


class BaseResult(BaseModel):
    """Base result with success/error handling."""

    success: bool = Field(description="Whether the operation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RunStats(BaseModel):
    """Statistics from an orchestrator run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workitems_executed_this_run: int = Field(default=0)
    total_workitems: int = Field(default=0)
    workitems_by_status: Dict[str, int] = Field(default_factory=dict)
    claims_by_stage: Dict[str, int] = Field(default_factory=dict)
    evidence_count: int = Field(default=0)
    uncertainty_count: int = Field(default=0)
    synthesis: Optional[Dict[str, Any]] = Field(
        default=None, description="Synthesis results"
    )
    reasoning_trace: Optional[ReasoningTrace] = Field(default=None)


class AskResult(BaseResult):
    """Result of ``ask`` / ``verify``: findings, claims, evidence and the verdict."""

    question: str = Field(default="", description="The original question or claim")
    database: str = Field(
        default="",
        description="Path of the database file the run wrote; empty when it was temporary",
    )
    claims: List[Claim] = Field(
        default_factory=list, description="Research claims found"
    )
    evidence: List[Evidence] = Field(
        default_factory=list, description="Supporting evidence"
    )
    uncertainties: List[Uncertainty] = Field(
        default_factory=list, description="Open uncertainties"
    )
    stats: Optional[RunStats] = Field(default=None, description="Execution statistics")
    artefact_content: Optional[str] = Field(
        default=None, description="Full artefact markdown — the canonical output"
    )
    verdict: Optional[EpistemicVerdict] = Field(
        default=None,
        description=(
            "The run's single decision record — direction, posterior, "
            "provenance, per-claim breakdown. A caller gets the verdict "
            "here without parsing ``artefact_content``."
        ),
    )
