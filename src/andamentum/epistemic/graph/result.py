"""Result type for the epistemic graph End node.

This is the value inside End(EpistemicResult(...)) — what the graph
returns when it terminates. A Pydantic model, per the dialect's Result
convention (the ``End[T]`` payload is never a bare dataclass).

Architecture: Layer 1 (framework-agnostic, typed record)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .quarantine import OperationLogEntry, QuarantineRecord


class EpistemicResult(BaseModel):
    """Final output of an epistemic graph run."""

    objective_id: str
    status: str  # "complete", "partial", "no_claims", "retrieval_failed"
    successful: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)
    operations_log: list[OperationLogEntry] = Field(default_factory=list)

    # Termination reason: "complete", "no_claims", "partial", "retrieval_failed"
    termination_reason: str = ""

    quarantined: list[QuarantineRecord] = Field(default_factory=list)

    retrieval_failed: bool = False
