"""Layer-1 record types for the epistemic graph run.

``QuarantineRecord`` — when an operation raises, the central runner
records one so downstream nodes can skip the entity and the final report
can surface the failure. Fail-loud: no silent degradation.

``OperationLogEntry`` — the per-operation trace line the runner appends
for the progress callback and final stats.

Both are typed boundary schemas (Law 7): they ride on ``EpistemicResult``,
the graph's ``End`` payload, so they must not be untyped dicts.

Architecture: Layer 1 (framework-agnostic, pure records)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class QuarantineRecord(BaseModel):
    """Records that an entity was quarantined because an operation raised."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    entity_type: str
    operation: str
    exception_type: str
    message: str


class OperationLogEntry(BaseModel):
    """One operation execution, recorded for tracing and final stats."""

    operation: str
    entity_id: str
    success: bool
    message: str
