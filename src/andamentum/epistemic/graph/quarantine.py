"""Quarantine record for entities whose operations failed.

When an operation raises, the central runner records a QuarantineRecord
so downstream nodes can skip the entity and the final report can surface
the failure to the user. Fail-loud: no silent degradation.

Architecture: Layer 1 (framework-agnostic, pure dataclass)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuarantineRecord:
    """Records that an entity was quarantined because an operation raised."""

    entity_id: str
    entity_type: str
    operation: str
    exception_type: str
    message: str
