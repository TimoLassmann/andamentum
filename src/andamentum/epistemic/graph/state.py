"""Mutable state for the epistemic graph.

Passed to every node via ctx.state. Nodes mutate it in place.

This is the single source of truth for pipeline progress within
one graph execution. Unlike the pattern scheduler (which queries
entity state from the repo), the graph state tracks what has been
DONE, not what entities EXIST.

Architecture: Layer 1 (framework-agnostic, pure dataclass)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .quarantine import QuarantineRecord


@dataclass
class EpistemicGraphState:
    """Shared mutable state for a single epistemic inquiry.

    Fields are grouped by pipeline phase. Nodes read and write
    these directly via ctx.state.
    """

    # ── Objective ────────────────────────────────────────────────
    objective_id: str = ""
    question: str = ""
    question_type: str | None = None
    skip_preplanning: bool = False

    # ── Evidence collection ─────────────────────────────────────
    evidence_extracted: bool = False

    # ── Claim creation ──────────────────────────────────────────
    claims_created: bool = False
    claim_ids: list[str] = field(default_factory=list)

    # ── Per-claim tracking ──────────────────────────────────────
    # Investigation cycle counts (Peirce cycling cap)
    investigation_counts: dict[str, int] = field(default_factory=dict)

    # Claims that have completed verification + integration
    verification_done: set[str] = field(default_factory=set)

    # Claims that have been abandoned or reached terminal stage
    terminal_claims: set[str] = field(default_factory=set)

    # ── Flow control (graph-managed, not on entities) ───────────
    # Claims needing re-scrutiny after uncertainty resolution or
    # investigation. Scrutinize node checks this IN ADDITION to
    # claims with scrutiny_verdict=None.
    claims_needing_rescrutiny: set[str] = field(default_factory=set)

    # Claims needing TMS revalidation after evidence changes.
    # Populated by graph nodes, consumed by _run_tms_sweep.
    claims_needing_tms: set[str] = field(default_factory=set)

    # ── Operation trace ─────────────────────────────────────────
    # Lightweight log for the progress callback and final stats
    operations_log: list[dict[str, Any]] = field(default_factory=list)
    successful: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    # Entities whose operations raised. Downstream nodes must skip these.
    quarantined: list[QuarantineRecord] = field(default_factory=list)
    _quarantined_ids: set[str] = field(default_factory=set, init=False)

    def log_operation(
        self, operation: str, entity_id: str, success: bool, message: str
    ) -> None:
        """Record an operation execution for tracing."""
        self.operations_log.append(
            {
                "operation": operation,
                "entity_id": entity_id,
                "success": success,
                "message": message,
            }
        )
        if success:
            self.successful += 1
        else:
            self.failed += 1
            self.errors.append(message)

    def quarantine(
        self,
        entity_id: str,
        entity_type: str,
        operation: str,
        exception: BaseException,
    ) -> None:
        """Record an operation failure. Appends a record; skip-set membership is idempotent."""
        record = QuarantineRecord(
            entity_id=entity_id,
            entity_type=entity_type,
            operation=operation,
            exception_type=type(exception).__name__,
            message=str(exception),
        )
        self.quarantined.append(record)
        self._quarantined_ids.add(entity_id)

    def is_quarantined(self, entity_id: str) -> bool:
        """Return True if this entity is quarantined from further operations."""
        return entity_id in self._quarantined_ids
