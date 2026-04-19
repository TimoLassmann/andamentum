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

    # ── Operation trace ─────────────────────────────────────────
    # Lightweight log for the progress callback and final stats
    operations_log: list[dict[str, Any]] = field(default_factory=list)
    successful: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

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
