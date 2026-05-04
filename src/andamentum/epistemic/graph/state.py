"""Mutable state for the epistemic graph.

Passed to every node via ctx.state. Nodes mutate it in place.

This is the single source of truth for pipeline progress within
one graph execution. Unlike the pattern scheduler (which queries
entity state from the repo), the graph state tracks what has been
DONE, not what entities EXIST.

Architecture: Layer 1 (framework-agnostic, pure dataclass)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .quarantine import QuarantineRecord


def _new_run_id() -> str:
    """Fresh per-graph-run identifier.

    Used to disambiguate execution-trace ``file_path`` entries when
    multiple graph runs share one DocumentStore — e.g. the decomposed
    orchestrator running N children, or a re-run of the same objective.
    Without this disambiguator, ``execution_step_<step>`` collides on
    the documents.file_path UNIQUE index.
    """
    return uuid.uuid4().hex[:12]


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

    # ── Run identity ─────────────────────────────────────────────
    # Per-graph-run disambiguator. Each fresh state gets a new id;
    # used to keep execution-trace file_paths unique when multiple
    # graph runs share a DocumentStore (decomposed runs, re-runs).
    run_id: str = field(default_factory=_new_run_id)

    # ── Claim creation ──────────────────────────────────────────
    claims_created: bool = False
    # Tracks the entity_ids of active (non-abandoned) claims at the end
    # of CreateClaims. Diagnostic / test-observation field — production
    # routing reads claims directly from the repo, not from this list.
    claim_ids: list[str] = field(default_factory=list)

    # ── Per-claim tracking ──────────────────────────────────────
    # Investigation cycle counts (Peirce cycling cap)
    investigation_counts: dict[str, int] = field(default_factory=dict)

    # Claims that have completed verification + integration
    verification_done: set[str] = field(default_factory=set)

    # Claims marked terminal by PromoteToSupported / PromoteSupported
    # (cycle_capped, abandoned-stale, refute-promoted, etc.). Diagnostic
    # / test-observation set — production routing reads claim flags
    # directly from the repo, not from this set.
    terminal_claims: set[str] = field(default_factory=set)

    # ── Retrieval health ────────────────────────────────────────
    # Number of consecutive extractions that returned zero content.
    # Incremented by ExtractEvidence nodes (Task B2); reset to 0
    # when a non-empty extraction lands.
    consecutive_empty_extractions: int = 0

    # Set to True when consecutive_empty_extractions crosses the
    # threshold. Downstream nodes (Task B3) check this and
    # short-circuit to CheckCompletion; the PosteriorReport
    # surfaces it as a distinct terminal_state.
    retrieval_failed: bool = False

    # ── Flow control (graph-managed, not on entities) ───────────
    # Claims needing re-scrutiny after uncertainty resolution or
    # investigation. Scrutinize node checks this IN ADDITION to
    # claims with scrutiny_verdict=None.
    claims_needing_rescrutiny: set[str] = field(default_factory=set)

    # Per-claim counter for the Scrutinize ↔ ResolveUncertainties
    # oscillation. Each pass through Scrutinize on a claim that was
    # marked for rescrutiny increments this. Once a claim's count hits
    # ``SCRUTINY_RESOLVE_CYCLE_CAP`` (defined in nodes.py),
    # ResolveUncertainties stops marking it for further rescrutiny and
    # Scrutinize stops accepting new rescrutiny requests for it. The
    # claim retains its current scrutiny_verdict and the graph proceeds
    # — analogous to ``investigation_counts`` capping the investigate
    # loop at 3. Without this cap, claims whose seed-text reliably
    # produces a single novel-but-near-duplicate uncertainty per
    # scrutiny pass would oscillate forever (decomposed cases hit this
    # in the wild on smoke_v12).
    scrutiny_resolve_cycles: dict[str, int] = field(default_factory=dict)

    # Claims needing TMS revalidation after evidence changes.
    # Populated by graph nodes, consumed by _run_tms_sweep.
    claims_needing_tms: set[str] = field(default_factory=set)

    # Set by CheckSynthesisDemand when it routes to SynthesizeInsufficient
    # (demand=needs_more, no eligible claims for loop-back). Carries the
    # structural diagnosis from the gate to the deterministic synthesis
    # operation so the artefact body can surface *why* the system
    # suspended judgment. None on every other path — Synthesize doesn't
    # read it.
    synthesis_insufficient_reason: str | None = None

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
