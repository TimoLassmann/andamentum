"""Pattern-based scheduler (DEPRECATED).

The pattern scheduler has been replaced by the pydantic-graph DAG
in ``andamentum.epistemic.graph``. This module is kept for backward
compatibility (WorkItem re-export) and reference. The Pattern class,
PatternScheduler, and WORK_PATTERNS are no longer used by the main
pipeline.
"""

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from .entities import EpistemicEntity, ClaimStage

# WorkItem moved to operations.base — re-export for backward compatibility
from .operations.base import WorkItem  # noqa: F401

if TYPE_CHECKING:
    from .repository import EpistemicRepository


# ══════════════════════════════════════════════════════════════════════════════
# OPERATION → TRACK MAPPING
# Maps operation names to routing track names (from routing.py).
# Operations not in this map are not subject to routing filters.
# ══════════════════════════════════════════════════════════════════════════════

OPERATION_TO_TRACK: dict[str, str] = {
    "adversarial_search": "adversarial",
    "assess_convergence": "convergence",
    "validate_deductively": "deductive",
    "verify_computationally": "computational",
    "analyze_argument": "argument",
    "contrastive_evaluation": "contrastive",
    "cross_claim_consistency": "consistency",
}


@dataclass
class Pattern:
    """Declarative rule: entity in state X needs operation Y.

    Attributes:
        entity_type: What kind of entity to look for
        filters: Required state for matching
        operation: What work to do when matched
        description: Human-readable description
    """

    entity_type: str
    filters: dict[str, Any]
    operation: str
    description: str = ""

    def matches(self, entity: EpistemicEntity) -> bool:
        """Check if entity matches this pattern's filters.

        Args:
            entity: Entity to check

        Returns:
            True if all filters match
        """
        for filter_key, expected in self.filters.items():
            # Handle comparison operators
            # NOTE: No getattr defaults — if a pattern references a field that
            # doesn't exist on the entity, AttributeError surfaces the bug.
            if filter_key.endswith("__gte"):
                field_name = filter_key[:-5]
                actual = getattr(entity, field_name)
                if actual < expected:
                    return False
            elif filter_key.endswith("__lte"):
                field_name = filter_key[:-5]
                actual = getattr(entity, field_name)
                if actual > expected:
                    return False
            elif filter_key.endswith("__gt"):
                field_name = filter_key[:-4]
                actual = getattr(entity, field_name)
                if actual <= expected:
                    return False
            elif filter_key.endswith("__lt"):
                field_name = filter_key[:-4]
                actual = getattr(entity, field_name)
                if actual >= expected:
                    return False
            elif filter_key.endswith("__ne"):
                field_name = filter_key[:-4]
                actual = getattr(entity, field_name)
                if actual == expected:
                    return False
            elif filter_key.endswith("__contains"):
                field_name = filter_key[:-10]
                actual = getattr(entity, field_name)
                if expected not in actual:
                    return False
            else:
                # Exact match
                actual = getattr(entity, filter_key)
                if actual != expected:
                    return False

        return True


# ══════════════════════════════════════════════════════════════════════════════
# WORK PATTERNS
# Complete pattern table from the refactor plan
# ══════════════════════════════════════════════════════════════════════════════


WORK_PATTERNS: list[Pattern] = [
    # ══════════════════════════════════════════════════════════════════
    # TMS: BELIEF MAINTENANCE (priority 1 — correctness-critical)
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="evidence",
        filters={"invalidated": True, "invalidation_cascaded": False},
        operation="invalidate_evidence",
        description="TMS: Cascade evidence invalidation",
    ),
    Pattern(
        entity_type="claim",
        filters={"needs_revalidation": True, "abandoned": False},
        operation="revalidate_claim",
        description="TMS: Re-validate claim stage gate",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 0-2: PRE-PLANNING
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="objective",
        filters={"phase": "new"},
        operation="clarify_question",
        description="Clarify the research question",
    ),
    Pattern(
        entity_type="objective",
        filters={"phase": "clarified", "question_type": None},
        operation="classify_question",
        description="Classify research question type for verification routing",
    ),
    Pattern(
        entity_type="objective",
        filters={"phase": "clarified"},
        operation="conceptual_analysis",
        description="Perform conceptual analysis",
    ),
    Pattern(
        entity_type="objective",
        filters={"phase": "analyzed"},
        operation="plan_task",
        description="Plan evidence collection",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: EVIDENCE COLLECTION
    # Plan creates evidence stubs with extracted=False
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="evidence",
        filters={"extracted": False},
        operation="extract_evidence",
        description="Extract content from evidence source",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 3b: INVESTIGATION (Peirce inquiry cycling)
    # When scrutiny produces doubt, investigate evidence gaps
    # ══════════════════════════════════════════════════════════════════
    # needs_resolution at any stage → investigate
    Pattern(
        entity_type="claim",
        filters={
            "scrutiny_verdict": "needs_resolution",
            "investigation_count__lt": 3,
            "abandoned": False,
        },
        operation="investigate_claim",
        description="Investigate evidence gaps after ambiguous scrutiny",
    ),
    # fail at HYPOTHESIS → investigate (can't demote below HYPOTHESIS)
    Pattern(
        entity_type="claim",
        filters={
            "scrutiny_verdict": "fail",
            "stage": ClaimStage.HYPOTHESIS.value,
            "investigation_count__lt": 3,
            "abandoned": False,
        },
        operation="investigate_claim",
        description="Investigate failed hypothesis before abandoning",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 3c: ABANDONMENT SAFETY NET
    # Catches HYPOTHESIS claims that exhausted investigation attempts
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.HYPOTHESIS.value,
            "scrutiny_verdict": "fail",
            "investigation_count__gte": 3,
            "abandoned": False,
        },
        operation="abandon_stale_claim",
        description="Abandon failed hypothesis after exhausting investigation",
    ),
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.HYPOTHESIS.value,
            "scrutiny_verdict": "needs_resolution",
            "investigation_count__gte": 3,
            "abandoned": False,
        },
        operation="abandon_stale_claim",
        description="Abandon unresolved hypothesis after exhausting investigation",
    ),
    # Demotion: SUPPORTED+ claims with exhausted investigation get demoted
    # (not abandoned — they have real evidence, just can't resolve scrutiny issues)
    Pattern(
        entity_type="claim",
        filters={
            "stage__ne": ClaimStage.HYPOTHESIS.value,
            "scrutiny_verdict": "needs_resolution",
            "investigation_count__gte": 3,
            "abandoned": False,
        },
        operation="demote_claim",
        description="Demote non-hypothesis claim after exhausting investigation",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: CLAIM CREATION
    # Two mutually exclusive modes:
    #   a) Verification mode: claim_to_verify is set → seed the exact
    #      claim (no LLM, no clustering, no assertion extraction).
    #   b) Research mode: claim_to_verify is None → explore evidence and
    #      propose novel claims via the full extraction pipeline.
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="objective",
        filters={
            "phase": "planned",
            "claims_proposed": False,
            "claim_to_verify__ne": None,
        },
        operation="seed_claim",
        description="Create claim from claim_to_verify (verification mode)",
    ),
    Pattern(
        entity_type="objective",
        filters={
            "phase": "planned",
            "claims_proposed": False,
            "claim_to_verify": None,
        },
        operation="propose_claims",
        description="Propose claims from evidence (research mode)",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: INITIAL SCRUTINY
    # All new claims get skeptic review
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="claim",
        filters={"scrutiny_verdict": None},
        operation="scrutinise_claim",
        description="Skeptic review of claim",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 5.5: ROUTING DEFAULTS
    # Pre-mark skipped verification tracks so promotion isn't blocked
    # Must fire BEFORE verification tracks (priority 4 < 5)
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.SUPPORTED.value,
            "scrutiny_verdict": "pass",
            "routing_applied": False,
        },
        operation="set_routing_defaults",
        description="Pre-mark skipped verification tracks based on question type routing",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: VERIFICATION TRACKS
    # Claims at SUPPORTED stage need verification before PROVISIONAL
    # ══════════════════════════════════════════════════════════════════
    # Adversarial search - seek disconfirming evidence
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.SUPPORTED.value, "adversarial_checked": False},
        operation="adversarial_search",
        description="Search for disconfirming evidence",
    ),
    # Cross-domain convergence - check independent evidence lines
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.SUPPORTED.value, "convergence_checked": False},
        operation="assess_convergence",
        description="Assess cross-domain convergence",
    ),
    # Deductive validation - first principles, consistency checks
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.SUPPORTED.value, "deductive_checked": False},
        operation="validate_deductively",
        description="Validate using first principles",
    ),
    # Computational verification - for verifiable claims only
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.SUPPORTED.value, "computational_checked": False},
        operation="verify_computationally",
        description="Verify computationally if applicable",
    ),
    # Contrastive evaluation - pairwise claim comparison
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.SUPPORTED.value, "contrastive_checked": False},
        operation="contrastive_evaluation",
        description="Pairwise contrastive evaluation of competing claims",
    ),
    # Cross-claim consistency - pairwise conflict check
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.SUPPORTED.value, "consistency_checked": False},
        operation="cross_claim_consistency",
        description="Check cross-claim consistency",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 6.5: ABDUCTIVE INTEGRATION
    # After adversarial search, holistically assess all evidence
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.SUPPORTED.value,
            "adversarial_checked": True,
            "integrated_assessment": None,
        },
        operation="integrate_evidence",
        description="Holistic evidence integration (Peirce abduction)",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: STAGE PROMOTION
    # Claims advance through stages when gates pass
    # ══════════════════════════════════════════════════════════════════
    # HYPOTHESIS → SUPPORTED (basic scrutiny passed)
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.HYPOTHESIS.value, "scrutiny_verdict": "pass"},
        operation="promote_claim",
        description="Promote from HYPOTHESIS to SUPPORTED",
    ),
    # SUPPORTED → PROVISIONAL (verification tracks complete)
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.SUPPORTED.value,
            "adversarial_checked": True,
            "convergence_checked": True,
            "deductive_checked": True,
            "contrastive_checked": True,
            "consistency_checked": True,
        },
        operation="promote_claim",
        description="Promote from SUPPORTED to PROVISIONAL",
    ),
    # PROVISIONAL → ROBUST (independent evidence + counterevidence addressed)
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.PROVISIONAL.value,
            "evidence_count__gte": 3,
            "adversarial_checked": True,
            "convergence_checked": True,
            "deductive_checked": True,
        },
        operation="promote_claim",
        description="Promote from PROVISIONAL to ROBUST",
    ),
    # ROBUST → ACTIONABLE (decision criteria met, predictions required)
    Pattern(
        entity_type="claim",
        filters={
            "stage": ClaimStage.ROBUST.value,
            "evidence_count__gte": 3,
            "predictions_generated": True,
        },
        operation="promote_claim",
        description="Promote from ROBUST to ACTIONABLE (requires predictions)",
    ),
    # Demotion on scrutiny failure (exclude HYPOTHESIS — can't demote further)
    Pattern(
        entity_type="claim",
        filters={"scrutiny_verdict": "fail", "stage__ne": ClaimStage.HYPOTHESIS.value},
        operation="demote_claim",
        description="Demote claim after scrutiny failure",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 8: UNCERTAINTY RESOLUTION
    # Only BLOCKING uncertainties must be resolved before synthesis
    # Non-blocking uncertainties are informational caveats
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="uncertainty",
        filters={"resolution": None, "is_blocking": True},
        operation="resolve_uncertainty",
        description="Resolve blocking uncertainty",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 8b: BATCH CONCERN DEDUP
    # After all blocking uncertainties are resolved, batch dedup any
    # remaining concerns they generated before creating new entities.
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="objective",
        filters={"pending_concerns_count__gt": 0},
        operation="deduplicate_concerns",
        description="Batch dedup buffered remaining concerns",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PHASE 9: SYNTHESIS
    # Freeze snapshot when claims are ready, then compile
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="objective",
        filters={"phase": "claims_done", "snapshot_id": None},
        operation="freeze_snapshot",
        description="Freeze snapshot of epistemic state",
    ),
    Pattern(
        entity_type="snapshot",
        filters={"snapshot_type": "final", "artefact_id": None},
        operation="synthesize_report",
        description="Synthesize report from snapshot",
    ),
    # ══════════════════════════════════════════════════════════════════
    # ARGUMENT ANALYSIS
    # Claims that passed scrutiny get argument quality analysis
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="claim",
        filters={"argument_analyzed": False, "scrutiny_verdict": "pass"},
        operation="analyze_argument",
        description="Analyze argument structure and quality",
    ),
    # ══════════════════════════════════════════════════════════════════
    # PREDICTION GENERATION
    # Robust claims get testable predictions (Lakatos)
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.ROBUST.value, "predictions_generated": False},
        operation="generate_prediction",
        description="Generate testable predictions from robust claim",
    ),
    # ══════════════════════════════════════════════════════════════════
    # DECISION TRACKING
    # ══════════════════════════════════════════════════════════════════
    Pattern(
        entity_type="claim",
        filters={"stage": ClaimStage.ACTIONABLE.value, "decision_recorded": False},
        operation="record_decision",
        description="Record decision based on actionable claim",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════


# Per-operation-type budgets: limits how many times SCOPE-CREATING operations
# can execute per run.  Operations not listed have no limit.
#
# Design principle: budgets gate SCOPE (how many entities are created), not
# EXECUTION (whether existing entities are fully processed).
#
# Scope-creating operations (budgeted):
#   clarify_question, conceptual_analysis, plan_task, propose_claims — run once
#   investigate_claim — capped by pattern filter (investigation_count < 3) AND here
#
# Processing operations (NOT budgeted — pattern filters guarantee idempotency):
#   scrutinise_claim, adversarial_search, assess_convergence, validate_deductively,
#   verify_computationally, promote_claim, demote_claim, analyze_argument, etc.
#   Each of these fires at most once per entity due to pattern filters on entity state.
#   Once a claim exists it MUST be fully processed; cutting off mid-pipeline
#   produces incomplete data that cannot be used in benchmarks or scoring.
#
# Synthesis ops are always exempt (see SYNTHESIS_OPS).
DEFAULT_OPERATION_BUDGETS: dict[str, int] = {
    # One-time bootstrap ops — safety caps only
    "clarify_question": 2,
    "classify_question": 2,
    "conceptual_analysis": 2,
    "plan_task": 2,
    "propose_claims": 2,
    "seed_claim": 2,
    # Investigation cycling — capped here AND by pattern filter (investigation_count < 3)
    "investigate_claim": 20,
}

# Synthesis operations are never budget-limited — they must always be able to run.
SYNTHESIS_OPS: set[str] = {
    "freeze_snapshot",
    "synthesize_report",
    "deduplicate_concerns",
}

# Per-(entity, operation) attempt limit.  Once an (entity, op) pair has been
# attempted this many times, it is permanently excluded for the rest of the run.
MAX_ENTITY_ATTEMPTS = 3


class PatternScheduler:
    """Match patterns and schedule work items.

    The scheduler:
    1. Queries entities matching patterns
    2. Filters out budget-exhausted operations and entity-exhausted pairs
    3. Creates work items sorted by priority

    All workflow logic lives in the patterns themselves.

    Two throttling mechanisms prevent budget waste:
    - Per-operation-type budgets cap how many times each op type can execute.
    - Per-(entity, operation) attempt limits permanently exclude pairs that
      have been tried MAX_ENTITY_ATTEMPTS times without advancing.
    """

    def __init__(
        self,
        repo: "EpistemicRepository",
        patterns: Optional[list[Pattern]] = None,
        operation_budgets: Optional[dict[str, int]] = None,
    ):
        """Initialize scheduler.

        Args:
            repo: Repository for entity queries
            patterns: Optional custom patterns (defaults to WORK_PATTERNS)
            operation_budgets: Optional per-operation budget overrides
        """
        self.repo = repo
        self.patterns = patterns or WORK_PATTERNS
        # Per-operation-type counters (total executions this run)
        self._op_counts: dict[str, int] = {}
        self._op_budgets = (
            operation_budgets
            if operation_budgets is not None
            else dict(DEFAULT_OPERATION_BUDGETS)
        )
        # Per-(entity, operation) attempt counters — permanent, never reset
        self._entity_attempts: dict[tuple[str, str], int] = {}

    def record_attempt(self, entity_id: str, operation: str) -> None:
        """Record a failed operation attempt on an entity.

        Called ONLY on failure (result.success=False or exception). Successful
        operations do not count against the attempt limit because Peirce cycling
        legitimately re-invokes operations (scrutiny, promote) after the
        epistemic state has changed. Only persistent failures indicate a
        broken entity that should be excluded.

        Capped at MAX_ENTITY_ATTEMPTS. Does NOT consume operation budget —
        only record_success does that.
        """
        key = (entity_id, operation)
        self._entity_attempts[key] = self._entity_attempts.get(key, 0) + 1

    def record_success(self, operation: str) -> None:
        """Record a successful operation (called after execution succeeds).

        Consumes one unit of the operation's budget. Failed operations
        do not consume budget — the system should not be penalized for
        provider errors, timeouts, or other transient failures.
        """
        self._op_counts[operation] = self._op_counts.get(operation, 0) + 1

    def _is_budget_exhausted(self, operation: str) -> bool:
        """Check if operation type has hit its per-run budget."""
        if operation in SYNTHESIS_OPS:
            return False  # Synthesis is never budget-limited
        budget = self._op_budgets.get(operation)
        if budget is None:
            return False  # No budget = unlimited
        return self._op_counts.get(operation, 0) >= budget

    def _is_entity_exhausted(self, entity_id: str, operation: str) -> bool:
        """Check if this (entity, operation) pair has exceeded attempt limit."""
        return (
            self._entity_attempts.get((entity_id, operation), 0) >= MAX_ENTITY_ATTEMPTS
        )

    def reset_entity_attempts(self, entity_id: str, operation: str | None = None) -> None:
        """Reset attempt counters for an entity after epistemic state changes.

        Called when new evidence is judged — earlier promote failures
        may no longer be predictive because the gate inputs changed.

        Args:
            entity_id: Entity to reset
            operation: If given, reset only this operation's counter.
                       If None, reset ALL operation counters for this entity.
        """
        if operation is not None:
            self._entity_attempts.pop((entity_id, operation), None)
        else:
            keys_to_remove = [k for k in self._entity_attempts if k[0] == entity_id]
            for k in keys_to_remove:
                del self._entity_attempts[k]

    async def get_pending_work(
        self, objective_id: Optional[str] = None
    ) -> list[WorkItem]:
        """Get all pending work items.

        Round-aware: resolve_uncertainty work is only returned when no
        other work exists. This ensures all uncertainty-creating operations
        finish before resolution starts, giving resolution access to the
        full pile of uncertainties at once.
        """
        work_items: list[WorkItem] = []

        for pattern in self.patterns:
            if self._is_budget_exhausted(pattern.operation):
                continue

            filters = dict(pattern.filters)
            if objective_id:
                filters["objective_id"] = objective_id

            try:
                entities = await self.repo.query(pattern.entity_type, **filters)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "Pattern query failed for %s (operation=%s): %s",
                    pattern.entity_type,
                    pattern.operation,
                    e,
                )
                continue

            for entity in entities:
                if pattern.matches(entity) and not self._is_entity_exhausted(
                    entity.entity_id, pattern.operation
                ):
                    work_items.append(
                        WorkItem(
                            entity_id=entity.entity_id,
                            entity_type=pattern.entity_type,
                            operation=pattern.operation,
                            metadata={
                                "pattern_description": pattern.description,
                                "objective_id": entity.objective_id,
                            },
                        )
                    )

        # Apply routing filter
        if work_items:
            work_items = await self._apply_routing_filter(work_items)

        # Round-aware scheduling: hold resolution work until everything
        # else is done.  This ensures the full pile of uncertainties is
        # visible before resolution starts.
        non_resolution = [w for w in work_items if w.operation != "resolve_uncertainty"]
        if non_resolution:
            return non_resolution

        # Only resolution work remains — return it
        return [w for w in work_items if w.operation == "resolve_uncertainty"]

    async def _apply_routing_filter(self, work_items: list[WorkItem]) -> list[WorkItem]:
        """Filter verification work items based on routing activation level.

        Three behaviors:
        - SKIP: never fire for this question type
        - PRIMARY: always fire
        - SECONDARY: fire only if a deterministic condition on the claim is met
          (e.g., adversarial fires only if evidence is conflicting)

        For backward compatibility, if the objective has no question_type,
        all tracks fire (no filtering).
        """
        from .routing import get_active_tracks, TrackActivation

        filtered: list[WorkItem] = []
        objective_cache: dict[str, Optional[str]] = {}
        claim_cache: dict[str, Any] = {}

        for item in work_items:
            track_name = OPERATION_TO_TRACK.get(item.operation)
            if track_name is None:
                filtered.append(item)
                continue

            obj_id = item.metadata.get("objective_id", "")
            if not obj_id:
                filtered.append(item)
                continue

            if obj_id not in objective_cache:
                try:
                    objective = await self.repo.get("objective", obj_id)
                    objective_cache[obj_id] = getattr(objective, "question_type", None)
                except Exception:
                    objective_cache[obj_id] = None

            question_type = objective_cache.get(obj_id)
            if not question_type:
                filtered.append(item)
                continue

            tracks = get_active_tracks(question_type)
            activation = tracks.get(track_name, TrackActivation.PRIMARY)

            if activation == TrackActivation.SKIP:
                continue

            if activation == TrackActivation.SECONDARY:
                # Secondary tracks fire only when a deterministic condition is met.
                # Load the claim to check its state.
                entity_id = item.entity_id
                if entity_id not in claim_cache:
                    try:
                        claim_cache[entity_id] = await self.repo.get("claim", entity_id)
                    except Exception:
                        claim_cache[entity_id] = None

                claim = claim_cache.get(entity_id)
                if claim and not self._secondary_condition_met(claim, track_name):
                    continue

            # PRIMARY or condition-met SECONDARY — include
            filtered.append(item)

        return filtered

    @staticmethod
    def _secondary_condition_met(claim: Any, track_name: str) -> bool:
        """Check whether a SECONDARY verification track should fire for this claim.

        Secondary tracks only fire when there's a specific reason — not by default.
        This prevents expensive operations (like adversarial search) from running
        on every claim when the routing says they're only needed sometimes.

        The conditions are deterministic checks on claim state, not LLM calls.

        Args:
            claim: The claim entity to check
            track_name: Which verification track is being considered

        Returns:
            True if the track should fire, False if it should be skipped
        """
        if track_name == "adversarial":
            # Fire on first pass (balance is None — hasn't been tested yet)
            # or if a prior adversarial run showed poor balance (< 0.6).
            # The only case where we skip: adversarial already ran AND the
            # balance was healthy (>= 0.6), meaning the claim survived.
            balance = getattr(claim, "adversarial_balance", None)
            if balance is not None and balance >= 0.6:
                return False  # Already tested, survived — skip re-run
            return True

        if track_name == "convergence":
            # Fire if the claim has 3+ evidence items but convergence hasn't
            # been checked yet — suggests enough data to assess independence.
            count = getattr(claim, "evidence_count", 0)
            checked = getattr(claim, "convergence_checked", False)
            return count >= 3 and not checked

        # Deductive and argument analysis are cheap (< 2s average).
        # Fire on first pass when SECONDARY — the cost of skipping them
        # and missing a logical issue is higher than the cost of running them.
        if track_name in ("deductive", "argument"):
            return True

        # Unknown track — fire by default (safe)
        return True

    async def get_next_work(
        self, objective_id: Optional[str] = None
    ) -> Optional[WorkItem]:
        """Get highest priority pending work item.

        Args:
            objective_id: Optional filter to single objective

        Returns:
            Next work item or None if no work pending
        """
        work_items = await self.get_pending_work(objective_id)
        return work_items[0] if work_items else None

    async def has_pending_work(self, objective_id: Optional[str] = None) -> bool:
        """Check if there is any pending work.

        Args:
            objective_id: Optional filter to single objective

        Returns:
            True if work is pending
        """
        work = await self.get_next_work(objective_id)
        return work is not None

    def get_patterns_for_operation(self, operation: str) -> list[Pattern]:
        """Get all patterns that trigger a specific operation.

        Args:
            operation: Operation name

        Returns:
            List of matching patterns
        """
        return [p for p in self.patterns if p.operation == operation]

    def describe_patterns(self) -> str:
        """Get human-readable description of all patterns."""
        lines = ["Work Patterns:", ""]
        for p in self.patterns:
            filter_str = ", ".join(f"{k}={v}" for k, v in p.filters.items())
            lines.append(f"  {p.entity_type}({filter_str}) → {p.operation}")
            if p.description:
                lines.append(f"      {p.description}")
        return "\n".join(lines)
