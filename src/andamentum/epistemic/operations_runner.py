"""Operations Runner - Entry point for epistemic operations.

This module provides the entry point for epistemic research operations
using the pattern-driven scheduler (Layer 1 architecture).

Workflow emerges from entity state + pattern matching:
- PatternScheduler matches entity states to find pending work
- Operations transform entity state
- No central workflow logic

Execution trace: every operation is recorded as an execution_step document
in the same database, enabling step-by-step replay in the Observer.

Architecture: Layer 1 (standalone package)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .confidence import AnswerConfidenceReport, PosteriorReport

from andamentum.document_store import DocumentStore

# Type for progress callback: (operation_type, workitem_id, success, message, outputs) -> None
ProgressCallback = Callable[[str, str, bool, str, dict[str, Any]], None]

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER RESULT
# ══════════════════════════════════════════════════════════════════════════════


class PatternSchedulerResult:
    """Result from pattern-driven scheduler run.

    Provides both pattern-scheduler native fields and compatibility
    properties used by CLI handlers.
    """

    def __init__(
        self,
        objective_id: str,
        iterations: int,
        successful: int,
        failed: int,
        status: str,
        errors: Optional[list[str]] = None,
        answer_confidence: Optional["AnswerConfidenceReport"] = None,
        posterior: Optional["PosteriorReport"] = None,
    ):
        self.objective_id = objective_id
        self.iterations = iterations
        self.successful = successful
        self.failed = failed
        self.status = status
        self.errors = errors or []
        self.answer_confidence = answer_confidence
        self.posterior = posterior

    @property
    def success(self) -> bool:
        """Whether the scheduler produced useful results.

        A run succeeds if it completed at least one operation successfully.
        Individual operation failures (e.g., agent rate limits, inapplicable
        verification) are expected noise in a multi-agent pipeline — they
        don't invalidate the findings that were produced.
        """
        return self.successful > 0


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION TRACE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


async def _snapshot_entity(
    repo: Any,
    entity_type: str,
    entity_id: str,
) -> Optional[dict[str, Any]]:
    """Snapshot entity state as a dict for before/after comparison."""
    try:
        entity = await repo.get(entity_type, entity_id)
        return entity.model_dump(mode="json")
    except Exception:
        return None


def _compute_state_changes(
    before: Optional[dict[str, Any]],
    after: Optional[dict[str, Any]],
) -> list[str]:
    """Compute list of field changes between two entity snapshots."""
    if before is None or after is None:
        return []
    changes: list[str] = []
    all_keys = sorted(set(before.keys()) | set(after.keys()))
    skip = {"updated_at"}  # Always changes, not interesting
    for key in all_keys:
        if key in skip:
            continue
        old = before.get(key)
        new = after.get(key)
        if old != new:
            old_s = _format_change_value(old)
            new_s = _format_change_value(new)
            changes.append(f"{key}: {old_s} → {new_s}")
    return changes


def _format_change_value(value: Any) -> str:
    """Format a value for state change display."""
    if value is None:
        return "null"
    if isinstance(value, str):
        if len(value) > 80:
            return f'"{value[:77]}..."'
        return f'"{value}"'
    if isinstance(value, list):
        if len(value) > 3:
            return f"[{len(value)} items]"
        return str(value)
    return str(value)


def _build_step_content(
    agent_calls: list[dict[str, Any]],
    entity_before: Optional[dict[str, Any]],
    entity_after: Optional[dict[str, Any]],
    state_changes: list[str],
) -> str:
    """Build markdown content for an execution_step document."""
    sections: list[str] = []

    # Agent I/O
    for i, call in enumerate(agent_calls):
        agent_label = "Agent Call" if len(agent_calls) == 1 else f"Agent Call {i + 1}"
        sections.append(f"## {agent_label}: {call.get('agent_name', 'unknown')}\n")
        sections.append("### Input\n```json")
        sections.append(json.dumps(call.get("input", {}), indent=2, default=str))
        sections.append("```\n")
        sections.append("### Output\n```")
        sections.append(str(call.get("raw_output", "")))
        sections.append("```\n")

    # State changes
    if state_changes:
        sections.append("## State Changes\n")
        for change in state_changes:
            sections.append(f"- {change}")
        sections.append("")

    return "\n".join(sections)


async def _record_step(
    store: DocumentStore,
    run_id: str,
    step_number: int,
    work: Any,
    result: Any,
    agent_calls: list[dict[str, Any]],
    entity_before: Optional[dict[str, Any]],
    entity_after: Optional[dict[str, Any]],
    obj_phase_before: Optional[str],
    obj_phase_after: Optional[str],
    started_at: datetime,
    completed_at: datetime,
    objective_id: str,
) -> None:
    """Record a single execution step as a document in the epistemic database."""
    state_changes = _compute_state_changes(entity_before, entity_after)
    content = _build_step_content(
        agent_calls, entity_before, entity_after, state_changes
    )

    duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    metadata: dict[str, Any] = {
        "epistemic_type": "execution_step",
        "objective_id": objective_id,
        "run_id": run_id,
        "step_number": step_number,
        "operation": work.operation,
        "pattern_description": work.metadata.get("pattern_description", ""),
        "target_entity_type": work.entity_type,
        "target_entity_id": work.entity_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_ms": duration_ms,
        "success": result.success,
        "message": result.message,
        "created_entities": result.created_entities,
        "validation_errors": result.validation_errors,
        "state_changes": state_changes,
        "agent_names": [c.get("agent_name", "") for c in agent_calls],
    }
    if obj_phase_before is not None:
        metadata["objective_phase_before"] = obj_phase_before
    if obj_phase_after is not None:
        metadata["objective_phase_after"] = obj_phase_after

    try:
        await store.add(
            f"execution_step_{step_number:03d}_{work.operation}.json",
            content=content,
            title=f"Step {step_number}: {work.operation}",
            metadata=metadata,
        )
    except Exception as e:
        logger.warning(f"Failed to record execution step {step_number}: {e}")


async def _record_termination(
    store: DocumentStore,
    repo: Any,
    run_id: str,
    step_number: int,
    objective_id: str,
    reason: str,
    iterations: int,
    successful: int,
    failed: int,
    run_started_at: datetime,
    errors: list[str],
) -> None:
    """Record why the scheduler loop terminated."""
    completed_at = datetime.now(timezone.utc)
    total_duration_ms = int((completed_at - run_started_at).total_seconds() * 1000)

    # Capture final entity states for debugging
    blocking_entities: list[dict[str, Any]] = []
    content_parts: list[str] = ["## Run Termination\n"]
    content_parts.append(f"- **Reason**: {reason}")
    content_parts.append(f"- **Steps completed**: {iterations}")
    content_parts.append(f"- **Successful**: {successful}")
    content_parts.append(f"- **Failed**: {failed}")
    content_parts.append(f"- **Total duration**: {total_duration_ms}ms")
    content_parts.append("")

    if errors:
        content_parts.append("## Errors\n")
        for err in errors:
            content_parts.append(f"- {err}")
        content_parts.append("")

    if reason == "no_matching_patterns":
        content_parts.append("## Entity State Analysis\n")
        content_parts.append("Entities that may be blocking progress:\n")

        # Check claims
        try:
            from .entities import Claim

            claims = await repo.query("claim", objective_id=objective_id)
            for c in claims:
                if not isinstance(c, Claim):
                    continue
                issues: list[str] = []
                if c.scrutiny_verdict == "needs_resolution":
                    issues.append(
                        "scrutiny_verdict='needs_resolution' (no pattern handles this)"
                    )
                if c.scrutiny_verdict is None:
                    issues.append("scrutiny_verdict=None (awaiting scrutiny)")
                if issues:
                    entity_info = {
                        "type": "claim",
                        "id": c.entity_id,
                        "stage": c.stage.value,
                        "issues": issues,
                    }
                    blocking_entities.append(entity_info)
                    content_parts.append(
                        f"- **Claim** `{c.entity_id[:12]}` (stage={c.stage.value})"
                    )
                    for issue in issues:
                        content_parts.append(f"  - {issue}")
        except Exception as e:
            content_parts.append(f"- Could not analyze claims: {e}")

        # Check objective phase
        try:
            from .entities import Objective

            obj = await repo.get("objective", objective_id)
            if isinstance(obj, Objective):
                content_parts.append(f"\n**Objective phase**: {obj.phase}")
                if obj.phase == "claims_proposed" and not blocking_entities:
                    content_parts.append(
                        "  Objective stuck at claims_proposed — check claim states above"
                    )
        except Exception:
            pass

    content = "\n".join(content_parts)

    metadata: dict[str, Any] = {
        "epistemic_type": "execution_step",
        "objective_id": objective_id,
        "run_id": run_id,
        "step_number": step_number,
        "operation": "_run_terminated",
        "started_at": completed_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_ms": 0,
        "success": reason in ("complete", "no_matching_patterns"),
        "message": f"Run terminated: {reason}",
        "termination_reason": reason,
        "total_iterations": iterations,
        "total_successful": successful,
        "total_failed": failed,
        "total_duration_ms": total_duration_ms,
        "blocking_entities": blocking_entities,
        "created_entities": [],
        "validation_errors": [],
    }

    try:
        await store.add(
            f"execution_step_{step_number:03d}__run_terminated.json",
            content=content,
            title=f"Run Terminated: {reason}",
            metadata=metadata,
        )
    except Exception as e:
        logger.warning(f"Failed to record termination: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS FORCE-TRIGGER
# ══════════════════════════════════════════════════════════════════════════════


async def _force_synthesis_if_needed(
    repo: Any,
    objective_id: str,
    scheduler: Any,
) -> bool:
    """Force-trigger synthesis if no artefact has been produced yet.

    When the scheduler runs out of work (all budgets/attempts exhausted),
    this ensures we still get a final artefact by setting the objective
    phase to 'claims_done' so freeze_snapshot + synthesize_report patterns
    match.

    Returns True if synthesis was force-triggered (caller should re-query
    the scheduler), False if synthesis already happened or no claims exist.
    """
    from .entities import Objective, Snapshot

    try:
        obj = await repo.get("objective", objective_id)
    except Exception:
        return False

    if not isinstance(obj, Objective):
        return False

    # Check if artefact already exists
    snapshots = await repo.query("snapshot", objective_id=objective_id)
    for s in snapshots:
        if isinstance(s, Snapshot) and s.artefact_id is not None:
            return False  # Already have an artefact

    # Check if there are any claims to synthesize
    claims = await repo.query("claim", objective_id=objective_id)
    if not claims:
        return False  # Nothing to synthesize

    # Force synthesis: set phase to claims_done and clear snapshot_id
    if obj.phase != "claims_done":
        obj.phase = "claims_done"
        obj.snapshot_id = None
        await repo.save(obj)
        logger.info("Force-triggered synthesis: set objective phase to claims_done")
        return True

    # Phase is already claims_done but no snapshot yet — scheduler should pick it up
    if obj.snapshot_id is None:
        return True  # freeze_snapshot pattern should match

    return False


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════


async def run_research_question(
    question: str,
    database_name: str = "epistemic_research",
    max_iterations: Optional[int] = 50,
    verbose: bool = False,
    skip_preplanning: bool = False,
    model: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    provider: str = "all",
    providers: Optional[dict[str, Any]] = None,
    quality_scorer: Optional[Any] = None,
    db_dir: Optional[str] = None,
    operation_budgets: Optional[dict[str, int]] = None,
) -> PatternSchedulerResult:
    """Run a research question through the epistemic pipeline.

    Uses the pattern-driven scheduler where workflow emerges from
    entity state + pattern matching. No central workflow logic.

    Full pipeline:
        CLARIFY_QUESTION → CONCEPTUAL_ANALYSIS → PLAN_TASK →
        EXTRACT_EVIDENCE → PROPOSE_CLAIMS → SCRUTINISE_CLAIM →
        [verification tracks] → PROMOTE_CLAIM → FREEZE_SNAPSHOT →
        SYNTHESIZE_REPORT

    Args:
        question: The research question to investigate
        database_name: Name of the database to use
        max_iterations: Maximum scheduler iterations (None for unlimited)
        verbose: Print detailed output
        skip_preplanning: Skip CLARIFY_QUESTION and CONCEPTUAL_ANALYSIS
        model: Optional LLM model to use (e.g., "openai:gpt-4o-mini")
        progress_callback: Optional callback for progress updates.
            Signature: (operation_type, entity_id, success, message, outputs) -> None
        provider: Evidence provider to use: "web_search" or "all"
        providers: Optional dict of named EvidenceProvider instances to inject.
            Overrides the default provider setup when given.
        quality_scorer: Optional QualityScorer instance for evidence quality scoring.
        db_dir: Custom directory for database files. When provided, the database
            is written here instead of ~/.config/andamentum/databases/.

    Returns:
        PatternSchedulerResult with execution summary
    """
    from .repository import EpistemicRepository
    from .patterns import PatternScheduler
    from .operations import create_operations, OperationResult
    from .entities import Objective
    from .evidence_gathering import get_default_gatherer

    # Initialize database and repository
    from .storage import DocumentStoreAdapter

    store = DocumentStore.for_database(database_name, db_dir=db_dir)
    await store.initialize()
    repo = EpistemicRepository(DocumentStoreAdapter(store))

    # Resume if an objective already exists in this database, otherwise create one.
    existing_objectives = await repo.query("objective")
    if existing_objectives:
        objective_id = existing_objectives[0].objective_id
        if verbose:
            phase = getattr(existing_objectives[0], "phase", "unknown")
            logger.info(f"Resuming objective: {objective_id} (phase={phase})")
    else:
        objective_id = f"obj_{uuid.uuid4().hex[:12]}"
        starting_phase = "analyzed" if skip_preplanning else "new"
        objective = Objective(
            entity_id=objective_id,
            objective_id=objective_id,
            description=question,
            phase=starting_phase,
        )
        await repo.save(objective)
        if verbose:
            logger.info(f"Created objective: {objective_id} (phase={starting_phase})")

    # Create agent runner, evidence gatherer, and operations
    from .runner import DefaultAgentRunner

    if not model:
        raise ValueError(
            "model is required for run_research_question. "
            "Pass --model or set ANDAMENTUM_MAIN_LLM_MODEL."
        )
    agent_runner = DefaultAgentRunner(model=model)

    # Auto-load providers based on `provider` string when no explicit providers given
    if providers is None and provider == "all":
        from .providers import get_biomedical_providers

        providers = get_biomedical_providers()

    evidence_gatherer = (
        get_default_gatherer(model=model, providers=providers) if model else None
    )
    operations = create_operations(
        repo,
        agent_runner,
        evidence_gatherer=evidence_gatherer,
        quality_scorer=quality_scorer,
    )

    # Create pattern scheduler with operation budgets
    scheduler = PatternScheduler(repo, operation_budgets=operation_budgets)

    # Execution trace
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_started_at = datetime.now(timezone.utc)
    step_number = 0

    # Run scheduler loop
    iterations = 0
    successful = 0
    failed = 0
    errors: list[str] = []
    synthesis_forced = False  # Guard against infinite force-trigger loops

    while max_iterations is None or iterations < max_iterations:
        # Get next work item
        work = await scheduler.get_next_work(objective_id)

        if work is None:
            # No schedulable work — check if synthesis has been produced.
            # If not, force-trigger synthesis so we always get an artefact.
            if not synthesis_forced:
                synthesis_forced = True
                forced = await _force_synthesis_if_needed(repo, objective_id, scheduler)
                if forced:
                    # Re-enter loop to execute the synthesis work
                    continue

            if verbose:
                logger.info("No more pending work")
            # Record termination
            termination_reason = (
                "complete" if successful > 0 else "no_matching_patterns"
            )
            await _record_termination(
                store,
                repo,
                run_id,
                step_number + 1,
                objective_id,
                termination_reason,
                iterations,
                successful,
                failed,
                run_started_at,
                errors,
            )
            break

        iterations += 1
        step_number += 1

        # Record attempt BEFORE execution (permanent tracking)
        scheduler.record_attempt(work.entity_id, work.operation)

        if verbose:
            logger.info(
                f"[{iterations}] {work.operation} on {work.entity_type}:{work.entity_id[:8]}"
            )

        # Get operation handler
        op = operations.get(work.operation)
        if not op:
            failed += 1
            error_msg = f"Unknown operation: {work.operation}"
            errors.append(error_msg)
            if verbose:
                logger.warning(error_msg)
            continue

        # Capture state BEFORE execution
        entity_before = await _snapshot_entity(repo, work.entity_type, work.entity_id)
        obj_before = await _snapshot_entity(repo, "objective", objective_id)
        obj_phase_before = obj_before.get("phase") if obj_before else None

        # Reset agent call tracking
        op._agent_calls = []

        # Execute operation
        started_at = datetime.now(timezone.utc)
        try:
            result: OperationResult = await op.execute(work)

            completed_at = datetime.now(timezone.utc)

            # Capture state AFTER execution
            entity_after = await _snapshot_entity(
                repo, work.entity_type, work.entity_id
            )
            obj_after = await _snapshot_entity(repo, "objective", objective_id)
            obj_phase_after = obj_after.get("phase") if obj_after else None

            # Record execution step
            await _record_step(
                store,
                run_id,
                step_number,
                work,
                result,
                op._agent_calls,
                entity_before,
                entity_after,
                obj_phase_before,
                obj_phase_after,
                started_at,
                completed_at,
                objective_id,
            )

            if result.success:
                successful += 1
                scheduler.record_success(work.operation)
                if progress_callback:
                    progress_callback(
                        work.operation,
                        work.entity_id,
                        True,
                        result.message,
                        {"created_entities": result.created_entities},
                    )
            else:
                failed += 1
                errors.append(result.message)
                if progress_callback:
                    progress_callback(
                        work.operation,
                        work.entity_id,
                        False,
                        result.message,
                        {"validation_errors": result.validation_errors},
                    )

            if verbose:
                status = "✓" if result.success else "✗"
                logger.info(f"  {status} {result.message}")

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            failed += 1
            error_msg = f"{work.operation} failed: {str(e)}"
            errors.append(error_msg)
            if verbose:
                logger.error(f"  ✗ {error_msg}")

            # Record failed step (create a minimal OperationResult for the trace)
            from .operations import OperationResult as _OR

            error_result = _OR(
                success=False,
                entity_id=work.entity_id,
                message=error_msg,
            )
            await _record_step(
                store,
                run_id,
                step_number,
                work,
                error_result,
                op._agent_calls,
                entity_before,
                None,
                obj_phase_before,
                None,
                started_at,
                completed_at,
                objective_id,
            )
    else:
        # Loop ended due to max_iterations — try to force synthesis before giving up
        forced = await _force_synthesis_if_needed(repo, objective_id, scheduler)
        if forced:
            # Execute ONLY synthesis operations (freeze_snapshot + synthesize_report).
            from .patterns import SYNTHESIS_OPS

            for _ in range(5):  # enough for freeze_snapshot + synthesize_report
                all_work = await scheduler.get_pending_work(objective_id)
                synthesis_work = [w for w in all_work if w.operation in SYNTHESIS_OPS]
                if not synthesis_work:
                    break
                work = synthesis_work[0]
                op = operations.get(work.operation)
                if not op:
                    break
                scheduler.record_attempt(work.entity_id, work.operation)
                try:
                    result = await op.execute(work)
                    if result.success:
                        successful += 1
                    else:
                        failed += 1
                        logger.warning(
                            f"Post-budget synthesis {work.operation} failed: {result.message}"
                        )
                except Exception as e:
                    failed += 1
                    logger.warning(
                        f"Post-budget synthesis {work.operation} exception: {e}"
                    )

        await _record_termination(
            store,
            repo,
            run_id,
            step_number + 1,
            objective_id,
            "max_iterations",
            iterations,
            successful,
            failed,
            run_started_at,
            errors,
        )

    # Determine final status
    if failed == 0 and iterations > 0:
        status = "complete"
    elif iterations >= (max_iterations or 0) and max_iterations is not None:
        status = "max_iterations"
    else:
        status = "partial"

    if verbose:
        logger.info(f"Scheduler complete: {successful} successful, {failed} failed")

    # Compute post-hoc confidence from structural signals (deterministic, no LLM)
    answer_confidence_report = None
    posterior_report = None
    if successful > 0:
        try:
            from .confidence import compute_answer_confidence

            answer_confidence_report = await compute_answer_confidence(
                repo, objective_id
            )
        except Exception as e:
            logger.warning(f"Answer confidence computation failed: {e}")

        try:
            from .confidence import compute_posterior

            posterior_report = await compute_posterior(repo, objective_id)
        except Exception as e:
            logger.warning(f"Posterior computation failed: {e}")

    return PatternSchedulerResult(
        objective_id=objective_id,
        iterations=iterations,
        successful=successful,
        failed=failed,
        status=status,
        errors=errors,
        answer_confidence=answer_confidence_report,
        posterior=posterior_report,
    )


async def get_research_summary(
    database_name: str,
    objective_id: Optional[str] = None,
) -> dict:
    """Get a summary of research results from the database.

    Args:
        database_name: Name of the database
        objective_id: Optional specific objective ID (uses first if not provided)

    Returns:
        Dictionary with claims, evidence, and uncertainties
    """
    store = DocumentStore.for_database(database_name)
    await store.initialize()

    # Find objective
    if not objective_id:
        results = await store.find_by_metadata(
            {
                "epistemic_type": "objective",
            },
            limit=1,
        )
        if not results:
            return {"error": "No objective found"}
        objective_id = results[0].metadata.get("objective_id")

    # Collect claims
    claims = await store.find_by_metadata(
        {
            "epistemic_type": "claim",
            "objective_id": objective_id,
        }
    )

    # Collect evidence
    evidence = await store.find_by_metadata(
        {
            "epistemic_type": "evidence",
            "objective_id": objective_id,
        }
    )

    # Collect uncertainties
    uncertainties = await store.find_by_metadata(
        {
            "epistemic_type": "uncertainty",
            "objective_id": objective_id,
        }
    )

    # Collect artefacts
    artefacts = await store.find_by_metadata(
        {
            "epistemic_type": "artefact",
            "objective_id": objective_id,
        }
    )

    return {
        "objective_id": objective_id,
        "claims_count": len(claims),
        "evidence_count": len(evidence),
        "uncertainties_count": len(uncertainties),
        "artefacts_count": len(artefacts),
        "claims": [
            {
                "claim_id": c.metadata.get("claim_id"),
                "statement": c.metadata.get("statement", "")[:100],
                "stage": c.metadata.get("stage"),
            }
            for c in claims
        ],
        "artefacts": [
            {
                "artefact_id": a.metadata.get("artefact_id"),
                "artefact_type": a.metadata.get("artefact_type"),
            }
            for a in artefacts
        ],
    }
