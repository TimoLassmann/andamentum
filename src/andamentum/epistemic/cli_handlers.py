"""CLI Handlers for Epistemic System.

Async handlers for CLI commands. Designed to return structured data
that can be easily translated to web app responses.

Architecture: Layer 4 (Application)
"""

import logging
from collections import defaultdict
from typing import Dict, Any, Optional, List, Literal

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.markup import escape as rich_escape

# Old orchestrator removed - now using operations-based system
from .trace_renderers import render_timeline, render_flow, render_claims, render_all
from .config import ResearchConfig
from .result_models import (
    InitResult,
    RunResult,
    RunStats,
    ObjectiveStats,
    StatusResult,
    DebateResult,
    ClaimsResult,
    EvidenceResult,
    UncertaintiesResult,
    DecisionsResult,
    LogResult,
    AskResult,
    CleanupResult,
    ReportResult,
    VerificationEvidence,
)
from .primitives import ClaimStage, Claim, Evidence, Uncertainty
from .trace import ReasoningTrace
from andamentum.document_store import DocumentStore
from .storage import DocumentStoreAdapter
from andamentum.document_store.lifecycle import (
    delete_database,
    get_db_path,
    get_databases_dir,
)

logger = logging.getLogger(__name__)

# Type for trace display mode
TraceMode = Literal["timeline", "flow", "claims", "debate", "all", "none"]

console = Console()


async def _get_objective_id_from_db(store: DocumentStore) -> Optional[str]:
    """Get the first objective ID from the database using metadata query."""
    results = await store.find_by_metadata(
        {
            "epistemic_type": "objective",
        },
        limit=1,
    )
    if results:
        return results[0].metadata.get("objective_id")
    return None


async def _gather_primitives_from_db(
    store: DocumentStore,
    objective_id: str,
) -> tuple[List[Claim], List[Evidence], List[Uncertainty]]:
    """Gather claims, evidence, and uncertainties from the database.

    Converts document metadata back to proper primitive objects.
    Uses the from_metadata class methods on primitives for proper reconstruction.

    Args:
        store: DocumentStore instance
        objective_id: Objective ID to query

    Returns:
        Tuple of (claims, evidence, uncertainties)
    """
    # Gather claims - statement is stored in metadata
    claim_docs = await store.find_by_metadata(
        {
            "epistemic_type": "claim",
            "objective_id": objective_id,
        }
    )
    claims: List[Claim] = []
    for doc_meta in claim_docs:
        meta = doc_meta.metadata  # DocumentMetadata.metadata is the nested dict
        if not meta:
            continue
        # Use from_metadata for proper reconstruction
        claims.append(Claim.from_metadata(meta))

    # Sort by stage priority (actionable > robust > provisional > supported > hypothesis)
    stage_priority = {
        ClaimStage.ACTIONABLE: 0,
        ClaimStage.ROBUST: 1,
        ClaimStage.PROVISIONAL: 2,
        ClaimStage.SUPPORTED: 3,
        ClaimStage.HYPOTHESIS: 4,
    }
    claims.sort(key=lambda c: stage_priority.get(c.stage, 5))

    # Gather evidence - need to read full documents for content
    evidence_meta_docs = await store.find_by_metadata(
        {
            "epistemic_type": "evidence",
            "objective_id": objective_id,
        }
    )
    evidence: List[Evidence] = []
    for doc_meta in evidence_meta_docs:
        meta = doc_meta.metadata
        if not meta:
            continue
        # Read full document — content is the full entity JSON
        full_doc = await store.read(doc_meta.doc_id)
        content = ""
        if full_doc and full_doc.content:
            import json

            try:
                entity_data = json.loads(full_doc.content)
                content = entity_data.get("extracted_content", "")
            except (json.JSONDecodeError, TypeError):
                content = full_doc.content
        # Use from_metadata for proper reconstruction
        evidence.append(Evidence.from_metadata(meta, content=content))

    # Gather uncertainties - need to read full documents for description
    uncertainty_meta_docs = await store.find_by_metadata(
        {
            "epistemic_type": "uncertainty",
            "objective_id": objective_id,
        }
    )
    uncertainties: List[Uncertainty] = []
    for doc_meta in uncertainty_meta_docs:
        meta = doc_meta.metadata
        if not meta:
            continue
        # Read full document — content is the full entity JSON
        full_doc = await store.read(doc_meta.doc_id)
        description = ""
        if full_doc and full_doc.content:
            # Content is JSON — extract the description field
            import json

            try:
                entity_data = json.loads(full_doc.content)
                description = entity_data.get("description", "")
            except (json.JSONDecodeError, TypeError):
                description = full_doc.content
        # Use from_metadata for proper reconstruction
        uncertainties.append(Uncertainty.from_metadata(meta, description=description))

    return claims, evidence, uncertainties


async def _get_synthesis_from_artefact(
    store: DocumentStore,
    objective_id: str,
) -> Dict[str, Any]:
    """Get synthesis content from the artefact.

    The artefact is the ONE canonical output of the epistemic system.
    It contains the full research report as markdown.

    Args:
        store: DocumentStore instance
        objective_id: Objective ID to query

    Returns:
        Synthesis dict with the full artefact content and extracted confidence
    """
    artefact_docs = await store.find_by_metadata(
        {
            "epistemic_type": "artefact",
            "objective_id": objective_id,
        }
    )

    if not artefact_docs:
        return {}

    # Get the first artefact - need full document for content
    doc_meta = artefact_docs[0]
    full_doc = await store.read(doc_meta.doc_id)
    content = ""
    if full_doc and full_doc.content:
        # Content is JSON — extract the markdown content field
        import json

        try:
            entity_data = json.loads(full_doc.content)
            content = entity_data.get("content", "")
        except (json.JSONDecodeError, TypeError):
            content = full_doc.content

    # Extract confidence from the header line: > **Confidence:** HIGH | ...
    confidence = "medium"
    import re

    match = re.search(
        r"\*\*Confidence:\*\*\s+(HIGH|MEDIUM|LOW|NONE)", content, re.IGNORECASE
    )
    if match:
        confidence = match.group(1).lower()

    return {
        "summary": content,
        "confidence": confidence,
    }


async def handle_init(
    name: str,
    objective: str,
    artefact_specs: Optional[List[str]] = None,
    *,
    model: str,
    verbose: bool = False,
) -> InitResult:
    """Initialize a new epistemic project.

    Creates a database with an objective and initial workitem ready for
    the scheduler to process.

    Args:
        name: Project/database name
        objective: Research objective description
        artefact_specs: (Legacy, ignored) Expected deliverables
        model: (Legacy, ignored) Model to use
        verbose: Print detailed output

    Returns:
        InitResult with objective_id and status
    """
    import uuid
    from datetime import datetime
    from .primitives import (
        WorkItemType,
        WorkItemStatus,
    )

    # Initialize database
    store = DocumentStore.for_database(name)
    await store.initialize()

    # Create objective
    objective_id = f"obj_{uuid.uuid4().hex[:12]}"

    # Store the objective
    metadata = {
        "epistemic_type": "objective",
        "objective_id": objective_id,
        "status": "active",
        "created_at": datetime.now().isoformat(),
    }
    await store.add(
        file_path=f"objective_{objective_id[:8]}.json",
        content=objective,
        title=f"Objective: {objective[:50]}...",
        metadata=metadata,
    )

    # Create initial workitem (start with CLARIFY_QUESTION)
    workitem_id = f"wi_{uuid.uuid4().hex[:12]}"
    wi_metadata = {
        "epistemic_type": "workitem",
        "workitem_id": workitem_id,
        "objective_id": objective_id,
        "operation_type": WorkItemType.CLARIFY_QUESTION.value,
        "workitem_status": WorkItemStatus.QUEUED.value,
        "inputs": {"query": objective},
        "dependencies": [],
        "priority": 5,
        "retry_count": 0,
        "created_at": datetime.now().isoformat(),
    }
    import json

    await store.add(
        file_path=f"workitem_{workitem_id[:8]}.json",
        content=json.dumps({"query": objective}, indent=2),
        title=f"WorkItem: {WorkItemType.CLARIFY_QUESTION.value}",
        metadata=wi_metadata,
    )

    if verbose:
        console.print(f"[green]✓ Created epistemic project: {name}[/green]")
        console.print(f"[dim]Objective ID: {objective_id}[/dim]")
        console.print(f"[dim]Description: {objective}[/dim]")
        console.print(f"[dim]Initial workitem: {workitem_id} (CLARIFY_QUESTION)[/dim]")

    return InitResult(
        success=True,
        objective_id=objective_id,
        database_name=name,
        description=objective,
    )


async def handle_run(
    name: str,
    objective_id: Optional[str] = None,
    max_items: Optional[int] = None,
    max_retries: int = 3,
    *,
    model: str,
    verbose: bool = False,
) -> RunResult:
    """Run the epistemic scheduler on an existing project.

    Uses the database-centric operations architecture where each operation:
    1. Reads what it needs from the database
    2. Runs its agent with appropriate context
    3. Writes results back to the database
    4. Creates follow-up workitems as needed

    Args:
        name: Project/database name
        objective_id: Specific objective to run (optional, uses first if not specified)
        max_items: Maximum scheduler iterations
        max_retries: (Legacy, ignored) Maximum retries
        model: (Legacy, ignored) Model to use
        verbose: Print detailed output

    Returns:
        RunResult with execution stats
    """
    from .operations_runner import run_research_question

    if verbose:
        console.print(f"[cyan]Running epistemic scheduler for: {name}[/cyan]")
        if objective_id:
            console.print(f"[dim]Objective: {objective_id}[/dim]")

    # Get objective description from database
    store = DocumentStore.for_database(name)
    await store.initialize()

    if not objective_id:
        objective_id = await _get_objective_id_from_db(store)
        if not objective_id:
            return RunResult(
                success=False, error="No objectives found. Run 'epistemic init' first."
            )

    # Get the objective's question
    objective_docs = await store.find_by_metadata(
        {
            "epistemic_type": "objective",
            "objective_id": objective_id,
        },
        limit=1,
    )
    question = (
        objective_docs[0].metadata.get("description", "") if objective_docs else ""
    )

    # Run the pattern scheduler
    from .providers.openalex import OpenAlexQualityScorer

    scheduler_result = await run_research_question(
        question=question or "Research objective",
        database_name=name,
        max_iterations=max_items or 50,
        verbose=verbose,
        quality_scorer=OpenAlexQualityScorer(),
    )

    if verbose:
        console.print(f"  Iterations: {scheduler_result.iterations}")
        console.print(f"  Successful: {scheduler_result.successful}")
        console.print(f"  Failed: {scheduler_result.failed}")
        if scheduler_result.errors:
            console.print("[yellow]Errors:[/yellow]")
            for err in scheduler_result.errors[:5]:
                console.print(f"  - {err}")

    # Gather stats from database
    claims, evidence, uncertainties = await _gather_primitives_from_db(
        store, objective_id
    )

    # Build claims_by_stage
    claims_by_stage: Dict[str, int] = {}
    for claim in claims:
        stage_str = claim.stage.value
        claims_by_stage[stage_str] = claims_by_stage.get(stage_str, 0) + 1

    # Convert to typed RunStats model
    stats = RunStats(
        workitems_executed_this_run=scheduler_result.iterations,
        total_workitems=scheduler_result.iterations,
        workitems_by_status={
            "completed": scheduler_result.successful,
            "failed": scheduler_result.failed,
        },
        claims_by_stage=claims_by_stage,
        evidence_count=len(evidence),
        uncertainty_count=len(uncertainties),
        synthesis=await _get_synthesis_from_artefact(store, objective_id),
        reasoning_trace=None,
    )

    return RunResult(
        success=scheduler_result.success,
        objective_id=objective_id,
        stats=stats,
    )


async def handle_status(
    name: str,
    objective_id: Optional[str] = None,
    show_claims: bool = False,
    show_workitems: bool = False,
    verbose: bool = False,
) -> StatusResult:
    """Get status of an epistemic project.

    Args:
        name: Project/database name
        objective_id: Specific objective (optional)
        show_claims: Include claims summary
        show_workitems: Include workitems summary
        verbose: Print detailed output

    Returns:
        StatusResult with status information
    """
    store = DocumentStore.for_database(name)
    await store.initialize()

    from .stats import get_objective_stats as _get_stats

    # Get objective
    if not objective_id:
        objective_id = await _get_objective_id_from_db(store)
        if not objective_id:
            return StatusResult(success=False, error="No objectives found")

    stats_dict = await _get_stats(store, objective_id)

    # Always print for CLI (verbose controls detail level)
    _print_status(stats_dict, show_claims, show_workitems, verbose)

    # Convert dict to typed ObjectiveStats model
    stats = ObjectiveStats(
        objective_id=stats_dict.get("objective_id", objective_id),
        evidence_count=stats_dict.get("evidence_count", 0),
        claims_by_stage=stats_dict.get("claims_by_stage", {}),
        uncertainties_unresolved=stats_dict.get("uncertainties_unresolved", 0),
        uncertainties_resolved=stats_dict.get("uncertainties_resolved", 0),
        decisions_active=stats_dict.get("decisions_active", 0),
        decisions_reversed=stats_dict.get("decisions_reversed", 0),
        workitems_queued=stats_dict.get("workitems_queued", 0),
        workitems_done=stats_dict.get("workitems_done", 0),
        workitems_failed=stats_dict.get("workitems_failed", 0),
        snapshots=stats_dict.get("snapshots", 0),
        artefacts=stats_dict.get("artefacts", 0),
    )

    return StatusResult(
        success=True,
        objective_id=objective_id,
        stats=stats,
    )


async def handle_debate(
    name: str,
    objective_id: Optional[str] = None,
    output_format: str = "cli",
    *,
    model: str,
) -> DebateResult:
    """View adversarial debate summary for an epistemic project.

    Note: The debate feature has been removed. Use handle_status or handle_ask
    to view research results including adversarial analysis.
    """
    console.print("[yellow]The 'debate' command has been removed.[/yellow]")
    console.print(
        "[dim]Use 'status' to view claims and adversarial balance, or 'ask' to run a new research question.[/dim]"
    )
    return DebateResult(
        success=False, error="Debate feature removed. Use 'status' or 'ask' instead."
    )


async def handle_claims(
    name: str,
    stage: Optional[str] = None,
    objective_id: Optional[str] = None,
    verbose: bool = False,
) -> ClaimsResult:
    """List claims for an epistemic project.

    Args:
        name: Project/database name
        stage: Filter by stage (hypothesis, supported, provisional, robust, actionable)
        objective_id: Specific objective (optional)
        verbose: Print detailed output

    Returns:
        ClaimsResult with claims list
    """
    store = DocumentStore.for_database(name)
    await store.initialize()

    from .repository import EpistemicRepository

    repo = EpistemicRepository(DocumentStoreAdapter(store))

    # Get objective
    if not objective_id:
        objective_id = await _get_objective_id_from_db(store)
        if not objective_id:
            return ClaimsResult(success=False, error="No objectives found")

    # Parse stage filter
    stage_filter = None
    if stage:
        try:
            stage_filter = ClaimStage(stage)
        except ValueError:
            return ClaimsResult(success=False, error=f"Invalid stage: {stage}")

    if stage_filter:
        claims = await repo.get_claims_for_objective(
            objective_id, stage=stage_filter.value
        )
    else:
        claims = await repo.get_claims_for_objective(objective_id)

    # Always print for CLI (verbose controls detail level)
    _print_claims(claims, stage, verbose)

    return ClaimsResult(
        success=True,
        objective_id=objective_id,
        claims=claims,
        count=len(claims),
    )


async def handle_evidence(
    name: str,
    claim_id: Optional[str] = None,
    objective_id: Optional[str] = None,
    verbose: bool = False,
    include_verification: bool = True,
) -> EvidenceResult:
    """List evidence for an epistemic project.

    Args:
        name: Project/database name
        claim_id: Filter evidence for specific claim
        objective_id: Specific objective (optional)
        verbose: Print detailed output
        include_verification: Include verification evidence (computational, adversarial, etc.)

    Returns:
        EvidenceResult with evidence list
    """
    store = DocumentStore.for_database(name)
    await store.initialize()

    from .repository import EpistemicRepository
    from .stats import get_all_verification_evidence

    repo = EpistemicRepository(DocumentStoreAdapter(store))

    # Get objective
    if not objective_id:
        objective_id = await _get_objective_id_from_db(store)
        if not objective_id:
            return EvidenceResult(success=False, error="No objectives found")

    evidence = await repo.get_evidence_for_objective(objective_id)

    # Filter by claim if specified
    if claim_id:
        claim = await repo.get_claim(claim_id)
        if claim:
            evidence = [e for e in evidence if e.evidence_id in claim.evidence_ids]

    # Get verification evidence if requested
    verification_evidence_dict: Optional[Dict[str, List[Any]]] = None
    if include_verification:
        verification_evidence_dict = await get_all_verification_evidence(
            store, objective_id
        )
        # Filter verification evidence by claim if specified
        if claim_id:
            for key in verification_evidence_dict:
                verification_evidence_dict[key] = [
                    e
                    for e in verification_evidence_dict[key]
                    if hasattr(e, "claim_id") and e.claim_id == claim_id
                ]

    # Always print for CLI (verbose controls detail level)
    _print_evidence(evidence, verbose)
    if verification_evidence_dict:
        _print_verification_evidence(verification_evidence_dict, verbose)

    # Build VerificationEvidence model if we have verification data
    verification_model: Optional[VerificationEvidence] = None
    verification_count = 0
    if include_verification and verification_evidence_dict:
        verification_model = VerificationEvidence(
            adversarial=verification_evidence_dict.get("adversarial", []),
            computational=verification_evidence_dict.get("computational", []),
            convergent=verification_evidence_dict.get("convergent", []),
            temporal=verification_evidence_dict.get("temporal", []),
            deductive=verification_evidence_dict.get("deductive", []),
        )
        verification_count = sum(len(v) for v in verification_evidence_dict.values())

    return EvidenceResult(
        success=True,
        objective_id=objective_id,
        evidence=evidence,
        count=len(evidence),
        verification_evidence=verification_model,
        verification_count=verification_count,
    )


async def handle_uncertainties(
    name: str,
    blocking_only: bool = False,
    objective_id: Optional[str] = None,
    verbose: bool = False,
) -> UncertaintiesResult:
    """List uncertainties for an epistemic project.

    Args:
        name: Project/database name
        blocking_only: Only show unresolved blocking uncertainties
        objective_id: Specific objective (optional)
        verbose: Print detailed output

    Returns:
        UncertaintiesResult with uncertainties list
    """
    store = DocumentStore.for_database(name)
    await store.initialize()

    from .repository import EpistemicRepository

    repo = EpistemicRepository(DocumentStoreAdapter(store))

    # Get objective
    if not objective_id:
        objective_id = await _get_objective_id_from_db(store)
        if not objective_id:
            return UncertaintiesResult(success=False, error="No objectives found")

    if blocking_only:
        uncertainties = await repo.get_blocking_uncertainties(objective_id)
    else:
        uncertainties = await repo.get_uncertainties_for_objective(objective_id)

    # Always print for CLI (verbose controls detail level)
    _print_uncertainties(uncertainties, blocking_only, verbose)

    return UncertaintiesResult(
        success=True,
        objective_id=objective_id,
        uncertainties=uncertainties,
        count=len(uncertainties),
    )


async def handle_decisions(
    name: str,
    include_reversed: bool = False,
    objective_id: Optional[str] = None,
    verbose: bool = False,
) -> DecisionsResult:
    """List decisions for an epistemic project.

    Args:
        name: Project/database name
        include_reversed: Include reversed decisions
        objective_id: Specific objective (optional)
        verbose: Print detailed output

    Returns:
        DecisionsResult with decisions list
    """
    store = DocumentStore.for_database(name)
    await store.initialize()

    from .repository import EpistemicRepository

    repo = EpistemicRepository(DocumentStoreAdapter(store))

    # Get objective
    if not objective_id:
        objective_id = await _get_objective_id_from_db(store)
        if not objective_id:
            return DecisionsResult(success=False, error="No objectives found")

    decisions = await repo.get_decisions_for_objective(objective_id, include_reversed)

    # Always print for CLI (verbose controls detail level)
    _print_decisions(decisions, include_reversed, verbose)

    return DecisionsResult(
        success=True,
        objective_id=objective_id,
        decisions=decisions,
        count=len(decisions),
    )


async def handle_log(
    name: str,
    limit: int = 50,
    objective_id: Optional[str] = None,
    verbose: bool = False,
) -> LogResult:
    """View audit log for an epistemic project.

    Note: The event log feature is not available in the standalone package.
    Use 'status' to view project information.
    """
    console.print(
        "[yellow]The 'log' command is not available in the standalone package.[/yellow]"
    )
    console.print("[dim]Use 'status' to view project information.[/dim]")
    return LogResult(
        success=False, error="Log feature not available. Use 'status' instead."
    )


async def _print_operation_profile(store: "DocumentStore") -> None:
    """Print a timing profile of operations from execution_step documents.

    Reads the execution_step metadata already stored by the operations runner
    and aggregates by operation type. No new recording needed.
    """
    try:
        steps = await store.find_by_metadata(
            {"epistemic_type": "execution_step"}, limit=500
        )
    except Exception:
        return  # Silently skip if store doesn't support this query

    if not steps:
        return

    # Aggregate by operation
    op_stats: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {"calls": 0, "total_ms": 0, "failures": 0}
    )

    for step in steps:
        meta: dict[str, Any] = (
            step.metadata
            if isinstance(step.metadata, dict)
            else getattr(step, "metadata", {})
        )
        # DocumentStore returns DocumentMetadata objects with a .metadata dict
        if hasattr(meta, "metadata"):
            meta = getattr(meta, "metadata")
        op = meta.get("operation", "unknown")
        duration = meta.get("duration_ms", 0)
        success = meta.get("success", True)

        op_stats[op]["calls"] += 1
        op_stats[op]["total_ms"] += duration
        if not success:
            op_stats[op]["failures"] += 1

    if not op_stats:
        return

    # Sort by total time descending
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["total_ms"], reverse=True)

    total_time_ms = sum(s["total_ms"] for s in op_stats.values())
    total_calls = sum(s["calls"] for s in op_stats.values())

    table = Table(
        title="Operation Profile", border_style="dim", show_edge=False, pad_edge=False
    )
    table.add_column("Operation", style="bold", min_width=28)
    table.add_column("Calls", justify="right", style="cyan", min_width=5)
    table.add_column("Total", justify="right", min_width=8)
    table.add_column("Avg", justify="right", min_width=8)
    table.add_column("% Time", justify="right", min_width=7)
    table.add_column("Fail", justify="right", min_width=4)

    for op_name, stats in sorted_ops:
        calls = int(stats["calls"])
        total_ms = int(stats["total_ms"])
        avg_ms = total_ms // calls if calls > 0 else 0
        pct = (total_ms / total_time_ms * 100) if total_time_ms > 0 else 0
        failures = int(stats["failures"])

        # Format durations
        if total_ms >= 60_000:
            total_str = f"{total_ms / 60_000:.1f}m"
        elif total_ms >= 1_000:
            total_str = f"{total_ms / 1_000:.1f}s"
        else:
            total_str = f"{total_ms}ms"

        if avg_ms >= 1_000:
            avg_str = f"{avg_ms / 1_000:.1f}s"
        else:
            avg_str = f"{avg_ms}ms"

        pct_str = f"{pct:.0f}%"
        fail_str = f"[red]{failures}[/red]" if failures > 0 else "[dim]0[/dim]"

        # Highlight slow operations
        style = "bold yellow" if pct >= 25 else ""
        table.add_row(
            op_name, str(calls), total_str, avg_str, pct_str, fail_str, style=style
        )

    # Footer
    if total_time_ms >= 60_000:
        total_str = f"{total_time_ms / 60_000:.1f}m"
    elif total_time_ms >= 1_000:
        total_str = f"{total_time_ms / 1_000:.1f}s"
    else:
        total_str = f"{total_time_ms}ms"

    table.add_section()
    table.add_row("TOTAL", str(total_calls), total_str, "", "100%", "", style="bold")

    console.print()
    console.print(table)


async def handle_ask(
    question: str,
    name: Optional[str] = None,
    max_items: int = 50,
    max_retries: int = 3,
    *,
    model: str,
    embedding_model: Optional[str] = None,
    keep: bool = False,
    verbose: bool = False,
    trace: TraceMode = "timeline",
    evidence_agent: Optional[str] = None,
    force_quick: bool = False,
    research_config: Optional[ResearchConfig] = None,
    output_path: Optional[str] = None,
    provider: str = "all",
    db_dir: Optional[str] = None,
) -> AskResult:
    """Ask a research question and get validated findings.

    This is the primary interface to the epistemic system. It uses the
    database-centric operations architecture where each operation:
    1. Reads what it needs from the database
    2. Runs its agent with appropriate context
    3. Writes results back to the database
    4. Creates follow-up workitems as needed

    The full pipeline:
        CLARIFY_QUESTION → CONCEPTUAL_ANALYSIS → PLAN_TASK → COLLECT_EVIDENCE →
        EXTRACT_EVIDENCE → PROPOSE_CLAIMS → SCRUTINISE_CLAIM → [verification] →
        PROMOTE_CLAIM → FREEZE_SNAPSHOT → SYNTHESIZE_REPORT

    Args:
        question: Research question to investigate
        name: Project name (auto-generated if not specified)
        max_items: Maximum scheduler iterations
        max_retries: (Legacy, ignored) Maximum retries per workitem
        model: LLM model to use (e.g., "anthropic:claude-haiku-4-5", "openai:gpt-4o-mini")
        keep: Keep project database after completion
        verbose: Print detailed progress
        trace: Trace visualization mode (timeline, flow, claims, all, none)
        evidence_agent: (Legacy, ignored) Evidence agent override
        force_quick: Skip preplanning (clarification + conceptual analysis)
        research_config: (Legacy, ignored) Research configuration
        output_path: Path to save HTML report

    Returns:
        AskResult with findings, claims, evidence, and uncertainties
    """
    import uuid
    import logging
    from typing import Any as TypingAny
    from .operations_runner import run_research_question

    # Suppress noisy logs during spinner (restore after)
    root_logger = logging.getLogger()
    original_level = root_logger.level

    # Auto-generate project name if not specified
    if not name:
        name = f"ask_{uuid.uuid4().hex[:8]}"

    # Progress callback for real-time updates
    step_count = [0]  # Use list to allow mutation in closure

    def progress_callback(
        operation_type: str,
        workitem_id: str,
        success: bool,
        message: str,
        outputs: dict[str, TypingAny],
    ) -> None:
        """Display progress for each completed operation."""
        step_count[0] += 1

        # Map operation types to short display names
        op_display = {
            "clarify_question": "Clarify",
            "conceptual_analysis": "Analyze",
            "plan_task": "Plan",
            "collect_evidence": "Collect",
            "extract_evidence": "Extract",
            "propose_claims": "Propose",
            "scrutinise_claim": "Scrutinize",
            "promote_claim": "Promote",
            "demote_claim": "Demote",
            "freeze_snapshot": "Snapshot",
            "synthesize_report": "Synthesize",
            "verify_computationally": "Verify",
            "adversarial_search": "Challenge",
            "assess_convergence": "Converge",
            "resolve_uncertainty": "Resolve",
        }.get(operation_type, operation_type.replace("_", " ").title())

        if success:
            console.print(f"  [green]✓[/green] [bold]{op_display}[/bold]: {message}")
        else:
            console.print(f"  [red]✗[/red] [bold]{op_display}[/bold]: {message}")

    # Always show the question header
    console.print()
    console.print(
        Panel(
            f"[bold]{question}[/bold]",
            title="[bold cyan]Epistemic Analysis[/bold cyan]",
            subtitle=f"[dim]project: {name} | max iterations: {max_items}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if force_quick:
        console.print("[dim]Skipping preplanning (starting at PLAN_TASK)[/dim]")

    console.print()
    console.print("[bold blue]Research Pipeline[/bold blue]")

    try:
        # Suppress INFO logs - keep WARNING level so important issues still surface
        root_logger.setLevel(logging.WARNING)

        # Always use progress callback for visibility
        from .providers.openalex import OpenAlexQualityScorer

        # Derive operation budgets from research config.
        # When principled budgets are active, max_iterations is only
        # an emergency backstop — the per-operation budgets control scope.
        if research_config is None:
            research_config = ResearchConfig.light()
        op_budgets = research_config.operation_budgets()

        # Only apply max_items if the user explicitly set it on the CLI.
        # Otherwise let the per-operation budgets be the sole scope control.
        iterations_limit = max_items if max_items != 50 else 500

        scheduler_result = await run_research_question(
            question=question,
            database_name=name,
            max_iterations=iterations_limit,
            verbose=verbose,
            skip_preplanning=force_quick,
            model=model,
            embedding_model=embedding_model,
            progress_callback=progress_callback,
            provider=provider,
            db_dir=db_dir,
            quality_scorer=OpenAlexQualityScorer(),
            operation_budgets=op_budgets,
        )

        console.print()
        if scheduler_result.failed > 0:
            console.print(
                f"[yellow]Completed {scheduler_result.successful} operations, {scheduler_result.failed} failed[/yellow]"
            )
        else:
            console.print(
                f"[green]Completed {scheduler_result.successful} operations[/green]"
            )

        # Gather results from the database
        store = DocumentStore.for_database(name, db_dir=db_dir)
        await store.initialize()

        # Show operation timing profile
        await _print_operation_profile(store)

        objective_id = await _get_objective_id_from_db(store)
        if not objective_id:
            raise ValueError("No objective found in database")

        if verbose:
            console.print(f"  [green]✓[/green] Objective: {objective_id[:12]}...")

        # Gather primitives
        if verbose:
            all_claims, evidence, uncertainties = await _gather_primitives_from_db(
                store, objective_id
            )
        else:
            with console.status(
                "[bold blue]Gathering findings...[/bold blue]", spinner="dots"
            ):
                all_claims, evidence, uncertainties = await _gather_primitives_from_db(
                    store, objective_id
                )

        # Get synthesis from artefact
        synthesis = await _get_synthesis_from_artefact(store, objective_id)

        # Restore logging level before displaying results
        root_logger.setLevel(original_level)

        # Build stats dict for display compatibility
        stats: Dict[str, Any] = {
            "workitems_executed_this_run": scheduler_result.iterations,
            "total_workitems": scheduler_result.iterations,
            "operations_successful": scheduler_result.successful,
            "operations_failed": scheduler_result.failed,
            "workitems_by_status": {
                "completed": scheduler_result.successful,
                "failed": scheduler_result.failed,
            },
            "claims_by_stage": {},
            "evidence_count": len(evidence),
            "uncertainty_count": len(uncertainties),
            "synthesis": synthesis if synthesis else None,
        }

        # Count claims by stage
        for claim in all_claims:
            stage_str = claim.stage.value
            stats["claims_by_stage"][stage_str] = (
                stats["claims_by_stage"].get(stage_str, 0) + 1
            )

        # Display results (always, not just verbose)
        _print_ask_results(
            question,
            all_claims,
            evidence,
            uncertainties,
            stats,
            verbose,
            trace,
            None,
            answer_confidence=getattr(scheduler_result, "answer_confidence", None),
            posterior=getattr(scheduler_result, "posterior", None),
        )

        # Generate HTML report if output path specified
        if output_path:
            from pathlib import Path
            from .report_generator import ReportGenerator

            output_file = Path(output_path)
            generator = ReportGenerator(store, name)
            if await generator.save_html(
                output_path=output_file,
                objective_id=objective_id,
                model_name=model,
            ):
                console.print(
                    f"\n[green]✓[/green] HTML report saved to: [bold]{output_file}[/bold]"
                )

                # Also generate typeset version for comparison
                try:
                    report_data = await generator.extract_report_data(
                        objective_id=objective_id,
                        model_name=model,
                    )
                    if report_data:
                        from .typeset_report import build_typeset_report
                        from andamentum.typeset import render_to_file

                        atoms = build_typeset_report(report_data)
                        new_file = output_file.with_stem(output_file.stem + "_new")
                        render_to_file(atoms, new_file)
                        console.print(
                            f"[green]✓[/green] Typeset report saved to: [bold]{new_file}[/bold]"
                        )
                except Exception as e:
                    console.print(
                        f"\n[yellow]Warning: Typeset report failed: {e}[/yellow]"
                    )
            else:
                console.print(
                    "\n[yellow]Warning: Failed to generate HTML report[/yellow]"
                )

        # Clean up if not keeping
        if not keep:
            if delete_database(name):
                if verbose:
                    console.print("[dim]Cleaned up ephemeral project database[/dim]")
        else:
            console.print(f"\n[dim]Project saved: {name}[/dim]")
            console.print(
                f"[dim]Inspect with: andamentum-epistemic status {name} -v[/dim]"
            )

        # Convert stats dict to typed RunStats model
        typed_stats = RunStats(
            workitems_executed_this_run=scheduler_result.iterations,
            total_workitems=scheduler_result.iterations,
            workitems_by_status=stats["workitems_by_status"],
            claims_by_stage=stats["claims_by_stage"],
            evidence_count=len(evidence),
            uncertainty_count=len(uncertainties),
            synthesis=synthesis if synthesis else None,
            reasoning_trace=None,
        )

        return AskResult(
            success=scheduler_result.success,
            question=question,
            project_name=name,
            claims=all_claims,
            evidence=evidence,
            uncertainties=uncertainties,
            stats=typed_stats,
            kept=keep,
            artefact_content=synthesis.get("summary") if synthesis else None,
        )

    except Exception as e:
        # Restore logging level
        root_logger.setLevel(original_level)
        console.print(f"\n[red]Error: {rich_escape(str(e))}[/red]")

        # Clean up on error if not keeping
        if not keep:
            try:
                delete_database(name)
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup database on error: {cleanup_err}")

        return AskResult(
            success=False,
            error=str(e),
            question=question,
            project_name=name,
        )


def _print_ask_results(
    question: str,
    claims: list,
    evidence: list,
    uncertainties: list,
    stats: Dict[str, Any],
    verbose: bool = False,
    trace_mode: TraceMode = "timeline",
    reasoning_trace: Optional[ReasoningTrace] = None,
    answer_confidence: Optional[Any] = None,
    posterior: Optional[Any] = None,
) -> None:
    """Print results from an ask query in a nice format.

    Display priority:
    1. ANSWER - The synthesized findings (always shown prominently)
    2. Evidence sources summary
    3. Open questions (if any)
    4. Detailed claims table (verbose or trace mode)
    5. Execution stats (verbose only)
    6. Reasoning trace (if requested)

    Args:
        question: The original research question
        claims: List of claims found
        evidence: List of evidence items
        uncertainties: List of uncertainties
        stats: Run statistics including synthesis
        verbose: Whether to print verbose details
        trace_mode: Trace visualization mode (timeline, flow, claims, all, none)
        reasoning_trace: ReasoningTrace object for visualization
    """

    console.print()

    # Check if we have synthesis results (artefact content)
    synthesis = stats.get("synthesis", {})

    # Color code by answer confidence level
    ac_level = answer_confidence.level if answer_confidence else "none"
    color_map = {
        "high": "green",
        "moderate": "cyan",
        "low": "yellow",
        "insufficient": "red",
        "none": "red",
    }

    # === PRIMARY OUTPUT: THE ARTEFACT ===
    if synthesis:
        artefact_content = synthesis.get("summary", "")
        title_color = color_map.get(ac_level, "white")
        confidence_label = (
            f"{ac_level} confidence" if answer_confidence else "no confidence score"
        )

        if artefact_content:
            console.print(
                Panel(
                    Markdown(artefact_content),
                    title=f"[bold {title_color}]Research Report[/bold {title_color}] [dim]({confidence_label})[/dim]",
                    border_style=title_color,
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel(
                    "[dim]No findings synthesized[/dim]",
                    title=f"[bold {title_color}]Research Report[/bold {title_color}]",
                    border_style=title_color,
                    padding=(1, 2),
                )
            )

    # === SCORES: Answer confidence + Posterior ===
    if answer_confidence:
        ac = answer_confidence
        ac_color = color_map.get(ac.level, "white")
        console.print()
        console.print(
            f"[bold]Answer confidence:[/bold] [{ac_color}]{ac.confidence:.2f} ({ac.level.upper()})[/{ac_color}]"
            f"  [dim]{ac.passes}/{ac.passes + ac.failures} checks passed[/dim]"
        )
        for check in ac.checks:
            status = "[green]✓[/green]" if check.passed else "[red]✗[/red]"
            tradition = f" [dim][{check.tradition}][/dim]" if check.tradition else ""
            console.print(f"  {status} {check.name}{tradition}")

    if posterior:
        po = posterior
        console.print(
            f"[bold]Posterior confidence:[/bold] {po.posterior:.2%}"
            f"  [dim]({po.supporting_count} supporting, {po.contradicting_count} contradicting)[/dim]"
        )

        # Show errors if any (HONEST FAILURE REPORTING)
        errors = stats.get("errors", [])
        if errors:
            console.print()
            console.print("[bold yellow]Issues Encountered[/bold yellow]")
            for error in errors:
                console.print(f"  [yellow]•[/yellow] {error}")

    else:
        # Fallback: No artefact - show claims directly as the answer
        console.print(Rule("[bold green]Research Complete[/bold green]", style="green"))

        if claims:
            # Show ALL claims as the answer - never truncate (per CLAUDE.md "No Output Truncation")
            console.print()
            console.print("[bold]Key Findings:[/bold]")
            for i, claim in enumerate(claims, 1):
                stage_icon = (
                    "✓" if claim.stage.value in ("robust", "actionable") else "○"
                )
                console.print(f"  {stage_icon} {claim.statement}")
        else:
            console.print(
                Panel(
                    "[yellow]No findings established.[/yellow]\n"
                    "[dim]The research did not produce validated claims.[/dim]",
                    border_style="yellow",
                )
            )

    # === SECONDARY OUTPUT: Open questions (unresolved uncertainties only) ===
    unresolved = (
        [u for u in uncertainties if not u.is_resolved] if uncertainties else []
    )
    # Separate blocking (genuine open questions) from non-blocking (caveats)
    open_questions = [u for u in unresolved if u.is_blocking]
    caveats = [u for u in unresolved if not u.is_blocking]

    if open_questions:
        console.print()
        console.print(
            f"[bold yellow]Open Questions ({len(open_questions)})[/bold yellow]"
        )
        for i, u in enumerate(open_questions, 1):
            desc = (
                u.description.split("\n")[0] if "\n" in u.description else u.description
            )
            console.print(f"  [yellow]{i}.[/yellow] {desc}")

    if caveats:
        console.print()
        console.print(f"[bold dim]Caveats ({len(caveats)})[/bold dim]")
        for i, u in enumerate(caveats, 1):
            desc = (
                u.description.split("\n")[0] if "\n" in u.description else u.description
            )
            console.print(f"  [dim]{i}.[/dim] [dim]{desc}[/dim]")

    # === DETAILED OUTPUT: Claims table (verbose or trace mode only) ===
    show_details = verbose or trace_mode in ("claims", "all")

    if show_details and claims:
        console.print()
        claims_table = Table(
            title="[bold]Claims by Stage[/bold]",
            show_header=True,
            header_style="bold",
            border_style="blue",
            title_style="bold blue",
            expand=True,
        )
        claims_table.add_column("Stage", style="bold", width=12)
        claims_table.add_column("Claim", style="white", ratio=3)
        claims_table.add_column("Scope", style="dim", ratio=1)

        stage_styles = {
            "hypothesis": "[yellow]HYPOTHESIS[/yellow]",
            "supported": "[blue]SUPPORTED[/blue]",
            "provisional": "[cyan]PROVISIONAL[/cyan]",
            "robust": "[green]ROBUST[/green]",
            "actionable": "[bold green]ACTIONABLE[/bold green]",
        }

        for claim in claims:
            stage_display = stage_styles.get(
                claim.stage.value, claim.stage.value.upper()
            )
            scope = claim.scope if claim.scope else "-"
            claims_table.add_row(stage_display, claim.statement, scope)

        console.print(claims_table)

    # === VERBOSE OUTPUT: Execution stats ===
    if verbose and stats:
        console.print()

        # Compact stats line
        ops_ok = stats.get("operations_successful", 0)
        ops_fail = stats.get("operations_failed", 0)
        workitems = stats.get("workitems_executed_this_run", 0)

        stats_line = f"[dim]Stats: {workitems} work items, {ops_ok} ops succeeded"
        if ops_fail > 0:
            stats_line += f", [red]{ops_fail} failed[/red]"
        stats_line += "[/dim]"
        console.print(stats_line)

    # === TRACE OUTPUT: Reasoning trace visualization ===
    if trace_mode != "none":
        console.print()
        # Handle trace modes
        if trace_mode == "debate":
            console.print(
                "[dim]Debate trace mode has been removed. Use 'timeline', 'flow', 'claims', or 'all'.[/dim]"
            )
        elif reasoning_trace:
            # Standard trace modes use reasoning_trace
            if trace_mode == "timeline":
                render_timeline(reasoning_trace, console)
            elif trace_mode == "flow":
                render_flow(reasoning_trace, console)
            elif trace_mode == "claims":
                render_claims(reasoning_trace, console)
            elif trace_mode == "all":
                render_all(reasoning_trace, console)

    console.print()


# --- Display Helpers ---


def _print_run_stats(stats: Dict[str, Any]) -> None:
    """Print run statistics."""
    console.print(Panel("[bold green]Run Complete[/bold green]", border_style="green"))

    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row(
        "WorkItems Executed", str(stats.get("workitems_executed_this_run", 0))
    )
    table.add_row("Total Claims", str(stats.get("total_claims", 0)))
    table.add_row("Total Evidence", str(stats.get("total_evidence", 0)))
    table.add_row(
        "Unresolved Uncertainties", str(stats.get("uncertainties_unresolved", 0))
    )

    console.print(table)


def _print_status(
    stats: Dict[str, Any],
    show_claims: bool,
    show_workitems: bool,
    verbose: bool = False,
) -> None:
    """Print status information."""
    console.print(Panel("[bold cyan]Epistemic Status[/bold cyan]", border_style="cyan"))

    # Summary table
    table = Table(show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Claims", str(stats.get("total_claims", 0)))
    table.add_row("Total Evidence", str(stats.get("total_evidence", 0)))
    table.add_row(
        "Unresolved Uncertainties", str(stats.get("uncertainties_unresolved", 0))
    )

    console.print(table)

    # Claims by stage
    if "claims_by_stage" in stats:
        console.print("\n[bold]Claims by Stage:[/bold]")
        for stage, count in stats["claims_by_stage"].items():
            bar = "█" * count
            console.print(f"  {stage}: {bar} ({count})")


def _print_claims(
    claims: list, stage_filter: Optional[str], verbose: bool = False
) -> None:
    """Print claims list."""
    title = f"Claims ({stage_filter})" if stage_filter else "All Claims"
    console.print(
        Panel(f"[bold]{title}[/bold] - {len(claims)} total", border_style="blue")
    )

    for claim in claims:
        stage_color = {
            "hypothesis": "yellow",
            "supported": "blue",
            "provisional": "cyan",
            "robust": "green",
            "actionable": "bold green",
        }.get(claim.stage.value, "white")

        console.print(
            f"\n[{stage_color}][{claim.stage.value}][/{stage_color}] {claim.claim_id[:12]}..."
        )
        console.print(f"  {claim.statement}")
        console.print(f"  [dim]Scope: {claim.scope}[/dim]")
        console.print(
            f"  [dim]Evidence: {len(claim.evidence_ids)} | Uncertainties: {len(claim.uncertainty_ids)}[/dim]"
        )


def _print_evidence(evidence: list, verbose: bool = False) -> None:
    """Print evidence list."""
    console.print(
        Panel(f"[bold]Evidence[/bold] - {len(evidence)} total", border_style="green")
    )

    for ev in evidence:
        console.print(f"\n[cyan][{ev.source_type}][/cyan] {ev.evidence_id[:12]}...")
        console.print(f"  Source: {ev.source_ref}")
        if ev.extracted_content:
            # Show ALL extracted content - never truncate (per CLAUDE.md "No Output Truncation")
            console.print(f"  [dim]{ev.extracted_content}[/dim]")


def _print_verification_evidence(verification: dict, verbose: bool = False) -> None:
    """Print verification evidence (computational, adversarial, convergent, temporal)."""
    total = sum(len(v) for v in verification.values())
    if total == 0:
        return

    console.print(
        Panel(
            f"[bold]Verification Evidence[/bold] - {total} total",
            border_style="magenta",
        )
    )

    # Computational evidence
    for ev in verification.get("computational", []):
        verdict_color = (
            "green"
            if ev.final_verdict == "SUPPORTED"
            else "red"
            if ev.final_verdict == "REFUTED"
            else "bright_yellow"
            if ev.final_verdict == "TEST_FAILED"
            # Weaker than REFUTED
            else "yellow"
        )
        console.print(f"\n[magenta][COMPUTATIONAL][/magenta] {ev.evidence_id[:12]}...")
        console.print(f"  Claim: {ev.claim_id[:12]}...")
        console.print(
            f"  Verdict: [{verdict_color}]{ev.final_verdict}[/{verdict_color}] (confidence: {ev.confidence:.2f})"
        )
        if ev.reproducible:
            console.print("  [green]✓ Reproducible[/green]")
        if ev.explanation:
            # Show ALL explanation - never truncate (per CLAUDE.md "No Output Truncation")
            console.print(f"  [dim]{ev.explanation}[/dim]")

    # Adversarial evidence
    for ev in verification.get("adversarial", []):
        verdict_color = (
            "green"
            if ev.verdict == "SUPPORTED"
            else "red"
            if ev.verdict == "REFUTED"
            else "yellow"
        )
        console.print(f"\n[magenta][ADVERSARIAL][/magenta] {ev.evidence_id[:12]}...")
        console.print(f"  Claim: {ev.claim_id[:12]}...")
        console.print(
            f"  Verdict: [{verdict_color}]{ev.verdict}[/{verdict_color}] (balance: {ev.adversarial_balance:.2f})"
        )
        if ev.counterarguments:
            console.print(f"  Counterarguments: {len(ev.counterarguments)}")
            # Show ALL counterarguments - never truncate (per CLAUDE.md "No Output Truncation")
            for ca in ev.counterarguments:
                console.print(f"    - [{ca.category.value}] {ca.summary}")
        console.print(f"  Recommendation: {ev.recommendation}")

    # Convergent evidence
    for ev in verification.get("convergent", []):
        verdict_color = (
            "green"
            if ev.verdict == "CONVERGENT"
            else "yellow"
            if ev.verdict == "PARTIAL"
            else "red"
        )
        console.print(f"\n[magenta][CONVERGENT][/magenta] {ev.evidence_id[:12]}...")
        console.print(f"  Claim: {ev.claim_id[:12]}...")
        console.print(
            f"  Verdict: [{verdict_color}]{ev.verdict}[/{verdict_color}] (strength: {ev.convergence_strength:.2f})"
        )
        console.print(f"  Independent domains: {ev.num_independent_domains}")
        if ev.domain_clusters:
            # Show ALL domain clusters - never truncate (per CLAUDE.md "No Output Truncation")
            for cluster in ev.domain_clusters:
                label = cluster.cluster_label or f"Cluster {cluster.cluster_id[:8]}"
                console.print(f"    - {label}: {cluster.cluster_size} items")

    # Temporal evidence
    for ev in verification.get("temporal", []):
        verdict_color = (
            "green"
            if ev.verdict == "CONFIRMED"
            else "red"
            if ev.verdict == "REFUTED"
            else "yellow"
        )
        console.print(f"\n[magenta][TEMPORAL][/magenta] {ev.evidence_id[:12]}...")
        console.print(f"  Claim: {ev.claim_id[:12]}...")
        console.print(f"  Verdict: [{verdict_color}]{ev.verdict}[/{verdict_color}]")
        console.print(
            f"  Predictions: {ev.resolved_predictions}/{ev.total_predictions} resolved"
        )
        console.print(f"  Confirmed: {ev.confirmed_count}, Refuted: {ev.refuted_count}")
        if ev.confirmation_rate > 0:
            console.print(f"  Confirmation rate: {ev.confirmation_rate:.1%}")


def _print_uncertainties(
    uncertainties: list, blocking_only: bool, verbose: bool = False
) -> None:
    """Print uncertainties list grouped by blocking/non-blocking and resolution status."""
    title = "Blocking Uncertainties" if blocking_only else "All Uncertainties"
    console.print(
        Panel(
            f"[bold]{title}[/bold] - {len(uncertainties)} total", border_style="yellow"
        )
    )

    # Group uncertainties
    blocking_unresolved = [
        u for u in uncertainties if u.is_blocking and not u.resolved_at
    ]
    non_blocking_unresolved = [
        u for u in uncertainties if not u.is_blocking and not u.resolved_at
    ]
    resolved = [u for u in uncertainties if u.resolved_at]

    if blocking_unresolved:
        console.print("\n[bold red]Blocking (unresolved)[/bold red]")
        for u in blocking_unresolved:
            console.print(
                f"\n[red]✗[/red] [{u.uncertainty_type.value}] {u.uncertainty_id[:12]}..."
            )
            console.print(f"  {u.description}")
            if u.affected_claim_ids:
                console.print(
                    f"  [dim]Affects: {len(u.affected_claim_ids)} claims[/dim]"
                )

    if non_blocking_unresolved:
        console.print("\n[bold yellow]Non-blocking (caveats)[/bold yellow]")
        for u in non_blocking_unresolved:
            console.print(
                f"\n[yellow]~[/yellow] [{u.uncertainty_type.value}] {u.uncertainty_id[:12]}..."
            )
            console.print(f"  {u.description}")
            if u.affected_claim_ids:
                console.print(
                    f"  [dim]Affects: {len(u.affected_claim_ids)} claims[/dim]"
                )

    if resolved:
        console.print(f"\n[bold green]Resolved ({len(resolved)})[/bold green]")
        for u in resolved:
            console.print(
                f"\n[green]✓[/green] [{u.uncertainty_type.value}] {u.uncertainty_id[:12]}..."
            )
            console.print(f"  {u.description}")
            if u.resolution:
                console.print(f"  [dim italic]Resolution: {u.resolution}[/dim italic]")


def _print_decisions(
    decisions: list, include_reversed: bool, verbose: bool = False
) -> None:
    """Print decisions list."""
    title = "All Decisions" if include_reversed else "Active Decisions"
    console.print(
        Panel(
            f"[bold]{title}[/bold] - {len(decisions)} total", border_style="bright_cyan"
        )
    )

    for d in decisions:
        # Status indicator
        if d.reversed_at:
            status = "[red]REVERSED[/red]"
            style = "dim"
        else:
            status = "[green]ACTIVE[/green]"
            style = "white"

        reversible = (
            "[cyan]reversible[/cyan]"
            if d.reversible
            else "[yellow]irreversible[/yellow]"
        )

        console.print(f"\n{status} {d.decision_id[:12]}... ({reversible})")
        console.print(f"  [{style}]{d.statement}[/{style}]")
        # Show ALL claim IDs - never truncate (per CLAUDE.md "No Output Truncation")
        console.print(
            f"  [dim]Based on claims: {', '.join(cid[:8] for cid in d.claim_ids)}[/dim]"
        )

        if d.justification:
            # Show ALL justification - never truncate (per CLAUDE.md "No Output Truncation")
            console.print(f"  [dim italic]{d.justification}[/dim italic]")

        if d.reversed_at:
            console.print(
                f"  [red]Reversed: {d.reversal_reason or 'No reason given'}[/red]"
            )


async def handle_cleanup(
    older_than_days: int = 7,
    dry_run: bool = False,
    verbose: bool = False,
) -> CleanupResult:
    """Clean up old ephemeral epistemic databases.

    Removes ask_* databases and their _raw directories that are older than
    the specified number of days. Named databases (without ask_ prefix) are
    never deleted.

    Args:
        older_than_days: Delete databases older than this many days (default: 7)
        dry_run: Show what would be deleted without actually deleting
        verbose: Show detailed progress

    Returns:
        CleanupResult with cleanup statistics
    """
    from datetime import datetime, timedelta

    databases_dir = get_databases_dir()
    if not databases_dir.exists():
        console.print("[yellow]No databases directory found[/yellow]")
        return CleanupResult(success=True, deleted=0, freed_bytes=0, dry_run=dry_run)

    cutoff_time = datetime.now() - timedelta(days=older_than_days)
    deleted_count = 0
    freed_bytes = 0
    candidates = []

    # Find all ask_* databases
    for item in databases_dir.iterdir():
        if item.name.startswith("ask_"):
            # Check modification time
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            if mtime < cutoff_time:
                # Calculate size
                if item.is_file():
                    size = item.stat().st_size
                else:
                    size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                candidates.append((item, mtime, size))

    if not candidates:
        console.print(
            f"[green]No ephemeral databases older than {older_than_days} days found[/green]"
        )
        return CleanupResult(success=True, deleted=0, freed_bytes=0, dry_run=dry_run)

    # Group by database name (db + raw dir)
    db_groups = {}
    for item, mtime, size in candidates:
        # Extract base name (remove .db or _raw suffix)
        base_name = item.name.replace(".db", "").replace("_raw", "")
        if base_name not in db_groups:
            db_groups[base_name] = {"items": [], "total_size": 0, "mtime": mtime}
        db_groups[base_name]["items"].append(item)
        db_groups[base_name]["total_size"] += size

    console.print("\n[bold]Epistemic Database Cleanup[/bold]")
    console.print(
        f"Found {len(db_groups)} ephemeral projects older than {older_than_days} days\n"
    )

    if dry_run:
        console.print("[yellow]DRY RUN - no files will be deleted[/yellow]\n")

    for base_name, info in sorted(db_groups.items()):
        size_mb = info["total_size"] / (1024 * 1024)
        age_days = (datetime.now() - info["mtime"]).days

        if verbose or dry_run:
            console.print(f"  {base_name}: {size_mb:.2f} MB, {age_days} days old")

        if not dry_run:
            # Use delete_database which handles both .db and _raw
            try:
                delete_database(base_name)
                deleted_count += 1
                freed_bytes += info["total_size"]
            except Exception as e:
                console.print(f"  [red]Failed to delete {base_name}: {e}[/red]")

    if dry_run:
        total_mb = sum(info["total_size"] for info in db_groups.values()) / (
            1024 * 1024
        )
        console.print(
            f"\n[yellow]Would delete {len(db_groups)} projects, freeing {total_mb:.2f} MB[/yellow]"
        )
    else:
        freed_mb = freed_bytes / (1024 * 1024)
        console.print(
            f"\n[green]Deleted {deleted_count} projects, freed {freed_mb:.2f} MB[/green]"
        )

    # Also clean up orphan _raw directories (where .db was already deleted)
    orphan_count = 0
    orphan_bytes = 0
    for item in databases_dir.iterdir():
        if (
            item.name.startswith("ask_")
            and item.name.endswith("_raw")
            and item.is_dir()
        ):
            base_name = item.name.replace("_raw", "")
            db_file = databases_dir / f"{base_name}.db"
            if not db_file.exists():
                # Orphan _raw directory - no matching .db
                if not dry_run:
                    import shutil

                    size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    shutil.rmtree(item)
                    orphan_count += 1
                    orphan_bytes += size
                    if verbose:
                        console.print(f"  Cleaned orphan: {item.name}")

    if orphan_count > 0:
        orphan_mb = orphan_bytes / (1024 * 1024)
        console.print(
            f"[green]Cleaned {orphan_count} orphan directories, freed {orphan_mb:.2f} MB[/green]"
        )

    return CleanupResult(
        success=True,
        deleted=deleted_count + orphan_count,
        freed_bytes=freed_bytes + orphan_bytes,
        dry_run=dry_run,
    )


async def handle_report(
    name: str,
    output_path: str,
    objective_id: Optional[str] = None,
    *,
    model: str,
    verbose: bool = False,
) -> ReportResult:
    """Generate HTML report from existing epistemic database.

    This command generates a standalone HTML report from an existing epistemic
    database without re-running the investigation. Useful for generating reports
    from completed research projects.

    Args:
        name: Database name (epistemic project name)
        output_path: Path to save the HTML file (e.g., report.html)
        objective_id: Specific objective to report on (uses first if not specified)
        model: Model name for report metadata
        verbose: Print detailed progress

    Returns:
        ReportResult with generation status and statistics
    """
    from pathlib import Path
    from .report_generator import ReportGenerator

    # Validate database exists
    db_path = get_db_path(name)
    if not db_path.exists():
        console.print(f"[red]Database not found: {name}[/red]")
        console.print(f"[dim]Expected at: {db_path}[/dim]")
        return ReportResult(success=False, error=f"Database not found: {name}")

    # Initialize store
    store = DocumentStore.for_database(name)
    await store.initialize()

    # Get objective ID if not specified
    if not objective_id:
        objective_id = await _get_objective_id_from_db(store)
        if not objective_id:
            return ReportResult(success=False, error="No objectives found in database")

    if verbose:
        console.print(f"[cyan]Generating HTML report from: {name}[/cyan]")
        console.print(f"[dim]Objective: {objective_id}[/dim]")

    # Generate report
    output_file = Path(output_path)
    generator = ReportGenerator(store, name)

    # Extract data to get counts
    report_data = await generator.extract_report_data(
        objective_id=objective_id,
        model_name=model,
    )

    if not report_data:
        console.print("[red]No data found for report generation[/red]")
        return ReportResult(success=False, error="No data found for report generation")

    # Save HTML
    if await generator.save_html(
        output_path=output_file,
        objective_id=objective_id,
        model_name=model,
    ):
        console.print(
            f"\n[green]✓[/green] HTML report saved to: [bold]{output_file}[/bold]"
        )
        console.print(
            f"[dim]Claims: {len(report_data.claims)}, Evidence: {len(report_data.evidence)}, Uncertainties: {len(report_data.uncertainties)}[/dim]"
        )

        return ReportResult(
            success=True,
            database_name=name,
            output_path=str(output_file),
            claims_count=len(report_data.claims),
            evidence_count=len(report_data.evidence),
            uncertainties_count=len(report_data.uncertainties),
        )
    else:
        console.print("[red]Failed to generate HTML report[/red]")
        return ReportResult(success=False, error="Failed to save HTML report")
