"""CLI Handlers for Epistemic System.

Async handlers for CLI commands. Designed to return structured data
that can be easily translated to web app responses.

Architecture: Layer 4 (Application)
"""

import logging
from collections import defaultdict
from typing import Dict, Any, Optional, Literal

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.markup import escape as rich_escape

# Old orchestrator removed - now using operations-based system
from .trace_renderers import render_timeline, render_flow, render_claims, render_all
from .result_models import (
    AskResult,
)
from .trace import ReasoningTrace
from pathlib import Path

from andamentum.document_store import Store

logger = logging.getLogger(__name__)

# Type for trace display mode
TraceMode = Literal["timeline", "flow", "claims", "debate", "all", "none"]

console = Console()


async def _print_operation_profile(store: "Store") -> None:
    """Print a timing profile of operations from execution_step documents.

    Reads the execution_step metadata already stored by the operations runner
    and aggregates by operation type. No new recording needed.
    """
    try:
        steps = await store.list(where={"epistemic_type": "execution_step"}, limit=500)
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
    *,
    model: str,
    embedding_model: str,
    database: Optional[str | Path] = None,
    report: Optional[str | Path] = None,
    verbose: bool = False,
    trace: TraceMode = "timeline",
    force_quick: bool = False,
    mode: Literal["verify", "research"] = "research",
    provider: str = "all",
    searxng_url: Optional[str] = None,
    browser: bool = False,
) -> AskResult:
    """Run ``ask`` / ``verify`` and print the progress and the findings.

    The command-line face of :func:`andamentum.epistemic.ask` and
    :func:`andamentum.epistemic.verify`: the same run, with the progress lines,
    the operation profile and the results shown on the console. The result is
    the same :class:`AskResult`.

    Args:
        question: Research question, or (``mode="verify"``) the claim text
        model: LLM model to use (e.g., "anthropic:claude-haiku-4-5")
        embedding_model: Ollama embedding model
        database: Path of the database file to keep; without it the run uses a
            temporary folder that is removed afterwards
        report: Path to save the HTML report
        verbose: Print detailed progress
        trace: Trace visualization mode (timeline, flow, claims, all, none)
        force_quick: Skip preplanning (clarification + conceptual analysis)
        mode: ``"research"`` attempts decomposition; ``"verify"`` treats
            ``question`` as a single claim text and skips decomposition.

    Raises:
        Whatever the run raises; the caller turns it into an error line.
    """
    import logging
    from typing import Any as TypingAny

    from .ask_verify import _run

    # Suppress noisy logs during spinner (restore after)
    root_logger = logging.getLogger()
    original_level = root_logger.level

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
            subtitle=f"[dim]{database if database else 'temporary run'}[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if force_quick:
        console.print("[dim]Skipping preplanning (starting at PLAN_TASK)[/dim]")

    console.print()
    console.print("[bold blue]Research Pipeline[/bold blue]")

    seen: dict[str, Any] = {}

    async def after_run(run_result: Any, store: Store) -> None:
        seen["run"] = run_result
        console.print()
        errored = len(run_result.quarantined)
        gate_rejected = run_result.failed - errored
        if errored > 0:
            console.print(
                f"[red]Completed {run_result.successful} operations; "
                f"{errored} errored, {gate_rejected} gate-blocked[/red]"
            )
        elif gate_rejected > 0:
            console.print(
                f"[yellow]Completed {run_result.successful} operations; "
                f"{gate_rejected} gate-blocked[/yellow]"
            )
        else:
            console.print(
                f"[green]Completed {run_result.successful} operations[/green]"
            )
        if run_result.retrieval_failed:
            console.print(
                "[red]⚠ Retrieval failed: evidence extraction returned empty "
                "content 3+ times consecutively. Posterior is uninformative.[/red]"
            )
        # Show operation timing profile
        await _print_operation_profile(store)

    try:
        # Suppress INFO logs - keep WARNING level so important issues still surface
        root_logger.setLevel(logging.WARNING)
        result = await _run(
            question,
            mode=mode,
            model=model,
            embedding_model=embedding_model,
            searxng_url=searxng_url,
            browser=browser,
            database=database,
            report=report,
            provider=provider,
            skip_preplanning=force_quick,
            verbose=verbose,
            progress_callback=progress_callback,
            after_run=after_run,
        )
    finally:
        # Restore logging level before displaying results
        root_logger.setLevel(original_level)

    run_result = seen["run"]
    synthesis = result.stats.synthesis if result.stats else None
    stats: Dict[str, Any] = {
        "workitems_executed_this_run": run_result.iterations,
        "total_workitems": run_result.iterations,
        "operations_successful": run_result.successful,
        "operations_errored": len(run_result.quarantined),
        "operations_gate_blocked": run_result.failed - len(run_result.quarantined),
        "synthesis": synthesis,
    }
    _print_ask_results(
        question,
        result.claims,
        result.evidence,
        result.uncertainties,
        stats,
        verbose,
        trace,
        None,
        verdict=result.verdict,
    )
    if report:
        console.print(f"\n[green]✓[/green] HTML report saved to: [bold]{report}[/bold]")
    if database:
        console.print(f"[dim]Run saved: {database}[/dim]")
    return result


def _print_ask_results(
    question: str,
    claims: list,
    evidence: list,
    uncertainties: list,
    stats: Dict[str, Any],
    verbose: bool = False,
    trace_mode: TraceMode = "timeline",
    reasoning_trace: Optional[ReasoningTrace] = None,
    *,
    verdict: Optional[Any] = None,
) -> None:
    """Print results from an ask query in a nice format.

    Display priority:
    1. ANSWER - The synthesized findings (always shown prominently)
    2. The run's verdict (label + posterior + provenance)
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

    # === PRIMARY OUTPUT: THE ARTEFACT ===
    if synthesis:
        artefact_content = synthesis.get("summary", "")

        if artefact_content:
            console.print(
                Panel(
                    Markdown(artefact_content),
                    title="[bold green]Research Report[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel(
                    "[dim]No findings synthesized[/dim]",
                    title="[bold green]Research Report[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

    # === SCORE: the run's verdict, rendered from the one object ===
    if verdict:
        from .entities.verdict import render_verdict_line
        from .report_data import judgment_reliability_line

        console.print(f"[bold]Verdict:[/bold] {render_verdict_line(verdict)}")
        po = verdict.posterior_report
        if po is not None:
            console.print(
                f"  [dim]({po.supporting_count} supporting, "
                f"{po.contradicting_count} contradicting)[/dim]"
            )
        # How reliable were the judgments underneath that verdict. Same
        # sentence the HTML report prints in its "How confident are we?"
        # row — one measurement, one wording. Speaks only when a count
        # actually raised something, so a run with nothing to report
        # prints exactly what it printed before this line existed.
        reliability = judgment_reliability_line(verdict)
        if reliability:
            console.print(f"  [dim]{rich_escape(reliability)}[/dim]")

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
        ops_errored = stats.get("operations_errored", 0)
        ops_gate_blocked = stats.get("operations_gate_blocked", 0)
        workitems = stats.get("workitems_executed_this_run", 0)

        stats_line = f"[dim]Stats: {workitems} work items, {ops_ok} ops succeeded"
        if ops_errored > 0:
            stats_line += f", [red]{ops_errored} errored[/red]"
        if ops_gate_blocked > 0:
            stats_line += f", [yellow]{ops_gate_blocked} gate-blocked[/yellow]"
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
