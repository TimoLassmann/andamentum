"""Trace Renderers - Visualization of epistemic reasoning traces.

This module provides Rich-based renderers for epistemic reasoning traces:
1. Timeline - Chronological view of operations (DEFAULT)
2. Flow - Pipeline DAG visualization
3. Claims - Per-claim evidence/uncertainty lineage

Architecture: Layer 4 (Application) - Uses Rich for console rendering
"""

from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.markup import escape as rich_escape

from .trace import ReasoningTrace, TraceStep, ClaimLineage


# Color scheme for operations (consistent with EpistemicConsole)
OPERATION_COLORS = {
    "PLAN": "magenta",
    "COLLECT": "green",
    "EXTRACT": "green",
    "PROPOSE": "yellow",
    "SCRUTINISE": "red",
    "PROMOTE": "cyan",
    "FREEZE": "blue",
    "COMPILE": "white",
}


def render_timeline(trace: ReasoningTrace, console: Console) -> None:
    """Render compact timeline to console.

    Shows chronological operations with timestamps, inputs, and outputs.
    This is the DEFAULT trace view - compact but informative.

    Args:
        trace: ReasoningTrace with steps to render
        console: Rich Console instance
    """
    if not trace.steps:
        console.print("[dim]No trace steps recorded[/dim]")
        return

    # Header
    console.print()
    console.print(
        Panel(
            "[bold]REASONING TIMELINE[/bold]",
            border_style="blue",
            padding=(0, 2),
        )
    )

    # Group consecutive operations of same type
    grouped_steps = _group_consecutive_steps(trace.steps)

    for group in grouped_steps:
        _render_timeline_group(group, console)

    # Footer with summary
    console.print()
    success_count = sum(1 for s in trace.steps if s.success)
    fail_count = len(trace.steps) - success_count

    summary = f"[dim]Total: {len(trace.steps)} operations"
    if fail_count > 0:
        summary += f" ({success_count} [green]succeeded[/green], {fail_count} [red]failed[/red])"
    summary += f" | {trace.evidence_count} evidence | {trace.claim_count} claims | {trace.uncertainty_count} uncertainties[/dim]"
    console.print(summary)
    console.print()


def _group_consecutive_steps(steps: List[TraceStep]) -> List[List[TraceStep]]:
    """Group consecutive steps of the same operation type."""
    if not steps:
        return []

    groups: List[List[TraceStep]] = []
    current_group: List[TraceStep] = [steps[0]]

    for step in steps[1:]:
        if step.operation_display == current_group[0].operation_display:
            current_group.append(step)
        else:
            groups.append(current_group)
            current_group = [step]

    groups.append(current_group)
    return groups


def _render_timeline_group(steps: List[TraceStep], console: Console) -> None:
    """Render a group of timeline steps."""
    if not steps:
        return

    first_step = steps[0]
    op_name = first_step.operation_display
    color = OPERATION_COLORS.get(op_name, "white")
    timestamp = first_step.timestamp.strftime("%H:%M:%S")

    # Header line with timestamp and operation
    count_suffix = f" ({len(steps)})" if len(steps) > 1 else ""
    status_icon = (
        "[green]✓[/green]" if all(s.success for s in steps) else "[red]✗[/red]"
    )

    header = f"[dim]{timestamp}[/dim] │ [{color}]{op_name}{count_suffix}[/{color}] {status_icon}"
    console.print(header)

    # Details for each step
    for i, step in enumerate(steps):
        prefix = "├──" if i < len(steps) - 1 else "└──"

        # Description
        console.print(f"         │ [dim]{prefix}[/dim] {step.description}")

        # Outputs (show all - NO TRUNCATION)
        for output in step.outputs:
            console.print(f"         │     [dim]→ {output}[/dim]")

        # Error if failed
        if not step.success and step.error:
            console.print(
                f"         │     [red]Error: {rich_escape(str(step.error))}[/red]"
            )

    console.print("         │")


def render_flow(trace: ReasoningTrace, console: Console) -> None:
    """Render pipeline flow diagram to console.

    Shows the DAG structure of operations with fan-out/fan-in.

    Args:
        trace: ReasoningTrace with steps to render
        console: Rich Console instance
    """
    if not trace.steps:
        console.print("[dim]No trace steps recorded[/dim]")
        return

    # Header
    console.print()
    console.print(
        Panel(
            "[bold]REASONING FLOW[/bold]",
            border_style="blue",
            padding=(0, 2),
        )
    )

    # Group by operation type (preserving order)
    grouped = trace.get_grouped_steps()

    # Define the canonical pipeline order
    pipeline_order = [
        "PLAN",
        "COLLECT",
        "EXTRACT",
        "PROPOSE",
        "SCRUTINISE",
        "PROMOTE",
        "FREEZE",
        "COMPILE",
    ]

    prev_count = 1
    for op_name in pipeline_order:
        if op_name not in grouped:
            continue

        steps = grouped[op_name]
        count = len(steps)
        color = OPERATION_COLORS.get(op_name, "white")

        # Connection from previous phase
        if prev_count == 1 and count > 1:
            # Fan-out
            console.print("                              │")
            branches = "─" * 10
            console.print(f"              ┌{branches}┬{branches}┐")
            console.print("              │" + " " * 10 + "│" + " " * 10 + "│")
        elif prev_count > 1 and count == 1:
            # Fan-in
            branches = "─" * 10
            console.print("              │" + " " * 10 + "│" + " " * 10 + "│")
            console.print(f"              └{branches}┴{branches}┘")
            console.print("                              │")
        else:
            # Simple connection
            console.print("                              │")

        # Render phase box(es)
        if count == 1:
            step = steps[0]
            _render_flow_box(
                op_name, step.description, step.success, color, console, single=True
            )
        else:
            # Multiple parallel operations
            _render_flow_boxes_parallel(op_name, steps, color, console)

        prev_count = count

    console.print("                              │")
    console.print("                              ▼")
    console.print("                        [bold green]COMPLETE[/bold green]")
    console.print()


def _render_flow_box(
    op_name: str,
    description: str,
    success: bool,
    color: str,
    console: Console,
    single: bool = True,
) -> None:
    """Render a single flow box."""
    status = "[green]✓[/green]" if success else "[red]✗[/red]"
    width = 30

    if single:
        # Centered single box
        padding = " " * 14
        console.print(f"{padding}┌{'─' * width}┐")
        name_line = f" [{color}]{op_name}[/{color}] {status}"
        console.print(
            f"{padding}│{name_line:^{width + 20}}│"
        )  # Extra space for color codes

        # Truncate description if needed
        desc_display = (
            description[: width - 4] if len(description) > width - 4 else description
        )
        console.print(f"{padding}│  {desc_display:<{width - 4}}  │")
        console.print(f"{padding}└{'─' * width}┘")
    else:
        # Part of parallel boxes (rendered differently)
        console.print(f"┌{'─' * 20}┐")
        console.print(f"│ [{color}]{op_name}[/{color}] {status}│")
        console.print(f"└{'─' * 20}┘")


def _render_flow_boxes_parallel(
    op_name: str, steps: List[TraceStep], color: str, console: Console
) -> None:
    """Render multiple parallel flow boxes side by side (simplified)."""
    # For simplicity, render as a list rather than true parallel boxes
    console.print("              ▼" + " " * 10 + "▼" + " " * 10 + "▼")

    for i, step in enumerate(steps):
        status = "[green]✓[/green]" if step.success else "[red]✗[/red]"
        prefix = f"[{color}]{op_name}[/{color}]"

        # Extract key info from description
        short_desc = (
            step.description[:40] if len(step.description) > 40 else step.description
        )

        console.print(f"              {prefix} #{i + 1}: {short_desc} {status}")

        # Show outputs (NO TRUNCATION)
        for output in step.outputs:
            console.print(f"                    [dim]→ {output}[/dim]")


def render_claims(trace: ReasoningTrace, console: Console) -> None:
    """Render claim lineage cards to console.

    Shows per-claim evidence/uncertainty links and promotion history.

    Args:
        trace: ReasoningTrace with claim lineages to render
        console: Rich Console instance
    """
    if not trace.claim_lineages:
        console.print("[dim]No claims to display[/dim]")
        return

    # Header
    console.print()
    console.print(
        Panel(
            "[bold]CLAIM LINEAGE[/bold]",
            border_style="blue",
            padding=(0, 2),
        )
    )

    for lineage in trace.claim_lineages:
        _render_claim_card(lineage, console)

    console.print()


def _render_claim_card(lineage: ClaimLineage, console: Console) -> None:
    """Render a single claim lineage card."""
    # Stage color
    stage_colors = {
        "hypothesis": "yellow",
        "supported": "blue",
        "provisional": "cyan",
        "robust": "green",
        "actionable": "bold green",
    }
    stage_color = stage_colors.get(lineage.stage.lower(), "white")

    # Header
    console.print()
    header_table = Table(
        show_header=False, box=box.DOUBLE_EDGE, border_style="blue", width=80
    )
    header_table.add_column("Content", width=78)

    # Claim statement
    header_table.add_row(f"[bold]CLAIM:[/bold] {lineage.statement}")
    header_table.add_row(
        f"[bold]Stage:[/bold] [{stage_color}]{lineage.stage.upper()}[/{stage_color}]"
    )
    if lineage.scope:
        header_table.add_row(f"[bold]Scope:[/bold] [dim]{lineage.scope}[/dim]")

    console.print(header_table)

    # Two-column layout: Evidence | Uncertainties
    content_table = Table(
        show_header=True, box=box.SIMPLE, border_style="dim", width=80
    )
    content_table.add_column("Supporting Evidence", width=38, style="green")
    content_table.add_column("Uncertainties", width=38, style="yellow")

    # Prepare evidence rows
    evidence_lines = []
    for ev in lineage.supporting_evidence:
        ev_type = ev.get("source_type", "unknown")
        ev_ref = ev.get("source_ref", "")
        verified = ev.get("verified", False)
        verify_tag = "" if verified else " [dim yellow][unverified][/dim yellow]"
        # Show full reference - NO TRUNCATION
        evidence_lines.append(f"[{ev_type}] {ev_ref}{verify_tag}")
        preview = ev.get("extracted_content_preview", "")
        if preview:
            evidence_lines.append(f'[dim]"{preview}"[/dim]')
        evidence_lines.append("")  # Spacing

    # Prepare uncertainty rows
    uncertainty_lines = []
    for unc in lineage.uncertainties:
        unc_type = unc.get("uncertainty_type", "unknown")
        desc = unc.get("description", "")
        resolved = "[dim](resolved)[/dim]" if unc.get("is_resolved") else ""
        uncertainty_lines.append(f"[{unc_type}] {desc} {resolved}")
        uncertainty_lines.append("")  # Spacing

    # Pad to equal length
    max_lines = max(len(evidence_lines), len(uncertainty_lines))
    while len(evidence_lines) < max_lines:
        evidence_lines.append("")
    while len(uncertainty_lines) < max_lines:
        uncertainty_lines.append("")

    # Add rows
    for ev_line, unc_line in zip(evidence_lines, uncertainty_lines):
        content_table.add_row(ev_line, unc_line)

    if evidence_lines or uncertainty_lines:
        console.print(content_table)

    # Promotion path
    if lineage.promotion_path:
        console.print()
        console.print("[bold]Promotion Path:[/bold]")
        for promo in lineage.promotion_path:
            from_stage = promo.get("from_stage", "?")
            to_stage = promo.get("to_stage", "?")
            justification = promo.get("justification", "")
            console.print(f"  {from_stage.upper()} → {to_stage.upper()}")
            if justification:
                console.print(f"  [dim]{justification}[/dim]")
    else:
        # No promotions yet - show what's needed
        console.print()
        console.print(f"[dim]Awaiting promotion from {lineage.stage.upper()}[/dim]")

    # Assumptions
    if lineage.assumptions:
        console.print()
        console.print("[bold]Assumptions:[/bold]")
        for assumption in lineage.assumptions:
            console.print(f"  [dim]• {assumption}[/dim]")


def render_all(trace: ReasoningTrace, console: Console) -> None:
    """Render all trace visualizations.

    Args:
        trace: ReasoningTrace to render
        console: Rich Console instance
    """
    render_timeline(trace, console)
    render_flow(trace, console)
    render_claims(trace, console)
