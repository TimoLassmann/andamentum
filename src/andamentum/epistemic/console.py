"""Rich Console Output for Epistemic System.

Provides detailed, colorful console output for debugging and monitoring
the epistemic orchestrator's execution.

Architecture: Layer 4 (Application)
"""

import textwrap
from typing import Dict, Any, Optional, List
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.markup import escape as rich_escape

from .primitives import WorkItemType


class EpistemicConsole:
    """Rich console output for epistemic system visibility.

    Provides formatted output showing:
    - Current execution stage
    - Agent inputs and outputs
    - Validation results
    - System state (claims, evidence, uncertainties)

    Transaction Box Model:
    Each WorkItem is displayed as a transaction with READ/TRANSFORM/WRITE sections.
    """

    # Color scheme for operation types
    OP_COLORS = {
        WorkItemType.PLAN_TASK: "magenta",
        WorkItemType.COLLECT_EVIDENCE: "green",
        WorkItemType.EXTRACT_EVIDENCE: "green",
        WorkItemType.PROPOSE_CLAIMS: "yellow",
        WorkItemType.SCRUTINISE_CLAIM: "red",
        WorkItemType.PROMOTE_CLAIM: "cyan",
        WorkItemType.FREEZE_SNAPSHOT: "blue",
        WorkItemType.SYNTHESIZE_REPORT: "white",
        WorkItemType.DECIDE: "bright_cyan",
    }

    # Intent descriptions for TRANSFORM section
    OPERATION_INTENTS = {
        WorkItemType.PLAN_TASK: "Generate research tasks for the objective",
        WorkItemType.COLLECT_EVIDENCE: "Search for sources relevant to the topic",
        WorkItemType.EXTRACT_EVIDENCE: "Extract key information from source",
        WorkItemType.PROPOSE_CLAIMS: "Propose claims based on evidence",
        WorkItemType.SCRUTINISE_CLAIM: "Identify issues and uncertainties",
        WorkItemType.PROMOTE_CLAIM: "Evaluate claim for stage promotion",
        WorkItemType.FREEZE_SNAPSHOT: "Select claims for snapshot",
        WorkItemType.SYNTHESIZE_REPORT: "Generate human-facing output",
        WorkItemType.DECIDE: "Make decision based on claims",
    }

    def __init__(self, enabled: bool = True):
        """Initialize console output.

        Args:
            enabled: Whether to show output (set False for quiet mode)
        """
        self.enabled = enabled
        self.console = Console()
        self._start_time: Optional[datetime] = None

    @property
    def _box_width(self) -> int:
        """Calculate box width dynamically based on terminal width."""
        terminal_width = self.console.width or 80
        # Leave margin of 4 for box borders and safety
        return max(40, terminal_width - 4)

    def _separator(self) -> str:
        """Generate separator line based on current terminal width."""
        return "├" + "─" * self._box_width + "┤"

    def _wrap_text(self, text: str, prefix: str = "│ ", indent: str = "", suffix: str = "") -> List[str]:
        """Wrap text to fit within the transaction box, maintaining prefix on each line.

        Args:
            text: The text to wrap
            prefix: Box prefix (e.g., "│ " or "│   ")
            indent: Additional indent after prefix (e.g., "  ")
            suffix: Suffix to append (e.g., "[/dim]" for markup)

        Returns:
            List of wrapped lines with prefix and suffix applied
        """
        # Calculate available width: terminal width minus prefix, indent, suffix, and safety margin
        # Rich markup tags don't count toward visual width, but we need to account for actual chars
        terminal_width = self.console.width or 80
        prefix_len = len(prefix) + len(indent)
        # Leave margin for box edge
        available_width = max(30, terminal_width - prefix_len - 5)

        # Wrap the text
        wrapped = textwrap.wrap(text, width=available_width, break_long_words=True, break_on_hyphens=True)

        # Return lines with prefix, indent, and suffix
        if not wrapped:
            return [f"{prefix}{indent}{suffix}"]
        return [f"{prefix}{indent}{line}{suffix}" for line in wrapped]

    def _print_wrapped(self, text: str, prefix: str = "│ ", indent: str = "", suffix: str = "") -> None:
        """Print text wrapped to fit within the transaction box."""
        for line in self._wrap_text(text, prefix, indent, suffix):
            self.console.print(line)

    def start_run(self, objective_id: str, description: str) -> None:
        """Show run start banner."""
        if not self.enabled:
            return

        self._start_time = datetime.now()

        panel = Panel(
            f"[bold white]{description}[/bold white]\n\n"
            f"[dim]Objective ID: {objective_id}[/dim]",
            title="[bold cyan]🔬 Epistemic Investigation Started[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True,
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def end_run(self, stats: Dict[str, Any]) -> None:
        """Show run completion summary."""
        if not self.enabled:
            return

        elapsed = ""
        if self._start_time:
            elapsed = f" in {(datetime.now() - self._start_time).total_seconds():.1f}s"

        # Build summary table
        table = Table(title="Run Summary", box=box.SIMPLE)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("WorkItems Executed", str(stats.get("workitems_executed_this_run", 0)))
        table.add_row("Evidence Collected", str(stats.get("evidence_count", 0)))

        # Claims by stage
        claims_by_stage = stats.get("claims_by_stage", {})
        total_claims = sum(claims_by_stage.values())
        table.add_row("Total Claims", str(total_claims))

        for stage, count in claims_by_stage.items():
            if count > 0:
                table.add_row(f"  └─ {stage}", str(count))

        table.add_row("Open Questions", str(stats.get("uncertainties_unresolved", 0)))
        table.add_row("Queued WorkItems", str(stats.get("workitems_queued", 0)))
        table.add_row("Failed WorkItems", str(stats.get("workitems_failed", 0)))

        self.console.print()
        self.console.print(table)

        # Status message
        if stats.get("workitems_failed", 0) > 0:
            self.console.print(f"\n[bold red]⚠ Completed with failures{elapsed}[/bold red]")
        elif stats.get("workitems_queued", 0) > 0:
            self.console.print(f"\n[bold yellow]⏸ Paused (more work pending){elapsed}[/bold yellow]")
        else:
            self.console.print(f"\n[bold green]✓ Complete{elapsed}[/bold green]")
        self.console.print()

    def show_stage(self, stage: str, message: str) -> None:
        """Show a stage transition."""
        if not self.enabled:
            return

        self.console.print(f"\n[bold blue]▶ {stage}[/bold blue]: {message}")

    def start_workitem(self, operation: WorkItemType, description: str) -> None:
        """Start a transaction box for the workitem."""
        if not self.enabled:
            return

        color = self.OP_COLORS.get(operation, "white")
        op_name = operation.value.upper()

        # Calculate padding: box_width - "─ " - op_name - " "
        padding = max(0, self._box_width - 3 - len(op_name))

        self.console.print()
        self.console.print(f"╭─ [bold {color}]{op_name}[/bold {color}] " + "─" * padding + "╮")
        self._print_wrapped(description)

    def show_input(self, input_package: Dict[str, Any]) -> None:
        """Show the input being sent to the agent - NO TRUNCATION."""
        if not self.enabled:
            return

        # Show key inputs (not everything - but no truncation of values)
        # Keys align with orchestrator's _build_input_package output
        important_keys = [
            "topic", "query", "objective_description", "claim_statement",
            "source_ref", "evidence_summary", "source_content", "minimum_stage",
            # Index-based claim inputs (orchestrator writes these)
            "claims_for_review",    # FREEZE_SNAPSHOT
            "available_claims",     # DECIDE
            "claim_count",          # SYNTHESIZE_REPORT (now uses typed _snapshot_claims)
            "decision_context",     # DECIDE
        ]
        shown = {}
        for key in important_keys:
            if key in input_package:
                shown[key] = input_package[key]

        if shown:
            self.console.print("[dim]│ Input:[/dim]")
            for key, value in shown.items():
                # For multi-line values, indent continuation lines
                # Escape values to prevent Rich markup interpretation
                if isinstance(value, str) and "\n" in value:
                    lines = value.split("\n")
                    self.console.print(f"[dim]│   {key}:[/dim]")
                    for line in lines:
                        self._print_wrapped(rich_escape(line), prefix="[dim]│     ", suffix="[/dim]")
                else:
                    self._print_wrapped(f"{key}: {rich_escape(str(value))}", prefix="[dim]│   ", suffix="[/dim]")

    def show_read_section(self, read_data: Dict[str, Any]) -> None:
        """Show what data was READ from state - NO TRUNCATION.

        This is the READ part of the READ/TRANSFORM/WRITE transaction model.
        Shows what data was loaded from the database before the agent runs.
        """
        if not self.enabled:
            return

        self.console.print("│")
        self.console.print("│ [bold cyan]📖 READ[/bold cyan]")
        self.console.print(self._separator())

        has_content = False

        # Topic/objective context
        if topic := read_data.get("topic"):
            has_content = True
            self._print_wrapped(f"Topic: {rich_escape(str(topic))}", prefix="│   ")

        # Claims being operated on
        if claims := read_data.get("claims"):
            has_content = True
            for claim in claims:
                statement = claim.get("statement", "N/A")
                self._print_wrapped(f"Claim: {rich_escape(str(statement))}", prefix="│   ")
                stage = claim.get("stage", "N/A")
                evidence_ids = claim.get("evidence_ids", [])
                self._print_wrapped(f"Stage: {stage} | Evidence: {evidence_ids}", prefix="│     ")

        # Evidence being used
        if evidence := read_data.get("evidence"):
            has_content = True
            for ev in evidence:
                short_id = ev.get("short_id", ev.get("evidence_id", "N/A")[:8] if ev.get("evidence_id") else "N/A")
                source_ref = ev.get("source_ref", "N/A")
                self._print_wrapped(f"Evidence {short_id}: {rich_escape(str(source_ref))}", prefix="│   ")

        # Evidence summary (for operations that get summarized evidence)
        if evidence_summary := read_data.get("evidence_summary"):
            has_content = True
            self.console.print("│   Evidence Summary:")
            # Show full summary, handle multi-line, wrap each line
            for line in str(evidence_summary).split("\n"):
                self._print_wrapped(rich_escape(line), prefix="│     ")

        # Sources being processed
        if sources := read_data.get("sources"):
            has_content = True
            for src in sources:
                url_or_ref = src.get("url", src.get("source_ref", "N/A"))
                self._print_wrapped(f"Source: {rich_escape(str(url_or_ref))}", prefix="│   ")

        # If nothing specific, show generic message
        if not has_content:
            self.console.print("│   (No prior state - initial operation)")

        self.console.print("│")

    def show_transform_section(self, operation: WorkItemType, agent_name: str, output: Dict[str, Any]) -> None:
        """Show the TRANSFORM operation and its output - NO TRUNCATION.

        This is the TRANSFORM part of the READ/TRANSFORM/WRITE transaction model.
        Shows what agent processed the data and its full output.
        """
        if not self.enabled:
            return

        self.console.print("│ [bold yellow]🔄 TRANSFORM[/bold yellow]")
        self.console.print(self._separator())
        self._print_wrapped(f"Agent: {agent_name}", prefix="│   ")
        self._print_wrapped(f"Intent: {self.OPERATION_INTENTS.get(operation, 'Process data')}", prefix="│   ")
        self.console.print("│")
        self.console.print("│   [bold]Agent Output:[/bold]")

        # Render full output using the shared helper
        self._render_output_content(output, operation, prefix="│     ")

        self.console.print("│")

    def show_write_section(self, write_summary: Dict[str, Any]) -> None:
        """Show what will be WRITTEN to state - NO TRUNCATION.

        This is the WRITE part of the READ/TRANSFORM/WRITE transaction model.
        Shows what state changes were applied to the database.
        """
        if not self.enabled:
            return

        self.console.print("│ [bold green]💾 WRITE[/bold green]")
        self.console.print(self._separator())
        self.console.print("│   State Changes:")

        has_changes = False

        # Evidence created
        if evidence_count := write_summary.get("evidence_created", 0):
            has_changes = True
            self._print_wrapped(f"• Created {evidence_count} evidence record(s)", prefix="│     ")
            for ev in write_summary.get("evidence_details", []):
                short_id = ev.get("short_id", "N/A")
                source_ref = ev.get("source_ref", "N/A")
                self._print_wrapped(f"- {short_id}: {rich_escape(str(source_ref))}", prefix="│       ")

        # Claims created
        if claims_created := write_summary.get("claims_created", 0):
            has_changes = True
            self._print_wrapped(f"• Created {claims_created} claim(s)", prefix="│     ")
            for claim in write_summary.get("claims_details", []):
                stmt = claim.get("statement", "N/A")
                self._print_wrapped(f"- {rich_escape(str(stmt))}", prefix="│       ")

        # Claims promoted
        if claims_promoted := write_summary.get("claims_promoted"):
            has_changes = True
            for promo in claims_promoted:
                to_stage = promo.get("to_stage", "N/A")
                self._print_wrapped(f"• Promoted claim to {to_stage}", prefix="│     ")

        # Uncertainties created
        if uncertainties := write_summary.get("uncertainties_created", 0):
            has_changes = True
            self._print_wrapped(f"• Created {uncertainties} uncertainty record(s)", prefix="│     ")

        # WorkItems queued
        if workitems := write_summary.get("workitems_queued", 0):
            has_changes = True
            self._print_wrapped(f"• Queued {workitems} new workitem(s)", prefix="│     ")

        # Snapshot created
        if write_summary.get("snapshot_created"):
            has_changes = True
            snapshot_id = write_summary.get("snapshot_id", "N/A")
            self._print_wrapped(f"• Created snapshot: {snapshot_id}", prefix="│     ")

        # Artefact created
        if write_summary.get("artefact_created"):
            has_changes = True
            artefact_title = write_summary.get("artefact_title", "N/A")
            self._print_wrapped(f"• Created artefact: {rich_escape(str(artefact_title))}", prefix="│     ")

        # Decision made
        if write_summary.get("decision_created"):
            has_changes = True
            self._print_wrapped("• Created decision record", prefix="│     ")

        if not has_changes:
            self.console.print("│     (No state changes)")

        self.console.print("│")

    def _render_output_content(
        self, output: Dict[str, Any], operation: WorkItemType, prefix: str = "│     ", suffix: str = ""
    ) -> None:
        """Render operation-specific output content - NO TRUNCATION.

        This is a shared helper used by both show_output() and show_transform_section().
        All content is displayed in full without any truncation.
        Long lines are wrapped to fit within the terminal, maintaining the box prefix.

        Args:
            output: The agent output dictionary
            operation: The operation type for format-specific rendering
            prefix: String to prepend to each line (e.g., "│     " or "[dim]│   ")
            suffix: String to append to each line (e.g., "" or "[/dim]")
        """

        def p(text: str, indent: str = "") -> None:
            """Print a line with prefix, optional indent, and suffix - WITH WRAPPING.

            Text is automatically escaped to prevent Rich markup interpretation
            of LLM-generated content (e.g., if output contains '[red]' or '[/red]').
            """
            self._print_wrapped(rich_escape(text), prefix=prefix, indent=indent, suffix=suffix)

        if operation == WorkItemType.PLAN_TASK:
            # New format: evidence_strategy, verification_strategy, focus_areas, planning_rationale
            if "evidence_strategy" in output:
                evidence_strategy = output.get("evidence_strategy", [])
                verification_strategy = output.get("verification_strategy", [])
                focus_areas = output.get("focus_areas", [])
                planning_rationale = output.get("planning_rationale", "")

                # Show evidence providers
                if evidence_strategy:
                    providers = [e.get("provider", "unknown") for e in evidence_strategy]
                    p(f"Evidence providers: {', '.join(providers)}")

                # Show verification methods
                if verification_strategy:
                    methods = [v.get("method", "unknown") for v in verification_strategy]
                    p(f"Verification methods: {', '.join(methods)}")

                # Show focus areas if any
                if focus_areas:
                    p(f"Focus areas: {', '.join(focus_areas)}")

                # Show rationale
                if planning_rationale:
                    p(f"Rationale: {planning_rationale}")

            else:
                # Legacy format: tasks list
                tasks_list = output.get("tasks", [])
                p(f"Created {len(tasks_list)} tasks:")
                for i, task in enumerate(tasks_list):
                    task_type = task.get("task_type", "unknown")
                    desc = task.get("description", "")
                    priority = task.get("priority", 5)
                    p(f"{i+1}. ({task_type}) {desc} - priority {priority}", "  ")

        elif operation == WorkItemType.COLLECT_EVIDENCE:
            sources = output.get("sources", [])
            p(f"Found {len(sources)} sources:")
            for i, source in enumerate(sources):
                source_type = source.get("source_type", "unknown")
                url = source.get("url", "")
                summary = source.get("summary", "")
                p(f"{i+1}. [{source_type}] {url}", "  ")
                if summary:
                    p(f"Summary: {summary}", "     ")

        elif operation == WorkItemType.EXTRACT_EVIDENCE:
            source_ref = output.get("source_ref", "")
            source_type = output.get("source_type", "")
            quotes = output.get("relevant_quotes", [])
            limitations = output.get("limitations", [])
            p(f"Source: [{source_type}] {source_ref}")
            if quotes:
                p(f"Quotes ({len(quotes)}):")
                for i, quote in enumerate(quotes):
                    p(f'{i+1}. "{quote}"', "  ")
            if limitations:
                p(f"Limitations ({len(limitations)}):")
                for lim in limitations:
                    p(f"* {lim}", "  ")

        elif operation == WorkItemType.PROPOSE_CLAIMS:
            # Handle new list[item_schema] format
            claims_list = output.get("claims", [])
            if claims_list:
                p(f"Proposed {len(claims_list)} claims:")
                for i, claim in enumerate(claims_list):
                    statement = claim.get("statement", "")
                    scope = claim.get("scope", "")
                    p(f"{i+1}. {statement}", "  ")
                    if scope:
                        p(f"Scope: {scope}", "     ")
            else:
                # Fallback for old format
                statements = output.get("statements", [])
                scopes = output.get("scopes", [])
                p(f"Proposed {len(statements)} claims:")
                for i, stmt in enumerate(statements):
                    scope = scopes[i] if i < len(scopes) else ""
                    p(f"{i+1}. {stmt}", "  ")
                    if scope:
                        p(f"Scope: {scope}", "     ")

        elif operation == WorkItemType.SCRUTINISE_CLAIM:
            issues = output.get("issues_found", [])
            issue_types = output.get("issue_types", [])
            rec = output.get("recommendation", "unknown")
            passes = output.get("passes_scrutiny", False)
            p(f"Passes scrutiny: {passes}")
            p(f"Recommendation: {rec}")
            if issues:
                p(f"Issues ({len(issues)}):")
                for i, issue in enumerate(issues):
                    itype = issue_types[i] if i < len(issue_types) else "unknown"
                    p(f"{i+1}. [{itype}] {issue}", "  ")

        elif operation == WorkItemType.PROMOTE_CLAIM:
            claim_id = output.get("claim_id", "")
            proposed = output.get("proposed_stage", "")
            ready = output.get("ready_for_promotion", False)
            justification = output.get("justification", "")
            p(f"Claim: {claim_id}")
            p(f"Proposed stage: {proposed}")
            p(f"Ready for promotion: {ready}")
            if justification:
                p(f"Justification: {justification}")

        elif operation == WorkItemType.FREEZE_SNAPSHOT:
            include_indices = output.get("include_indices", [])
            min_stage = output.get("minimum_stage", "")
            rationale = output.get("snapshot_rationale", "")
            p(f"Minimum stage: {min_stage}")
            p(f"Included claims ({len(include_indices)}):")
            for idx in include_indices:
                p(f"* index {idx}", "  ")
            if rationale:
                p(f"Rationale: {rationale}")

        elif operation == WorkItemType.SYNTHESIZE_REPORT:
            title = output.get("title", "")
            paragraphs = output.get("paragraphs", [])
            p(f"Title: {title}")
            p(f"Paragraphs ({len(paragraphs)}):")
            for i, para in enumerate(paragraphs):
                p(f"{i+1}. {para}", "  ")

        elif operation == WorkItemType.DECIDE:
            statement = output.get("statement", "")
            justification = output.get("justification", "")
            claim_ids = output.get("claim_ids", [])
            reversible = output.get("reversible", True)
            reversal_conditions = output.get("reversal_conditions", "")
            p(f"Statement: {statement}")
            p(f"Reversible: {reversible}")
            p(f"Based on claims ({len(claim_ids)}):")
            for cid in claim_ids:
                p(f"- {cid}", "  ")
            if justification:
                p(f"Justification: {justification}")
            if reversal_conditions:
                p(f"Reversal conditions: {reversal_conditions}")

        else:
            # Generic: show all keys and values - NO TRUNCATION
            p("Full output:")
            for key, value in output.items():
                if isinstance(value, list):
                    p(f"{key} ({len(value)} items):", "  ")
                    for i, item in enumerate(value):
                        p(f"{i+1}. {item}", "    ")
                else:
                    p(f"{key}: {value}", "  ")

    def show_output(self, output: Dict[str, Any], operation: WorkItemType) -> None:
        """Show the output from the agent - NO TRUNCATION.

        This is a legacy method that wraps _render_output_content with dim styling.
        Prefer show_transform_section() for the new transaction box model.
        """
        if not self.enabled:
            return

        self.console.print("[dim]│ Output:[/dim]")
        self._render_output_content(output, operation, prefix="[dim]│   ", suffix="[/dim]")

    def show_validation(self, valid: bool, errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None) -> None:
        """Show validation result."""
        if not self.enabled:
            return

        if valid:
            self.console.print("[dim]│ Validation: [green]✓ passed[/green][/dim]")
        else:
            self.console.print("[dim]│ Validation: [red]✗ failed[/red][/dim]")
            for err in (errors or []):
                self._print_wrapped(f"✗ {rich_escape(err)}", prefix="[red]│   ", suffix="[/red]")

        for warn in (warnings or []):
            self._print_wrapped(f"⚠ {rich_escape(warn)}", prefix="[yellow]│   ", suffix="[/yellow]")

    def end_workitem(self, success: bool, output_count: int, duration_s: float = 0, error: Optional[str] = None) -> None:
        """Close the transaction box."""
        if not self.enabled:
            return

        if success:
            duration_str = f" in {duration_s:.1f}s" if duration_s > 0 else ""
            self._print_wrapped(f"[green]✓ Complete ({output_count} outputs){duration_str}[/green]")
        else:
            error_msg = rich_escape(str(error)) if error else "Unknown error"
            self._print_wrapped(f"[red]✗ Failed: {error_msg}[/red]")

        self.console.print("╰" + "─" * self._box_width + "╯")

    def show_state_summary(self, stats: Dict[str, Any]) -> None:
        """Show current state summary."""
        if not self.enabled:
            return

        evidence = stats.get("evidence_count", 0)
        claims_by_stage = stats.get("claims_by_stage", {})
        total_claims = sum(claims_by_stage.values())
        uncertainties = stats.get("uncertainties_unresolved", 0)
        queued = stats.get("workitems_queued", 0)

        self.console.print(
            f"\n[dim]State: {evidence} evidence | {total_claims} claims | "
            f"{uncertainties} questions | {queued} pending tasks[/dim]"
        )

    def show_blocked(self, pending_count: int) -> None:
        """Show that execution is blocked waiting for dependencies."""
        if not self.enabled:
            return

        self.console.print(
            f"\n[yellow]⏸ Blocked: {pending_count} workitems waiting on dependencies[/yellow]"
        )

    def show_error(self, message: str) -> None:
        """Show an error message."""
        if not self.enabled:
            return

        self.console.print(f"\n[bold red]Error: {rich_escape(message)}[/bold red]")

    def show_info(self, message: str) -> None:
        """Show an info message."""
        if not self.enabled:
            return

        self.console.print(f"[dim]{message}[/dim]")
