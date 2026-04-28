"""Progress reporter for the deep_research search cycle.

The CLI installs a ``RichReporter`` when ``--verbose`` is set so the user
can watch the generate→verify→search loop unfold in real time: which
queries were proposed, which were accepted, which were rejected and why,
and how many search results came back.

Library callers and tests pass ``NoopReporter()`` (the default on
``NodeDeps``) to silence the channel completely.

This is intentionally NOT Python logging — progress events have different
needs than warnings/errors:

- progress events should produce a structured, hierarchical, *visual*
  display in the terminal (cycle → slot → attempt)
- warnings/errors should appear in logs regardless of UI mode

Both coexist: nodes call ``reporter.query_accepted(...)`` for progress
events AND ``logger.warning(...)`` for actual problems.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SearchReporter(Protocol):
    """Callbacks invoked by deep_research nodes at progress checkpoints.

    Every method takes only kwargs so the interface is forward-compatible:
    new fields can be added without breaking existing implementations.
    """

    def cycle_starting(
        self,
        *,
        iteration: int,
        mode: str,
        target_count: int,
        gaps: list[str],
    ) -> None: ...

    def slot_starting(self, *, slot: int) -> None: ...

    def query_generated(
        self,
        *,
        slot: int,
        attempt: int,
        query: str,
        rationale: str,
    ) -> None: ...

    def query_accepted(
        self, *, slot: int, query: str, reason: str
    ) -> None: ...

    def query_rejected(
        self,
        *,
        slot: int,
        attempt: int,
        query: str,
        reason: str,
    ) -> None: ...

    def slot_exhausted(
        self, *, slot: int, new_target_count: int
    ) -> None: ...

    def parallel_search_starting(self, *, queries: list[str]) -> None: ...

    def query_search_complete(
        self, *, query: str, n_results: int, error: str | None
    ) -> None: ...

    def cycle_complete(self) -> None: ...


class NoopReporter:
    """Default reporter — silently drops every event.

    Used when ``--verbose`` is off and in every test that doesn't care
    about progress UI. Fields all return None; no I/O.
    """

    def cycle_starting(self, **_kwargs) -> None: pass
    def slot_starting(self, **_kwargs) -> None: pass
    def query_generated(self, **_kwargs) -> None: pass
    def query_accepted(self, **_kwargs) -> None: pass
    def query_rejected(self, **_kwargs) -> None: pass
    def slot_exhausted(self, **_kwargs) -> None: pass
    def parallel_search_starting(self, **_kwargs) -> None: pass
    def query_search_complete(self, **_kwargs) -> None: pass
    def cycle_complete(self) -> None: pass


class RichReporter:
    """Pretty-prints search-cycle progress via a ``rich.console.Console``.

    Output is sequential (no Live region) — events arrive in order and
    natural newline rendering is the cleanest visual story. Indentation
    and Rich markup carry the cycle → slot → attempt hierarchy.
    """

    def __init__(self, console) -> None:
        self.console = console

    def cycle_starting(
        self,
        *,
        iteration: int,
        mode: str,
        target_count: int,
        gaps: list[str],
    ) -> None:
        self.console.print()
        header = (
            f"[bold cyan]▶ Cycle {iteration}[/bold cyan]  "
            f"[dim](mode=[white]{mode}[/white], "
            f"target=[white]{target_count}[/white])[/dim]"
        )
        self.console.print(header)
        if gaps:
            self.console.print(
                f"  [dim]gaps:[/dim] {', '.join(gaps)}"
            )

    def slot_starting(self, *, slot: int) -> None:
        self.console.print(f"  [dim]Slot {slot}[/dim]")

    def query_generated(
        self,
        *,
        slot: int,
        attempt: int,
        query: str,
        rationale: str,
    ) -> None:
        marker = "•" if attempt == 1 else "↻"
        self.console.print(
            f"    [white]{marker}[/white] generated  "
            f"[bold]{query}[/bold]"
        )
        if rationale:
            self.console.print(f"      [dim]{rationale}[/dim]")

    def query_accepted(
        self, *, slot: int, query: str, reason: str
    ) -> None:
        self.console.print(
            f"    [green]✓ accepted[/green]  [dim italic]{reason}[/dim italic]"
        )

    def query_rejected(
        self,
        *,
        slot: int,
        attempt: int,
        query: str,
        reason: str,
    ) -> None:
        self.console.print(
            f"    [yellow]✗ rejected[/yellow]  "
            f"[dim italic]{reason}[/dim italic]"
        )

    def slot_exhausted(
        self, *, slot: int, new_target_count: int
    ) -> None:
        self.console.print(
            f"    [yellow]⚠ slot exhausted[/yellow] "
            f"[dim]→ tightening target to {new_target_count}[/dim]"
        )

    def parallel_search_starting(self, *, queries: list[str]) -> None:
        self.console.print()
        self.console.print(
            f"  [bold cyan]▶ Searching {len(queries)} "
            f"{'query' if len(queries) == 1 else 'queries'} in parallel…[/bold cyan]"
        )

    def query_search_complete(
        self, *, query: str, n_results: int, error: str | None
    ) -> None:
        if error:
            self.console.print(
                f"    [red]✗[/red] {query!r}  "
                f"[red]→ ERROR:[/red] [dim]{error}[/dim]"
            )
        else:
            self.console.print(
                f"    [green]✓[/green] {query!r}  "
                f"[dim]→ {n_results} {'result' if n_results == 1 else 'results'}[/dim]"
            )

    def cycle_complete(self) -> None:
        # Sequential output — nothing to flush; the next cycle's
        # cycle_starting call adds its own leading blank line.
        pass
