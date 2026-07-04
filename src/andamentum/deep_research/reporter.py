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

    def query_accepted(self, *, slot: int, query: str, reason: str) -> None: ...

    def query_rejected(
        self,
        *,
        slot: int,
        attempt: int,
        query: str,
        reason: str,
    ) -> None: ...

    def slot_exhausted(self, *, slot: int, new_target_count: int) -> None: ...

    def parallel_search_starting(self, *, queries: list[str]) -> None: ...

    def query_search_complete(
        self, *, query: str, n_results: int, error: str | None
    ) -> None: ...

    def fetch_starting(self, *, n_pages: int) -> None: ...

    def fetch_complete(
        self,
        *,
        url: str,
        success: bool,
        n_words: int,
        error: str | None,
    ) -> None: ...

    def summarize_starting(self, *, n_pages: int) -> None: ...

    def page_summarized(self, *, url: str, relevance: float, summary: str) -> None: ...

    def synthesis_starting(self, *, n_summaries: int, max_relevance: float) -> None: ...

    def cycle_complete(self) -> None: ...


class NoopReporter:
    """Default reporter — silently drops every event.

    Used when ``--verbose`` is off and in every test that doesn't care
    about progress UI. Fields all return None; no I/O.
    """

    def cycle_starting(self, **_kwargs) -> None:
        pass

    def slot_starting(self, **_kwargs) -> None:
        pass

    def query_generated(self, **_kwargs) -> None:
        pass

    def query_accepted(self, **_kwargs) -> None:
        pass

    def query_rejected(self, **_kwargs) -> None:
        pass

    def slot_exhausted(self, **_kwargs) -> None:
        pass

    def parallel_search_starting(self, **_kwargs) -> None:
        pass

    def query_search_complete(self, **_kwargs) -> None:
        pass

    def fetch_starting(self, **_kwargs) -> None:
        pass

    def fetch_complete(self, **_kwargs) -> None:
        pass

    def summarize_starting(self, **_kwargs) -> None:
        pass

    def page_summarized(self, **_kwargs) -> None:
        pass

    def synthesis_starting(self, **_kwargs) -> None:
        pass

    def cycle_complete(self) -> None:
        pass


# Shared default sink for worker ``reporter=`` parameters — one stateless
# instance, so workers can call reporter methods unconditionally (the
# dialect's "Protocol-typed sink defaulted to no-op").
NOOP_REPORTER: SearchReporter = NoopReporter()


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
            self.console.print(f"  [dim]gaps:[/dim] {', '.join(gaps)}")

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
            f"    [white]{marker}[/white] generated  [bold]{query}[/bold]"
        )
        if rationale:
            self.console.print(f"      [dim]{rationale}[/dim]")

    def query_accepted(self, *, slot: int, query: str, reason: str) -> None:
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
            f"    [yellow]✗ rejected[/yellow]  [dim italic]{reason}[/dim italic]"
        )

    def slot_exhausted(self, *, slot: int, new_target_count: int) -> None:
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
                f"    [red]✗[/red] {query!r}  [red]→ ERROR:[/red] [dim]{error}[/dim]"
            )
        else:
            self.console.print(
                f"    [green]✓[/green] {query!r}  "
                f"[dim]→ {n_results} {'result' if n_results == 1 else 'results'}[/dim]"
            )

    def fetch_starting(self, *, n_pages: int) -> None:
        self.console.print()
        self.console.print(
            f"  [bold cyan]▶ Fetching {n_pages} "
            f"{'page' if n_pages == 1 else 'pages'}…[/bold cyan]"
        )

    def fetch_complete(
        self,
        *,
        url: str,
        success: bool,
        n_words: int,
        error: str | None,
    ) -> None:
        short = self._short_url(url)
        if success:
            self.console.print(
                f"    [green]✓[/green] {short}  [dim]({n_words:,} words)[/dim]"
            )
        else:
            self.console.print(
                f"    [red]✗[/red] {short}  "
                f"[red dim]→ {error or 'unknown error'}[/red dim]"
            )

    def summarize_starting(self, *, n_pages: int) -> None:
        self.console.print()
        self.console.print(
            f"  [bold cyan]▶ Summarising {n_pages} "
            f"{'page' if n_pages == 1 else 'pages'}…[/bold cyan]"
        )

    def page_summarized(self, *, url: str, relevance: float, summary: str) -> None:
        short = self._short_url(url)
        # Color-code relevance: green ≥ 0.6, yellow 0.3–0.6, red < 0.3.
        if relevance >= 0.6:
            colour = "green"
            mark = "✓"
        elif relevance >= 0.3:
            colour = "yellow"
            mark = "·"
        else:
            colour = "red"
            mark = "⊘"
        self.console.print(
            f"    [{colour}]{mark}[/{colour}] {short}  "
            f"[dim]relevance {relevance:.2f}[/dim]"
        )
        # Show the first ~120 chars of the summary as a clue at low relevance.
        if relevance < 0.3 and summary:
            preview = summary.strip().replace("\n", " ")[:120]
            self.console.print(f"      [dim italic]{preview}…[/dim italic]")

    def synthesis_starting(self, *, n_summaries: int, max_relevance: float) -> None:
        self.console.print()
        if n_summaries == 0:
            self.console.print(
                "  [bold yellow]⚠ Synthesising with no page summaries — "
                "report will note 'no pages fetched'.[/bold yellow]"
            )
        elif max_relevance < 0.3:
            self.console.print(
                f"  [bold yellow]⚠ Synthesising {n_summaries} "
                f"{'summary' if n_summaries == 1 else 'summaries'} "
                f"(max relevance {max_relevance:.2f}) — limited evidence; "
                f"report will be framed as partial.[/bold yellow]"
            )
        else:
            self.console.print(
                f"  [bold cyan]▶ Synthesising {n_summaries} "
                f"{'summary' if n_summaries == 1 else 'summaries'} "
                f"(max relevance {max_relevance:.2f})…[/bold cyan]"
            )

    def cycle_complete(self) -> None:
        # Sequential output — nothing to flush; the next cycle's
        # cycle_starting call adds its own leading blank line.
        pass

    @staticmethod
    def _short_url(url: str) -> str:
        """Trim a URL to a readable in-terminal label."""
        from urllib.parse import urlparse

        try:
            p = urlparse(url)
        except Exception:
            return url
        host = p.netloc.replace("www.", "")
        path = p.path.rstrip("/")
        last = path.rsplit("/", 1)[-1] if path else ""
        if last and len(host + "/" + last) <= 60:
            return f"{host}/{last}"
        return host or url
