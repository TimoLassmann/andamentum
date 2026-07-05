"""Progress reporting for the forge pipeline — a Port, a no-op, and a live dashboard.

The CLI installs a :class:`RichReporter` when ``--verbose`` is set so the user can watch
the brief become a system in real time: the ten-stage spine as a checklist, the active
stage spinning with its elapsed time, a live bar while node bodies are authored, and the
sandbox audit ticking through its checks. Library callers and tests use the default
:class:`NoopReporter` (silent).

This follows the same Port shape as ``deep_research.reporter``: every method is
keyword-only so the interface stays forward-compatible, and it is *not* Python logging —
progress events want a structured, hierarchical, visual display, not a log line.

Leaf module (dialect Law 2): ``rich`` + stdlib only; no graph engine, no sibling
worker. It holds no domain logic — only presentation — so the stage *detail* strings are
computed by the caller (``graph.py`` reads them off the run state) and the per-node /
audit structure is passed in as data and formatted here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class ForgeReporter(Protocol):
    """Callbacks the forge pipeline invokes at progress checkpoints.

    ``graph.py`` drives the stage-level events (``planned`` / ``stage_*`` / ``run_finished``)
    off the graph iteration and the run state; the build and audit workers emit the
    sub-step events (``build_starting`` / ``node_*`` / ``audit_check``).
    """

    def planned(self, *, stages: list[str]) -> None: ...

    def stage_started(self, *, name: str) -> None: ...

    def stage_finished(self, *, name: str, detail: str) -> None: ...

    def stage_failed(self, *, name: str, error: str) -> None: ...

    def build_starting(self, *, total: int) -> None: ...

    def node_building(
        self,
        *,
        node: str,
        kind: str,
        index: int,
        total: int,
        attempt: int,
        phase: str,
    ) -> None: ...

    def node_built(
        self, *, node: str, status: str, attempts: int, detail: str
    ) -> None: ...

    def audit_check(self, *, name: str, status: str, detail: str) -> None: ...

    def run_finished(self, *, works: bool, stage_reached: str) -> None: ...


class NoopReporter:
    """The default reporter — drops every event. No I/O, no rich dependency touched."""

    def planned(self, **_kwargs: object) -> None:
        pass

    def stage_started(self, **_kwargs: object) -> None:
        pass

    def stage_finished(self, **_kwargs: object) -> None:
        pass

    def stage_failed(self, **_kwargs: object) -> None:
        pass

    def build_starting(self, **_kwargs: object) -> None:
        pass

    def node_building(self, **_kwargs: object) -> None:
        pass

    def node_built(self, **_kwargs: object) -> None:
        pass

    def audit_check(self, **_kwargs: object) -> None:
        pass

    def run_finished(self, **_kwargs: object) -> None:
        pass


# --- the live dashboard ---------------------------------------------------------

_GLYPH = {
    "pending": ("·", "dim"),
    "done": ("✓", "green"),
    "failed": ("✗", "bold red"),
}


@dataclass
class _Stage:
    """One row of the dashboard."""

    status: str = "pending"  # pending | active | done | failed
    detail: str = ""
    started_at: float = 0.0
    elapsed: float = 0.0


@dataclass
class _Build:
    """The live sub-state shown under the Build row while bodies are authored."""

    total: int = 0
    done: int = 0
    node: str = ""
    line: str = ""


@dataclass
class _Audit:
    """The live sub-state shown under the Audit row while checks run."""

    line: str = ""
    results: list[tuple[str, str]] = field(default_factory=list)  # (name, status)


def _bar(done: int, total: int, width: int = 12) -> str:
    """A compact unicode progress bar: ``▕███████▒▒▒▒▏``."""
    if total <= 0:
        return ""
    filled = round(width * done / total)
    return "▕" + "█" * filled + "▒" * (width - filled) + "▏"


class RichReporter:
    """A live ``rich`` dashboard of the forge run. Start it (``start()`` / ``with``)
    before the run and stop it after; the CLI owns that lifecycle.

    The reporter is itself the ``Live`` renderable — ``__rich__`` rebuilds the panel from
    the current state on every refresh, so the active stage's spinner and elapsed timer
    animate while a node awaits. Events only mutate state; the refresh thread paints.
    """

    def __init__(
        self, console, *, brief: str, model: str, dest: str | None = None
    ) -> None:
        from rich.spinner import Spinner

        self.console = console
        self._brief = brief
        self._model = model
        self._dest = dest
        self._order: list[str] = []
        self._stages: dict[str, _Stage] = {}
        self._build = _Build()
        self._audit = _Audit()
        self._spinner = Spinner("dots", style="cyan")  # one instance → it animates
        self._live = None

    # -- lifecycle --

    def start(self) -> None:
        from rich.live import Live

        self._live = Live(
            self, console=self.console, refresh_per_second=12, transient=False
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.refresh()  # paint the final state once more
            self._live.stop()
            self._live = None

    def __enter__(self) -> RichReporter:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- events --

    def planned(self, *, stages: list[str]) -> None:
        for name in stages:
            if name not in self._stages:
                self._order.append(name)
                self._stages[name] = _Stage()

    def stage_started(self, *, name: str) -> None:
        st = self._stages.get(name)
        if st is None:
            return
        # Each audit pass starts from a clean result set (append would stack rows).
        if name == "Audit":
            self._audit = _Audit()
        st.status = "active"
        st.started_at = time.monotonic()

    def stage_finished(self, *, name: str, detail: str) -> None:
        st = self._stages.get(name)
        if st is None:
            return
        st.status = "done"
        st.detail = detail
        if st.started_at:
            st.elapsed = time.monotonic() - st.started_at

    def stage_failed(self, *, name: str, error: str) -> None:
        st = self._stages.get(name)
        if st is None:
            return
        st.status = "failed"
        st.detail = error

    def build_starting(self, *, total: int) -> None:
        self._build = _Build(total=total)

    def node_building(
        self,
        *,
        node: str,
        kind: str,
        index: int,
        total: int,
        attempt: int,
        phase: str,
    ) -> None:
        self._build.total = total
        self._build.done = index - 1
        self._build.node = node
        suffix = f"attempt {attempt}" if attempt > 1 else phase
        self._build.line = f"{node} · {kind} · {suffix}"

    def node_built(self, *, node: str, status: str, attempts: int, detail: str) -> None:
        self._build.done += 1
        mark = {"filled": "✓", "kept": "~", "unfillable": "✗"}.get(status, "·")
        tail = f" — {detail}" if detail else ""
        self._build.line = f"{mark} {node} ({status}, {attempts} attempt{'s' if attempts != 1 else ''}){tail}"

    def audit_check(self, *, name: str, status: str, detail: str) -> None:
        if status == "running":
            self._audit.line = f"{name}: {detail}"
        else:
            self._audit.results.append((name, status))
            self._audit.line = ""

    def run_finished(self, *, works: bool, stage_reached: str) -> None:
        pass  # the dashboard already shows the final per-stage state

    # -- rendering --

    def __rich__(self):
        from rich.console import Group
        from rich.rule import Rule
        from rich.table import Table
        from rich.text import Text

        head = Text()
        head.append("forge ", style="bold cyan")
        head.append("⬢  ", style="cyan")
        head.append(self._brief, style="bold")
        sub = Text(no_wrap=True)
        sub.append(self._model, style="dim")
        if self._dest:
            sub.append("  ·  → ", style="dim")
            sub.append(self._dest, style="dim")

        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=2, justify="center")
        grid.add_column(min_width=10, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right", no_wrap=True)

        for name in self._order:
            st = self._stages[name]
            if st.status == "active":
                glyph = self._spinner
                label = Text(name, style="bold cyan")
            else:
                ch, style = _GLYPH[st.status]
                glyph = Text(ch, style=style)
                label = Text(name, style="white" if st.status == "done" else "dim")
            detail = Text(
                st.detail or ("—" if st.status == "active" else ""), style="dim"
            )
            elapsed = Text(f"{st.elapsed:.1f}s" if st.elapsed else "", style="dim")
            grid.add_row(glyph, label, detail, elapsed)
            if name == "Build" and st.status == "active":
                grid.add_row("", Text(""), self._build_row(), Text(""))
            if name == "Audit" and st.status == "active" and self._audit.line:
                grid.add_row(
                    "",
                    Text(""),
                    Text(f"   ⠿ {self._audit.line}", style="dim"),
                    Text(""),
                )

        return Group(head, sub, Rule(style="grey37"), grid, Rule(style="grey37"))

    def _build_row(self):
        from rich.text import Text

        b = self._build
        t = Text()
        t.append(f"   {_bar(b.done, b.total)} ", style="cyan")
        t.append(f"{b.done}/{b.total}  ", style="dim")
        if b.line:
            t.append(f"└ {b.line}", style="dim")
        return t
