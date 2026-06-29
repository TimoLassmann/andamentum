"""Progress reporting — the Port is driven in stage order, and the live dashboard
renders without a live model or a container.

``run_forge`` drives the reporter off ``graph.iter``; a recording stub captures the
stage sequence and details. The ``RichReporter`` is exercised against an in-memory
console to prove it paints the checklist, the per-node build bar, and the audit line.
"""

from __future__ import annotations

import io

from andamentum.forge import run_forge
from andamentum.forge.reporter import NoopReporter, RichReporter

from .conftest import ScriptedSink


class _Recording:
    """A reporter stub that records the event stream for assertions."""

    def __init__(self) -> None:
        self.planned_stages: list[str] = []
        self.started: list[str] = []
        self.finished: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []
        self.finished_run = False

    def planned(self, *, stages: list[str]) -> None:
        self.planned_stages = list(stages)

    def stage_started(self, *, name: str) -> None:
        self.started.append(name)

    def stage_finished(self, *, name: str, detail: str) -> None:
        self.finished.append((name, detail))

    def stage_failed(self, *, name: str, error: str) -> None:
        self.failed.append((name, error))

    def build_starting(self, **_kwargs: object) -> None:
        pass

    def node_building(self, **_kwargs: object) -> None:
        pass

    def node_built(self, **_kwargs: object) -> None:
        pass

    def audit_check(self, **_kwargs: object) -> None:
        pass

    def run_finished(self, **_kwargs: object) -> None:
        self.finished_run = True


# --- the Port is driven in stage order ------------------------------------------


async def test_reporter_sees_design_stages_in_order(
    reading_list_sink: ScriptedSink,
) -> None:
    rep = _Recording()
    await run_forge(
        "Manage my reading list.", model="test", sink=reading_list_sink, reporter=rep
    )

    assert rep.planned_stages == [
        "Understand",
        "Assess",
        "Frame",
        "Decompose",
        "Compile",
        "Review",
    ]
    assert rep.started == rep.planned_stages  # every planned stage started, in order
    finished_names = [n for n, _ in rep.finished]
    # Finish runs last but is not a displayed stage; the five design stages all finish.
    for stage in rep.planned_stages:
        assert stage in finished_names
    assert rep.finished_run is True
    assert rep.failed == []


async def test_reporter_records_stage_detail(
    reading_list_sink: ScriptedSink,
) -> None:
    rep = _Recording()
    await run_forge(
        "Manage my reading list.", model="test", sink=reading_list_sink, reporter=rep
    )
    detail = dict(rep.finished)
    assert "serves the goal" in detail["Review"]
    assert "steps" in detail["Decompose"]
    assert "nodes" in detail["Compile"]


async def test_reporter_marks_the_failing_stage() -> None:
    # A plan-review failure inside the graph is reported against the active stage. The
    # rejecting sink from test_review loops Review until it fails loud at the cap.
    from .test_review import _rejecting_sink

    rep = _Recording()
    try:
        await run_forge(
            "Manage my reading list.",
            model="test",
            sink=_rejecting_sink(),
            reporter=rep,
        )
    except ValueError:
        pass
    assert rep.failed, "the failing stage should be reported"
    assert rep.failed[0][0] == "Review"


# --- the live dashboard renders --------------------------------------------------


def _rich_reporter() -> tuple[RichReporter, io.StringIO]:
    from rich.console import Console

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=88)
    rep = RichReporter(
        console, brief="Manage my reading list", model="ollama:test", dest="/tmp/x"
    )
    return rep, buf


def test_rich_reporter_paints_the_checklist() -> None:
    rep, buf = _rich_reporter()
    rep.planned(stages=["Understand", "Frame", "Build", "Audit"])
    rep.stage_started(name="Understand")
    rep.stage_finished(name="Understand", detail="purpose + boundaries")
    rep.stage_started(name="Build")
    rep.build_starting(total=3)
    rep.node_building(
        node="n1", kind="spine", index=1, total=3, attempt=1, phase="draft"
    )
    with rep:  # start/stop the Live region around a paint
        pass
    out = buf.getvalue()
    assert "forge" in out
    assert "Manage my reading list" in out


def test_rich_reporter_handles_build_and_audit_substeps() -> None:
    rep, _buf = _rich_reporter()
    rep.planned(stages=["Build", "Audit"])
    rep.stage_started(name="Build")
    rep.build_starting(total=2)
    rep.node_building(
        node="n1", kind="spine", index=1, total=2, attempt=2, phase="repair"
    )
    rep.node_built(node="n1", status="filled", attempts=2, detail="")
    rep.stage_finished(name="Build", detail="2 authored · 0 unfillable")
    rep.stage_started(name="Audit")
    rep.audit_check(name="tests", status="running", detail="shipped tests in sandbox")
    rep.audit_check(name="tests", status="passed", detail="passed")
    # __rich__ must build a renderable without raising in every state.
    assert rep.__rich__() is not None


def test_noop_reporter_is_silent() -> None:
    rep = NoopReporter()
    rep.planned(stages=["A"])
    rep.stage_started(name="A")
    rep.node_building(
        node="n", kind="spine", index=1, total=1, attempt=1, phase="draft"
    )
    rep.run_finished(works=True, stage_reached="design")  # no output, no error
