"""Stage 3 (build) + stage 4 (audit): agents author code, the sandbox verifies it.

The build tests use the scripted draft stub (a contract-valid body, no live model). The
audit test uses the real ``SubprocessSandbox`` — it genuinely runs the built system's
shipped tests out-of-process, proving the authored system works (no container needed).
"""

from __future__ import annotations

from pathlib import Path

from andamentum.core import AgentDefinition
from andamentum.forge import build_system, compile_spec, render, run_forge
from andamentum.forge.extract import discover_holes
from andamentum.forge.sandbox import SubprocessSandbox
from andamentum.forge.schemas import (
    BodyVerdict,
    DesignPlan,
    ForgeWhy,
    NodeDraft,
    PieceOut,
)
from andamentum.forge.spec import NodeKind
from pydantic import BaseModel

from .conftest import FakeSandbox, ScriptedSink


def _plan() -> DesignPlan:
    return DesignPlan(
        why=ForgeWhy(
            purpose="Help manage a reading list.",
            boundary_in="a request",
            boundary_out="an answer",
        ),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Parse the request.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["parsed_request"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Answer the request.",
                kind=NodeKind.HEAD,
                consumes=["parsed_request"],
                produces=["answer"],
            ),
        ],
    )


async def test_build_fills_every_hole(
    tmp_path: Path, reading_list_sink: ScriptedSink
) -> None:
    spec = compile_spec(_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name
    assert discover_holes(pkg), "expected at least one spine-body hole to fill"

    report = await build_system(spec, pkg, sink=reading_list_sink, attempt_cap=3)
    assert report.all_filled, report.unfillable
    assert not discover_holes(pkg), "every hole should be filled after build"


class _AlwaysObjectsSink(ScriptedSink):
    """The component manager always objects, but the drafted body passes every static
    gate — proves a gate-valid body is KEPT (not downgraded) with the concern recorded."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "component_manager":
            return BodyVerdict(
                implements_job=False, issue="does not use parsed_request"
            )
        return await super().run(defn, **kwargs)


class _ObjectsOnceSink(ScriptedSink):
    """The component manager objects on its FIRST verdict per build, then passes — proves
    the manager's issue feeds the in-loop repair (the node fills, no concern recorded)."""

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self._manager_calls = 0

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "component_manager":
            self._manager_calls += 1
            if self._manager_calls == 1:
                return BodyVerdict(
                    implements_job=False, issue="does not use parsed_request"
                )
            return BodyVerdict(implements_job=True, issue="")
        return await super().run(defn, **kwargs)


def _sink_kwargs() -> dict[str, object]:
    return dict(
        why=ForgeWhy(
            purpose="Help the user manage a personal reading list.",
            boundary_in="a natural-language request",
            boundary_out="a text answer",
        ),
        areas=["core"],
        jobs_by_area={"core": ["Answer the request."]},
    )


async def test_static_valid_body_with_manager_objection_is_kept(
    tmp_path: Path,
) -> None:
    spec = compile_spec(_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name

    sink = _AlwaysObjectsSink(**_sink_kwargs())  # type: ignore[arg-type]
    report = await build_system(spec, pkg, sink=sink, attempt_cap=3)

    # The body passed every static gate, so it is KEPT — never downgraded to unfillable.
    assert report.all_filled, report.unfillable
    assert not discover_holes(pkg), "the kept body should leave no hole"
    # The unresolved manager objection is recorded as an advisory concern.
    assert report.concerns, "expected a recorded BuildConcern"
    assert all(c.issue == "does not use parsed_request" for c in report.concerns)
    concern_nodes = {c.node for c in report.concerns}
    assert concern_nodes <= {f.node for f in report.filled}


async def test_manager_objection_feeds_the_repair_loop(tmp_path: Path) -> None:
    spec = compile_spec(_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name

    sink = _ObjectsOnceSink(**_sink_kwargs())  # type: ignore[arg-type]
    report = await build_system(spec, pkg, sink=sink, attempt_cap=3)

    assert report.all_filled, report.unfillable
    # The first node's manager objected once then passed → at least 2 attempts, no concern.
    assert report.concerns == []
    assert any(f.attempts >= 2 for f in report.filled)


class _InventsDepsSink(ScriptedSink):
    """Authors an otherwise contract-valid body that reaches for an UNDECLARED dependency
    (`ctx.deps.repo_url`). Proves the deps gate catches the small-model wiring bug at build
    — the exact failure that previously slipped to a runtime AttributeError."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name in ("build_draft", "build_repair"):
            from .conftest import _draft_body

            valid = _draft_body(str(kwargs.get("context", "")))
            return PieceOut(body="_ = ctx.deps.repo_url\n" + valid)
        return await super().run(defn, **kwargs)


async def test_undeclared_dep_access_is_caught_at_build(tmp_path: Path) -> None:
    spec = compile_spec(_plan())
    render(spec, tmp_path)
    pkg = tmp_path / spec.name

    sink = _InventsDepsSink(**_sink_kwargs())  # type: ignore[arg-type]
    report = await build_system(spec, pkg, sink=sink, attempt_cap=2)

    # The body invents ctx.deps.repo_url every attempt → the gate rejects it → honest
    # unfillable, NEVER a runtime AttributeError. (The hole is restored to a known state.)
    assert report.unfillable, "a body inventing an undeclared dep must fail the gate"
    assert any("repo_url" in u.last_error for u in report.unfillable), report.unfillable
    assert discover_holes(pkg), "the rejected node's hole is restored"


async def test_full_pipeline_with_fake_sandbox(
    tmp_path: Path, reading_list_sink: ScriptedSink
) -> None:
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=reading_list_sink,
        sandbox=FakeSandbox(),
    )
    assert result.stage_reached == "audit"
    assert result.build is not None and result.build.all_filled
    assert result.audit is not None
    assert (
        result.audit.requirements is not None and result.audit.requirements.meets_brief
    )


async def test_audit_runs_the_built_system_out_of_process(
    tmp_path: Path, reading_list_sink: ScriptedSink
) -> None:
    # The strong proof: author the whole system, then RUN its shipped tests in a real
    # subprocess (assembles + smoke-runs the graph end-to-end). No container, no model.
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        sink=reading_list_sink,
        sandbox=SubprocessSandbox(),
    )
    assert result.stage_reached == "audit"
    assert result.audit is not None
    failed = [f"{c.name}: {c.detail}" for c in result.audit.checks if not c.passed]
    assert result.audit.works, failed
