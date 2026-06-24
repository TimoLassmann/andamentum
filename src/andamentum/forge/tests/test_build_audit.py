"""Stage 3 (build) + stage 4 (audit): agents author code, the sandbox verifies it.

The build tests use the scripted draft stub (a contract-valid body, no live model). The
audit test uses the real ``SubprocessSandbox`` — it genuinely runs the built system's
shipped tests out-of-process, proving the authored system works (no container needed).
"""

from __future__ import annotations

from pathlib import Path

from andamentum.forge import build_system, compile_spec, render, run_forge
from andamentum.forge.extract import discover_holes
from andamentum.forge.sandbox import SubprocessSandbox
from andamentum.forge.schemas import DesignPlan, ForgeWhy, NodeDraft
from andamentum.forge.spec import NodeKind

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
