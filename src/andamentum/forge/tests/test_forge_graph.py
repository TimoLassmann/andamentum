"""End-to-end forge graph runs, driven by a stub agent (no live model)."""

from __future__ import annotations

from pathlib import Path

from andamentum.forge import run_forge

from .conftest import ScriptedSink


async def test_design_only(reading_list_sink: ScriptedSink) -> None:
    result = await run_forge(
        "Manage my reading list.", model="test", sink=reading_list_sink
    )
    assert result.design_only
    assert result.report is None
    spec = result.spec
    assert len(spec.nodes) == 2
    assert any(n.kind.value == "head" for n in spec.nodes)


async def test_render_stage_verifies(
    tmp_path: Path, reading_list_sink: ScriptedSink
) -> None:
    result = await run_forge(
        "Manage my reading list.",
        model="test",
        dest=tmp_path,
        stop_after="render",
        sink=reading_list_sink,
    )
    assert result.stage_reached == "render"
    assert result.report is not None
    failed = [f"{c.name}: {c.detail}" for c in result.report.checks if not c.passed]
    assert result.works, failed
    assert result.rendered_files
    # the produced package is a real, importable agentic graph
    assert any(f.endswith("graph.py") for f in result.rendered_files)


async def test_blank_brief_is_rejected_at_the_door(
    reading_list_sink: ScriptedSink,
) -> None:
    try:
        await run_forge("   ", model="test", sink=reading_list_sink)
    except ValueError as e:
        assert "blank" in str(e)
    else:  # pragma: no cover
        raise AssertionError("blank brief should raise")
