"""Plan-manager grounding (Tier 1a) — deterministic per-area coverage, and the clean
pass-through of the Review node.

``plan_coverage`` is a pure function; the Review node is exercised end-to-end through
``run_forge`` with a scripted sink (no live model, no container).
"""

from __future__ import annotations

from andamentum.forge import run_forge
from andamentum.forge.review import plan_coverage
from andamentum.forge.schemas import (
    FindingKind,
    ForgeWhy,
    NodeDraft,
)
from andamentum.forge.spec import NodeKind

from .conftest import ScriptedSink

_WHY = ForgeWhy(
    purpose="Help the user manage a personal reading list.",
    boundary_in="a natural-language request",
    boundary_out="a text answer",
)


def _draft(node_id: str, area: str, job: str) -> NodeDraft:
    return NodeDraft(id=node_id, area=area, job=job, kind=NodeKind.SPINE)


# --- Tier 1a: deterministic coverage --------------------------------------------


def test_plan_coverage_flags_uncovered_area() -> None:
    drafts = [_draft("n1", "core", "Parse the request.")]
    findings = plan_coverage(_WHY, ["core", "persistence"], drafts)
    assert len(findings) == 1
    f = findings[0]
    assert f.kind is FindingKind.UNCOVERED_AREA
    assert "persistence" in f.detail


def test_plan_coverage_clean_when_every_area_owns_a_step() -> None:
    drafts = [
        _draft("n1", "core", "Parse the request."),
        _draft("n2", "persistence", "Save the entry."),
    ]
    assert plan_coverage(_WHY, ["core", "persistence"], drafts) == []


# --- the Review node: clean pass-through ----------------------------------------


async def test_clean_plan_passes_through(
    reading_list_sink: ScriptedSink,
) -> None:
    result = await run_forge(
        "Manage my reading list.", model="test", sink=reading_list_sink
    )
    assert result.design_only
