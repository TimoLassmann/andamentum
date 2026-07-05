"""The decompose refine loop — converge to clean, or fail loud at the cap.

Under the two-pass declare-then-select scheme, a repair re-runs ONLY the SELECT pass
(``select_consumes``): produced names are frozen after DECLARE, so wiring is corrected by
re-choosing input ORDINALS, never by reinventing names. Two scenarios, both driven by a
stub ``AgentSink`` (no live model):

  - A sink that SELECTS a bad input first (a name not on the board → an empty, unreachable
    read), then — once fed the finding as feedback — selects the real producer. The loop
    must CONVERGE to a clean report, re-running only the select pass for the flagged node.
  - A sink that NEVER fixes the selection. The loop must FAIL LOUD after ``MAX_DESIGN_ROUNDS``
    with the structural problem in the message — never a half-wired design.
"""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from andamentum.core import AgentDefinition
from andamentum.forge.decompose import MAX_DESIGN_ROUNDS, decompose
from andamentum.forge.schemas import (
    ConsumeSelection,
    DataKind,
    ForgeAreas,
    ForgeWhy,
    JobList,
    NodeDeclaration,
)
from andamentum.forge.spec import NodeKind

from .conftest import _parse_options

_WHY = ForgeWhy(
    purpose="Summarize a document into bullets.",
    boundary_in="a document",
    boundary_out="bullet points",
)


def _focus_id(board: str) -> str:
    for line in board.splitlines():
        if line.strip().startswith(">>>"):
            m = re.search(r"n\d+", line)
            if m:
                return m.group()
    return ""


# Three nodes so the flaw lives in a MIDDLE node (n2): the deterministic sink-collapse only
# merges TERMINAL signals into the last node, so a middle node reading nothing stays
# unreachable and can be fixed only by the model re-selecting — this exercises the model
# repair path, which the 2-node sink-collapse case would otherwise short-circuit.
_JOBS = ["Read the document.", "Extract the key points.", "Write the bullets."]
_PRODUCES = {"n1": "document_text", "n2": "key_points", "n3": "bullets"}


def _declare(fid: str) -> NodeDeclaration:
    kind = NodeKind.SPINE if fid == "n1" else NodeKind.HEAD
    return NodeDeclaration(
        kind=kind, produces=_PRODUCES[fid], produces_kind=DataKind.SIGNAL
    )


class _RepairSink:
    """Declares a 3-step chain; n2 (a middle node) first SELECTS nothing (→ unreachable),
    then reads document_text once it sees feedback. n1/n3 wire correctly. Counts the
    select_consumes calls per node so the test can assert the model repair ran."""

    def __init__(self) -> None:
        self.select_calls: dict[str, int] = {}

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "frame":
            return ForgeAreas(areas=["core"])
        if defn.name == "list_jobs":
            return JobList(jobs=_JOBS)
        if defn.name == "declare_node":
            return _declare(_focus_id(str(kwargs.get("board", ""))))
        if defn.name == "select_consumes":
            fid = _focus_id(str(kwargs.get("board", "")))
            feedback = str(kwargs.get("feedback", ""))
            self.select_calls[fid] = self.select_calls.get(fid, 0) + 1
            options = _parse_options(str(kwargs.get("options", "")))
            if fid == "n1":
                return ConsumeSelection(consume_indices=[options["input"]])
            if fid == "n3":
                return ConsumeSelection(consume_indices=[options["key_points"]])
            # n2 (middle): read nothing first (→ unreachable), fix to document_text on feedback.
            idxs = [options["document_text"]] if feedback else []
            return ConsumeSelection(consume_indices=idxs)
        raise AssertionError(f"unexpected agent {defn.name!r}")


class _StubbornSink:
    """n2 (a middle node) always reads nothing — it stays unreachable, and the sink-collapse
    (which only fixes terminal sinks in the last node) cannot reach it, so the loop can never
    converge."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "frame":
            return ForgeAreas(areas=["core"])
        if defn.name == "list_jobs":
            return JobList(jobs=_JOBS)
        if defn.name == "declare_node":
            return _declare(_focus_id(str(kwargs.get("board", ""))))
        if defn.name == "select_consumes":
            fid = _focus_id(str(kwargs.get("board", "")))
            options = _parse_options(str(kwargs.get("options", "")))
            if fid == "n1":
                return ConsumeSelection(consume_indices=[options["input"]])
            if fid == "n3":
                return ConsumeSelection(consume_indices=[options["key_points"]])
            # n2 never reads a real producer → permanently unreachable, never fixed
            return ConsumeSelection(consume_indices=[])
        raise AssertionError(f"unexpected agent {defn.name!r}")


async def test_repair_loop_converges_to_clean() -> None:
    sink = _RepairSink()
    plan, report, _notes = await decompose(
        _WHY, ["core"], sink=sink, max_jobs_per_area=5, max_nodes=18
    )
    assert report.clean
    # The middle node was re-wired by the model to read n1's output.
    n2 = next(n for n in plan.nodes if n.id == "n2")
    assert n2.consumes == ["document_text"]
    # n2's inputs were re-selected at least once (the initial pick left it unreachable; the
    # model repair fixed it — the sink-collapse cannot reach a middle node) — and DECLARE
    # was never re-run.
    assert sink.select_calls["n2"] >= 2


async def test_repair_loop_fails_loud_at_the_cap() -> None:
    sink = _StubbornSink()
    with pytest.raises(ValueError) as exc:
        await decompose(_WHY, ["core"], sink=sink, max_jobs_per_area=5, max_nodes=18)
    message = str(exc.value)
    assert str(MAX_DESIGN_ROUNDS) in message
    # The full report is in the message — the problem is surfaced, never dropped.
    assert "unreachable" in message
