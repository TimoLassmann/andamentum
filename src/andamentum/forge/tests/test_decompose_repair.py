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


class _RepairSink:
    """Declares a clean board (n1→document_text, n2→bullets); n2 first SELECTS a name not
    on the board (an empty read), then selects the real producer once it sees feedback.
    Counts the select_consumes calls per node so the test can assert one repair."""

    def __init__(self) -> None:
        self.select_calls: dict[str, int] = {}

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "frame":
            return ForgeAreas(areas=["core"])
        if defn.name == "list_jobs":
            return JobList(jobs=["Read the document.", "Write the bullets."])
        if defn.name == "declare_node":
            fid = _focus_id(str(kwargs.get("board", "")))
            if fid == "n1":
                return NodeDeclaration(
                    kind=NodeKind.SPINE,
                    produces="document_text",
                    produces_kind=DataKind.SIGNAL,
                )
            return NodeDeclaration(
                kind=NodeKind.HEAD,
                produces="bullets",
                produces_kind=DataKind.SIGNAL,
            )
        if defn.name == "select_consumes":
            board = str(kwargs.get("board", ""))
            feedback = str(kwargs.get("feedback", ""))
            fid = _focus_id(board)
            self.select_calls[fid] = self.select_calls.get(fid, 0) + 1
            options = _parse_options(str(kwargs.get("options", "")))
            if fid == "n1":
                return ConsumeSelection(consume_indices=[options["input"]])
            # n2: pick a name nothing produces first (→ empty read); fix on feedback.
            wanted = "document_text" if feedback else "sources"
            idxs = [options[wanted]] if wanted in options else []
            return ConsumeSelection(consume_indices=idxs)
        raise AssertionError(f"unexpected agent {defn.name!r}")


class _StubbornSink:
    """Always selects a non-existent input for n2 — the loop can never converge."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "frame":
            return ForgeAreas(areas=["core"])
        if defn.name == "list_jobs":
            return JobList(jobs=["Read the document.", "Write the bullets."])
        if defn.name == "declare_node":
            fid = _focus_id(str(kwargs.get("board", "")))
            if fid == "n1":
                return NodeDeclaration(kind=NodeKind.SPINE, produces="document_text")
            return NodeDeclaration(kind=NodeKind.HEAD, produces="bullets")
        if defn.name == "select_consumes":
            fid = _focus_id(str(kwargs.get("board", "")))
            options = _parse_options(str(kwargs.get("options", "")))
            if fid == "n1":
                return ConsumeSelection(consume_indices=[options["input"]])
            # never reads a real producer → n2 stays unreachable, the flaw is never fixed
            return ConsumeSelection(consume_indices=[])
        raise AssertionError(f"unexpected agent {defn.name!r}")


async def test_repair_loop_converges_to_clean() -> None:
    sink = _RepairSink()
    plan, report, _notes = await decompose(
        _WHY, ["core"], sink=sink, max_jobs_per_area=5, max_nodes=18
    )
    assert report.clean
    # The board is wired: n2 reads what n1 produces.
    n2 = next(n for n in plan.nodes if n.id == "n2")
    assert n2.consumes == ["document_text"]
    # n2's inputs were re-selected at least once (the initial pick left it unreachable; the
    # repair fixed it) — and DECLARE was never re-run.
    assert sink.select_calls["n2"] >= 2


async def test_repair_loop_fails_loud_at_the_cap() -> None:
    sink = _StubbornSink()
    with pytest.raises(ValueError) as exc:
        await decompose(_WHY, ["core"], sink=sink, max_jobs_per_area=5, max_nodes=18)
    message = str(exc.value)
    assert str(MAX_DESIGN_ROUNDS) in message
    # The full report is in the message — the problem is surfaced, never dropped.
    assert "unreachable" in message
