"""The decompose refine loop — converge to clean, or fail loud at the cap.

Two scenarios, both driven by a stub ``AgentSink`` (no live model):

  - A sink that types a MESSY board first (a dangling read), then — once fed the finding
    as feedback — returns a fixed typing. The loop must CONVERGE to a clean report.
  - A sink that NEVER fixes the flaw. The loop must FAIL LOUD after ``MAX_DESIGN_ROUNDS``
    with the structural problem in the message — never a half-wired design.
"""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from andamentum.core import AgentDefinition
from andamentum.forge.decompose import MAX_DESIGN_ROUNDS, decompose
from andamentum.forge.schemas import (
    DataKind,
    ForgeAreas,
    ForgeWhy,
    JobList,
    NodeTyping,
)
from andamentum.forge.spec import NodeKind

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
    """Types n1 cleanly; types n2 with a dangling read first, then fixes it once it sees
    feedback. Counts the type_node calls per node so the test can assert one repair."""

    def __init__(self) -> None:
        self.type_calls: dict[str, int] = {}

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "frame":
            return ForgeAreas(areas=["core"])
        if defn.name == "list_jobs":
            return JobList(jobs=["Read the document.", "Write the bullets."])
        if defn.name == "type_node":
            board = str(kwargs.get("board", ""))
            feedback = str(kwargs.get("feedback", ""))
            fid = _focus_id(board)
            self.type_calls[fid] = self.type_calls.get(fid, 0) + 1
            if fid == "n1":
                return NodeTyping(
                    kind=NodeKind.SPINE,
                    consumes=["input"],
                    produces=["document_text"],
                    produces_kind=DataKind.SIGNAL,
                )
            # n2: read a name nothing produces first; fix to the real producer on feedback.
            reads = "document_text" if feedback else "sources"
            return NodeTyping(
                kind=NodeKind.HEAD,
                consumes=[reads],
                produces=["bullets"],
                produces_kind=DataKind.SIGNAL,
            )
        raise AssertionError(f"unexpected agent {defn.name!r}")


class _StubbornSink:
    """Always types n2 with the same dangling read — the loop can never converge."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "frame":
            return ForgeAreas(areas=["core"])
        if defn.name == "list_jobs":
            return JobList(jobs=["Read the document.", "Write the bullets."])
        if defn.name == "type_node":
            fid = _focus_id(str(kwargs.get("board", "")))
            if fid == "n1":
                return NodeTyping(
                    kind=NodeKind.SPINE,
                    consumes=["input"],
                    produces=["document_text"],
                )
            return NodeTyping(
                kind=NodeKind.HEAD,
                consumes=["sources"],  # never produced — the flaw is never fixed
                produces=["bullets"],
            )
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
    # n2 was re-typed at least once (initial typing left a dangling read; the repair fixed it).
    assert sink.type_calls["n2"] >= 2


async def test_repair_loop_fails_loud_at_the_cap() -> None:
    sink = _StubbornSink()
    with pytest.raises(ValueError) as exc:
        await decompose(_WHY, ["core"], sink=sink, max_jobs_per_area=5, max_nodes=18)
    message = str(exc.value)
    assert str(MAX_DESIGN_ROUNDS) in message
    # The full report is in the message — the problem is surfaced, never dropped.
    assert "sources" in message
    assert "dangling_read" in message
