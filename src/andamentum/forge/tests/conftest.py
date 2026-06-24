"""Test fixtures — a scripted agent stub so the whole forge graph runs with no model.

``ScriptedSink`` satisfies the ``AgentSink`` Port (the dialect agent test seam): it
answers each design head from a canned script, keyed by agent name and — for
``type_node`` — by the focus node id parsed from the board. The fallback typing
produces a unique datum per node, so an under-specified script never collides.
"""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from andamentum.core import AgentDefinition
from andamentum.forge.schemas import (
    DataKind,
    ForgeAreas,
    ForgeWhy,
    JobList,
    NodeTyping,
)
from andamentum.forge.spec import NodeKind


def _focus_id(board: str) -> str:
    for line in board.splitlines():
        if line.strip().startswith(">>>"):
            m = re.search(r"n\d+", line)
            if m:
                return m.group()
    return ""


class ScriptedSink:
    """A canned ``AgentSink`` for tests — no live model."""

    def __init__(
        self,
        *,
        why: ForgeWhy,
        areas: list[str],
        jobs_by_area: dict[str, list[str]],
        typings: dict[str, NodeTyping] | None = None,
    ) -> None:
        self.why = why
        self.areas = areas
        self.jobs_by_area = jobs_by_area
        self.typings = typings or {}

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "understand":
            return self.why
        if defn.name == "frame":
            return ForgeAreas(areas=self.areas)
        if defn.name == "list_jobs":
            area = str(kwargs.get("area", ""))
            return JobList(jobs=self.jobs_by_area.get(area, []))
        if defn.name == "type_node":
            fid = _focus_id(str(kwargs.get("board", "")))
            return self.typings.get(fid) or NodeTyping(
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=[f"out_{fid}"],
                produces_kind=DataKind.SIGNAL,
            )
        raise AssertionError(f"unexpected agent {defn.name!r}")


@pytest.fixture
def reading_list_sink() -> ScriptedSink:
    """A small coherent script: parse the request (spine) → answer it (head)."""
    return ScriptedSink(
        why=ForgeWhy(
            purpose="Help the user manage a personal reading list.",
            boundary_in="a natural-language request",
            boundary_out="a text answer",
        ),
        areas=["core"],
        jobs_by_area={"core": ["Parse the request.", "Answer the request."]},
        typings={
            "n1": NodeTyping(
                kind=NodeKind.SPINE, consumes=["input"], produces=["parsed_request"]
            ),
            "n2": NodeTyping(
                kind=NodeKind.HEAD, consumes=["parsed_request"], produces=["answer"]
            ),
        },
    )
