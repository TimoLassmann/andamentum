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
    CriticVerdict,
    DataKind,
    ForgeAreas,
    ForgeWhy,
    JobList,
    NodeTyping,
    PieceOut,
    RequirementsVerdict,
    SandboxResult,
)
from andamentum.forge.spec import NodeKind


def _focus_id(board: str) -> str:
    for line in board.splitlines():
        if line.strip().startswith(">>>"):
            m = re.search(r"n\d+", line)
            if m:
                return m.group()
    return ""


def _parse_fields(context: str, header: str) -> list[tuple[str, str]]:
    """The `ctx.state.<name>: <ann>` lines under a header (e.g. 'YOU MUST SET')."""
    out: list[tuple[str, str]] = []
    capture = False
    for line in context.splitlines():
        if line.startswith(header):
            capture = True
            continue
        if capture:
            m = re.match(r"\s+ctx\.state\.(\w+):\s*(.*)", line)
            if m:
                out.append((m.group(1), m.group(2).strip()))
            elif line.strip():
                capture = False
    return out


def _parse_successors(context: str) -> list[str]:
    for line in context.splitlines():
        if line.startswith("RETURN exactly one"):
            seg = line.split(":", 1)[1].split("—")[0]
            return [s.strip() for s in seg.split(",") if s.strip()]
    return []


def _draft_body(context: str) -> str:
    """Synthesise a contract-valid spine body from the draft context: read every declared
    input, set every declared output, return the first declared successor. Enough to pass
    all static gates (contract, purity, fail-loud, read/write coverage) and make the smoke
    graph run — the stub stands in for a real model."""
    lines = [
        f"_ = ctx.state.{name}" for name, _ in _parse_fields(context, "YOU MAY READ")
    ]
    for name, ann in _parse_fields(context, "YOU MUST SET"):
        lines.append(
            f"ctx.state.{name} = {'0' if ann.startswith('int') else chr(39) + 'x' + chr(39)}"
        )
    target = next((s for s in _parse_successors(context) if s != "End"), None)
    lines.append(f"return {target}()" if target else "return End('done')")
    return "\n".join(lines)


class FakeSandbox:
    """A stub ``SandboxPort`` — returns a scripted verdict, runs nothing."""

    def __init__(
        self, *, exit_code: int = 0, stdout: str = "", stderr: str = ""
    ) -> None:
        self._result = SandboxResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    def run(
        self,
        argv,
        *,
        cwd=None,
        extra_path=None,
        timeout=30,
        mem_mb=512,
        allow_network=False,
    ) -> SandboxResult:
        return self._result


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
        # --- stage 3/4 authoring + audit heads ---
        if defn.name in ("build_draft", "build_repair"):
            return PieceOut(body=_draft_body(str(kwargs.get("context", ""))))
        if defn.name == "requirements":
            return RequirementsVerdict(meets_brief=True, gaps=[])
        if defn.name == "critic":
            return CriticVerdict(issues=[])
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
