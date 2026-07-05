"""Test fixtures — a scripted agent stub so the whole forge graph runs with no model.

``ScriptedSink`` satisfies the ``AgentSink`` Port (the dialect agent test seam): it
answers each design head from a canned script, keyed by agent name and — for the two-pass
node design heads (``declare_node`` / ``select_consumes``) — by the focus node id parsed
from the board. A script's intent per node is a ``NodeScript`` (kind + the names it means
to consume/produce); the sink translates that intent into the real head shapes: a
``NodeDeclaration`` for the DECLARE pass and (by mapping the intended consume NAMES onto
the numbered option list the SELECT pass is shown) a ``ConsumeSelection`` of ORDINALS. The
fallback declaration produces a unique datum per node, so an under-specified script never
collides.
"""

from __future__ import annotations

import re

import pytest
from pydantic import BaseModel

from andamentum.core import AgentDefinition
from andamentum.forge.schemas import (
    ConsumeSelection,
    CriticVerdict,
    DataKind,
    Fitness,
    ForgeAreas,
    ForgeWhy,
    JobList,
    NodeDeclaration,
    PieceOut,
    RequirementsVerdict,
    SandboxResult,
)
from andamentum.forge.spec import NodeControl, NodeKind


class NodeScript(BaseModel):
    """A test's intended I/O for one node, expressed in NAMES (as a human would).

    The sink turns it into the real two-pass head shapes: ``produces[0]`` becomes the
    DECLARE pass's single produced name, and ``consumes`` (names) are mapped onto ordinals
    against the numbered option list the SELECT pass is shown.
    """

    kind: NodeKind = NodeKind.SPINE
    consumes: list[str] = []
    produces: list[str] = []
    produces_kind: DataKind = DataKind.SIGNAL
    control: NodeControl = NodeControl.NONE
    network: bool = False


def _parse_options(options: str) -> dict[str, int]:
    """Map each option NAME → its ordinal, parsed from the SELECT pass's numbered list
    (``'0. input — …'`` / ``'3. ranked_items — produced by …'``)."""
    out: dict[str, int] = {}
    for line in options.splitlines():
        m = re.match(r"\s*(\d+)\.\s+(\S+)", line)
        if m:
            out[m.group(2)] = int(m.group(1))
    return out


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
        typings: dict[str, NodeScript] | None = None,
    ) -> None:
        self.why = why
        self.areas = areas
        self.jobs_by_area = jobs_by_area
        self.typings = typings or {}

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "understand":
            return self.why
        if defn.name == "fitness":
            # Default to a buildable rung-1 verdict so the gate passes; tests that
            # exercise refusal subclass this and override the fitness answer.
            return Fitness(
                realizable_as_function=True,
                rung="function",
                reason="stub: treated as a function",
                suggested_reshape="",
            )
        if defn.name == "frame":
            return ForgeAreas(areas=self.areas)
        if defn.name == "list_jobs":
            area = str(kwargs.get("area", ""))
            return JobList(jobs=self.jobs_by_area.get(area, []))
        if defn.name == "declare_node":
            fid = _focus_id(str(kwargs.get("board", "")))
            script = self.typings.get(fid)
            if script is not None and script.produces:
                return NodeDeclaration(
                    kind=script.kind,
                    produces=script.produces[0],
                    produces_kind=script.produces_kind,
                    control=script.control,
                    network=script.network,
                )
            return NodeDeclaration(kind=NodeKind.SPINE, produces=f"out_{fid}")
        if defn.name == "select_consumes":
            fid = _focus_id(str(kwargs.get("board", "")))
            script = self.typings.get(fid)
            wanted = script.consumes if script is not None else ["input"]
            name_to_index = _parse_options(str(kwargs.get("options", "")))
            indices = [name_to_index[w] for w in wanted if w in name_to_index]
            return ConsumeSelection(consume_indices=indices)
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
            "n1": NodeScript(
                kind=NodeKind.SPINE, consumes=["input"], produces=["parsed_request"]
            ),
            "n2": NodeScript(
                kind=NodeKind.HEAD, consumes=["parsed_request"], produces=["answer"]
            ),
        },
    )
