"""Acceptance: a rung-2 (stateful) function builds and actually remembers across runs.

End-to-end through the real pipeline: a brief whose design declares a durable entity is
rendered + built, then the generated run-entry is called TWICE against the same database
file — the second run must see what the first saved. The design is all-spine, so the entry
makes no model calls and the proof is fully offline (no Ollama, no container).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from andamentum.core import AgentDefinition
from andamentum.forge import run_forge
from andamentum.forge.schemas import (
    DataKind,
    Fitness,
    ForgeWhy,
    PieceOut,
)
from andamentum.forge.spec import NodeKind
from pydantic import BaseModel

from .conftest import (
    FakeSandbox,
    NodeScript,
    ScriptedSink,
    _parse_fields,
    _parse_successors,
)


def _concat_body(context: str) -> str:
    """Author a body that concatenates every declared input into every declared output, so
    the loaded prior value visibly flows into the result — proving persistence, not just
    that the graph ran. ``or ''`` tolerates the first-run empty value."""
    reads = [n for n, _ in _parse_fields(context, "YOU MAY READ")]
    writes = [n for n, _ in _parse_fields(context, "YOU MUST SET")]
    succ = next((s for s in _parse_successors(context) if s != "End"), None)
    expr = " + ".join(f"str(ctx.state.{r} or '')" for r in reads) or "''"
    lines = [f"ctx.state.{w} = {expr}" for w in writes]
    if succ:
        lines.append(f"return {succ}()")
    elif writes:
        lines.append(f"return End(ctx.state.{writes[0]})")
    else:
        lines.append("return End('done')")
    return "\n".join(lines)


class _AppendSink(ScriptedSink):
    """A rung-2 design (one durable entity) whose authored bodies carry inputs forward."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "fitness":
            return Fitness(
                realizable_as_function=True,
                rung="stateful_function",
                reason="declares a durable entity loaded at start and saved at end",
                suggested_reshape="",
            )
        if defn.name in ("build_draft", "build_repair"):
            return PieceOut(body=_concat_body(str(kwargs.get("context", ""))))
        return await super().run(defn, **kwargs)


def _stateful_sink() -> _AppendSink:
    return _AppendSink(
        why=ForgeWhy(
            purpose="Remember and update a value across runs.",
            boundary_in="a request",
            boundary_out="the updated value",
        ),
        areas=["core"],
        jobs_by_area={
            "core": ["update the saved value with the request", "format the answer"]
        },
        typings={
            # n1 reads the saved value + the request and rewrites the saved value: a
            # read-modify-write durable entity (the §7 round-trip signature).
            "n1": NodeScript(
                kind=NodeKind.SPINE,
                consumes=["saved_value", "input"],
                produces=["saved_value"],
                produces_kind=DataKind.ENTITY,
            ),
            "n2": NodeScript(
                kind=NodeKind.SPINE,
                consumes=["saved_value"],
                produces=["answer"],
                produces_kind=DataKind.SIGNAL,
            ),
        },
    )


async def test_stateful_function_remembers_across_runs(tmp_path: Path) -> None:
    out_dir = tmp_path / "pkg"
    result = await run_forge(
        "Remember and update a value across runs.",
        model="test",
        dest=out_dir,
        stop_after="build",
        sink=_stateful_sink(),
        sandbox=FakeSandbox(),
    )
    assert result.build is not None and result.build.all_filled, result.build
    assert result.spec.entities, "the design should declare a durable entity"
    assert result.spec.entities[0].state_field, (
        "the entity should be bound to a State field"
    )
    name = result.spec.name

    db = str(tmp_path / "memory.db")
    added = str(out_dir) not in sys.path
    if added:
        sys.path.insert(0, str(out_dir))
    saved_mods = {
        k: v for k, v in sys.modules.items() if k == name or k.startswith(name + ".")
    }
    for k in list(saved_mods):
        del sys.modules[k]
    try:
        pkg = importlib.import_module(name)
        run = getattr(pkg, f"run_{name}")
        first = await run("alpha", model="unused", store=db)
        second = await run("beta", model="unused", store=db)  # same file → remembers
        third = await run("gamma", model="unused", store=None)  # in-memory → forgets
    finally:
        for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[k]
        sys.modules.update(saved_mods)
        if added and str(out_dir) in sys.path:
            sys.path.remove(str(out_dir))

    assert "alpha" in first
    # The second run loaded what the first saved → its output carries the first input.
    assert "alpha" in second and "beta" in second, (
        f"second run did not remember the first: {second!r}"
    )
    # A fresh in-memory run starts blank — it remembers nothing.
    assert "alpha" not in third and "gamma" in third, third
