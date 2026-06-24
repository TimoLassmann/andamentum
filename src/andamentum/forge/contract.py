"""Per-node contracts and the holes the renderer leaves — spec-derived, deterministic.

A ``Hole`` is an intentional ``NotImplementedError`` the renderer left for real logic;
the builder fills it. A ``NodeContract`` is the typed bundle of what a node may read,
must write, and may return — the single object the builder fills *against* and the
draft/repair prompts are built *from*, so "what this node must do" has one source.

Ported from the ``forge`` dump (``spec/holes.py`` + ``spec/contract.py``). Leaf worker
file: ``pydantic`` + stdlib + the sibling ``spec`` only; no graph engine.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from .spec import END, ModelSpec, NodeKind, SystemSpec


class HoleKind(str, Enum):
    SPINE_BODY = "spine_body"  # spine node run(): the whole body is the hole
    ROUTING = "routing"  # multi-successor head run(): route on `out`
    GATE_DECIDE = "gate_decide"  # HumanGate decide(): route on the human's answer


class Hole(BaseModel):
    """An intentional ``NotImplementedError`` the renderer left for real logic."""

    node: str  # the node class name
    kind: HoleKind
    # Rich fields — populated by AST discovery; empty for spec-derived holes.
    method: str = ""  # "run" or "decide"
    file: Path | None = None
    hint: str = ""  # the NotImplementedError message
    signature: str = ""  # the def / async def line, as it appears in source
    context: str = ""  # preamble already in the body (routing: the run_head call)

    model_config = {"arbitrary_types_allowed": True}


class IOField(BaseModel):
    """One field a node reads or writes — a State field name with its type."""

    name: str
    annotation: str
    optional: bool = False


class NodeContract(BaseModel):
    """Everything needed to fill (and test) one node, with no further spec lookups."""

    node: str
    kind: NodeKind
    reads: list[IOField]
    writes: list[IOField]
    successors: list[str]  # node class names, or the END sentinel ("End")
    agent_output: ModelSpec | None = None  # the `out` schema, for head nodes

    @property
    def terminal(self) -> bool:
        return END in self.successors


def node_contract(spec: SystemSpec, node_name: str) -> NodeContract:
    """Resolve ``node_name``'s contract against ``spec``'s State and agents."""
    node = next(n for n in spec.nodes if n.name == node_name)
    state_by_name = {f.name: f for f in spec.state.fields}
    primary = spec.input.primary_text_field

    def resolve(name: str) -> IOField:
        f = state_by_name.get(name)
        if f is not None:
            return IOField(name=f.name, annotation=f.annotation, optional=f.optional)
        if name == primary:
            return IOField(name=name, annotation="str", optional=False)
        return IOField(name=name, annotation="str", optional=True)

    agent_output: ModelSpec | None = None
    if node.kind is NodeKind.HEAD and node.agent:
        agent_output = next(
            (a.output for a in spec.agents if a.name == node.agent), None
        )

    return NodeContract(
        node=node.name,
        kind=node.kind,
        reads=[resolve(n) for n in node.reads],
        writes=[resolve(n) for n in node.writes],
        successors=list(node.successors),
        agent_output=agent_output,
    )
