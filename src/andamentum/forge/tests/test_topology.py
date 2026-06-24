"""The dialect topology test — the forge graph is statically well-formed.

Reflects over each step's ``run`` return annotation (the declared successors), then
asserts: every step is reachable from the entry, every path can reach ``End``, and no
step dead-ends. This is the structural guarantee behind resume + audit (dialect L4).
"""

from __future__ import annotations

import types
import typing

from pydantic_graph import BaseNode, End

from andamentum.forge.graph import (
    Audit,
    Build,
    Compile,
    Decompose,
    Finish,
    Frame,
    Render,
    Understand,
    Verify,
)

_NODES = [Understand, Frame, Decompose, Compile, Render, Verify, Build, Audit, Finish]
_ENTRY = Understand


def _successors(cls: type[BaseNode]) -> set[str]:
    ret = typing.get_type_hints(cls.run)["return"]
    # Only a Union return declares multiple successors; split on it alone, so a bare
    # `End[ForgeResult]` is not mistaken for its payload type (whose get_args it is).
    is_union = typing.get_origin(ret) in (typing.Union, types.UnionType)
    members = list(typing.get_args(ret)) if is_union else [ret]
    out: set[str] = set()
    for m in members:
        if m is End or typing.get_origin(m) is End:
            out.add("End")
        elif isinstance(m, type) and issubclass(m, BaseNode):
            out.add(m.__name__)
    return out


def test_every_step_reaches_a_successor() -> None:
    for cls in _NODES:
        assert _successors(cls), f"{cls.__name__} declares no successor"


def test_all_reachable_and_end_reachable() -> None:
    adj = {cls.__name__: _successors(cls) for cls in _NODES}
    reached: set[str] = set()
    end_reachable = False
    stack = [_ENTRY.__name__]
    while stack:
        cur = stack.pop()
        if cur in reached:
            continue
        reached.add(cur)
        for s in adj.get(cur, set()):
            if s == "End":
                end_reachable = True
            elif s not in reached:
                stack.append(s)

    all_names = {cls.__name__ for cls in _NODES}
    assert all_names <= reached, f"unreachable: {all_names - reached}"
    assert end_reachable, "no End reachable from the entry"


def test_branch_points_are_the_stage_gates() -> None:
    # Compile / Verify / Build each early-exit to Finish on `stop_after` (or continue) —
    # the branches that make the topology test load-bearing. Every branch includes Finish.
    branchy = {cls.__name__ for cls in _NODES if len(_successors(cls)) > 1}
    assert branchy == {"Compile", "Verify", "Build"}, branchy
    for cls in _NODES:
        succ = _successors(cls)
        if len(succ) > 1:
            assert "Finish" in succ, (
                f"{cls.__name__} branch must include the Finish early-exit"
            )
