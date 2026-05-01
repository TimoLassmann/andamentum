"""Graph topology as a Python value, derived from run() return annotations.

Phase 0 of the Move-3 plan. The graph topology — which nodes can
transition to which — is already encoded in each node's ``run()``
return type annotation: pydantic-graph reads it to build the graph,
and pyright enforces the body returns only declared successors.

This module exposes that information as a dict-of-sets you can
inspect, iterate, diff, and assert against — *without running the
graph*. It uses ``typing.get_type_hints`` and ``typing.get_args`` to
reflect over each node's annotation; no metadata duplication is
required.

Usage::

    >>> from andamentum.epistemic.graph.topology import topology
    >>> t = topology()
    >>> t[AbandonOrDemote]
    frozenset({Scrutinize, PromoteToSupported})

    >>> # Static check: CheckCompletion must NOT be in
    >>> # AbandonOrDemote's successors. The recurring routing bug.
    >>> assert CheckCompletion not in t[AbandonOrDemote]

The reflection works on any ``BaseNode`` subclass; the new ``Node``
base class (in ``base.py``) is not required. This means we can run
topology() against the existing ``nodes.py`` immediately and validate
its global properties before any node migration begins.
"""

from __future__ import annotations

import typing
from typing import Union

from pydantic_graph import BaseNode, End


def _extract_successors(annotation: object) -> frozenset[type[BaseNode]]:
    """Pull successor node classes out of a return type annotation.

    Handles three shapes:

    * ``NodeA`` — a single node class
    * ``Union[NodeA, NodeB, ...]`` — explicit alternation
    * ``End[T]`` and ``"NodeA"`` (forward ref) — End[T] is filtered out
      (terminal); forward refs are resolved by ``get_type_hints``.

    Returns a frozenset of node classes (excluding ``End[...]``).
    """
    out: set[type[BaseNode]] = set()
    args = typing.get_args(annotation)
    candidates: list[object] = list(args) if args else [annotation]
    for c in candidates:
        # End[T] is a generic terminal — skip.
        origin = typing.get_origin(c)
        if origin is End or c is End:
            continue
        # Plain class.
        if isinstance(c, type) and issubclass(c, BaseNode):
            out.add(c)
            continue
        # Generic Union nested inside another Union — recurse.
        if origin is Union:
            out.update(_extract_successors(c))
            continue
    return frozenset(out)


def topology(
    nodes: "list[type[BaseNode]] | None" = None,
) -> dict[type[BaseNode], frozenset[type[BaseNode]]]:
    """Return ``{node_class: {successor_classes}}`` for each given node.

    When ``nodes`` is None, defaults to all ``BaseNode`` subclasses
    importable from ``andamentum.epistemic.graph.nodes``. Each entry
    is derived from the node's ``run()`` return type annotation;
    no node-side metadata is read.

    Successors are *direct* — not transitive. Use a graph traversal
    to compute reachability (see ``reachable_from`` below).
    """
    if nodes is None:
        from . import nodes as nodes_module
        from .base import Node as _NodeBase

        nodes = [
            cls
            for name in dir(nodes_module)
            if isinstance(cls := getattr(nodes_module, name), type)
            and issubclass(cls, BaseNode)
            and cls is not BaseNode
            # Skip abstract bases (BaseNode subclasses that don't
            # define a concrete graph node themselves). Currently
            # this is just our own ``Node`` class from base.py;
            # any future intermediate base classes should be added
            # to this exclusion. A class is treated as "abstract"
            # if its run() method is inherited rather than defined.
            and cls is not _NodeBase
            and "run" in cls.__dict__
        ]

    out: dict[type[BaseNode], frozenset[type[BaseNode]]] = {}
    for node_cls in nodes:
        # Resolve forward refs (e.g. Union["NodeA", "NodeB"]) using the
        # node's own module globals. ``include_extras=False`` strips
        # any Annotated[] wrappers we don't care about here.
        try:
            hints = typing.get_type_hints(
                node_cls.run, include_extras=False
            )
        except Exception:
            # If the run method has unresolvable forward refs, fall
            # back to raw annotation lookup. This shouldn't happen for
            # graph nodes but let's not crash on it.
            hints = getattr(node_cls.run, "__annotations__", {})
        return_annotation = hints.get("return")
        if return_annotation is None:
            out[node_cls] = frozenset()
            continue
        out[node_cls] = _extract_successors(return_annotation)
    return out


def reachable_from(
    start: type[BaseNode],
    topo: "dict[type[BaseNode], frozenset[type[BaseNode]]] | None" = None,
) -> frozenset[type[BaseNode]]:
    """Return the set of nodes reachable from ``start`` (transitive
    closure of successors). Includes ``start`` itself."""
    if topo is None:
        topo = topology()
    seen: set[type[BaseNode]] = set()
    stack: list[type[BaseNode]] = [start]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(topo.get(cur, frozenset()))
    return frozenset(seen)


def all_nodes(
    topo: "dict[type[BaseNode], frozenset[type[BaseNode]]] | None" = None,
) -> frozenset[type[BaseNode]]:
    """Return the set of all nodes in the topology — keys plus
    successors. Useful for "is everything reachable?" checks."""
    if topo is None:
        topo = topology()
    out: set[type[BaseNode]] = set(topo.keys())
    for succ in topo.values():
        out.update(succ)
    return frozenset(out)
