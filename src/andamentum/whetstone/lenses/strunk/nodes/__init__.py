"""Strunk sub-graph nodes.

Each node is a ``pydantic_graph.BaseNode`` subclass that declares its
``kind`` as a ClassVar (one of ``NodeKind.DETERMINISTIC``,
``NodeKind.AGENT``, ``NodeKind.CONTROL``). The structural tests in
``../tests/test_node_kinds.py`` walk every class registered here and
enforce that AGENT nodes carry ``model`` and ``output_model`` ClassVars
while DETERMINISTIC nodes never import the core agent runner.
"""

from __future__ import annotations
