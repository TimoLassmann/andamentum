"""Structural tests for the Strunk sub-graph's NodeKind discipline.

These are static / introspection-only tests — they make no LLM call,
need no fixtures, and run in milliseconds. They enforce the discipline
the user can grep for: every node declares its ``kind`` as a ClassVar,
agent nodes declare ``model`` and ``output_model``, deterministic
nodes never import the core agent runner.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from andamentum.whetstone.lenses.strunk.graph import NODE_CLASSES
from andamentum.whetstone.lenses.strunk.kinds import NodeKind


def test_every_node_declares_kind():
    for cls in NODE_CLASSES:
        assert hasattr(cls, "kind"), f"{cls.__name__} is missing 'kind' ClassVar"
        assert isinstance(cls.kind, NodeKind), (  # type: ignore[attr-defined]
            f"{cls.__name__}.kind is not a NodeKind: {cls.kind!r}"  # type: ignore[attr-defined]
        )


def test_every_node_declares_reads_and_writes():
    for cls in NODE_CLASSES:
        assert hasattr(cls, "reads")
        assert isinstance(cls.reads, frozenset)  # type: ignore[attr-defined]
        assert hasattr(cls, "writes")
        assert isinstance(cls.writes, frozenset)  # type: ignore[attr-defined]


def test_agent_nodes_declare_model_and_output_model():
    for cls in NODE_CLASSES:
        if cls.kind != NodeKind.AGENT:  # type: ignore[attr-defined]
            continue
        assert hasattr(cls, "model"), (
            f"{cls.__name__} is AGENT but missing 'model' ClassVar"
        )
        assert isinstance(cls.model, str)  # type: ignore[attr-defined]
        assert hasattr(cls, "output_model"), (
            f"{cls.__name__} is AGENT but missing 'output_model' ClassVar"
        )
        assert cls.output_model is not None  # type: ignore[attr-defined]
        # And a rule number / source so the topology reflects it.
        assert hasattr(cls, "rule_number")
        assert hasattr(cls, "rule_source")


def test_agent_nodes_have_unique_rule_numbers():
    """If two AGENT nodes report the same rule_number, downstream
    Finding categories collide and dedup gets confused."""
    seen = {}
    for cls in NODE_CLASSES:
        if cls.kind != NodeKind.AGENT:  # type: ignore[attr-defined]
            continue
        rn = cls.rule_number  # type: ignore[attr-defined]
        assert rn not in seen, (
            f"{cls.__name__} and {seen[rn].__name__} both claim rule_number={rn}"
        )
        seen[rn] = cls


def _module_imports(cls: type) -> list[str]:
    """Return every imported module name and ``from X import Y`` target."""
    source_file = inspect.getsourcefile(cls)
    assert source_file is not None
    tree = ast.parse(Path(source_file).read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for alias in node.names:
                names.append(f"{base}.{alias.name}" if base else alias.name)
    return names


def test_deterministic_nodes_do_not_import_agent_machinery():
    """A DETERMINISTIC node's module must not import the agent runner
    or any agent-related symbol.

    The discipline is enforced statically: walking the AST of the
    node module's file means a violation is caught even before any
    test imports the node. Catches accidentally-LLM-using
    'deterministic' helpers."""
    forbidden = {
        "andamentum.core.agents",
        "AgentRunner",
        "AgentDefinition",
        "build_pydantic_ai_agent",
    }
    for cls in NODE_CLASSES:
        if cls.kind != NodeKind.DETERMINISTIC:  # type: ignore[attr-defined]
            continue
        for imp in _module_imports(cls):
            for bad in forbidden:
                assert bad not in imp, (
                    f"{cls.__name__} is DETERMINISTIC but its module "
                    f"imports {imp!r} which contains {bad!r}"
                )


def test_agent_nodes_import_agent_definition():
    """An AGENT node should reference an AgentDefinition (its own).
    Catches the inverse mistake — an LLM-using node that forgot to
    declare itself as one."""
    for cls in NODE_CLASSES:
        if cls.kind != NodeKind.AGENT:  # type: ignore[attr-defined]
            continue
        imports = _module_imports(cls)
        assert any("AgentDefinition" in imp for imp in imports), (
            f"{cls.__name__} is AGENT but module imports no AgentDefinition"
        )
