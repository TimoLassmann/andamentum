"""Contract validator for graph nodes (Phase 0 skeleton).

Phase 0 of the Move-3 plan. The contract validator iterates over the
registry of nodes that opt in by inheriting from
``andamentum.epistemic.graph.base.Node`` and asserts:

1. The body of ``run()`` only accesses ``ctx.state.<field>`` for
   fields in ``reads``; it only mutates ``ctx.state.<field>`` for
   fields in ``writes``.
2. Every ``_run_op(OpClass, ...)`` call's first argument is a class
   in ``operations``.

(There is no successors check — pyright + pydantic-graph already
enforce that the body returns only nodes in the ``run()`` return
type annotation. See the plan's "What pydantic-graph already provides"
section.)

In Phase 0 the registry is empty (no node has been migrated to
``Node`` yet). This file establishes the test surface so it's wired
into CI from day one. Phase 1 fills in the first contracts; later
phases populate the rest.
"""

from __future__ import annotations

from andamentum.epistemic.graph.base import Node


def _nodes_with_contracts() -> list[type[Node]]:
    """Discover all subclasses of ``Node`` (the contract-bearing
    base). Empty in Phase 0; grows as Move 3 phases land."""
    return list(Node.__subclasses__())


def test_registry_is_empty_or_nonempty() -> None:
    """Sanity test: this file imports cleanly, the Node base exists,
    and the registry is well-formed. It exists so that even before
    any node migrates, the contract-validator file is exercised by
    pytest collection."""
    nodes = _nodes_with_contracts()
    assert isinstance(nodes, list)


def test_every_contracted_node_has_required_metadata() -> None:
    """Each ``Node`` subclass must declare reads, writes, operations,
    and post_invariants — even as empty defaults. Catches regressions
    where someone subclasses Node but forgets to set the metadata
    (the empty default is a feature, not a foot-gun)."""
    for node_cls in _nodes_with_contracts():
        assert hasattr(node_cls, "reads"), f"{node_cls.__name__} missing reads"
        assert hasattr(node_cls, "writes"), f"{node_cls.__name__} missing writes"
        assert hasattr(node_cls, "operations"), (
            f"{node_cls.__name__} missing operations"
        )
        assert hasattr(node_cls, "post_invariants"), (
            f"{node_cls.__name__} missing post_invariants"
        )


# Phase 1+ will add:
#
# * test_node_body_only_touches_declared_state — AST walks the run()
#   method and asserts every ctx.state.X access maps to reads, every
#   assignment to writes (cross-cutting fields allowlisted).
#
# * test_node_only_dispatches_declared_operations — AST walks the
#   run() method and asserts every _run_op(OpClass, ...) call's first
#   argument is in `operations`.
#
# These checks are deferred to Phase 1 so they can be developed
# alongside the first migrated nodes (CheckCompletion, Synthesize),
# where the contract structure is concrete.
