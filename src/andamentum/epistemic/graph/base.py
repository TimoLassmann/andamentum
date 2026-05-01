"""Node base class with explicit contract metadata.

Phase 0 of the Move-3 plan (`docs/superpowers/plans/2026-05-01-graph-node-contracts.md`).

Existing nodes inherit from ``pydantic_graph.BaseNode`` directly. This
module adds a thin subclass — ``Node`` — that carries class-level
metadata declaring each node's state I/O, operation dispatch, and
post-conditions. **Successors are NOT duplicated as metadata** — they
are encoded in the ``run()`` method's return type annotation, which
pyright and pydantic-graph already enforce.

Existing nodes are not migrated yet; that's Phase 1+. This module just
provides the foundation. Phase 0's only behaviour change is that
contract metadata is now *available* — no node uses it yet.

The validator (``test_node_contracts.py``) iterates over the registry
of nodes that opt in by inheriting from ``Node`` (rather than
``BaseNode`` directly). Until Phase 1, that registry is empty.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Optional,
)

from pydantic_graph import BaseNode

from .deps import EpistemicDeps
from .result import EpistemicResult
from .state import EpistemicGraphState

if TYPE_CHECKING:
    from ..entities import Claim
    from ..operations.base import BaseOperation


# ── Invariant typedef ────────────────────────────────────────────────


# An Invariant is a predicate over (state, claims) that returns None
# when the invariant holds, or a non-empty violation message string
# when it doesn't. Used for post_invariants — checks that must hold
# after a node runs.
#
# Example: ``no_stranded_claims`` (in ``invariants.py``) returns None
# when no claim is at SUPPORTED with integrated_assessment=None and
# not in verification_done; returns a message like "stranded: <id>"
# when the invariant is violated.
Invariant = Callable[
    [EpistemicGraphState, "list[Claim]"], Optional[str]
]


# ── Node base class ──────────────────────────────────────────────────


class Node(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Epistemic graph node with explicit contract metadata.

    Subclass this instead of ``BaseNode`` directly to declare:

    * ``reads`` — state fields the node reads from ``ctx.state``
    * ``writes`` — state fields the node mutates on ``ctx.state``
    * ``operations`` — operation classes dispatched via ``_run_op``
    * ``post_invariants`` — predicates that must hold over (state, claims)
      after the node runs

    Successors are encoded in the ``run()`` method's return type
    annotation; do not duplicate them here. Pyright and pydantic-graph
    already enforce the annotation; the ``topology()`` helper reads it
    to build the graph as data.

    Empty defaults so existing tests keep passing during the migration.
    Phase 1+ migrates each node to this base and fills in the metadata.
    """

    # State fields this node reads from ``ctx.state``. Validator asserts
    # the body only accesses ``ctx.state.<field>`` for fields in this
    # set. Cross-cutting fields (``operations_log``, ``successful``,
    # ``failed``) are written by ``_run_op`` itself, not by node
    # bodies — they're allowlisted in the validator.
    reads: ClassVar[frozenset[str]] = frozenset()

    # State fields this node writes on ``ctx.state``. Same validator
    # rules as ``reads``.
    writes: ClassVar[frozenset[str]] = frozenset()

    # Operations this node dispatches via ``_run_op``. Validator
    # asserts the body's ``_run_op(OpClass, ...)`` calls all use a
    # class in this set.
    operations: ClassVar[frozenset["type[BaseOperation]"]] = frozenset()

    # Predicates that must hold over (state, claims) AFTER this node
    # runs. Each is an ``Invariant``. The reachability test loops over
    # state patterns and asserts every node's post_invariants hold
    # after running.
    post_invariants: ClassVar[tuple[Invariant, ...]] = ()
