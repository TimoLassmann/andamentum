"""Structural wiring guards.

These tests detect a class of bug we hit twice: an operation gets
registered (or a state field gets declared) but no graph node ever
references it, leaving it dormant. The fact that ``operations/__init__``
re-exports ``ReflectOnGapsOperation`` does NOT mean the graph ever
calls it. Only a name reference inside ``graph/nodes.py`` does.

These tests grep the graph package source for name references. They are
deliberately textual rather than AST-based: a textual reference is
sufficient to know a name is at least *visible* to the graph; an AST
walk would catch dynamically-resolved names too, but we don't dispatch
operations dynamically. Future graph nodes that resolve operation
classes by string lookup would fail this test, which is the correct
signal — make the wiring static, or add the string-lookup path to the
allowlist below with a justification.

If you delete an operation: also delete it from ``OPERATION_CLASSES``
in ``operations/__init__.py``. That keeps this test honest.

A third guard lives here for a closely-related shape: an operation that
IS wired, but wired with a dependency missing. Two graph nodes construct
their operation inline rather than through ``_make_op`` (they need a
``providers`` kwarg ``_make_op`` doesn't know about), so a dependency
added to ``_make_op`` and forgotten there is absent on the live graph
path and present everywhere else — invisible to every unit test that
builds the operation by hand. That is exactly how ``judge_model`` came
to be missing from the primary evidence-gathering path.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations import OPERATION_CLASSES
from andamentum.epistemic.operations.base import BaseOperation


GRAPH_NODES = Path(__file__).parent.parent / "graph" / "nodes.py"
GRAPH_DIR = Path(__file__).parent.parent / "graph"


# Operations that are intentionally NOT graph-driven and shouldn't fail
# this test. Each entry needs a justification so that future readers can
# tell the difference between "intentionally outside the graph" and
# "dormant code".
OPERATIONS_INTENTIONALLY_OUTSIDE_GRAPH: dict[str, str] = {
    # InvalidateEvidenceOperation and RevalidateClaimOperation are both
    # invoked from `_run_tms_sweep` (which is itself called from graph
    # nodes after evidence-invalidating operations). They DO appear in
    # nodes.py textually, so they shouldn't actually need this allowlist
    # — kept here as a placeholder showing the right pattern for future
    # entries.
}


# State fields exempt from the reader-presence check. Most fields read
# by `state.<field>` will match; private attrs and ones that exist for
# bookkeeping only go here.
STATE_FIELDS_INTENTIONALLY_UNREAD: set[str] = {
    "_quarantined_ids",  # internal dedup set behind the public quarantine API
}


def _read_graph_source() -> str:
    """Concatenate every .py file under graph/ for textual searches."""
    out: list[str] = []
    for f in sorted(GRAPH_DIR.glob("*.py")):
        out.append(f.read_text())
    return "\n".join(out)


class TestEveryOperationHasGraphCaller:
    """Every entry in OPERATION_CLASSES must be referenced (by class name)
    somewhere in the graph package. The two dormancy bugs we hit
    (``DecomposeQuestionOperation`` un-wired in v0.3, then
    ``ReflectOnGapsOperation`` left dormant after the post-audit fix
    queue) would both have failed this test."""

    def test_every_op_class_has_a_graph_reference(self) -> None:
        graph_source = _read_graph_source()
        dormant: list[str] = []
        for op_name, op_class in OPERATION_CLASSES.items():
            if op_name in OPERATIONS_INTENTIONALLY_OUTSIDE_GRAPH:
                continue
            class_name = op_class.__name__
            if class_name not in graph_source:
                dormant.append(f"{op_name} ({class_name})")
        assert not dormant, (
            "These operations are registered in OPERATION_CLASSES but "
            "no graph node references their class name — they are "
            "dormant. Either wire them into a graph node, delete them "
            "from OPERATION_CLASSES, or add them to "
            "OPERATIONS_INTENTIONALLY_OUTSIDE_GRAPH with a justification:"
            "\n  - " + "\n  - ".join(dormant)
        )

    def test_every_op_class_has_an_operation_string_caller(self) -> None:
        """Stronger version: each registered op_name must also appear as
        a string literal in graph source (the third arg to _run_op).
        Catches the case where a class is referenced in an import but
        never actually dispatched."""
        graph_source = _read_graph_source()
        unused: list[str] = []
        for op_name in OPERATION_CLASSES:
            if op_name in OPERATIONS_INTENTIONALLY_OUTSIDE_GRAPH:
                continue
            # Match either single or double quoted string.
            if (
                f'"{op_name}"' not in graph_source
                and f"'{op_name}'" not in graph_source
            ):
                unused.append(op_name)
        assert not unused, (
            "These operation names appear in OPERATION_CLASSES but never "
            "as a quoted string in any graph node — _run_op needs the "
            "operation name as its 6th argument, so absence here means "
            "the op is never actually executed:"
            "\n  - " + "\n  - ".join(unused)
        )


class TestEveryStateFieldHasReader:
    """Every public field on EpistemicGraphState must be read somewhere
    in the graph package — not just written. A field that's only
    written-to (or only declared) is dead state. The
    ``Objective.reflection_rounds`` field we just deleted is the
    entity-side analogue of this."""

    def test_every_state_field_is_referenced_outside_state_module(
        self,
    ) -> None:
        # Read sources OTHER than state.py — definition + write-back are
        # there; we want to know if anyone else reads the field.
        non_state_source = "\n".join(
            f.read_text()
            for f in sorted(GRAPH_DIR.glob("*.py"))
            if f.name != "state.py"
        )
        unread: list[str] = []
        for field_name in EpistemicGraphState.__dataclass_fields__:
            if field_name in STATE_FIELDS_INTENTIONALLY_UNREAD:
                continue
            # Look for `state.<field>` or `.<field>` (the deps + node
            # context attribute access patterns).
            if (
                f"state.{field_name}" not in non_state_source
                and f".{field_name}" not in non_state_source
            ):
                unread.append(field_name)
        assert not unread, (
            "These EpistemicGraphState fields are declared but never "
            "read by any graph node — they are dead state. Either use "
            "them in a node decision, delete them, or add them to "
            "STATE_FIELDS_INTENTIONALLY_UNREAD with a justification:"
            "\n  - " + "\n  - ".join(unread)
        )


# Constructor keywords that are legitimately specific to one inline
# construction and therefore excluded from the parity comparison. Each
# entry needs a justification.
INLINE_CONSTRUCTION_EXTRA_KWARGS: dict[str, str] = {
    # The reason these two constructions bypass _make_op at all: the
    # dispatch operations take a provider registry that _make_op has no
    # way to supply.
    "providers": "provider registry — the reason _make_op is bypassed",
}


def _base_operation_positional_names() -> list[str]:
    """``BaseOperation.__init__``'s positional parameters, in order."""
    params = list(inspect.signature(BaseOperation.__init__).parameters.values())
    return [
        p.name
        for p in params
        if p.name != "self" and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


def _dependency_names(call: ast.Call) -> set[str]:
    """The BaseOperation dependencies a construction call supplies.

    Positional arguments are resolved against ``BaseOperation.__init__``'s
    parameter order so that ``Op(repo, runner, embedding_model=...)`` and
    ``Op(repo=..., agent_runner=..., embedding_model=...)`` compare equal.
    """
    positional = _base_operation_positional_names()
    names = {positional[i] for i in range(len(call.args)) if i < len(positional)}
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs splat — cannot compare statically
            continue
        if kw.arg in INLINE_CONSTRUCTION_EXTRA_KWARGS:
            continue
        names.add(kw.arg)
    return names


def _operation_construction_calls(tree: ast.AST) -> dict[str, ast.Call]:
    """Every ``SomethingOperation(...)`` call in the tree, by class name."""
    found: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id.endswith("Operation"):
            found[func.id] = node
    return found


class TestInlineOperationConstructionsMatchMakeOp:
    """The two ``_make_op``-bypassing constructions must pass the same
    dependency set ``_make_op`` passes.

    ``PlanEvidence._run_dispatch`` and ``ResolveUncertainties``'
    ``_run_investigate`` build their operation by hand because it takes a
    ``providers`` kwarg. Everything else about the dependency set must be
    identical, or the operation behaves differently on the live graph
    path than the same operation reached through ``_make_op`` — and no
    unit test that constructs the operation directly can see the
    difference. This test compares the two argument sets structurally, so
    the NEXT dependency added to ``_make_op`` cannot silently skip them.
    """

    def _tree(self) -> ast.AST:
        return ast.parse(GRAPH_NODES.read_text())

    def _make_op_dependencies(self, tree: ast.AST) -> set[str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_make_op":
                calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
                assert calls, "_make_op no longer constructs anything"
                return _dependency_names(calls[0])
        raise AssertionError("_make_op not found in graph/nodes.py")

    def test_make_op_passes_every_base_operation_dependency(self) -> None:
        """Guards the reference set itself: if ``_make_op`` stops passing
        a ``BaseOperation`` dependency, comparing against it would pin the
        gap rather than catch it."""
        tree = self._tree()
        supplied = self._make_op_dependencies(tree)
        expected = set(_base_operation_positional_names()) | {"judge_model"}
        # ``validator`` is deliberately not threaded from deps — every
        # operation builds its own DefaultValidator and nothing in the
        # graph supplies one.
        expected.discard("validator")
        assert supplied == expected, (
            "_make_op no longer passes every BaseOperation dependency. "
            f"Missing: {sorted(expected - supplied)}. Extra: "
            f"{sorted(supplied - expected)}. Add the new dependency here "
            "AND to the inline constructions in _run_dispatch / "
            "_run_investigate."
        )

    def test_inline_constructions_pass_the_same_dependencies(self) -> None:
        tree = self._tree()
        reference = self._make_op_dependencies(tree)
        calls = _operation_construction_calls(tree)

        drifted: list[str] = []
        for class_name in ("DispatchGatherOperation", "InvestigateClaimOperation"):
            call = calls.get(class_name)
            assert call is not None, (
                f"{class_name} is no longer constructed inline in "
                "graph/nodes.py — if it now goes through _make_op, drop it "
                "from this test."
            )
            supplied = _dependency_names(call)
            if supplied != reference:
                drifted.append(
                    f"{class_name}: missing {sorted(reference - supplied)}, "
                    f"extra {sorted(supplied - reference)}"
                )

        assert not drifted, (
            "These inline operation constructions in graph/nodes.py do not "
            "pass the same dependency set as _make_op. A dependency missing "
            "here is missing ONLY on the live graph path — every unit test "
            "that builds the operation directly still passes. This is how "
            "judge_model came to be absent from the primary evidence-"
            "gathering path:\n  - " + "\n  - ".join(drifted)
        )
