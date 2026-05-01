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
"""

from __future__ import annotations

from pathlib import Path

from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations import OPERATION_CLASSES


GRAPH_NODES = (
    Path(__file__).parent.parent / "graph" / "nodes.py"
)
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
