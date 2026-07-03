"""Step 4 — critic node attribution: the ``NodeFinding`` schema and the rapidfuzz
reconciliation of a critic-named node string to a real spec node name (§4.4 signal 4)."""

from __future__ import annotations

from andamentum.forge.audit import _reconcile_node
from andamentum.forge.schemas import CriticVerdict, NodeFinding

_NODES = ["ParseTheRequest", "NormaliseTheRequest", "AnswerTheRequest"]


def test_node_finding_is_flat_two_string_fields() -> None:
    f = NodeFinding(node="ParseTheRequest", issue="hardcoded stand-in")
    assert f.node == "ParseTheRequest"
    assert f.issue == "hardcoded stand-in"


def test_critic_verdict_carries_node_findings() -> None:
    v = CriticVerdict(issues=[NodeFinding(node="AnswerTheRequest", issue="drops input")])
    assert v.issues[0].node == "AnswerTheRequest"


def test_reconcile_exact_name_wins() -> None:
    assert _reconcile_node("NormaliseTheRequest", _NODES) == "NormaliseTheRequest"


def test_reconcile_near_miss_maps_to_real_node() -> None:
    # A slightly-off name (missing 'The') is reconciled to the real node.
    assert _reconcile_node("ParseRequest", _NODES) == "ParseTheRequest"


def test_reconcile_unreconcilable_is_dropped() -> None:
    assert _reconcile_node("SomethingEntirelyDifferent", _NODES) == ""


def test_reconcile_against_empty_node_list_drops() -> None:
    assert _reconcile_node("ParseTheRequest", []) == ""
