"""Topology-reflection tests for the Strunk sub-graph.

The topology is a Python value (a dict) you can assert against
directly. These tests catch the recurring class of routing bug where
a node's declared successor doesn't actually lead anywhere useful.
"""

from __future__ import annotations

from andamentum.whetstone.lenses.strunk.topology import topology


EXPECTED_NODES = {
    "DeterministicScreen",
    "R11ActiveVoice",
    "R13OmitNeedlessWords",
    "ResolveDemands",
    "Aggregate",
}


def test_topology_lists_every_node():
    t = topology()
    assert set(t.keys()) == EXPECTED_NODES


def test_topology_linear_chain():
    """Phase A: DeterministicScreen → R11 → R13 → ResolveDemands → Aggregate → End."""
    t = topology()
    assert t["DeterministicScreen"]["successors"] == ["R11ActiveVoice"]
    assert t["R11ActiveVoice"]["successors"] == ["R13OmitNeedlessWords"]
    assert t["R13OmitNeedlessWords"]["successors"] == ["ResolveDemands"]
    assert t["ResolveDemands"]["successors"] == ["Aggregate"]
    assert t["Aggregate"]["successors"] == ["End"]


def test_topology_reachability_from_entry_node():
    """Every node is reachable from the entry, and End is reachable."""
    t = topology()
    visited: set[str] = set()
    frontier = ["DeterministicScreen"]
    while frontier:
        node = frontier.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for succ in t.get(node, {}).get("successors", []):
            frontier.append(succ)
    assert EXPECTED_NODES <= visited
    assert "End" in visited


def test_topology_kinds():
    t = topology()
    assert t["DeterministicScreen"]["kind"] == "deterministic"
    assert t["R11ActiveVoice"]["kind"] == "agent"
    assert t["R13OmitNeedlessWords"]["kind"] == "agent"
    assert t["ResolveDemands"]["kind"] == "control"
    assert t["Aggregate"]["kind"] == "control"


def test_topology_no_self_loops():
    for name, entry in topology().items():
        assert name not in entry["successors"], (
            f"{name} declares itself as a successor"
        )


def test_topology_no_orphan_successors():
    t = topology()
    known = set(t.keys()) | {"End"}
    for name, entry in t.items():
        for succ in entry["successors"]:
            assert succ in known, (
                f"{name} declares unknown successor {succ!r}"
            )


def test_topology_agent_metadata_present():
    t = topology()
    for name in ("R11ActiveVoice", "R13OmitNeedlessWords"):
        entry = t[name]
        assert "rule_number" in entry
        assert "rule_source" in entry
        assert "model" in entry
        assert "output_model" in entry
        assert "Strunk" in entry["rule_source"]


def test_topology_only_aggregate_terminates():
    terminators = [
        name for name, entry in topology().items() if "End" in entry["successors"]
    ]
    assert terminators == ["Aggregate"]
