"""Pure unit tests for the engine-free ``assemble`` worker — no model, no graph.

Each test builds a tiny ``NodeDraft`` board (names already canonical) and asserts the
matched ``DataGraph``: the producer/consumer maps, the edges, and that fan-in,
fan-out, and loop back-edges all fall out of name-matching.
"""

from __future__ import annotations

from andamentum.forge.assemble import assemble
from andamentum.forge.schemas import NodeDraft
from andamentum.forge.spec import NodeControl


def _node(
    node_id: str,
    consumes: list[str],
    produces: list[str],
    *,
    control: NodeControl = NodeControl.NONE,
) -> NodeDraft:
    return NodeDraft(id=node_id, consumes=consumes, produces=produces, control=control)


def test_linear_chain_maps_and_edges() -> None:
    nodes = [
        _node("n1", ["input"], ["parsed"]),
        _node("n2", ["parsed"], ["answer"]),
    ]
    graph = assemble(nodes)

    assert graph.writers == {"parsed": ["n1"], "answer": ["n2"]}
    assert graph.readers == {"parsed": ["n2"]}
    assert graph.edges == [("n1", "n2")]
    assert graph.inputs == ["input"]


def test_fan_in_one_node_reads_two_upstream_outputs() -> None:
    nodes = [
        _node("n1", ["input"], ["left"]),
        _node("n2", ["input"], ["right"]),
        _node("n3", ["left", "right"], ["merged"]),
    ]
    graph = assemble(nodes)

    assert graph.readers["left"] == ["n3"]
    assert graph.readers["right"] == ["n3"]
    # Both upstream producers point at the joining node — a fan-in.
    assert ("n1", "n3") in graph.edges
    assert ("n2", "n3") in graph.edges


def test_fan_out_two_nodes_read_one_output() -> None:
    nodes = [
        _node("n1", ["input"], ["shared"]),
        _node("n2", ["shared"], ["a"]),
        _node("n3", ["shared"], ["b"]),
    ]
    graph = assemble(nodes)

    assert graph.writers["shared"] == ["n1"]
    assert sorted(graph.readers["shared"]) == ["n2", "n3"]
    # One producer points at two consumers — a fan-out.
    assert ("n1", "n2") in graph.edges
    assert ("n1", "n3") in graph.edges


def test_loop_back_edge_is_an_edge_to_an_earlier_node() -> None:
    # n2 (a checkpoint) reads n3's output: a back edge n3 -> n2 forms the bounded loop.
    nodes = [
        _node("n1", ["input"], ["seed"]),
        _node("n2", ["seed", "refined"], ["candidate"], control=NodeControl.CHECKPOINT),
        _node("n3", ["candidate"], ["refined"]),
    ]
    graph = assemble(nodes)

    assert ("n2", "n3") in graph.edges  # forward
    assert ("n3", "n2") in graph.edges  # the back edge — the loop falls out of matching


def test_input_tokens_collected_not_treated_as_a_produced_variable() -> None:
    nodes = [_node("n1", ["brief"], ["out"])]
    graph = assemble(nodes)

    assert graph.inputs == ["brief"]
    assert "brief" not in graph.readers  # an input token is not a normal read edge
    assert graph.edges == []  # nothing produces 'brief', so no internal edge
