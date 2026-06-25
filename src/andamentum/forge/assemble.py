"""Worker: assemble a declared node board into a data DAG (puzzle-fit).

Pure and engine-free — stdlib + the sibling schemas only. Given the freely-declared
``consumes``/``produces`` names on each ``NodeDraft`` (already canonicalised by the
caller via :func:`naming.canonical_datum`), it matches producers to consumers and
yields a typed :class:`DataGraph`:

  - ``writers``  variable name → the node ids that produce it
  - ``readers``  variable name → the node ids that consume it
  - ``edges``    (producer node id, consumer node id) for every matched variable
  - ``inputs``   the input tokens actually read by some node (the graph's door)

The grammar falls out of the matching: a node reading two upstream outputs is a
fan-in; two nodes reading one output is a fan-out; a back edge (a producer reading a
later producer's output) is a loop. Nothing is forced linear, and no diagnosis happens
here — :mod:`diagnose` inspects this graph. Building the maps is fully deterministic:
iteration follows the node order on the board and the declared name order.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .schemas import INPUT_TOKENS, NodeDraft


class DataGraph(BaseModel):
    """The data DAG matched from a node board — producers, consumers, edges, inputs."""

    writers: dict[str, list[str]] = Field(default_factory=dict)
    readers: dict[str, list[str]] = Field(default_factory=dict)
    edges: list[tuple[str, str]] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)


def assemble(nodes: list[NodeDraft]) -> DataGraph:
    """Match producers→consumers across the board and return the typed data DAG."""
    writers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    inputs: list[str] = []
    seen_inputs: set[str] = set()

    for node in nodes:
        for name in node.produces:
            writers.setdefault(name, []).append(node.id)
        for name in node.consumes:
            if name in INPUT_TOKENS:
                if name not in seen_inputs:
                    seen_inputs.add(name)
                    inputs.append(name)
                continue
            readers.setdefault(name, []).append(node.id)

    # One edge per (producer, consumer) for every variable both a writer and a reader name.
    edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for node in nodes:
        for name in node.consumes:
            for producer in writers.get(name, ()):
                edge = (producer, node.id)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)

    return DataGraph(writers=writers, readers=readers, edges=edges, inputs=inputs)


__all__ = ["DataGraph", "assemble"]
