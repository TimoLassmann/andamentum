"""Node kinds for the Strunk sub-graph.

Every node in the Strunk lens declares its ``kind`` as a ``ClassVar``.
The discipline makes the deterministic-vs-LLM boundary visible at the
type level and at grep time, and lets a single test enforce that
``DETERMINISTIC`` nodes never reach for the agent runner.

Three kinds, deliberately:

* ``DETERMINISTIC`` — pure function. No LLM call, no I/O beyond
  reading and writing the graph state. Output is a function of input,
  so the node is testable with plain ``assertEquals`` on a fixture.
* ``AGENT`` — invokes an LLM through the executor in ``StrunkLensDeps``.
  Declares ``model`` and ``output_model`` ClassVars. Non-deterministic.
* ``CONTROL`` — graph plumbing (fan-out, aggregation, demand routing).
  Deterministic in mechanism, but its purpose is flow control rather
  than checking a Strunk rule.
"""

from __future__ import annotations

from enum import StrEnum


class NodeKind(StrEnum):
    """How a node operates. Declared as a ClassVar on every node."""

    DETERMINISTIC = "deterministic"
    AGENT = "agent"
    CONTROL = "control"
