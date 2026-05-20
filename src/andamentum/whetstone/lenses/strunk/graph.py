"""Strunk sub-graph wiring.

Per the foundational principle: zero domain logic here. This module
just declares the node list, exposes the canonical node tuple
``NODE_CLASSES`` for topology reflection / tests, and provides the
``run_strunk_lens`` entrypoint that builds initial state and runs the
graph.

Sequence (linear; each rule node makes ONE LLM call per section):

    DeterministicScreen → R11ActiveVoice → R13OmitNeedlessWords
                       → ResolveDemands → Aggregate
                       → End[list[Finding]]
"""

from __future__ import annotations

from pydantic_graph import Graph

from ...schemas import Finding
from ...structural.types import SectionRef
from .nodes.aggregate import Aggregate
from .nodes.deterministic_screen import DeterministicScreen
from .nodes.r11_active_voice import R11ActiveVoice
from .nodes.r13_omit_needless_words import R13OmitNeedlessWords
from .nodes.resolve_demands import ResolveDemands
from .state import StrunkLensDeps, StrunkLensState


# Canonical ordered list of node classes in the sub-graph. Used by
# ``topology()`` and the structural tests. Adding a new rule node
# means adding it here AND inserting it into the run() return-type
# chain at the right point.
NODE_CLASSES: tuple[type, ...] = (
    DeterministicScreen,
    R11ActiveVoice,
    R13OmitNeedlessWords,
    ResolveDemands,
    Aggregate,
)


strunk_graph: Graph[StrunkLensState, StrunkLensDeps, list[Finding]] = Graph(
    nodes=list(NODE_CLASSES),  # type: ignore[arg-type]
)


async def run_strunk_lens(
    section: SectionRef,
    *,
    deps: StrunkLensDeps,
) -> list[Finding]:
    """Run the full Strunk sub-graph against one section.

    Builds a fresh ``StrunkLensState`` around the section, kicks off
    the graph at ``DeterministicScreen``, and returns the public
    ``Finding`` list produced by ``Aggregate``.
    """
    state = StrunkLensState(section=section)
    result = await strunk_graph.run(DeterministicScreen(), state=state, deps=deps)
    return result.output
