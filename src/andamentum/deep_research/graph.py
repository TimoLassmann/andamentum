"""Assemble research workflow graph."""

from __future__ import annotations

from pydantic_graph import GraphBuilder, StepContext

from .models import EvidenceReport
from .nodes import (
    AnalyzeGaps,
    FetchPhase,
    GenerateOne,
    NodeDeps,
    ParallelSearch,
    PlanResearch,
    PrepareSearchCycle,
    RefineSearch,
    SummarizePages,
    Synthesize,
    Verify,
)
from .state import ResearchState

# Assemble graph with all node classes. The search cycle is decomposed
# into PrepareSearchCycle → GenerateOne ⇄ Verify → ParallelSearch (see
# nodes.py for the per-slot retry loop and skip-and-tighten logic).
#
# pydantic-graph v2 has no ``Graph(nodes=[...])`` constructor: nodes are
# registered on a GraphBuilder and the run is entered through an input
# rather than a start-node instance, so a one-line entry step constructs
# the first node. The nodes themselves stay plain ``BaseNode`` subclasses.
_builder = GraphBuilder[ResearchState, NodeDeps, None, EvidenceReport](
    state_type=ResearchState,
    deps_type=NodeDeps,
    output_type=EvidenceReport,
)


@_builder.step
async def enter(ctx: StepContext[ResearchState, NodeDeps, None]) -> PlanResearch:
    """Entry step: hand the graph its first node."""
    return PlanResearch()


_builder.add(
    _builder.edge_from(_builder.start_node).to(enter),
    _builder.node(PlanResearch),
    _builder.node(PrepareSearchCycle),
    _builder.node(GenerateOne),
    _builder.node(Verify),
    _builder.node(ParallelSearch),
    _builder.node(FetchPhase),
    _builder.node(SummarizePages),
    _builder.node(AnalyzeGaps),
    _builder.node(RefineSearch),
    _builder.node(Synthesize),
)

research_graph = _builder.build()
