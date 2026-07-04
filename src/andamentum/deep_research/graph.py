"""Assemble research workflow graph."""

from __future__ import annotations

from pydantic_graph import Graph

from .nodes import (
    AnalyzeGaps,
    FetchPhase,
    GenerateOne,
    ParallelSearch,
    PlanResearch,
    PrepareSearchCycle,
    RefineSearch,
    SummarizePages,
    Synthesize,
    Verify,
)

# Assemble graph with all node classes. The search cycle is decomposed
# into PrepareSearchCycle → GenerateOne ⇄ Verify → ParallelSearch (see
# nodes.py for the per-slot retry loop and skip-and-tighten logic).
research_graph = Graph(
    nodes=[
        PlanResearch,
        PrepareSearchCycle,
        GenerateOne,
        Verify,
        ParallelSearch,
        FetchPhase,
        SummarizePages,
        AnalyzeGaps,
        RefineSearch,
        Synthesize,
    ]
)
