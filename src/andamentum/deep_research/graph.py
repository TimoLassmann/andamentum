"""Assemble research workflow graph.

Requires the [llm] optional extra: ``pip install mosaic-deep-research[llm]``
"""

from pydantic_graph import Graph

from .nodes import PlanResearch, SearchPhase, FetchPhase, SummarizePages, AnalyzeGaps, RefineSearch, Synthesize

# Assemble graph with all node classes
research_graph = Graph(
    nodes=[
        PlanResearch,
        SearchPhase,
        FetchPhase,
        SummarizePages,
        AnalyzeGaps,
        RefineSearch,
        Synthesize,
    ]
)
