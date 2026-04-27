"""Assemble the whetstone v2 graph.

Per the foundational principle: this file contains zero LLM calls and
zero domain logic. It just declares the node list. Adding a phase later
means adding the node class to this list (and editing the previous
phase's terminal node to return the new node instead of End).

Edges (which node returns which next):

    HarvestSource ──► ChunkAndScan
                         │
                         ├─ deps.model is None ─► End[ReviewResult]
                         │
                         └─ deps.model is set  ─► CriticalRead
                                                       │
                                                       └► ReflectAndInvestigate
                                                              │
                                                              └► EditSections
                                                                    │
                                                                    └► Challenge
                                                                          │
                                                                          └► AuthorQuestions
                                                                                 │
                                                                                 └► Synthesise
                                                                                        │
                                                                                        └► End[ReviewResult]
"""

from pydantic_graph import Graph

from .nodes import (
    AuthorQuestions,
    Challenge,
    ChunkAndScan,
    CriticalRead,
    EditSections,
    HarvestSource,
    ReflectAndInvestigate,
    Synthesise,
)

review_graph = Graph(
    nodes=[
        HarvestSource,
        ChunkAndScan,
        CriticalRead,
        ReflectAndInvestigate,
        EditSections,
        Challenge,
        AuthorQuestions,
        Synthesise,
    ]
)
