"""Assemble the whetstone v2 graph.

Per the foundational principle: this file contains zero LLM calls and
zero domain logic. It just declares the node list. Adding a phase later
means adding the node class to this list (and editing the previous
phase's terminal node to return the new node instead of End).

Both pipelines share ``HarvestSource`` and ``ChunkAndScan``;
``ChunkAndScan`` branches on ``state.mode`` to dispatch into either
``CriticalRead`` (mode="review") or ``ExtractKeywords`` (mode="panel").

Edges (review mode):

    HarvestSource ──► ChunkAndScan
                         │
                         ├─ deps.model is None        ─► End[ReviewResult]
                         │
                         ├─ state.mode == "panel"     ─► ExtractKeywords ─► GenerateExpertPanel
                         │                                                       │
                         │                                                       └► ExpertReview
                         │                                                            │
                         │                                                            └► PanelSynthesise
                         │                                                                  │
                         │                                                                  └► End[ReviewResult]
                         │
                         └─ default (mode="review")   ─► CriticalRead
                                                              │
                                                              └► ReflectAndInvestigate
                                                                     │
                                                                     └► EditSections
                                                                            │
                                                                            └► Challenge
                                                                                  │
                                                                                  └► AuthorQuestions
                                                                                         │
                                                                                         └► ReconcileClaims
                                                                                                │
                                                                                                └► Consolidate
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
    Consolidate,
    CriticalRead,
    CustomReviewer,
    EditSections,
    EvaluateGuidelineItems,
    ExpertReview,
    ExtractCheckableItems,
    ExtractKeywords,
    GenerateExpertPanel,
    HarvestSource,
    NoveltyCheck,
    PanelSynthesise,
    ReconcileClaims,
    ReflectAndInvestigate,
    Synthesise,
)

review_graph = Graph(
    nodes=[
        HarvestSource,
        ChunkAndScan,
        # Review-mode nodes
        CriticalRead,
        ReflectAndInvestigate,
        NoveltyCheck,
        EditSections,
        Challenge,
        AuthorQuestions,
        ReconcileClaims,
        Consolidate,
        Synthesise,
        # Panel-mode nodes
        ExtractKeywords,
        GenerateExpertPanel,
        ExpertReview,
        PanelSynthesise,
        # Guidelines-mode nodes
        ExtractCheckableItems,
        EvaluateGuidelineItems,
        # Custom-criteria-mode node
        CustomReviewer,
    ]
)
