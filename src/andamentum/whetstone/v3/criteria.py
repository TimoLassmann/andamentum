"""Pluggable criterion sets — the review *content*, kept out of the graph.

A `Criterion` is just a name + a list of atomic, checkable questions + which
document-model facets it reads. SPECS is the default set for academic
documents; other document types (or the user) select a different set. The graph
never names "Story" or "Evaluations" — it fans the one generic review node over
whichever set is active.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Facet = Literal["claims", "gists", "citations"]


class Criterion(BaseModel):
    name: str
    questions: list[str] = Field(default_factory=list)
    facets: list[Facet] = Field(default_factory=lambda: ["claims", "gists"])


# ── SPECS — the academic default set ────────────────────────────────────────

SPECS: list[Criterion] = [
    Criterion(
        name="Story",
        questions=[
            "Is the problem the work addresses stated clearly?",
            "Is the claimed gap or limitation in prior work made explicit?",
            "Is the core contribution stated clearly, and does the rest of the "
            "document actually deliver it?",
            "Do the central claims follow from the evidence presented?",
        ],
    ),
    Criterion(
        name="Presentation",
        questions=[
            "Are there passages that are unclear, disorganised, or hard to follow?",
            "Is the main message of each major section easy to locate?",
            "Is terminology used consistently across the document?",
        ],
    ),
    Criterion(
        name="Evaluations",
        questions=[
            "Are claimed results backed by evidence (data, experiments, "
            "comparisons) in the document?",
            "Where results are claimed, are baselines / comparisons / metrics "
            "adequate and clearly defined?",
            "Is enough detail given to trust or reproduce the evaluation?",
        ],
    ),
    Criterion(
        name="Correctness",
        questions=[
            "Are there equations, derivations, algorithms, or numerical "
            "statements that look wrong or internally inconsistent?",
            "Do numbers stated in different places agree with each other?",
        ],
    ),
    Criterion(
        name="Significance",
        questions=[
            "Are novelty or contribution claims supported, or overstated "
            "relative to prior work?",
            "Are there obvious missing comparisons or related work the document "
            "should address?",
        ],
        facets=["claims", "gists", "citations"],
    ),
]


# ── Selection by document type (general; SPECS is one set among several) ─────

_SETS: dict[str, list[Criterion]] = {"academic": SPECS}


def criterion_set_for(document_type: str) -> list[Criterion]:
    """The active criterion set for a document type. Falls back to SPECS until
    other sets are defined — the mechanism is general, the content is data."""
    return _SETS.get(document_type, SPECS)
