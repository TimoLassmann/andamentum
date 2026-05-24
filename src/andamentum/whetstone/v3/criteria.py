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


# ── EXTERNAL_COMMS — blog posts, articles, emails, op-eds, press releases ────

EXTERNAL_COMMS: list[Criterion] = [
    Criterion(
        name="Hook",
        questions=[
            "Does the opening earn the reader's attention — is there a clear "
            "reason to keep reading by the second or third sentence?",
            "Is the core message of the piece visible early, rather than "
            "buried under setup?",
        ],
    ),
    Criterion(
        name="Argument",
        questions=[
            "Is the main claim or take stated clearly enough that a reader "
            "could repeat it back in one sentence?",
            "Do the supporting points actually support the main claim, or "
            "are they adjacent material that doesn't move the argument?",
            "Are there obvious counterpoints the piece should address but "
            "ignores?",
        ],
    ),
    Criterion(
        name="Evidence",
        questions=[
            "Are factual claims backed up — data, examples, attribution, "
            "named sources?",
            "Where the piece quotes or paraphrases others, is the source "
            "identifiable and the framing fair?",
        ],
        facets=["claims", "gists", "citations"],
    ),
    Criterion(
        name="Voice",
        questions=[
            "Is the tone consistent throughout, or do passages slip into a "
            "different register (academic, marketing-speak, defensive)?",
            "Is the tone appropriate for the apparent audience?",
        ],
    ),
    Criterion(
        name="Clarity",
        questions=[
            "Are there passages that are unclear, jargon-heavy, or hard to "
            "follow for a general reader?",
            "Are sentences and paragraphs the right length, or do they "
            "stack into walls of text the reader will bounce off?",
        ],
    ),
]


# ── GENERAL — notes, drafts, books, technical docs, internal writeups ────────

GENERAL: list[Criterion] = [
    Criterion(
        name="Purpose",
        questions=[
            "Is the goal of the document clear — what is this for, who reads "
            "it, what should they do after reading?",
            "Are there sections or passages that don't serve the stated "
            "purpose?",
        ],
    ),
    Criterion(
        name="Structure",
        questions=[
            "Is the document organised logically — does the order of "
            "sections make sense for the reader's path through the material?",
            "Is the main point of each section easy to locate?",
        ],
    ),
    Criterion(
        name="Completeness",
        questions=[
            "Does the document cover what its scope implies it should? Are "
            "there obvious gaps a reader would notice?",
            "Are there sections that are clearly stubs or placeholders "
            "needing more content?",
        ],
    ),
    Criterion(
        name="Clarity",
        questions=[
            "Are there passages that are confusing, ambiguous, or hard to "
            "follow?",
            "Is terminology used consistently?",
        ],
    ),
]


# ── Selection by document type ──────────────────────────────────────────────

_SETS: dict[str, list[Criterion]] = {
    "academic": SPECS,
    "external_communication": EXTERNAL_COMMS,
    "general": GENERAL,
}


def criterion_set_for(document_type: str) -> list[Criterion]:
    """The active criterion set for a document type.

    Three sets ship today: ``academic`` → SPECS (Story/Presentation/
    Evaluations/Correctness/Significance), ``external_communication`` →
    EXTERNAL_COMMS (Hook/Argument/Evidence/Voice/Clarity),
    ``general`` → GENERAL (Purpose/Structure/Completeness/Clarity).

    Unknown document types fall back to GENERAL — the safest neutral
    set. The earlier behaviour (silent fallback to SPECS for everything)
    was structurally wrong: applying Evaluations or Correctness to an
    essay surfaces forced findings (\"the essay lacks a baseline\") that
    are useless to the author.

    The mechanism is general, the content is data — add new sets by
    extending ``_SETS``.
    """
    return _SETS.get(document_type, GENERAL)
