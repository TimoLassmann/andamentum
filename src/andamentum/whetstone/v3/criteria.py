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
            "Are there obvious counterpoints the piece should address but ignores?",
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
            "Are there sections or passages that don't serve the stated purpose?",
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
            "Are there passages that are confusing, ambiguous, or hard to follow?",
            "Is terminology used consistently?",
        ],
    ),
]


# ── ESSAY — personal essays, narrative essays, opinion essays ───────────────

ESSAY: list[Criterion] = [
    Criterion(
        name="Thesis",
        questions=[
            "Is there an arguable claim the essay is making — something a "
            "thoughtful reader could disagree with?",
            "Is the claim non-obvious, or does it restate something the "
            "intended reader would already accept?",
            "Can the thesis be located in the text, or does the reader have "
            "to reconstruct it from scattered hints?",
        ],
    ),
    Criterion(
        name="Narrative arc",
        questions=[
            "Does the essay build toward its insight, or does it state the "
            "conclusion early and then circle it?",
            "Does each section earn its place in the sequence — would "
            "reordering or cutting a section damage the argument?",
            "Does the ending land somewhere the opening could not have "
            "reached directly?",
        ],
    ),
    Criterion(
        name="Specificity",
        questions=[
            "Is the writing grounded in concrete detail — scenes, objects, "
            "moments — rather than abstract assertion?",
            "When the essay makes a general claim, is it anchored to at "
            "least one specific instance the reader can picture?",
            "Are the examples particular to this author's view, or could "
            "they have come from anyone writing on the topic?",
        ],
    ),
    Criterion(
        name="Voice",
        questions=[
            "Is there a distinctive authorial presence on the page — a "
            "sensibility the reader could recognise in another piece?",
            "Does the voice sustain through the essay, or does it flatten "
            "into generic prose in the middle sections?",
        ],
    ),
    Criterion(
        name="Fresh observation",
        questions=[
            "Does the essay say something the intended reader has not "
            "already heard many times on this topic?",
            "Is the prose free of cliché, stock phrasing, and received "
            "wisdom presented as insight?",
            "Where the essay reaches for a familiar idea, does it push past "
            "the familiar framing into something the author actually saw?",
        ],
    ),
]


# ── TUTORIAL — how-tos, technical walkthroughs, cookbooks ───────────────────

TUTORIAL: list[Criterion] = [
    Criterion(
        name="Goal",
        questions=[
            "Is what the reader will accomplish stated explicitly before the "
            "first step?",
            "Will the reader know, by the end, whether they succeeded — is "
            "there a concrete end state to check against?",
        ],
    ),
    Criterion(
        name="Prerequisites",
        questions=[
            "Are required tools, versions, accounts, and prior knowledge "
            "listed before the first step?",
            "If the reader is missing a prerequisite, will they find out "
            "before they start work — not three steps in?",
        ],
    ),
    Criterion(
        name="Step ordering",
        questions=[
            "Does each step follow naturally from the previous one, with no "
            "leaps that assume unstated prior knowledge?",
            "Are there gaps between steps where the reader has to guess at "
            "an intermediate action?",
            "Does any step depend on state produced by a later step?",
        ],
    ),
    Criterion(
        name="Correctness",
        questions=[
            "Are commands, code, file paths, and measurements accurate as "
            "written — would copy-pasting them work?",
            "If a reader followed the steps literally, with no improvisation, "
            "would they reach the stated goal?",
            "Are version-specific details flagged, so a reader on a different "
            "version knows what may differ?",
        ],
    ),
    Criterion(
        name="Completeness",
        questions=[
            "Does the tutorial cover the failure modes a real reader will "
            "hit, or only the happy path?",
            "When something can go wrong at a step, is there guidance on "
            "what the failure looks like and how to recover?",
            "Are common variations of the reader's setup acknowledged, or "
            "is one environment assumed without saying so?",
        ],
    ),
]


# ── CREATIVE — short fiction, memoir, narrative non-fiction ─────────────────

CREATIVE: list[Criterion] = [
    Criterion(
        name="Premise",
        questions=[
            "Is the central situation interesting in itself — would the "
            "reader want to know how it resolves?",
            "Is there a question the piece is asking, implicitly or "
            "explicitly, that the reader can feel pulling them forward?",
        ],
    ),
    Criterion(
        name="Character & voice",
        questions=[
            "Are the characters distinct from one another in speech, action, "
            "and the details they notice?",
            "Does the narrative voice match the material — is the register "
            "right for what is being told?",
            "Would the reader recognise a character from a single line of "
            "their dialogue or a single gesture?",
        ],
    ),
    Criterion(
        name="Scene & sensory grounding",
        questions=[
            "Does the prose put the reader inside specific moments — places, "
            "objects, bodies — rather than summarising at a distance?",
            "Is exposition kept to what the scene needs, or does it stack "
            "up between the moments that carry the story?",
            "When the piece tells the reader something, does it also show "
            "the reader enough to feel it?",
        ],
    ),
    Criterion(
        name="Tension",
        questions=[
            "Does each scene have stakes — something the character wants, "
            "risks, or stands to lose?",
            "Does the piece sustain forward momentum, or are there stretches "
            "where nothing is at risk and nothing changes?",
        ],
    ),
    Criterion(
        name="Prose craft",
        questions=[
            "Does the language carry rhythm and image, or does it default to "
            "the first phrasing that came to hand?",
            "Is the prose economical — are there sentences, modifiers, or "
            "whole paragraphs that could be cut without loss?",
            "Where the piece reaches for an image or metaphor, is it fresh, "
            "or one the reader has met many times before?",
        ],
    ),
]


# ── Selection by document type ──────────────────────────────────────────────

_SETS: dict[str, list[Criterion]] = {
    "academic": SPECS,
    "external_communication": EXTERNAL_COMMS,
    "essay": ESSAY,
    "tutorial": TUTORIAL,
    "creative": CREATIVE,
    "general": GENERAL,
}


def criterion_set_for(document_type: str) -> list[Criterion]:
    """The active criterion set for a document type.

    Six sets ship today: ``academic`` → SPECS (Story/Presentation/
    Evaluations/Correctness/Significance), ``external_communication`` →
    EXTERNAL_COMMS (Hook/Argument/Evidence/Voice/Clarity),
    ``essay`` → ESSAY (Thesis/Narrative arc/Specificity/Voice/Fresh
    observation), ``tutorial`` → TUTORIAL (Goal/Prerequisites/Step
    ordering/Correctness/Completeness), ``creative`` → CREATIVE
    (Premise/Character & voice/Scene & sensory grounding/Tension/Prose
    craft), and ``general`` → GENERAL (Purpose/Structure/Completeness/
    Clarity).

    Unknown document types fall back to GENERAL — the safest neutral
    set. The earlier behaviour (silent fallback to SPECS for everything)
    was structurally wrong: applying Evaluations or Correctness to an
    essay surfaces forced findings (\"the essay lacks a baseline\") that
    are useless to the author.

    The mechanism is general, the content is data — add new sets by
    extending ``_SETS``.
    """
    return _SETS.get(document_type, GENERAL)
