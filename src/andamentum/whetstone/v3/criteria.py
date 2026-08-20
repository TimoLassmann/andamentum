"""The criterion registry — the review *content*, kept out of the graph.

A `Criterion` is just a name + a list of atomic, checkable questions + which
document-model facets it reads. SPECS is the default set for academic
documents; other document types (or the user) select a different set. The graph
never names "Story" or "Evaluations" — it fans the one generic review node over
whichever set is active.

Every criterion this package ships is a **module-level constant** and every
document-type set is a list of those constants — never a fresh `Criterion`
literal. Consumers that select criteria (a document type here, a reviewer
profile elsewhere) reference the same object, so each question has exactly one
wording and a copy-paste that produced a second one fails the identity test in
``tests/test_criterion_registry.py``.

The registry is deliberately larger than the union of the six document-type
sets: criteria written for a reviewer profile live here too, so that a profile
selects rather than carries its questions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ``gists`` is inert. Nothing in the package branches on it: ``review._project``
# gates the CLAIMS block on ``claims`` and the CITATIONS line on ``citations``,
# but the SECTIONS block — which is where section gists are rendered — is
# appended to every criterion prompt unconditionally. The member is kept
# because shipped criteria declare it and removing it would change shipped
# data; new criteria omit it rather than imply a gate that does not exist.
Facet = Literal["claims", "gists", "citations"]


class Criterion(BaseModel):
    name: str
    questions: list[str] = Field(default_factory=list)
    facets: list[Facet] = Field(default_factory=lambda: ["claims", "gists"])

    # Frozen so that a registry constant cannot be edited in place by whoever
    # happens to hold a reference. Without this, "this profile's criterion *is*
    # the registry constant" would be a guarantee about identity and none about
    # content. Nothing in the package mutates a Criterion.
    model_config = {"frozen": True}


# ── The walk instruction ────────────────────────────────────────────────────
#
# Criteria that must read the document in order carry this as their first
# question. The shared review scaffold does not invite a full read — its tool
# discipline block leans the other way — so the instruction belongs to the
# criteria that need it, in one wording, rather than being restated per
# criterion.
_WALK_THE_DOCUMENT = (
    "Call read_section on every section id in the SECTIONS block, in the "
    "order listed, before answering. Do not answer from the gists."
)


# ── SPECS — the academic default set ────────────────────────────────────────

ACADEMIC_STORY = Criterion(
    name="Story",
    questions=[
        "Is the problem the work addresses stated clearly?",
        "Is the claimed gap or limitation in prior work made explicit?",
        "Is the core contribution stated clearly, and does the rest of the "
        "document actually deliver it?",
        "Do the central claims follow from the evidence presented?",
    ],
)

ACADEMIC_PRESENTATION = Criterion(
    name="Presentation",
    questions=[
        "Are there passages that are unclear, disorganised, or hard to follow?",
        "Is the main message of each major section easy to locate?",
        "Is terminology used consistently across the document?",
    ],
)

ACADEMIC_EVALUATIONS = Criterion(
    name="Evaluations",
    questions=[
        "Are claimed results backed by evidence (data, experiments, "
        "comparisons) in the document?",
        "Where results are claimed, are baselines / comparisons / metrics "
        "adequate and clearly defined?",
        "Is enough detail given to trust or reproduce the evaluation?",
    ],
)

ACADEMIC_CORRECTNESS = Criterion(
    name="Correctness",
    questions=[
        "Are there equations, derivations, algorithms, or numerical "
        "statements that look wrong or internally inconsistent?",
        "Do numbers stated in different places agree with each other?",
    ],
)

ACADEMIC_SIGNIFICANCE = Criterion(
    name="Significance",
    questions=[
        "Are novelty or contribution claims supported, or overstated "
        "relative to prior work?",
        "Are there obvious missing comparisons or related work the document "
        "should address?",
    ],
    facets=["claims", "gists", "citations"],
)

SPECS: list[Criterion] = [
    ACADEMIC_STORY,
    ACADEMIC_PRESENTATION,
    ACADEMIC_EVALUATIONS,
    ACADEMIC_CORRECTNESS,
    ACADEMIC_SIGNIFICANCE,
]


# ── EXTERNAL_COMMS — blog posts, articles, emails, op-eds, press releases ────

EXTERNAL_COMMS_HOOK = Criterion(
    name="Hook",
    questions=[
        "Does the opening earn the reader's attention — is there a clear "
        "reason to keep reading by the second or third sentence?",
        "Is the core message of the piece visible early, rather than "
        "buried under setup?",
    ],
)

EXTERNAL_COMMS_ARGUMENT = Criterion(
    name="Argument",
    questions=[
        "Is the main claim or take stated clearly enough that a reader "
        "could repeat it back in one sentence?",
        "Do the supporting points actually support the main claim, or "
        "are they adjacent material that doesn't move the argument?",
        "Are there obvious counterpoints the piece should address but ignores?",
    ],
)

EXTERNAL_COMMS_EVIDENCE = Criterion(
    name="Evidence",
    questions=[
        "Are factual claims backed up — data, examples, attribution, named sources?",
        "Where the piece quotes or paraphrases others, is the source "
        "identifiable and the framing fair?",
    ],
    facets=["claims", "gists", "citations"],
)

EXTERNAL_COMMS_VOICE = Criterion(
    name="Voice",
    questions=[
        "Is the tone consistent throughout, or do passages slip into a "
        "different register (academic, marketing-speak, defensive)?",
        "Is the tone appropriate for the apparent audience?",
    ],
)

EXTERNAL_COMMS_CLARITY = Criterion(
    name="Clarity",
    questions=[
        "Are there passages that are unclear, jargon-heavy, or hard to "
        "follow for a general reader?",
        "Are sentences and paragraphs the right length, or do they "
        "stack into walls of text the reader will bounce off?",
    ],
)

EXTERNAL_COMMS: list[Criterion] = [
    EXTERNAL_COMMS_HOOK,
    EXTERNAL_COMMS_ARGUMENT,
    EXTERNAL_COMMS_EVIDENCE,
    EXTERNAL_COMMS_VOICE,
    EXTERNAL_COMMS_CLARITY,
]


# ── GENERAL — notes, drafts, books, technical docs, internal writeups ────────

GENERAL_PURPOSE = Criterion(
    name="Purpose",
    questions=[
        "Is the goal of the document clear — what is this for, who reads "
        "it, what should they do after reading?",
        "Are there sections or passages that don't serve the stated purpose?",
    ],
)

GENERAL_STRUCTURE = Criterion(
    name="Structure",
    questions=[
        "Is the document organised logically — does the order of "
        "sections make sense for the reader's path through the material?",
        "Is the main point of each section easy to locate?",
    ],
)

GENERAL_COMPLETENESS = Criterion(
    name="Completeness",
    questions=[
        "Does the document cover what its scope implies it should? Are "
        "there obvious gaps a reader would notice?",
        "Are there sections that are clearly stubs or placeholders "
        "needing more content?",
    ],
)

GENERAL_CLARITY = Criterion(
    name="Clarity",
    questions=[
        "Are there passages that are confusing, ambiguous, or hard to follow?",
        "Is terminology used consistently?",
    ],
)

GENERAL: list[Criterion] = [
    GENERAL_PURPOSE,
    GENERAL_STRUCTURE,
    GENERAL_COMPLETENESS,
    GENERAL_CLARITY,
]


# ── ESSAY — personal essays, narrative essays, opinion essays ───────────────

ESSAY_THESIS = Criterion(
    name="Thesis",
    questions=[
        "Is there an arguable claim the essay is making — something a "
        "thoughtful reader could disagree with?",
        "Is the claim non-obvious, or does it restate something the "
        "intended reader would already accept?",
        "Can the thesis be located in the text, or does the reader have "
        "to reconstruct it from scattered hints?",
    ],
)

ESSAY_NARRATIVE_ARC = Criterion(
    name="Narrative arc",
    questions=[
        "Does the essay build toward its insight, or does it state the "
        "conclusion early and then circle it?",
        "Does each section earn its place in the sequence — would "
        "reordering or cutting a section damage the argument?",
        "Does the ending land somewhere the opening could not have reached directly?",
    ],
)

ESSAY_SPECIFICITY = Criterion(
    name="Specificity",
    questions=[
        "Is the writing grounded in concrete detail — scenes, objects, "
        "moments — rather than abstract assertion?",
        "When the essay makes a general claim, is it anchored to at "
        "least one specific instance the reader can picture?",
        "Are the examples particular to this author's view, or could "
        "they have come from anyone writing on the topic?",
    ],
)

ESSAY_VOICE = Criterion(
    name="Voice",
    questions=[
        "Is there a distinctive authorial presence on the page — a "
        "sensibility the reader could recognise in another piece?",
        "Does the voice sustain through the essay, or does it flatten "
        "into generic prose in the middle sections?",
    ],
)

ESSAY_FRESH_OBSERVATION = Criterion(
    name="Fresh observation",
    questions=[
        "Does the essay say something the intended reader has not "
        "already heard many times on this topic?",
        "Is the prose free of cliché, stock phrasing, and received "
        "wisdom presented as insight?",
        "Where the essay reaches for a familiar idea, does it push past "
        "the familiar framing into something the author actually saw?",
    ],
)

ESSAY: list[Criterion] = [
    ESSAY_THESIS,
    ESSAY_NARRATIVE_ARC,
    ESSAY_SPECIFICITY,
    ESSAY_VOICE,
    ESSAY_FRESH_OBSERVATION,
]


# ── TUTORIAL — how-tos, technical walkthroughs, cookbooks ───────────────────

TUTORIAL_GOAL = Criterion(
    name="Goal",
    questions=[
        "Is what the reader will accomplish stated explicitly before the first step?",
        "Will the reader know, by the end, whether they succeeded — is "
        "there a concrete end state to check against?",
    ],
)

TUTORIAL_PREREQUISITES = Criterion(
    name="Prerequisites",
    questions=[
        "Are required tools, versions, accounts, and prior knowledge "
        "listed before the first step?",
        "If the reader is missing a prerequisite, will they find out "
        "before they start work — not three steps in?",
    ],
)

TUTORIAL_STEP_ORDERING = Criterion(
    name="Step ordering",
    questions=[
        "Does each step follow naturally from the previous one, with no "
        "leaps that assume unstated prior knowledge?",
        "Are there gaps between steps where the reader has to guess at "
        "an intermediate action?",
        "Does any step depend on state produced by a later step?",
    ],
)

TUTORIAL_CORRECTNESS = Criterion(
    name="Correctness",
    questions=[
        "Are commands, code, file paths, and measurements accurate as "
        "written — would copy-pasting them work?",
        "If a reader followed the steps literally, with no improvisation, "
        "would they reach the stated goal?",
        "Are version-specific details flagged, so a reader on a different "
        "version knows what may differ?",
    ],
)

TUTORIAL_COMPLETENESS = Criterion(
    name="Completeness",
    questions=[
        "Does the tutorial cover the failure modes a real reader will "
        "hit, or only the happy path?",
        "When something can go wrong at a step, is there guidance on "
        "what the failure looks like and how to recover?",
        "Are common variations of the reader's setup acknowledged, or "
        "is one environment assumed without saying so?",
    ],
)

TUTORIAL: list[Criterion] = [
    TUTORIAL_GOAL,
    TUTORIAL_PREREQUISITES,
    TUTORIAL_STEP_ORDERING,
    TUTORIAL_CORRECTNESS,
    TUTORIAL_COMPLETENESS,
]


# ── CREATIVE — short fiction, memoir, narrative non-fiction ─────────────────

CREATIVE_PREMISE = Criterion(
    name="Premise",
    questions=[
        "Is the central situation interesting in itself — would the "
        "reader want to know how it resolves?",
        "Is there a question the piece is asking, implicitly or "
        "explicitly, that the reader can feel pulling them forward?",
    ],
)

CREATIVE_CHARACTER_AND_VOICE = Criterion(
    name="Character & voice",
    questions=[
        "Are the characters distinct from one another in speech, action, "
        "and the details they notice?",
        "Does the narrative voice match the material — is the register "
        "right for what is being told?",
        "Would the reader recognise a character from a single line of "
        "their dialogue or a single gesture?",
    ],
)

CREATIVE_SCENE_AND_SENSORY_GROUNDING = Criterion(
    name="Scene & sensory grounding",
    questions=[
        "Does the prose put the reader inside specific moments — places, "
        "objects, bodies — rather than summarising at a distance?",
        "Is exposition kept to what the scene needs, or does it stack "
        "up between the moments that carry the story?",
        "When the piece tells the reader something, does it also show "
        "the reader enough to feel it?",
    ],
)

CREATIVE_TENSION = Criterion(
    name="Tension",
    questions=[
        "Does each scene have stakes — something the character wants, "
        "risks, or stands to lose?",
        "Does the piece sustain forward momentum, or are there stretches "
        "where nothing is at risk and nothing changes?",
    ],
)

CREATIVE_PROSE_CRAFT = Criterion(
    name="Prose craft",
    questions=[
        "Does the language carry rhythm and image, or does it default to "
        "the first phrasing that came to hand?",
        "Is the prose economical — are there sentences, modifiers, or "
        "whole paragraphs that could be cut without loss?",
        "Where the piece reaches for an image or metaphor, is it fresh, "
        "or one the reader has met many times before?",
    ],
)

CREATIVE: list[Criterion] = [
    CREATIVE_PREMISE,
    CREATIVE_CHARACTER_AND_VOICE,
    CREATIVE_SCENE_AND_SENSORY_GROUNDING,
    CREATIVE_TENSION,
    CREATIVE_PROSE_CRAFT,
]


# ── Criteria selected by reviewer profiles ──────────────────────────────────
#
# These are in the registry but in none of the six document-type sets. A
# reviewer profile selects them; nothing selects them by document type, so no
# existing run changes because they exist. Each one was written because no
# shipped criterion asks its question — the alternatives (reuse, merge) were
# checked question by question and rejected.
#
# ``facets`` is stated explicitly on every one of them, and ``gists`` is
# omitted throughout: it gates nothing (see the note beside ``Facet``), so
# declaring it would imply a projection rule that does not exist. A criterion
# that needs the document in reading order carries the walk instruction as its
# first question instead.

EVIDENCE_SUPPORT = Criterion(
    name="Evidence support",
    questions=[
        "For each claim, which sentence, number, figure reference or table "
        "reference in the text is the evidence for it? Where there is none, "
        "say so.",
        "Where does the claim's strength exceed its evidence: a causal verb "
        "(causes, drives, leads to, explains) resting on an association; "
        '"significant", "robust", "strong", "clear" or "consistent" without '
        "the number or comparison that would make it so; a superlative "
        "(first, best, only, novel) with nothing it is compared against; a "
        "mechanism named where only its outcome was measured; a statistically "
        "detectable effect described as if it were large or important; "
        '"demonstrates", "establishes" or "confirms" where the evidence '
        'supports "suggests" or "is consistent with"?',
        'Where is evidence asserted but not present in the text: "data not '
        'shown", "we confirmed", "as expected", a figure cited for something '
        "it does not show?",
        "Where the same overreach recurs in more than one place, report the "
        "strongest instance once and name the other locations in the issue "
        "text. Do not report claims already hedged to their evidence.",
    ],
    facets=["claims"],
)

SCOPE_OF_CLAIMS = Criterion(
    name="Scope of claims",
    questions=[
        "Where does the text generalise beyond what was studied: from one "
        "system, dataset, condition, setting or population to a general "
        "statement; from a specific measurement to the broader thing it "
        "stands for; from an example to a rule?",
        "Do the opening and any summary state the same scope as the results, "
        "or a wider one? Name each word in them that widens scope beyond what "
        "the results support.",
        "Where does a limitation stated in one place fail to constrain a "
        "claim made elsewhere?",
    ],
    facets=["claims"],
)

INTERNAL_CONSISTENCY = Criterion(
    name="Internal consistency",
    questions=[
        "Does every number that appears more than once agree across all its "
        "mentions (counts, sample sizes, percentages, effect sizes, "
        "thresholds, units, versions, dates), in the body, captions, tables "
        "and any summary? For each disagreement, give both values and both "
        "locations in the issue text — one quote cannot carry two places.",
        "Does every percentage have a denominator, every measurement a unit, "
        "every comparison a reference point, and every statistical statement "
        "the test and quantity it rests on, stated somewhere in the text?",
        "Where does one passage contradict or quietly revise another: a "
        "procedure described one way and used another, an aim stated and a "
        "different aim answered, a result summarised differently in two "
        "places? Name both passages and say which the surrounding text "
        "supports.",
    ],
    facets=["claims"],
)

REPRODUCIBILITY = Criterion(
    name="Reproducibility",
    questions=[
        _WALK_THE_DOCUMENT,
        "Could another group repeat this from the text alone? Walk the "
        "procedure as described and name each point where you would have to "
        "guess: a parameter, threshold, cut-off, version, quantity, setting "
        "or rule that is used but not stated; a step whose input is not the "
        "output of any earlier step; a resource (data, code, material, tool) "
        "referred to but not identified precisely enough to obtain.",
        "For each gap, would guessing wrong plausibly change the result? "
        "Report those in full; list the rest in one line.",
        "Where does the text describe what was done in words that allow more "
        'than one procedure: "normalised", "filtered", "cleaned", "standard", '
        '"as previously described" with no pointer to where?',
    ],
    facets=[],
)

MOTIVATED_CHOICES = Criterion(
    name="Motivated choices",
    questions=[
        _WALK_THE_DOCUMENT,
        "List each choice where a different reasonable choice was available: "
        "a threshold, a method or model, an inclusion or exclusion rule, a "
        "comparison set, a metric, a stopping point.",
        "For each, is a reason given? Where none is, would a different "
        "reasonable choice plausibly change the conclusion? Report those; "
        "ignore choices that are conventional and inconsequential.",
        "Where a reason is given, does it justify this choice over the "
        "alternatives, or only justify making some choice? Name the nearest "
        "alternative it does not rule out.",
    ],
    facets=[],
)

EVALUATION_ADEQUACY = Criterion(
    name="Evaluation adequacy",
    questions=[
        "For each evaluation, what is it used to show, and can it show that? "
        "Describe the case in which the evaluation would pass even though the "
        "claim is false.",
        "Is the quantity reported the one the claim is about, or a proxy for "
        "it? Where a proxy, is the link stated and defended?",
        "Where the same data serves to both find and confirm a result, is "
        "there an independent check?",
        "Where a statistical test or model is used, does it match the data as "
        "described (independence, pairing, distribution, the number of "
        "comparisons made), and is the size of the effect given and "
        "interpreted rather than only whether it is significant?",
        "Where a sample size or number of replicates is stated, is it "
        "justified? Where it is not, does the strength of the claim depend "
        "on it?",
    ],
    facets=["claims"],
)

PROMISE_AND_DELIVERY = Criterion(
    name="Promise and delivery",
    questions=[
        _WALK_THE_DOCUMENT,
        "List each promise the opening makes — the stated aims, the gap the "
        "work says it fills, the contributions, or this kind of document's "
        "equivalent. For each, name the passage in the body that delivers it, "
        "or say that none does.",
        "Which substantial results or points in the body are not promised by "
        "the opening? Should the opening promise them, or should they go?",
        "Read the closing against the body: which sentences claim more than "
        "the body showed, and which undersell something the body did show?",
        "Can a reader state in one sentence what this document adds beyond "
        "what they already expected? If not, which sentence in the opening "
        "was supposed to do that job, and what does it say instead?",
    ],
    facets=["claims"],
)

CONTRIBUTION = Criterion(
    name="Contribution",
    questions=[
        "Would a reader outside this project care about this document, and "
        "can they say why from the text alone? Name the sentence that is "
        "supposed to tell them.",
        "What does this add that the reader did not already have — a result, "
        "a method, a way of seeing the problem, a decision they can now make? "
        "Where the text claims to add something, is the addition stated "
        "concretely or only asserted?",
        "Where does the document assume the reader already agrees that the "
        "subject matters, rather than establishing it?",
    ],
    facets=["claims", "citations"],
)

READER_PATH = Criterion(
    name="Reader path",
    questions=[
        _WALK_THE_DOCUMENT,
        "Read as a competent reader from an adjacent field who has not seen "
        "this work. Mark every point where you would stop and why: a term "
        "used before it is defined or in a sense that has shifted since; a "
        "step or inference you cannot follow from what came before; a figure, "
        "table or result cited before you know what to look for in it; a "
        "passage whose purpose in the document you cannot state; a run of "
        "detail whose bearing on the argument you cannot see.",
        "At each stop, what would you need there: a definition, a sentence of "
        "motivation, a forward reference, a reordering?",
        "Report stops in document order.",
    ],
    facets=[],
)

ASSUMED_KNOWLEDGE = Criterion(
    name="Assumed knowledge",
    questions=[
        _WALK_THE_DOCUMENT,
        "Which named methods, datasets, conventions, notations or prior "
        "results does the text use without a one-line gloss, that a reader "
        "from an adjacent field would not know?",
        "Which steps in the reasoning does the text treat as obvious that "
        "such a reader would need spelled out?",
        "Which abbreviations or symbols are used before they are expanded or defined?",
    ],
    facets=[],
)

STRONGEST_OBJECTION = Criterion(
    name="Strongest objection",
    questions=[
        "Assume the central claim is wrong. What is the strongest argument a "
        "competent hostile expert would make: a confound or artefact the "
        "design cannot exclude; a way the result could arise without the "
        "claimed mechanism; a prior result the text contradicts or ignores; a "
        "scope the data cannot carry; a step where the same data both finds "
        "and confirms a result?",
        "For each objection: does the text already answer it? If yes, drop "
        "it. If partly, say what the full answer would require — an analysis, "
        "a comparison, a sentence, a narrower claim.",
        "Rank the surviving objections by how much of the contribution each "
        "would remove if it held. Report the top few, in that order.",
        "For each, name what in the draft would defeat it, if anything does.",
    ],
    facets=["claims"],
)

MISSING_COMPARISON = Criterion(
    name="Missing comparison",
    questions=[
        "What comparison, control, baseline or alternative would a hostile "
        "reviewer say is absent: the simpler approach that might do as well; "
        "the condition that isolates the claimed factor; the independent data "
        "that would show the result is not particular to this data; the prior "
        "method the work should have been set against?",
        "For each, would its absence change how the central claim reads? "
        "Report those; drop the rest.",
        "Where a comparison is present, is it fair: same data, same "
        "conditions, same effort at tuning, same metric?",
        "For each, say what including it would show or rule out.",
    ],
    facets=["claims", "citations"],
)


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

    Returns a fresh list each call, for the reason ``probe_bank_for``
    already gives on the checklist side: handing back the module-level set
    would let a caller that appends its own criterion mutate the package
    constant for the life of the process, silently changing every later
    run. A shallow copy is enough because :class:`Criterion` is frozen —
    the elements are still the registry constants, so identity checks
    against them hold.

    The mechanism is general, the content is data — add new sets by
    extending ``_SETS``.
    """
    return list(_SETS.get(document_type, GENERAL))
