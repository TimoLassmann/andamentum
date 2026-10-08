"""The default probe banks — the checklist *content*, kept out of the machinery.

:mod:`.probes` fixes the response format, the quota semantics and the two
verification routes. This module fixes nothing: it is a list of
:class:`~.probes.Probe` values, and swapping it changes what gets checked
without touching a line of code. Fourteen probes ship across the banks; they
are a default, not a schema, and a caller with a house checklist should build
its own list rather than bend these.

Three rules govern what may go in a bank, and each of them is a decision that
cost something to make:

**Quota where the defect is near-universal, ``None`` where it is not.** A quota
turns the task into a forced ranking with a fixed output size, which is exactly
what makes a small model productive and two runs comparable. But a quota is
also an instruction to produce that many findings. Every real draft contains
choices made without a stated reason, so ranking them is informative and
``unmotivated_choice`` carries a quota. A draft that contradicts itself is the
exception rather than the rule, so ``internal_disagreement`` carries ``None``
and is told an empty result is a legitimate answer — asking for "the 3 worst
contradictions" in a document with none manufactures three, which is the
failure mode this whole design is most exposed to.

**Tags are optional, per probe.** ``population / setting / timescale`` is a
decomposition of overreach that happens to fit empirical work; a memo
overreaches along other axes. Where the failure modes of a defect class are
genuinely known the tag set buys precision in the coverage table, and where
they are not, an untagged probe returns quote plus issue and works everywhere.

**No probe names a section.** Nothing in the document model knows which section
is Methods, and a blog post has none. Probes describe the defect and let the
agent find it with ``read_section`` / ``search_paper``; a probe that says "in
the methods section" silently no-ops on most documents.

The defect classes are all relations *inside* the text — claim against
evidence, aim against content, statement against statement, instruction against
what the reader has been given so far. That is what lets the same nine ids
re-skin from an academic paper to a blog post: only the wording changes, and
the coverage table still lines up across document types.
"""

from __future__ import annotations

import logging

from .probes import Probe

logger = logging.getLogger("andamentum.whetstone.v3")


# ── The academic core — eight probes ────────────────────────────────────────
#
# The ninth, the open slot, is defined below the list and appended in
# ACADEMIC_PROBES.

_ACADEMIC_CORE: list[Probe] = [
    Probe(
        id="unmotivated_choice",
        title="Choices made without a stated reason",
        instruction=(
            "Find the 3 choices this document makes without stating a reason "
            "for them. A choice is a threshold, cutoff, sample size, tool, "
            "model, dataset, parameter value, inclusion or exclusion rule, or "
            "comparison baseline — anything that could defensibly have been "
            "set to something else. A choice is unmotivated when the document "
            "states what was chosen but nowhere states why that rather than "
            "an alternative; the reason may sit anywhere in the document, so "
            "search before you decide it is absent. Return the verbatim "
            "sentence or clause where the choice is stated, and say what a "
            "reader cannot tell about why it was made. Rank by how much the "
            "document's conclusions would change if the choice had gone the "
            "other way."
        ),
        shape="presence",
        quota=3,
    ),
    Probe(
        id="claim_outruns_evidence",
        title="Claims the evidence supports least well",
        instruction=(
            "Find the 3 claims this document's own reported evidence supports "
            "least well. For each claim, look for the strongest result, "
            "measurement, example or cited source the document offers for it. "
            "A claim outruns its evidence when that support is missing "
            "entirely, is weaker than the claim's wording, was obtained under "
            "conditions or on a population the claim does not restrict itself "
            "to, shows things occurring together where the claim asserts one "
            "causes the other, or rests on a single instance where the claim "
            "is general. Return the verbatim claim as the quote, name the "
            "closest supporting evidence you found or state that there is "
            "none, and describe the gap between the two. Tag which of the "
            "listed relations the gap is."
        ),
        shape="presence",
        quota=3,
        tags=(
            "no_result",
            "weaker_than_claimed",
            "different_population",
            "association_not_cause",
            "single_case",
        ),
    ),
    # Absence-shaped AND quota'd, which the quota rule above warns against.
    # Kept deliberately, on two grounds. Missing information is near-universal
    # in a real draft — the condition the rule requires for a quota — where
    # self-contradiction, the case the rule was written about, is not. And an
    # absence finding is the one kind that must clear search-term refutation
    # before it reaches a reader (.verify drops it the moment any term turns
    # up), so the manufactured-finding risk a quota carries is already met by
    # the verification route rather than left to the model's restraint.
    Probe(
        id="missing_to_act",
        title="Information needed to act, and absent",
        instruction=(
            "Find the 3 things a reader would need in order to repeat, apply "
            "or build on this work and cannot find anywhere in the document. "
            "Report only things whose absence blocks action: a step of a "
            "procedure that is referred to but never described, a value or "
            "setting that is used but never given, where the underlying "
            "material or data can be obtained, the identity or version of a "
            "tool whose behaviour differs between versions, or the rule by "
            "which something was decided, counted or classified. Search the "
            "document for each one before reporting it — this check reports "
            "genuine absence only, and anything stated in any wording counts "
            "as present. Return the missing item as a short noun phrase, plus "
            "2 to 4 search terms that would appear in the document if it were "
            "present. There is nothing to quote here; do not invent a quote."
        ),
        shape="absence",
        quota=3,
        tags=(
            "procedure",
            "parameter",
            "data_access",
            "tool_version",
            "criterion",
        ),
    ),
    Probe(
        id="internal_disagreement",
        title="Statements that cannot both be true",
        instruction=(
            "Report every pair of statements in this document that cannot "
            "both be true as written. That includes a number, count, date or "
            "unit given differently in two places; a term defined one way and "
            "used another; a result described at one size in one place and a "
            "different size elsewhere; and a statement of what was done that "
            "contradicts another statement of what was done. A difference of "
            "emphasis, framing or tone is not a disagreement — the two "
            "statements must be jointly impossible. Return the verbatim text "
            "of one of the two conflicting statements as the quote, and give "
            "the other one inside the issue so a reader can check both at "
            "once. Reporting nothing is a legitimate answer here: if the "
            "document does not contradict itself, say so by returning no "
            "findings rather than by stretching a near-miss."
        ),
        shape="presence",
        quota=None,
    ),
    Probe(
        id="reach_beyond_scope",
        title="Statements that reach past what was examined",
        instruction=(
            "Find the 3 statements that assert more than this document "
            "examined. Compare the scope of each concluding, generalising or "
            "recommending statement against the scope of the work the "
            "document describes: who or what was actually looked at, under "
            "what conditions, over what period, at what size of effect, and "
            "through what mechanism. A statement reaches beyond scope when it "
            "covers a group, a setting, a duration, a magnitude or a causal "
            "mechanism wider than the document's own material covers. A "
            "statement that is explicitly hedged to the scope actually "
            "studied does not count. Return the verbatim overreaching "
            "statement as the quote and name the narrower scope the document "
            "actually supports. Tag the dimension the reach happens along."
        ),
        shape="presence",
        quota=3,
        tags=(
            "population",
            "setting",
            "timescale",
            "magnitude",
            "mechanism",
        ),
    ),
    Probe(
        id="unanswered_aim",
        title="Aims without answers, results without aims",
        instruction=(
            "Report every mismatch between what this document says it will do "
            "and what it delivers. Two directions count. First, something the "
            "document states it will address, answer, show or cover, for "
            "which no corresponding result, argument or discussion appears "
            "anywhere later — tag that aim_unanswered. Second, a substantial "
            "result or passage that answers no aim, question or purpose the "
            "document ever states — tag that result_orphan. Search the "
            "document for the missing half before reporting either; a "
            "partial, brief or differently-worded answer counts as delivered. "
            "Return the missing half as the missing item: for aim_unanswered "
            "name the aim and the answer that never comes, and for "
            "result_orphan name the result and the aim it would have needed. "
            "Give 2 to 4 search terms that would appear if the missing half "
            "were present. Reporting nothing is a legitimate answer here."
        ),
        shape="absence",
        quota=None,
        tags=("aim_unanswered", "result_orphan"),
    ),
    Probe(
        id="term_drift",
        title="Terms that change meaning mid-document",
        instruction=(
            "Find the 3 terms this document uses in more than one sense, or "
            "swaps for a different word without saying whether a distinction "
            "is meant. A term drifts when its meaning, scope or referent "
            "changes between two uses and the document does not mark the "
            "change, or when two words are used interchangeably for one "
            "concept so a reader cannot tell whether they name the same "
            "thing. Ordinary synonym variation in ordinary prose is not "
            "drift; the term must be one the document's argument depends on. "
            "Return the verbatim sentence containing the later, drifted use "
            "as the quote, and state both senses the term carries and where "
            "the earlier one appears. Rank by how much confusion the drift "
            "causes a reader."
        ),
        shape="presence",
        quota=3,
    ),
    Probe(
        id="alternative_explanation",
        title="Preferred explanations with an unaddressed rival",
        instruction=(
            "Find the 2 places where this document settles on one explanation "
            "for something it observed without ruling out an obvious "
            "competing one. For each, the document must assert or imply that "
            "one thing accounts for another, while a different account of the "
            "same observation is consistent with everything the document "
            "reports — for example how the material examined was selected, an "
            "uncontrolled difference between the things compared, the order "
            "in which events happened, an artefact of how something was "
            "measured or reported, or chance. Return the verbatim statement "
            "of the preferred explanation as the quote, and name the "
            "competing account it leaves unaddressed. A place where the "
            "document explicitly considers a rival and gives a reason for "
            "rejecting it does not count."
        ),
        shape="presence",
        quota=2,
    ),
]


# The open slot is deliberate insurance against taxonomy blindness. Every
# other probe can only find defects someone anticipated when writing this
# bank; a document's worst problem is regularly one no list names. One
# unconstrained slot costs a single call and is the only part of the bank
# that can report something the design did not foresee. It carries a quota
# of 1 rather than None because the point is to force the question to be
# answered, not to let a model that found nothing convenient skip it.
_OPEN_SLOT_ACADEMIC = Probe(
    id="open_slot",
    title="Biggest problem the other checks would miss",
    instruction=(
        "Name the single biggest problem with this document that none of the "
        "other checks in this run would have caught. Those checks look for "
        "choices made without a stated reason, claims outrunning their "
        "evidence, information needed to act on the work and absent, "
        "statements that contradict each other, conclusions reaching past "
        "what was examined, stated aims that go unanswered, terms that drift "
        "in meaning, and preferred explanations with an unaddressed rival — "
        "so do not report one of those here. Report instead whatever a "
        "careful, well-disposed reader of this document would name as its "
        "most serious remaining weakness. Return the verbatim passage where "
        "the problem is most visible as the quote, and state the problem "
        "plainly enough that the author knows what to change."
    ),
    shape="presence",
    quota=1,
)


ACADEMIC_PROBES: list[Probe] = [*_ACADEMIC_CORE, _OPEN_SLOT_ACADEMIC]
"""The core bank, worded for research writing."""


# ── The general core — the same eight defect classes, re-skinned ────────────
#
# Same ids, same shapes, same quotas: this is a wording change, not a
# different checklist. Keeping the ids stable is what lets a coverage table
# from a blog post be read against one from a paper, and it is why the
# re-skin deliberately invents no new defect classes — a class that only
# exists for one document type belongs in that type's variants below. The
# ninth id, the open slot, is re-skinned below the list in the same way.

_GENERAL_CORE: list[Probe] = [
    Probe(
        id="unmotivated_choice",
        title="Decisions asserted without a reason",
        instruction=(
            "Find the 3 decisions this document asserts without giving a "
            "reason for them. A decision is any point where the document says "
            "something is done, used, chosen, recommended or ruled out in a "
            "particular way: a tool, an amount, a deadline, an audience, an "
            "approach, a rule about what is included or excluded. It is "
            "unmotivated when the document says what was decided but nowhere "
            "says why that rather than the obvious alternative; the reason "
            "may sit anywhere in the document, so search before you decide it "
            "is absent. Return the verbatim sentence or clause stating the "
            "decision, and say what a reader cannot tell about why it was "
            "made. Rank by how much the rest of the document depends on it."
        ),
        shape="presence",
        quota=3,
    ),
    Probe(
        id="claim_outruns_evidence",
        title="Assertions with the least support",
        instruction=(
            "Find the 3 statements this document asserts most confidently "
            "while offering the least support for them. Support is anything "
            "in the document a reader could check for themselves: a number, a "
            "worked example, a named source, a described case, a quotation, a "
            "demonstration. A statement outruns its support when there is "
            "none at all, when what is offered is weaker or narrower than the "
            "statement's wording, when the support comes from a different "
            "group or situation than the statement covers, when things merely "
            "occurring together are presented as one causing the other, or "
            "when one case is used to carry a general claim. Return the "
            "verbatim statement as the quote, name the closest support you "
            "found or state that there is none, and describe the gap. Tag "
            "which of the listed relations the gap is."
        ),
        shape="presence",
        quota=3,
        tags=(
            "no_result",
            "weaker_than_claimed",
            "different_population",
            "association_not_cause",
            "single_case",
        ),
    ),
    # Absence-shaped AND quota'd against the quota rule, for the reasons given
    # at the academic bank's copy of this probe: the defect is near-universal,
    # and search-term refutation in .verify is the mitigation the rule wants.
    Probe(
        id="missing_to_act",
        title="Information the reader needs to act, and cannot find",
        instruction=(
            "Find the 3 pieces of information a reader would need in order to "
            "act on this document and cannot find in it. Report only things "
            "whose absence blocks action: a step that is referred to but "
            "never described, a value, setting or amount that is used but "
            "never given, where to obtain something the document tells the "
            "reader to use, the identity or version of a thing that behaves "
            "differently between versions, or the rule by which something is "
            "to be judged or decided. Search the document for each one before "
            "reporting it — this check reports genuine absence only, and "
            "anything stated in any wording counts as present. Return the "
            "missing item as a short noun phrase, plus 2 to 4 search terms "
            "that would appear in the document if it were present. There is "
            "nothing to quote here; do not invent a quote."
        ),
        shape="absence",
        quota=3,
        tags=(
            "procedure",
            "parameter",
            "data_access",
            "tool_version",
            "criterion",
        ),
    ),
    Probe(
        id="internal_disagreement",
        title="Statements that cannot both be true",
        instruction=(
            "Report every pair of statements in this document that cannot "
            "both be true as written. That includes a number, count, date or "
            "unit given differently in two places; a term used with one "
            "meaning here and another there; a description of what happened "
            "that contradicts another description of the same thing; and "
            "advice in one place that the document elsewhere warns against. A "
            "difference of emphasis, framing or tone is not a disagreement — "
            "the two statements must be jointly impossible. Return the "
            "verbatim text of one of the two conflicting statements as the "
            "quote, and give the other one inside the issue so a reader can "
            "check both at once. Reporting nothing is a legitimate answer "
            "here: if the document does not contradict itself, return no "
            "findings rather than stretching a near-miss."
        ),
        shape="presence",
        quota=None,
    ),
    Probe(
        id="reach_beyond_scope",
        title="Statements that claim more than the document shows",
        instruction=(
            "Find the 3 statements that claim more than this document shows. "
            "Compare the scope of each conclusion, generalisation or "
            "recommendation against the scope of what the document actually "
            "covers: which people or cases it draws on, in what setting, over "
            "what period, at what size of effect, and through what mechanism. "
            "A statement reaches beyond scope when it covers a group, a "
            "setting, a duration, a magnitude or a causal mechanism wider "
            "than the document's own material. A statement explicitly hedged "
            "to what the document covers does not count. Return the verbatim "
            "overreaching statement as the quote and name the narrower scope "
            "the document actually supports. Tag the dimension of the reach."
        ),
        shape="presence",
        quota=3,
        tags=(
            "population",
            "setting",
            "timescale",
            "magnitude",
            "mechanism",
        ),
    ),
    Probe(
        id="unanswered_aim",
        title="Promises without delivery, content without purpose",
        instruction=(
            "Report every mismatch between what this document promises and "
            "what it delivers. Two directions count. First, something the "
            "document says it will explain, answer, show or cover, for which "
            "no corresponding passage appears anywhere later — tag that "
            "aim_unanswered. Second, a substantial passage that serves no "
            "purpose the document ever states — tag that result_orphan. "
            "Search the document for the missing half before reporting "
            "either; a partial, brief or differently-worded delivery counts. "
            "Return the missing half as the missing item: for aim_unanswered "
            "name the promise and the delivery that never comes, and for "
            "result_orphan name the passage and the purpose it would have "
            "needed. Give 2 to 4 search terms that would appear if the "
            "missing half were present. Reporting nothing is a legitimate "
            "answer here."
        ),
        shape="absence",
        quota=None,
        tags=("aim_unanswered", "result_orphan"),
    ),
    Probe(
        id="term_drift",
        title="Words that change meaning mid-document",
        instruction=(
            "Find the 3 words or phrases this document uses in more than one "
            "sense, or swaps for a different word without saying whether a "
            "distinction is meant. A term drifts when its meaning, scope or "
            "referent changes between two uses and the document does not mark "
            "the change, or when two words are used interchangeably for one "
            "thing so a reader cannot tell whether they mean the same thing. "
            "Ordinary synonym variation in ordinary prose is not drift; the "
            "term must be one the document's point depends on. Return the "
            "verbatim sentence containing the later, drifted use as the "
            "quote, and state both senses the term carries and where the "
            "earlier one appears. Rank by how much confusion it causes."
        ),
        shape="presence",
        quota=3,
    ),
    Probe(
        id="alternative_explanation",
        title="Explanations with an unaddressed rival",
        instruction=(
            "Find the 2 places where this document settles on one explanation "
            "for something without ruling out an obvious competing one. For "
            "each, the document must assert or imply that one thing accounts "
            "for another, while a different account of the same thing is "
            "consistent with everything the document says — for example that "
            "the cases it draws on were unusual, that something else changed "
            "at the same time, that the order of events explains it, that the "
            "effect is an artefact of how it was noticed or reported, or that "
            "it was chance. Return the verbatim statement of the preferred "
            "explanation as the quote, and name the competing account it "
            "leaves unaddressed. A place where the document explicitly "
            "considers a rival and gives a reason for rejecting it does not "
            "count."
        ),
        shape="presence",
        quota=2,
    ),
]


# Same insurance slot, general wording. See the comment above the academic one.
_OPEN_SLOT_GENERAL = Probe(
    id="open_slot",
    title="Biggest problem the other checks would miss",
    instruction=(
        "Name the single biggest problem with this document that none of the "
        "other checks in this run would have caught. Those checks look for "
        "decisions asserted without a reason, statements with little support, "
        "information the reader needs and cannot find, statements that "
        "contradict each other, claims reaching past what the document shows, "
        "promises that go undelivered, words that drift in meaning, and "
        "explanations with an unaddressed rival — so do not report one of "
        "those here. Report instead whatever a careful, well-disposed reader "
        "of this document would name as its most serious remaining weakness. "
        "Return the verbatim passage where the problem is most visible as the "
        "quote, and state the problem plainly enough that the author knows "
        "what to change."
    ),
    shape="presence",
    quota=1,
)


GENERAL_PROBES: list[Probe] = [*_GENERAL_CORE, _OPEN_SLOT_GENERAL]
"""The core bank re-skinned for non-academic prose. Same ids, same shapes."""


# ── Per-type variants ───────────────────────────────────────────────────────
#
# Five probes that only earn their call on some document types. Each is still
# a relation inside the text, not a domain checklist: an instruction against
# what the reader has been given, an assertion against its attribution, a
# conclusion against the detail that would produce it, a passage against the
# rest of the document, a term against its explanation. They sit before the
# open slot in each bank so the insurance slot stays last.

_ORDER_DEPENDENCY = Probe(
    id="order_dependency",
    title="Instructions that depend on something not yet given",
    instruction=(
        "Find the 3 instructions that depend on something the document has "
        "not given the reader by that point. An instruction depends on "
        "something when carrying it out requires a value, a file, an account, "
        "an installed component, a completed earlier action or a piece of "
        "knowledge that the document supplies only later, or never supplies "
        "at all. A reader who has followed the document from the start and in "
        "order is the test: if they would have to stop and go elsewhere, the "
        "instruction qualifies. Return the verbatim instruction as the quote, "
        "and name what the reader would not yet have and where — if anywhere "
        "— the document supplies it. Rank by how early in the document the "
        "reader would be stopped."
    ),
    shape="presence",
    quota=3,
    tags=(
        "undefined_input",
        "unstated_setup",
        "defined_later",
        "assumed_state",
    ),
)

_UNATTRIBUTED_SOURCE = Probe(
    id="unattributed_source",
    title="External facts with no checkable source",
    instruction=(
        "Find the 3 statements of external fact this document presents "
        "without a source a reader could check. A statement of external fact "
        "is one the document reports rather than demonstrates: a number or "
        "proportion, a quotation or paraphrase of someone else, a finding "
        "attributed to research or to an organisation, a forecast, or an "
        "assertion that people generally agree on something. It is "
        "unattributed when no name, publication, dataset, link or date "
        "anywhere in the document identifies where it came from. The "
        "document's own reasoning and the author's own experience are not "
        "external facts and do not count. Return the verbatim statement as "
        "the quote and say what the reader is being asked to take on trust. "
        "Tag which kind of statement it is."
    ),
    shape="presence",
    quota=3,
    tags=(
        "statistic",
        "quotation",
        "research_finding",
        "prediction",
        "consensus_claim",
    ),
)

_UNDEFINED_TERM = Probe(
    id="undefined_term",
    title="Specialist terms used but never explained",
    instruction=(
        "Report every term this document uses as if the reader already knew "
        "it and never explains. A term qualifies when it is specific to a "
        "field, a product, a project or an organisation — a name, a set of "
        "initials never written out, a piece of jargon — and the document "
        "neither defines it, glosses it in passing, nor makes its meaning "
        "plain from the sentence around it. Search the document for an "
        "explanation before reporting one: a definition anywhere, in any "
        "wording, counts as present. Return the missing explanation as the "
        "missing item — the term, plus what a reader would need to be told "
        "about it — and 2 to 4 search terms that would appear if the document "
        "did explain it. Reporting nothing is a legitimate answer here."
    ),
    shape="absence",
    quota=None,
)

_STATED_NOT_SHOWN = Probe(
    id="stated_not_shown",
    title="Conclusions asserted instead of rendered",
    instruction=(
        "Find the 3 places where this document tells the reader a conclusion "
        "about something instead of giving them what would produce it. This "
        "happens when the text asserts a quality, a feeling, a judgment or a "
        "significance — that someone is a certain way, that a moment "
        "mattered, that a situation was hard — while nothing around it "
        "renders the specifics a reader could draw that conclusion from for "
        "themselves. An assertion that IS surrounded by concrete detail "
        "carrying it does not count, however plainly it is worded. Return the "
        "verbatim asserting sentence as the quote and name the detail the "
        "reader is missing. Rank by how much the document's effect depends on "
        "the reader believing the assertion."
    ),
    shape="presence",
    quota=3,
)

_INERT_PASSAGE = Probe(
    id="inert_passage",
    title="Passages that could be cut without loss",
    instruction=(
        "Find the 2 passages that could be removed without the document "
        "losing anything. A passage is inert when it introduces no "
        "information, image, argument or change the rest of the document does "
        "not already carry: restating a point already made, elaborating "
        "something already clear, or bridging between two parts that need no "
        "bridge. Judge by what would be lost, not by how well it is written — "
        "a well-written passage that repeats an earlier one is inert, and a "
        "clumsy passage carrying the only instance of something is not. "
        "Return the first sentence of the passage, verbatim, as the quote, "
        "and say what it repeats or why nothing is lost by cutting it."
    ),
    shape="presence",
    quota=2,
)


EXTERNAL_COMMUNICATION_PROBES: list[Probe] = [
    *_GENERAL_CORE,
    _UNATTRIBUTED_SOURCE,
    _UNDEFINED_TERM,
    _OPEN_SLOT_GENERAL,
]
"""Blog posts, articles, emails, op-eds, press releases."""


ESSAY_PROBES: list[Probe] = [
    *_GENERAL_CORE,
    _STATED_NOT_SHOWN,
    _INERT_PASSAGE,
    _OPEN_SLOT_GENERAL,
]
"""Personal, narrative and opinion essays."""


TUTORIAL_PROBES: list[Probe] = [
    *_GENERAL_CORE,
    _ORDER_DEPENDENCY,
    _UNDEFINED_TERM,
    _OPEN_SLOT_GENERAL,
]
"""How-tos, technical walkthroughs, cookbooks."""


CREATIVE_PROBES: list[Probe] = [
    *_GENERAL_CORE,
    _STATED_NOT_SHOWN,
    _INERT_PASSAGE,
    _OPEN_SLOT_GENERAL,
]
"""Short fiction, memoir, narrative non-fiction."""


# ── Selection by document type ──────────────────────────────────────────────

_BANKS: dict[str, list[Probe]] = {
    "academic": ACADEMIC_PROBES,
    "external_communication": EXTERNAL_COMMUNICATION_PROBES,
    "essay": ESSAY_PROBES,
    "tutorial": TUTORIAL_PROBES,
    "creative": CREATIVE_PROBES,
    "general": GENERAL_PROBES,
}


def probe_bank_for(document_type: str) -> list[Probe]:
    """The active probe bank for a document type.

    Mirrors ``criteria.criterion_set_for``: the same six types, and an
    unknown type falls back to GENERAL_PROBES rather than to the academic
    bank. Falling back to academic would be actively harmful here in a way
    it is not for a criterion set — a quota'd probe asked about a document
    it does not fit still returns its quota, so running
    ``claim_outruns_evidence`` over a short story produces three confident
    findings about missing evidence in a document that was never making an
    empirical claim.

    The fallback is logged because it is a real loss of precision, not a
    neutral default: a document type reaching this function unrecognised
    usually means the classifier upstream emitted something the banks do
    not cover, and that is worth seeing in the run log rather than
    inferring later from bland findings.

    Returns a fresh list each call. Handing back the module-level bank would
    let a caller that appends its own probe mutate the package constant for
    the life of the process, silently changing every later run. A shallow copy
    is enough because :class:`Probe` is frozen.
    """
    bank = _BANKS.get(document_type)
    if bank is None:
        logger.warning(
            "[v3.checklist.bank] unknown document type %r — falling back to "
            "the general bank (%d probes). Known types: %s",
            document_type,
            len(GENERAL_PROBES),
            ", ".join(sorted(_BANKS)),
        )
        return list(GENERAL_PROBES)
    return list(bank)


# ── Selection by id ─────────────────────────────────────────────────────────
#
# A reviewer profile selects probes by id rather than by document type, so it
# needs a lookup the per-type banks do not provide. Nothing here changes a
# probe's text or a bank's membership: ``probe_bank_for`` and every bank list
# are untouched, so an existing checklist run resolves exactly the probes it
# always did.
#
# The five per-type variants above are private on purpose — they are not a
# bank and selecting one by hand only makes sense with an id in front of you —
# so this is the one door onto them.

_CROSS_TYPE_PROBES: tuple[Probe, ...] = (
    _ORDER_DEPENDENCY,
    _UNATTRIBUTED_SOURCE,
    _UNDEFINED_TERM,
    _STATED_NOT_SHOWN,
    _INERT_PASSAGE,
)

ALL_PROBE_IDS: frozenset[str] = frozenset(
    p.id for bank in (*_BANKS.values(), _CROSS_TYPE_PROBES) for p in bank
)
"""Every probe id this package ships, across all six banks and the variants.

Membership only. Two ids in here name *different* ``Probe`` objects depending
on the document type (``unmotivated_choice`` has an academic wording and a
general one), which is why the resolver below takes the type as well.
"""


def probe_for(probe_id: str, *, document_type: str) -> Probe:
    """The document-type-appropriate variant of one probe, by id.

    Resolution order is the type's own bank first, then the cross-type
    variants. The order matters because the same id names two different probes:
    ``unmotivated_choice`` asks about an academic paper's choices in
    ACADEMIC_PROBES and about a general document's in GENERAL_PROBES, and a
    profile that selects it should get the wording that matches the document in
    front of it. The cross-type variants exist in one form only, so they
    resolve the same way whatever the type.

    An unrecognised document type resolves through ``probe_bank_for``, which
    falls back to the general bank and logs that it did — one behaviour for
    unknown types, not two.

    Raises ``ValueError`` naming the id and the known ids when the id is not
    one this package ships. A profile that names a probe which no longer
    exists is a bug in the profile, and dropping it silently would produce a
    coverage table with a missing row and no explanation for it.
    """
    for probe in (*probe_bank_for(document_type), *_CROSS_TYPE_PROBES):
        if probe.id == probe_id:
            return probe
    raise ValueError(
        f"probe_for: unknown probe id {probe_id!r}. Known ids: "
        f"{', '.join(sorted(ALL_PROBE_IDS))}."
    )
