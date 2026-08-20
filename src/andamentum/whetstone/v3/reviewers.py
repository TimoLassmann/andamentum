"""Reviewer profiles — a named bundle of what the caller could type by hand.

A profile is the values the review and checklist pipelines already accept
(criteria, probes, editor criteria, tracked-change author) plus one thing no
argument can express: a *brief* that fixes the reviewer's stance, the bar a
finding must clear, and what the reviewer will not report. ``claims`` and the
equivalent inline arguments are the same mechanism, one named and one typed
out. Anything a profile does stays doable inline, and an inline value replaces
the profile's field rather than merging with it — which is what lets someone
start from ``methods``, change one thing, and see exactly what they changed.

**Where the content lives.** A profile *selects*; it does not carry questions.

  • Criteria — :mod:`andamentum.whetstone.v3.criteria`. Every shipped
    ``Criterion`` is a module-level constant there and the profiles below hold
    references to those constants, so each question has exactly one wording.
    The registry is deliberately larger than the union of the six
    document-type sets: twelve criteria in it exist for these profiles and for
    nothing else.
  • Probes — :mod:`andamentum.whetstone.v3.checklist.bank`. A profile holds
    probe *ids* rather than ``Probe`` objects, because the same id names a
    different probe on different document types; ``bank.probe_for`` resolves
    an id once the type is known.
  • Editor criteria — :data:`andamentum.whetstone.v3.editor
    .DEFAULT_EDITOR_CRITERIA` is the default the profiles fall through to.
    They are free text, not a closed set.

**The default reviewer is a profile like any other.** ``general`` sets every
selectable field to ``None``, meaning "the document type's own", so no consumer
needs a "no profile" branch and a ``general`` run is byte-identical to the run
the same arguments produced before profiles existed.

**An empty tuple is not the same as ``None``.** ``criteria=()`` and
``probes=()`` mean "skip that pass" and are used only by ``editor``. Note that
whetstone review treats an empty criterion set as "run everything except the
criterion calls" rather than as a caller error (which is what
``andamentum.llm_judge`` does with the same input): claim extraction, synthesis
and self-critique still run, so an empty criteria tuple is a cheap run and not
a free one. ``editor_criteria=()`` is refused at construction, because
resolution coerces a falsy editor criteria list back to the default and the
tuple would therefore say something it cannot deliver.

**What is thin, recorded rather than hidden.** A profile that adds only a name
over questions the type sets already ask is worse than no profile, so the two
that come closest are named here.

``adversary`` is the one whose value rests most on its bar and least on new
questions. ``ACADEMIC_SIGNIFICANCE`` already asks "are there obvious missing
comparisons or related work the document should address?", and
``EXTERNAL_COMMS_ARGUMENT`` already asks "are there obvious counterpoints the
piece should address but ignores?". What ``MISSING_COMPARISON`` and
``STRONGEST_OBJECTION`` add over those two lines is enumeration of the kinds, a
fairness check, the "does the draft already answer it" filter, and ranking by
how much of the contribution each objection would remove. Those four mechanisms
are the whole claim to being a different reviewer — and one of the two criteria
is a bar alone. If the bar turns out not to change what a model reports, the
honest move is to fold these two criteria into ``claims`` and drop the profile,
not to ship a renamed ``ACADEMIC_SIGNIFICANCE``. That is what the planted-defect
fixture exists to decide, and ``adversary`` is its primary subject.

``editor`` is a profile in the weakest sense: no criteria, no probes, and a
brief plus a list of editor criteria. It earns its place as a name for a
combination people will otherwise mistype, not as a new capability.

The design predicted a third thin case at ``methods`` against
``ACADEMIC_CORRECTNESS``. That prediction was wrong: ``ACADEMIC_CORRECTNESS``
asks about numerical consistency and asks nothing about whether the work could
be repeated from the text.

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .._document_type import DOCUMENT_TYPES
from ..renderers.docx import DEFAULT_AI_AUTHOR
from ..schemas import ConfigurationLine, ReviewerHeader
from .checklist.bank import probe_bank_for, probe_for
from .checklist.probes import Probe
from .criteria import (
    ASSUMED_KNOWLEDGE,
    CONTRIBUTION,
    EVALUATION_ADEQUACY,
    EVIDENCE_SUPPORT,
    INTERNAL_CONSISTENCY,
    MISSING_COMPARISON,
    MOTIVATED_CHOICES,
    PROMISE_AND_DELIVERY,
    READER_PATH,
    REPRODUCIBILITY,
    SCOPE_OF_CLAIMS,
    STRONGEST_OBJECTION,
    Criterion,
    criterion_set_for,
)
from .editor import DEFAULT_EDITOR_CRITERIA

ALL_DOCUMENT_TYPES: frozenset[str] = frozenset(DOCUMENT_TYPES)
"""The six document types a reviewer can be declared to fit."""

DEFAULT_PROFILE_NAME = "default"
"""The profile that runs when no reviewer is named."""

# Appended to a profile's identity to make the tracked-change author. The
# default author itself is imported, never re-typed — four distinct default
# author strings already exist in this package and only one of them is
# reachable, so a fifth literal would be a real hazard rather than a tidiness
# complaint. A profile supplies only the identity half: it can change who the
# reviewer is, never whether the edits are marked as machine-written.
_AI_AUTHOR_SUFFIX = "(andamentum-whetstone, AI)"

# Where a resolved value came from, written for a human to read. The command
# line prints these verbatim beside the value, so they name the *source* rather
# than the argument that carried it — the same profile resolves identically
# whether it was reached from a flag or from a Python call.
_SOURCE_INLINE = "supplied inline"
_SOURCE_GUIDELINES = "supplied guidelines"
_SOURCE_EDITOR_DEFAULT = "editor default"
_SOURCE_UNDETERMINED = "not determined"

# The document type before a classifier has run. It is a real value on the way
# in — every entry point accepts it — and it is the one value that cannot
# select anything, so the two document-type fallbacks below report themselves
# as undetermined rather than guessing a set the caller never asked for.
UNRESOLVED_DOCUMENT_TYPE = "auto"


@dataclass(frozen=True, kw_only=True)
class ReviewerProfile:
    """A named reviewer: what it asks, from what stance, on what documents.

    Keyword-only because five of the fields are strings a caller could
    transpose — the project's confusable-parameter rule applies to
    constructors, and a swapped ``identity`` and ``description`` would be
    invisible to a type checker and visible only in the finished report.

    Frozen because the profiles below are package data that several pipelines
    read concurrently; a profile that could be edited in place would make
    "which reviewer produced this" unanswerable after the fact.
    """

    name: str
    """The value given to ``--reviewer`` and the suffix on output filenames.

    Lowercase, one word, no punctuation, so it works as an argument value and
    as a filename component everywhere.
    """

    identity: str
    """The reviewer as it appears to the reader: report header, and — with the
    machine-authorship marker appended — the Word tracked-change author. Short,
    because Word shows it in the margin."""

    description: str
    """One line for ``--list-reviewers``. The user picks from this without
    reading any code, so it says what the reviewer does and what it costs."""

    brief: str
    """Stance, bar and exclusions, three sentences at most.

    Read by every model call the profile makes and printed at the top of the
    report, so a reader knows the stance the findings came from. The only field
    that changes how a question is *read*, and the only one that costs a prompt
    change. Written as operations (a finding is X; do not report Y) rather than
    as a persona: a persona makes a model longer and more generic, a bar makes
    it shorter and sharper.
    """

    criteria: tuple[Criterion, ...] | None
    """The review pipeline's criterion set.

    References to registry constants, never copies. ``None`` means the document
    type's own set. An empty tuple skips the criterion pass.
    """

    probes: tuple[str, ...] | None
    """The checklist pipeline's probe ids, in the order the coverage table
    should show them. ``None`` means the document type's own bank. An empty
    tuple skips the checklist pass."""

    editor_criteria: tuple[str, ...] | None
    """The editor's criteria. ``None`` means the editor's own default list.

    An empty tuple is refused: resolution coerces a falsy editor criteria list
    back to the default, so ``()`` would promise a behaviour it cannot
    deliver. A profile whose findings no rewrite can answer leaves
    ``editor_criteria=None`` and lets the editor decline them one by one, which
    it already does and reports as such.
    """

    fits: frozenset[str] = ALL_DOCUMENT_TYPES
    """The document types this reviewer makes sense on. Read only by
    :func:`fits`. A miss warns and proceeds; it never refuses."""

    def __post_init__(self) -> None:
        if self.editor_criteria == ():
            raise ValueError(
                f"ReviewerProfile {self.name!r}: editor_criteria=() is not a "
                "way to run the editor without criteria — resolution replaces "
                "a falsy editor criteria list with the default, and the editor "
                "prompt substitutes a placeholder for an empty one, so the "
                "editor would run either way. Pass None for the editor's "
                "default list, or a non-empty tuple."
            )


@dataclass(frozen=True, kw_only=True)
class ResolvedReviewer:
    """What one reviewer, one document type and one set of inline values mean.

    The output of :func:`resolve` and the only thing the pipelines read: by the
    time this exists every precedence question has been answered, so no node
    and no renderer has to ask again where a value came from.

    Keyword-only for the same reason :class:`ReviewerProfile` is: ``author``,
    ``brief`` and ``fit_warning`` are three strings in a row, and a transposed
    pair would put the reviewer's stance in the Word margin and the author name
    into every prompt — invisible to a type checker, and visible only in the
    finished document.
    """

    criteria: list[Criterion] | None
    """The active criterion set, or ``None`` when the caller supplied
    ``guidelines_text`` and the criteria are still to be extracted from it by
    the pipeline (the one thing resolution cannot do, because it needs a model
    call). An empty list means the criterion pass is skipped."""

    probes: list[Probe]
    """The active probe bank, in coverage-table order. Empty means the
    checklist pass has nothing to run."""

    editor_criteria: list[str]
    """The editor's criteria. Always concrete: the editor's default is
    knowable here, so resolution does not hand the question back."""

    author: str
    """The tracked-change author, marker included."""

    brief: str
    """The reviewer's brief, empty for the default reviewer. Rendered into
    each model call's prompt and into the report header."""

    configuration: list[ConfigurationLine]
    """Every resolved value beside where it came from, in the order the command
    line prints them: criteria, probes, editor criteria, author.

    Populated for **every** profile including the default one, because the
    question "what is this run actually going to do" is asked before the run
    and the answer is the same whether or not the report will repeat it. The
    header below carries the same list, and is suppressed for the default
    reviewer.
    """

    header: ReviewerHeader | None
    """The block the renderers print, or ``None`` for the default reviewer —
    for which there is nothing to say that the report does not already say."""

    fit_warning: str
    """Empty when the reviewer fits the document type. Otherwise the one line
    that names the pair and the way out; it is also carried on ``header``."""


# ── The default profiles ────────────────────────────────────────────────────
#
# Seven. The specialists earn their place by asking questions the document-type
# sets do not ask, from a stated stance, with a stated bar: two reviewers given
# disjoint subsets of the same questions would be one reviewer under two names,
# while two reviewers reading the same passage from different stances produce
# different findings and agree only where the agreement is worth something.
#
# The bar is what keeps six reviewers from producing a longer to-do list.
#
# A few overlaps are deliberate: `missing_to_act` sits in both `methods` and
# `reader`, and `unmotivated_choice` in both `methods` and `adversary`. Same
# probe, different stance, differently phrased finding — and a defect both of
# them raise independently is the one to fix first.

DEFAULT_PROFILE = ReviewerProfile(
    name=DEFAULT_PROFILE_NAME,
    # Never rendered: the header block is suppressed for this profile and the
    # tracked-change author stays the package default verbatim.
    identity="Whetstone reviewer",
    description=(
        "The document type's own criteria and probes; what runs when no "
        "reviewer is named."
    ),
    brief="",
    criteria=None,
    probes=None,
    editor_criteria=None,
)

CLAIMS_PROFILE = ReviewerProfile(
    name="claims",
    identity="Claims reviewer",
    description=(
        "Checks every conclusion against the evidence stated in the text and "
        "flags where the claim outruns it."
    ),
    brief=(
        "Read each claim as a proposition and locate the evidence in the text "
        "that supports it. Report only gaps that would change what a reader "
        "concludes. Do not comment on prose, structure, or whether the work "
        "matters."
    ),
    criteria=(EVIDENCE_SUPPORT, SCOPE_OF_CLAIMS, INTERNAL_CONSISTENCY),
    probes=("claim_outruns_evidence", "reach_beyond_scope", "internal_disagreement"),
    # The one rewrite these findings license is bringing the verb down to the
    # evidence.
    editor_criteria=("hedging",),
    # Not creative: a short story makes no claims to check against evidence,
    # and every one of these criteria would still return its quota.
    fits=frozenset(
        {"academic", "external_communication", "essay", "tutorial", "general"}
    ),
)

METHODS_PROFILE = ReviewerProfile(
    name="methods",
    identity="Methods reviewer",
    description=(
        "Asks whether another group could repeat the work from the text alone "
        "and whether each choice is motivated."
    ),
    brief=(
        "Read as someone about to repeat this with only the text in hand. "
        "Report only what would block the repetition or plausibly change the "
        "result. Do not comment on significance, framing, or prose."
    ),
    # Two of the three read the whole document in order via read_section, which
    # is what makes them a methods review rather than a sweep over gists — and
    # what makes them fail rather than degrade on a document of more than
    # roughly a hundred sections.
    criteria=(REPRODUCIBILITY, MOTIVATED_CHOICES, EVALUATION_ADEQUACY),
    # No open slot. It is the bank's catch-all, and its instruction names the
    # other eight core probes and says not to report those — inside a narrowed
    # bank that tells it to suppress exactly the defect classes this profile
    # dropped.
    probes=("unmotivated_choice", "missing_to_act", "order_dependency"),
    editor_criteria=None,
    fits=frozenset({"academic", "tutorial", "general"}),
)

FRAMING_PROFILE = ReviewerProfile(
    name="framing",
    identity="Framing reviewer",
    description=(
        "Checks that what the opening promises is what the body delivers and "
        "the closing does not inflate, and that a reader could say what the "
        "document adds."
    ),
    brief=(
        "Read the opening as promises and the body as their delivery. Report "
        "only where the two differ, or where a reader could not say what the "
        "document adds. Do not report missing detail, prose, or evidence gaps "
        "inside a single claim."
    ),
    criteria=(PROMISE_AND_DELIVERY, CONTRIBUTION),
    probes=("unanswered_aim", "inert_passage"),
    editor_criteria=None,
)

READER_PROFILE = ReviewerProfile(
    name="reader",
    identity="Reader advocate",
    description=(
        "Reads once as a capable outsider and reports every place they stop: "
        "an undefined or drifting term, a jump, a passage with no visible "
        "purpose."
    ),
    brief=(
        "Read once, in order, as a competent reader from an adjacent field "
        "who has not seen this work. Report where you stop and what you "
        "needed there. Do not judge whether the work is correct or important."
    ),
    # Two calls rather than one: a single pass over a whole document is thin
    # coverage, and the stops and the assumed-knowledge check are different
    # reads.
    criteria=(READER_PATH, ASSUMED_KNOWLEDGE),
    probes=("term_drift", "undefined_term", "missing_to_act"),
    editor_criteria=("clarity",),
)

ADVERSARY_PROFILE = ReviewerProfile(
    name="adversary",
    identity="Devil's advocate",
    description=(
        "Builds the strongest case a hostile expert would make and reports "
        "only the objections the draft does not already answer."
    ),
    brief=(
        "Assume the central claim is wrong and look for the argument that "
        "shows it. Report only objections the draft does not already answer, "
        "and do not soften. Do not comment on prose."
    ),
    criteria=(STRONGEST_OBJECTION, MISSING_COMPARISON),
    # No `stated_not_shown`: it is the show-don't-tell craft rule, written for
    # essays and fiction, and rewording it would change checklist findings on
    # every existing essay and creative run.
    probes=("alternative_explanation", "unmotivated_choice"),
    editor_criteria=None,
)

EDITOR_PROFILE = ReviewerProfile(
    name="editor",
    identity="Line editor",
    description=(
        "Prose only. Tracked changes for clarity, concision, grammar, "
        "precision and hedging; no commentary on the argument. In a list it "
        "is one pass after the last reviewer, over all of their findings. On "
        "its own it runs no review criteria and no checks, but still costs "
        "claim extraction, synthesis and self-critique."
    ),
    # Two sentences, not five: everything else the design gave this profile is
    # already in the editor's own prompt, and a brief that repeats the scaffold
    # is two places to edit. The "read twice" bar is the one part the scaffold
    # does not have — it names constructions, never that test.
    brief=(
        "Do not evaluate the argument. Rewrite sentences a careful reader "
        "would have to read twice."
    ),
    criteria=(),
    probes=(),
    editor_criteria=("clarity", "concision", "grammar", "precision", "hedging"),
)


PROFILES: Mapping[str, ReviewerProfile] = MappingProxyType(
    {
        # Display order for --list-reviewers, and the order the recommended
        # sequence walks: the free one first, then the three that change what
        # goes out, then the reader, then the editor last — prose that will
        # still change is wasted effort to edit.
        p.name: p
        for p in (
            DEFAULT_PROFILE,
            CLAIMS_PROFILE,
            METHODS_PROFILE,
            FRAMING_PROFILE,
            READER_PROFILE,
            ADVERSARY_PROFILE,
            EDITOR_PROFILE,
        )
    }
)
"""Every profile this package ships, keyed by name, in display order.

Read-only: a profile added or replaced at runtime would make two runs of the
same name incomparable, which is most of what a named reviewer is for.
"""


def fits(*, profile: ReviewerProfile, document_type: str) -> bool:
    """Does this reviewer make sense on this kind of document?

    The one predicate. Everything that needs the answer — the resolver below,
    the fit marker in ``--list-reviewers``, the warning in the report header —
    asks here, so the compatibility matrix has exactly one definition.

    A miss is not a refusal. Document type is usually classified rather than
    stated, the user wrote the draft and asked for this reviewer, and the
    failure mode is a page of findings about something the document was never
    doing — not damage. Substituting a different reviewer would be worse:
    nobody asked for the substitute, and it breaks comparability between runs.
    """
    return document_type in profile.fits


def may_misfit(*, profile: ReviewerProfile) -> bool:
    """Could :func:`fits` ever say no about this reviewer?

    Asked before spending a classifier call. Document type is resolved lazily —
    a run that supplies its own criteria never needs one — so a reviewer that
    fits all six types must not be the reason a run classifies: the answer is
    known without the question. Only the two narrowed profiles say yes.
    """
    return profile.fits != ALL_DOCUMENT_TYPES


def author_for(*, profile: ReviewerProfile, patch_author: str | None = None) -> str:
    """The tracked-change author: who reviewed, and that a machine wrote it.

    One definition, called by :func:`resolve` and by the command line, because
    a report header naming one author while the tracked changes carry another
    is the kind of disagreement nobody checks for.

    A profile supplies only the identity half; the machine-authorship marker is
    appended here and a profile cannot remove it. The one path that writes an
    unmarked name is an explicit ``patch_author``, which the command line
    already gates behind its own confirmation flag.
    """
    if patch_author is not None:
        return patch_author
    if profile.name == DEFAULT_PROFILE_NAME:
        return DEFAULT_AI_AUTHOR
    return f"{profile.identity} {_AI_AUTHOR_SUFFIX}"


def resolve(
    *,
    profile: ReviewerProfile,
    document_type: str,
    criteria: list[Criterion] | None = None,
    guidelines_text: str | None = None,
    probes: list[Probe] | None = None,
    editor_criteria: list[str] | None = None,
    patch_author: str | None = None,
) -> ResolvedReviewer:
    """Turn a reviewer, a document type and any inline values into one config.

    The single place precedence lives. Every field falls through the same three
    steps — the caller's own value, then the profile's, then the document
    type's default — and each inline value **replaces** its field entirely
    rather than merging into it. Replacement is what makes the effective
    configuration statable in one line per field, which is what
    ``--list-reviewers`` and the report header print.

    ::

        criteria         inline  >  guidelines text  >  profile  >  document type
        probes           inline  >  profile  >  document type
        editor criteria  inline  >  profile  >  the editor's default
        author           inline  >  profile identity + marker  >  package default

    Called **after** the document type is resolved, never before: the type
    decides which variant of a probe id a profile gets, and under automatic
    classification the type is only known once the pipeline has classified the
    document. Resolving earlier would mean classifying twice.

    ``criteria=None`` on the result is the one deferred answer — it means the
    caller passed ``guidelines_text`` and the criteria come from decomposing it,
    which needs a model call and therefore belongs to the pipeline. Every other
    field is concrete.

    An unknown probe id raises ``ValueError`` from ``bank.probe_for`` rather
    than being dropped, because a silently missing row in a coverage table is
    indistinguishable from a defect class that came back clean.

    ``document_type="auto"`` means the type has not been resolved yet, which
    is a real state: a review run that was given its own criteria never needs
    a classifier, and spending one so this function could fill in a probe bank
    the run will not use would be a model call bought for a display line.
    Every value that needs the type — the criterion set, the probe bank, and a
    profile's probe **ids**, which name different probes on different types —
    then yields nothing and says ``not determined`` rather than selecting the
    general set behind the caller's back. Each
    pipeline resolves the type before the value **it** consumes can fall
    through: review classifies whenever the criterion set would come from the
    type, checklist whenever the bank would.
    """
    lines: list[ConfigurationLine] = []
    profile_source = f"reviewer {profile.name}"
    type_known = document_type != UNRESOLVED_DOCUMENT_TYPE
    type_source = (
        f"document type {document_type}" if type_known else _SOURCE_UNDETERMINED
    )

    # ── criteria ────────────────────────────────────────────────────────────
    resolved_criteria: list[Criterion] | None
    if criteria is not None:
        resolved_criteria = list(criteria)
        criteria_source = _SOURCE_INLINE
    elif guidelines_text is not None:
        resolved_criteria = None
        criteria_source = _SOURCE_GUIDELINES
    elif profile.criteria is not None:
        resolved_criteria = list(profile.criteria)
        criteria_source = profile_source
    else:
        resolved_criteria = criterion_set_for(document_type) if type_known else []
        criteria_source = type_source
    lines.append(
        ConfigurationLine(
            label="criteria",
            value=(
                "extracted from the supplied guidelines"
                if resolved_criteria is None
                else _join(c.name for c in resolved_criteria)
            ),
            source=criteria_source,
        )
    )

    # ── probes ──────────────────────────────────────────────────────────────
    resolved_probes: list[Probe]
    if probes is not None:
        resolved_probes = list(probes)
        probes_source = _SOURCE_INLINE
    elif profile.probes is not None and type_known:
        resolved_probes = [
            probe_for(probe_id, document_type=document_type)
            for probe_id in profile.probes
        ]
        probes_source = profile_source
    elif profile.probes is not None:
        # A profile holds probe *ids*, and the same id names a differently
        # worded probe on different document types — so an id cannot be
        # resolved before the type is. Resolving it anyway would take the
        # general bank through ``probe_bank_for``'s unknown-type fallback and
        # log a warning per id about a bank the run is not going to use; every
        # review run under automatic classification would carry that noise.
        # The value nobody has determined is reported as undetermined, exactly
        # as the criterion set above is.
        resolved_probes = []
        probes_source = _SOURCE_UNDETERMINED
    else:
        resolved_probes = probe_bank_for(document_type) if type_known else []
        probes_source = type_source
    lines.append(
        ConfigurationLine(
            label="probes",
            value=_join(p.id for p in resolved_probes),
            source=probes_source,
        )
    )

    # ── editor criteria ─────────────────────────────────────────────────────
    resolved_editor_criteria: list[str]
    if editor_criteria:
        # Falsy rather than ``is not None``: an empty inline list means "not
        # supplied", which is what ``run_review_v3`` did with it before this
        # function existed and what ``ReviewerProfile.__post_init__`` cites
        # when it refuses ``editor_criteria=()``. It is also the only reading
        # the editor can honour — an empty list reaching the prompt is
        # replaced there by a placeholder and the editor proposes edits
        # anyway, so "no editor criteria" is not a state this pipeline has.
        resolved_editor_criteria = list(editor_criteria)
        editor_source = _SOURCE_INLINE
    elif profile.editor_criteria is not None:
        resolved_editor_criteria = list(profile.editor_criteria)
        editor_source = profile_source
    else:
        resolved_editor_criteria = list(DEFAULT_EDITOR_CRITERIA)
        editor_source = _SOURCE_EDITOR_DEFAULT
    lines.append(
        ConfigurationLine(
            label="editor criteria",
            value=_join(resolved_editor_criteria),
            source=editor_source,
        )
    )

    # ── author ──────────────────────────────────────────────────────────────
    author = author_for(profile=profile, patch_author=patch_author)
    if patch_author is not None:
        author_source = _SOURCE_INLINE
    elif profile.name == DEFAULT_PROFILE_NAME:
        author_source = "package default"
    else:
        author_source = profile_source
    lines.append(ConfigurationLine(label="author", value=author, source=author_source))

    fit_warning = (
        ""
        if not type_known or fits(profile=profile, document_type=document_type)
        else (
            f"{profile.name} reviewer on a document classified as "
            f"{document_type}; if the classification is wrong, pass "
            "--document-type"
        )
    )

    header: ReviewerHeader | None = None
    if profile.name != DEFAULT_PROFILE_NAME:
        from andamentum import __version__

        header = ReviewerHeader(
            identity=profile.identity,
            description=profile.description,
            brief=profile.brief,
            fit_warning=fit_warning,
            configuration=lines,
            version=__version__,
        )

    return ResolvedReviewer(
        configuration=lines,
        criteria=resolved_criteria,
        probes=resolved_probes,
        editor_criteria=resolved_editor_criteria,
        author=author,
        brief=profile.brief,
        header=header,
        fit_warning=fit_warning,
    )


def _join(values: Iterable[str]) -> str:
    """Comma-separated, or the word "none" — never an empty configuration line.

    An empty string beside a label reads as a rendering bug; "none" reads as
    the answer, which for ``criteria`` and ``probes`` it genuinely is.
    """
    joined = ", ".join(values)
    return joined or "none"
