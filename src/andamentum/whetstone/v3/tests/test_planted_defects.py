"""Are the specialists six reviewers, or one reviewer under six names?

One draft, six planted defects, one per profile. Each specialist is run over
the whole draft and asked three questions:

1. **Did it find its own defect?** If a specialist cannot find the defect its
   criteria and its brief were written for, the questions are wrong — sharpen
   them (design section 10, step two).
2. **Did it stay out of the others' lanes?** The sharper and more fragile of
   the first two. Six reviewers that all report all six defects are one
   reviewer with six author names.
3. **Did its brief do any of that work?** This is the design's riskiest
   assumption — that a briefing changes what a model reports when the
   questions are held fixed — and questions 1 and 2 cannot answer it, because
   each specialist also carries its own criteria and separation between two
   specialists is fully explained by the questions alone. So the third test
   runs *one* reviewer twice, changing only ``brief``. If the two arms are
   indistinguishable, the ``brief`` field goes, the three prompt slots go, and
   the ``editor`` profile — a brief and a flag alias, nothing else — goes with
   them. The same shape is applied to the one criterion whose claim to being
   more than a rename rests on a bar rather than on new questions,
   ``Missing comparison`` question 2.

They are separate tests on purpose. "Found nothing" and "found everything" are
opposite failures with opposite fixes: the first is a question that is too
narrow or too abstract, the second is an exclusion the brief does not state
firmly enough. A single combined assertion would report both as one red line.

**These tests call a real model and are deselected by default** (marker
``cloud``, like every other model-calling test in this repository). Run them by
hand::

    uv run pytest src/andamentum/whetstone/v3/tests/test_planted_defects.py -m cloud -s

Run once per model class in use and record the result per profile — that is the
honest answer to which reviewer needs which model, and it changes as models
change. Expect the two profiles that rest on a judgement rather than on an
enumeration (``adversary``, and the bar inside ``methods``) to degrade first:
a smaller model answers "would guessing wrong plausibly change the result" with
generic objections that pass the quote anchor because any quote will do.

The fixture's own integrity — six anchors, each appearing exactly once, no two
overlapping — is checked by the deterministic tests at the bottom of this file,
which run on every commit. A fixture that has drifted would make every model
result above meaningless, and finding that out during a hand-run is finding it
out too late.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from dotenv import load_dotenv

from andamentum.whetstone.schemas import ReviewResult
from andamentum.whetstone.v3.criteria import MISSING_COMPARISON
from andamentum.whetstone.v3.graph import run_review_v3
from andamentum.whetstone.v3.reviewers import DEFAULT_PROFILE_NAME, PROFILES, ReviewerProfile

load_dotenv()

# The models this fixture has been run against. One entry is a run; add an id
# to run the same fixture on another model class (design section 14 asks for
# exactly that, per profile). Kept as a literal rather than read from the
# environment: which model produced a result is part of the result.
MODELS_UNDER_TEST = ("openai:gpt-5.4-nano",)

# Stated rather than classified. The classifier would spend a call to reach
# the same answer, and a fixture whose document type could drift between runs
# would make two runs incomparable for a reason that has nothing to do with
# the reviewers.
DOCUMENT_TYPE = "academic"


# ── the draft ───────────────────────────────────────────────────────────────
#
# Two pages, one defect per specialist, each planted in a sentence of its own
# so that a finding either quotes it or does not. The subject matter is
# deliberately ordinary — a tool, a corpus, a timing comparison — because the
# thing under test is whether a stance changes what a reviewer reports, not
# whether a model knows a field.

SOURCE = """\
# Hedgerow: grouping short text records on one machine

## Summary

Teams that keep large collections of short written records — support notes,
incident logs, meeting minutes — need to group them by what they are about.
The usual approach compares every record against every other one, and at a
million records that comparison is most of the cost. We describe Hedgerow, a
grouping method that runs on a single machine rather than a cluster, and we
report a run over 1.2 million customer support notes. We also set out how the
method behaves when records arrive out of order.

## Background

Grouping records by meaning is an old problem, and the method most teams reach
for compares dense representations of every pair of records. That method is
well understood and its output is easy to inspect, but its cost grows with the
square of the collection, so a team with a million records either buys a
cluster or gives up on regrouping often enough to be useful. Work on cheaper
grouping has mostly gone in two directions: reduce the number of pairs that
have to be compared, or reduce the cost of comparing one pair. Hedgerow takes
the second direction.

## The method

Hedgerow runs in three passes over the collection.

Each record is first reduced to its stem profile before comparison. The second
pass sorts records by the size of that reduction, so that records carrying the
least information are considered last and cannot pull an otherwise coherent
group apart. The third pass walks the sorted list once and assigns each record
either to an existing group or to a new one of its own.

The assignment step compares a record against the records already placed and
scores each comparison between zero and one. Pairs scoring above the
similarity threshold were merged into one group. A record that scores below it
against every existing group starts a group of its own, which is why the
number of groups is an output of the method rather than an input to it.

We implemented all three passes in a single process with no external services,
so the method can be run on a laptop by anyone holding the records.

## Evaluation

We ran Hedgerow and the standard pairwise method over the same collection of
customer support notes and measured wall-clock time and grouping quality.

Before running Hedgerow we removed exact duplicates, which took the collection
from 1.2 million records to 0.7 million. The baseline was run on the full 1.2
million.

Hedgerow finished in 41 minutes and the baseline in 190 minutes, so the new
grouping stage is the source of the gain. Both runs used the same machine and
the same process count, and neither was run more than once.

For quality we asked three people to read a sample of 200 groups from each
method and mark each group as coherent or not. Groups produced from the
shorter representation scored 0.71 against the baseline's 0.64. The speed-up
therefore demonstrates that shorter representations cause better grouping.

## Discussion

Hedgerow makes regrouping cheap enough to do weekly rather than quarterly,
which is the difference between a collection that reflects how a team talks
today and one that reflects how it talked last year.

The evaluation of the performance of the implementation of the grouping stage,
which was carried out by the team over a period of several weeks, was, given
that the collection had already been prepared in advance beforehand, a
straightforward undertaking.

The method assumes that records are short. On longer records the reduction
step throws away most of what distinguishes one from another, and we would
expect the grouping quality to fall well below what the pairwise method
reaches. We have not measured where that boundary sits.

## Conclusion

Hedgerow groups a large collection of short records on one machine in a
fraction of the time the standard method takes, and the groups it produces
read as well to a human as the ones the standard method produces. Teams
holding collections of this shape can regroup them often enough to keep them
useful.
"""


@dataclass(frozen=True, kw_only=True)
class _Defect:
    """One planted defect: whose it is, what it is, and where it sits.

    Keyword-only because ``profile``, ``defect`` and ``anchor`` are three
    strings in a row and a transposition would silently retarget the whole
    test — the confusable-parameter rule applies here for the same reason it
    applies to library code.
    """

    profile: str
    defect: str
    anchor: str

    @property
    def span(self) -> tuple[int, int]:
        """Where the anchor sits in :data:`SOURCE`, as a half-open range."""
        start = SOURCE.index(self.anchor)
        return start, start + len(self.anchor)


DEFECTS: tuple[_Defect, ...] = (
    _Defect(
        profile="claims",
        defect="a causal conclusion drawn from two numbers that only differ",
        anchor=(
            "The speed-up\ntherefore demonstrates that shorter representations "
            "cause better grouping."
        ),
    ),
    _Defect(
        profile="methods",
        defect="a threshold the method turns on, used but never stated",
        anchor=(
            "Pairs scoring above the\nsimilarity threshold were merged into one group."
        ),
    ),
    _Defect(
        profile="framing",
        defect="a promise the opening makes that no passage in the body keeps",
        anchor=(
            "We also set out how the\nmethod behaves when records arrive out of order."
        ),
    ),
    _Defect(
        profile="reader",
        defect="a term the whole method rests on, used without a gloss",
        anchor="Each record is first reduced to its stem profile before comparison.",
    ),
    _Defect(
        profile="adversary",
        defect=(
            "a timing comparison run on collections of different sizes, with the "
            "difference stated one paragraph earlier and never answered"
        ),
        anchor=(
            "Hedgerow finished in 41 minutes and the baseline in 190 minutes, so "
            "the new\ngrouping stage is the source of the gain."
        ),
    ),
    _Defect(
        profile="editor",
        defect="a sentence a careful reader has to read twice",
        anchor=(
            "The evaluation of the performance of the implementation of the "
            "grouping stage,\nwhich was carried out by the team over a period of "
            "several weeks, was, given\nthat the collection had already been "
            "prepared in advance beforehand, a\nstraightforward undertaking."
        ),
    ),
)

BY_PROFILE: dict[str, _Defect] = {defect.profile: defect for defect in DEFECTS}

# The editor is excluded from the exclusivity half. It rewrites prose wherever
# prose needs rewriting, and five of the six planted defects are sentences —
# so "the line editor did not touch another specialist's defect" would be a
# test of where the sentences happen to be, not of the reviewer. What the
# editor is held to instead is that it reports no findings at all: its
# criteria are empty, so its whole output is rewrites.
FINDING_PROFILES = tuple(d.profile for d in DEFECTS if d.profile != "editor")


# ── running one specialist ──────────────────────────────────────────────────

_CACHE: dict[tuple[str, str], ReviewResult] = {}


async def _run_profile(
    *, profile: ReviewerProfile, model: str, label: str
) -> ReviewResult:
    """One review of the fixture by one reviewer, memoised under *label*.

    The label rather than the profile is the cache key because the comparison
    tests below run the *same* named reviewer twice with one field changed,
    and two runs that must differ cannot share a cache entry. Keyword-only:
    ``model`` and ``label`` are two strings in a row.

    The cache holds a finished ``ReviewResult`` — a plain value, bound to no
    event loop — so it survives between tests, and a run costs several calls
    per criterion.
    """
    key = (label, model)
    if key not in _CACHE:
        _CACHE[key] = await run_review_v3(
            SOURCE,
            model=model,
            document_type=DOCUMENT_TYPE,
            reviewer=profile,
            # A reviewer that selects no criteria makes no criterion call; its
            # output is edits, and without this it would produce nothing.
            editor=profile.criteria == (),
        )
    return _CACHE[key]


async def _run(*, profile_name: str, model: str) -> ReviewResult:
    """One review of the fixture by one shipped reviewer, as shipped."""
    return await _run_profile(
        profile=PROFILES[profile_name], model=model, label=profile_name
    )


def _overlaps(*, span: tuple[int, int], start: int, end: int) -> bool:
    return start < span[1] and end > span[0]


def _spans(result: ReviewResult) -> list[tuple[int, int, str]]:
    """Every located span this review produced, findings and edits alike."""
    located = [
        (quote.char_start, quote.char_end, finding.title)
        for finding in result.findings
        for quote in finding.quotes
    ]
    located += [(edit.char_start, edit.char_end, edit.title) for edit in result.edits]
    return located


def _hits(result: ReviewResult, defect: _Defect) -> list[str]:
    """The titles of whatever this review said about this defect's sentence."""
    return [
        title
        for start, end, title in _spans(result)
        if _overlaps(span=defect.span, start=start, end=end)
    ]


# ── the model tests ─────────────────────────────────────────────────────────


@pytest.mark.cloud
@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
@pytest.mark.parametrize("profile_name", [d.profile for d in DEFECTS])
async def test_each_specialist_finds_its_own_defect(
    profile_name: str, model: str
) -> None:
    """The first half: the questions reach the defect they were written for."""
    defect = BY_PROFILE[profile_name]
    result = await _run(profile_name=profile_name, model=model)
    hits = _hits(result, defect)
    assert hits, (
        f"the {profile_name} reviewer said nothing about {defect.defect}, "
        f"which was planted for it at {defect.span}. It reported "
        f"{len(result.findings)} finding(s) and {len(result.edits)} edit(s) "
        f"elsewhere. Sharpen the question, not the brief."
    )


@pytest.mark.cloud
@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
@pytest.mark.parametrize("profile_name", FINDING_PROFILES)
async def test_each_specialist_stays_in_its_lane(profile_name: str, model: str) -> None:
    """The second half, and the design's kill test for the ``brief`` field.

    A specialist that reports the other five specialists' defects as well as
    its own is a general reviewer wearing a name. If this fails across the
    board, the honest conclusion is that the brief is not doing work: remove
    the field and the three prompt slots, and drop the ``editor`` profile with
    them, since it is a brief and nothing else.
    """
    result = await _run(profile_name=profile_name, model=model)
    trespasses = {
        other.profile: _hits(result, other)
        for other in DEFECTS
        if other.profile != profile_name and _hits(result, other)
    }
    assert not trespasses, (
        f"the {profile_name} reviewer reported defects belonging to "
        f"{', '.join(sorted(trespasses))}: {trespasses}"
    )


@pytest.mark.cloud
@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
async def test_the_line_editor_rewrites_and_does_not_review(model: str) -> None:
    """Its criteria are empty, so a finding from it would be a routing bug."""
    result = await _run(profile_name="editor", model=model)
    assert result.findings == []
    assert result.edits, "the line editor's whole output is rewrites"


@pytest.mark.cloud
@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
async def test_report_pairwise_overlap_between_specialists(model: str) -> None:
    """Watched, not gated: how much of each pair's output lands together.

    The design asks for this number on a real draft; the fixture is the
    standing stand-in, because a real draft cannot be checked into the
    repository and a number nobody can reproduce is not a number. Run with
    ``-s`` to see the matrix.

    It asserts nothing about the values on purpose. A rising diagonal-adjacent
    figure between two profiles is a prompt to ask whether they are still two
    reviewers — a question for a person, not a threshold. The only assertion
    is that every specialist produced something to compare, since an empty row
    would make the whole matrix read as perfect separation.
    """
    outputs = {
        name: _spans(await _run(profile_name=name, model=model))
        for name in FINDING_PROFILES
    }
    width = max(len(name) for name in outputs)
    print(f"\npairwise overlap of located spans — {model}")
    print(" " * (width + 2) + "  ".join(f"{name:>10}" for name in outputs))
    for name, spans in outputs.items():
        cells = []
        for other, other_spans in outputs.items():
            shared = sum(
                any(
                    _overlaps(span=(start, end), start=o_start, end=o_end)
                    for o_start, o_end, _ in other_spans
                )
                for start, end, _ in spans
            )
            cells.append("—" if other == name else f"{shared} of {len(spans)}")
        print(f"{name:<{width}}  " + "  ".join(f"{cell:>10}" for cell in cells))
    for name, spans in outputs.items():
        assert spans, f"the {name} reviewer located nothing to compare"


# ── does the brief do work? (the design's kill test) ────────────────────────
#
# The lane test above cannot answer this on its own, and saying that it does
# would be the more comfortable mistake: each specialist also carries its own
# criteria, so separation between two specialists is fully explained by the
# questions alone. The only way to see the brief is to hold everything else
# fixed and take it away.
#
# The consequence is stated in the design and is not negotiable after the
# fact: if a specialist's findings with its brief are not distinguishable from
# its findings with an empty one, the `brief` field goes, the three prompt
# slots go, and the `editor` profile goes with them — it is a brief and a flag
# alias, and nothing else.


def _empty_brief(name: str) -> ReviewerProfile:
    """The same reviewer with the one field under test removed."""
    return replace(PROFILES[name], brief="")


def _summarise(result: ReviewResult) -> str:
    return (
        f"{len(result.findings)} finding(s), {len(result.edits)} edit(s), "
        f"{len(_spans(result))} located span(s)"
    )


def _trespasses(result: ReviewResult, *, own: str) -> dict[str, list[str]]:
    return {
        other.profile: _hits(result, other)
        for other in DEFECTS
        if other.profile != own and _hits(result, other)
    }


@pytest.mark.cloud
@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
@pytest.mark.parametrize("profile_name", [d.profile for d in DEFECTS])
async def test_the_brief_changes_what_a_specialist_reports(
    profile_name: str, model: str
) -> None:
    """One reviewer, twice: as shipped, and with ``brief=""``.

    Everything else is held fixed — same criteria, same document, same model,
    same document type — so any difference between the two runs is the brief
    and can be nothing else.

    Two things are measured and both are printed, because the decision this
    test informs is about specificity as much as about counts: how much each
    arm reported, and how far outside its own lane each arm went. The
    assertion is the weaker of the two claims — that the arms are
    distinguishable at all. A brief that cannot manage that is not doing work.
    """
    shipped = await _run(profile_name=profile_name, model=model)
    barless = await _run_profile(
        profile=_empty_brief(profile_name),
        model=model,
        label=f"{profile_name}:no-brief",
    )
    shipped_trespass = _trespasses(shipped, own=profile_name)
    barless_trespass = _trespasses(barless, own=profile_name)
    print(f"\n{profile_name} — {model}")
    print(f"  with brief:  {_summarise(shipped)}, out of lane: {shipped_trespass}")
    print(f"  empty brief: {_summarise(barless)}, out of lane: {barless_trespass}")

    assert _spans(shipped) != _spans(barless), (
        f"the {profile_name} reviewer reported the same spans with its brief "
        f"and with an empty one ({_summarise(shipped)}). If this holds across "
        "the specialists and across models, the brief mechanism is not doing "
        "work: remove the field and the three prompt slots, and drop the "
        "editor profile with them."
    )


@pytest.mark.cloud
@pytest.mark.parametrize("model", MODELS_UNDER_TEST)
async def test_the_missing_comparison_bar_changes_what_is_reported(
    model: str,
) -> None:
    """The thin-layer test the decision record asks for by name.

    ``Missing comparison`` question 1 enumerates the kinds of absent
    comparison and question 3 asks whether a present one is fair; question 2
    is purely a bar — *would its absence change how the central claim reads?
    Report those; drop the rest.* Without that bar the criterion is the one
    line the academic set already carries ("are there obvious missing
    comparisons the document should address?") with a longer preamble, and
    the honest move would be to drop ``adversary`` and fold its criteria into
    ``claims`` rather than ship a renamed one.

    So: the adversary, twice, with only that question removed.
    """
    barless_criterion = MISSING_COMPARISON.model_copy(
        update={
            "questions": [
                question
                for question in MISSING_COMPARISON.questions
                if not question.startswith("For each, would its absence change")
            ]
        }
    )
    assert len(barless_criterion.questions) == len(MISSING_COMPARISON.questions) - 1, (
        "the bar was not found — question 2 of Missing comparison has been "
        "reworded, and this test is comparing the criterion against itself"
    )
    profile = PROFILES["adversary"]
    with_bar = await _run(profile_name="adversary", model=model)
    without_bar = await _run_profile(
        profile=replace(
            profile,
            criteria=tuple(
                barless_criterion if c is MISSING_COMPARISON else c
                for c in profile.criteria or ()
            ),
        ),
        model=model,
        label="adversary:no-bar",
    )
    print(f"\nmissing comparison bar — {model}")
    print(f"  with the bar:    {_summarise(with_bar)}")
    print(f"  without the bar: {_summarise(without_bar)}")

    assert _spans(with_bar) != _spans(without_bar), (
        "the bar in Missing comparison question 2 changed nothing "
        f"({_summarise(with_bar)}). The criterion is then the academic set's "
        "one-line ancestor with a longer preamble: drop the adversary and "
        "fold its criteria into claims rather than ship a renamed one."
    )


def test_removing_the_brief_changes_nothing_else() -> None:
    """Deterministic guard on the comparison itself.

    If ``replace(profile, brief="")`` ever changed a second field, the model
    tests above would be measuring two things and reporting them as one.
    """
    for name in PROFILES:
        stripped = _empty_brief(name)
        assert stripped.brief == ""
        assert replace(stripped, brief=PROFILES[name].brief) == PROFILES[name]


def test_the_bar_this_thin_layer_test_removes_is_still_there() -> None:
    """Deterministic, so a reworded question 2 is caught on every commit
    rather than during a hand-run of the model test above."""
    bars = [
        question
        for question in MISSING_COMPARISON.questions
        if question.startswith("For each, would its absence change")
    ]
    assert len(bars) == 1


# ── the fixture's own integrity (deterministic, runs on every commit) ───────


def test_one_defect_per_specialist() -> None:
    """Every profile that produces findings has exactly one planted defect."""
    planted = [d.profile for d in DEFECTS]
    assert len(planted) == len(set(planted))
    assert set(planted) == set(PROFILES) - {DEFAULT_PROFILE_NAME}


def test_every_anchor_appears_in_the_draft_exactly_once() -> None:
    """An anchor that matched twice would make every hit ambiguous."""
    for defect in DEFECTS:
        assert SOURCE.count(defect.anchor) == 1, defect.profile


def test_no_two_anchors_overlap() -> None:
    """Overlapping anchors would make "found its own" and "trespassed" the
    same event."""
    spans = sorted(defect.span for defect in DEFECTS)
    for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
        assert first_end <= second_start


def test_the_draft_is_about_two_pages() -> None:
    """Long enough to have a body to walk, short enough to run by hand."""
    assert 600 <= len(SOURCE.split()) <= 1200


def test_the_overlap_test_is_the_one_being_used() -> None:
    """The half-open comparison, pinned — an off-by-one here would report
    every adjacent quote as a hit and turn the lane test into noise."""
    span = (10, 20)
    assert _overlaps(span=span, start=15, end=16)
    assert _overlaps(span=span, start=5, end=11)
    assert _overlaps(span=span, start=19, end=25)
    assert not _overlaps(span=span, start=0, end=10)
    assert not _overlaps(span=span, start=20, end=30)
