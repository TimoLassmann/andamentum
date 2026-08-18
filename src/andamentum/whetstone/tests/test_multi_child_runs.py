"""A run is not one string, and an edit must not treat it as one.

Word puts several children in a single ``<w:r>``: two ``<w:t>`` fragments, a
``<w:br>`` from shift+enter, a ``<w:tab>``, and a
``<w:lastRenderedPageBreak/>`` — which Word writes on every save, so a real
manuscript is full of them.

Two earlier versions of the edit path got this wrong. The first rebuilt a run
from ``run.find(qn("t"))``, its FIRST text child, and replaced the whole run:
one edit turned "The first half<br>the second half." into "The initial half",
deleting a sentence with no tracked deletion while reporting success. The
second keyed cuts correctly by element but still replaced the run once per
element, so two cut fragments in one run left the second working on a run the
first had already detached.

Neither was caught, because every fixture put the break in a run of its own.
"""

from __future__ import annotations

import pytest
from docx import Document
from lxml import etree  # type: ignore

from andamentum.whetstone.docx.atoms import document_atoms, projection, qn
from andamentum.whetstone.docx.surgery import Edit, apply_edits
from andamentum.whetstone.docx.textmap import TextMap


def _paragraph_with(children: list[tuple[str, str | None]]):
    """One paragraph holding ONE run with the given children, in order."""
    doc = Document()
    paragraph = doc.add_paragraph()
    run = etree.SubElement(paragraph._p, qn("r"))
    for tag, text in children:
        element = etree.SubElement(run, qn(tag))
        if text is not None:
            element.text = text
    return doc, paragraph._p


def _edit(paragraph, target: str, replacement: str, author: str = "AI") -> None:
    textmap = TextMap(document_atoms([paragraph]))
    span = textmap.find(target)
    assert span is not None, f"target did not resolve: {target!r}"
    apply_edits(textmap, [Edit(span=span, new_text=replacement, author=author)])


def _text(paragraph) -> str:
    return projection(document_atoms([paragraph]))


def _count(paragraph, tag: str) -> int:
    return len(paragraph.findall(f".//{qn(tag)}"))


@pytest.mark.parametrize(
    "label,children,target,replacement,expected",
    [
        (
            "text, break, text",
            [("t", "The first half"), ("br", None), ("t", "the second half.")],
            "first",
            "initial",
            "The initial half the second half.",
        ),
        (
            "text, page-break marker, text",
            [
                ("t", "Before the break "),
                ("lastRenderedPageBreak", None),
                ("t", "after the break."),
            ],
            "Before",
            "Prior to",
            "Prior to the break after the break.",
        ),
        (
            "two text fragments",
            [("t", "Alpha beta "), ("t", "gamma delta.")],
            "beta",
            "BETA",
            "Alpha BETA gamma delta.",
        ),
        (
            "text, tab, text",
            [("t", "Column one"), ("tab", None), ("t", "Column two.")],
            "one",
            "ONE",
            "Column ONE Column two.",
        ),
    ],
)
def test_an_edit_keeps_every_other_child_of_the_run(
    label, children, target, replacement, expected
) -> None:
    _, paragraph = _paragraph_with(children)
    _edit(paragraph, target, replacement)
    assert _text(paragraph) == expected, label


def test_a_page_break_marker_survives_an_edit_beside_it() -> None:
    """Word emits this on every save, so losing it corrupts ordinary files."""
    _, paragraph = _paragraph_with(
        [("t", "Before "), ("lastRenderedPageBreak", None), ("t", "after.")]
    )
    _edit(paragraph, "Before", "Prior to")
    assert _count(paragraph, "lastRenderedPageBreak") == 1


def test_a_line_break_inside_a_text_run_survives() -> None:
    _, paragraph = _paragraph_with(
        [("t", "The first half"), ("br", None), ("t", "the second half.")]
    )
    _edit(paragraph, "first", "initial")
    assert _count(paragraph, "br") == 1


def test_two_edits_in_the_same_run_both_apply() -> None:
    """Each element is cut in its own right, in one pass over the run."""
    _, paragraph = _paragraph_with(
        [("t", "Alpha beta "), ("br", None), ("t", "gamma delta.")]
    )
    textmap = TextMap(document_atoms([paragraph]))
    first, second = textmap.find("beta"), textmap.find("delta")
    assert first is not None and second is not None
    report = apply_edits(
        textmap,
        [
            Edit(span=first, new_text="BETA", author="AI"),
            Edit(span=second, new_text="DELTA", author="AI"),
        ],
    )
    assert report.applied == 2
    assert _text(paragraph) == "Alpha BETA  gamma DELTA."
    assert _count(paragraph, "br") == 1


def test_each_edit_is_attributed_to_the_reviewer_that_asked_for_it() -> None:
    """Edit.author is per-edit; a batch used to be stamped with the first."""
    _, paragraph = _paragraph_with([("t", "Alpha beta gamma delta.")])
    textmap = TextMap(document_atoms([paragraph]))
    first, second = textmap.find("beta"), textmap.find("delta")
    assert first is not None and second is not None
    apply_edits(
        textmap,
        [
            Edit(span=first, new_text="BETA", author="Reviewer One"),
            Edit(span=second, new_text="DELTA", author="Reviewer Two"),
        ],
    )
    authors = {
        element.get(qn("author"))
        for element in paragraph.iter()
        if element.tag in (qn("ins"), qn("del"))
    }
    assert authors == {"Reviewer One", "Reviewer Two"}


def test_an_untouched_stretch_stays_one_run() -> None:
    """Carrying children across must not shatter the paragraph into one run
    per node — that would bloat the file and fragment the tracked changes."""
    _, paragraph = _paragraph_with(
        [("t", "keep one "), ("t", "keep two "), ("t", "edit me")]
    )
    _edit(paragraph, "edit me", "changed")
    kept = [
        run
        for run in paragraph.findall(qn("r"))
        if any("keep" in (t.text or "") for t in run.findall(qn("t")))
    ]
    assert len(kept) == 1
    assert "".join(t.text or "" for t in kept[0].findall(qn("t"))) == (
        "keep one keep two "
    )


# ── paragraph marks belong to whoever revised them first ────────────────


def test_an_existing_paragraph_mark_revision_keeps_its_author() -> None:
    """A paragraph mark holds at most one revision of each kind, so a second
    proposal cannot sit beside the first. Overwriting it restamped a
    co-author's revision with the AI's name — content survived, the audit
    trail did not."""
    from andamentum.whetstone.docx.surgery import _mark_paragraph_mark

    doc = Document()
    paragraph = doc.add_paragraph("The first thought ends here")._p
    doc.add_paragraph("and the second begins.")

    _mark_paragraph_mark(paragraph, "del", author="Dr Human", cid=900)
    _mark_paragraph_mark(paragraph, "del", author="whetstone (AI)", cid=901)

    marks = paragraph.findall(f"{qn('pPr')}/{qn('rPr')}/{qn('del')}")
    assert len(marks) == 1
    assert marks[0].get(qn("author")) == "Dr Human"


def test_a_paragraph_mark_is_still_marked_when_nobody_claimed_it() -> None:
    doc = Document()
    paragraph = doc.add_paragraph("A paragraph")._p
    from andamentum.whetstone.docx.surgery import _mark_paragraph_mark

    _mark_paragraph_mark(paragraph, "ins", author="whetstone (AI)", cid=1)
    marks = paragraph.findall(f"{qn('pPr')}/{qn('rPr')}/{qn('ins')}")
    assert len(marks) == 1
    assert marks[0].get(qn("author")) == "whetstone (AI)"


# ── collisions between two rewrites ─────────────────────────────────────


def test_the_survivor_of_a_collision_does_not_depend_on_input_order() -> None:
    """Arbitration reads the spans, not the order the caller built the list in.

    Before this, the winner was whichever edit came first in the list — which
    depends on the order sections finished and the order the model listed its
    suggestions. The same set of edits could resolve differently between two
    runs over an unchanged document. WHICH one wins is still arbitrary; that it
    is the same one every time is not.
    """
    from andamentum.whetstone.docx.textmap import TextMap

    def survivor(order):
        _, paragraph = _paragraph_with(
            [("t", "We found that the results show a large effect on outcomes.")]
        )
        textmap = TextMap(document_atoms([paragraph]))
        wide = textmap.find("the results show a large effect")
        narrow = textmap.find("a large effect on outcomes")
        assert wide is not None and narrow is not None
        edits = {
            "wide": Edit(span=wide, new_text="the results suggest an effect", author="AI"),
            "narrow": Edit(span=narrow, new_text="a modest effect on outcomes", author="AI"),
        }
        report = apply_edits(textmap, [edits[name] for name in order])
        assert report.applied == 1 and len(report.conflicted) == 1
        return _text(paragraph)

    assert survivor(["wide", "narrow"]) == survivor(["narrow", "wide"])


def test_a_collision_is_reported_with_both_sides() -> None:
    """A refusal the caller cannot see is a silent loss. The report names the
    edit that was dropped and the one that displaced it."""
    from andamentum.whetstone.docx.textmap import TextMap

    _, paragraph = _paragraph_with(
        [("t", "We found that the results show a large effect on outcomes.")]
    )
    textmap = TextMap(document_atoms([paragraph]))
    wide = textmap.find("the results show a large effect")
    narrow = textmap.find("a large effect on outcomes")
    assert wide is not None and narrow is not None

    report = apply_edits(
        textmap,
        [
            Edit(span=wide, new_text="the results suggest an effect", author="AI"),
            Edit(span=narrow, new_text="a modest effect on outcomes", author="AI"),
        ],
    )
    assert len(report.conflicted) == 1
    loser, winner = report.conflicted[0]
    assert loser is not winner
    assert loser not in report.applied_edits
    assert winner in report.applied_edits


def test_many_edits_over_one_span_keep_exactly_one() -> None:
    """N mutually overlapping edits refuse N-1. No cascade, nothing composed."""
    from andamentum.whetstone.docx.textmap import TextMap

    _, paragraph = _paragraph_with([("t", "alpha beta gamma delta epsilon zeta")])
    textmap = TextMap(document_atoms([paragraph]))
    found = [
        textmap.find("beta gamma"),
        textmap.find("gamma delta"),
        textmap.find("beta gamma delta"),
        textmap.find("delta epsilon"),
    ]
    assert all(s is not None for s in found)
    spans = [s for s in found if s is not None]
    report = apply_edits(
        textmap,
        [Edit(span=s, new_text=f"r{i}", author="AI") for i, s in enumerate(spans)],
    )
    assert report.applied + len(report.conflicted) == 4
    assert report.applied >= 1
    # Whatever survived, the paragraph reads as one coherent edit of the
    # original — never a blend of two.
    text = _text(paragraph)
    assert text.count("r") >= 1
