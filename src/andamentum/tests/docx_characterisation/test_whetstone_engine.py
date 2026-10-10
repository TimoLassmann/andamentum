"""Characterisation of whetstone's tracked-change engine (Engine W).

These tests pin what the engine does today, read back from the saved .docx:
the text a reader sees after accepting every change, the text after rejecting
every change, the revision markup present, the comment markers and the
footnote references. They do not pin implementation details (function
internals, intermediate atom streams).

Where the engine does something surprising, the assertion states what it
actually does, and the comment above the test says so. A change in any
asserted value makes the test fail.

Fixtures are the synthetic documents in ``andamentum.tests.docx_corpus``.
Revision dates are not pinned; they are wall-clock timestamps.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pytest
from lxml import etree  # type: ignore

from andamentum.tests.docx_corpus import builder
from andamentum.tests.docx_corpus._xml import _CT, _REL, _para, _run, _write_package
from andamentum.whetstone.docx.annotate import Note, Rewrite, apply_annotations
from andamentum.docx_edit.atoms import document_atoms, projection
from andamentum.whetstone.docx.footnotes import add_footnotes
from andamentum.whetstone.docx.low_level import DocxEditor

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
AUTHOR = "Reviewer"
ORIGINAL_PARAGRAPHS = [
    "Jane Doe recorded the first SaltLog reading at Apollo.",
    "The second paragraph follows on from the first.",
]


def _q(tag: str) -> str:
    return f"{{{W}}}{tag}"


# ── helpers: drive the engine ───────────────────────────────────────────


def _reviewed_text(source: Path) -> str:
    """The text the reviewer reads: the projection of the document's paragraphs."""
    editor = DocxEditor(str(source))
    return projection(document_atoms(editor.paragraph_elements))


def _span(text: str, needle: str, nth: int = 0) -> tuple[int, int]:
    """Offsets of the *nth* occurrence of *needle* in the reviewed text."""
    index = -1
    for _ in range(nth + 1):
        index = text.index(needle, index + 1)
    return index, index + len(needle)


def _annotate(
    source: Path,
    target: Path,
    *,
    rewrites: tuple[Rewrite, ...] = (),
    notes: tuple[Note, ...] = (),
):
    """Run the annotation path over *source* and save the result to *target*."""
    editor = DocxEditor(str(source), author=AUTHOR)
    reviewed = projection(document_atoms(editor.paragraph_elements))
    report = apply_annotations(
        editor, rewrites=list(rewrites), notes=list(notes), reviewed_text=reviewed
    )
    editor.write(str(target))
    return report


def _rewrite(text: str, needle: str, replacement: str, *, nth: int = 0, label: str = "") -> Rewrite:
    start, end = _span(text, needle, nth)
    return Rewrite(start, end, replacement, AUTHOR, label=label)


# ── helpers: read the saved file back ───────────────────────────────────


def _body(path: Path):
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("word/document.xml")).find(_q("body"))


def _part(path: Path, name: str):
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read(name))


def _paragraph_text(element, mode: str) -> str:
    """Text of one element under *mode*: ``accept`` or ``reject`` every change."""
    parts: list[str] = []
    for child in element:
        tag = child.tag
        if not isinstance(tag, str):
            continue
        if mode == "accept" and tag == _q("del"):
            continue
        if mode == "reject" and tag == _q("ins"):
            continue
        if tag == _q("t") or (mode == "reject" and tag == _q("delText")):
            parts.append(child.text or "")
        else:
            parts.append(_paragraph_text(child, mode))
    return "".join(parts)


def _resolved(path: Path, mode: str) -> list[str]:
    """Paragraphs after accepting or rejecting every change, Word-style.

    A paragraph whose own mark is removed by the change (a deleted mark when
    accepting, an inserted mark when rejecting) merges into the next paragraph.
    """
    removed_mark = "del" if mode == "accept" else "ins"
    lines: list[str] = []
    carry = ""
    for paragraph in _body(path).findall(_q("p")):
        text = carry + _paragraph_text(paragraph, mode)
        carry = ""
        mark = paragraph.find(f"{_q('pPr')}/{_q('rPr')}/{_q(removed_mark)}")
        if mark is not None:
            carry = text
        else:
            lines.append(text)
    if carry:
        lines.append(carry)
    return lines


def _accepted(path: Path) -> list[str]:
    return _resolved(path, "accept")


def _rejected(path: Path) -> list[str]:
    return _resolved(path, "reject")


def _revisions(path: Path) -> list[tuple[str, int, str, str]]:
    """Every revision in document order as (kind, id, author, text).

    A paragraph-mark revision has kind ``ins-mark`` or ``del-mark`` and empty text.
    """
    found: list[tuple[str, int, str, str]] = []
    for element in _body(path).iter(_q("ins"), _q("del")):
        kind = "ins" if element.tag == _q("ins") else "del"
        revision_id = int(element.get(_q("id")))
        author = element.get(_q("author"))
        if element.getparent().tag == _q("rPr"):
            found.append((f"{kind}-mark", revision_id, author, ""))
        else:
            text = "".join(
                node.text or "" for node in element.iter(_q("t"), _q("delText"))
            )
            found.append((kind, revision_id, author, text))
    return found


def _comments(path: Path) -> dict[int, tuple[str, str]]:
    """Comment id → (author, text), from ``word/comments.xml``."""
    root = _part(path, "word/comments.xml")
    out: dict[int, tuple[str, str]] = {}
    for comment in root.findall(_q("comment")):
        text = "".join(node.text or "" for node in comment.iter(_q("t")))
        out[int(comment.get(_q("id")))] = (comment.get(_q("author")), text)
    return out


def _text_between_comment_markers(path: Path, comment_id: int) -> str:
    """Text a comment's start and end markers bracket, read in document order."""
    active = False
    collected: list[str] = []
    for element in _body(path).iter():
        if element.tag == _q("commentRangeStart") and element.get(_q("id")) == str(comment_id):
            active = True
        elif element.tag == _q("commentRangeEnd") and element.get(_q("id")) == str(comment_id):
            active = False
        elif active and element.tag == _q("t"):
            collected.append(element.text or "")
    return "".join(collected)


def _signature(element) -> tuple:
    """Structure, attributes and text of an element, for byte-free comparison."""
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_signature(child) for child in element if isinstance(child.tag, str)),
    )


# ── replace / insert / delete inside one paragraph ──────────────────────


def test_replacing_a_word_leaves_one_tracked_deletion_and_one_tracked_insertion(
    tmp_path: Path,
) -> None:
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    report = _annotate(source, target, rewrites=(_rewrite(text, "first", "second"),))

    assert report.applied_rewrites == 1
    assert report.conflicted_rewrites == []
    # Insertion is written before the deletion it replaces (ins id 1000, del id 1001).
    assert _revisions(target) == [
        ("ins", 1000, AUTHOR, "second"),
        ("del", 1001, AUTHOR, "first"),
    ]
    assert _accepted(target) == [
        "Jane Doe recorded the second SaltLog reading at Apollo.",
        ORIGINAL_PARAGRAPHS[1],
    ]
    assert _rejected(target) == ORIGINAL_PARAGRAPHS


def test_replacing_a_word_does_not_touch_the_second_paragraph(tmp_path: Path) -> None:
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    _annotate(source, target, rewrites=(_rewrite(text, "first", "second"),))

    second = _body(target).findall(_q("p"))[1]
    assert second.find(f".//{_q('ins')}") is None
    assert second.find(f".//{_q('del')}") is None
    assert "".join(node.text or "" for node in second.iter(_q("t"))) == (
        "The second paragraph follows on from the first."
    )


def test_insertion_at_a_position_adds_tracked_text_and_keeps_the_original_run(
    tmp_path: Path,
) -> None:
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    _annotate(source, target, rewrites=(_rewrite(text, "Apollo", "Apollo Lab"),))

    assert _revisions(target) == [("ins", 1000, AUTHOR, " Lab")]
    assert _accepted(target) == [
        "Jane Doe recorded the first SaltLog reading at Apollo Lab.",
        ORIGINAL_PARAGRAPHS[1],
    ]
    assert _rejected(target) == ORIGINAL_PARAGRAPHS


def test_deleting_text_leaves_a_tracked_deletion_and_no_insertion(tmp_path: Path) -> None:
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    _annotate(source, target, rewrites=(_rewrite(text, " first", ""),))

    assert _revisions(target) == [("del", 1000, AUTHOR, " first")]
    assert _accepted(target) == [
        "Jane Doe recorded the SaltLog reading at Apollo.",
        ORIGINAL_PARAGRAPHS[1],
    ]
    assert _rejected(target) == ORIGINAL_PARAGRAPHS


# ── paragraph boundaries ────────────────────────────────────────────────


def test_newline_in_new_text_inserts_a_tracked_paragraph(tmp_path: Path) -> None:
    """A tracked paragraph break: the first paragraph's mark is inserted, the new text follows."""
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    report = _annotate(
        source,
        target,
        rewrites=(_rewrite(text, "at Apollo.", "at Apollo.\nA new line."),),
    )

    assert report.paragraphs_created == 1
    assert report.paragraphs_merged == 0
    assert _revisions(target) == [
        ("ins-mark", 1000, AUTHOR, ""),
        ("ins", 1001, AUTHOR, "A new line."),
    ]
    assert _accepted(target) == [
        "Jane Doe recorded the first SaltLog reading at Apollo.",
        "A new line.",
        ORIGINAL_PARAGRAPHS[1],
    ]
    assert _rejected(target) == ORIGINAL_PARAGRAPHS


def test_deleting_a_whole_paragraph_marks_the_break_and_the_text_deleted(
    tmp_path: Path,
) -> None:
    """The break before paragraph two is attributed to paragraph one's mark.

    Deleting paragraph two therefore marks paragraph one's mark as deleted; on
    accept the two paragraphs merge into one and paragraph two's text is gone.
    """
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    report = _annotate(
        source,
        target,
        rewrites=(
            _rewrite(text, "\nThe second paragraph follows on from the first.", ""),
        ),
    )

    assert report.paragraphs_merged == 1
    assert _revisions(target) == [
        ("del-mark", 1001, AUTHOR, ""),
        ("del", 1000, AUTHOR, "The second paragraph follows on from the first."),
    ]
    assert _accepted(target) == [ORIGINAL_PARAGRAPHS[0]]
    assert _rejected(target) == ORIGINAL_PARAGRAPHS


def test_deleting_paragraph_text_without_its_break_leaves_an_empty_paragraph(
    tmp_path: Path,
) -> None:
    """Deleting only the words of a paragraph leaves its paragraph mark in place.

    The paragraph therefore stays, empty, on accept. Removing the paragraph
    needs its mark deleted as well (the test above does that).
    """
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    report = _annotate(
        source,
        target,
        rewrites=(_rewrite(text, "The second paragraph follows on from the first.", ""),),
    )

    assert report.paragraphs_merged == 0
    assert _revisions(target) == [
        ("del", 1000, AUTHOR, "The second paragraph follows on from the first."),
    ]
    assert _accepted(target) == [ORIGINAL_PARAGRAPHS[0], ""]


# ── comments ────────────────────────────────────────────────────────────


def test_comment_brackets_its_range_and_is_referenced_after_the_range(
    tmp_path: Path,
) -> None:
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)
    start, end = _span(text, "SaltLog")

    report = _annotate(
        source,
        target,
        notes=(Note(start, end, "Define SaltLog.", AUTHOR),),
    )

    assert report.applied_notes == 1
    assert report.applied_rewrites == 0
    assert _comments(target) == {1: (AUTHOR, "Define SaltLog.")}
    assert _text_between_comment_markers(target, 1) == "SaltLog"

    # The reference run follows the range end directly, in the same paragraph.
    paragraph = _body(target).findall(_q("p"))[0]
    children = [child for child in paragraph if isinstance(child.tag, str)]
    end_index = next(
        index
        for index, child in enumerate(children)
        if child.tag == _q("commentRangeEnd")
    )
    assert children[end_index].get(_q("id")) == "1"
    reference_run = children[end_index + 1]
    assert reference_run.tag == _q("r")
    assert reference_run.find(_q("commentReference")).get(_q("id")) == "1"

    # Nothing about the visible text changes: no revision is written for a comment.
    assert _revisions(target) == []
    assert _accepted(target) == ORIGINAL_PARAGRAPHS


def test_comment_package_parts_and_relationship_are_written(tmp_path: Path) -> None:
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)
    start, end = _span(text, "SaltLog")

    _annotate(source, target, notes=(Note(start, end, "Define SaltLog.", AUTHOR),))

    rels = _part(target, "word/_rels/document.xml.rels")
    assert any(
        rel.get("Type") == f"{_REL}/comments" and rel.get("Target") == "comments.xml"
        for rel in rels
    )
    types = _part(target, "[Content_Types].xml")
    assert any(
        override.get("PartName") == "/word/comments.xml"
        and override.get("ContentType") == f"{_CT}.comments+xml"
        for override in types
    )


def test_comment_across_a_run_boundary_brackets_both_runs(tmp_path: Path) -> None:
    """The word ``Saline`` is two runs: bold ``Sali`` and plain ``ne``.

    The range is cut at the run boundaries and the start marker sits before the
    first run, the end marker after the second.
    """
    source = builder.formatted_runs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)
    start, end = _span(text, "Saline")

    _annotate(source, target, notes=(Note(start, end, "Check word.", AUTHOR),))

    assert _text_between_comment_markers(target, 1) == "Saline"
    assert _accepted(target) == [builder.FORMATTED_RUNS_TEXT]
    # The bold run is the first run the range covers; its formatting is untouched.
    paragraph = _body(target).find(_q("p"))
    start_marker = paragraph.find(_q("commentRangeStart"))
    following_run = start_marker.getnext()
    assert following_run.tag == _q("r")
    assert following_run.find(f"{_q('rPr')}/{_q('b')}") is not None
    assert following_run.find(_q("t")).text == "Sali"


# ── footnotes ───────────────────────────────────────────────────────────


def _placeholder_baseline(path: Path) -> Path:
    """A baseline document whose footnote marks are placeholder runs of their own."""
    body = _para(
        _run("Jane Doe recorded readings")
        + _run("⟦FN1⟧")
        + _run(" at Apollo.")
        + _run("⟦FN2⟧")
    ) + _para(_run("The second paragraph."))
    return _write_package(path, body, parts={}, overrides=[], doc_rels=[])


def test_add_footnotes_replaces_placeholders_with_numbered_references(
    tmp_path: Path,
) -> None:
    baseline = _placeholder_baseline(tmp_path / "base.docx")

    add_footnotes(baseline, ["Invented note one.", "Invented note two & more."])

    references = [
        element
        for element in _body(baseline).iter(_q("footnoteReference"))
    ]
    assert [reference.get(_q("id")) for reference in references] == ["1", "2"]
    for reference in references:
        run = reference.getparent()
        assert run.find(f"{_q('rPr')}/{_q('vertAlign')}").get(_q("val")) == "superscript"

    footnotes = _part(baseline, "word/footnotes.xml")
    by_id = {
        footnote.get(_q("id")): "".join(node.text or "" for node in footnote.iter(_q("t")))
        for footnote in footnotes.findall(_q("footnote"))
    }
    assert by_id == {
        "-1": "",
        "0": "",
        "1": "Invented note one.",
        "2": "Invented note two & more.",
    }
    assert footnotes.findall(_q("footnote"))[0].get(_q("type")) == "separator"
    assert footnotes.findall(_q("footnote"))[1].get(_q("type")) == "continuationSeparator"

    # The placeholders are gone; the reader sees the text with the marks in place.
    assert _accepted(baseline) == [
        "Jane Doe recorded readings at Apollo.",
        "The second paragraph.",
    ]


def test_add_footnotes_registers_the_part_in_rels_and_content_types(tmp_path: Path) -> None:
    baseline = _placeholder_baseline(tmp_path / "base.docx")

    add_footnotes(baseline, ["Invented note one.", "Invented note two."])

    rels = _part(baseline, "word/_rels/document.xml.rels")
    assert any(
        rel.get("Id") == "rIdAndamentumFootnotes"
        and rel.get("Type") == f"{_REL}/footnotes"
        and rel.get("Target") == "footnotes.xml"
        for rel in rels
    )
    types = _part(baseline, "[Content_Types].xml")
    assert any(
        override.get("PartName") == "/word/footnotes.xml"
        and override.get("ContentType") == f"{_CT}.footnotes+xml"
        for override in types
    )


def test_add_footnotes_refuses_a_placeholder_inside_a_text_run(tmp_path: Path) -> None:
    """A placeholder that shares a run with other words is refused.

    add_footnotes does not split runs, so the placeholder must be the whole
    text of its run; otherwise it raises ValueError.
    """
    path = _write_package(
        tmp_path / "bad.docx",
        _para(_run("Jane Doe recorded readings⟦FN1⟧ at Apollo.")),
        parts={},
        overrides=[],
        doc_rels=[],
    )

    with pytest.raises(ValueError, match="footnote placeholder 1 found 0 times"):
        add_footnotes(path, ["Invented note one."])


def test_add_footnotes_refuses_a_document_that_already_has_footnotes(
    tmp_path: Path,
) -> None:
    baseline = _placeholder_baseline(tmp_path / "base.docx")
    add_footnotes(baseline, ["Invented note one.", "Invented note two."])

    with pytest.raises(ValueError, match="already has footnotes"):
        add_footnotes(baseline, ["Another note."])


def test_a_tracked_rewrite_next_to_a_footnote_keeps_the_reference_in_place(
    tmp_path: Path,
) -> None:
    baseline = _placeholder_baseline(tmp_path / "base.docx")
    add_footnotes(baseline, ["Invented note one.", "Invented note two."])
    target = tmp_path / "out.docx"
    text = _reviewed_text(baseline)

    _annotate(baseline, target, rewrites=(_rewrite(text, "Apollo", "Lab"),))

    references = [element.get(_q("id")) for element in _body(target).iter(_q("footnoteReference"))]
    assert references == ["1", "2"]
    assert _revisions(target) == [
        ("ins", 1000, AUTHOR, "Lab"),
        ("del", 1001, AUTHOR, "Apollo"),
    ]
    assert _accepted(target) == [
        "Jane Doe recorded readings at Lab.",
        "The second paragraph.",
    ]


# ── documents that already carry tracked changes ────────────────────────


def test_edit_on_a_document_with_tracked_changes_numbers_above_existing_ids(
    tmp_path: Path,
) -> None:
    source = builder.existing_tracked(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    _annotate(source, target, rewrites=(_rewrite(text, "SaltLog", "Saline"),))

    assert _revisions(target) == [
        ("ins", 101, "Jane Doe", "every meal"),
        ("del", 102, "Jane Doe", " on paper"),
        ("ins", 103, AUTHOR, "Saline"),
        ("del", 104, AUTHOR, "SaltLog"),
    ]
    assert _accepted(target) == ["Participants logged every meal in the Saline app."]
    # Rejecting everything restores the text as it stood before the edit.
    assert _rejected(target) == ["Participants logged  on paper in the SaltLog app."]


def test_co_author_revision_elements_are_unchanged_by_an_edit_elsewhere(
    tmp_path: Path,
) -> None:
    source = builder.existing_tracked(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    _annotate(source, target, rewrites=(_rewrite(text, "SaltLog", "Saline"),))

    def co_author_marks(path: Path) -> list[tuple]:
        return [
            _signature(element)
            for element in _body(path).iter(_q("ins"), _q("del"))
            if element.get(_q("author")) == "Jane Doe"
        ]

    before = co_author_marks(source)
    after = co_author_marks(target)
    assert len(before) == 2
    assert after == before


def test_edit_inside_a_co_author_insertion_nests_the_deletion_in_it(
    tmp_path: Path,
) -> None:
    """The reviewer's deletion of ``every`` is nested inside Jane Doe's insertion.

    This is the markup whetstone's CLAUDE.md describes for a second reviewer
    editing text an earlier author inserted. Jane Doe's ``w:ins`` keeps its
    author and id; the reviewer's ``w:del`` (id 103) sits inside it, and the
    reviewer's insertion ``each`` (id 104) is a sibling placed before Jane
    Doe's insertion. Whether Word displays this nesting as intended has not
    been checked in Word.
    """
    source = builder.existing_tracked(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    _annotate(source, target, rewrites=(_rewrite(text, "every", "each"),))

    assert _accepted(target) == ["Participants logged each meal in the SaltLog app."]

    jane = next(
        element
        for element in _body(target).iter(_q("ins"))
        if element.get(_q("author")) == "Jane Doe"
    )
    assert jane.get(_q("id")) == "101"
    nested = jane.find(_q("del"))
    assert nested is not None
    assert nested.get(_q("author")) == AUTHOR
    assert nested.get(_q("id")) == "103"
    assert "".join(node.text or "" for node in nested.iter(_q("delText"))) == "every"

    sibling = jane.getprevious()
    assert sibling.tag == _q("ins")
    assert sibling.get(_q("author")) == AUTHOR
    assert sibling.get(_q("id")) == "104"


def test_comment_added_to_a_document_with_comments_takes_the_next_free_id(
    tmp_path: Path,
) -> None:
    source = builder.existing_comments(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)
    start, end = _span(text, "in the pilot")

    _annotate(source, target, notes=(Note(start, end, "Name the pilot.", AUTHOR),))

    comments = _comments(target)
    assert sorted(comments) == [0, 1, 2, 3]
    assert comments[3] == (AUTHOR, "Name the pilot.")
    assert comments[0] == ("Reviewer A", "Is this causal?")


# ── overlapping edits ───────────────────────────────────────────────────


def test_overlapping_rewrite_is_refused_and_reported_not_applied(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two rewrites share the word ``first``. The one starting earlier wins.

    The later one is reported in ``conflicted_rewrites`` and logged as DROPPED;
    its replacement text does not reach the file at all.
    """
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)
    winner = _rewrite(text, "recorded the first", "logged the second", label="one")
    loser = Rewrite(
        *_span(text, "first SaltLog"), "third SaltLog", "Other", label="two"
    )

    # The surgery module logs to "andamentum.whetstone". The CLI tests turn
    # propagation off on that logger and never restore it, so the capture
    # handler is attached to it directly instead of relying on propagation.
    logger = logging.getLogger("andamentum.whetstone")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="andamentum.whetstone"):
            report = _annotate(source, target, rewrites=(winner, loser))
    finally:
        logger.removeHandler(caplog.handler)

    assert report.total == 2
    assert report.applied_rewrites == 1
    assert [rewrite.label for rewrite in report.conflicted_rewrites] == ["two"]
    assert "edit DROPPED" in caplog.text
    assert _accepted(target) == [
        "Jane Doe logged the second SaltLog reading at Apollo.",
        ORIGINAL_PARAGRAPHS[1],
    ]
    with zipfile.ZipFile(target) as archive:
        assert b"third" not in archive.read("word/document.xml")
    assert all(author != "Other" for _, _, author, _ in _revisions(target))


# ── a run boundary inside the edited span ───────────────────────────────


def test_replacement_spanning_a_run_boundary_keeps_each_run_formatting(
    tmp_path: Path,
) -> None:
    """``Saline`` spans bold ``Sali`` and plain ``ne``; each deleted run keeps its own format.

    The inserted word inherits the formatting of the character before the
    span, here the plain space after ``The``. The untouched runs (italic
    ``slowly``, superscript ``2``, coloured ``red``) are carried through as is.
    """
    source = builder.formatted_runs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)

    _annotate(source, target, rewrites=(_rewrite(text, "Saline", "Brine"),))

    assert _accepted(target) == [
        "The Brine flows slowly from H2O into red tubes."
    ]
    assert _rejected(target) == [builder.FORMATTED_RUNS_TEXT]
    assert _revisions(target) == [
        ("ins", 1002, AUTHOR, "Brine"),
        ("del", 1000, AUTHOR, "Sali"),
        ("del", 1001, AUTHOR, "ne"),
    ]

    deleted_sali = _body(target).find(f".//{_q('del')}[@{_q('id')}='1000']/{_q('r')}")
    assert deleted_sali.find(f"{_q('rPr')}/{_q('b')}") is not None
    deleted_ne = _body(target).find(f".//{_q('del')}[@{_q('id')}='1001']/{_q('r')}")
    assert deleted_ne.find(_q("rPr")) is None

    untouched = {
        run.find(_q("t")).text: run.find(f"{_q('rPr')}/*").tag.split("}")[1]
        for run in _body(target).findall(f".//{_q('r')}")
        if run.find(f"{_q('rPr')}/*") is not None and run.find(_q("t")) is not None
    }
    assert untouched == {
        "slowly": "i",
        "2": "vertAlign",
        "red": "color",
    }


def test_note_overlapping_a_rewrite_is_moved_onto_the_new_text(tmp_path: Path) -> None:
    """A note whose span overlaps a rewrite is placed, not dropped.

    The note covers ``recorded the first SaltLog`` and the rewrite replaces
    ``first``. The rewrite is applied, the note is placed on the text that now
    stands in its place, and the report lists the note in ``moved_notes``.
    """
    source = builder.plain_two_paragraphs(tmp_path / "src")
    target = tmp_path / "out.docx"
    text = _reviewed_text(source)
    rewrite = _rewrite(text, "first", "second", label="rewrite")
    note_start, note_end = _span(text, "recorded the first SaltLog")

    report = _annotate(
        source,
        target,
        rewrites=(rewrite,),
        notes=(Note(note_start, note_end, "Overlap note.", AUTHOR),),
    )

    assert report.applied_rewrites == 1
    assert report.conflicted_rewrites == []
    assert report.applied_notes == 1
    assert report.shadowed_notes == []
    assert [note.body for note in report.moved_notes] == ["Overlap note."]
    assert _comments(target) == {1: (AUTHOR, "Overlap note.")}
    assert _text_between_comment_markers(target, 1) == "recorded the second SaltLog"
    assert _accepted(target)[0] == (
        "Jane Doe recorded the second SaltLog reading at Apollo."
    )
