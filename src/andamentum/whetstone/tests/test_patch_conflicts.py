"""Regression tests for how overlapping and ambiguous text edits are applied.

Every case here was reproduced against the real machinery before the fix. The
short version of what used to happen when two reviewers wanted to rewrite the
same sentence:

- a pattern occurring three times in a paragraph was rewritten three times,
  including in sentences no reviewer had read;
- two edits on overlapping spans could BOTH apply and compose into a sentence
  neither proposed, with the run reporting complete success;
- a finding's comment on a passage an edit rewrote was written into
  ``comments.xml``, never anchored, and reported as applied — so the rewrite
  landed and the reason for it disappeared with no error anywhere.

The last one is why these matter beyond tidiness: it is the normal case as soon
as a finding and the edit that fixes it point at the same span.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from docx import Document

from andamentum.whetstone.docx.patch_editor import (
    EditPlacement,
    PatchDocxEditor,
    find_occurrences,
    resolve_edit_conflicts,
)
from andamentum.whetstone.models import DocumentPatch

SENTENCE = "We found that the results show a significant effect on patient outcomes."
OLD_SPAN = "the results show a significant effect"
NEW_SPAN = "the results are consistent with an effect"


def _docx(path: Path, paragraphs: list[str]) -> Path:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    doc.save(str(path))
    return path


def _edit(pattern: str, new_text: str, why: str, confidence: float) -> DocumentPatch:
    return DocumentPatch(
        patch_type="text_edit",
        text_pattern=pattern,
        original_text=pattern,
        new_text=new_text,
        explanation=why,
        confidence=confidence,
    )


def _comment(pattern: str, body: str, confidence: float = 0.9) -> DocumentPatch:
    return DocumentPatch(
        patch_type="comment",
        text_pattern=pattern,
        comment_text=body,
        explanation=body,
        confidence=confidence,
    )


# ── the arbitration rule, as a pure function ────────────────────────────


def _placement(start: int, end: int, *, paragraph: int = 0, name: str = "x"):
    return EditPlacement(
        patch=_edit(name, "replacement", name, 0.8),
        author="tester",
        paragraph_index=paragraph,
        start=start,
        end=end,
        target=name,
        replacement="replacement",
    )


def test_disjoint_spans_are_all_accepted() -> None:
    first, second = _placement(0, 5, name="a"), _placement(10, 15, name="b")
    accepted, conflicted = resolve_edit_conflicts([first, second])
    assert accepted == [first, second]
    assert conflicted == []


def test_touching_spans_do_not_count_as_overlapping() -> None:
    """``[0,5)`` and ``[5,9)`` are adjacent, not overlapping — both may apply."""
    first, second = _placement(0, 5, name="a"), _placement(5, 9, name="b")
    accepted, _ = resolve_edit_conflicts([first, second])
    assert len(accepted) == 2


def test_overlapping_spans_keep_the_first_and_name_the_winner() -> None:
    first, second = _placement(0, 20, name="a"), _placement(10, 30, name="b")
    accepted, conflicted = resolve_edit_conflicts([first, second])
    assert accepted == [first]
    # The loser is paired with the placement that displaced it, so the caller
    # can report which edit won rather than emitting a bare count.
    assert conflicted == [(second, first)]


def test_same_span_in_different_paragraphs_is_not_a_conflict() -> None:
    first = _placement(0, 20, paragraph=0, name="a")
    second = _placement(0, 20, paragraph=1, name="b")
    accepted, conflicted = resolve_edit_conflicts([first, second])
    assert len(accepted) == 2
    assert conflicted == []


def test_find_occurrences_reports_every_position() -> None:
    assert find_occurrences("ab", "ab_ab_ab") == [0, 3, 6]
    assert find_occurrences("zz", "ab") == []
    assert find_occurrences("", "ab") == []


# ── applied against a real document ─────────────────────────────────────


def test_repeated_pattern_is_refused_rather_than_rewritten_everywhere(
    tmp_path: Path,
) -> None:
    """A pattern with three matches identifies no single span, so it is refused.

    ``str.replace`` with no count used to rewrite all three. The producer knows
    which occurrence it read; this layer does not, and guessing rewrites a
    sentence the author never had reviewed.
    """
    paragraph = (
        "The data shows a trend in cohort A. "
        "The data shows a trend in cohort B. "
        "The data shows a trend in cohort C."
    )
    editor = PatchDocxEditor(
        str(_docx(tmp_path / "in.docx", [paragraph])), author="tester"
    )
    result = editor.apply_patches(
        [_edit("The data shows", "The data show", "subject-verb agreement", 0.9)]
    )

    assert result.applied_patches == 0
    assert len(result.failed_patches) == 1
    assert editor.paragraphs[0].modified == paragraph


def test_unique_pattern_still_applies(tmp_path: Path) -> None:
    """The control for the test above — one match is not ambiguous."""
    editor = PatchDocxEditor(
        str(_docx(tmp_path / "in.docx", [SENTENCE])), author="tester"
    )
    result = editor.apply_patches([_edit("patient outcomes", "clinical outcomes", "x", 0.9)])

    assert result.applied_patches == 1
    assert "clinical outcomes" in editor.paragraphs[0].modified


def test_overlapping_edits_never_compose(tmp_path: Path) -> None:
    """Two edits over one span produce one of the two rewrites, never a blend.

    Both used to apply: the first rewrite left the second's pattern intact, so
    the paragraph ended up saying something neither reviewer wrote.
    """
    substantive = _edit(
        OLD_SPAN,
        NEW_SPAN,
        "overclaim relative to sample size",
        0.4,
    )
    cosmetic = _edit(
        "a significant effect on patient outcomes",
        "a significant effect on outcomes",
        "concision",
        0.95,
    )
    editor = PatchDocxEditor(
        str(_docx(tmp_path / "in.docx", [SENTENCE])), author="tester"
    )
    result = editor.apply_patches([substantive, cosmetic])

    solo_substantive = SENTENCE.replace(OLD_SPAN, NEW_SPAN)
    solo_cosmetic = SENTENCE.replace(
        "a significant effect on patient outcomes", "a significant effect on outcomes"
    )
    assert editor.paragraphs[0].modified in {solo_substantive, solo_cosmetic}
    assert result.applied_patches == 1
    assert result.conflict_failures == 1


def test_conflict_is_counted_apart_from_a_location_miss(tmp_path: Path) -> None:
    """The cause and the remedy differ, so the counts must not be pooled."""
    editor = PatchDocxEditor(
        str(_docx(tmp_path / "in.docx", [SENTENCE])), author="tester"
    )
    result = editor.apply_patches(
        [
            _edit(OLD_SPAN, NEW_SPAN, "overclaim", 0.9),
            _edit("a significant effect", "a large effect", "wording", 0.5),
            _edit("text that is simply not in the document", "x", "missing", 0.9),
        ]
    )

    assert result.conflict_failures == 1
    assert result.location_failures == 1
    assert result.applied_patches == 1


def test_disjoint_edits_in_one_paragraph_both_apply(tmp_path: Path) -> None:
    editor = PatchDocxEditor(
        str(_docx(tmp_path / "in.docx", [SENTENCE])), author="tester"
    )
    result = editor.apply_patches(
        [
            _edit("We found that", "We observed that", "voice", 0.9),
            _edit("patient outcomes", "clinical outcomes", "precision", 0.8),
        ]
    )

    assert result.applied_patches == 2
    assert result.conflict_failures == 0
    text = editor.paragraphs[0].modified
    assert "We observed that" in text
    assert "clinical outcomes" in text


# ── comments beside edits ───────────────────────────────────────────────


def _finalize(tmp_path: Path, patches: list[DocumentPatch]):
    from andamentum.whetstone.docx.finalization import finalize_reviewed_document

    out = tmp_path / "reviewed.docx"
    _, result = finalize_reviewed_document(
        original_file_path=_docx(tmp_path / "in.docx", [SENTENCE]),
        patches=patches,
        review_summary="summary",
        issues_count=1,
        output_path=out,
        author="tester",
    )
    with zipfile.ZipFile(out) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        names = archive.namelist()
        comments_xml = (
            archive.read("word/comments.xml").decode("utf-8")
            if "word/comments.xml" in names
            else ""
        )
    return result, document_xml, comments_xml


def test_comment_anchors_onto_text_an_edit_inserted(tmp_path: Path) -> None:
    """A rewrite and the reason for it can coexist on the same sentence.

    The write-time index used to walk only direct-child ``<w:r>`` runs. Every
    run the rebuild produces for an edited span is nested inside a ``<w:ins>``
    or ``<w:del>`` container, so an edited passage was unreachable and the
    comment could not be placed anywhere. Insertions are indexed now: they are
    the paragraph as it reads once the changes are accepted.
    """
    result, document_xml, comments_xml = _finalize(
        tmp_path,
        [
            _edit(OLD_SPAN, NEW_SPAN, "overclaim relative to sample size", 0.9),
            _comment(NEW_SPAN, "Overclaim: n=12 does not support 'significant'."),
        ],
    )

    assert result.applied_patches == 2
    assert "n=12" in comments_xml
    assert "commentRangeStart" in document_xml
    assert "commentReference" in document_xml


def test_comment_on_a_rewritten_span_fails_loudly(tmp_path: Path) -> None:
    """Quoting the OLD wording after an edit replaced it is now a reported miss.

    It used to report success: the comment resolved against the pre-edit text,
    then missed at write time against the rebuilt runs, leaving an orphan body
    in ``comments.xml`` that Word does not show. Comments resolve after edits
    now, against the same text the write pass indexes, so the failure surfaces
    where the caller can see it.
    """
    result, document_xml, _ = _finalize(
        tmp_path,
        [
            _edit(OLD_SPAN, NEW_SPAN, "overclaim", 0.9),
            _comment(OLD_SPAN, "Overclaim: n=12 does not support 'significant'."),
        ],
    )

    assert result.applied_patches == 1
    assert [p.patch_type for p in result.failed_patches] == ["comment"]
    assert "commentRangeStart" not in document_xml


def test_comment_on_an_untouched_span_is_unaffected_by_an_edit_elsewhere(
    tmp_path: Path,
) -> None:
    """Ordering edits before comments must not cost an ordinary comment."""
    result, document_xml, comments_xml = _finalize(
        tmp_path,
        [
            _edit("We found that", "We observed that", "voice", 0.9),
            _comment("patient outcomes", "Which outcomes, specifically?"),
        ],
    )

    assert result.applied_patches == 2
    assert "specifically" in comments_xml
    assert "commentRangeStart" in document_xml
