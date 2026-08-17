"""End-to-end behaviour of a patch batch against a real .docx.

Every case here was reproduced against the shipped machinery before the fix.
What used to happen when two annotations landed on one sentence:

- a pattern occurring three times in a paragraph was rewritten three times,
  including in sentences no reviewer had read;
- two edits on overlapping spans could BOTH apply and compose into a sentence
  neither proposed, with the run reporting complete success;
- a finding's comment on a passage an edit rewrote was written into
  ``comments.xml``, never anchored, and reported as applied — so the rewrite
  landed and the reason for it disappeared with no error anywhere.

The unit-level arbitration tests live beside the code that now does the work,
in ``test_surgery.py``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from docx import Document

from andamentum.whetstone.docx.patch_editor import PatchDocxEditor
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


def _text(editor: PatchDocxEditor) -> str:
    from andamentum.whetstone.docx.atoms import document_atoms, projection

    return projection(document_atoms(editor.paragraph_elements))


# ── ambiguity ───────────────────────────────────────────────────────────


def test_a_repeated_pattern_is_refused_rather_than_rewritten_everywhere(
    tmp_path: Path,
) -> None:
    """Three matches identify no single passage, so the patch is refused.

    ``str.replace`` with no count used to rewrite all three, including in
    sentences the reviewer never quoted.
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
    assert _text(editor) == paragraph


def test_a_unique_pattern_applies(tmp_path: Path) -> None:
    editor = PatchDocxEditor(
        str(_docx(tmp_path / "in.docx", [SENTENCE])), author="tester"
    )
    result = editor.apply_patches(
        [_edit("patient outcomes", "clinical outcomes", "precision", 0.9)]
    )

    assert result.applied_patches == 1
    assert "clinical outcomes" in _text(editor)


# ── conflicts ───────────────────────────────────────────────────────────


def test_overlapping_edits_never_compose(tmp_path: Path) -> None:
    """Two edits over one span give one of the two rewrites, never a blend."""
    editor = PatchDocxEditor(
        str(_docx(tmp_path / "in.docx", [SENTENCE])), author="tester"
    )
    result = editor.apply_patches(
        [
            _edit(OLD_SPAN, NEW_SPAN, "overclaim relative to sample size", 0.4),
            _edit(
                "a significant effect on patient outcomes",
                "a significant effect on outcomes",
                "concision",
                0.95,
            ),
        ]
    )

    solo_substantive = SENTENCE.replace(OLD_SPAN, NEW_SPAN)
    solo_cosmetic = SENTENCE.replace(
        "a significant effect on patient outcomes", "a significant effect on outcomes"
    )
    assert _text(editor) in {solo_substantive, solo_cosmetic}
    assert result.applied_patches == 1
    assert result.conflict_failures == 1


def test_a_conflict_is_counted_apart_from_a_location_miss(tmp_path: Path) -> None:
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
    text = _text(editor)
    assert "We observed that" in text and "clinical outcomes" in text


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
        names = archive.namelist()
        document_xml = archive.read("word/document.xml").decode("utf-8")
        comments_xml = (
            archive.read("word/comments.xml").decode("utf-8")
            if "word/comments.xml" in names
            else ""
        )
    return result, document_xml, comments_xml


def test_a_rewrite_and_the_reason_for_it_can_share_a_sentence(
    tmp_path: Path,
) -> None:
    """The comment anchors onto the replacement text, beside the tracked change."""
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


def test_a_comment_body_is_never_written_without_an_anchor(tmp_path: Path) -> None:
    """Quoting the pre-edit wording is a reported miss, not a silent orphan.

    It used to report success: the comment resolved against the pre-edit text,
    then missed at write time against the rebuilt runs, leaving a body in
    ``comments.xml`` that Word does not show.
    """
    result, document_xml, comments_xml = _finalize(
        tmp_path,
        [
            _edit(OLD_SPAN, NEW_SPAN, "overclaim", 0.9),
            _comment(OLD_SPAN, "Overclaim: n=12 does not support 'significant'."),
        ],
    )

    assert result.applied_patches == 1
    assert [p.patch_type for p in result.failed_patches] == ["comment"]
    assert "n=12" not in comments_xml
    assert "commentRangeStart" not in document_xml


def test_a_comment_elsewhere_is_unaffected_by_an_edit(tmp_path: Path) -> None:
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


def test_every_comment_body_written_has_a_range_in_the_document(
    tmp_path: Path,
) -> None:
    """The count that used to lie: 8 comments reported applied, 2 visible."""
    result, document_xml, comments_xml = _finalize(
        tmp_path,
        [
            _comment("We found that", "One"),
            _comment("patient outcomes", "Two"),
            _comment("not present in the text at all", "Three"),
        ],
    )

    bodies = comments_xml.count("<w:comment ")
    ranges = document_xml.count("commentRangeStart")
    assert bodies == ranges == 2
    assert result.applied_comments == 2
    assert len(result.failed_patches) == 1
