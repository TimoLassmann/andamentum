"""Tests for ``andamentum-whetstone --apply-patches``.

The patch-only CLI path is the analogue of the standalone
``apply-patches.py`` from the manuscript-tools plugin: take a .docx, take
a JSON list of ``DocumentPatch``, write a tracked-changes .docx. These
tests exercise the full end-to-end CLI surface against a real .docx
fixture (no mocks), plus the argparse-time validation rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from docx import Document

from andamentum.whetstone.cli import _build_parser, _validate_args, main


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _make_docx(path: Path, paragraphs: list[str]) -> None:
    """Write a minimal .docx with the given paragraphs to ``path``."""
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))


def _read_docx_xml_text(path: Path) -> str:
    """Concatenate ALL ``<w:t>`` text in a .docx body.

    Unlike ``Paragraph.text``, this surfaces text inside ``<w:ins>``
    (tracked insertions), which is exactly what the patch editor writes
    when applying a text_edit patch.
    """
    doc = Document(str(path))
    body = doc.element.body
    return "".join(t.text or "" for t in body.iter(f"{{{NS['w']}}}t"))


def _tracked_insertions(path: Path) -> list[str]:
    """Return the text content of every ``<w:ins>`` element in the document."""
    doc = Document(str(path))
    body = doc.element.body
    out: list[str] = []
    for ins in body.iter(f"{{{NS['w']}}}ins"):
        out.append("".join(t.text or "" for t in ins.iter(f"{{{NS['w']}}}t")))
    return out


def _tracked_deletions(path: Path) -> list[str]:
    """Return the text content of every ``<w:del>`` element in the document."""
    doc = Document(str(path))
    body = doc.element.body
    out: list[str] = []
    for d in body.iter(f"{{{NS['w']}}}del"):
        # Tracked deletions store text in <w:delText> not <w:t>
        out.append("".join(t.text or "" for t in d.iter(f"{{{NS['w']}}}delText")))
    return out


# ── argparse-time validation ────────────────────────────────────────────


def test_apply_patches_requires_docx_input(tmp_path: Path) -> None:
    """--apply-patches rejects a non-.docx INPUT."""
    md = tmp_path / "draft.md"
    md.write_text("# hi")
    patches = tmp_path / "patches.json"
    patches.write_text("[]")

    parser = _build_parser()
    args = parser.parse_args(
        [
            str(md),
            "--apply-patches",
            str(patches),
            "--out",
            str(tmp_path / "out.docx"),
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_apply_patches_requires_docx_out(tmp_path: Path) -> None:
    """--apply-patches rejects a non-.docx --out."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["hello"])
    patches = tmp_path / "patches.json"
    patches.write_text("[]")

    parser = _build_parser()
    args = parser.parse_args(
        [
            str(src),
            "--apply-patches",
            str(patches),
            "--out",
            str(tmp_path / "out.md"),
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_apply_patches_rejects_missing_patches_file(tmp_path: Path) -> None:
    """--apply-patches rejects a path that doesn't exist."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["hello"])

    parser = _build_parser()
    args = parser.parse_args(
        [
            str(src),
            "--apply-patches",
            str(tmp_path / "missing.json"),
            "--out",
            str(tmp_path / "out.docx"),
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_apply_patches_skips_model_requirement(tmp_path: Path) -> None:
    """In patch-only mode --model is NOT required."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["hello"])
    patches = tmp_path / "patches.json"
    patches.write_text("[]")

    parser = _build_parser()
    args = parser.parse_args(
        [
            str(src),
            "--apply-patches",
            str(patches),
            "--out",
            str(tmp_path / "out.docx"),
        ]
    )
    _validate_args(args)  # must not raise


def test_patch_report_without_apply_patches_rejected(tmp_path: Path) -> None:
    """--patch-report only makes sense in patch-only mode."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            str(tmp_path / "out.md"),
            "--patch-report",
            str(tmp_path / "report.md"),
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_patch_author_on_the_review_path_still_needs_the_override(
    tmp_path: Path,
) -> None:
    """Naming an author is allowed in review mode; naming one silently is not.

    Review mode accepts ``--patch-author`` because a second review round needs
    a second name. The misconduct rail is unchanged by that: the flag on its
    own is still refused, on this path exactly as on the patch-only one.
    """
    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            str(tmp_path / "out.md"),
            "--model",
            "stub",
            "--patch-author",
            "Alice",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_patch_author_with_override_is_accepted_on_the_review_path(
    tmp_path: Path,
) -> None:
    """The pair together is how round two gets its own name in Word."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            str(tmp_path / "out.docx"),
            "--model",
            "stub",
            "--patch-author",
            "Reviewer B",
            "--allow-author-override",
        ]
    )
    _validate_args(args)  # does not exit


def test_naming_an_author_with_no_docx_output_is_refused(tmp_path: Path) -> None:
    """Only a .docx carries tracked changes, so only a .docx has an author."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "paper.md",
            "--out",
            str(tmp_path / "review.md"),
            "--model",
            "stub",
            "--patch-author",
            "Reviewer B",
            "--allow-author-override",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


# ── end-to-end CLI run ─────────────────────────────────────────────────


def test_apply_patches_text_edit_end_to_end(tmp_path: Path) -> None:
    """A text_edit patch rewrites prose; the output .docx contains the new text."""
    src = tmp_path / "draft.docx"
    _make_docx(
        src,
        [
            "The quick brown fox jumps over the lazy dog.",
            "A second paragraph for context.",
        ],
    )

    patches = [
        {
            "patch_type": "text_edit",
            "text_pattern": "lazy dog",
            "new_text": "sleeping dog",
            "explanation": "More accurate description",
            "confidence": 0.9,
        }
    ]
    patches_path = tmp_path / "patches.json"
    patches_path.write_text(json.dumps(patches))

    out = tmp_path / "out.docx"
    main(
        [
            str(src),
            "--apply-patches",
            str(patches_path),
            "--out",
            str(out),
        ]
    )

    assert out.exists()
    # Track changes encode the diff token-by-token: the editor inserts
    # only the changed words ("sleeping") and deletes the original
    # ("lazy"), reusing the unchanged " dog" run. So check word-level
    # rather than full-phrase containment.
    inserted = " ".join(_tracked_insertions(out))
    deleted = " ".join(_tracked_deletions(out))
    assert "sleeping" in inserted
    assert "lazy" in deleted


def test_apply_patches_with_report_prepends_section(tmp_path: Path) -> None:
    """--patch-report prepends a review-summary section to the output."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["Some text to review."])

    patches_path = tmp_path / "patches.json"
    patches_path.write_text("[]")
    report = tmp_path / "report.md"
    report.write_text("Reviewer noticed nothing structural.")

    out = tmp_path / "out.docx"
    main(
        [
            str(src),
            "--apply-patches",
            str(patches_path),
            "--out",
            str(out),
            "--patch-report",
            str(report),
        ]
    )

    text = _read_docx_xml_text(out)
    assert "Reviewer noticed nothing structural." in text
    assert "Some text to review." in text


def test_apply_patches_invalid_json_array(tmp_path: Path) -> None:
    """Patches file must be a JSON array."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["hello"])
    patches_path = tmp_path / "patches.json"
    patches_path.write_text(json.dumps({"patches": []}))  # object, not array

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                str(src),
                "--apply-patches",
                str(patches_path),
                "--out",
                str(tmp_path / "out.docx"),
            ]
        )
    assert excinfo.value.code == 1


def test_apply_patches_invalid_patch_object(tmp_path: Path) -> None:
    """A malformed patch entry yields a config-error exit (1)."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["hello"])
    patches_path = tmp_path / "patches.json"
    # Missing required fields for text_edit (no new_text, no explanation)
    patches_path.write_text(
        json.dumps([{"patch_type": "text_edit", "text_pattern": "hello"}])
    )

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                str(src),
                "--apply-patches",
                str(patches_path),
                "--out",
                str(tmp_path / "out.docx"),
            ]
        )
    assert excinfo.value.code == 1


# ── AI-author attribution gate ─────────────────────────────────────────


def test_default_author_is_ai_attributed(tmp_path: Path) -> None:
    """With no --patch-author flag, the docx is tagged as AI-authored."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["The quick brown fox jumps over the lazy dog."])
    patches_path = tmp_path / "patches.json"
    patches_path.write_text(
        json.dumps(
            [
                {
                    "patch_type": "text_edit",
                    "text_pattern": "lazy dog",
                    "new_text": "sleeping dog",
                    "explanation": "X",
                    "confidence": 0.9,
                }
            ]
        )
    )
    out = tmp_path / "out.docx"
    main([str(src), "--apply-patches", str(patches_path), "--out", str(out)])

    # The tracked-insert author attribute should contain the AI default.
    doc = Document(str(out))
    body = doc.element.body
    ins_authors = {
        ins.get(f"{{{NS['w']}}}author") for ins in body.iter(f"{{{NS['w']}}}ins")
    }
    assert any("andamentum-whetstone" in (a or "") for a in ins_authors)


def test_patch_author_override_requires_explicit_flag(tmp_path: Path) -> None:
    """--patch-author NAME without --allow-author-override refuses to run."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["hello"])
    patches_path = tmp_path / "patches.json"
    patches_path.write_text("[]")

    parser = _build_parser()
    args = parser.parse_args(
        [
            str(src),
            "--apply-patches",
            str(patches_path),
            "--out",
            str(tmp_path / "out.docx"),
            "--patch-author",
            "Dr Smith",
        ]
    )
    with pytest.raises(SystemExit) as excinfo:
        _validate_args(args)
    assert excinfo.value.code == 1


def test_allow_author_override_without_patch_author_rejected(
    tmp_path: Path,
) -> None:
    """--allow-author-override on its own is meaningless and refused."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["hello"])
    patches_path = tmp_path / "patches.json"
    patches_path.write_text("[]")

    parser = _build_parser()
    args = parser.parse_args(
        [
            str(src),
            "--apply-patches",
            str(patches_path),
            "--out",
            str(tmp_path / "out.docx"),
            "--allow-author-override",
        ]
    )
    with pytest.raises(SystemExit):
        _validate_args(args)


def test_explicit_author_override_works_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With both flags set, the override takes effect and prints a warning."""
    src = tmp_path / "draft.docx"
    _make_docx(src, ["The quick brown fox jumps over the lazy dog."])
    patches_path = tmp_path / "patches.json"
    patches_path.write_text(
        json.dumps(
            [
                {
                    "patch_type": "text_edit",
                    "text_pattern": "lazy dog",
                    "new_text": "sleeping dog",
                    "explanation": "X",
                    "confidence": 0.9,
                }
            ]
        )
    )
    out = tmp_path / "out.docx"
    main(
        [
            str(src),
            "--apply-patches",
            str(patches_path),
            "--out",
            str(out),
            "--patch-author",
            "Dr Smith",
            "--allow-author-override",
        ]
    )

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "Dr Smith" in captured.err
    assert "research misconduct" in captured.err

    doc = Document(str(out))
    body = doc.element.body
    ins_authors = {
        ins.get(f"{{{NS['w']}}}author") for ins in body.iter(f"{{{NS['w']}}}ins")
    }
    assert any(a == "Dr Smith" for a in ins_authors)


# ── dropped-annotation warning (review / checklist docx path) ───────────


def _report(**kwargs):
    from andamentum.whetstone.docx.annotate import AnnotationReport

    return AnnotationReport(**kwargs)


def test_shadowed_comments_are_reported_not_swallowed(capsys) -> None:
    """A partial .docx write must say so.

    ``render_docx``'s return value used to be discarded by the review and
    checklist paths, so a run that placed 2 of 7 comments reported plain
    success. Observed on a real manuscript.
    """
    from rich.console import Console

    from andamentum.whetstone.cli import _warn_on_dropped_annotations
    from andamentum.whetstone.docx.annotate import Note

    note = Note(start=0, end=4, body="b", author="a")
    _warn_on_dropped_annotations(
        Console(stderr=True),
        _report(total=7, applied_notes=2, shadowed_notes=[note] * 5),
    )

    err = capsys.readouterr().err
    # Out of the comments attempted (5 dropped + 2 placed), not out of 7.
    assert "5 of 7 comment" in err
    # The reader must be told the findings are not lost, only unplaced.
    assert "still listed in the review report" in err


def test_dropped_edits_are_not_reported_as_merely_unplaced(capsys) -> None:
    """A dropped edit is data loss and must not get the comment reassurance.

    Comments carry findings, and every finding is also in the prepended report,
    so an unplaced comment loses nothing. Edits are in no report — a dropped
    edit vanishes. The single "all findings are still listed" line used to
    cover both, telling the author nothing was lost when a rewrite was.
    """
    from rich.console import Console

    from andamentum.whetstone.cli import _warn_on_dropped_annotations
    from andamentum.whetstone.docx.annotate import Rewrite

    rewrite = Rewrite(
        start=0,
        end=31,
        new_text="…",
        author="a",
        label="the results show a large effect",
    )
    _warn_on_dropped_annotations(
        Console(stderr=True), _report(total=4, conflicted_rewrites=[rewrite])
    )

    err = capsys.readouterr().err
    assert "NOT applied" in err
    assert "overlaps an edit already accepted" in err
    # The edit is named so the author can find what went missing.
    assert "the results show" in err
    # And the comment-only reassurance must NOT appear.
    assert "still listed in the review report" not in err


def test_fully_placed_run_prints_no_annotation_warning(capsys) -> None:
    """No noise when every annotation landed."""
    from rich.console import Console

    from andamentum.whetstone.cli import _warn_on_dropped_annotations

    _warn_on_dropped_annotations(
        Console(stderr=True), _report(total=7, applied_rewrites=4, applied_notes=3)
    )
    assert capsys.readouterr().err == ""
