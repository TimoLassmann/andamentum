"""MarkdownAdapter over a compose bundle (synthetic content)."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.compose.bundle import Bundle
from andamentum.compose.harness.kernel.adapter import DocumentAdapter
from andamentum.compose.harness.kernel.ops import (
    Bind,
    CommentOp,
    Delete,
    Insert,
    InsertParagraph,
    Replace,
    Transaction,
)
from andamentum.compose.harness.md import MarkdownAdapter


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    path = tmp_path / "apollo"
    bundle = Bundle.create(
        path, title="Apollo Labs report", metadata={"project_id": "proj-1"}
    )
    bundle.add_section(
        title="Intro", body="Apollo Labs tested SaltLog.\n\nIt ran 12 trials."
    )
    bundle.add_section(title="Results", body="Latency fell by 30 percent.")
    bundle.add_section(title="Empty")
    return path


def _txn(*ops) -> Transaction:
    return Transaction(ops=list(ops), author="tester", reason="r", source="test")


def test_is_a_document_adapter() -> None:
    adapter: DocumentAdapter = MarkdownAdapter()
    assert adapter is not None


def test_view_from_sections(bundle_dir: Path) -> None:
    view = MarkdownAdapter().open(bundle_dir)
    assert view.project_id == "proj-1"
    assert [(p.id, p.text) for p in view.paragraphs] == [
        ("s01:h", "Intro"),
        ("s01:0", "Apollo Labs tested SaltLog."),
        ("s01:1", "It ran 12 trials."),
        ("s02:h", "Results"),
        ("s02:0", "Latency fell by 30 percent."),
        ("s03:h", "Empty"),
    ]
    assert [p.index for p in view.paragraphs] == list(range(6))
    assert (
        view.paragraphs[0].heading_level == 1
        and view.paragraphs[1].heading_level is None
    )
    assert view.sections() == [
        ("Intro", ["s01:0", "s01:1"]),
        ("Results", ["s02:0"]),
        ("Empty", []),
    ]
    assert MarkdownAdapter().open(bundle_dir) == view  # ids stable across re-opens


def test_replace_insert_delete(bundle_dir: Path, tmp_path: Path) -> None:
    adapter = MarkdownAdapter()
    view = adapter.open(bundle_dir)
    out = tmp_path / "out"
    result = adapter.apply(
        view,
        _txn(
            Replace(paragraph_id="s02:0", start=16, end=18, text="45", expect="30"),
            Insert(paragraph_id="s01:1", at=0, text="Overall, "),
            Delete(paragraph_id="s01:0", start=0, end=7, expect="Apollo "),
        ),
        output=out,
    )
    assert result.applied == 3 and result.refused == []
    assert result.untouched.ok and result.untouched.changed_outside == []
    texts = [p.text for p in adapter.open(out).paragraphs]
    assert texts == [
        "Intro",
        "Labs tested SaltLog.",
        "Overall, It ran 12 trials.",
        "Results",
        "Latency fell by 45 percent.",
        "Empty",
    ]
    assert adapter.open(bundle_dir) == view  # input untouched


def test_expect_mismatch_refused_not_guessed(bundle_dir: Path, tmp_path: Path) -> None:
    adapter = MarkdownAdapter()
    view = adapter.open(bundle_dir)
    result = adapter.apply(
        view,
        _txn(
            Replace(paragraph_id="s02:0", start=0, end=3, text="X", expect="Lat"),
            Replace(paragraph_id="s02:0", start=0, end=3, text="X", expect="Wrong"[:3]),
        ),
        output=tmp_path / "out",
    )
    assert result.applied == 1 and len(result.refused) == 1
    assert "expected" in result.refused[0]


def test_insert_paragraph_after_block_and_after_heading(
    bundle_dir: Path, tmp_path: Path
) -> None:
    adapter = MarkdownAdapter()
    view = adapter.open(bundle_dir)
    result = adapter.apply(
        view,
        _txn(
            InsertParagraph(after_paragraph_id="s01:0", text="A new middle paragraph."),
            InsertParagraph(
                after_paragraph_id="s03:h", text="Body for the empty section."
            ),
        ),
        output=tmp_path / "out",
    )
    assert result.applied == 2 and result.untouched.ok
    assert [p.text for p in adapter.open(tmp_path / "out").paragraphs] == [
        "Intro",
        "Apollo Labs tested SaltLog.",
        "A new middle paragraph.",
        "It ran 12 trials.",
        "Results",
        "Latency fell by 30 percent.",
        "Empty",
        "Body for the empty section.",
    ]


def test_heading_replace_renames_section(bundle_dir: Path, tmp_path: Path) -> None:
    adapter = MarkdownAdapter()
    view = adapter.open(bundle_dir)
    adapter.apply(
        view,
        _txn(
            Replace(
                paragraph_id="s02:h", start=0, end=7, text="Findings", expect="Results"
            )
        ),
        output=tmp_path / "out",
    )
    out = Bundle.open(tmp_path / "out")
    assert out.section("s02").title == "Findings"
    assert out.section("s02").revision == 2


def test_unsafe_edits_are_refused(bundle_dir: Path, tmp_path: Path) -> None:
    adapter = MarkdownAdapter()
    view = adapter.open(bundle_dir)
    result = adapter.apply(
        view,
        _txn(
            Insert(paragraph_id="s02:0", at=0, text="a\n\nb"),
            Replace(
                paragraph_id="s02:h",
                start=0,
                end=7,
                text="Two\nlines",
                expect="Results",
            ),
            Delete(
                paragraph_id="s02:0",
                start=0,
                end=27,
                expect="Latency fell by 30 percent.",
            ),
            Insert(paragraph_id="nope", at=0, text="x"),
            Insert(paragraph_id="s02:0", at=999, text="x"),
            InsertParagraph(after_paragraph_id="s02:0", text="x", style="Heading 2"),
        ),
        output=tmp_path / "out",
    )
    assert result.applied == 0 and len(result.refused) == 6
    assert [p.text for p in adapter.open(tmp_path / "out").paragraphs] == [
        p.text for p in view.paragraphs
    ]


def test_unsupported_ops_raise(bundle_dir: Path, tmp_path: Path) -> None:
    adapter = MarkdownAdapter()
    view = adapter.open(bundle_dir)
    for op in (
        CommentOp(paragraph_id="s02:0", start=0, end=3, text="c"),
        Bind(paragraph_id="s02:0", start=0, end=3, anchor_id="a", kind="span"),
    ):
        with pytest.raises(NotImplementedError):
            adapter.apply(view, _txn(op), output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_stale_view_and_existing_output(bundle_dir: Path, tmp_path: Path) -> None:
    adapter = MarkdownAdapter()
    view = adapter.open(bundle_dir)
    bundle = Bundle.open(bundle_dir)
    section = bundle.section("s02")
    bundle.write_section("s02", body="Changed.", expected_revision=section.revision)
    with pytest.raises(ValueError, match="changed since"):
        adapter.apply(
            view,
            _txn(Insert(paragraph_id="s01:0", at=0, text="x")),
            output=tmp_path / "o1",
        )
    fresh = adapter.open(bundle_dir)
    (tmp_path / "o2").mkdir()
    (tmp_path / "o2" / "f").write_text("x")
    with pytest.raises(FileExistsError):
        adapter.apply(
            fresh,
            _txn(Insert(paragraph_id="s01:0", at=0, text="x")),
            output=tmp_path / "o2",
        )


def test_untouched_section_files_are_byte_identical(
    bundle_dir: Path, tmp_path: Path
) -> None:
    adapter = MarkdownAdapter()
    view = adapter.open(bundle_dir)
    adapter.apply(
        view,
        _txn(Insert(paragraph_id="s02:0", at=0, text="So: ")),
        output=tmp_path / "out",
    )
    for name in ("s01.md", "s03.md"):
        assert (bundle_dir / "sections" / name).read_bytes() == (
            tmp_path / "out" / "sections" / name
        ).read_bytes()
