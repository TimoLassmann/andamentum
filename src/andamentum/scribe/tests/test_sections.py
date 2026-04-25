"""Tests for section query and replacement."""

import pytest

from andamentum.scribe.api import Document, Heading, Paragraph


def _seed(monkeypatch, tmp_path) -> Document:
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Introduction", level=1))
    doc.append(Paragraph("Intro body 1."))
    doc.append(Paragraph("Intro body 2."))
    doc.append(Heading("Methods", level=1))
    doc.append(Paragraph("Methods body."))
    doc.append(Heading("Sub-method", level=2))
    doc.append(Paragraph("Sub body."))
    doc.append(Heading("Results", level=1))
    doc.append(Paragraph("Results body."))
    return doc


def test_list_sections_returns_top_level_headings_in_order(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    sections = doc.list_sections()
    names = [s["name"] for s in sections]
    assert names == ["Introduction", "Methods", "Results"]


def test_list_sections_reports_block_and_word_counts(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    sections = doc.list_sections()
    intro = next(s for s in sections if s["name"] == "Introduction")
    # 2 paragraph blocks under "Introduction"
    assert intro["block_count"] == 2
    # 4 words: "Intro body 1." + "Intro body 2." → "Intro body 1 Intro body 2" → 6
    assert intro["word_count"] == 6


def test_section_returns_blocks_under_heading(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    blocks = doc.section("Methods")
    types = [b.type for b in blocks]
    # Methods heading + paragraph + sub-heading + sub-paragraph
    assert types == ["heading", "paragraph", "heading", "paragraph"]
    assert blocks[0].content == "Methods"


def test_section_unknown_name_raises(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    with pytest.raises(KeyError, match="Nope"):
        doc.section("Nope")


def test_replace_section_swaps_body_blocks(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    doc.replace_section(
        "Introduction",
        "First new paragraph.\n\nSecond new paragraph.",
        reason="rewrite",
    )

    blocks = doc.section("Introduction")
    # Heading preserved + 2 new paragraphs
    assert [b.type for b in blocks] == ["heading", "paragraph", "paragraph"]
    assert blocks[1].content == "First new paragraph."
    assert blocks[2].content == "Second new paragraph."


def test_replace_section_preserves_following_sections(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    doc.replace_section("Introduction", "New intro.")
    names = [s["name"] for s in doc.list_sections()]
    assert names == ["Introduction", "Methods", "Results"]


def test_replace_section_writes_revisions_for_each_removed_block(monkeypatch, tmp_path):
    from andamentum.scribe.database import open_db

    doc = _seed(monkeypatch, tmp_path)
    doc.replace_section("Introduction", "New intro body.", reason="rewrite")

    with open_db("t") as conn:
        rows = conn.execute(
            "SELECT reason FROM scribe_revisions WHERE reason = ?",
            ("rewrite",),
        ).fetchall()
    # Two original paragraph blocks removed → two revision rows logging the deletion
    assert len(rows) == 2
