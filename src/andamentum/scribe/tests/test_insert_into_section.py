"""Tests for Document.insert_into_section."""

import pytest

from andamentum.scribe.api import Document, Figure, Heading, Paragraph


def _seed(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Intro", level=1))
    doc.append(Paragraph("Intro body."))
    doc.append(Heading("Methods", level=1))
    doc.append(Paragraph("Methods body."))
    doc.append(Heading("Results", level=1))
    doc.append(Paragraph("Results body."))
    return doc


def test_insert_into_section_appends_at_end(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    bid = doc.insert_into_section("Methods", Paragraph("Extra methods text."))
    assert isinstance(bid, str)

    methods = doc.section("Methods")
    # heading + original paragraph + new paragraph
    assert [b.type for b in methods] == ["heading", "paragraph", "paragraph"]
    assert methods[-1].content == "Extra methods text."


def test_insert_into_section_preserves_following_sections(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    doc.insert_into_section("Intro", Paragraph("Added."))
    names = [s["name"] for s in doc.list_sections()]
    assert names == ["Intro", "Methods", "Results"]
    # Methods still has its original content
    methods = doc.section("Methods")
    assert any(b.content == "Methods body." for b in methods)


def test_insert_into_section_at_start(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    doc.insert_into_section("Methods", Paragraph("First."), position="start")
    methods = doc.section("Methods")
    # heading + new paragraph + original paragraph
    assert [b.type for b in methods] == ["heading", "paragraph", "paragraph"]
    assert methods[1].content == "First."


def test_insert_into_section_unknown_raises(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    with pytest.raises(KeyError, match="Nope"):
        doc.insert_into_section("Nope", Paragraph("x"))


def test_insert_into_section_invalid_position_raises(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="position"):
        doc.insert_into_section("Methods", Paragraph("x"), position="middle")


def test_insert_into_section_works_with_figure(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    bid = doc.insert_into_section(
        "Results",
        Figure(path="x.png", caption="cap", label="fig:x"),
    )
    assert isinstance(bid, str)
    blocks = doc.section("Results")
    assert any(
        b.type == "figure" and b.metadata.get("label") == "fig:x" for b in blocks
    )
