"""Tests for block → typeset atom conversion."""

from andamentum.scribe.api import Document, Figure, Heading, Paragraph, Table
from andamentum.scribe.render_typeset import to_typeset_atoms


def test_heading_becomes_heading_atom(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(Heading("Intro", level=2))

    atoms = to_typeset_atoms(doc)
    assert len(atoms) == 1
    assert atoms[0]["kind"] == "heading"
    assert atoms[0]["content"] == "Intro"
    assert atoms[0]["level"] == 2


def test_paragraph_becomes_prose_atom(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(Paragraph("Hello *world*."))

    atoms = to_typeset_atoms(doc)
    assert atoms[0]["kind"] == "prose"
    assert atoms[0]["content"] == "Hello *world*."


def test_figure_becomes_card_atom_with_caption(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(Figure(path="x.png", caption="An overview.", label="fig:x"))

    atoms = to_typeset_atoms(doc)
    assert atoms[0]["kind"] == "card"
    assert "An overview." in atoms[0]["content"]
    assert "x.png" in atoms[0]["content"]


def test_table_becomes_prose_atom_with_markdown_table(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(
        Table(
            rows=[["Col A", "Col B"], ["1", "2"]],
            header_row=True,
            caption="demo",
            label="tab:demo",
        )
    )

    atoms = to_typeset_atoms(doc)
    assert atoms[0]["kind"] == "prose"
    # Markdown table syntax — pipe-delimited
    assert "| Col A | Col B |" in atoms[0]["content"]
    assert "| 1 | 2 |" in atoms[0]["content"]


def test_atoms_validate_against_typeset_validator(monkeypatch, tmp_path):
    from andamentum.typeset.atoms import validate_document

    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="T", database="t")
    doc.append(Heading("H", level=1))
    doc.append(Paragraph("P"))

    atoms = to_typeset_atoms(doc)
    # Cast to the broader Mapping type expected by typeset's validator.
    validated = validate_document(list(atoms))  # raises if invalid
    assert len(validated) == 2
