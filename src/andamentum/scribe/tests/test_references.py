"""Reference management and Document.citations() tests."""

import pytest

from andamentum.scribe.api import Document, Paragraph


def test_add_reference_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.add_reference(cite_key="smith2023", bibtex="@article{smith2023, ...}")

    refs = doc.references()
    assert len(refs) == 1
    assert refs[0].cite_key == "smith2023"


def test_add_reference_duplicate_key_raises(monkeypatch, tmp_path):
    import sqlite3

    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.add_reference(cite_key="smith2023")
    with pytest.raises(sqlite3.IntegrityError):
        doc.add_reference(cite_key="smith2023")


def test_citations_returns_keys_used_in_paragraphs(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Paragraph("As shown by [@smith2023] and [@jones2024]."))
    doc.append(Paragraph("Repeated: [@smith2023]."))

    keys = doc.citations()
    assert sorted(keys) == ["jones2024", "smith2023"]
