"""Tests for Document.create and Document.open."""

import pytest

from andamentum.scribe.api import Document


def test_create_returns_document_with_id_and_title(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="My Paper", database="test")
    assert doc.title == "My Paper"
    assert isinstance(doc.id, str)
    assert len(doc.id) >= 8  # uuid hex prefix


def test_open_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="Original", database="test")
    doc_id = doc.id

    reopened = Document.open(doc_id, database="test")
    assert reopened.title == "Original"
    assert reopened.id == doc_id


def test_open_unknown_id_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    Document.create(title="exists", database="test")  # ensures DB file exists
    with pytest.raises(KeyError, match="not found"):
        Document.open("does-not-exist", database="test")


def test_create_records_template_when_provided(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="test", template="nature.docx")
    reopened = Document.open(doc.id, database="test")
    assert reopened.template == "nature.docx"
