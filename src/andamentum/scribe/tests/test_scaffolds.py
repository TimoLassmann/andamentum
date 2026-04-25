"""Tests for built-in scaffolds."""

import pytest

from andamentum.scribe.api import Document
from andamentum.scribe.scaffolds import SCAFFOLDS


def test_article_scaffold_creates_standard_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t", scaffold="article")
    sections = [s["name"] for s in doc.list_sections()]
    assert sections == [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "References",
    ]


def test_grant_scaffold_creates_standard_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="G", database="t", scaffold="grant")
    sections = [s["name"] for s in doc.list_sections()]
    assert sections == [
        "Specific Aims",
        "Background and Significance",
        "Innovation",
        "Approach",
        "Timeline and Milestones",
        "References",
    ]


def test_scaffold_includes_guide_metadata_on_placeholder_paragraphs(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t", scaffold="article")
    intro_blocks = doc.section("Introduction")
    # heading + placeholder paragraph
    assert intro_blocks[1].type == "paragraph"
    assert "guide" in intro_blocks[1].metadata
    assert "funnel" in intro_blocks[1].metadata["guide"].lower()


def test_unknown_scaffold_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Unknown scaffold"):
        Document.create(title="P", database="t", scaffold="bogus")


def test_scaffolds_constant_is_well_formed():
    for name, sections in SCAFFOLDS.items():
        assert isinstance(name, str)
        assert len(sections) >= 2
        for section_name, guide in sections:
            assert isinstance(section_name, str) and section_name
            assert isinstance(guide, str)
