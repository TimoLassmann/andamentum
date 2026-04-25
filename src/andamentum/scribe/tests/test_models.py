"""Pydantic model tests."""

import pytest

from andamentum.scribe.models import (
    Block,
    BlockType,
    Reference,
    StaleRevisionError,
    ValidationIssue,
)


def test_block_paragraph_minimal():
    b = Block(id="b1", doc_id="d1", type="paragraph", content="Hello.", position=0)
    assert b.revision == 1
    assert b.metadata == {}
    assert b.parent_id is None


def test_block_heading_carries_level_in_metadata():
    b = Block(
        id="b2",
        doc_id="d1",
        type="heading",
        content="Introduction",
        position=0,
        metadata={"level": 1},
    )
    assert b.metadata["level"] == 1


def test_block_figure_carries_path_caption_label():
    b = Block(
        id="b3",
        doc_id="d1",
        type="figure",
        content="",
        position=0,
        metadata={"path": "fig1.png", "caption": "Overview", "label": "fig:overview"},
    )
    assert b.metadata["label"] == "fig:overview"


def test_block_table_carries_rows():
    b = Block(
        id="b4",
        doc_id="d1",
        type="table",
        content="",
        position=0,
        metadata={
            "rows": [["a", "b"], ["1", "2"]],
            "header_row": True,
            "caption": "Demo",
            "label": "tab:demo",
        },
    )
    assert b.metadata["rows"][0] == ["a", "b"]


def test_block_rejects_unknown_type():
    with pytest.raises(ValueError):
        Block(id="bx", doc_id="d1", type="bogus", content="", position=0)  # type: ignore[arg-type]


def test_reference_minimal():
    r = Reference(id="r1", doc_id="d1", cite_key="smith2023")
    assert r.bibtex_entry is None


def test_validation_issue_severity_constraint():
    ValidationIssue(severity="error", message="missing", location="b1")
    with pytest.raises(ValueError):
        ValidationIssue(severity="catastrophic", message="x", location="b1")  # type: ignore[arg-type]


def test_stale_revision_error_carries_context():
    err = StaleRevisionError(block_id="b1", expected=1, actual=3)
    assert "b1" in str(err)
    assert err.expected == 1
    assert err.actual == 3


def test_block_type_literal_values():
    expected = {"paragraph", "heading", "figure", "table"}
    assert set(BlockType.__args__) == expected
