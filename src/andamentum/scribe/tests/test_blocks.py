"""Tests for block append and query."""

from andamentum.scribe.api import Document, Figure, Heading, Paragraph, Table


def test_append_returns_block_id(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    bid = doc.append(Heading("Intro", level=1))
    assert isinstance(bid, str)
    assert len(bid) >= 8


def test_append_increments_position(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Intro", level=1))
    doc.append(Paragraph("First paragraph."))
    doc.append(Paragraph("Second."))

    blocks = doc.query()
    positions = [b.position for b in blocks]
    assert positions == [0, 1, 2]


def test_query_filters_by_type(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Intro", level=1))
    doc.append(Paragraph("Body."))
    doc.append(Figure(path="f.png", caption="C", label="fig:c"))

    paragraphs = doc.query(type="paragraph")
    assert len(paragraphs) == 1
    assert paragraphs[0].content == "Body."


def test_query_returns_blocks_in_position_order(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    doc = Document.create(title="P", database="t")
    for i in range(5):
        doc.append(Paragraph(f"P{i}"))
    blocks = doc.query()
    assert [b.content for b in blocks] == [f"P{i}" for i in range(5)]


def test_factory_heading_carries_level():
    blk = Heading("Methods", level=2)
    assert blk["type"] == "heading"
    assert blk["metadata"]["level"] == 2


def test_factory_figure_carries_metadata():
    blk = Figure(path="x.png", caption="cap", label="fig:x")
    assert blk["type"] == "figure"
    assert blk["metadata"] == {
        "path": "x.png",
        "caption": "cap",
        "label": "fig:x",
        "width_in": None,
    }


def test_factory_figure_with_width():
    blk = Figure(path="x.png", caption="c", label="fig:x", width_in=4.5)
    assert blk["metadata"]["width_in"] == 4.5


def test_factory_table_carries_rows_and_caption():
    blk = Table(
        rows=[["a", "b"], ["1", "2"]],
        header_row=True,
        caption="demo",
        label="tab:demo",
    )
    assert blk["type"] == "table"
    assert blk["metadata"]["rows"] == [["a", "b"], ["1", "2"]]
    assert blk["metadata"]["caption"] == "demo"


def test_factory_heading_rejects_invalid_level():
    import pytest

    with pytest.raises(ValueError):
        Heading("X", level=0)
    with pytest.raises(ValueError):
        Heading("X", level=7)
