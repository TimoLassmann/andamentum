"""Tests for andamentum.figures.scribe_glue."""

import pytest

from andamentum.figures.scribe_glue import insert_figure
from andamentum.scribe.api import Document, Heading, Paragraph


def _seed(monkeypatch, tmp_path) -> Document:
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path / "scribe_db"))
    doc = Document.create(title="P", database="t")
    doc.append(Heading("Introduction", level=1))
    doc.append(Paragraph("Intro body."))
    doc.append(Heading("Results", level=1))
    doc.append(Paragraph("Results body."))
    return doc


def test_insert_figure_renders_file_and_inserts_block(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)

    bid = insert_figure(
        doc,
        "Results",
        output_dir=tmp_path / "figures",
        caption="Bench results.",
        label="fig:bench",
        data={"Method": ["A", "B", "C"], "Score": [0.5, 0.7, 0.9]},
        kind="bar",
        x="Method",
        y="Score",
    )

    assert isinstance(bid, str)
    assert (tmp_path / "figures" / "fig_bench.png").exists()

    results = doc.section("Results")
    fig_blocks = [b for b in results if b.type == "figure"]
    assert len(fig_blocks) == 1
    assert fig_blocks[0].metadata["label"] == "fig:bench"
    assert fig_blocks[0].metadata["caption"] == "Bench results."
    assert "fig_bench.png" in fig_blocks[0].metadata["path"]


def test_insert_figure_creates_output_dir_if_missing(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    nested = tmp_path / "deep" / "nested" / "figs"

    insert_figure(
        doc,
        "Results",
        output_dir=nested,
        caption="x",
        label="fig:x",
        data={"A": [1, 2], "B": [3, 4]},
        kind="bar",
        x="A",
        y="B",
    )
    assert nested.exists()


def test_insert_figure_unknown_section_raises(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    with pytest.raises(KeyError):
        insert_figure(
            doc,
            "NoSuchSection",
            output_dir=tmp_path / "figures",
            caption="x",
            label="fig:x",
            data={"A": [1, 2], "B": [3, 4]},
            kind="bar",
            x="A",
            y="B",
        )


def test_insert_figure_filename_override(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    insert_figure(
        doc,
        "Results",
        output_dir=tmp_path / "figures",
        caption="c",
        label="fig:y",
        filename="custom_name.png",
        data={"A": [1, 2], "B": [3, 4]},
        kind="bar",
        x="A",
        y="B",
    )
    assert (tmp_path / "figures" / "custom_name.png").exists()


def test_insert_figure_appends_png_to_extensionless_filename(monkeypatch, tmp_path):
    doc = _seed(monkeypatch, tmp_path)
    insert_figure(
        doc,
        "Results",
        output_dir=tmp_path / "figures",
        caption="c",
        label="fig:z",
        filename="bare_name",
        data={"A": [1, 2], "B": [3, 4]},
        kind="bar",
        x="A",
        y="B",
    )
    assert (tmp_path / "figures" / "bare_name.png").exists()
