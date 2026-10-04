"""Tests for andamentum.figures.bundle_glue."""

import pytest

from andamentum.figures.bundle_glue import insert_figure
from andamentum.compose.bundle import Bundle, BundleError, StaleRevisionError

DATA = {"Method": ["A", "B", "C"], "Score": [0.5, 0.7, 0.9]}


def _bundle(tmp_path) -> Bundle:
    b = Bundle.create(tmp_path / "b", title="P")
    b.add_section(title="Introduction", body="Intro body.")
    b.add_section(title="Results", body="Results body.")
    return b


def test_insert_figure_copies_png_and_appends_markdown(tmp_path):
    b = _bundle(tmp_path)
    section = insert_figure(
        b,
        "s02",
        caption="Bench results.",
        label="fig:bench",
        data=DATA,
        kind="bar",
        x="Method",
        y="Score",
    )
    assert (tmp_path / "b" / "figures" / "bench.png").is_file()
    assert section.revision == 2
    assert section.body.endswith("![Bench results.](figures/bench.png){#fig:bench}")
    assert b.section("s01").body == "Intro body."
    assert b.validate() == []


def test_insert_figure_unknown_section_raises_before_rendering(tmp_path):
    b = _bundle(tmp_path)
    with pytest.raises(BundleError):
        insert_figure(
            b,
            "s99",
            caption="x",
            label="x",
            data=DATA,
            kind="bar",
            x="Method",
            y="Score",
        )
    assert list((tmp_path / "b" / "figures").iterdir()) == []


def test_insert_figure_stale_revision(tmp_path):
    b = _bundle(tmp_path)
    with pytest.raises(StaleRevisionError):
        insert_figure(
            b,
            "s02",
            caption="x",
            label="x",
            expected_revision=7,
            data=DATA,
            kind="bar",
            x="Method",
            y="Score",
        )


def test_insert_figure_renders_into_docx(tmp_path):
    from docx import Document as DocxDocument

    b = _bundle(tmp_path)
    insert_figure(
        b,
        "s02",
        caption="c",
        label="fig:z",
        data=DATA,
        kind="bar",
        x="Method",
        y="Score",
    )
    out = b.render(tmp_path / "o.docx", format="docx")
    assert len(DocxDocument(str(out)).inline_shapes) == 1
