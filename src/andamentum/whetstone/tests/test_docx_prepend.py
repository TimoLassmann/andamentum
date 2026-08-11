"""Tests for ``DocxEditor.prepend_review_section`` markdown parsing.

Exercises the markdown→Word styling cascade: headings, lists,
blockquotes, horizontal rules, and the trailing page break. We assert
on the OOXML that comes out (style names + structural elements) rather
than rendering a real .docx — that keeps the test fast and avoids
needing Word to verify by eye.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _editor_for_blank_doc(tmp_path: Path):
    """Build a PatchDocxEditor over a fresh empty .docx file."""
    from andamentum.whetstone.docx.patch_editor import PatchDocxEditor

    src = tmp_path / "blank.docx"
    Document().save(str(src))
    return PatchDocxEditor(str(src), author="Whetstone Test")


def _prepended_body_paragraphs(editor):
    """Return the list of `<w:p>` elements that were prepended."""
    tree = editor.trees[editor._doc_path]
    body = tree.getroot().find(".//w:body", namespaces=NS)
    return body.findall("w:p", namespaces=NS)


def _style_name(p_elem) -> str | None:
    """Return the pStyle val attribute, or None if no style is set."""
    pStyle = p_elem.find("./w:pPr/w:pStyle", namespaces=NS)
    if pStyle is None:
        return None
    return pStyle.get(f"{{{NS['w']}}}val")


def _paragraph_text(p_elem) -> str:
    return "".join(t.text or "" for t in p_elem.xpath(".//w:t", namespaces=NS))


def _last_paragraph_has_page_break(paragraphs) -> bool:
    """The trailing page break is in the last prepended paragraph."""
    last = paragraphs[-1]
    return last.find(".//w:br[@w:type='page']", namespaces=NS) is not None


# ── Headings ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "markdown_prefix,expected_style",
    [
        ("# ", "Heading1"),
        ("## ", "Heading2"),
        ("### ", "Heading3"),
        ("#### ", "Heading4"),
        ("##### ", "Heading5"),
        ("###### ", "Heading6"),
    ],
)
def test_heading_levels(tmp_path, markdown_prefix, expected_style):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section(f"{markdown_prefix}Title text")
    paragraphs = _prepended_body_paragraphs(editor)
    # First paragraph is the one we wrote; last is the page break.
    assert _style_name(paragraphs[0]) == expected_style
    assert "Title text" in _paragraph_text(paragraphs[0])


def test_heading_supports_inline_bold(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("## Title with **bold** in it")
    paragraphs = _prepended_body_paragraphs(editor)
    p = paragraphs[0]
    assert _style_name(p) == "Heading2"
    # The bold portion lives in a run with <w:rPr><w:b/></w:rPr>
    bold_run_text = [t.text for t in p.xpath(".//w:r[w:rPr/w:b]/w:t", namespaces=NS)]
    assert "bold" in bold_run_text


# ── Bullet lists ───────────────────────────────────────────────────────


def test_bullet_dash(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("- first\n- second\n- third")
    paragraphs = _prepended_body_paragraphs(editor)
    items = paragraphs[:3]
    assert all(_style_name(p) == "ListBullet" for p in items)
    assert _paragraph_text(items[0]) == "first"
    assert _paragraph_text(items[1]) == "second"
    assert _paragraph_text(items[2]) == "third"


def test_bullet_star(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("* alpha\n* beta")
    paragraphs = _prepended_body_paragraphs(editor)
    assert _style_name(paragraphs[0]) == "ListBullet"
    assert _style_name(paragraphs[1]) == "ListBullet"


def test_bullet_with_inline_bold(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("- item with **emphasis** inside")
    paragraphs = _prepended_body_paragraphs(editor)
    p = paragraphs[0]
    assert _style_name(p) == "ListBullet"
    bold_run_text = [t.text for t in p.xpath(".//w:r[w:rPr/w:b]/w:t", namespaces=NS)]
    assert "emphasis" in bold_run_text


def test_double_star_not_treated_as_bullet(tmp_path):
    """``**Whole-line bold**`` should NOT be parsed as a bullet."""
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("**Bold paragraph**")
    paragraphs = _prepended_body_paragraphs(editor)
    assert _style_name(paragraphs[0]) is None  # no list style
    assert _paragraph_text(paragraphs[0]) == "Bold paragraph"


# ── Numbered lists ─────────────────────────────────────────────────────


def test_numbered_single_digit(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("1. first\n2. second\n3. third")
    paragraphs = _prepended_body_paragraphs(editor)
    items = paragraphs[:3]
    assert all(_style_name(p) == "ListNumber" for p in items)
    assert _paragraph_text(items[0]) == "first"
    assert _paragraph_text(items[1]) == "second"


def test_numbered_multi_digit(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("12. twelfth\n100. hundredth")
    paragraphs = _prepended_body_paragraphs(editor)
    assert _style_name(paragraphs[0]) == "ListNumber"
    assert _paragraph_text(paragraphs[0]) == "twelfth"
    assert _style_name(paragraphs[1]) == "ListNumber"
    assert _paragraph_text(paragraphs[1]) == "hundredth"


# ── Blockquotes ────────────────────────────────────────────────────────


def test_blockquote(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("> Recommendation: Minor Revisions")
    paragraphs = _prepended_body_paragraphs(editor)
    assert _style_name(paragraphs[0]) == "Quote"
    assert "Minor Revisions" in _paragraph_text(paragraphs[0])


def test_blockquote_with_inline_bold(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("> **Recommendation:** Minor Revisions")
    paragraphs = _prepended_body_paragraphs(editor)
    p = paragraphs[0]
    assert _style_name(p) == "Quote"
    bold_run_text = [t.text for t in p.xpath(".//w:r[w:rPr/w:b]/w:t", namespaces=NS)]
    assert "Recommendation:" in bold_run_text


# ── Horizontal rules ───────────────────────────────────────────────────


def test_horizontal_rule(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("---")
    paragraphs = _prepended_body_paragraphs(editor)
    # The horizontal-rule paragraph has a bottom border in pPr
    p = paragraphs[0]
    bottom = p.find("./w:pPr/w:pBdr/w:bottom", namespaces=NS)
    assert bottom is not None
    assert bottom.get(f"{{{NS['w']}}}val") == "single"


# ── Page break separation ──────────────────────────────────────────────


def test_trailing_page_break_always_added(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("# Title\n\nSome prose.")
    paragraphs = _prepended_body_paragraphs(editor)
    # Last prepended paragraph carries the page break
    assert _last_paragraph_has_page_break(paragraphs)


def test_page_break_is_after_review_content(tmp_path):
    """Original document body lives AFTER the page break — verify
    structural ordering even if we can't open Word here."""
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("- one\n- two")
    paragraphs = _prepended_body_paragraphs(editor)
    # 2 list items + page break paragraph = 3 prepended; the existing
    # blank doc had one empty paragraph that's now last.
    assert _style_name(paragraphs[0]) == "ListBullet"
    assert _style_name(paragraphs[1]) == "ListBullet"
    # The third should contain the page break
    page_break = paragraphs[2].find(".//w:br[@w:type='page']", namespaces=NS)
    assert page_break is not None


# ── Mixed content (real-world shape) ───────────────────────────────────


def test_mixed_content_panel_synthesis_shape(tmp_path):
    """The shape v2's panel synthesis emits should produce the right
    cascade of Word styles."""
    editor = _editor_for_blank_doc(tmp_path)
    md = """\
## Panel Synthesis

> **Recommendation: Minor Revisions** (confidence: high) — average score **8.0/10**

The contribution is sound; novelty framing needs tightening.

### Consensus strengths

- clear writing
- sound methodology
- well-cited

### Key decision factors

- sound rigor
- clarity"""
    editor.prepend_review_section(md)
    paragraphs = _prepended_body_paragraphs(editor)

    styles = [_style_name(p) for p in paragraphs]
    assert "Heading2" in styles  # ## Panel Synthesis
    assert "Heading3" in styles  # ### Consensus strengths / Key decision factors
    assert "Quote" in styles  # > Recommendation: ...
    assert styles.count("ListBullet") >= 5  # 3 strengths + 2 factors

    # Trailing page break
    assert _last_paragraph_has_page_break(paragraphs)


# ── Markdown tables ────────────────────────────────────────────────────


def _prepended_tables(editor):
    """Return the `<w:tbl>` elements that were prepended into the body."""
    tree = editor.trees[editor._doc_path]
    body = tree.getroot().find(".//w:body", namespaces=NS)
    return body.findall("w:tbl", namespaces=NS)


def _table_cell_texts(tbl_elem) -> list[list[str]]:
    """Return the table as a list of rows, each a list of cell strings."""
    rows = []
    for tr in tbl_elem.findall("w:tr", namespaces=NS):
        rows.append(
            [
                "".join(t.text or "" for t in tc.xpath(".//w:t", namespaces=NS))
                for tc in tr.findall("w:tc", namespaces=NS)
            ]
        )
    return rows


COVERAGE_TABLE = """\
| Check | Findings | Severity | Ran |
| --- | --- | --- | --- |
| Sample size | 2 | major | yes |
| Blinding | 0 | none | yes |"""


def test_markdown_table_becomes_word_table(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section(COVERAGE_TABLE)

    tables = _prepended_tables(editor)
    assert len(tables) == 1

    rows = _table_cell_texts(tables[0])
    # Header + 2 body rows, 4 columns each — the separator row is dropped.
    assert len(rows) == 3
    assert all(len(row) == 4 for row in rows)
    assert rows[0] == ["Check", "Findings", "Severity", "Ran"]
    assert rows[1] == ["Sample size", "2", "major", "yes"]
    assert rows[2] == ["Blinding", "0", "none", "yes"]

    # Nothing leaked into the paragraph stream as literal pipe text.
    paragraph_text = " ".join(
        _paragraph_text(p) for p in _prepended_body_paragraphs(editor)
    )
    assert "|" not in paragraph_text


def test_markdown_table_header_row_is_bold(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section(COVERAGE_TABLE)

    tbl = _prepended_tables(editor)[0]
    header_tr, first_body_tr = tbl.findall("w:tr", namespaces=NS)[:2]

    bold_header = [
        t.text for t in header_tr.xpath(".//w:r[w:rPr/w:b]/w:t", namespaces=NS)
    ]
    assert bold_header == ["Check", "Findings", "Severity", "Ran"]
    # Body rows stay unbolded.
    assert first_body_tr.xpath(".//w:r[w:rPr/w:b]/w:t", namespaces=NS) == []


def test_markdown_table_cell_gets_inline_markdown(tmp_path):
    """A `**major**` in a cell must render as formatting, not literally."""
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section(
        "| Check | Severity |\n| --- | --- |\n| Sample size | **major** |"
    )

    tbl = _prepended_tables(editor)[0]
    body_tr = tbl.findall("w:tr", namespaces=NS)[1]
    bold_text = [t.text for t in body_tr.xpath(".//w:r[w:rPr/w:b]/w:t", namespaces=NS)]
    assert bold_text == ["major"]
    assert "**" not in "".join(_table_cell_texts(tbl)[1])


def test_ragged_table_falls_back_to_paragraphs(tmp_path, monkeypatch):
    """A row with the wrong cell count must not break the render.

    The warning is asserted by intercepting the module logger rather than
    via ``caplog``: another test in this package attaches its own handler
    to the root logger, and the record never reaches caplog's.
    """
    from andamentum.whetstone.docx import low_level

    warnings: list[str] = []
    monkeypatch.setattr(
        low_level.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    ragged = "| Check | Severity |\n| --- | --- |\n| Sample size | major | extra |"

    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section(ragged)

    assert _prepended_tables(editor) == []
    paragraphs = _prepended_body_paragraphs(editor)
    assert _paragraph_text(paragraphs[0]) == "| Check | Severity |"
    assert _paragraph_text(paragraphs[2]) == "| Sample size | major | extra |"
    assert any("ragged table" in message for message in warnings)


def test_pipe_lines_without_separator_row_stay_paragraphs(tmp_path):
    """Pipe-wrapped prose is not a table — no separator row, no `<w:tbl>`."""
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("| not really a table |")

    assert _prepended_tables(editor) == []
    assert _paragraph_text(_prepended_body_paragraphs(editor)[0]) == (
        "| not really a table |"
    )


def test_escaped_pipe_in_cell_does_not_add_a_column(tmp_path):
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section(
        "| Check | Note |\n| --- | --- |\n| Power | a \\| b |"
    )

    tbl = _prepended_tables(editor)[0]
    assert _table_cell_texts(tbl)[1] == ["Power", "a | b"]


def test_table_between_prose_keeps_document_order(tmp_path):
    """Review mode's report block is unaffected: prose still surrounds
    the table in body order, and the page break stays last."""
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section(
        "## Coverage\n\n| Check | Ran |\n| --- | --- |\n| Blinding | yes |\n\nDone."
    )

    tree = editor.trees[editor._doc_path]
    body = tree.getroot().find(".//w:body", namespaces=NS)
    tags = [child.tag.split("}")[1] for child in body]
    assert "tbl" in tags
    # Heading and the trailing prose sit either side of the table.
    assert tags.index("tbl") > 0
    assert tags.index("tbl") < len(tags) - 1
    assert _last_paragraph_has_page_break(_prepended_body_paragraphs(editor))


def test_empty_lines_preserve_spacing(tmp_path):
    """Blank input lines remain blank paragraphs (vertical spacing)."""
    editor = _editor_for_blank_doc(tmp_path)
    editor.prepend_review_section("# Title\n\n\nNew paragraph after gap")
    paragraphs = _prepended_body_paragraphs(editor)
    # Title + empty + empty + paragraph + page break + ...
    assert _style_name(paragraphs[0]) == "Heading1"
    # Two empty paragraphs follow
    assert _paragraph_text(paragraphs[1]) == ""
    assert _paragraph_text(paragraphs[2]) == ""
    assert "New paragraph after gap" in _paragraph_text(paragraphs[3])
