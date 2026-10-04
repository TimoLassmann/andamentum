"""The body markdown subset."""

from andamentum.compose.bundle.markdown import (
    ImageBlock,
    ListBlock,
    ParagraphBlock,
    TableBlock,
    blocks_to_markdown,
    parse_blocks,
)


def test_paragraphs_join_lines_and_split_on_blank():
    blocks = parse_blocks("one\ntwo\n\nthree")
    assert blocks == [ParagraphBlock("one two"), ParagraphBlock("three")]


def test_lists_bullet_and_numbered():
    blocks = parse_blocks("- a\n- b\n\n1. x\n2. y")
    assert blocks == [
        ListBlock(ordered=False, items=["a", "b"]),
        ListBlock(ordered=True, items=["x", "y"]),
    ]


def test_list_directly_followed_by_other_kind_without_blank_line():
    blocks = parse_blocks("- a\n1. x\ntext")
    assert [type(b) for b in blocks] == [ListBlock, ListBlock, ParagraphBlock]


def test_table_first_row_is_header_separator_dropped():
    blocks = parse_blocks("| A | B |\n|---|---|\n| 1 | 2 |\n| 3 |")
    assert blocks == [TableBlock([["A", "B"], ["1", "2"], ["3", ""]])]


def test_image_with_and_without_label():
    blocks = parse_blocks("![Cap](figures/a.png){#fig:one}\n\n![No label](b.png)")
    assert blocks == [
        ImageBlock("Cap", "figures/a.png", "one"),
        ImageBlock("No label", "b.png", ""),
    ]


def test_heading_line_is_plain_paragraph():
    assert parse_blocks("# Not a heading") == [ParagraphBlock("# Not a heading")]


def test_italic_marker_is_not_a_list():
    assert parse_blocks("*emphasis* start") == [ParagraphBlock("*emphasis* start")]


def test_roundtrip_to_markdown():
    text = "para\n\n- a\n- b\n\n1. x\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n![C](p.png){#fig:l}"
    assert blocks_to_markdown(parse_blocks(text)) == text
