"""Drafted bodies as Word paragraphs: blocks, lists, tables, markers, placeholders."""

from __future__ import annotations

from andamentum.compose.harness.draft.body import (
    Cite,
    clean_markers,
    parse_body,
    sentence_spans,
    strip_inline_markdown,
)


def test_blank_lines_split_paragraphs() -> None:
    paragraphs = parse_body("First paragraph.\n\nSecond one,\ncontinued here.")
    assert [p.text for p in paragraphs] == [
        "First paragraph.",
        "Second one, continued here.",
    ]
    assert all(p.list_kind is None for p in paragraphs)


def test_list_items_are_their_own_paragraphs() -> None:
    paragraphs = parse_body("- one\n- two\n\n1. first\n2) second")
    assert [(p.text, p.list_kind, p.list_marker) for p in paragraphs] == [
        ("one", "bullet", "-"),
        ("two", "bullet", "-"),
        ("first", "number", "1."),
        ("second", "number", "2."),
    ]


def test_list_item_continuation_lines_join_the_item() -> None:
    (item,) = parse_body("- a long item\n  that wraps")
    assert item.text == "a long item that wraps"


def test_pipe_table_becomes_one_paragraph_per_row() -> None:
    rows = parse_body("| group | score |\n|---|---|\n| treatment | 53.8 |")
    assert [p.text for p in rows] == ["group | score", "treatment | 53.8"]
    assert all(p.table_row for p in rows)


def test_inline_markdown_is_removed() -> None:
    assert strip_inline_markdown("a **bold** and *slanted* `code` word") == (
        "a bold and slanted code word"
    )
    assert strip_inline_markdown("2 * 3 and 4 * 5") == "2 * 3 and 4 * 5"


def test_marker_before_the_full_stop_moves_ahead_of_it() -> None:
    text, cites = clean_markers("It rose. [[f1]] Then it fell.")
    assert text == "It rose. Then it fell."
    assert cites == (Cite(at=7, fact_ids=("f1",)),)
    assert text[: cites[0].at] == "It rose"


def test_marker_inside_a_sentence_keeps_the_words_apart() -> None:
    text, cites = clean_markers("Alpha [[f1]] beta.")
    assert text == "Alpha beta."
    assert cites == (Cite(at=5, fact_ids=("f1",)),)


def test_adjacent_markers_are_one_cite() -> None:
    text, cites = clean_markers("Both agree [[f1]][[f2]] [[f3]].")
    assert text == "Both agree."
    assert cites == (Cite(at=10, fact_ids=("f1", "f2", "f3")),)


def test_marker_at_the_end_without_punctuation() -> None:
    text, cites = clean_markers("A claim [[f9]]")
    assert text == "A claim"
    assert cites == (Cite(at=7, fact_ids=("f9",)),)


def test_cites_survive_paragraph_stripping() -> None:
    (paragraph,) = parse_body("  - An item [[f2]].")
    assert paragraph.text == "An item."
    assert paragraph.cites == (Cite(at=7, fact_ids=("f2",)),)


def test_to_fill_ranges_and_sentences() -> None:
    (paragraph,) = parse_body("It rose by 5%. [TO FILL: the sample size] Then more.")
    ((start, end),) = paragraph.to_fill()
    assert paragraph.text[start:end] == "[TO FILL: the sample size]"
    assert [paragraph.text[s:e] for s, e in sentence_spans(paragraph.text)] == [
        "It rose by 5%.",
        "[TO FILL: the sample size] Then more.",
    ]


def test_empty_body_has_no_paragraphs() -> None:
    assert parse_body("  \n\n ") == []
