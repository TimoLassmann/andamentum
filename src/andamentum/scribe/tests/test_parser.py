"""Markdown body parser tests (citation spans + verify markers + inline runs)."""

from andamentum.scribe.parser import (
    extract_citation_keys,
    find_unresolved_markers,
    inline_runs,
)


def test_extract_single_citation():
    assert extract_citation_keys("See [@smith2023].") == ["smith2023"]


def test_extract_multiple_citations():
    text = "Both [@smith2023] and [@jones2024] disagree."
    assert extract_citation_keys(text) == ["smith2023", "jones2024"]


def test_extract_handles_grouped_citations():
    text = "Many studies [@smith2023; @jones2024; @lee2022]."
    assert extract_citation_keys(text) == ["smith2023", "jones2024", "lee2022"]


def test_extract_ignores_email_like_atsigns():
    assert extract_citation_keys("Contact me at [me@example.com].") == []


def test_extract_returns_empty_for_no_citations():
    assert extract_citation_keys("plain text") == []


def test_find_unresolved_markers_verify():
    text = "Foundational work [verify] established the field."
    assert find_unresolved_markers(text) == ["verify"]


def test_find_unresolved_markers_citation_needed():
    text = "Some claim [citation needed]."
    assert find_unresolved_markers(text) == ["citation needed"]


def test_find_unresolved_markers_returns_empty_for_clean_text():
    assert find_unresolved_markers("Clean.") == []


def test_inline_runs_plain_text():
    runs = inline_runs("plain text")
    assert runs == [("plain text", set())]


def test_inline_runs_bold():
    runs = inline_runs("normal **bold** more")
    assert runs == [
        ("normal ", set()),
        ("bold", {"bold"}),
        (" more", set()),
    ]


def test_inline_runs_italic():
    runs = inline_runs("normal *italic* more")
    assert runs == [
        ("normal ", set()),
        ("italic", {"italic"}),
        (" more", set()),
    ]


def test_inline_runs_inline_code():
    runs = inline_runs("call `f(x)` now")
    assert runs == [
        ("call ", set()),
        ("f(x)", {"code"}),
        (" now", set()),
    ]
