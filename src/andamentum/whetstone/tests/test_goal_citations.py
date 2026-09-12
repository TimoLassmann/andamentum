"""Tests for the deterministic reference-list consistency check.

Pure functions, no LLM, no I/O — every case here is a plain string in,
report out.
"""

from __future__ import annotations

from andamentum.whetstone.goal.citations import (
    ReferenceMismatch,
    check_reference_consistency,
    _expand_numeric_marker,
)
from andamentum.whetstone.v3.model import Section


def _section(id: str, title: str, text: str, start: int) -> Section:
    return Section(id=id, title=title, text=text, start=start, end=start + len(text))


def test_expand_numeric_marker_single() -> None:
    assert _expand_numeric_marker("[12]") == {12}


def test_expand_numeric_marker_range() -> None:
    assert _expand_numeric_marker("[3-5]") == {3, 4, 5}


def test_expand_numeric_marker_list() -> None:
    assert _expand_numeric_marker("[12, 13]") == {12, 13}


def test_expand_numeric_marker_mixed() -> None:
    assert _expand_numeric_marker("[3-5, 8]") == {3, 4, 5, 8}


def test_no_reference_section_found() -> None:
    text = "Body text with a [1] marker and no list at all."
    sections = [_section("s1", "Introduction", text, 0)]
    report = check_reference_consistency(text, sections)
    assert report.style == "no_reference_section_found"
    assert report.mismatches == []
    assert not report.satisfied


def test_author_year_only_is_none_detected() -> None:
    body = "As shown (Smith et al., 2020), the effect holds."
    ref_text = "References\nSmith J. (2020). A paper."
    text = body + "\n" + ref_text
    sections = [
        _section("s1", "Body", body, 0),
        _section("s2", "References", ref_text, len(body) + 1),
    ]
    report = check_reference_consistency(text, sections)
    assert report.style == "none_detected"
    assert not report.satisfied


def test_numeric_bracket_fully_consistent() -> None:
    body = "First claim [1]. Second claim [2, 3]."
    ref_text = "References\n[1] Smith 2020.\n[2] Jones 2019.\n[3] Lee 2021."
    text = body + "\n" + ref_text
    sections = [
        _section("s1", "Body", body, 0),
        _section("s2", "References", ref_text, len(body) + 1),
    ]
    report = check_reference_consistency(text, sections)
    assert report.style == "numeric_bracket"
    assert report.mismatches == []
    assert report.satisfied


def test_numeric_bracket_cited_not_listed() -> None:
    body = "A claim [1] and another [2]."
    ref_text = "References\n[1] Smith 2020."
    text = body + "\n" + ref_text
    sections = [
        _section("s1", "Body", body, 0),
        _section("s2", "References", ref_text, len(body) + 1),
    ]
    report = check_reference_consistency(text, sections)
    assert not report.satisfied
    assert report.mismatches == [
        ReferenceMismatch(marker="[2]", kind="cited_not_listed")
    ]


def test_numeric_bracket_listed_not_cited() -> None:
    body = "Only claim [1] appears."
    ref_text = "References\n[1] Smith 2020.\n[2] Jones 2019, never cited."
    text = body + "\n" + ref_text
    sections = [
        _section("s1", "Body", body, 0),
        _section("s2", "References", ref_text, len(body) + 1),
    ]
    report = check_reference_consistency(text, sections)
    assert not report.satisfied
    assert [m.kind for m in report.mismatches] == ["listed_not_cited"]
    assert report.mismatches[0].marker == "[2]"


def test_pandoc_key_fully_consistent() -> None:
    body = "As shown [@smith2020] and [@jones2019]."
    ref_text = "References\n[@smith2020] Smith 2020.\n[@jones2019] Jones 2019."
    text = body + "\n" + ref_text
    sections = [
        _section("s1", "Body", body, 0),
        _section("s2", "References", ref_text, len(body) + 1),
    ]
    report = check_reference_consistency(text, sections)
    assert report.style == "pandoc_key"
    assert report.satisfied


def test_pandoc_key_cited_not_listed() -> None:
    body = "As shown [@smith2020] and [@missing2021]."
    ref_text = "References\n[@smith2020] Smith 2020."
    text = body + "\n" + ref_text
    sections = [
        _section("s1", "Body", body, 0),
        _section("s2", "References", ref_text, len(body) + 1),
    ]
    report = check_reference_consistency(text, sections)
    assert not report.satisfied
    assert report.mismatches[0].marker == "[@missing2021]"
    assert report.mismatches[0].kind == "cited_not_listed"


def test_reference_section_matched_case_insensitively() -> None:
    body = "A claim [1]."
    ref_text = "BIBLIOGRAPHY\n[1] Smith 2020."
    text = body + "\n" + ref_text
    sections = [
        _section("s1", "Body", body, 0),
        _section("s2", "Bibliography", ref_text, len(body) + 1),
    ]
    report = check_reference_consistency(text, sections)
    assert report.style == "numeric_bracket"
    assert report.satisfied
