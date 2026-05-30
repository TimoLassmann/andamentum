"""Unit tests for the normalised anchor utility (framework-free)."""

from __future__ import annotations

from andamentum.whetstone.docx.anchor import (
    DocIndex,
    normalize_with_map,
)


# ---------------------------------------------------------------------------
# normalize_with_map
# ---------------------------------------------------------------------------


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    norm, idx = normalize_with_map("The   Quick\n\nBrown")
    assert norm == "the quick brown"
    assert len(idx) == len(norm)


def test_normalize_strips_markdown_markers() -> None:
    norm, _ = normalize_with_map("## 1 Introduction")
    assert norm == "1 introduction"  # '#' dropped, whitespace collapsed


def test_normalize_strips_emphasis_and_links() -> None:
    norm, _ = normalize_with_map("see *Figure* [1] and `code`")
    assert "*" not in norm and "[" not in norm and "`" not in norm
    assert norm == "see figure 1 and code"


def test_index_map_points_back_to_original() -> None:
    text = "##  Hello   World"
    norm, idx = normalize_with_map(text)
    assert norm == "hello world"
    # The 'h' of normalized maps to the 'H' in the original.
    assert text[idx[0]] == "H"
    # The 'w' maps to 'W'.
    w_pos = norm.index("w")
    assert text[idx[w_pos]] == "W"


def test_normalize_empty_and_markers_only() -> None:
    assert normalize_with_map("")[0] == ""
    assert normalize_with_map("##  **  ")[0] == ""


# ---------------------------------------------------------------------------
# DocIndex.find — single segment
# ---------------------------------------------------------------------------


def test_find_within_single_run() -> None:
    idx = DocIndex([[("r1", "The methods were significantly robust.")]])
    span = idx.find("significantly robust")
    assert span is not None
    assert span.start_key == "r1"
    assert span.end_key == "r1"
    seg_text = "The methods were significantly robust."
    assert seg_text[span.start_char] == "s"  # start of "significantly"
    assert seg_text[span.end_char] == "t"  # end of "robust"


def test_find_markdown_target_in_single_paragraph() -> None:
    """A markdown-flavoured target finds plain body text in one paragraph."""
    idx = DocIndex([[("r1", "Studying a biological research question is hard.")]])
    span = idx.find("*Studying* a biological research question")
    assert span is not None
    assert span.start_key == "r1"


def test_find_target_spanning_heading_and_body_paragraphs() -> None:
    """The real failure case: ``## Heading\\n\\nBody`` against two paragraphs."""
    idx = DocIndex(
        [
            [("h1", "1 Introduction")],
            [("b1", "Studying a biological question.")],
        ]
    )
    span = idx.find("## 1 Introduction\n\nStudying a biological question")
    assert span is not None
    assert span.start_key == "h1"
    assert span.end_key == "b1"


# ---------------------------------------------------------------------------
# DocIndex.find — cross-run within a paragraph
# ---------------------------------------------------------------------------


def test_find_spans_two_runs_in_one_paragraph() -> None:
    idx = DocIndex(
        [
            [
                ("r1", "the methods were "),
                ("r2", "significantly "),
                ("r3", "robust and clear"),
            ]
        ]
    )
    span = idx.find("significantly robust")
    assert span is not None
    assert span.start_key == "r2"
    assert span.end_key == "r3"
    assert "significantly "[span.start_char] == "s"
    assert "robust and clear"[span.end_char] == "t"  # end of "robust"


def test_find_absent_returns_none() -> None:
    idx = DocIndex([[("r1", "the methods were robust")]])
    assert idx.find("completely different text") is None


def test_paragraph_separator_prevents_false_join() -> None:
    """End of one paragraph and start of the next must not fuse."""
    idx = DocIndex([[("r1", "introduction")], [("r2", "studying")]])
    assert idx.find("introductionstudying") is None
    assert idx.find("introduction studying") is not None


# ---------------------------------------------------------------------------
# DocIndex.closest — diagnostics
# ---------------------------------------------------------------------------


def test_closest_returns_similar_region() -> None:
    idx = DocIndex(
        [
            [("r1", "Studying a biological research question often involves work. ")],
            [("r2", "We measured everything in triplicate.")],
        ]
    )
    score, snippet = idx.closest(
        "## 1 Introduction\n\nStudying a biological research question often"
    )
    assert score > 0.3
    assert "biological research question" in snippet


def test_closest_empty_when_nothing_similar() -> None:
    idx = DocIndex([[("r1", "alpha beta gamma")]])
    score, _snippet = idx.closest("zzzz qqqq")
    # No shared substring of meaning; score low.
    assert score < 0.5


def test_closest_scores_local_window_not_whole_document() -> None:
    """A long target whose region is almost entirely present should score
    HIGH locally — even though it's a tiny fraction of the whole document.
    Guards the global-ratio bug (which scored ~0.01 for exactly this case)."""
    body = (
        "the widget catalogue is released under a permissive licence an "
        "archived release is deposited in a public repository the evaluation "
        "corpus comprising benchmark snapshots for all forty scenarios"
    )
    # A big document so a global ratio would be tiny.
    idx = DocIndex([[("r1", "lorem ipsum " * 200 + body + " dolor sit " * 200)]])
    target = (
        "under a permissive licence; an archived release is deposited in "
        "[a public repository]. The evaluation corpus, comprising benchmark "
        "snapshots for all forty scenarios"
    )
    score, snippet = idx.closest(target)
    assert score > 0.8, f"local score should be high, got {score}"
    assert "public repository" in snippet
