"""Tests for v3 layer-1 tools (read_section, search_paper).

Pure-Python: builds a small synthetic DocumentModel directly, constructs
a duck-typed RunContext namespace pointing at DocDeps, and exercises the
tools' return values. No LLM, no pydantic-ai agent, no fixtures from
real corpus papers — just shape and behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from pydantic_ai import RunContext

from andamentum.whetstone.v3.model import DocumentModel, Section
from andamentum.whetstone.v3.tools import (
    DocDeps,
    _section_id_at,
    _substring_positions,
    read_section,
    search_paper,
)


# ── fixtures ─────────────────────────────────────────────────────────


def _model() -> DocumentModel:
    """Synthetic three-section paper with predictable char offsets."""
    source = (
        "We propose Adam.\n\n"  # chars 0-17  ("We propose Adam." is 0-15, \n\n is 16-17)
        "It combines AdaGrad and RMSProp.\n\n"  # chars 18-51
        "We prove convergence under bounded gradients.\n"  # chars 52-97
    )
    return DocumentModel(
        source=source,
        sections=[
            Section(
                id="abstract",
                title="Abstract",
                text="We propose Adam.",
                start=0,
                end=16,
            ),
            Section(
                id="1",
                title="Introduction",
                text="It combines AdaGrad and RMSProp.",
                start=18,
                end=50,
            ),
            Section(
                id="2",
                title="Convergence",
                text="We prove convergence under bounded gradients.",
                start=52,
                end=97,
            ),
        ],
    )


def _ctx(model: DocumentModel) -> RunContext[DocDeps]:
    """Duck-typed RunContext stub. From the tool's perspective, RunContext
    is just `deps` access; constructing a real one requires a model and
    other plumbing we don't need for these unit tests. Cast at the
    boundary so the call sites stay clean and pyright-quiet."""
    return cast(RunContext[DocDeps], SimpleNamespace(deps=DocDeps(document_model=model)))


# ── read_section ─────────────────────────────────────────────────────


async def test_read_section_returns_text_for_known_id() -> None:
    text = await read_section(_ctx(_model()), "1")
    assert text == "It combines AdaGrad and RMSProp."


async def test_read_section_returns_error_string_for_unknown_id() -> None:
    result = await read_section(_ctx(_model()), "999")
    assert isinstance(result, str)
    assert "no section with id" in result
    assert "999" in result  # the bad id is echoed so the agent can correct


async def test_read_section_returns_abstract_when_asked() -> None:
    text = await read_section(_ctx(_model()), "abstract")
    assert text == "We propose Adam."


# ── search_paper: substring mode (default) ───────────────────────────


async def test_search_paper_substring_finds_known_term() -> None:
    matches = await search_paper(_ctx(_model()), "AdaGrad")
    assert isinstance(matches, list)
    assert len(matches) == 1
    assert matches[0]["section_id"] == "1"
    assert "AdaGrad" in matches[0]["snippet"]


async def test_search_paper_substring_is_case_insensitive() -> None:
    lower = await search_paper(_ctx(_model()), "adam")
    upper = await search_paper(_ctx(_model()), "ADAM")
    assert isinstance(lower, list) and isinstance(upper, list)
    assert len(lower) == 1
    assert len(upper) == 1
    assert lower[0]["position"] == upper[0]["position"]


async def test_search_paper_substring_returns_empty_when_absent() -> None:
    matches = await search_paper(_ctx(_model()), "transformer")
    assert matches == []


async def test_search_paper_substring_caps_at_max_results() -> None:
    source = "the " * 20
    model = DocumentModel(
        source=source,
        sections=[Section(id="x", title="X", text=source, start=0, end=len(source))],
    )
    matches = await search_paper(_ctx(model), "the", max_results=3)
    assert isinstance(matches, list)
    assert len(matches) == 3


async def test_search_paper_substring_snippet_includes_context() -> None:
    matches = await search_paper(_ctx(_model()), "AdaGrad")
    assert isinstance(matches, list)
    snippet = matches[0]["snippet"]
    # snippet should include surrounding words, not just the matched term
    assert len(snippet) > len("AdaGrad")
    assert ("combines" in snippet) or ("RMSProp" in snippet)


async def test_search_paper_substring_tags_section_id_per_hit() -> None:
    matches = await search_paper(_ctx(_model()), "Adam")
    assert isinstance(matches, list)
    # "Adam" appears in the abstract; should be tagged with section "abstract"
    assert matches[0]["section_id"] == "abstract"


async def test_search_paper_empty_query_returns_empty() -> None:
    """Empty needle would match every position; refuse to be silly."""
    matches = await search_paper(_ctx(_model()), "")
    assert matches == []


# ── search_paper: regex mode ─────────────────────────────────────────


async def test_search_paper_regex_alternation_finds_multiple() -> None:
    matches = await search_paper(_ctx(_model()), r"(adam|adagrad)", regex=True)
    assert isinstance(matches, list)
    # "Adam" in abstract + "AdaGrad" in intro
    assert len(matches) >= 2


async def test_search_paper_regex_character_class() -> None:
    source = "Theorem 1 says X. See Theorem 12. Then Theorem 3.\n"
    model = DocumentModel(
        source=source,
        sections=[Section(id="x", title="X", text=source, start=0, end=len(source))],
    )
    matches = await search_paper(_ctx(model), r"Theorem [0-9]+", regex=True)
    assert isinstance(matches, list)
    assert len(matches) == 3


async def test_search_paper_regex_compile_error_returns_string() -> None:
    result = await search_paper(_ctx(_model()), "foo[", regex=True)
    assert isinstance(result, str)
    assert "invalid regex" in result.lower()


async def test_search_paper_regex_rejects_overlong_pattern() -> None:
    result = await search_paper(_ctx(_model()), "a" * 250, regex=True)
    assert isinstance(result, str)
    assert "too long" in result.lower()


async def test_search_paper_regex_word_boundary_excludes_substrings() -> None:
    """`\\bAdam\\b` should match the standalone word but not 'adamantly'."""
    source = "Adam is mentioned. We are adamantly opposed.\n"
    model = DocumentModel(
        source=source,
        sections=[Section(id="x", title="X", text=source, start=0, end=len(source))],
    )
    matches = await search_paper(_ctx(model), r"\bAdam\b", regex=True)
    assert isinstance(matches, list)
    # Only "Adam" the word should match — not "adamantly"
    assert len(matches) == 1


# ── pure helpers ─────────────────────────────────────────────────────


def test_substring_positions_finds_non_overlapping() -> None:
    """Non-overlapping advancement: 'aaaa' should yield 2 hits, not 3."""
    positions = _substring_positions("aa", "aaaa", max_results=10)
    assert positions == [(0, 2), (2, 4)]


def test_section_id_at_finds_containing_section() -> None:
    sections = [
        Section(id="a", title="A", text="x" * 10, start=0, end=10),
        Section(id="b", title="B", text="y" * 10, start=12, end=22),
    ]
    assert _section_id_at(5, sections) == "a"
    assert _section_id_at(15, sections) == "b"
    # Position between sections (the gap)
    assert _section_id_at(11, sections) == "?"
    # Position past the last section
    assert _section_id_at(30, sections) == "?"
