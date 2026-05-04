"""Unit tests for ``document_store.chunker_adapter``.

Tests the ``Unit → Chunk`` mapping in isolation — no Ollama, no LLM,
no real chunker invocation. Constructs synthetic ``Unit`` records by
hand so the test failure modes are localised to the adapter.
"""

from __future__ import annotations

from andamentum.chunker.types import Unit
from andamentum.document_store.chunker_adapter import (
    _compute_section_paths,
    units_to_chunks,
)


def _unit(text: str, start: int, title: str = "") -> Unit:
    """Build a Unit covering source[start : start + len(text)]."""
    return Unit(
        id="t",
        title=title or text[:20],
        text=text,
        kind="prose",
        source_start=start,
        source_end=start + len(text),
        complete=True,
        anchor_match_method="exact",
    )


def test_section_path_single_h2_section():
    src = "## Introduction\n\nFirst paragraph here."
    units = [_unit(src, 0, title="Introduction")]
    paths = _compute_section_paths(src, units)
    assert paths == ["Introduction"]


def test_section_path_nested_h2_h3():
    src = (
        "## Methods\n\n"
        "Methods intro paragraph.\n\n"
        "### ODE Solver\n\n"
        "ODE solver details paragraph here.\n"
    )
    # Unit covering the H3 sub-section
    h3_offset = src.index("### ODE Solver")
    units = [_unit(src[h3_offset:], h3_offset, title="ODE Solver")]
    paths = _compute_section_paths(src, units)
    assert paths == ["Methods > ODE Solver"]


def test_section_path_h1_then_h2():
    src = (
        "# Paper Title\n\n"
        "## Introduction\n\n"
        "Intro body here.\n"
    )
    intro_offset = src.index("## Introduction")
    units = [_unit(src[intro_offset:], intro_offset, title="Introduction")]
    paths = _compute_section_paths(src, units)
    assert paths == ["Paper Title > Introduction"]


def test_section_path_preamble_before_first_heading_is_empty():
    src = "Preamble before any heading.\n\n## Section A\n\nbody"
    # Unit covering the preamble text starts at offset 0, before any heading
    preamble = "Preamble before any heading."
    units = [_unit(preamble, 0)]
    paths = _compute_section_paths(src, units)
    assert paths == [""]


def test_section_path_no_headings_at_all():
    src = "Plain prose. No headings anywhere in this document."
    units = [_unit(src, 0)]
    paths = _compute_section_paths(src, units)
    assert paths == [""]


def test_section_path_unit_at_h2_boundary_picks_that_section():
    """A unit whose source_start equals a heading offset belongs to that heading."""
    src = "## Intro\n\nA\n\n## Methods\n\nB"
    methods_offset = src.index("## Methods")
    units = [_unit(src[methods_offset:], methods_offset, title="Methods")]
    paths = _compute_section_paths(src, units)
    assert paths == ["Methods"]


def test_units_to_chunks_assigns_sequential_indices():
    src = "## A\n\nbody A\n\n## B\n\nbody B"
    a_off = src.index("## A")
    b_off = src.index("## B")
    units = [
        _unit(src[a_off:b_off].rstrip(), a_off, title="A"),
        _unit(src[b_off:], b_off, title="B"),
    ]
    chunks = units_to_chunks(src, units)
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.section_path for c in chunks] == ["A", "B"]


def test_units_to_chunks_byte_identical_round_trip():
    src = "## Methods\n\nFirst paragraph.\n\n## Results\n\nSecond paragraph."
    m_off = src.index("## Methods")
    r_off = src.index("## Results")
    units = [
        _unit(src[m_off : r_off - 2], m_off, title="Methods"),
        _unit(src[r_off:], r_off, title="Results"),
    ]
    chunks = units_to_chunks(src, units)
    for c in chunks:
        assert src[c.start_char : c.end_char] == c.text


def test_units_to_chunks_empty_units_returns_empty_list():
    chunks = units_to_chunks("any source", [])
    assert chunks == []


def test_section_path_three_level_nesting():
    src = (
        "# Book\n\n"
        "## Chapter 1\n\n"
        "Chapter 1 intro.\n\n"
        "### Section 1.1\n\n"
        "Section 1.1 body here.\n"
    )
    deep_offset = src.index("### Section 1.1")
    units = [_unit(src[deep_offset:], deep_offset, title="Section 1.1")]
    paths = _compute_section_paths(src, units)
    assert paths == ["Book > Chapter 1 > Section 1.1"]
