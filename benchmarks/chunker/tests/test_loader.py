"""Tests for case loading + anchor resolution."""

import pytest

from benchmarks.chunker.loader import LoadError, load_case


def test_load_case_resolves_anchors_to_offsets(tmp_path):
    src = "# Title\n\nFirst paragraph here.\n\nSecond paragraph next."
    (tmp_path / "x.input.md").write_text(src)
    (tmp_path / "x.truth.json").write_text("""{
        "convention": "paragraph = unit",
        "expected_f1_floor": 0.7,
        "boundary_tolerance_chars": 50,
        "domain": "general",
        "units": [
            {"title": "First", "start_anchor": "First paragraph", "end_anchor": "paragraph here."},
            {"title": "Second", "start_anchor": "Second paragraph", "end_anchor": "paragraph next."}
        ]
    }""")

    case = load_case(tmp_path / "x.truth.json")
    assert case.name == "x"
    assert case.domain == "general"
    assert case.expected_f1_floor == 0.7
    assert len(case.truth.units) == 2
    # Anchors resolved to actual offsets
    assert case.source[
        case.truth.units[0].start_offset : case.truth.units[0].end_offset
    ].startswith("First paragraph")


def test_load_case_raises_for_missing_anchor(tmp_path):
    (tmp_path / "x.input.md").write_text("Hello world.")
    (tmp_path / "x.truth.json").write_text("""{
        "convention": "x",
        "expected_f1_floor": 0.5,
        "boundary_tolerance_chars": 50,
        "domain": "general",
        "units": [
            {"title": "t", "start_anchor": "totally not here", "end_anchor": "world."}
        ]
    }""")
    with pytest.raises(LoadError, match="start_anchor"):
        load_case(tmp_path / "x.truth.json")


def test_load_case_resolves_anchors_in_order(tmp_path):
    """Same anchor appearing twice resolves by document order, like the chunker."""
    src = "Hello world. Hello world. Hello world."
    (tmp_path / "x.input.md").write_text(src)
    (tmp_path / "x.truth.json").write_text("""{
        "convention": "x",
        "expected_f1_floor": 0.5,
        "boundary_tolerance_chars": 50,
        "domain": "general",
        "units": [
            {"title": "1", "start_anchor": "Hello world", "end_anchor": "Hello world."},
            {"title": "2", "start_anchor": "Hello world", "end_anchor": "Hello world."}
        ]
    }""")
    case = load_case(tmp_path / "x.truth.json")
    # Second unit's start should be after first unit's end
    assert case.truth.units[1].start_offset >= case.truth.units[0].end_offset
