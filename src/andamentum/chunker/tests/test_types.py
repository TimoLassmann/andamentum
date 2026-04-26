"""Tests for chunker pydantic types."""

import pytest

from andamentum.chunker.types import (
    ChunkingFailedError,
    ChunkingResult,
    Gap,
    NextUnitResult,
    Unit,
)


def test_next_unit_result_found_minimal():
    r = NextUnitResult(
        found=True,
        title="Intro",
        start_anchor="Multiple sequence alignment is",
        end_anchor="propose a new method.",
        kind="prose",
    )
    assert r.found is True
    assert r.complete is True  # default
    assert r.skip_to == ""


def test_next_unit_result_not_found():
    r = NextUnitResult(found=False, skip_to="end of nav block")
    assert r.found is False
    assert r.title == ""


def test_unit_carries_provenance_and_metadata():
    u = Unit(
        id="u1",
        title="Intro",
        text="Multiple sequence alignment is foundational. We propose a new method.",
        kind="prose",
        source_start=0,
        source_end=70,
        complete=True,
        anchor_match_method="exact",
    )
    assert u.text.startswith("Multiple")
    assert u.source_end - u.source_start == 70


def test_gap_carries_position_and_length():
    g = Gap(source_start=100, source_end=250, text="...skipped content...")
    assert g.length == 150


def test_chunking_result_coverage_math():
    units = [
        Unit(
            id="u1",
            title="t1",
            text="x" * 80,
            kind="prose",
            source_start=0,
            source_end=80,
            complete=True,
            anchor_match_method="exact",
        )
    ]
    gaps = [Gap(source_start=80, source_end=100, text="x" * 20)]
    r = ChunkingResult(
        units=units,
        gaps=gaps,
        total_chars=100,
        model_calls=2,
        retries_used=0,
        windows_processed=1,
    )
    assert r.coverage == pytest.approx(0.8)
    assert r.gap_fraction == pytest.approx(0.2)


def test_chunking_failed_error_carries_diagnostics():
    err = ChunkingFailedError(
        cursor=1234,
        attempted_models=["ollama:gemma:9b", "openai:gpt-4o-mini"],
        last_validator_messages=["anchor 'foo' not found"],
        message="extraction stalled",
    )
    assert err.cursor == 1234
    assert "ollama" in str(err)
