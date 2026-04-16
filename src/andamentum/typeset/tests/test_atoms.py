"""Tests for andamentum.typeset.atoms validation."""

from __future__ import annotations

import warnings

import pytest

from andamentum.typeset.atoms import (
    ATOM_KINDS,
    validate_atom,
    validate_document,
)


def test_valid_prose() -> None:
    atom = {"kind": "prose", "content": "Hello, world."}
    result = validate_atom(atom, 0)
    assert result["kind"] == "prose"
    assert result["content"] == "Hello, world."


def test_missing_kind_defaults_to_prose() -> None:
    atom = {"content": "No kind given."}
    result = validate_atom(atom, 0)
    assert result["kind"] == "prose"


def test_unknown_kind_falls_back_to_prose() -> None:
    atom = {"kind": "bogus", "content": "Unknown kind."}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validate_atom(atom, 0)
    assert result["kind"] == "prose"
    assert any("bogus" in str(w.message) for w in caught)


def test_missing_required_field_raises() -> None:
    # heading requires "content"
    atom = {"kind": "heading"}
    with pytest.raises(ValueError, match="content") as exc_info:
        validate_atom(atom, 3)
    assert "3" in str(exc_info.value)


def test_invalid_callout_tone_raises() -> None:
    atom = {"kind": "callout", "content": "Watch out!", "tone": "danger"}
    with pytest.raises(ValueError, match="tone"):
        validate_atom(atom, 1)


def test_invalid_items_variant_raises() -> None:
    atom = {"kind": "items", "entries": ["a", "b"], "variant": "diagonal"}
    with pytest.raises(ValueError, match="variant"):
        validate_atom(atom, 2)


def test_all_seven_kinds_are_present() -> None:
    assert len(ATOM_KINDS) == 7
    expected = {"heading", "prose", "callout", "items", "aside", "card", "reference"}
    assert ATOM_KINDS == expected


def test_empty_document() -> None:
    result = validate_document([])
    assert result == []


def test_non_list_raises() -> None:
    with pytest.raises(ValueError, match="list"):
        validate_document("string")  # type: ignore[arg-type]


def test_non_dict_atom_raises() -> None:
    with pytest.raises(ValueError, match="dict"):
        validate_document(["string"])  # type: ignore[arg-type]
