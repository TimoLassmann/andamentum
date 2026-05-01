"""Tests for the Demand object — Phase 0 of the lazy-escalation plan.

This module's only job in Phase 0 is to provide a working Pydantic
model. Phases 1+ wire it into the graph; Phase 0 just establishes the
shape and the canonical constructors so future code can rely on them.

The tests pin:

  1. The two-helper-constructor ergonomics (``Demand.satisfied`` /
     ``Demand.needs``) so future code consistently uses the same
     idiom for the two semantic cases.
  2. The flat-schema contract (3 fields, all top-level, no nesting)
     so small Ollama models will reliably fill the schema in Phases
     1+ when they emit demands.
  3. Round-trip safety via ``model_dump`` / ``model_validate``,
     which Phase 4's graph state may need for serialisation.
"""

from __future__ import annotations

from andamentum.epistemic.demand import Demand


# ── Direct construction ──────────────────────────────────────────────


def test_construct_with_all_fields() -> None:
    """Direct construction works for cases that need to set every
    field explicitly (e.g. demand round-tripped from a serialised
    form)."""
    d = Demand(
        needs_more=True,
        justification="No RCT mortality data in the current evidence pool.",
        target_hint="ClinicalTrials.gov or Cochrane",
    )
    assert d.needs_more is True
    assert "RCT mortality" in d.justification
    assert d.target_hint == "ClinicalTrials.gov or Cochrane"


def test_target_hint_defaults_to_empty_string() -> None:
    """The hint is optional. Generators that can't suggest a target
    leave it blank; the consuming layer decides on its own."""
    d = Demand(needs_more=True, justification="Need more evidence on mechanism.")
    assert d.target_hint == ""


# ── Helper constructors ──────────────────────────────────────────────


def test_satisfied_constructor() -> None:
    d = Demand.satisfied(
        justification="Cochrane review explicitly reports no included studies "
        "had mortality outcomes; question resolved in the negative."
    )
    assert d.needs_more is False
    assert "Cochrane" in d.justification
    assert d.target_hint == ""


def test_satisfied_constructor_without_justification() -> None:
    """Justification is optional on the satisfied path — recommended
    for observability but not enforced. The model permits empty
    justification because some satisfaction paths (e.g. cycle-cap
    fallthrough) genuinely have nothing to add."""
    d = Demand.satisfied()
    assert d.needs_more is False
    assert d.justification == ""


def test_needs_constructor() -> None:
    d = Demand.needs(
        justification="No mechanistic data on intermittent fasting and survival "
        "pathways.",
        target_hint="biology / mechanistic literature",
    )
    assert d.needs_more is True
    assert "mechanistic" in d.justification
    assert d.target_hint == "biology / mechanistic literature"


def test_needs_constructor_without_target_hint() -> None:
    """Hint is optional. The default empty string is correct: an
    empty hint is a real signal ('no specific suggestion'), not a
    missing field."""
    d = Demand.needs(
        justification="Single supportive paper; need triangulating evidence."
    )
    assert d.needs_more is True
    assert d.target_hint == ""


# ── Schema flatness for small-LLM compatibility ──────────────────────


def test_schema_is_three_fields_all_top_level() -> None:
    """Pinned: the schema must remain flat with exactly three top-
    level fields. Small LLMs (Ollama tier) reliably fill flat
    schemas; nested objects, deep enums, or Union types reduce
    fill reliability sharply.

    If a future change adds nesting, this test fires and forces a
    deliberate decision: either revert to flat, or document the
    nesting cost in the plan + this test.
    """
    schema = Demand.model_json_schema()
    properties = schema.get("properties", {})
    assert set(properties.keys()) == {"needs_more", "justification", "target_hint"}, (
        f"Demand schema has unexpected fields: {set(properties.keys())}. "
        "Phase 0 of the lazy-escalation plan committed to a 3-field flat "
        "schema for small-LLM compatibility. Adding fields requires "
        "updating the plan and this test together."
    )

    # Each field must be a primitive type (string or boolean), not a
    # nested object reference.
    for name, prop in properties.items():
        assert "type" in prop, (
            f"Field {name!r} has no top-level 'type' — likely a nested "
            "ref or Union. Small LLMs fill these unreliably; flatten "
            "before merging."
        )
        assert prop["type"] in ("string", "boolean"), (
            f"Field {name!r} has type={prop['type']!r}; only string and "
            "boolean are flat-schema-compatible at this layer."
        )


# ── Round-trip via model_dump / model_validate ───────────────────────


def test_model_dump_round_trip() -> None:
    """The Demand will travel through the graph state and possibly
    through DB metadata. ``model_dump`` → ``model_validate`` must be
    lossless (and identity-equal on key fields)."""
    original = Demand.needs(
        justification="Conflicting findings on observational vs RCT outcomes.",
        target_hint="systematic review or meta-analysis",
    )
    payload = original.model_dump()
    restored = Demand.model_validate(payload)
    assert restored.needs_more == original.needs_more
    assert restored.justification == original.justification
    assert restored.target_hint == original.target_hint


def test_model_dump_omits_no_fields() -> None:
    """All three fields appear in the dump even when at default
    values. Future code that filters on dump output (e.g. for
    diff-rendering a demand chain) shouldn't have to handle missing
    keys."""
    d = Demand.satisfied()
    payload = d.model_dump()
    assert set(payload.keys()) == {"needs_more", "justification", "target_hint"}
