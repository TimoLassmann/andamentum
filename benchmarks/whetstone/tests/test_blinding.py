"""Pure-logic tests for judge blinding / de-blinding (no LLM)."""

from __future__ import annotations

from benchmarks.whetstone.adjudicate import (
    _deblind_bucket,
    _deblind_text,
    _deblind_verdict,
    _mapping,
    _reblind_bucket_label,
)


def test_mapping_is_seeded_and_deterministic() -> None:
    assert _mapping("paper-x", 1, swap=False) == _mapping("paper-x", 1, swap=False)


def test_swap_flips_the_mapping() -> None:
    base = _mapping("paper-x", 1, swap=False)
    swapped = _mapping("paper-x", 1, swap=True)
    assert base["system_1"] != swapped["system_1"]
    assert base["system_2"] != swapped["system_2"]


def test_deblind_bucket_follows_mapping() -> None:
    m = {"system_1": "A", "system_2": "B"}  # system_1 = whetstone
    assert _deblind_bucket("system_1_only", m) == "a_only"
    assert _deblind_bucket("system_2_only", m) == "b_only"
    assert _deblind_bucket("both", m) == "both"
    # swapped mapping reverses it
    ms = {"system_1": "B", "system_2": "A"}
    assert _deblind_bucket("system_1_only", ms) == "b_only"


def test_deblind_verdict_follows_mapping() -> None:
    m = {"system_1": "A", "system_2": "B"}
    assert _deblind_verdict("system_1", m) == "whetstone"
    assert _deblind_verdict("system_2", m) == "whole-doc"
    assert _deblind_verdict("comparable", m) == "comparable"


def test_pure_position_bias_is_caught_as_inconsistent() -> None:
    """If the judge ALWAYS picks 'system_1' (pure position bias), the two
    order-swapped runs de-blind to opposite systems → flagged inconsistent."""
    m0 = _mapping("p", 1, swap=False)
    m1 = _mapping("p", 1, swap=True)
    v0 = _deblind_verdict("system_1", m0)
    v1 = _deblind_verdict("system_1", m1)
    assert v0 != v1  # one is whetstone, the other whole-doc → disagreement


def test_genuine_preference_is_consistent() -> None:
    """A judge that truly prefers arm A picks the system A maps to in each run,
    so both de-blind to 'whetstone' → consistent."""
    m0 = _mapping("p", 1, swap=False)
    m1 = _mapping("p", 1, swap=True)
    # In each run, pick whichever system label currently maps to arm A.
    pick0 = "system_1" if m0["system_1"] == "A" else "system_2"
    pick1 = "system_1" if m1["system_1"] == "A" else "system_2"
    assert _deblind_verdict(pick0, m0) == _deblind_verdict(pick1, m1) == "whetstone"


def test_deblind_text_replaces_system_labels() -> None:
    m = {"system_1": "A", "system_2": "B"}
    out = _deblind_text("System 1 caught more than system 2 did.", m)
    assert "System 1" not in out and "system 2" not in out
    assert "whetstone" in out and "whole-document" in out


def test_reblind_bucket_label() -> None:
    m = {"system_1": "A", "system_2": "B"}
    assert _reblind_bucket_label("a_only", m) == "System 1 only"
    assert _reblind_bucket_label("b_only", m) == "System 2 only"
    assert _reblind_bucket_label("both", m) == "both systems"
