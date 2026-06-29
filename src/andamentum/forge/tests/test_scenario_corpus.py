"""Validate the scenario corpus is well-formed.

These checks enforce the structural invariants of the corpus itself; later
phases consume CORPUS as their acceptance fixture.
"""

from __future__ import annotations

from andamentum.forge.tests.scenario_corpus import CORPUS, Scenario

FUNCTION_RUNGS = {"function", "stateful_function"}
REFUSE_RUNGS = {"app", "agent", "service"}
ALL_RUNGS = FUNCTION_RUNGS | REFUSE_RUNGS


def test_corpus_non_empty() -> None:
    assert CORPUS, "CORPUS must contain at least one scenario"


def test_expected_matches_rung() -> None:
    for s in CORPUS:
        if s.rung in FUNCTION_RUNGS:
            assert s.expected == "build", (
                f"function-rung scenario must be 'build': {s.brief!r}"
            )
        else:
            assert s.rung in REFUSE_RUNGS, f"unknown rung {s.rung!r}: {s.brief!r}"
            assert s.expected == "refuse", (
                f"non-function-rung scenario must be 'refuse': {s.brief!r}"
            )


def test_build_iff_function_rung() -> None:
    for s in CORPUS:
        assert (s.expected == "build") == (s.rung in FUNCTION_RUNGS), s.brief
        assert (s.expected == "refuse") == (s.rung in REFUSE_RUNGS), s.brief


def test_refuse_cases_have_reshape() -> None:
    for s in CORPUS:
        if s.expected == "refuse":
            assert s.note.strip(), f"refuse case needs a reshape note: {s.brief!r}"
            assert s.axis.strip(), (
                f"refuse case needs a disqualifying axis: {s.brief!r}"
            )


def test_build_cases_have_no_axis() -> None:
    for s in CORPUS:
        if s.expected == "build":
            assert s.axis == "", f"in-scope scenario must have empty axis: {s.brief!r}"


def test_all_five_rungs_present() -> None:
    present = {s.rung for s in CORPUS}
    assert present == ALL_RUNGS, f"missing rungs: {ALL_RUNGS - present}"


def test_scenarios_are_frozen() -> None:
    s = CORPUS[0]
    assert isinstance(s, Scenario)
    try:
        s.brief = "mutated"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Scenario must be frozen (immutable)")
