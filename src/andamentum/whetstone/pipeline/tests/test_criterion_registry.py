"""The criterion registry holds one object per question set — no copies.

The guarantee under test: every criterion any consumer can reach is a
module-level constant in ``criteria.py``. Document-type sets select from the
registry; reviewer profiles will select from the same registry. A copy-paste
that produced a second `Criterion` with the same intent — and, in time,
divergent wording — fails here rather than shipping two questions that a reader
of the report cannot tell apart.

Identity (``is``), not equality, because equality is exactly what a fresh copy
would still satisfy on the day it was made.
"""

from __future__ import annotations

import pytest

from andamentum.whetstone.v3 import criteria as registry
from andamentum.whetstone.v3.criteria import Criterion, criterion_set_for

DOCUMENT_TYPES = (
    "academic",
    "external_communication",
    "essay",
    "tutorial",
    "creative",
    "general",
)

# The six shipped document-type sets, by the name the module gives them.
TYPE_SETS = {
    "academic": registry.SPECS,
    "external_communication": registry.EXTERNAL_COMMS,
    "essay": registry.ESSAY,
    "tutorial": registry.TUTORIAL,
    "creative": registry.CREATIVE,
    "general": registry.GENERAL,
}


def _registry_constants() -> dict[int, str]:
    """Every module-level `Criterion` constant, keyed by object identity."""
    return {
        id(value): name
        for name, value in vars(registry).items()
        if isinstance(value, Criterion) and name.isupper()
    }


def test_registry_constants_are_upper_snake_case() -> None:
    """A `Criterion` bound at module level is part of the public registry, so
    it carries a registry name — not a lowercase working variable."""
    stray = [
        name
        for name, value in vars(registry).items()
        if isinstance(value, Criterion) and not name.isupper()
    ]
    assert stray == [], f"module-level Criterion not named as a constant: {stray}"


@pytest.mark.parametrize("document_type", DOCUMENT_TYPES)
def test_every_criterion_in_a_type_set_is_a_registry_constant(
    document_type: str,
) -> None:
    constants = _registry_constants()
    for criterion in TYPE_SETS[document_type]:
        assert id(criterion) in constants, (
            f"{document_type!r} set contains a Criterion named "
            f"{criterion.name!r} that is not a module-level constant in "
            "criteria.py — it is a literal or a copy. Bind it as a constant "
            "and reference that."
        )


@pytest.mark.parametrize("document_type", (*DOCUMENT_TYPES, "unknown-type", ""))
def test_criterion_set_for_returns_registry_constants(document_type: str) -> None:
    """The public selector copies the *list* but never the criteria in it."""
    constants = _registry_constants()
    selected = criterion_set_for(document_type)
    assert selected, "a document type must select at least one criterion"
    for criterion in selected:
        assert id(criterion) in constants


@pytest.mark.parametrize("document_type", DOCUMENT_TYPES)
def test_criterion_set_for_copies_the_list_not_the_criteria(
    document_type: str,
) -> None:
    """A caller may append to what it is handed without editing the package."""
    shipped = TYPE_SETS[document_type]
    selected = criterion_set_for(document_type)
    assert selected is not shipped
    assert all(a is b for a, b in zip(selected, shipped, strict=True))


def test_criterion_is_frozen() -> None:
    """Without this the identity guarantee is worth nothing: whoever holds a
    reference to a registry constant could edit its questions in place and
    change every later run in the process."""
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        registry.ACADEMIC_STORY.name = "Something else"


def test_profile_criteria_are_in_the_registry_and_in_no_type_set() -> None:
    """The criteria written for reviewer profiles are registry members, so a
    profile can select them by reference — and are in none of the six document
    -type sets, so no existing run changes because they exist."""
    profile_criteria = [
        registry.EVIDENCE_SUPPORT,
        registry.SCOPE_OF_CLAIMS,
        registry.INTERNAL_CONSISTENCY,
        registry.REPRODUCIBILITY,
        registry.MOTIVATED_CHOICES,
        registry.EVALUATION_ADEQUACY,
        registry.PROMISE_AND_DELIVERY,
        registry.CONTRIBUTION,
        registry.READER_PATH,
        registry.ASSUMED_KNOWLEDGE,
        registry.STRONGEST_OBJECTION,
        registry.MISSING_COMPARISON,
    ]
    constants = _registry_constants()
    shipped = {id(c) for shipped_set in TYPE_SETS.values() for c in shipped_set}

    for criterion in profile_criteria:
        assert id(criterion) in constants
        assert criterion.questions, f"{criterion.name!r} has no questions"
        assert id(criterion) not in shipped, (
            f"{criterion.name!r} was added to a document-type set; that "
            "changes findings on every run of that type."
        )


def test_profile_criteria_do_not_declare_the_inert_gists_facet() -> None:
    """``gists`` gates nothing — section gists reach every criterion prompt
    unconditionally. New criteria omit it rather than imply a projection rule
    that does not exist. Shipped criteria keep theirs; changing them would
    change shipped data for no gain."""
    constants = _registry_constants()
    shipped = {id(c) for shipped_set in TYPE_SETS.values() for c in shipped_set}
    for object_id, name in constants.items():
        if object_id in shipped:
            continue
        criterion = getattr(registry, name)
        assert "gists" not in criterion.facets, (
            f"{name} declares the inert 'gists' facet"
        )


def test_walking_criteria_carry_the_walk_instruction_first() -> None:
    """The criteria that need the document in reading order say so in one
    shared wording, as their first question."""
    walking = [
        registry.REPRODUCIBILITY,
        registry.MOTIVATED_CHOICES,
        registry.PROMISE_AND_DELIVERY,
        registry.READER_PATH,
        registry.ASSUMED_KNOWLEDGE,
    ]
    first = {c.questions[0] for c in walking}
    assert len(first) == 1, f"the walk instruction has drifted: {first}"
    assert "read_section" in first.pop()
