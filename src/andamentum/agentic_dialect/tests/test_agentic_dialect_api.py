"""The public API surface: laws, law lookup, roles, for_role."""

from __future__ import annotations

import pytest

from andamentum.agentic_dialect import for_role, law, laws, roles


def test_laws_are_l1_through_l9_in_order() -> None:
    assert [lw.id for lw in laws()] == [f"L{i}" for i in range(1, 10)]


def test_law_lookup_is_case_insensitive() -> None:
    assert law("l4").id == "L4"
    assert law("L4").name == "Routing is static, declarative, and deterministic"


def test_unknown_law_raises() -> None:
    with pytest.raises(KeyError):
        law("L99")


def test_for_role_worker_carries_only_its_slice() -> None:
    text = for_role("worker")
    assert "L2" in text and "L7" in text and "L8" in text
    assert "L4" not in text  # the worker slice excludes routing
    assert text.startswith("You are writing a worker")


def test_roles_are_listed() -> None:
    assert {"worker", "orchestrator", "state", "agent", "entry", "reviewer"} <= set(
        roles()
    )


def test_unknown_role_raises() -> None:
    with pytest.raises(KeyError):
        for_role("nope")
