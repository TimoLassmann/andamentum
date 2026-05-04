"""Tests for Objective entity-level invariants.

The seed-mode invariant: ``claim_to_verify`` and ``decomposition`` are
mutually exclusive. ``CreateClaims`` (graph/nodes.py) branches on
``claim_to_verify`` first, so silently allowing both would discard the
decomposition. The Pydantic validator on ``Objective`` refuses the bad
state at construction time so the precedence rule is documented in the
error rather than buried in graph routing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from andamentum.epistemic.entities.decomposition import (
    Decomposition,
    SubInvestigation,
)
from andamentum.epistemic.entities.objective import Objective


def test_claim_to_verify_alone_is_valid() -> None:
    """Single-seed mode (claim_to_verify only) constructs cleanly."""
    obj = Objective(
        entity_id="o1",
        objective_id="o1",
        description="Aspirin reduces MI risk in healthy adults.",
        claim_to_verify="Aspirin reduces MI risk in healthy adults.",
    )
    assert obj.claim_to_verify is not None
    assert obj.decomposition is None


def test_decomposition_alone_is_valid() -> None:
    """Multi-seed mode (decomposition only) constructs cleanly."""
    decomp = Decomposition(
        sub_investigations=[
            SubInvestigation(id="s1", seed_claim="X is true.", rationale="r"),
        ],
        combination_rule="AND",
    )
    obj = Objective(
        entity_id="o1",
        objective_id="o1",
        description="Does X hold under conditions Y?",
        decomposition=decomp,
    )
    assert obj.claim_to_verify is None
    assert obj.decomposition is not None


def test_neither_is_valid() -> None:
    """Open-research mode (neither set) constructs cleanly."""
    obj = Objective(
        entity_id="o1",
        objective_id="o1",
        description="What do we know about X?",
    )
    assert obj.claim_to_verify is None
    assert obj.decomposition is None


def test_both_set_raises() -> None:
    """Both claim_to_verify and decomposition is the precedence footgun.

    Pydantic should refuse with a message that names the precedence
    rule so the next consumer doesn't have to read CreateClaims source
    to figure out why their decomposition didn't fire.
    """
    decomp = Decomposition(
        sub_investigations=[
            SubInvestigation(id="s1", seed_claim="X is true.", rationale="r"),
        ],
        combination_rule="AND",
    )
    with pytest.raises(ValidationError) as exc_info:
        Objective(
            entity_id="o1",
            objective_id="o1",
            description="Aspirin reduces MI risk.",
            claim_to_verify="Aspirin reduces MI risk.",
            decomposition=decomp,
        )
    msg = str(exc_info.value)
    # The error must name the precedence rule explicitly.
    assert "claim_to_verify" in msg
    assert "decomposition" in msg
    assert "mutually exclusive" in msg or "silently" in msg
