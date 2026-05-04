"""Tests for Phase 3: routing forces verificatory in seed/multi-seed mode.

Bug context: when the parent Objective's classifier output (e.g.
"explanatory" for a declarative SciFact claim) cascaded into
RunVerification's routing-profile lookup, convergence track was demoted
to SECONDARY, A2's termination signal didn't fire, and the
Scrutinize↔Resolve loop relied on the cycle cap rather than convergence
to terminate.

Fix: ``Objective.is_verification_task()`` returns True when
``claim_to_verify`` or ``decomposition`` is set. RunVerification reads
this and forces ``question_type = "verificatory"`` for routing-profile
lookup, regardless of the LLM's classification of the parent's text.
"""

from __future__ import annotations

import pytest

from andamentum.epistemic.entities import Objective


class TestObjectiveIsVerificationTask:
    def test_no_seed_no_decomposition_returns_false(self) -> None:
        """A bare Objective (open research) is NOT a verification task."""
        obj = Objective(description="What's known about X?")
        assert obj.is_verification_task() is False

    def test_claim_to_verify_set_returns_true(self) -> None:
        """Single-seed mode is verification."""
        obj = Objective(description="dummy", claim_to_verify="X is true")
        assert obj.is_verification_task() is True

    def test_decomposition_with_subs_returns_true(self) -> None:
        """Multi-seed mode is verification."""
        obj = Objective(
            description="parent",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "alpha", "rationale": "ra"},
                    {"id": "B", "seed_claim": "beta", "rationale": "rb"},
                ],
                "combination_rule": "AND",
                "rationale": "all must hold",
            },
        )
        assert obj.is_verification_task() is True

    def test_decomposition_with_empty_subs_returns_false(self) -> None:
        """Defensive: a decomposition shell with no sub-investigations is
        NOT a verification task. Shouldn't happen in production but the
        guard prevents accidental verificatory routing on a degenerate
        decomposition."""
        obj = Objective(
            description="parent",
            decomposition={
                "sub_investigations": [],
                "combination_rule": "AND",
                "rationale": "empty",
            },
        )
        assert obj.is_verification_task() is False

    def test_both_claim_and_decomposition_rejected(self) -> None:
        """Construction-time invariant: claim_to_verify and decomposition
        are mutually exclusive. Earlier behaviour silently picked single-
        claim mode and discarded the decomposition; the validator on
        Objective now refuses the bad state at construction time so the
        precedence rule is documented in the error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="mutually exclusive"):
            Objective(
                description="parent",
                claim_to_verify="X is true",
                decomposition={
                    "sub_investigations": [
                        {"id": "A", "seed_claim": "alpha", "rationale": "ra"},
                    ],
                    "combination_rule": "AND",
                    "rationale": "ignored",
                },
            )


class TestRoutingOverrideSemantics:
    """The integration test: when the Objective is a verification task,
    routing reads as verificatory regardless of its question_type. We
    test this at the routing-profile level since RunVerification's
    integration is straightforward (read the Objective, branch on
    is_verification_task)."""

    def test_explanatory_parent_with_decomposition_routes_verificatory(
        self,
    ) -> None:
        """The case 54 scenario: parent classified as explanatory, has
        decomposition. Routing should pick verificatory profile."""
        obj = Objective(
            description="AMPK activation increases fibrosis",
            question_type="explanatory",
            decomposition={
                "sub_investigations": [
                    {
                        "id": "A",
                        "seed_claim": "AMPK has cytokine effects",
                        "rationale": "mechanism",
                    },
                ],
                "combination_rule": "AND",
                "rationale": "verify mechanism",
            },
        )
        # Mirror RunVerification's choice logic. If is_verification_task,
        # the routing profile key is "verificatory" regardless of qt.
        if obj.is_verification_task():
            routing_qt = "verificatory"
        else:
            routing_qt = obj.question_type or "verificatory"
        assert routing_qt == "verificatory"

    def test_explanatory_parent_without_decomposition_routes_explanatory(
        self,
    ) -> None:
        """Sanity: when Objective is genuinely an open-research
        explanatory inquiry (no claim_to_verify, no decomposition), the
        classifier's verdict is honored."""
        obj = Objective(
            description="Why does X happen?",
            question_type="explanatory",
        )
        if obj.is_verification_task():
            routing_qt = "verificatory"
        else:
            routing_qt = obj.question_type or "verificatory"
        assert routing_qt == "explanatory"

    def test_comparative_parent_with_decomposition_routes_verificatory(
        self,
    ) -> None:
        """Comparative parent decomposed into binary sub-claims: each
        child is verifying a specific claim, so routing should be
        verificatory even though the parent is comparative."""
        obj = Objective(
            description="Is A better than B?",
            question_type="comparative",
            decomposition={
                "sub_investigations": [
                    {
                        "id": "A",
                        "seed_claim": "A is more effective",
                        "rationale": "efficacy",
                    },
                    {
                        "id": "B",
                        "seed_claim": "A has fewer side effects",
                        "rationale": "safety",
                    },
                ],
                "combination_rule": "WEIGHTED_AND",
                "rationale": "verify each criterion",
            },
        )
        if obj.is_verification_task():
            routing_qt = "verificatory"
        else:
            routing_qt = obj.question_type or "verificatory"
        assert routing_qt == "verificatory"
