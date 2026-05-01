"""Tests for the per-step work caps from Phase 1 of the efficiency plan.

Each test pins a specific cap so that:
  1. The cap value is documented as a contract, not just a constant.
  2. Future changes to the cap surface as test failures, prompting a
     re-benchmark rather than a silent drift in cost.
  3. The cap is exercised at the actual call site, not just unit-tested
     against the constant — so refactors that move the constant but
     don't apply it in the new location get caught.
"""

from __future__ import annotations

from andamentum.epistemic.operations.integration import _CANDIDATE_IDS
from andamentum.epistemic.operations.verification import (
    MAX_ADVERSARIAL_FRAMINGS,
    MAX_ADVERSARIAL_TEMPLATES,
)


# ── Cap values are the contract ──────────────────────────────────────
#
# An earlier iteration of this plan also capped MAX_EXTRAS_PER_STUB
# at 3 in operations/evidence.py. The Phase 1 benchmark showed that
# cap was too aggressive: with extras capped at 3 per stub on EVERY
# round's extraction, the system systematically lost evidence breadth
# across all rounds (rather than recovering it from later rounds, as
# the plan's open-decision #1 had assumed). All 3 claims abandoned,
# IBE never ran, posterior fell back to the no-data 0.5 fallback.
# The cap was reverted; if revisited, formulate as a per-claim total
# cap rather than a per-stub cap.


def test_adversarial_query_count_is_five() -> None:
    """Adversarial search generates 3 deterministic templates and
    2 LLM-generated framings = 5 queries total per claim. Each query's
    hits are evaluated by an LLM; the cap halves downstream evaluation
    cost vs. the previous 5+3=8 split.
    """
    assert MAX_ADVERSARIAL_TEMPLATES == 3
    assert MAX_ADVERSARIAL_FRAMINGS == 2
    assert MAX_ADVERSARIAL_TEMPLATES + MAX_ADVERSARIAL_FRAMINGS == 5


def test_ibe_candidates_capped_at_three() -> None:
    """The IBE chain enumerates up to 3 candidates per claim. Each
    candidate is then scored on loveliness + likeliness (2 LLM calls
    each), so the cap halves IBE cost vs. the previous K=5.
    """
    assert len(_CANDIDATE_IDS) == 3
    # The IDs themselves are the IBE chain's slot keys; the contract
    # is that they're stable across the chain so each operation's
    # filter logic stays valid.
    assert _CANDIDATE_IDS == ["A", "B", "C"]


def test_max_slot_retries_is_two() -> None:
    """The deep_research per-slot generate→verify retry budget is 2.
    Each retry costs 2 LLM calls (generate + verify), so capping at 2
    bounds wasted work on slots where the first attempt was poor. The
    skip-and-tighten fallback still fires after the budget is exhausted.
    """
    from andamentum.deep_research.nodes import MAX_SLOT_RETRIES

    assert MAX_SLOT_RETRIES == 2


# ── Caps actually fire at the call site ──────────────────────────────


def test_adversarial_framings_list_truncated_to_max() -> None:
    """The framings list is sliced to MAX_ADVERSARIAL_FRAMINGS at the
    call site. If a future refactor moves the constant but forgets to
    apply the slice, this test fails — the constant alone isn't enough.

    Reads the source rather than running the operation (which needs a
    full agent runner). The assertion is structural: the constant and
    the slice both refer to the same number.
    """
    from pathlib import Path

    src = (
        Path(__file__).parent.parent
        / "operations"
        / "verification.py"
    ).read_text()
    # Look for the slice pattern at the framings list literal.
    assert "][:MAX_ADVERSARIAL_FRAMINGS]" in src, (
        "AdversarialSearchOperation no longer slices its framings list "
        "by MAX_ADVERSARIAL_FRAMINGS. The constant exists but isn't "
        "being applied — the cap is silently inert."
    )


