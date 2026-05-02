"""Tests pinning the adversarial / IBE / slot-retry budgets.

History:

* An earlier version of this file pinned reduced budgets from
  Phase 1 of the (now-reverted) efficiency plan:
  ``MAX_ADVERSARIAL_TEMPLATES=3``, ``MAX_ADVERSARIAL_FRAMINGS=2``,
  ``_CANDIDATE_IDS=["A","B","C"]``, ``MAX_SLOT_RETRIES=2``.

* Two reverts happened:
  - ``MAX_EXTRAS_PER_STUB=3`` (per-stub extras cap) — reverted because
    it stripped evidence breadth across rounds, causing all-claims-
    abandoned outcomes.
  - The four budgets above — reverted (2026-05-02) because benchmark
    runs showed convergence degradation: claims more often hit cycle
    caps before IBE could fire. Restored to the pre-Phase-1 values.

These tests now pin the restored (pre-Phase-1) values so a future
optimization attempt that re-cuts these budgets without first
proving convergence is preserved fails this test loud.
"""

from __future__ import annotations

from andamentum.epistemic.operations.integration import _CANDIDATE_IDS
from andamentum.epistemic.operations.verification import (
    MAX_ADVERSARIAL_FRAMINGS,
    MAX_ADVERSARIAL_TEMPLATES,
)


def test_adversarial_query_count_is_eight() -> None:
    """Adversarial search generates 5 deterministic templates and
    3 LLM-generated framings = 8 queries total per claim. Reducing
    these (we tried 3+2=5 in the reverted Phase 1 efficiency cut)
    causes convergence degradation: less counter-evidence diversity
    → claims more often hit cycle caps before IBE.
    """
    assert MAX_ADVERSARIAL_TEMPLATES == 5
    assert MAX_ADVERSARIAL_FRAMINGS == 3
    assert MAX_ADVERSARIAL_TEMPLATES + MAX_ADVERSARIAL_FRAMINGS == 8


def test_ibe_candidates_at_five() -> None:
    """IBE chain enumerates up to 5 candidates per claim. The
    reverted Phase 1 cut to K=3 reduced abductive diversity such
    that the IBE chain more often produced wishy-washy verdicts.
    Five gives enough comparative breadth to make the chain's
    "best explanation" judgment meaningful."""
    assert len(_CANDIDATE_IDS) == 5
    assert _CANDIDATE_IDS == ["A", "B", "C", "D", "E"]


def test_max_slot_retries_is_three() -> None:
    """The deep_research per-slot generate→verify retry budget is 3.
    The reverted Phase 1 cut to 2 gave the generator one fewer
    chance to recover from a poor first draft, contributing to
    weaker validated-query pools.
    """
    from andamentum.deep_research.nodes import MAX_SLOT_RETRIES

    assert MAX_SLOT_RETRIES == 3


# ── Caps still applied at the call site ──────────────────────────────


def test_adversarial_framings_list_truncated_to_max() -> None:
    """The framings list is sliced to MAX_ADVERSARIAL_FRAMINGS at
    the call site. If a future refactor moves the constant but
    forgets to apply the slice, this test fails — the constant
    alone isn't enough.

    Reads the source rather than running the operation (which needs
    a full agent runner). The assertion is structural: the slice
    pattern uses the constant.
    """
    from pathlib import Path

    src = (
        Path(__file__).parent.parent / "operations" / "verification.py"
    ).read_text()
    assert "][:MAX_ADVERSARIAL_FRAMINGS]" in src, (
        "AdversarialSearchOperation no longer slices its framings list "
        "by MAX_ADVERSARIAL_FRAMINGS. The constant exists but isn't "
        "being applied — the cap is silently inert."
    )
