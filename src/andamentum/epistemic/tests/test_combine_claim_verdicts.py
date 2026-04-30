"""Tests for Phase 4: combine_claim_verdicts and the CombineClaimVerdicts
graph node.

Phase 4 of the multi-seed-claim refactor. The combination logic moved
from decomposed_runner.combine_sub_verdicts (per-PipelineResult input)
to graph/combination.combine_claim_verdicts (per-Claim input). The
graph node CombineClaimVerdicts runs between PromoteSupported and
CheckCompletion, applies the decomposition's combination_rule, and
stashes the result on objective.decomposition["combined_verdict"] for
FreezeSnapshot to promote onto the Snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Objective, Snapshot
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.combination import (
    CombinedVerdict,
    combine_claim_verdicts,
    extract_weights_from_decomposition,
)
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import CombineClaimVerdicts
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


def _make_claim(
    *,
    sub_id: str | None = None,
    verdict: str | None = None,
    confidence: float = 0.0,
    abandoned: bool = False,
    cycle_capped: bool = False,
) -> Claim:
    return Claim(
        objective_id="test-obj",
        statement=f"claim {sub_id or '?'}",
        scope="scope",
        stage=ClaimStage.SUPPORTED,
        sub_investigation_id=sub_id,
        integrated_assessment=verdict,
        integrated_confidence=confidence if verdict else None,
        abandoned=abandoned,
        cycle_capped=cycle_capped,
    )


# ── Pure-function tests ───────────────────────────────────────────────


class TestCombineClaimVerdicts:
    def test_and_returns_min(self) -> None:
        claims = [
            _make_claim(sub_id="A", verdict="supports", confidence=0.8),
            _make_claim(sub_id="B", verdict="supports", confidence=0.6),
            _make_claim(sub_id="C", verdict="contradicts", confidence=0.5),
        ]
        c = combine_claim_verdicts(claims, "AND")
        # supports@0.8→0.9; supports@0.6→0.8; contradicts@0.5→0.25
        # min = 0.25 → contradicts
        assert c.posterior == pytest.approx(0.25)
        assert c.verdict == "contradicts"

    def test_or_returns_max(self) -> None:
        claims = [
            _make_claim(sub_id="A", verdict="contradicts", confidence=0.5),
            _make_claim(sub_id="B", verdict="supports", confidence=0.7),
        ]
        c = combine_claim_verdicts(claims, "OR")
        # contradicts@0.5→0.25; supports@0.7→0.85
        assert c.posterior == pytest.approx(0.85)
        assert c.verdict == "supports"

    def test_weighted_and_uses_weights(self) -> None:
        claims = [
            _make_claim(sub_id="A", verdict="supports", confidence=0.8),
            _make_claim(sub_id="B", verdict="contradicts", confidence=0.4),
        ]
        # supports@0.8 → 0.9; contradicts@0.4 → 0.3
        # weighted: 0.9*3 + 0.3*1 = 3.0; / 4 = 0.75
        c = combine_claim_verdicts(
            claims, "WEIGHTED_AND", weights=[3.0, 1.0]
        )
        assert c.posterior == pytest.approx(0.75)

    def test_union_returns_none_posterior(self) -> None:
        claims = [
            _make_claim(sub_id="A", verdict="supports", confidence=0.8),
            _make_claim(sub_id="B", verdict="contradicts", confidence=0.6),
        ]
        c = combine_claim_verdicts(claims, "UNION")
        assert c.posterior is None
        assert c.verdict == "union"

    def test_excludes_capped_from_aggregation(self) -> None:
        """Phase 2 partial-cap fix at the combiner level: capped claims
        are excluded; the rest aggregate normally."""
        claims = [
            _make_claim(sub_id="A", verdict="supports", confidence=0.8),
            _make_claim(sub_id="B", cycle_capped=True),
            _make_claim(sub_id="C", verdict="supports", confidence=0.7),
        ]
        c = combine_claim_verdicts(claims, "AND")
        # Aggregates A (0.9) and C (0.85) only; B is capped.
        assert c.posterior == pytest.approx(0.85)
        assert c.n_capped == 1

    def test_excludes_abandoned_from_aggregation(self) -> None:
        claims = [
            _make_claim(sub_id="A", verdict="supports", confidence=0.8),
            _make_claim(sub_id="B", abandoned=True),
            _make_claim(sub_id="C", verdict="supports", confidence=0.7),
        ]
        c = combine_claim_verdicts(claims, "AND")
        assert c.posterior == pytest.approx(0.85)
        assert c.n_abandoned == 1

    def test_no_verdict_claims_excluded(self) -> None:
        """Claims without an integration verdict (HYPOTHESIS-stuck or
        no-IBE-run) drop from numeric combination."""
        claims = [
            _make_claim(sub_id="A", verdict="supports", confidence=0.8),
            _make_claim(sub_id="B"),  # No verdict
            _make_claim(sub_id="C", verdict="supports", confidence=0.7),
        ]
        c = combine_claim_verdicts(claims, "AND")
        assert c.n_no_verdict == 1
        # Min of 0.9 and 0.85 = 0.85.
        assert c.posterior == pytest.approx(0.85)

    def test_all_excluded_returns_no_data(self) -> None:
        claims = [
            _make_claim(sub_id="A", abandoned=True),
            _make_claim(sub_id="B", cycle_capped=True),
        ]
        c = combine_claim_verdicts(claims, "AND")
        assert c.posterior is None
        assert c.verdict == "no_data"

    def test_unknown_rule_raises(self) -> None:
        claims = [_make_claim(sub_id="A", verdict="supports", confidence=0.8)]
        with pytest.raises(ValueError, match="Unknown combination_rule"):
            combine_claim_verdicts(claims, "MAJORITY")

    def test_negative_weight_raises(self) -> None:
        claims = [
            _make_claim(sub_id="A", verdict="supports", confidence=0.8),
            _make_claim(sub_id="B", verdict="supports", confidence=0.6),
        ]
        with pytest.raises(ValueError, match="non-negative"):
            combine_claim_verdicts(
                claims, "WEIGHTED_AND", weights=[1.0, -1.0]
            )

    def test_mismatched_weights_length_raises(self) -> None:
        claims = [
            _make_claim(sub_id="A", verdict="supports", confidence=0.8),
            _make_claim(sub_id="B", verdict="supports", confidence=0.6),
        ]
        with pytest.raises(ValueError, match="length"):
            combine_claim_verdicts(claims, "WEIGHTED_AND", weights=[1.0])


# ── extract_weights_from_decomposition ────────────────────────────────


class TestExtractWeights:
    def test_pulls_weights_in_claim_order(self) -> None:
        decomposition = {
            "sub_investigations": [
                {"id": "A", "weight": 1.0},
                {"id": "B", "weight": 2.0},
                {"id": "C", "weight": 0.5},
            ]
        }
        claims = [
            _make_claim(sub_id="A"),
            _make_claim(sub_id="B"),
            _make_claim(sub_id="C"),
        ]
        weights = extract_weights_from_decomposition(decomposition, claims)
        assert weights == [1.0, 2.0, 0.5]

    def test_returns_none_when_unmatched_claim(self) -> None:
        """Defensive: if any claim has a sub_id not in the decomposition,
        fall back to no-weights (caller uses simple mean)."""
        decomposition = {
            "sub_investigations": [
                {"id": "A", "weight": 1.0},
                {"id": "B", "weight": 2.0},
            ]
        }
        claims = [
            _make_claim(sub_id="A"),
            _make_claim(sub_id="B"),
            _make_claim(sub_id="C"),  # Not in decomposition
        ]
        weights = extract_weights_from_decomposition(decomposition, claims)
        assert weights is None

    def test_returns_none_when_no_decomposition(self) -> None:
        claims = [_make_claim(sub_id=None)]
        assert extract_weights_from_decomposition(None, claims) is None
        assert extract_weights_from_decomposition({}, claims) is None


# ── Snapshot.combined_verdict round-trip ──────────────────────────────


class TestSnapshotCombinedVerdict:
    async def test_field_round_trips(self, tmp_path: Path) -> None:
        store = DocumentStore.for_database("snap_round", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent")
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        snapshot = Snapshot(
            objective_id=obj.entity_id,
            claim_ids=["c1", "c2"],
            combined_verdict={
                "posterior": 0.75,
                "verdict": "supports",
                "combination_rule": "WEIGHTED_AND",
                "claim_posteriors": [0.9, 0.7],
                "n_capped": 0,
                "n_no_verdict": 0,
                "n_abandoned": 0,
                "explanation": "weighted mean",
            },
        )
        await repo.save(snapshot)

        loaded = await repo.get("snapshot", snapshot.entity_id)
        assert loaded.combined_verdict is not None
        assert loaded.combined_verdict["posterior"] == 0.75
        assert loaded.combined_verdict["verdict"] == "supports"

    async def test_default_is_none(self, tmp_path: Path) -> None:
        """Snapshots without a combined_verdict (open-research) leave
        the field as None."""
        store = DocumentStore.for_database("snap_default", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent")
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        snapshot = Snapshot(
            objective_id=obj.entity_id, claim_ids=["c1", "c2"]
        )
        await repo.save(snapshot)
        loaded = await repo.get("snapshot", snapshot.entity_id)
        assert loaded.combined_verdict is None


# ── CombineClaimVerdicts graph node ───────────────────────────────────


class _FakeRunContext:
    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps) -> None:
        self.state = state
        self.deps = deps


class TestCombineClaimVerdictsNode:
    async def test_no_decomposition_is_noop(self, tmp_path: Path) -> None:
        """When the Objective has no decomposition (open research), the
        node passes through without writing anything."""
        store = DocumentStore.for_database("combine_noop", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await CombineClaimVerdicts().run(ctx)  # type: ignore[arg-type]

        reloaded = await repo.get("objective", obj.entity_id)
        # decomposition is None, so no combined_verdict written.
        assert reloaded.decomposition is None

    async def test_combines_and_stashes_on_objective(
        self, tmp_path: Path
    ) -> None:
        """With decomposition + matching claims, the node writes the
        CombinedVerdict to objective.decomposition["combined_verdict"]."""
        store = DocumentStore.for_database("combine_stash", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="parent",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "alpha", "rationale": "ra",
                     "weight": 1.0},
                    {"id": "B", "seed_claim": "beta", "rationale": "rb",
                     "weight": 1.0},
                ],
                "combination_rule": "AND",
                "rationale": "both must hold",
            },
            combination_rule="AND",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        # Two claims with verdicts, in decomposition order (A, B).
        for sub_id, conf in (("A", 0.8), ("B", 0.6)):
            c = _make_claim(
                sub_id=sub_id, verdict="supports", confidence=conf
            )
            c.objective_id = obj.entity_id
            await repo.save(c)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await CombineClaimVerdicts().run(ctx)  # type: ignore[arg-type]

        reloaded = await repo.get("objective", obj.entity_id)
        assert reloaded.decomposition is not None
        cv = reloaded.decomposition["combined_verdict"]
        assert cv["combination_rule"] == "AND"
        # min over [0.9 (A), 0.8 (B)] = 0.8
        assert cv["posterior"] == pytest.approx(0.8)
        assert cv["verdict"] == "supports"

    async def test_orders_claims_by_decomposition(
        self, tmp_path: Path
    ) -> None:
        """The node aligns claims with the decomposition order, not
        repo-query order. This ensures weighted_and's weight alignment
        is deterministic regardless of save order."""
        store = DocumentStore.for_database("combine_order", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="parent",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "α", "rationale": "ra",
                     "weight": 3.0},
                    {"id": "B", "seed_claim": "β", "rationale": "rb",
                     "weight": 1.0},
                ],
                "combination_rule": "WEIGHTED_AND",
                "rationale": "weighted",
            },
            combination_rule="WEIGHTED_AND",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        # Save in REVERSE order of decomposition to confirm ordering.
        for sub_id, conf in (("B", 0.4), ("A", 0.8)):
            c = _make_claim(
                sub_id=sub_id, verdict="supports", confidence=conf
            )
            c.objective_id = obj.entity_id
            await repo.save(c)

        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        await CombineClaimVerdicts().run(ctx)  # type: ignore[arg-type]

        reloaded = await repo.get("objective", obj.entity_id)
        assert reloaded.decomposition is not None
        cv = reloaded.decomposition["combined_verdict"]
        # A has weight 3 and posterior 0.9; B has weight 1 and 0.7.
        # weighted = (0.9*3 + 0.7*1) / 4 = 3.4/4 = 0.85.
        assert cv["posterior"] == pytest.approx(0.85)


class TestCombinedVerdictType:
    """Sanity: imported correctly from graph/combination.py."""

    def test_dataclass_has_expected_fields(self) -> None:
        cv = CombinedVerdict(
            posterior=0.5,
            verdict="insufficient",
            combination_rule="AND",
            claim_posteriors=[0.5],
            n_capped=0,
            n_no_verdict=0,
            n_abandoned=0,
            explanation="x",
        )
        assert cv.posterior == 0.5
