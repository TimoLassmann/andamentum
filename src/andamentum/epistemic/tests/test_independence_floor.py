"""Tests for the independence floor (Reichenbach, Tier A — exact identifiers).

Covers ``count_independent_sources`` (pure) and its integration into
``detect_convergence``: evidence that provably traces to too few distinct
sources cannot count as cross-domain convergence, no matter how the domain
classifier splits it. See harness-backstops item 4.
"""

from __future__ import annotations

from ..convergence_detector import count_independent_sources, detect_convergence
from ..primitives import (
    CausalRole,
    DataSourceType,
    DomainClassification,
    MethodType,
    TemporalApproach,
)


def _mk(eid: str, method: MethodType, source: DataSourceType, causal: CausalRole):
    return DomainClassification(
        evidence_id=eid,
        claim_id="claim-1",
        method_type=method,
        data_source=source,
        temporal=TemporalApproach.CROSS_SECTIONAL,
        causal_role=causal,
        classification_confidence=0.9,
        classification_method="agent",
        classification_notes="test",
    )


# Three deliberately distinct domains → convergence fires without a floor.
_ITEMS = [{"evidence_id": f"ev{i}", "content": f"c{i}"} for i in range(3)]
_CLASSIFICATIONS = [
    _mk(
        "ev0",
        MethodType.EXPERIMENTAL,
        DataSourceType.PRIMARY,
        CausalRole.INTERVENTIONAL,
    ),
    _mk(
        "ev1",
        MethodType.OBSERVATIONAL,
        DataSourceType.SECONDARY,
        CausalRole.PHENOMENOLOGICAL,
    ),
    _mk(
        "ev2",
        MethodType.COMPUTATIONAL,
        DataSourceType.SYNTHETIC,
        CausalRole.MECHANISTIC,
    ),
]


class TestCountIndependentSources:
    def test_shared_key_collapses_to_one(self) -> None:
        assert count_independent_sources(["doi:a", "doi:a", "doi:a"]) == 1

    def test_unkeyed_items_each_count_separately(self) -> None:
        # Absence of a shared exact identifier is not evidence of dependence.
        assert count_independent_sources([None, None, None]) == 3

    def test_mixed_keyed_and_unkeyed(self) -> None:
        # Two share a DOI (→1), one unkeyed (→1) = 2 distinct sources.
        assert count_independent_sources(["doi:a", "doi:a", None]) == 2

    def test_all_distinct(self) -> None:
        assert count_independent_sources(["doi:a", "pmid:b", None]) == 3

    def test_empty(self) -> None:
        assert count_independent_sources([]) == 0


class TestFloorIntegration:
    def test_baseline_convergence_without_keys(self) -> None:
        r = detect_convergence(
            _ITEMS, "claim-1", "obj-1", precomputed_classifications=_CLASSIFICATIONS
        )
        assert r.convergence_detected is True
        assert r.verdict == "CONVERGENT"

    def test_shared_source_downgrades_convergence(self) -> None:
        # Same DOI on all three (same paper via three providers) → not
        # independent corroboration, even across three domain classifications.
        r = detect_convergence(
            _ITEMS,
            "claim-1",
            "obj-1",
            precomputed_classifications=_CLASSIFICATIONS,
            independent_source_keys=["doi:10.1/x", "doi:10.1/x", "doi:10.1/x"],
        )
        assert r.convergence_detected is False
        assert r.verdict == "PARTIAL"
        assert "Independence floor" in r.explanation

    def test_distinct_sources_do_not_trigger_floor(self) -> None:
        # Genuinely distinct sources: the floor must not bite (safe direction —
        # it only downgrades illusory convergence, never real convergence).
        r = detect_convergence(
            _ITEMS,
            "claim-1",
            "obj-1",
            precomputed_classifications=_CLASSIFICATIONS,
            independent_source_keys=["doi:a", "doi:b", "doi:c"],
        )
        assert r.convergence_detected is True
        assert r.verdict == "CONVERGENT"

    def test_unkeyed_evidence_does_not_trigger_floor(self) -> None:
        # No identifiers extracted (web/notes) → each counts as its own source,
        # so the floor stays out of the way (it needs PROOF of shared identity).
        r = detect_convergence(
            _ITEMS,
            "claim-1",
            "obj-1",
            precomputed_classifications=_CLASSIFICATIONS,
            independent_source_keys=[None, None, None],
        )
        assert r.convergence_detected is True
        assert r.verdict == "CONVERGENT"

    def test_two_shared_one_distinct_stays_convergent(self) -> None:
        # 2 distinct sources ≥ min_independent_domains (2) → floor does not fire.
        r = detect_convergence(
            _ITEMS,
            "claim-1",
            "obj-1",
            precomputed_classifications=_CLASSIFICATIONS,
            independent_source_keys=["doi:a", "doi:a", "doi:b"],
        )
        assert r.convergence_detected is True
