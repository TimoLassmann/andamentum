"""Tests for evidence deduplication module."""

import pytest
import numpy as np
from unittest.mock import patch, AsyncMock

from ..dedup import deduplicate_evidence, _build_cluster, EvidenceCluster


def _make_embeddings(
    n_clusters: int, n_per_cluster: int, dim: int = 10, noise: float = 0.05
):
    """Generate synthetic embeddings with known cluster structure.

    Each cluster has a random centroid. Members are centroid + small noise.
    Returns (embeddings_matrix, expected_labels).
    """
    rng = np.random.RandomState(42)
    embeddings = []
    labels = []
    for c in range(n_clusters):
        centroid = rng.randn(dim)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        for _ in range(n_per_cluster):
            member = centroid + rng.randn(dim) * noise
            member = member / np.linalg.norm(member)
            embeddings.append(member)
            labels.append(c)
    return np.array(embeddings), labels


class TestBuildCluster:
    def test_singleton(self):
        emb = np.array([[1.0, 0.0, 0.0]])
        cluster = _build_cluster(emb, [0])
        assert cluster.medoid_index == 0
        assert cluster.representative_indices == [0]
        assert cluster.member_indices == [0]
        assert cluster.count == 1

    def test_two_members(self):
        emb = np.array([[1.0, 0.0], [0.9, 0.1]])
        cluster = _build_cluster(emb, [0, 1])
        assert cluster.count == 2
        assert cluster.medoid_index in [0, 1]
        # dedup.py returns medoid only; best-quality is added by operations layer
        assert len(cluster.representative_indices) == 1

    def test_large_cluster_has_one_representative(self):
        """Medoid only — best-quality added later by operations layer."""
        emb = np.random.RandomState(42).randn(10, 5)
        cluster = _build_cluster(emb, list(range(10)))
        assert cluster.count == 10
        assert len(cluster.representative_indices) == 1  # Medoid only
        assert cluster.representative_indices[0] == cluster.medoid_index

    def test_all_indices_in_members(self):
        # Matrix must be large enough to hold the highest index (14)
        emb = np.random.RandomState(42).randn(15, 3)
        cluster = _build_cluster(emb, [2, 5, 8, 11, 14])
        assert set(cluster.member_indices) == {2, 5, 8, 11, 14}


class TestDeduplicateEvidence:
    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await deduplicate_evidence([], embedding_model="test-model")
        assert result == []

    @pytest.mark.asyncio
    async def test_single_document(self):
        with patch(
            "andamentum.epistemic.dedup.embed_documents", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [[[1.0, 0.0, 0.0]]]  # 1 doc, 1 chunk
            result = await deduplicate_evidence(
                ["single doc"], embedding_model="test-model"
            )
            assert len(result) == 1
            assert result[0].count == 1
            assert result[0].medoid_index == 0

    @pytest.mark.asyncio
    async def test_identical_documents_cluster_together(self):
        """N near-identical documents should produce fewer clusters than documents."""
        with patch(
            "andamentum.epistemic.dedup.embed_documents", new_callable=AsyncMock
        ) as mock_embed:
            # 5 nearly identical embeddings — each doc has 1 chunk
            base = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
            embeddings = [
                base + np.random.RandomState(i).randn(5) * 0.01 for i in range(5)
            ]
            mock_embed.return_value = [[e.tolist()] for e in embeddings]

            result = await deduplicate_evidence(
                ["doc"] * 5, min_cluster_size=2, embedding_model="test-model"
            )

            # All documents must be accounted for
            total_members = sum(c.count for c in result)
            assert total_members == 5
            # With near-identical embeddings, HDBSCAN must produce fewer clusters than documents
            assert len(result) < 5

    @pytest.mark.asyncio
    async def test_distinct_documents_form_separate_clusters(self):
        """Documents about different topics should form different clusters."""
        with patch(
            "andamentum.epistemic.dedup.embed_documents", new_callable=AsyncMock
        ) as mock_embed:
            emb, _ = _make_embeddings(n_clusters=3, n_per_cluster=4, dim=10, noise=0.02)
            mock_embed.return_value = [[e] for e in emb.tolist()]

            texts = [f"doc_{i}" for i in range(12)]
            result = await deduplicate_evidence(
                texts, min_cluster_size=2, embedding_model="test-model"
            )

            # Should have ~3 clusters (HDBSCAN may merge very close ones)
            total_members = sum(c.count for c in result)
            assert total_members == 12  # All documents accounted for
            # Expect 3 clusters, but allow some flexibility
            non_singleton = [c for c in result if c.count > 1]
            assert len(non_singleton) >= 2  # At least 2 real clusters found

    @pytest.mark.asyncio
    async def test_all_indices_covered(self):
        """Every input index must appear in exactly one cluster."""
        with patch(
            "andamentum.epistemic.dedup.embed_documents", new_callable=AsyncMock
        ) as mock_embed:
            emb, _ = _make_embeddings(n_clusters=2, n_per_cluster=5, dim=8, noise=0.03)
            mock_embed.return_value = [[e] for e in emb.tolist()]

            result = await deduplicate_evidence(
                [f"doc_{i}" for i in range(10)],
                min_cluster_size=2,
                embedding_model="test-model",
            )

            all_indices = []
            for cluster in result:
                all_indices.extend(cluster.member_indices)
            assert sorted(all_indices) == list(range(10))

    @pytest.mark.asyncio
    async def test_representatives_are_subset_of_members(self):
        with patch(
            "andamentum.epistemic.dedup.embed_documents", new_callable=AsyncMock
        ) as mock_embed:
            emb, _ = _make_embeddings(n_clusters=2, n_per_cluster=6, dim=8, noise=0.02)
            mock_embed.return_value = [[e] for e in emb.tolist()]

            result = await deduplicate_evidence(
                [f"doc_{i}" for i in range(12)],
                min_cluster_size=2,
                embedding_model="test-model",
            )

            for cluster in result:
                assert set(cluster.representative_indices).issubset(
                    set(cluster.member_indices)
                )
                assert cluster.medoid_index in cluster.representative_indices

    @pytest.mark.asyncio
    async def test_corroboration_count_matches_member_count(self):
        with patch(
            "andamentum.epistemic.dedup.embed_documents", new_callable=AsyncMock
        ) as mock_embed:
            emb, _ = _make_embeddings(n_clusters=2, n_per_cluster=4, dim=8, noise=0.02)
            mock_embed.return_value = [[e] for e in emb.tolist()]

            result = await deduplicate_evidence(
                [f"doc_{i}" for i in range(8)],
                min_cluster_size=2,
                embedding_model="test-model",
            )

            for cluster in result:
                assert cluster.count == len(cluster.member_indices)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — dedup wired into operations
# ══════════════════════════════════════════════════════════════════════════════

from types import SimpleNamespace  # noqa: E402

from andamentum.document_store import DocumentStore  # noqa: E402
from ..repository import EpistemicRepository  # noqa: E402
from ..entities.objective import Objective  # noqa: E402
from ..entities.evidence import Evidence  # noqa: E402
from ..entities.claim import Claim  # noqa: E402
from ..primitives import ClaimStage  # noqa: E402
from ..operations import ProposeClaimsOperation  # noqa: E402
from ..operations.base import OperationInput  # noqa: E402


class TestDedupIntegration:
    @pytest.fixture
    async def store(self, tmp_path):
        s = DocumentStore.for_database("test", db_dir=tmp_path)
        await s.initialize()
        return s

    @pytest.fixture
    async def repo(self, store):
        return EpistemicRepository(store)

    def _make_runner(self):
        """Mock runner that returns canned responses for assertion extraction and claim drafting."""
        call_count = [0]

        class Runner:
            def __init__(self):
                self.calls: list[tuple[str, dict]] = []

            async def run(self, agent_name: str, **kwargs: object) -> SimpleNamespace:
                self.calls.append((agent_name, kwargs))
                if agent_name == "epistemic_screen_relevance":
                    return SimpleNamespace(is_relevant=True)
                elif agent_name == "epistemic_extract_assertion":
                    call_count[0] += 1
                    return SimpleNamespace(assertion=f"Assertion {call_count[0]}")
                elif agent_name == "epistemic_draft_claim":
                    return SimpleNamespace(
                        statement="Test claim", scope="General", direction="supports"
                    )
                elif agent_name == "epistemic_judge_evidence":
                    return SimpleNamespace(
                        verdict="supports", reasoning="Test judgment"
                    )
                return SimpleNamespace()

        return Runner()

    @pytest.mark.asyncio
    async def test_propose_claims_promotes_every_cluster_to_representatives(self, repo):
        """ProposeClaimsOperation should give every cluster representatives.

        The old top-K=5 cap was retired: clustering's job is to enforce
        independence in the count, not to discard work. Cost-bounded LLM
        consumers cap themselves locally with LLM_PANEL_CAP.
        """
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="test",
            phase="planned",
            claims_proposed=False,
        )
        await repo.save(obj)

        # Create 8 evidence items with varying quality
        qualities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        for i in range(8):
            ev = Evidence(
                objective_id="obj-1",
                extracted=True,
                extracted_content=f"Evidence content {i}",
                source_ref=f"https://example.com/{i}",
                quality_score=qualities[i],
            )
            await repo.save(ev)

        runner = self._make_runner()

        with (
            patch(
                "andamentum.epistemic.operations.claims.deduplicate_evidence"
            ) as mock_dedup,
            patch(
                "andamentum.epistemic.embeddings.embed_texts", new_callable=AsyncMock
            ) as mock_embed,
            patch(
                "andamentum.epistemic.similarity.group_by_similarity"
            ) as mock_cluster,
        ):
            # 6 clusters: 2 real clusters + 4 singletons
            # Cluster A (indices 0,1): best quality 0.9
            # Cluster B (indices 2,3): best quality 0.7
            # Singleton C (index 4): quality 0.5
            # Singleton D (index 5): quality 0.4
            # Singleton E (index 6): quality 0.3
            # Singleton F (index 7): quality 0.2
            mock_dedup.return_value = [
                EvidenceCluster(
                    medoid_index=0,
                    representative_indices=[0],
                    member_indices=[0, 1],
                    count=2,
                ),
                EvidenceCluster(
                    medoid_index=2,
                    representative_indices=[2],
                    member_indices=[2, 3],
                    count=2,
                ),
                EvidenceCluster(
                    medoid_index=4,
                    representative_indices=[4],
                    member_indices=[4],
                    count=1,
                ),
                EvidenceCluster(
                    medoid_index=5,
                    representative_indices=[5],
                    member_indices=[5],
                    count=1,
                ),
                EvidenceCluster(
                    medoid_index=6,
                    representative_indices=[6],
                    member_indices=[6],
                    count=1,
                ),
                EvidenceCluster(
                    medoid_index=7,
                    representative_indices=[7],
                    member_indices=[7],
                    count=1,
                ),
            ]
            mock_embed.return_value = [
                [0.1] * 10
            ] * 5  # Enough for assertion clustering
            mock_cluster.return_value = [
                [i] for i in range(5)
            ]  # Each assertion its own cluster

            op = ProposeClaimsOperation(repo, runner, embedding_model="test-model")
            work = OperationInput(
                entity_id="obj-1", entity_type="objective", operation="propose_claims"
            )
            result = await op.execute(work)

        assert result.success

        # Check statuses
        all_evidence = await repo.query("evidence", objective_id="obj-1")
        statuses = {}
        for e in all_evidence:
            statuses[e.entity_id] = e.cluster_status

        # Every cluster contributes representatives; no items are deferred.
        # The exact rep/corroborative split inside multi-member clusters
        # depends on the best-quality augmentation step, which may add a
        # second rep when quality ordering puts the medoid below another
        # member. The robust contract is: every item is either rep or
        # corroborative, and no item is deferred.
        representative_count = sum(
            1 for s in statuses.values() if s == "representative"
        )
        deferred_count = sum(1 for s in statuses.values() if s == "deferred")
        corroborative_count = sum(1 for s in statuses.values() if s == "corroborative")

        # 6 clusters → at least 6 representatives (one per cluster).
        assert representative_count >= 6
        assert deferred_count == 0
        # Every item lands in one of the two categories.
        assert representative_count + corroborative_count == 8

    @pytest.mark.asyncio
    async def test_propose_claims_only_extracts_from_representatives(self, repo):
        """Assertion extraction should only run on representative evidence, not corroborative or deferred."""
        obj = Objective(
            entity_id="obj-2",
            objective_id="obj-2",
            description="test",
            phase="planned",
            claims_proposed=False,
        )
        await repo.save(obj)

        # Create 8 evidence items across 7 clusters (exceeds K=5, so 2 are deferred)
        for i in range(8):
            ev = Evidence(
                objective_id="obj-2",
                extracted=True,
                extracted_content=f"Evidence content {i}",
                source_ref=f"https://example.com/{i}",
                quality_score=0.9 - i * 0.1,
            )
            await repo.save(ev)

        runner = self._make_runner()

        with (
            patch(
                "andamentum.epistemic.operations.claims.deduplicate_evidence"
            ) as mock_dedup,
            patch(
                "andamentum.epistemic.embeddings.embed_texts", new_callable=AsyncMock
            ) as mock_embed,
            patch(
                "andamentum.epistemic.similarity.group_by_similarity"
            ) as mock_cluster,
        ):
            # 7 clusters: 1 cluster of 2 items + 6 singletons
            # Top-K=5 selects clusters ranked by quality: indices 0,1 (0.9), 2 (0.7), 3 (0.6), 4 (0.5), 5 (0.4)
            # Deferred: indices 6 (0.3), 7 (0.2)
            mock_dedup.return_value = [
                EvidenceCluster(
                    medoid_index=0,
                    representative_indices=[0],
                    member_indices=[0, 1],
                    count=2,
                ),
                EvidenceCluster(
                    medoid_index=2,
                    representative_indices=[2],
                    member_indices=[2],
                    count=1,
                ),
                EvidenceCluster(
                    medoid_index=3,
                    representative_indices=[3],
                    member_indices=[3],
                    count=1,
                ),
                EvidenceCluster(
                    medoid_index=4,
                    representative_indices=[4],
                    member_indices=[4],
                    count=1,
                ),
                EvidenceCluster(
                    medoid_index=5,
                    representative_indices=[5],
                    member_indices=[5],
                    count=1,
                ),
                EvidenceCluster(
                    medoid_index=6,
                    representative_indices=[6],
                    member_indices=[6],
                    count=1,
                ),
                EvidenceCluster(
                    medoid_index=7,
                    representative_indices=[7],
                    member_indices=[7],
                    count=1,
                ),
            ]
            # Embedding stubs for assertion clustering (5 representatives = 5 assertions)
            mock_embed.return_value = [[0.1] * 10] * 5
            mock_cluster.return_value = [[i] for i in range(5)]

            op = ProposeClaimsOperation(repo, runner, embedding_model="test-model")
            work = OperationInput(
                entity_id="obj-2", entity_type="objective", operation="propose_claims"
            )
            await op.execute(work)

        # Extract calls fire on every representative. 7 clusters guarantees
        # at least 7 reps; the size-2 cluster may augment to 2 reps (if the
        # query order puts a non-medoid as best-quality), so the upper bound
        # is 8. The contract: every cluster contributes, no corroborative is
        # extracted.
        extract_calls = [
            c for c in runner.calls if c[0] == "epistemic_extract_assertion"
        ]
        assert 7 <= len(extract_calls) <= 8

    @pytest.mark.asyncio
    async def test_best_quality_member_added_to_representatives(self, repo):
        """If the best-quality member is not medoid or boundary, it should be added."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="test",
            phase="planned",
            claims_proposed=False,
        )
        await repo.save(obj)

        # Create 5 evidence items — the item with quality 0.9 has highest quality
        # but must NOT be in the initial representative set returned by the mock.
        qualities = [0.5, 0.4, 0.3, 0.9, 0.2]
        for i in range(5):
            ev = Evidence(
                objective_id="obj-1",
                extracted=True,
                extracted_content=f"Evidence content {i}",
                source_ref=f"https://example.com/{i}",
                quality_score=qualities[i],
            )
            await repo.save(ev)

        runner = self._make_runner()

        with (
            patch(
                "andamentum.epistemic.operations.claims.deduplicate_evidence"
            ) as mock_dedup,
            patch(
                "andamentum.epistemic.embeddings.embed_texts", new_callable=AsyncMock
            ) as mock_embed,
            patch(
                "andamentum.epistemic.similarity.group_by_similarity"
            ) as mock_cluster,
        ):
            # 1 cluster of 5 items: best-quality member is NOT in the initial
            # representative set so that select_top_k_evidence must add it.
            # We use side_effect to find the best-quality index dynamically
            # because the repo may return evidence in a different order than
            # insertion order.
            async def _dedup_side_effect(
                texts, *, min_cluster_size=2, embedding_model=""
            ):
                # texts correspond 1:1 with the extracted list passed to deduplicate_evidence.
                # Find the best-quality index dynamically so the test is order-independent.
                from andamentum.epistemic.entities.evidence import Evidence as _Ev

                ev_list = await repo.query(
                    "evidence", objective_id="obj-1", extracted=True
                )
                extracted_ordered = [
                    e
                    for e in ev_list
                    if isinstance(e, _Ev)
                    and e.extracted
                    and e.extracted_content
                    and not e.invalidated
                ]
                best_idx = max(
                    range(len(extracted_ordered)),
                    key=lambda i: extracted_ordered[i].quality_score or 0.0,
                )
                n = len(extracted_ordered)
                # Build 3 initial reps that do NOT include best_idx so augmentation adds it.
                # Pick the first 3 indices from [0..n-1] that are not best_idx.
                initial_reps = [i for i in range(n) if i != best_idx][:3]
                medoid_idx = initial_reps[0]
                return [
                    EvidenceCluster(
                        medoid_index=medoid_idx,
                        representative_indices=initial_reps,
                        member_indices=list(range(n)),
                        count=n,
                    ),
                ]

            mock_dedup.side_effect = _dedup_side_effect
            # Representatives after augmentation = initial_reps(3) + best_idx(1) = 4 items,
            # so assertion clustering receives 4 assertions.
            mock_embed.return_value = [[0.1] * 10] * 4  # For assertion clustering
            mock_cluster.return_value = [[i] for i in range(4)]

            op = ProposeClaimsOperation(repo, runner, embedding_model="test-model")
            work = OperationInput(
                entity_id="obj-1", entity_type="objective", operation="propose_claims"
            )
            await op.execute(work)

        all_evidence = await repo.query("evidence", objective_id="obj-1")
        representatives = [
            e for e in all_evidence if e.cluster_status == "representative"
        ]

        # Should have 4 representatives: 3 from initial_reps + 1 best_quality added
        assert len(representatives) == 4

        # The best-quality item (quality 0.9) should be a representative
        rep_qualities = [e.quality_score for e in representatives]
        assert 0.9 in rep_qualities


class TestDownstreamFiltering:
    @pytest.fixture
    async def store(self, tmp_path):
        s = DocumentStore.for_database("test", db_dir=tmp_path)
        await s.initialize()
        return s

    @pytest.fixture
    async def repo(self, store):
        return EpistemicRepository(store)

    @pytest.mark.asyncio
    async def test_scrutiny_excludes_corroborative(self, repo):
        """ScrutiniseClaimOperation should not include corroborative evidence in summaries."""
        from andamentum.epistemic.operations import ScrutiniseClaimOperation

        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        # Create one representative and one corroborative evidence
        ev_rep = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Representative finding about X",
            cluster_status="representative",
            quality_score=0.7,
        )
        ev_corr = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Corroborative finding about X (same thing)",
            cluster_status="corroborative",
            quality_score=0.6,
        )
        await repo.save(ev_rep)
        await repo.save(ev_corr)

        claim = Claim(
            statement="test",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[ev_rep.entity_id, ev_corr.entity_id],
            evidence_count=2,
        )
        await repo.save(claim)

        op = ScrutiniseClaimOperation(repo, None)
        summaries = await op._gather_evidence_summaries(claim)

        # Should only have 1 summary (representative), not 2
        assert len(summaries) == 1
        assert "Representative finding" in summaries[0]

    @pytest.mark.asyncio
    async def test_deferred_evidence_excluded_from_scrutiny(self, repo):
        """Deferred evidence should not appear in scrutiny summaries."""
        from andamentum.epistemic.operations import ScrutiniseClaimOperation

        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        ev_rep = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Representative finding",
            cluster_status="representative",
            quality_score=0.8,
        )
        ev_def = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Deferred finding",
            cluster_status="deferred",
            quality_score=0.2,
        )
        await repo.save(ev_rep)
        await repo.save(ev_def)

        claim = Claim(
            statement="test",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[ev_rep.entity_id, ev_def.entity_id],
            evidence_count=2,
        )
        await repo.save(claim)

        op = ScrutiniseClaimOperation(repo, None)
        summaries = await op._gather_evidence_summaries(claim)

        assert len(summaries) == 1
        assert "Representative finding" in summaries[0]

    @pytest.mark.asyncio
    async def test_quality_weighted_sum_includes_corroborative(self, repo):
        """quality_weighted_evidence_sum sums quality across ALL judged evidence.

        Cluster_status no longer filters at the gate layer: clustering's role
        is to inform posterior weighting, not to disqualify evidence from the
        promote-time confidence calculation.
        """
        from andamentum.epistemic.gates import quality_weighted_evidence_sum

        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        ev_rep = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Representative finding",
            cluster_status="representative",
            quality_score=0.8,
        )
        ev_corr = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Corroborative finding",
            cluster_status="corroborative",
            quality_score=0.6,
        )
        await repo.save(ev_rep)
        await repo.save(ev_corr)

        claim = Claim(
            statement="test",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[ev_rep.entity_id, ev_corr.entity_id],
            evidence_count=2,
        )
        await repo.save(claim)

        total = await quality_weighted_evidence_sum(claim, repo)

        # Both items contribute their quality.
        assert abs(total - (0.8 + 0.6)) < 1e-6

    @pytest.mark.asyncio
    async def test_quality_weighted_sum_includes_deferred(self, repo):
        """quality_weighted_evidence_sum counts deferred evidence too.

        Existing databases may have evidence with the legacy `deferred`
        cluster_status. After dropping the gate-layer filter, those still
        contribute their quality (the `deferred` status is retired going
        forward but live in old data).
        """
        from andamentum.epistemic.gates import quality_weighted_evidence_sum

        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        ev_rep = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Representative finding",
            cluster_status="representative",
            quality_score=0.8,
        )
        ev_def = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Deferred finding",
            cluster_status="deferred",
            quality_score=0.6,
        )
        await repo.save(ev_rep)
        await repo.save(ev_def)

        claim = Claim(
            statement="test",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[ev_rep.entity_id, ev_def.entity_id],
            evidence_count=2,
        )
        await repo.save(claim)

        total = await quality_weighted_evidence_sum(claim, repo)

        # Both items contribute.
        assert abs(total - (0.8 + 0.6)) < 1e-6

    @pytest.mark.asyncio
    async def test_freeze_snapshot_includes_all_non_invalidated_evidence(self, repo):
        """FreezeSnapshotOperation captures the full evidence base.

        The snapshot is an audit record of the evidence at freeze time, not a
        pre-filtered slice. Consumers (synthesize_report) apply their own
        cap via top_n_representatives when building prompts, so a complete
        snapshot is information-preserving for downstream readers without
        bloating any LLM prompt.
        """
        from andamentum.epistemic.operations import FreezeSnapshotOperation
        from andamentum.epistemic.operations.base import OperationInput

        obj = Objective(
            entity_id="obj-freeze-1",
            objective_id="obj-freeze-1",
            description="test",
            phase="claims_done",
        )
        await repo.save(obj)

        ev_rep = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Representative finding",
            cluster_status="representative",
            quality_score=0.8,
        )
        ev_corr = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Corroborative finding",
            cluster_status="corroborative",
            quality_score=0.6,
        )
        ev_legacy = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Legacy evidence (unclustered)",
            quality_score=0.5,
        )
        await repo.save(ev_rep)
        await repo.save(ev_corr)
        await repo.save(ev_legacy)

        op = FreezeSnapshotOperation(repo, None)
        work = OperationInput(
            entity_id=obj.entity_id,
            entity_type="objective",
            operation="freeze_snapshot",
        )
        result = await op.execute(work)

        assert result.success

        snapshot_id = result.created_entities[0]
        from andamentum.epistemic.entities.snapshot import Snapshot

        snapshot = await repo.get("snapshot", snapshot_id)
        assert isinstance(snapshot, Snapshot)

        # All three non-invalidated evidence items are in the snapshot.
        assert ev_rep.entity_id in snapshot.evidence_ids
        assert ev_corr.entity_id in snapshot.evidence_ids
        assert ev_legacy.entity_id in snapshot.evidence_ids

    @pytest.mark.asyncio
    async def test_quality_weighted_sum_includes_unclustered(self, repo):
        """Legacy/unclustered evidence (no cluster_status) must still count toward quality sum."""
        from andamentum.epistemic.gates import quality_weighted_evidence_sum

        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        # Legacy evidence has no cluster_status field set (defaults to "unclustered")
        ev_legacy = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="Legacy finding",
            quality_score=0.9,
        )
        await repo.save(ev_legacy)

        claim = Claim(
            statement="test",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[ev_legacy.entity_id],
            evidence_count=1,
        )
        await repo.save(claim)

        total = await quality_weighted_evidence_sum(claim, repo)

        # Legacy evidence must be counted
        assert abs(total - 0.9) < 1e-6
