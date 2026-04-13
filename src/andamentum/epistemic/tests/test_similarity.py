"""Tests for the shared similarity utility (embed_and_group, validate_groups)."""

import math

import pytest
from unittest.mock import AsyncMock, MagicMock

from ..similarity import cosine_similarity, group_by_similarity, embed_and_group, validate_groups


# ── Helpers ─────────────────────────────────────────────────────────────


def _unit_vector(angle_degrees: float) -> list[float]:
    """2D unit vector at given angle. Easy to reason about similarity."""
    rad = math.radians(angle_degrees)
    return [math.cos(rad), math.sin(rad)]


def _make_cluster(center_angle: float, n: int, spread: float = 2.0) -> list[list[float]]:
    """Generate n unit vectors clustered around center_angle (degrees)."""
    return [_unit_vector(center_angle + i * spread / n) for i in range(n)]


# ── cosine_similarity ───────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_similar_vectors_high_similarity(self):
        a = _unit_vector(0)
        b = _unit_vector(10)
        assert cosine_similarity(a, b) == pytest.approx(math.cos(math.radians(10)), abs=1e-6)

    def test_dissimilar_vectors_low_similarity(self):
        a = _unit_vector(0)
        b = _unit_vector(80)
        assert cosine_similarity(a, b) == pytest.approx(math.cos(math.radians(80)), abs=1e-6)


# ── group_by_similarity ────────────────────────────────────────────────


class TestGroupBySimilarity:
    def test_empty_input(self):
        assert group_by_similarity([], 0.5) == []

    def test_single_item(self):
        assert group_by_similarity([[1.0, 0.0]], 0.5) == [[0]]

    def test_two_similar_items(self):
        embeddings = [_unit_vector(0), _unit_vector(5)]
        groups = group_by_similarity(embeddings, threshold=0.8)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1]

    def test_two_dissimilar_items(self):
        embeddings = [_unit_vector(0), _unit_vector(90)]
        groups = group_by_similarity(embeddings, threshold=0.8)
        assert len(groups) == 2

    def test_three_distinct_clusters(self):
        cluster_a = _make_cluster(0, 3, spread=2)
        cluster_b = _make_cluster(90, 3, spread=2)
        cluster_c = _make_cluster(180, 3, spread=2)
        embeddings = cluster_a + cluster_b + cluster_c
        groups = group_by_similarity(embeddings, threshold=0.9)
        assert len(groups) == 3
        group_sets = [set(g) for g in groups]
        assert {0, 1, 2} in group_sets
        assert {3, 4, 5} in group_sets
        assert {6, 7, 8} in group_sets

    def test_single_linkage_transitivity(self):
        """A-B-C chain: A~B and B~C but not A~C. Single-linkage groups all three."""
        embeddings = [_unit_vector(0), _unit_vector(20), _unit_vector(40)]
        groups = group_by_similarity(embeddings, threshold=0.9)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_all_singletons_at_high_threshold(self):
        embeddings = [_unit_vector(i * 30) for i in range(6)]
        groups = group_by_similarity(embeddings, threshold=0.999)
        assert len(groups) == 6

    def test_all_items_present(self):
        """Every index appears in exactly one group."""
        embeddings = [_unit_vector(i * 15) for i in range(10)]
        groups = group_by_similarity(embeddings, threshold=0.9)
        all_indices = sorted(idx for g in groups for idx in g)
        assert all_indices == list(range(10))

    def test_threshold_boundary(self):
        """Items exactly at the threshold are included."""
        angle = math.degrees(math.acos(0.8))
        embeddings = [_unit_vector(0), _unit_vector(angle)]
        groups = group_by_similarity(embeddings, threshold=0.8)
        assert len(groups) == 1


# ── embed_and_group ─────────────────────────────────────────────────────


class TestEmbedAndGroup:
    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await embed_and_group([], threshold=0.8, embedding_model="test-model")
        assert result == []

    @pytest.mark.asyncio
    async def test_single_item(self):
        result = await embed_and_group(["hello"], threshold=0.8, embedding_model="test-model")
        assert result == [[0]]

    @pytest.mark.asyncio
    async def test_groups_similar_texts(self):
        """Mock embed_texts to return known embeddings, verify grouping."""
        from unittest.mock import patch

        cluster_a = _make_cluster(0, 3, spread=2)
        cluster_b = _make_cluster(90, 2, spread=2)
        mock_embeddings = cluster_a + cluster_b

        with patch("andamentum.epistemic.embeddings.embed_texts", new=AsyncMock(return_value=mock_embeddings)):
            groups = await embed_and_group(
                ["a1", "a2", "a3", "b1", "b2"],
                threshold=0.9,
                embedding_model="test-model",
            )

        assert len(groups) == 2
        group_sets = [set(g) for g in groups]
        assert {0, 1, 2} in group_sets
        assert {3, 4} in group_sets


# ── validate_groups ─────────────────────────────────────────────────────


class TestValidateGroups:
    @pytest.mark.asyncio
    async def test_small_groups_pass_through(self):
        runner = MagicMock()
        runner.run = AsyncMock()

        groups = [[0], [1, 2]]
        result = await validate_groups(texts=["a", "b", "c"], groups=groups, runner=runner, min_group_size=3)

        runner.run.assert_not_called()
        assert result == [[0], [1, 2]]

    @pytest.mark.asyncio
    async def test_large_group_confirmed(self):
        mock_output = MagicMock()
        mock_output.subgroups = [[1, 2, 3, 4]]

        runner = MagicMock()
        runner.run = AsyncMock(return_value=mock_output)

        result = await validate_groups(texts=["a", "b", "c", "d"], groups=[[0, 1, 2, 3]], runner=runner)

        runner.run.assert_called_once()
        assert len(result) == 1
        assert sorted(result[0]) == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_large_group_split(self):
        mock_output = MagicMock()
        mock_output.subgroups = [[1, 2], [3, 4]]

        runner = MagicMock()
        runner.run = AsyncMock(return_value=mock_output)

        result = await validate_groups(texts=[""] * 50, groups=[[10, 20, 30, 40]], runner=runner)

        assert len(result) == 2
        assert sorted(result[0]) == [10, 20]
        assert sorted(result[1]) == [30, 40]

    @pytest.mark.asyncio
    async def test_mixed_small_and_large_groups(self):
        mock_output = MagicMock()
        mock_output.subgroups = [[1, 2, 3]]

        runner = MagicMock()
        runner.run = AsyncMock(return_value=mock_output)

        result = await validate_groups(
            texts=["a", "b", "c", "d", "e", "f"],
            groups=[[0], [1, 2], [3, 4, 5]],
            runner=runner,
        )

        assert runner.run.call_count == 1
        assert len(result) == 3
        assert result[0] == [0]
        assert result[1] == [1, 2]
        assert sorted(result[2]) == [3, 4, 5]

    @pytest.mark.asyncio
    async def test_custom_min_group_size(self):
        mock_output = MagicMock()
        mock_output.subgroups = [[1], [2]]

        runner = MagicMock()
        runner.run = AsyncMock(return_value=mock_output)

        result = await validate_groups(texts=[""] * 20, groups=[[5, 10]], runner=runner, min_group_size=2)

        runner.run.assert_called_once()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_invalid_item_numbers_ignored(self):
        mock_output = MagicMock()
        mock_output.subgroups = [[1, 2, 99]]

        runner = MagicMock()
        runner.run = AsyncMock(return_value=mock_output)

        result = await validate_groups(texts=["a", "b", "c"], groups=[[0, 1, 2]], runner=runner)

        assert len(result) == 1
        assert sorted(result[0]) == [0, 1]


# ── medoid ──────────────────────────────────────────────────────────────


class TestMedoid:
    def test_singleton(self):
        from andamentum.epistemic.similarity import medoid

        assert medoid([[1.0, 0.0]], [0]) == 0

    def test_medoid_is_most_central(self):
        from andamentum.epistemic.similarity import medoid

        embeddings = [_unit_vector(0), _unit_vector(10), _unit_vector(20)]
        assert medoid(embeddings, [0, 1, 2]) == 1

    def test_medoid_with_sparse_indices(self):
        from andamentum.epistemic.similarity import medoid

        embeddings = [_unit_vector(i * 10) for i in range(7)]
        assert medoid(embeddings, [1, 3, 5]) == 3


# ── assess_clustering ───────────────────────────────────────────────────


class TestAssessClustering:
    def test_well_separated_clusters(self):
        from andamentum.epistemic.similarity import assess_clustering

        embeddings = (
            _make_cluster(0, 4, spread=3)
            + _make_cluster(90, 4, spread=3)
            + _make_cluster(180, 4, spread=3)
        )
        groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

        quality = assess_clustering(embeddings, groups)

        assert quality.computable
        assert quality.silhouette > 0.5
        assert quality.interpretation in ("strong", "reasonable")
        assert len(quality.groups) == 3

    def test_edge_case_single_cluster(self):
        from andamentum.epistemic.similarity import assess_clustering

        embeddings = [_unit_vector(0), _unit_vector(5), _unit_vector(10)]
        quality = assess_clustering(embeddings, [[0, 1, 2]])
        assert not quality.computable

    def test_edge_case_all_singletons(self):
        from andamentum.epistemic.similarity import assess_clustering

        embeddings = [_unit_vector(i * 30) for i in range(5)]
        quality = assess_clustering(embeddings, [[0], [1], [2], [3], [4]])
        assert not quality.computable

    def test_per_group_breakdown(self):
        from andamentum.epistemic.similarity import assess_clustering

        tight = _make_cluster(0, 4, spread=1)
        loose = _make_cluster(90, 4, spread=15)
        embeddings = tight + loose
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]

        quality = assess_clustering(embeddings, groups)

        assert quality.computable
        assert len(quality.groups) == 2
        assert quality.groups[0].mean_intra_sim > quality.groups[1].mean_intra_sim


# ── Agent registration ─────────────────────────────────────────────────


class TestAgentRegistration:
    def test_validate_group_agent_registered(self):
        from andamentum.epistemic.agents import AGENT_REGISTRY

        assert "epistemic_validate_group" in AGENT_REGISTRY

    def test_validate_group_output_model(self):
        from andamentum.epistemic.agents import AGENT_REGISTRY
        from andamentum.epistemic.agents.output_models import ValidateGroupOutput

        defn = AGENT_REGISTRY["epistemic_validate_group"]
        assert defn.output_model is ValidateGroupOutput
