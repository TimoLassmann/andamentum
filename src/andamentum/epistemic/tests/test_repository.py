"""Tests for EpistemicRepository CRUD and query operations."""

import pytest
from ..repository import EntityNotFoundError
from ..entities import (
    Claim,
    Evidence,
    Uncertainty,
    UncertaintyType,
    Objective,
    Decision,
    Snapshot,
    Artefact,
)


class TestCRUD:
    async def test_save_and_get_claim(self, repo):
        c = Claim(entity_id="c-1", objective_id="obj-1", statement="Test claim")
        await repo.save(c)
        loaded = await repo.get_claim("c-1")
        assert loaded.statement == "Test claim"
        assert loaded.entity_type == "claim"

    async def test_save_and_get_evidence(self, repo):
        e = Evidence(entity_id="e-1", objective_id="obj-1", source_type="paper")
        await repo.save(e)
        loaded = await repo.get_evidence("e-1")
        assert loaded.source_type == "paper"

    async def test_save_and_get_objective(self, repo):
        o = Objective(entity_id="obj-1", objective_id="obj-1", description="Research Q")
        await repo.save(o)
        loaded = await repo.get_objective("obj-1")
        assert loaded.description == "Research Q"

    async def test_update_existing(self, repo):
        c = Claim(entity_id="c-1", objective_id="obj-1", statement="V1")
        await repo.save(c)
        c.statement = "V2"
        c.scrutiny_verdict = "pass"
        await repo.save(c)
        loaded = await repo.get_claim("c-1")
        assert loaded.scrutiny_verdict == "pass"

    async def test_delete(self, repo):
        c = Claim(entity_id="c-1", objective_id="obj-1", statement="X")
        await repo.save(c)
        assert await repo.exists("claim", "c-1")
        deleted = await repo.delete("claim", "c-1")
        assert deleted is True
        assert not await repo.exists("claim", "c-1")

    async def test_delete_nonexistent(self, repo):
        assert await repo.delete("claim", "no-such") is False

    async def test_entity_not_found(self, repo):
        with pytest.raises(EntityNotFoundError):
            await repo.get("claim", "no-such-id")

    async def test_unknown_entity_type(self, repo):
        with pytest.raises(KeyError, match="Unknown entity type"):
            await repo.get("spaceship", "id-1")

    async def test_exists(self, repo):
        assert not await repo.exists("claim", "c-1")
        c = Claim(entity_id="c-1", objective_id="obj-1", statement="X")
        await repo.save(c)
        assert await repo.exists("claim", "c-1")

    async def test_count(self, repo):
        for i in range(3):
            c = Claim(entity_id=f"c-{i}", objective_id="obj-1", statement=f"Claim {i}")
            await repo.save(c)
        assert await repo.count("claim") == 3
        assert await repo.count("evidence") == 0


class TestQuery:
    async def test_query_by_exact_match(self, repo):
        c1 = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="A",
            scrutiny_verdict="pass",
        )
        c2 = Claim(
            entity_id="c-2",
            objective_id="obj-1",
            statement="B",
            scrutiny_verdict="fail",
        )
        await repo.save(c1)
        await repo.save(c2)
        results = await repo.query("claim", scrutiny_verdict="pass")
        assert len(results) == 1
        assert results[0].entity_id == "c-1"

    async def test_query_by_none_filter(self, repo):
        """None filter matches documents where field is absent or None (SQL IS NULL)."""
        c1 = Claim(
            entity_id="c-1", objective_id="obj-1", statement="A", scrutiny_verdict=None
        )
        c2 = Claim(
            entity_id="c-2",
            objective_id="obj-1",
            statement="B",
            scrutiny_verdict="pass",
        )
        await repo.save(c1)
        await repo.save(c2)
        results = await repo.query("claim", scrutiny_verdict=None)
        assert len(results) == 1
        assert results[0].entity_id == "c-1"

    async def test_query_by_objective(self, repo):
        c1 = Claim(entity_id="c-1", objective_id="obj-1", statement="A")
        c2 = Claim(entity_id="c-2", objective_id="obj-2", statement="B")
        await repo.save(c1)
        await repo.save(c2)
        results = await repo.query("claim", objective_id="obj-1")
        assert len(results) == 1

    async def test_contains_filter(self, repo):
        c = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="A",
            evidence_ids=["e-1", "e-2"],
        )
        await repo.save(c)
        results = await repo.query("claim", evidence_ids__contains="e-1")
        assert len(results) == 1
        results = await repo.query("claim", evidence_ids__contains="e-99")
        assert len(results) == 0

    async def test_gte_filter(self, repo):
        c1 = Claim(
            entity_id="c-1", objective_id="o", statement="A", evidence_ids=["e1"]
        )
        c2 = Claim(
            entity_id="c-2",
            objective_id="o",
            statement="B",
            evidence_ids=["e1", "e2", "e3"],
        )
        await repo.save(c1)
        await repo.save(c2)
        results = await repo.query("claim", evidence_count__gte=2)
        assert len(results) == 1
        assert results[0].entity_id == "c-2"

    async def test_lte_filter(self, repo):
        c1 = Claim(
            entity_id="c-1", objective_id="o", statement="A", evidence_ids=["e1"]
        )
        c2 = Claim(
            entity_id="c-2",
            objective_id="o",
            statement="B",
            evidence_ids=["e1", "e2", "e3"],
        )
        await repo.save(c1)
        await repo.save(c2)
        results = await repo.query("claim", evidence_count__lte=1)
        assert len(results) == 1
        assert results[0].entity_id == "c-1"

    async def test_gt_lt_filter(self, repo):
        for i in range(5):
            c = Claim(
                entity_id=f"c-{i}",
                objective_id="o",
                statement=f"C{i}",
                modification_count=i,
            )
            await repo.save(c)
        results = await repo.query(
            "claim", modification_count__gt=2, modification_count__lt=4
        )
        assert len(results) == 1
        assert results[0].modification_count == 3


class TestConvenienceMethods:
    async def test_get_claims_for_objective(self, repo):
        c1 = Claim(entity_id="c-1", objective_id="obj-1", statement="A")
        c2 = Claim(entity_id="c-2", objective_id="obj-1", statement="B")
        c3 = Claim(entity_id="c-3", objective_id="obj-2", statement="C")
        for c in [c1, c2, c3]:
            await repo.save(c)
        claims = await repo.get_claims_for_objective("obj-1")
        assert len(claims) == 2

    async def test_get_evidence_for_objective(self, repo):
        e = Evidence(entity_id="e-1", objective_id="obj-1")
        await repo.save(e)
        evidence = await repo.get_evidence_for_objective("obj-1")
        assert len(evidence) == 1

    async def test_get_blocking_uncertainties(self, repo):
        u1 = Uncertainty(
            entity_id="u-1",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="Blocking",
        )
        u2 = Uncertainty(
            entity_id="u-2",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,
            description="Non-blocking",
        )
        await repo.save(u1)
        await repo.save(u2)
        blocking = await repo.get_blocking_uncertainties("obj-1")
        assert len(blocking) == 1
        assert blocking[0].entity_id == "u-1"

    async def test_get_decisions_excludes_reversed(self, repo):
        d1 = Decision(
            entity_id="d-1", objective_id="o", statement="Go", justification="Why"
        )
        d2 = Decision(
            entity_id="d-2", objective_id="o", statement="Stop", justification="Why"
        )
        d2.reverse("Changed mind")
        await repo.save(d1)
        await repo.save(d2)
        active = await repo.get_decisions_for_objective("o")
        assert len(active) == 1
        all_decisions = await repo.get_decisions_for_objective(
            "o", include_reversed=True
        )
        assert len(all_decisions) == 2

    async def test_typed_convenience_methods(self, repo):
        u = Uncertainty(entity_id="u-1", objective_id="o", description="Test")
        await repo.save(u)
        loaded = await repo.get_uncertainty("u-1")
        assert isinstance(loaded, Uncertainty)

        d = Decision(
            entity_id="d-1", objective_id="o", statement="X", justification="Y"
        )
        await repo.save(d)
        loaded_d = await repo.get_decision("d-1")
        assert isinstance(loaded_d, Decision)

        s = Snapshot(entity_id="s-1", objective_id="o")
        await repo.save(s)
        loaded_s = await repo.get_snapshot("s-1")
        assert isinstance(loaded_s, Snapshot)

        a = Artefact(entity_id="a-1", objective_id="o", snapshot_id="s-1")
        await repo.save(a)
        loaded_a = await repo.get_artefact("a-1")
        assert isinstance(loaded_a, Artefact)
