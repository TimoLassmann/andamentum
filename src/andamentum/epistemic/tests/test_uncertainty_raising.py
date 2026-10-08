"""D1 — idempotent uncertainty raising.

Self-feeding-loop plan, docs/epistemic/SELF_FEEDING_LOOP_PLAN.md §1 D1 and
§2 S1: raising the same issue (same type, near-duplicate description) on
one claim twice must create exactly ONE ``Uncertainty`` entity; the second
raise increments ``raise_count`` instead. Pinned here with a stub embedding
function (S1's unit test) plus coverage of the no-embedding-model exact-text
fallback and the "different claim" / "different type" / "genuinely
different description" non-dedup cases D1 requires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.document_store import Store
from andamentum.epistemic.entities import Objective, Uncertainty, UncertaintyType
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.epistemic.uncertainty_raising import raise_uncertainty


@pytest.fixture
async def repo(tmp_path: Path) -> EpistemicRepository:
    store = await Store.open(tmp_path / "uncertainty_raising.db")
    repo = EpistemicRepository(store)
    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return repo


async def _objective_id(repo: EpistemicRepository) -> str:
    rows = await repo.query("objective")
    return rows[0].entity_id


def _stub_embed_texts(monkeypatch: pytest.MonkeyPatch, groups: dict[str, list[float]]):
    """Patch ``uncertainty_raising.embed_texts`` to return a fixed vector per
    text, keyed by substring membership in ``groups`` — texts that share a
    group key are near-identical embeddings (cosine ~1.0); texts in
    different groups are orthogonal.
    """

    async def _fake_embed_texts(texts: list[str], *, model: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            for key, base_vec in groups.items():
                if key in text:
                    vectors.append(list(base_vec))
                    break
            else:
                raise AssertionError(f"No stub group matches text: {text!r}")
        return vectors

    monkeypatch.setattr(
        "andamentum.epistemic.uncertainty_raising.embed_texts", _fake_embed_texts
    )


# ── S1: embedding-based near-duplicate dedup ───────────────────────────


class TestEmbeddingDedup:
    async def test_second_near_duplicate_raise_increments_instead_of_creating(
        self, repo: EpistemicRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj_id = await _objective_id(repo)
        # Both descriptions map to the same embedding (same group "CONTRA"),
        # simulating two slightly different wordings of the same finding.
        _stub_embed_texts(monkeypatch, {"CONTRA": [1.0, 0.0, 0.0]})

        first, created_first = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="CONTRA: the sources disagree on dosage.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        assert created_first is True
        assert first.raise_count == 1

        second, created_second = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="CONTRA: sources still disagree about dosage here.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        assert created_second is False
        assert second.entity_id == first.entity_id
        assert second.raise_count == 2

        # Exactly one Uncertainty entity persisted for this claim/type.
        rows = await repo.query("uncertainty", objective_id=obj_id)
        assert len(rows) == 1
        reloaded = await repo.get("uncertainty", first.entity_id)
        assert reloaded.raise_count == 2

    async def test_dedup_matches_against_a_resolved_uncertainty_too(
        self, repo: EpistemicRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D1: dedup compares against existing uncertainties resolved OR
        unresolved — the whole point is that a re-raised, already-resolved
        contradiction does not reopen as a fresh blocking entity."""
        obj_id = await _objective_id(repo)
        _stub_embed_texts(monkeypatch, {"CONTRA": [1.0, 0.0, 0.0]})

        first, _ = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="CONTRA: dosage disagreement.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        first.resolve("Resolved: the higher dose was a typo in source B.")
        await repo.save(first)

        second, created = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="CONTRA: dosage disagreement persists.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        assert created is False
        assert second.entity_id == first.entity_id
        assert second.raise_count == 2
        # The increment did not disturb the resolution already recorded.
        assert second.resolution is not None

    async def test_different_type_does_not_dedup(
        self, repo: EpistemicRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj_id = await _objective_id(repo)
        _stub_embed_texts(monkeypatch, {"SAME": [1.0, 0.0, 0.0]})

        await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="SAME text",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        _second, created = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="SAME text",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        assert created is True

        rows = await repo.query("uncertainty", objective_id=obj_id)
        assert len(rows) == 2

    async def test_different_claim_does_not_dedup(
        self, repo: EpistemicRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj_id = await _objective_id(repo)
        _stub_embed_texts(monkeypatch, {"SAME": [1.0, 0.0, 0.0]})

        await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="SAME text",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        _second, created = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="SAME text",
            affected_claim_ids=["claim-2"],
            created_by="test",
            embedding_model="stub-embed",
        )
        assert created is True

        rows = await repo.query("uncertainty", objective_id=obj_id)
        assert len(rows) == 2

    async def test_genuinely_different_description_does_not_dedup(
        self, repo: EpistemicRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        obj_id = await _objective_id(repo)
        _stub_embed_texts(
            monkeypatch,
            {"ALPHA": [1.0, 0.0, 0.0], "BETA": [0.0, 1.0, 0.0]},
        )

        await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="ALPHA concern",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        _second, created = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="BETA concern",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model="stub-embed",
        )
        assert created is True

        rows = await repo.query("uncertainty", objective_id=obj_id)
        assert len(rows) == 2


# ── No embedding model: exact normalised-text fallback, never "always save" ─


class TestNoEmbeddingModelFallback:
    async def test_exact_normalised_match_dedups_without_an_embedding_model(
        self, repo: EpistemicRepository
    ) -> None:
        obj_id = await _objective_id(repo)

        first, created_first = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="  The Sources   Disagree on dosage.  ",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model=None,
        )
        assert created_first is True

        second, created_second = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="the sources disagree on dosage.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model=None,
        )
        assert created_second is False
        assert second.entity_id == first.entity_id
        assert second.raise_count == 2

    async def test_no_embedding_model_never_falls_back_to_always_save(
        self, repo: EpistemicRepository
    ) -> None:
        """A near-duplicate that is NOT an exact normalised match is still
        created as a new entity when there is no embedding model — D1 is
        explicit that the fallback is exact match, never "always create"."""
        obj_id = await _objective_id(repo)

        await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="The sources disagree on dosage.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model=None,
        )
        _second, created = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="The sources appear to disagree about the dosage used.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model=None,
        )
        assert created is True

        rows = await repo.query("uncertainty", objective_id=obj_id)
        assert len(rows) == 2


# ── Construction details on the create path ─────────────────────────────


class TestCreatePath:
    async def test_new_uncertainty_defaults_raise_count_to_one(
        self, repo: EpistemicRepository
    ) -> None:
        obj_id = await _objective_id(repo)
        unc, created = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,
            description="Some caveat.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model=None,
        )
        assert created is True
        assert unc.raise_count == 1
        assert isinstance(unc, Uncertainty)

    async def test_is_blocking_has_no_override_parameter(
        self, repo: EpistemicRepository
    ) -> None:
        """There is no ``is_blocking`` parameter on the helper:
        ``Uncertainty.model_post_init`` always recomputes ``is_blocking``
        from ``uncertainty_type``, so a caller-supplied override would be a
        false affordance. Passing it is a ``TypeError``, and the derived
        value on a newly created entity is exactly the type's default."""
        obj_id = await _objective_id(repo)
        with pytest.raises(TypeError):
            await raise_uncertainty(
                repo,
                objective_id=obj_id,
                uncertainty_type=UncertaintyType.CONTRADICTION,
                description="Claim conflicts with a sibling.",
                affected_claim_ids=["claim-1"],
                created_by="test",
                embedding_model=None,
                is_blocking=True,  # type: ignore[call-arg]
            )

        unc, _created = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="Claim conflicts with a sibling.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model=None,
        )
        # CONTRADICTION is always blocking, derived from the type.
        assert unc.is_blocking is True

    async def test_spawned_from_id_applies_on_create(
        self, repo: EpistemicRepository
    ) -> None:
        obj_id = await _objective_id(repo)
        unc, _created = await raise_uncertainty(
            repo,
            objective_id=obj_id,
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="A spawned concern.",
            affected_claim_ids=["claim-1"],
            created_by="test",
            embedding_model=None,
            spawned_from_id="parent-unc-1",
        )
        assert unc.spawned_from_id == "parent-unc-1"
