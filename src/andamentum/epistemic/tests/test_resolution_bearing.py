"""Self-feeding-loop plan D2: a resolution carries a bearing.

``ResolveUncertaintyOutput.bearing`` — ``supports_claim`` /
``refutes_claim`` / ``neutral`` — is stored on
``Uncertainty.resolution_bearing`` so the cap (D3) can read what a
resolved uncertainty said about the claim's direction, not just that it
was "resolved". Default is ``None`` on the output model precisely so an
older prompt, or a ``PromptedOutput`` fallback that omits the field, still
validates — and so "never assessed" (``None``) stays distinguishable from
"assessed as neutral" (``"neutral"``).

S3: the field round-trips through the repository, and the no-field case
(older/fallback output) stores ``None`` on the entity without failing.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.agents.output_models import ResolveUncertaintyOutput
from andamentum.epistemic.entities import Claim, Objective, Uncertainty
from andamentum.epistemic.entities.uncertainty import UncertaintyType
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.uncertainty import ResolveUncertaintyOperation
from andamentum.epistemic.repository import EpistemicRepository


# ── ResolveUncertaintyOutput itself: default and omitted-field cases ────


def test_bearing_defaults_to_none_when_field_is_omitted() -> None:
    """The `PromptedOutput` fallback case: an older prompt's JSON has no
    `bearing` key at all. Construction must still succeed, and the
    omission must read as "not assessed", not as "assessed as neutral"."""
    out = ResolveUncertaintyOutput.model_validate(
        {
            "uncertainty_id": "unc_1",
            "can_resolve": True,
            "resolution": "Addressed by the meta-analysis.",
            "remaining_concerns": [],
        }
    )
    assert out.bearing is None


def test_bearing_accepts_each_literal_value() -> None:
    for value in ("supports_claim", "refutes_claim", "neutral"):
        out = ResolveUncertaintyOutput(
            uncertainty_id="unc_1",
            can_resolve=True,
            resolution="x",
            bearing=value,
            remaining_concerns=[],
        )
        assert out.bearing == value


# ── Uncertainty.resolve(): bearing is optional and keyword-only ────────


def test_resolve_without_bearing_leaves_resolution_bearing_none() -> None:
    u = Uncertainty(objective_id="o", description="Unknown")
    u.resolve("Found the answer")
    assert u.resolution == "Found the answer"
    assert u.resolution_bearing is None


def test_resolve_with_bearing_stores_it() -> None:
    u = Uncertainty(objective_id="o", description="Unknown")
    u.resolve("The higher dose was a typo.", bearing="supports_claim")
    assert u.resolution_bearing == "supports_claim"


# ── S3: round-trips through the repository ──────────────────────────────


async def _repo(tmp_path: Path, db_name: str) -> EpistemicRepository:
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


async def test_resolution_bearing_round_trips_through_the_repository(
    tmp_path: Path,
) -> None:
    repo = await _repo(tmp_path, "bearing_round_trip")
    obj = Objective(
        description="Test objective",
        clarified_question="Test?",
        question_type="verificatory",
        claim_to_verify="Test?",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    u = Uncertainty(
        objective_id=obj.entity_id,
        description="A blocking concern.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=["c1"],
    )
    u.resolve("The refuting study used a different dose.", bearing="refutes_claim")
    await repo.save(u)

    reloaded = await repo.get("uncertainty", u.entity_id)
    assert isinstance(reloaded, Uncertainty)
    assert reloaded.resolution == "The refuting study used a different dose."
    assert reloaded.resolution_bearing == "refutes_claim"


async def test_unassessed_bearing_round_trips_as_none(tmp_path: Path) -> None:
    """The no-field case at the persistence layer: an uncertainty resolved
    with no bearing (e.g. the dedup path, or an older prompt whose output
    predates this field entirely) stores and reloads as None, not as a
    missing/erroring key."""
    repo = await _repo(tmp_path, "bearing_round_trip_none")
    obj = Objective(
        description="Test objective",
        clarified_question="Test?",
        question_type="verificatory",
        claim_to_verify="Test?",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    u = Uncertainty(
        objective_id=obj.entity_id,
        description="A blocking concern.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        affected_claim_ids=["c1"],
    )
    u.resolve("Deduplicated: same theme as [other].")  # no bearing kwarg
    await repo.save(u)

    reloaded = await repo.get("uncertainty", u.entity_id)
    assert isinstance(reloaded, Uncertainty)
    assert reloaded.resolution is not None
    assert reloaded.resolution_bearing is None


async def test_unresolved_uncertainty_round_trips_with_no_bearing(
    tmp_path: Path,
) -> None:
    """An uncertainty that was never resolved never had a bearing."""
    repo = await _repo(tmp_path, "bearing_round_trip_unresolved")
    obj = Objective(
        description="Test objective",
        clarified_question="Test?",
        question_type="verificatory",
        claim_to_verify="Test?",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    u = Uncertainty(
        objective_id=obj.entity_id,
        description="Still open.",
        uncertainty_type=UncertaintyType.UNKNOWN,
        affected_claim_ids=["c1"],
    )
    await repo.save(u)

    reloaded = await repo.get("uncertainty", u.entity_id)
    assert isinstance(reloaded, Uncertainty)
    assert reloaded.resolution is None
    assert reloaded.resolution_bearing is None


# ── ResolveUncertaintyOperation stores the bearing on save ──────────────


class _FakeAgentRunner:
    """Duck-typed AgentRunner returning a scripted ResolveUncertaintyOutput."""

    def __init__(self, output: ResolveUncertaintyOutput) -> None:
        self._output = output

    async def run(self, agent_name: str, **kwargs):
        return self._output


async def test_operation_stores_bearing_on_the_uncertainty_and_siblings(
    tmp_path: Path, monkeypatch
) -> None:
    repo = await _repo(tmp_path, "bearing_operation")
    obj = Objective(
        description="Test objective",
        clarified_question="Test?",
        question_type="verificatory",
        claim_to_verify="Test?",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="Study A and study B agree on dose-response.",
        scope="test",
    )
    await repo.save(claim)

    target = Uncertainty(
        objective_id=obj.entity_id,
        description="The dose in study A looks inconsistent with study B.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    await repo.save(target)

    sibling = Uncertainty(
        objective_id=obj.entity_id,
        description="The dose reported in study A conflicts with study B.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    await repo.save(sibling)

    output = ResolveUncertaintyOutput(
        uncertainty_id=target.entity_id,
        can_resolve=True,
        resolution="Study B's dose was a documented typo; both agree.",
        bearing="supports_claim",
        remaining_concerns=[],
    )

    op = ResolveUncertaintyOperation(
        repo=repo,
        agent_runner=_FakeAgentRunner(output),
        embedding_model="test-embed",
    )

    async def _fake_embed_texts(texts, *, model):
        # Same vector for near-duplicate texts -> cosine similarity 1.0,
        # well above DEDUP_SIMILARITY_THRESHOLD, so the sibling groups.
        return [[1.0, 0.0] for _ in texts]

    import andamentum.epistemic.embeddings as embeddings_mod

    monkeypatch.setattr(embeddings_mod, "embed_texts", _fake_embed_texts)

    result = await op.execute(
        OperationInput(
            entity_id=target.entity_id, entity_type="uncertainty", operation="resolve"
        )
    )

    assert result.success

    reloaded_target = await repo.get("uncertainty", target.entity_id)
    reloaded_sibling = await repo.get("uncertainty", sibling.entity_id)
    assert isinstance(reloaded_target, Uncertainty)
    assert isinstance(reloaded_sibling, Uncertainty)
    assert reloaded_target.resolution_bearing == "supports_claim"
    assert reloaded_sibling.resolution_bearing == "supports_claim"


async def test_operation_stores_none_when_output_omits_the_bearing_key(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end: an older-prompt/PromptedOutput-fallback output with no
    `bearing` key at all must round-trip as `resolution_bearing is None`
    on the reloaded entity — not as `"neutral"`."""
    repo = await _repo(tmp_path, "bearing_omitted_key")
    obj = Objective(
        description="Test objective",
        clarified_question="Test?",
        question_type="verificatory",
        claim_to_verify="Test?",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="Study A and study B agree on dose-response.",
        scope="test",
    )
    await repo.save(claim)

    target = Uncertainty(
        objective_id=obj.entity_id,
        description="The dose in study A looks inconsistent with study B.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[claim.entity_id],
    )
    await repo.save(target)

    output = ResolveUncertaintyOutput.model_validate(
        {
            "uncertainty_id": target.entity_id,
            "can_resolve": True,
            "resolution": "Study B's dose was a documented typo; both agree.",
            "remaining_concerns": [],
        }
    )
    assert output.bearing is None  # the omitted-key case this test exercises

    op = ResolveUncertaintyOperation(
        repo=repo,
        agent_runner=_FakeAgentRunner(output),
        embedding_model="test-embed",
    )

    async def _fake_embed_texts(texts, *, model):
        return [[1.0, 0.0] for _ in texts]

    import andamentum.epistemic.embeddings as embeddings_mod

    monkeypatch.setattr(embeddings_mod, "embed_texts", _fake_embed_texts)

    result = await op.execute(
        OperationInput(
            entity_id=target.entity_id, entity_type="uncertainty", operation="resolve"
        )
    )
    assert result.success

    reloaded_target = await repo.get("uncertainty", target.entity_id)
    assert isinstance(reloaded_target, Uncertainty)
    assert reloaded_target.resolution is not None
    assert reloaded_target.resolution_bearing is None


async def test_operation_unresolvable_branch_stores_no_bearing(
    tmp_path: Path,
) -> None:
    repo = await _repo(tmp_path, "bearing_unresolvable")
    obj = Objective(
        description="Test objective",
        clarified_question="Test?",
        question_type="verificatory",
        claim_to_verify="Test?",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    target = Uncertainty(
        objective_id=obj.entity_id,
        description="Cannot be resolved with what we have.",
        uncertainty_type=UncertaintyType.CONTRADICTION,
        affected_claim_ids=[],
    )
    await repo.save(target)

    output = ResolveUncertaintyOutput(
        uncertainty_id=target.entity_id,
        can_resolve=False,
        resolution="",
        bearing="neutral",
        remaining_concerns=[],
    )
    op = ResolveUncertaintyOperation(
        repo=repo, agent_runner=_FakeAgentRunner(output), embedding_model="test-embed"
    )

    result = await op.execute(
        OperationInput(
            entity_id=target.entity_id, entity_type="uncertainty", operation="resolve"
        )
    )
    assert result.success
    assert result.did_work is False

    reloaded = await repo.get("uncertainty", target.entity_id)
    assert isinstance(reloaded, Uncertainty)
    assert reloaded.resolution is not None
    assert reloaded.resolution_bearing is None
