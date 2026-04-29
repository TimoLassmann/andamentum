"""Tests asserting that silent fallbacks have been removed.

Every test here asserts a single property: a failure in a downstream call
(LLM agent, repo load, provider) either raises out of the operation or is
recorded on the graph state — never silently swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import _run_op
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.base import (
    BaseOperation,
    OperationInput,
    OperationResult,
)


class _RaisingOp(BaseOperation):
    """Test double: always raises."""

    entity_type = "claim"
    raised: Exception = RuntimeError("kaboom")

    async def execute(self, work: OperationInput) -> OperationResult:
        raise self.raised


@dataclass
class _StubDeps:
    """Minimal deps for _run_op — only fields the function reads."""

    repo: Any = None
    agent_runner: Any = None
    evidence_gatherer: Any = None
    quality_scorer: Any = None
    embedding_model: Any = None
    progress_callback: Any = None


@pytest.mark.asyncio
async def test_run_op_quarantines_entity_on_exception():
    state = EpistemicGraphState(objective_id="obj-1", question="q")
    deps = _StubDeps()

    result = await _run_op(
        _RaisingOp,
        cast(EpistemicDeps, deps),
        state,
        "claim-7",
        "claim",
        "scrutinize_claim",
    )

    # The result is surfaced as success=False (for logging), but the state
    # now carries a quarantine record — no silent degradation.
    assert result.success is False
    assert state.is_quarantined("claim-7")
    assert len(state.quarantined) == 1
    record = state.quarantined[0]
    assert record.entity_id == "claim-7"
    assert record.entity_type == "claim"
    assert record.operation == "scrutinize_claim"
    assert record.exception_type == "RuntimeError"
    assert "kaboom" in record.message


def test_epistemic_result_has_quarantined_field():
    from andamentum.epistemic.graph.result import EpistemicResult

    result = EpistemicResult(objective_id="obj-1", status="partial")
    # Default: empty list, not None
    assert result.quarantined == []


def test_pipeline_result_has_quarantined_field():
    from andamentum.epistemic.operations_runner import PipelineResult

    result = PipelineResult(
        objective_id="obj-1",
        iterations=0,
        successful=0,
        failed=0,
        status="partial",
    )
    assert result.quarantined == []


@pytest.mark.asyncio
async def test_propose_claims_propagates_screening_failure(tmp_path):
    """When epistemic_screen_relevance raises, ProposeClaimsOperation must
    propagate — the previous behavior (include-by-default) silently poisoned
    downstream evidence selection with unscreened items."""
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.entities import Evidence, Objective
    from andamentum.epistemic.operations.claims import ProposeClaimsOperation
    from andamentum.epistemic.repository import EpistemicRepository

    class _RaisingScreenRunner:
        async def run(self, agent_name: str, **kwargs):
            if agent_name == "epistemic_screen_relevance":
                raise RuntimeError("screening model timed out")
            # Other agents shouldn't be reached before screening
            raise AssertionError(
                f"Unexpected agent call {agent_name} before screening failed"
            )

    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="test question", clarified_question="q?")
    obj.objective_id = obj.entity_id  # Objectives are self-referential
    await repo.save(obj)
    ev = Evidence(
        objective_id=obj.entity_id,
        source_type="web_search",
        source_ref="http://example.org/x",
        extracted=True,
        extracted_content="some content",
    )
    await repo.save(ev)

    op = ProposeClaimsOperation(
        repo=repo,
        agent_runner=_RaisingScreenRunner(),
        evidence_gatherer=None,
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="screening model timed out"):
        await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="propose_claims",
            )
        )


@pytest.mark.asyncio
async def test_adversarial_check_propagates_counterquery_failure(tmp_path):
    """One failing framing must propagate. Previous: silently dropped to 2/3.
    New: the claim gets quarantined by _run_op."""
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.entities import Claim, Objective
    from andamentum.epistemic.entities.claim import ClaimStage
    from andamentum.epistemic.operations.verification import AdversarialSearchOperation
    from andamentum.epistemic.repository import EpistemicRepository

    class _OneFramingRaisesRunner:
        def __init__(self):
            self.calls = 0

        async def run(self, agent_name: str, **kwargs):
            if agent_name == "epistemic_generate_counterquery":
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("framing 2 failed")
                from types import SimpleNamespace

                return SimpleNamespace(query=f"q-{self.calls}", framing="test")
            raise AssertionError(f"Unexpected agent {agent_name}")

    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="q", clarified_question="q")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claim = Claim(
        objective_id=obj.entity_id,
        statement="X causes Y",
        scope="specific",
        stage=ClaimStage.HYPOTHESIS,
    )
    await repo.save(claim)

    op = AdversarialSearchOperation(
        repo=repo,
        agent_runner=_OneFramingRaisesRunner(),
        evidence_gatherer=None,
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="framing 2 failed"):
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="adversarial_check",
            )
        )


@pytest.mark.asyncio
async def test_adversarial_check_propagates_counterarg_eval_failure(tmp_path):
    """When epistemic_evaluate_counterargument raises on any hit, the
    operation must raise — do not build a default-scored Counterargument."""
    from types import SimpleNamespace

    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.entities import Claim, Objective
    from andamentum.epistemic.entities.claim import ClaimStage
    from andamentum.epistemic.operations.base import GatheredEvidence
    from andamentum.epistemic.operations.verification import AdversarialSearchOperation
    from andamentum.epistemic.repository import EpistemicRepository

    class _EvalFailsRunner:
        """Returns valid counterquery results, but raises on evaluator."""

        async def run(self, agent_name: str, **kwargs):
            if agent_name == "epistemic_generate_counterquery":
                return SimpleNamespace(
                    query="failed replication of X causes Y",
                    framing="replication_failures",
                )
            if agent_name == "epistemic_evaluate_counterargument":
                raise RuntimeError("evaluator failed")
            raise AssertionError(f"Unexpected agent call: {agent_name}")

    class _StubGatherer:
        """Returns one hit so evaluation is actually invoked."""

        async def gather(self, provider: str, query: str):
            return [
                GatheredEvidence(
                    content="Study finds no evidence for X causing Y",
                    source_ref="https://example.com/study",
                    source_type="web_search",
                )
            ]

    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="test question", clarified_question="q?")
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claim = Claim(
        objective_id=obj.entity_id,
        statement="X causes Y",
        scope="specific",
        stage=ClaimStage.HYPOTHESIS,
    )
    await repo.save(claim)

    op = AdversarialSearchOperation(
        repo=repo,
        agent_runner=_EvalFailsRunner(),
        evidence_gatherer=_StubGatherer(),  # type: ignore[arg-type]
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="evaluator failed"):
        from andamentum.epistemic.operations.base import OperationInput

        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="adversarial_check",
            )
        )


async def test_extract_evidence_raises_without_runner_or_gatherer(tmp_path):
    """When neither an agent runner nor a gatherer is wired up, extraction
    must raise — never fabricate `[Content from ...]` placeholders."""
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.entities import Evidence
    from andamentum.epistemic.operations.evidence import ExtractEvidenceOperation
    from andamentum.epistemic.repository import EpistemicRepository

    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    ev = Evidence(
        objective_id="obj-1",
        source_type="web_search",
        source_ref="http://example.org/paper",
    )
    await repo.save(ev)

    op = ExtractEvidenceOperation(
        repo=repo,
        agent_runner=None,  # no runner
        evidence_gatherer=None,  # no gatherer
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="no extractor"):
        await op.execute(
            OperationInput(
                entity_id=ev.entity_id,
                entity_type="evidence",
                operation="extract_evidence",
            )
        )


@pytest.mark.asyncio
async def test_domain_classifier_failure_propagates(tmp_path):
    """When epistemic_classify_evidence_domain raises, AssessConvergenceOperation
    must propagate — never silently fall back to default_classify."""
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.entities import Claim, Evidence, Objective
    from andamentum.epistemic.entities.claim import ClaimStage
    from andamentum.epistemic.operations.verification import AssessConvergenceOperation
    from andamentum.epistemic.repository import EpistemicRepository

    class _ClassifierFailsRunner:
        """Returns a valid response for any other agent, raises on domain classifier."""

        async def run(self, agent_name: str, **kwargs):
            if agent_name == "epistemic_classify_evidence_domain":
                raise RuntimeError("classifier failed")
            raise AssertionError(f"Unexpected agent call: {agent_name}")

    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="test question", clarified_question="q?")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="X causes Y",
        scope="specific",
        stage=ClaimStage.HYPOTHESIS,
    )
    await repo.save(claim)

    # Evidence must have extracted_content and a cluster_status that is NOT
    # "corroborative" or "deferred" — those are skipped before the classifier runs.
    ev = Evidence(
        objective_id=obj.entity_id,
        source_type="web_search",
        source_ref="http://example.org/study",
        extracted=True,
        extracted_content="Study finds evidence for X causing Y",
        cluster_status="primary",
    )
    ev_id = ev.entity_id
    await repo.save(ev)

    # Link evidence to claim
    claim.evidence_ids.append(ev_id)
    await repo.save(claim)

    op = AssessConvergenceOperation(
        repo=repo,
        agent_runner=_ClassifierFailsRunner(),
        evidence_gatherer=None,
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="classifier failed"):
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="assess_convergence",
            )
        )


@pytest.mark.asyncio
async def test_promote_claim_propagates_objective_load_failure(tmp_path):
    """When the objective can't be loaded, promotion must raise — silently
    falling back to default thresholds could promote claims under the wrong
    routing profile."""
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic import EntityNotFoundError
    from andamentum.epistemic.entities import Claim
    from andamentum.epistemic.entities.claim import ClaimStage
    from andamentum.epistemic.operations.base import OperationInput
    from andamentum.epistemic.operations.stage_management import PromoteClaimOperation
    from andamentum.epistemic.repository import EpistemicRepository

    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    # Save a claim whose objective_id points to nothing in the repo.
    # The try/except in stage_management.py previously swallowed the load error;
    # now it must propagate before even reaching gate validation.
    claim = Claim(
        objective_id="does-not-exist",
        statement="X causes Y",
        scope="specific",
        stage=ClaimStage.HYPOTHESIS,
    )
    await repo.save(claim)

    op = PromoteClaimOperation(
        repo=repo,
        agent_runner=None,
        evidence_gatherer=None,
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(EntityNotFoundError):
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="promote_claim",
            )
        )


@pytest.mark.asyncio
async def test_run_epistemic_graph_propagates_posterior_failure(monkeypatch, tmp_path):
    """When compute_posterior raises, run_epistemic_graph must propagate —
    a silent None posterior hides the headline result."""
    from andamentum.epistemic.graph import run_epistemic_graph
    from andamentum.epistemic.graph.result import EpistemicResult

    # Fake graph result with successful=1 so the posterior branch is reached.
    fake_epistemic_result = EpistemicResult(
        objective_id="obj-fake",
        status="complete",
        successful=1,
        failed=0,
    )

    class _FakeGraphRunResult:
        output = fake_epistemic_result

    class _FakeGraph:
        async def run(self, *args, **kwargs):
            return _FakeGraphRunResult()

    # Patch the graph object inside nodes so run_epistemic_graph uses our fake.
    monkeypatch.setattr(
        "andamentum.epistemic.graph.nodes.epistemic_graph",
        _FakeGraph(),
    )

    # Patch compute_posterior to raise.
    async def _raising_posterior(*args, **kwargs):
        raise RuntimeError("posterior computation failed")

    monkeypatch.setattr(
        "andamentum.epistemic.confidence.compute_posterior",
        _raising_posterior,
    )

    with pytest.raises(RuntimeError, match="posterior computation failed"):
        await run_epistemic_graph(
            question="Does X cause Y?",
            database_name="test_posterior_failure",
            model="test:stub",
            provider="none",
            providers={},
            db_dir=str(tmp_path),
        )


@pytest.mark.asyncio
async def test_convergence_evidence_load_failure_propagates(tmp_path):
    """When repo.get("evidence", eid) raises during the convergence loop,
    AssessConvergenceOperation must propagate — never silently skip the item
    and compute convergence on a partial evidence set."""
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.entities import Claim, Objective
    from andamentum.epistemic.entities.claim import ClaimStage
    from andamentum.epistemic.operations.base import OperationInput
    from andamentum.epistemic.operations.verification import AssessConvergenceOperation
    from andamentum.epistemic.repository import EpistemicRepository

    class _NeverCalledRunner:
        """Asserts that no agent is called — operation should raise before
        reaching the classifier."""

        async def run(self, agent_name: str, **kwargs):
            raise AssertionError(
                f"Agent {agent_name!r} called; expected operation to raise first"
            )

    class _RaiseOnGetEvidenceRepo:
        """Wraps a real repo but raises RuntimeError on any get("evidence", …)."""

        def __init__(self, real_repo: EpistemicRepository) -> None:
            self._real = real_repo

        async def get(self, kind: str, eid: str):
            if kind == "evidence":
                raise RuntimeError("storage backend error")
            return await self._real.get(kind, eid)

        def __getattr__(self, name: str):
            return getattr(self._real, name)

    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    real_repo = EpistemicRepository(store)

    obj = Objective(description="test question", clarified_question="q?")
    obj.objective_id = obj.entity_id
    await real_repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="X causes Y",
        scope="specific",
        stage=ClaimStage.HYPOTHESIS,
    )
    # Attach a bogus evidence_id — it will never exist in the store.
    claim.evidence_ids.append("bogus-evidence-id-does-not-exist")
    await real_repo.save(claim)

    stub_repo = _RaiseOnGetEvidenceRepo(real_repo)

    op = AssessConvergenceOperation(
        repo=stub_repo,  # type: ignore[arg-type]
        agent_runner=_NeverCalledRunner(),
        evidence_gatherer=None,
        quality_scorer=None,
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="storage backend error"):
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="assess_convergence",
            )
        )


@pytest.mark.asyncio
async def test_score_evidence_raises_when_no_scorer_available(tmp_path):
    """_score_evidence previously fabricated quality_score=0.1 ("default_minimum")
    when no scorer succeeded. That silently polluted posteriors with a fake
    quality signal. Now it raises so the failure is visible."""
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.entities import Evidence
    from andamentum.epistemic.operations.base import GatheredEvidence
    from andamentum.epistemic.operations.evidence import ExtractEvidenceOperation
    from andamentum.epistemic.repository import EpistemicRepository

    class _UnscoredGatherer:
        """Returns content but no quality_score — exactly the input shape that
        used to land in Path 4's default_minimum fabrication."""

        async def gather(self, source_type: str, query: str):
            return [
                GatheredEvidence(
                    content="some extracted passage about the claim",
                    source_ref=query,
                    source_type=source_type,
                    quality_score=None,  # the whole point
                )
            ]

    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    ev = Evidence(
        objective_id="obj-1",
        source_type="web_search",
        source_ref="http://example.org/paper",
    )
    await repo.save(ev)

    op = ExtractEvidenceOperation(
        repo=repo,
        agent_runner=None,  # no agent scoring
        evidence_gatherer=_UnscoredGatherer(),
        quality_scorer=None,  # no OpenAlex
        embedding_model=None,
    )
    with pytest.raises(RuntimeError, match="no scorer available"):
        await op.execute(
            OperationInput(
                entity_id=ev.entity_id,
                entity_type="evidence",
                operation="extract_evidence",
            )
        )


# ── Deferred-cluster visibility ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_select_top_k_evidence_promotes_every_cluster(monkeypatch, tmp_path):
    """select_top_k_evidence must produce a representative for every cluster.

    The old top-K cap discarded clusters past position K (tagging them
    deferred), which meant most of the gathered evidence never reached the
    posterior. After A4, clustering's job is to enforce independence in the
    count, not to discard work. deferred_count is always 0; LLM cost
    bounding lives in the consumers (LLM_PANEL_CAP).
    """
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.dedup import EvidenceCluster
    from andamentum.epistemic.entities import Evidence
    from andamentum.epistemic.operations.claims import select_top_k_evidence
    from andamentum.epistemic.repository import EpistemicRepository

    store = DocumentStore.for_database("test_no_defer", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    evidences: list[Evidence] = []
    for i in range(8):
        ev = Evidence(
            objective_id="obj-1",
            source_type="web_search",
            source_ref=f"http://example.org/ev{i}",
            extracted=True,
            extracted_content=f"Evidence content {i}",
            quality_score=0.5,
        )
        await repo.save(ev)
        evidences.append(ev)

    async def _stub_dedup(texts, min_cluster_size=2, *, embedding_model):
        return [
            EvidenceCluster(
                medoid_index=i,
                representative_indices=[i],
                member_indices=[i],
                count=1,
            )
            for i in range(len(texts))
        ]

    monkeypatch.setattr(
        "andamentum.epistemic.operations.claims.deduplicate_evidence", _stub_dedup
    )

    representatives, total_clusters, deferred_count = await select_top_k_evidence(
        repo,
        evidences,
        embedding_model="stub-model",
    )

    assert total_clusters == 8, f"Expected 8 total clusters, got {total_clusters}"
    assert deferred_count == 0, (
        f"Expected 0 deferred clusters (cap retired), got {deferred_count}"
    )
    assert len(representatives) == 8, (
        f"Expected one representative per cluster (8), got {len(representatives)}"
    )


@pytest.mark.asyncio
async def test_propose_claims_surfaces_cluster_count(monkeypatch, tmp_path):
    """ProposeClaimsOperation must report how many clusters it processed.

    The old contract reported a "deferred" count for clusters past the
    top-K cap. The cap is retired (every cluster now contributes
    representatives) so the message reports total clusters processed.
    """
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.dedup import EvidenceCluster
    from andamentum.epistemic.entities import Evidence, Objective
    from andamentum.epistemic.operations.base import OperationInput
    from andamentum.epistemic.operations.claims import ProposeClaimsOperation
    from andamentum.epistemic.repository import EpistemicRepository
    from types import SimpleNamespace

    # Stub deduplicate_evidence to return 8 singleton clusters
    async def _stub_dedup(texts, min_cluster_size=2, *, embedding_model):
        return [
            EvidenceCluster(
                medoid_index=i,
                representative_indices=[i],
                member_indices=[i],
                count=1,
            )
            for i in range(len(texts))
        ]

    monkeypatch.setattr(
        "andamentum.epistemic.operations.claims.deduplicate_evidence", _stub_dedup
    )

    # Stub embed_and_group to return single cluster for assertion clustering
    async def _stub_embed_and_group(texts, threshold=0.7, *, embedding_model):
        return [list(range(len(texts)))]

    monkeypatch.setattr(
        "andamentum.epistemic.similarity.embed_and_group",
        _stub_embed_and_group,
    )

    class _StubRunner:
        """Returns minimal valid outputs for all agents called."""

        async def run(self, agent_name: str, **kwargs):
            if agent_name == "epistemic_screen_relevance":
                return SimpleNamespace(is_relevant=True)
            if agent_name == "epistemic_extract_assertion":
                return SimpleNamespace(assertion="X is associated with Y")
            if agent_name == "epistemic_draft_claim":
                return SimpleNamespace(
                    statement="X is associated with Y",
                    scope="specific",
                    direction="positive",
                )
            if agent_name == "epistemic_judge_evidence":
                return SimpleNamespace(
                    verdict="supports", reasoning="evidence supports"
                )
            # judge_evidence
            return SimpleNamespace(verdict="supports", reasoning="ok")

    store = DocumentStore.for_database("test_deferred_claims", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="Does X cause Y?", clarified_question="Does X cause Y?")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # Create 8 evidence items (will produce 8 clusters, 3 deferred)
    for i in range(8):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="web_search",
            source_ref=f"http://example.org/paper{i}",
            extracted=True,
            extracted_content=f"Evidence content for claim {i}",
            quality_score=0.5,
        )
        await repo.save(ev)

    op = ProposeClaimsOperation(
        repo=repo,
        agent_runner=_StubRunner(),
        evidence_gatherer=None,
        quality_scorer=None,
        embedding_model="stub-model",
    )
    result = await op.execute(
        OperationInput(
            entity_id=obj.entity_id,
            entity_type="objective",
            operation="propose_claims",
        )
    )

    assert result.success is True, f"Expected success, got: {result.message}"
    assert "8 of 8 evidence clusters selected" in result.message, (
        f"Expected message to surface cluster count. Got:\n{result.message}"
    )
    assert "deferred" not in result.message, (
        f"deferred clusters should be retired. Got:\n{result.message}"
    )
