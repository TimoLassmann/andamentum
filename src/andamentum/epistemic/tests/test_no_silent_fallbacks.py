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
