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
        _RaisingOp, cast(EpistemicDeps, deps), state, "claim-7", "claim", "scrutinize_claim"
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
