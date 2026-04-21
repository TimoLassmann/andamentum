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
