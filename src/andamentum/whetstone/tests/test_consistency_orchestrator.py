"""Tests for the consistency-task orchestrator path."""

from dataclasses import dataclass
from typing import Any

from andamentum.whetstone import orchestrator
from andamentum.whetstone.agents.output_models import ConsistencyReviewOutput
from andamentum.whetstone.issues import DocumentIssue
from andamentum.whetstone.orchestrator import ReviewResult


@dataclass
class _FakeRunner:
    """Minimal AgentRunner stand-in.

    `returns` maps agent name → output object. Raises KeyError for
    unexpected calls so tests catch unintended dispatch.
    """

    returns: dict[str, Any]
    is_local: bool = False  # cloud model by default

    async def run(self, defn, **kwargs):  # noqa: ANN001
        return self.returns[defn.name]


async def test_consistency_merges_scanner_and_llm():
    # LLM returns a tense-drift issue
    llm_out = ConsistencyReviewOutput(
        issues=[
            DocumentIssue(
                issue_type="minor",
                category="consistency",
                title="Tense drift between methods and results",
                description="Methods use past tense; results use present.",
                agent_type="consistency_reviewer",
            ),
        ]
    )
    runner = _FakeRunner(returns={"consistency_reviewer": llm_out})

    result = ReviewResult(task="consistency")
    # Document has Figure 2 before Figure 1 — scanner should flag it
    doc = "First see Figure 2. Later Figure 1 explains."
    await orchestrator._run_consistency(runner, result, doc, verbose=False)  # type: ignore[arg-type]

    assert any(i.agent_type == "scanner:figure_order" for i in result.issues)
    assert any(i.agent_type == "consistency_reviewer" for i in result.issues)
    assert len(result.issues) == 2


async def test_consistency_no_scanner_findings():
    """Clean doc → only LLM issues in result."""
    llm_out = ConsistencyReviewOutput(issues=[])
    runner = _FakeRunner(returns={"consistency_reviewer": llm_out})
    result = ReviewResult(task="consistency")
    doc = "Clean text with no problems."
    await orchestrator._run_consistency(runner, result, doc, verbose=False)  # type: ignore[arg-type]
    assert result.issues == []
