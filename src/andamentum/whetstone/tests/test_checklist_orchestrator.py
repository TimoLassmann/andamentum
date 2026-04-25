"""Tests for the checklist-task orchestrator path."""

from dataclasses import dataclass, field
from typing import Callable

import pytest

from andamentum.whetstone import orchestrator
from andamentum.whetstone.agents.checklist import BASELINE_CHECKS
from andamentum.whetstone.agents.output_models import ExtractedChecklistNames
from andamentum.whetstone.models import ChecklistItem
from andamentum.whetstone.orchestrator import ReviewResult


@dataclass
class _FakeRunner:
    calls: list = field(default_factory=list)
    extractor_items: list[str] | None = None
    evaluator_factory: Callable[[str], ChecklistItem] | None = None
    is_local: bool = False  # cloud model by default; tests don't care about ordering

    async def run(self, defn, **kwargs):  # noqa: ANN001
        self.calls.append((defn.name, kwargs))
        if defn.name == "journal_guidelines_extractor":
            return ExtractedChecklistNames(items=self.extractor_items or [])
        if defn.name == "checklist_item_evaluator":
            assert self.evaluator_factory is not None
            return self.evaluator_factory(kwargs["check_name"])
        raise AssertionError(f"Unexpected agent call: {defn.name}")


def _ok_evaluator(check_name: str) -> ChecklistItem:
    return ChecklistItem(name=check_name, status="pass", notes="looks fine")


# Sample document that satisfies the deterministic baseline checks
_GOOD_DOC = """\
Jane Doe
Department of Computer Science, University of Somewhere

Keywords: reproducibility, methodology

Abstract: short abstract.

Figure 1: The setup.
Figure 2: The result.

Body references Figure 1 and Figure 2 in turn.

Table 1: Data.

Body references Table 1.

We had 50 participants. IRB approval obtained.
Conflict of interest: none.
Data availability: on request.
This work was supported by NIH grant X.

As shown [1] and [2].

References
[1] First.
[2] Second.
"""


async def test_checklist_baseline_only():
    runner = _FakeRunner(evaluator_factory=_ok_evaluator)
    result = ReviewResult(task="checklist")
    await orchestrator._run_checklist(
        runner,  # type: ignore[arg-type]
        result,
        _GOOD_DOC,
        guidelines=None,
        verbose=False,
    )

    assert len(result.checklist) == len(BASELINE_CHECKS)
    # All items tagged as baseline
    assert all(item.source == "baseline" for item in result.checklist)
    # Category tagging flowed through
    categories = {item.category for item in result.checklist}
    assert "abstract" in categories
    assert "figures" in categories
    # No journal-extractor calls
    assert not any(name == "journal_guidelines_extractor" for name, _ in runner.calls)


async def test_checklist_baseline_plus_journal():
    runner = _FakeRunner(
        extractor_items=["Funding disclosure complete", "Preprint policy respected"],
        evaluator_factory=_ok_evaluator,
    )
    result = ReviewResult(task="checklist")
    guidelines = "Short guidelines text."
    await orchestrator._run_checklist(
        runner,  # type: ignore[arg-type]
        result,
        _GOOD_DOC,
        guidelines=guidelines,
        verbose=False,
    )

    baseline_count = len(BASELINE_CHECKS)
    assert len(result.checklist) == baseline_count + 2
    journal_items = [i for i in result.checklist if i.source == "journal"]
    assert len(journal_items) == 2
    assert all(i.category == "journal" for i in journal_items)
    # Journal item names are what the extractor returned
    journal_names = {i.name for i in journal_items}
    assert "Funding disclosure complete" in journal_names


async def test_checklist_journal_item_failure_becomes_unclear():
    # First journal item fails, second succeeds
    def flaky(check_name: str) -> ChecklistItem:
        if check_name == "Will fail":
            raise RuntimeError("model timeout")
        return _ok_evaluator(check_name)

    runner = _FakeRunner(
        extractor_items=["Will fail", "Will succeed"],
        evaluator_factory=flaky,
    )
    result = ReviewResult(task="checklist")
    await orchestrator._run_checklist(
        runner,  # type: ignore[arg-type]
        result,
        _GOOD_DOC,
        guidelines="x",
        verbose=False,
    )

    journal_items = [i for i in result.checklist if i.source == "journal"]
    assert len(journal_items) == 2
    failed = next(i for i in journal_items if i.name == "Will fail")
    assert failed.status == "unclear"
    assert "model timeout" in failed.notes


async def test_checklist_baseline_evaluator_failure_raises():
    def always_fail(check_name: str) -> ChecklistItem:
        raise RuntimeError("hard failure")

    runner = _FakeRunner(evaluator_factory=always_fail)
    result = ReviewResult(task="checklist")
    with pytest.raises(RuntimeError):
        await orchestrator._run_checklist(
            runner,  # type: ignore[arg-type]
            result,
            _GOOD_DOC,
            guidelines=None,
            verbose=False,
        )


async def test_checklist_llm_item_name_is_authoritative():
    """Orchestrator overwrites whatever name the LLM returned."""

    def drifted(check_name: str) -> ChecklistItem:
        return ChecklistItem(name="DRIFTED NAME", status="pass", notes="")

    runner = _FakeRunner(evaluator_factory=drifted)
    result = ReviewResult(task="checklist")
    await orchestrator._run_checklist(
        runner,  # type: ignore[arg-type]
        result,
        _GOOD_DOC,
        guidelines=None,
        verbose=False,
    )

    for item in result.checklist:
        assert item.name != "DRIFTED NAME"
