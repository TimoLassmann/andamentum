"""Import-level and argument-validation smoke tests for the orchestrator.

Agent-backed tests require LLM credentials and run only under the
integration test target. See test_end_to_end.py (Task 10).
"""

import pytest

from andamentum.whetstone.orchestrator import ReviewResult, sharpen_document


def test_review_result_defaults():
    r = ReviewResult(task="edit")
    assert r.task == "edit"
    assert r.patches == []
    assert r.issues == []
    assert r.synthesis is None
    assert r.disciplines == []
    assert r.expert_profiles == []
    assert r.expert_reviews == []


async def test_sharpen_document_rejects_invalid_task():
    with pytest.raises(ValueError, match="Invalid task"):
        await sharpen_document("hello", task="bogus")


def test_review_result_checklist_default():
    r = ReviewResult(task="checklist")
    assert r.checklist == []


def test_review_result_checklist_accepts_items():
    from andamentum.whetstone.models import ChecklistItem
    items = [ChecklistItem(name="x", status="pass", notes="")]
    r = ReviewResult(task="checklist", checklist=items)
    assert len(r.checklist) == 1
