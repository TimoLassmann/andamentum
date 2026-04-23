"""Smoke tests for whetstone.issues (DocumentIssue)."""

import pytest
from pydantic import ValidationError

from andamentum.whetstone.issues import DocumentIssue


def test_issue_basic():
    i = DocumentIssue(
        issue_type="major",
        category="methodology",
        title="Missing control group",
        description="The experimental design lacks a proper control.",
        agent_type="methodology",
    )
    assert i.issue_type == "major"
    assert i.issue_id
    assert len(i.issue_id) == 8
    assert i.priority == "medium"  # default


def test_issue_type_must_be_valid():
    with pytest.raises(ValidationError):
        DocumentIssue(
            issue_type="catastrophic",  # not in Literal
            category="x",
            title="t",
            description="d",
            agent_type="a",
        )
