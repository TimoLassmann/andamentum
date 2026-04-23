"""Tests for ChecklistItem and BaselineCheck models."""

import pytest
from pydantic import ValidationError

from andamentum.whetstone.models import BaselineCheck, ChecklistItem


def test_checklist_item_minimal():
    item = ChecklistItem(name="Abstract word count", status="pass", notes="240 words on page 1")
    assert item.name == "Abstract word count"
    assert item.status == "pass"
    assert item.notes == "240 words on page 1"
    assert item.category == ""
    assert item.source == "baseline"


def test_checklist_item_all_fields():
    item = ChecklistItem(
        name="Keywords present", status="fail", notes="No keywords section",
        category="hygiene", source="baseline",
    )
    assert item.category == "hygiene"
    assert item.source == "baseline"


def test_checklist_item_rejects_bad_status():
    with pytest.raises(ValidationError):
        ChecklistItem(name="x", status="maybe", notes="")


def test_checklist_item_rejects_bad_source():
    with pytest.raises(ValidationError):
        ChecklistItem(name="x", status="pass", notes="y", source="other")


def test_baseline_check_deterministic():
    c = BaselineCheck(
        name="All figures referenced", category="figures",
        kind="deterministic", scanner="check_all_figures_referenced",
    )
    assert c.kind == "deterministic"
    assert c.scanner == "check_all_figures_referenced"
    assert c.prompt_hint is None


def test_baseline_check_llm():
    c = BaselineCheck(
        name="Abstract structured", category="abstract",
        kind="llm", prompt_hint="Look for background/methods/results/conclusion.",
    )
    assert c.kind == "llm"
    assert c.prompt_hint is not None
    assert c.scanner is None


def test_baseline_check_rejects_bad_kind():
    with pytest.raises(ValidationError):
        BaselineCheck(name="x", category="y", kind="guess")
