from __future__ import annotations

import pytest
from pydantic import ValidationError

from andamentum.compose.log import ActionLog
from andamentum.compose.schemas import (
    Brief,
    CheckResult,
    ComposeResult,
    Outline,
    ReaderQuestion,
    RoundRecord,
    Roles,
    SectionPlan,
    WorkItem,
)


def _brief() -> Brief:
    return Brief(
        task="Write to Jane Doe about the Apollo Labs pilot",
        genre="email",
        title="Apollo pilot",
        audience="Jane Doe",
        purpose="Confirm the pilot date",
        key_messages=["Pilot starts 1 March"],
        must_include=["1 March"],
        constraints=[],
        word_range=(50, 300),
        reader_questions=[ReaderQuestion(question="When?", expected_answer="1 March")],
    )


def test_roles_all_and_frozen():
    r = Roles.all("test:m")
    assert r.drafter == r.critic == r.judge == r.reader == "test:m"
    with pytest.raises(ValidationError):
        r.drafter = "x"  # type: ignore[misc]


def test_brief_frozen_and_roundtrip():
    b = _brief()
    assert Brief.model_validate_json(b.model_dump_json()) == b
    assert b.word_range == (50, 300)
    with pytest.raises(ValidationError):
        b.title = "x"  # type: ignore[misc]


def test_result_roundtrip(tmp_path):
    rr = RoundRecord(
        round=1,
        checks=[CheckResult(name="length", passed=True, details=[])],
        work=[WorkItem(level="section", section_id="s01", reason="r", source="length")],
        changed_sections=["s01"],
        goal_met=False,
    )
    res = ComposeResult(
        bundle=tmp_path,
        outputs=[tmp_path / "out" / "email.txt"],
        goal_met=False,
        stop_reason="cap_reached",
        rounds=[rr],
        open_questions=[],
        brief=_brief(),
        usage={"drafter": 10},
    )
    assert ComposeResult.model_validate_json(res.model_dump_json()) == res


def test_outline_roundtrip():
    o = Outline(
        sections=[
            SectionPlan(
                title="Body",
                purpose="p",
                key_points=["a"],
                fact_ids=["f1"],
                target_words=100,
            )
        ]
    )
    assert Outline.model_validate(o.model_dump()) == o


def test_log_records_and_usage(tmp_path):
    log = ActionLog(tmp_path)
    log.call(
        node="Draft",
        role="drafter",
        model="m",
        prompt="p",
        input_text="i",
        output_text="o",
        input_tokens=5,
        output_tokens=7,
    )
    log.call(
        node="Check",
        role="drafter",
        model="m",
        prompt="p",
        input_text="i",
        output_text="o2",
    )
    log.event(node="Evidence", event="facts_dropped", count=2)
    recs = log.records()
    assert len(recs) == 3 and recs[2]["count"] == 2
    assert set(recs[0]) >= {
        "ts",
        "node",
        "role",
        "model",
        "prompt_sha",
        "input_sha",
        "output_sha",
        "input_tokens",
        "output_tokens",
    }
    assert recs[1]["input_tokens"] is None
    assert log.usage() == {"drafter": 12}
