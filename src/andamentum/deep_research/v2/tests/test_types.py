"""The v2 records: validation, content addressing, and the result's exports."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from andamentum.references import LICENCE_NO_REDISTRIBUTE, Work

from ..types import (
    Budget,
    Candidate,
    Finding,
    LogEvent,
    Passage,
    RawFinding,
    ResearchResult,
    RunLog,
    passage_id_for,
)


def test_candidate_needs_a_source() -> None:
    with pytest.raises(ValidationError):
        Candidate(work=Work(), sources=[])


def test_passage_is_content_addressed_and_an_exact_slice() -> None:
    passage = Passage.make(work_id="w_1", start=10, text="Some verbatim text.")
    assert passage.end == 10 + len("Some verbatim text.")
    assert passage.passage_id == passage_id_for("w_1", "Some verbatim text.")
    other_work = Passage.make(work_id="w_2", start=10, text="Some verbatim text.")
    assert other_work.passage_id != passage.passage_id
    with pytest.raises(ValidationError, match="span"):
        Passage(
            passage_id=passage.passage_id,
            work_id="w_1",
            start=0,
            end=5,
            text="Some verbatim text.",
        )
    with pytest.raises(ValidationError, match="sha256"):
        Passage(passage_id="x", work_id="w_1", start=0, end=4, text="Some")


def test_a_snippet_follows_the_licence_of_its_work() -> None:
    restricted = Work(
        title="A restricted record",
        abstract="Text a licence forbids exporting.",
        licence=LICENCE_NO_REDISTRIBUTE,
    )
    assert restricted.redistributable is False
    candidate = Candidate(work=restricted, sources=["s2"], snippet="Text a licence")
    assert '"snippet":""' in candidate.model_dump_json().replace(" ", "")


def test_a_finding_is_numbered_from_a_raw_one() -> None:
    raw = RawFinding(
        subquestion_id="Q1",
        claim="Apollo cut latency by half.",
        quote="Apollo cut latency by half in the trial.",
        quote_verified=False,
        passage_id="p1",
        work_id="w1",
    )
    finding = Finding.from_raw(raw, number=7)
    assert finding.finding_id == "F7" and finding.quote_verified is False
    with pytest.raises(ValidationError):
        Finding(
            finding_id="x",
            subquestion_id="Q1",
            claim="c",
            quote="q",
            passage_id="p",
            work_id="w",
        )


def test_a_finding_from_an_old_file_is_verified_by_default() -> None:
    finding = Finding.model_validate(
        {
            "finding_id": "F1",
            "subquestion_id": "Q1",
            "claim": "c",
            "quote": "q",
            "passage_id": "p",
            "work_id": "w",
            "criterion_ids": ["Q1-C1"],  # a removed field is ignored
        }
    )
    assert finding.quote_verified is True


def test_the_budget_defaults_are_the_loop_defaults() -> None:
    budget = Budget()
    assert (budget.max_rounds, budget.reads_per_round, budget.window_chars) == (
        3,
        5,
        12_000,
    )
    assert budget.concurrency == 4
    with pytest.raises(ValidationError):
        Budget(window_chars=10)


def test_cited_work_ids_follow_the_report_order() -> None:
    def finding(number: int, work_id: str) -> Finding:
        return Finding(
            finding_id=f"F{number}",
            subquestion_id="Q1",
            claim=f"claim {number}",
            quote=f"quote {number}",
            passage_id=f"p{number}",
            work_id=work_id,
        )

    result = ResearchResult(
        run_id="r",
        question="q",
        mode="web",
        findings=[finding(1, "wa"), finding(2, "wb"), finding(3, "wa")],
        report_markdown="Second first [F2]. Then [F3, F1].",
    )
    assert result.cited_work_ids() == ["wb", "wa"]
    assert result.model_copy(update={"report_markdown": ""}).cited_work_ids() == [
        "wa",
        "wb",
    ]


def test_an_old_result_file_loads_with_the_removed_keys_ignored() -> None:
    old = (
        '{"run_id": "r", "question": "q", "mode": "web", "config_name": "hybrid", '
        '"criteria": {}, "coverage": [], "conflicts": [], "screening": []}'
    )
    assert ResearchResult.model_validate_json(old).run_id == "r"


def test_the_log_round_trips_and_renders_tables() -> None:
    log = RunLog(
        run_id="r1",
        events=[
            LogEvent(
                seq=0,
                t=0.0,
                step="Choose",
                event="chosen_work",
                message="m",
                data={
                    "subquestion": "Q1",
                    "title": "A",
                    "host": "a.org",
                    "chosen": True,
                },
            ),
            LogEvent(
                seq=1,
                t=1.0,
                step="Choose",
                event="chosen_work",
                message="m",
                data={
                    "subquestion": "Q1",
                    "title": "B",
                    "host": "b.org",
                    "chosen": False,
                },
            ),
        ],
    )
    assert RunLog.from_jsonl(log.to_jsonl()) == log
    markdown = log.to_markdown()
    assert "## Choose" in markdown and "| Q1 | A | a.org | True |" in markdown
