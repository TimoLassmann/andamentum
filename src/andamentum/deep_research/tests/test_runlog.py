"""The run log: explicit steps, whole text, the helpers of the loop's events."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from andamentum.references import Work

from .. import runlog
from ..runlog import FINAL_TOP, OUTPUT_CHARS, TEXT_CHARS, RunLogger, clipped, write_log
from ..types import Candidate, RunLog, SubQuestion, VerifyFlag


def logger() -> RunLogger:
    ticks = iter(float(n) for n in range(1000))
    return RunLogger(clock=lambda: next(ticks), run_id="r_test")


def cand(n: int, *, snippet: str = "") -> Candidate:
    return Candidate(
        work=Work(
            work_id=f"w{n}", title=f"Title {n}", url=f"https://site{n}.example/p"
        ),
        sources=["web"],
        snippet=snippet,
    )


def names(log: RunLogger) -> list[str]:
    return [e.event for e in log.log.events]


def test_an_event_needs_its_step() -> None:
    log = logger()
    with pytest.raises(TypeError):
        log.event("note", "no step")  # type: ignore[call-arg]
    log.event("note", "hello", step="Search", text="t")
    (event,) = log.log.events
    assert (event.seq, event.step, event.event, event.data) == (
        0,
        "Search",
        "note",
        {"text": "t"},
    )


def test_events_are_numbered_and_timed_on_the_runs_clock() -> None:
    log = logger()
    log.event("a", "x", step="Setup")
    log.event("b", "y", step="Plan")
    first, second = log.log.events
    assert (first.seq, second.seq) == (0, 1) and second.t > first.t


def test_the_log_round_trips_through_jsonl_and_renders_markdown(tmp_path: Path) -> None:
    log = logger()
    runlog.subquestions(log, [SubQuestion(id="Q1", text="Why?")])
    runlog.query_hits(
        log,
        [{"subquestion": "Q1", "query": "a b", "source": "web", "count": 2,
          "top": [{"rank": 1, "title": "T"}], "error": ""}],
    )  # fmt: skip
    md, jsonl = write_log(log.snapshot(), md_path=tmp_path / "run.log.md")
    assert jsonl.name == "run.log.jsonl"
    assert RunLog.from_jsonl(jsonl.read_text(encoding="utf-8")) == log.log
    text = md.read_text(encoding="utf-8")
    assert "## Plan" in text and "## Search" in text
    assert "| t (s) | sub-question | query | source | results | top titles |" in text


def test_clipped_cuts_only_beyond_the_limit() -> None:
    assert clipped("abc", limit=3) == ("abc", False)
    assert clipped("abcd", limit=3) == ("abc", True)
    assert TEXT_CHARS == 200_000 and OUTPUT_CHARS == 500


def test_an_llm_call_keeps_500_characters_of_output_and_files_under_its_step() -> None:
    class Out(BaseModel):
        text: str

    log = logger()
    runlog.llm_call(
        log, purpose="Out", step="Notes", model="m", prompt="p" * 10,
        output=Out(text="x" * 2000), seconds=1.234,
    )  # fmt: skip
    runlog.llm_call(
        log, purpose="Out", step="Nonsense", model="m", prompt="p", output=None,
        seconds=0.1, error="boom",
    )  # fmt: skip
    first, second = log.log.events
    assert first.step == "Notes" and len(str(first.data["output"])) == OUTPUT_CHARS
    assert first.data["prompt_chars"] == 10
    assert second.step == "Setup" and second.data["error"] == "boom"
    runlog.llm_call(
        None, purpose="x", step="Plan", model="m", prompt="p", output=None, seconds=0
    )


def test_records_log_degraded_notes_and_skipped_lines() -> None:
    log = logger()
    runlog.records(log, step="Read", degraded=["a"], notes=["b"], skipped=["c"])
    assert names(log) == ["degraded", "note", "skipped"]
    assert {e.step for e in log.log.events} == {"Read"}
    runlog.records(None, step="Read", degraded=["a"])


def test_search_results_are_logged_whole_per_query_and_source() -> None:
    log = logger()
    rows = [
        {
            "rank": 1,
            "url": "https://a.example/",
            "title": "A",
            "host": "a.example",
            "snippet": "s" * 900,
        }
    ]
    runlog.search_results(
        log,
        [{"subquestion": "Q1", "query": "q", "source": "web", "results": rows}],
        round_number=2,
    )
    (event,) = log.log.events
    assert event.step == "Search" and event.data["round"] == 2
    assert event.data["results"] == rows


def test_a_choice_lists_picks_reserves_and_what_was_not_chosen() -> None:
    log = logger()
    shown = [cand(n) for n in range(1, 6)]
    runlog.choice(
        log, subquestion="Q1", round_number=1, shown=shown, picks=shown[:2],
        reserves=shown[2:3], fallback=False,
    )  # fmt: skip
    choice, *rows = log.log.events
    assert choice.event == "choice" and choice.step == "Choose"
    assert choice.data["shown"] == 5 and choice.data["fallback"] is False
    assert [p["work_id"] for p in choice.data["picks"]] == ["w1", "w2"]  # type: ignore[union-attr]
    assert [p["work_id"] for p in choice.data["reserves"]] == ["w3"]  # type: ignore[union-attr]
    assert [p["work_id"] for p in choice.data["not_chosen"]] == ["w4", "w5"]  # type: ignore[union-attr]
    assert choice.data["picks"][0] == {  # type: ignore[index]
        "work_id": "w1", "url": "https://site1.example/p", "title": "Title 1", "host": "site1.example",
    }  # fmt: skip
    assert [r.data["chosen"] for r in rows] == [True, True, False, False, False]
    assert {r.event for r in rows} == {"chosen_work"}


def test_expanded_reflect_and_report_events() -> None:
    log = logger()
    runlog.expanded(
        log,
        subquestion="Q1",
        seeds=[Work(work_id="w1", title="Seed")],
        found=[cand(2), cand(3)],
    )
    runlog.reflected(
        log, subquestion="Q1", round_number=2, answered=True, answer="Likely 45 ms",
        confidence="high", gaps=[], next_queries=[],
    )  # fmt: skip
    runlog.report_written(log, report="# R\n" * 500, cited_works=3, findings=7)
    expanded, reflected, report = log.log.events
    assert (expanded.step, reflected.step, report.step) == (
        "Expand",
        "Reflect",
        "Write",
    )
    assert expanded.data["count"] == 2 and expanded.data["seeds"] == [
        {"work_id": "w1", "title": "Seed"}
    ]
    assert reflected.data["answer"] == "Likely 45 ms" and reflected.data["round"] == 2
    assert report.data["text"] == "# R\n" * 500 and report.data["chars"] == 2000


def test_flags_and_the_admitted_line() -> None:
    log = logger()
    runlog.flags(
        log, [VerifyFlag(kind="unknown_finding", ref="F9", detail="no such finding")]
    )
    runlog.admitted(log, before=2, after=5, step="Expand", via="citation expansion")
    flag, admitted = log.log.events
    assert flag.step == "Verify" and flag.data["ref"] == "F9"
    assert (
        admitted.step == "Expand"
        and admitted.data["new_works"] == 3
        and admitted.data["works_held"] == 5
    )


def test_the_final_ranking_keeps_the_first_twenty_with_basis_and_source_ranks() -> None:
    log = logger()
    works = [Work(work_id=f"w{n}", title=f"T{n}") for n in range(1, 31)]
    runlog.final_ranking(
        log, works=works, basis={"w1": "cited", "w2": "read"},
        source_ranks={"w1": {"web": 1}}, stop_reason="rounds",
    )  # fmt: skip
    (event,) = log.log.events
    assert event.step == "Finish" and event.data["works_ranked"] == 30
    ranked = event.data["works"]
    assert isinstance(ranked, list) and len(ranked) == FINAL_TOP
    assert ranked[0] == {
        "rank": 1,
        "work_id": "w1",
        "title": "T1",
        "basis": "cited",
        "source_ranks": {"web": 1},
    }
    assert [r["basis"] for r in ranked[1:3]] == ["read", "fused"]  # type: ignore[index]
    assert "| 1 | T1 | cited | web 1 |" in log.log.to_markdown()
