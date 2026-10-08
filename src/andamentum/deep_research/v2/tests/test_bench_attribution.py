"""Loss attribution: at which stage did the gold answer disappear?

Synthetic run logs built from ``LogEvent`` over invented questions and answers;
no run, no model, no network."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from andamentum.deep_research.v2.bench import cli
from andamentum.deep_research.v2.bench.arms import ArmOutput
from andamentum.deep_research.v2.bench.attribution import (
    DIAGNOSES,
    STAGES,
    collect_losses,
    trace_litsearch,
    trace_simpleqa,
)
from andamentum.deep_research.v2.bench.experiment import Experiment
from andamentum.deep_research.v2.bench.experiment_run import (
    DataDirs,
    Services,
    run_experiment,
)
from andamentum.deep_research.v2.types import LogEvent, LogStep, RunLog
from andamentum.references import Work

from .test_bench_experiment_run import (
    GOLD_BY_QUESTION,
    ScriptedArm,
    StubGrader,
    correct_behaviour,
    gold_title,
    output,
    write_litsearch,
    write_simpleqa,
)

GOLD = "teal7"
PAGE = "https://kilns.example.org/lids/apollo-7"
OTHER = "https://pots.example.org/other"

STEP: dict[str, LogStep] = {
    "search_results": "Search",
    "choice": "Choose",
    "fetched": "Read",
    "notes_input": "Notes",
    "finding_noted": "Notes",
    "reflect": "Reflect",
    "report": "Write",
    "final_ranking": "Finish",
}


def log_of(*events: tuple[str, dict[str, Any]]) -> RunLog:
    return RunLog(
        run_id="r",
        events=[
            LogEvent(
                seq=n,
                t=float(n),
                step=STEP.get(name, "Setup"),
                event=name,
                message=name,
                data=data,
            )
            for n, (name, data) in enumerate(events)
        ],
    )


def run_log(
    gold: str = GOLD,
    *,
    snippet: bool = True,
    picked: bool = True,
    reserved: bool = False,
    fetched_ok: bool = True,
    page_text: bool = True,
    window: bool = True,
    finding: bool = True,
    reflect: bool = True,
    report: bool = True,
) -> RunLog:
    """A run that carries *gold* through every stage not switched off."""
    row = {"work_id": "w1", "url": PAGE, "title": "Apollo kiln lids", "host": "x"}
    other = {"work_id": "w2", "url": OTHER, "title": "Pots", "host": "y"}
    chosen = {"picks": [row] if picked else [other], "reserves": []}
    if reserved:
        chosen = {"picks": [other], "reserves": [row]}
    page = f"The lid of the kiln is {gold}. " if page_text else "Nothing here. "
    return log_of(
        (
            "search_results",
            {
                "subquestion": "S1",
                "round": 1,
                "query": "kiln lid",
                "source": "web",
                "results": [
                    {
                        "rank": 1,
                        "url": PAGE,
                        "title": "Apollo kiln lids",
                        "host": "x",
                        "snippet": f"Lids come in colours; it is {gold}."
                        if snippet
                        else "Lids come in colours.",
                    },
                    {
                        "rank": 2,
                        "url": OTHER,
                        "title": "Pots",
                        "host": "y",
                        "snippet": "Pots.",
                    },
                ],
            },
        ),
        ("choice", {"subquestion": "S1", "round": 1, "shown": 2, **chosen}),
        (
            "fetched",
            {
                "work_id": "w1",
                "url": PAGE,
                "ok": fetched_ok,
                "text": page if fetched_ok else "",
            },
        ),
        (
            "notes_input",
            {
                "work_id": "w1",
                "url": PAGE,
                "text": page if window else "Some other part of the page. ",
            },
        ),
        (
            "finding_noted",
            {
                "work_id": "w1",
                "claim": f"The lid is {gold}" if finding else "The lid is coloured",
                "quote": "the lid",
                "verified": True,
            },
        ),
        (
            "reflect",
            {
                "subquestion": "S1",
                "answered": True,
                "answer": f"It is {gold}." if reflect else "It is unclear.",
            },
        ),
        (
            "report",
            {"text": f"The lid is {gold} [F1]." if report else "It is unclear."},
        ),
    )


def test_a_run_that_keeps_the_gold_to_the_end_is_present_at_every_stage() -> None:
    trace = trace_simpleqa(run_log(), GOLD)
    assert trace.stages == dict.fromkeys(STAGES, True)
    assert trace.diagnosis == "report_wrong"  # present everywhere: graded wrong anyway


@pytest.mark.parametrize(
    "switches, diagnosis, first_absent",
    [
        ({"snippet": False, "page_text": False}, "search_miss", "search_snippets"),
        (
            {"picked": False, "page_text": False, "fetched_ok": False},
            "not_chosen",
            "fetched_text",
        ),
        ({"fetched_ok": False, "page_text": False}, "unreadable", "fetched_text"),
        ({"window": False}, "window", "notes_input"),
        ({"finding": False}, "notes", "findings"),
        ({"reflect": False}, "reflect", "reflect"),
        ({"report": False}, "write", "report"),
    ],
)
def test_each_diagnosis_names_the_stage_the_gold_first_disappears_at(
    switches: dict[str, Any], diagnosis: str, first_absent: str
) -> None:
    trace = trace_simpleqa(run_log(**switches), GOLD)
    assert trace.diagnosis == diagnosis
    assert next(s for s in STAGES if not trace.stages[s]) == first_absent


def test_the_diagnoses_are_the_ones_the_spec_names_in_order() -> None:
    assert DIAGNOSES == (
        "no_log",
        "search_miss",
        "not_chosen",
        "unreadable",
        "window",
        "notes",
        "reflect",
        "write",
        "report_wrong",
    )
    assert STAGES == (
        "search_snippets",
        "fetched_text",
        "notes_input",
        "findings",
        "reflect",
        "report",
    )


def test_a_gold_in_a_snippet_of_a_page_nobody_chose_is_not_chosen() -> None:
    log = run_log(picked=False, page_text=False, fetched_ok=False)
    assert trace_simpleqa(log, GOLD).diagnosis == "not_chosen"


def test_a_reserve_counts_as_chosen() -> None:
    log = run_log(reserved=True, fetched_ok=False, page_text=False)
    assert trace_simpleqa(log, GOLD).diagnosis == "unreadable"


def test_a_chosen_page_that_was_read_without_the_gold_is_unreadable() -> None:
    # the fetch worked but the page text lacks the gold the snippet showed
    log = run_log(page_text=False)
    assert trace_simpleqa(log, GOLD).diagnosis == "unreadable"


def test_a_url_is_matched_by_its_canonical_form() -> None:
    log = run_log(picked=False, page_text=False, fetched_ok=False)
    chosen = log.events[1].data
    chosen["picks"] = [  # type: ignore[index]
        {"work_id": "w1", "url": PAGE + "/#top", "title": "t", "host": "x"}
    ]
    assert trace_simpleqa(log, GOLD).diagnosis == "unreadable"


def test_the_gold_is_matched_on_normalised_whole_words() -> None:
    log = log_of(
        (
            "search_results",
            {
                "results": [
                    {"url": PAGE, "title": "t", "snippet": "It is TEAL7, today."}
                ]
            },
        ),
        ("choice", {"picks": [{"url": PAGE}], "reserves": []}),
        ("fetched", {"url": PAGE, "ok": True, "text": "the colour: (teal7)!"}),
        ("notes_input", {"text": "the colour: (teal7)!"}),
        ("finding_noted", {"claim": "teal7x", "quote": "no match in this word"}),
    )
    trace = trace_simpleqa(log, GOLD)
    assert trace.stages["search_snippets"] and trace.stages["fetched_text"]
    assert trace.stages["notes_input"] and not trace.stages["findings"]  # whole words
    assert trace.diagnosis == "notes"


def test_a_quote_alone_counts_as_a_finding() -> None:
    log = run_log(finding=False)
    log.events[4].data["quote"] = f"the lid is {GOLD}"
    assert trace_simpleqa(log, GOLD).stages["findings"]


def test_a_title_counts_as_a_search_hit() -> None:
    log = log_of(
        (
            "search_results",
            {"results": [{"url": PAGE, "title": f"Lid {GOLD} guide", "snippet": ""}]},
        )
    )
    assert trace_simpleqa(log, GOLD).stages["search_snippets"]


@pytest.mark.parametrize("log", [None, RunLog(run_id="r")])
def test_no_log_is_its_own_diagnosis(log: RunLog | None) -> None:
    trace = trace_simpleqa(log, GOLD)
    assert trace.diagnosis == "no_log" and trace.stages == {}


def test_a_gold_with_no_letters_or_digits_is_refused() -> None:
    with pytest.raises(ValueError, match="no letters or digits"):
        trace_simpleqa(run_log(), "...")


# -- LitSearch -------------------------------------------------------------------------


def gold_work(number: int) -> Work:
    return Work(title=gold_title(number), local_id=f"local:{1000 + number}")


def lit_log(
    *,
    searched: bool = True,
    chosen: bool = True,
    read: bool = True,
    cited: bool = True,
    ranked: bool = True,
    rank: int = 3,
) -> RunLog:
    row = {"title": gold_title(0), "url": "https://kilns.example.org/p0"}
    unrelated = {
        "title": "A paper on something else entirely",
        "url": "https://z.org/q",
    }
    return log_of(
        ("search_results", {"results": [row if searched else unrelated]}),
        ("choice", {"picks": [row if chosen else unrelated], "reserves": []}),
        ("fetched", {**(row if read else unrelated), "ok": True, "text": "t"}),
        (
            "final_ranking",
            {
                "works": [
                    {
                        "rank": rank,
                        "title": gold_title(0),
                        "basis": "cited" if cited else "read",
                    }
                ]
                if ranked
                else []
            },
        ),
    )


@pytest.mark.parametrize(
    "switches, lost_at",
    [
        ({}, "found"),
        (
            {"searched": False, "chosen": False, "read": False, "ranked": False},
            "search_results",
        ),
        ({"chosen": False, "read": False, "rank": 99}, "choice"),
        ({"read": False, "rank": 99}, "read"),
        ({"cited": False}, "cited"),
        ({"rank": 21}, "ranked_top_20"),
        ({"rank": 20}, "found"),
    ],
)
def test_a_gold_work_is_lost_at_the_first_stage_it_is_missing_from(
    switches: dict[str, Any], lost_at: str
) -> None:
    [trace] = trace_litsearch(lit_log(**switches), [gold_work(0)])
    assert trace.lost_at == lost_at
    assert trace.title == gold_title(0)


def test_a_work_that_was_chosen_counts_as_found_even_if_no_search_listed_it() -> None:
    # a citation expansion adds candidates that no search returned
    [trace] = trace_litsearch(lit_log(searched=False), [gold_work(0)])
    assert trace.stages["search_results"] and trace.lost_at == "found"


def test_a_gold_work_is_matched_by_title_each_work_on_its_own() -> None:
    log = lit_log()
    traces = trace_litsearch(log, [gold_work(0), gold_work(1)])
    assert [t.lost_at for t in traces] == ["found", "search_results"]


def test_without_a_log_every_gold_work_is_lost_at_no_log() -> None:
    traces = trace_litsearch(None, [gold_work(0), gold_work(1)])
    assert [t.lost_at for t in traces] == ["no_log", "no_log"]


# -- the registry and the logs ------------------------------------------------------------


def wrong_log_for(question: str) -> RunLog:
    """A log whose gold got lost in the notes: the page had it, no finding does."""
    gold = GOLD_BY_QUESTION[question]
    return run_log(gold, finding=False)


def quiet_log_for(question: str) -> RunLog:
    """A log of a search that never found the gold."""
    return run_log(GOLD_BY_QUESTION[question], snippet=False, page_text=False)


def small_experiment() -> Experiment:
    return Experiment.model_validate(
        {
            "name": "tiny",
            "registered": "2026-10-08",
            "resamples": 200,
            "grader": "ollama:invented-grader-cloud",
            "sets": [{"name": "simpleqa", "n": 4, "seed": 0}],
            "arms": [
                {"name": "base", "system": "v2", "model": "ollama:m"},
                {"name": "new-engine", "system": "v2", "model": "ollama:m"},
            ],
            "baselines": {"simpleqa": "base"},
        }
    )


async def finished_run(tmp_path: Path) -> tuple[Path, DataDirs, Experiment]:
    """A run where new-engine answers two questions right and two wrong, each
    wrong one with a log that lost the gold at another stage."""
    sqa = write_simpleqa(tmp_path / "sqa")
    dirs = DataDirs(simpleqa=sqa)
    exp = small_experiment()
    calls = {"n": 0}

    def engine(question: str, _call: int) -> ArmOutput:
        calls["n"] += 1
        n = calls["n"]
        if n % 2 == 1:
            right = correct_behaviour(question, n)
            return right.model_copy(update={"log": run_log(GOLD_BY_QUESTION[question])})
        wrong = output(question, "I could not find it.")
        log = wrong_log_for(question) if n == 2 else quiet_log_for(question)
        return wrong.model_copy(update={"log": log})

    arms = {
        "base": ScriptedArm("base", correct_behaviour),
        "new-engine": ScriptedArm("new-engine", engine),
    }
    out = tmp_path / "out"
    await run_experiment(
        exp,
        out_dir=out,
        dirs=dirs,
        services=Services(
            answer_arm=lambda spec, _m: arms[spec.name],
            grader=StubGrader(),  # type: ignore[arg-type]
        ),
    )
    return out, dirs, exp


async def test_collect_losses_reads_the_registry_the_data_and_the_logs(
    tmp_path: Path,
) -> None:
    out, dirs, exp = await finished_run(tmp_path)
    found = collect_losses(exp, out_dir=out, dirs=dirs, arms=["new-engine"])
    [arm] = found.arms
    assert arm.arm == "new-engine" and arm.simpleqa_items == 4
    assert sorted(row.diagnosis for row in arm.simpleqa) == ["notes", "search_miss"]
    for row in arm.simpleqa:
        assert row.gold == GOLD_BY_QUESTION[row.question]
        assert row.item_id.startswith("SQ") and row.repeat == 1
    notes = next(r for r in arm.simpleqa if r.diagnosis == "notes")
    assert notes.stages["fetched_text"] and not notes.stages["findings"]


async def test_a_cell_without_a_log_is_no_log_and_every_arm_is_read_by_default(
    tmp_path: Path,
) -> None:
    out, dirs, exp = await finished_run(tmp_path)
    log_files = sorted((out / "logs" / "new-engine").glob("*-r1.log.jsonl"))
    assert len(log_files) == 4  # the engine's cells wrote logs, the base arm none
    assert not (out / "logs" / "base").exists()
    for path in log_files:
        path.unlink()
    again = collect_losses(exp, out_dir=out, dirs=dirs)
    assert [a.arm for a in again.arms] == ["base", "new-engine"]
    assert [r.diagnosis for r in again.arms[1].simpleqa] == ["no_log", "no_log"]


async def test_an_unknown_arm_or_a_missing_data_folder_is_refused(
    tmp_path: Path,
) -> None:
    out, dirs, exp = await finished_run(tmp_path)
    with pytest.raises(ValueError, match="not arms"):
        collect_losses(exp, out_dir=out, dirs=dirs, arms=["nobody"])
    with pytest.raises(ValueError, match="--simpleqa-dir"):
        collect_losses(exp, out_dir=out, dirs=DataDirs())
    with pytest.raises(FileNotFoundError, match="run the experiment first"):
        collect_losses(exp, out_dir=tmp_path / "empty", dirs=dirs)


async def test_the_markdown_has_the_counts_and_one_row_per_failed_item(
    tmp_path: Path,
) -> None:
    out, dirs, exp = await finished_run(tmp_path)
    text = collect_losses(exp, out_dir=out, dirs=dirs).to_markdown()
    assert "## SimpleQA, arm new-engine" in text
    assert "4 items, 2 not graded correct." in text
    assert "| diagnosis | count |" in text
    assert "| search_miss | 1 |" in text and "| notes | 1 |" in text
    assert "| item | question | gold | diagnosis | search_snippets |" in text
    rows = [line for line in text.splitlines() if line.startswith("| SQ")]
    assert len(rows) == 2
    assert all("Apollo kiln number" in row and "teal" in row for row in rows)
    notes_row = next(r for r in rows if "| notes |" in r)
    # present in the snippets, the fetched text and the notes input, then lost
    assert notes_row.endswith("| + | + | + | - | + | + |")


def test_the_command_prints_markdown_or_json_and_writes_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, dirs, exp = asyncio.run(finished_run(tmp_path))
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(exp.model_dump(mode="json")), encoding="utf-8")
    argv = [
        "experiment",
        "losses",
        str(path),
        "--out-dir",
        str(out),
        "--simpleqa-dir",
        str(dirs.simpleqa),
    ]
    assert cli.main(argv) == 0
    assert "| search_miss | 1 |" in capsys.readouterr().out
    assert cli.main([*argv, "--format", "json", "--arm", "new-engine"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["experiment"] == "tiny" and data["arms"][0]["arm"] == "new-engine"
    assert {r["diagnosis"] for r in data["arms"][0]["simpleqa"]} == {
        "search_miss",
        "notes",
    }
    target = tmp_path / "losses.md"
    assert cli.main([*argv, "--only-failed", "--out", str(target)]) == 0
    assert "# Losses in tiny" in target.read_text(encoding="utf-8")
    assert cli.main([*argv[:-2]]) == 1  # no --simpleqa-dir: a named error
    assert "--simpleqa-dir" in capsys.readouterr().err


async def test_litsearch_rows_name_the_stage_each_gold_work_was_lost_at(
    tmp_path: Path,
) -> None:
    root = write_litsearch(tmp_path / "ls", n=4)
    dirs = DataDirs(litsearch_local=root)
    exp = Experiment.model_validate(
        {
            "name": "lit",
            "registered": "2026-10-08",
            "resamples": 200,
            "sets": [{"name": "litsearch-web", "n": 4, "seed": 0}],
            "arms": [{"name": "new-engine", "system": "v2", "model": "ollama:m"}],
            "baselines": {"litsearch-web": "new-engine"},
        }
    )
    from andamentum.deep_research.v2.bench.experiment_run import load_items

    items = load_items(exp.set("litsearch-web"), dirs)
    gold_of = {i.question: i.litsearch.gold[0] for i in items if i.litsearch}
    order = [i.question for i in items]

    def engine(question: str, _call: int) -> ArmOutput:
        position = order.index(question)
        gold = gold_of[question]
        row = {"title": gold.title, "url": f"https://kilns.example.org/{position}"}
        # item 0 is found; item 1 is searched only; item 2 is never found
        found = position == 0
        searched = position <= 1
        log = log_of(
            ("search_results", {"results": [row] if searched else []}),
            ("choice", {"picks": [row] if found else [], "reserves": []}),
            ("fetched", {**row, "ok": True, "text": "t"} if found else {"ok": False}),
            (
                "final_ranking",
                {"works": [{"rank": 1, "title": gold.title, "basis": "cited"}]}
                if found
                else {"works": []},
            ),
        )
        kept = [gold] if found else []
        return output(question, "", kept).model_copy(update={"log": log})

    arms = {"new-engine": ScriptedArm("new-engine", engine)}
    out = tmp_path / "out"
    await run_experiment(
        exp,
        out_dir=out,
        dirs=dirs,
        services=Services(answer_arm=lambda spec, _m: arms[spec.name]),
    )
    found = collect_losses(exp, out_dir=out, dirs=dirs)
    [arm] = found.arms
    assert arm.litsearch_items == 4 and arm.simpleqa_items == 0
    by_query = {r.item_id: r.gold.lost_at for r in arm.litsearch}
    lost = sorted(by_query.values())
    assert lost == ["choice", "found", "search_results", "search_results"]
    markdown = found.to_markdown()
    assert "## LitSearch, arm new-engine" in markdown
    assert "4 queries, 4 gold works." in markdown
    assert "| first stage missing | gold works |" in markdown
    assert "| search_results | 2 |" in markdown
    shown = found.to_markdown(only_failed=True)
    assert "| found |" not in shown.split("| query |")[1]
    assert "| query | gold work | lost at |" in markdown
