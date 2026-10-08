"""Running and reporting an experiment with stand-in arms over invented items.

No model, no network: an arm is a small class that answers each question from a
script and the grader is a function, so what is tested is the experiment
machinery: item order, the registry, resuming, errors counted against the arm,
outages, the run log files of a cell, the report and the verdicts.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pytest

from andamentum.deep_research.bench import cli as bench_cli
from andamentum.deep_research.bench.arms import ArmOutput
from andamentum.deep_research.bench.experiment import ArmSpec, Experiment
from andamentum.deep_research.bench.experiment_report import (
    build_report,
    report_from_registry,
    stable_key,
)
from andamentum.deep_research.bench.experiment_run import (
    DataDirs,
    InfrastructureDown,
    Services,
    arm_record,
    build_answer_arm,
    cell_log_path,
    infrastructure_failure,
    load_items,
    run_experiment,
    set_mode,
)
from andamentum.deep_research.bench.litsearch_local import (
    LitSearchQuery,
)
from andamentum.deep_research.bench.metrics import Cost
from andamentum.deep_research.bench.registry import (
    ArmRecord,
    RegistryMismatch,
    RunRecord,
    StartRecord,
    read_records,
    registry_path,
)
from andamentum.deep_research.sources.base import SourceConfig
from andamentum.deep_research.sources.local_corpus import (
    CORPUS_FILE,
    CorpusRecord,
    build_index,
)
from andamentum.deep_research.types import (
    Finding,
    LogEvent,
    Passage,
    ResearchResult,
    RunLog,
    RunStats,
    SubQuestion,
)
from andamentum.references import Work

N_ROWS = 12


def question_of(number: int) -> str:
    return f"What is the lid colour of Apollo kiln number {number}?"


def gold_of(number: int) -> str:
    return f"teal{number}"


GOLD_BY_QUESTION = {question_of(n): gold_of(n) for n in range(1, N_ROWS + 1)}


def write_simpleqa(root: Path) -> Path:
    """An invented ``simple_qa_test_set.csv`` of twelve rows."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "simple_qa_test_set.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metadata", "problem", "answer"])
        for number in range(1, N_ROWS + 1):
            writer.writerow(["{}", question_of(number), gold_of(number)])
    return root


def gold_title(number: int) -> str:
    return f"An invented paper about kiln lids number {number}"


def write_litsearch(root: Path, *, n: int = 6) -> Path:
    """A local LitSearch folder: queries, a corpus and its index (invented)."""
    root.mkdir(parents=True, exist_ok=True)
    queries = [
        LitSearchQuery(
            query_id=f"LS{i:04d}",
            query=f"Which paper studies kiln lids number {i}?",
            query_set="inline_acl",
            specificity=1,
            quality=2,
            corpus_ids=[1000 + i],
        )
        for i in range(n)
    ]
    (root / "queries.jsonl").write_text(
        "".join(q.model_dump_json() + "\n" for q in queries), encoding="utf-8"
    )
    (root / CORPUS_FILE).write_text(
        "".join(
            CorpusRecord(
                corpus_id=str(1000 + i), title=gold_title(i), abstract="About kilns."
            ).model_dump_json()
            + "\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )
    build_index(root)
    return root


class StubGrader:
    """Grades by the gold string, as a stand-in for the grader model."""

    async def grade(self, *, question: str, gold: str, answer: str) -> str:
        if not answer.strip():
            return "not_attempted"
        return "correct" if gold in answer else "incorrect"


def work(number: int) -> Work:
    return Work(
        work_id=f"w_random{number}",
        doi=f"10.5555/invented.{number}",
        title=f"Invented study number {number}",
        year=2000 + number,
    )


def make_result(question: str, kept: list[Work]) -> ResearchResult:
    findings = [
        Finding(
            finding_id=f"F{i + 1}",
            subquestion_id="S1",
            claim="claim",
            quote=f"quote from {w.title}",
            passage_id=Passage.make(
                work_id=w.work_id, start=0, text=f"quote from {w.title}"
            ).passage_id,
            work_id=w.work_id,
        )
        for i, w in enumerate(kept)
    ]
    return ResearchResult(
        run_id="r",
        question=question,
        mode="web",
        subquestions=[SubQuestion(id="S1", text=question)],
        works=kept,
        findings=findings,
        report_markdown=" ".join(f"A claim [{f.finding_id}]." for f in findings),
        stats=RunStats(
            llm_calls=2, wall_s=1.5, step_seconds={"Plan": 0.5, "Read": 1.0}
        ),
    )


class ScriptedArm:
    """An answer arm that reads its behaviour from a callable."""

    def __init__(self, name: str, behave: Callable[[str, int], ArmOutput]) -> None:
        self.name = name
        self.behave = behave
        self.calls: list[str] = []

    async def run(self, question: str) -> ArmOutput:
        self.calls.append(question)
        return self.behave(question, len(self.calls))


def output(
    question: str, answer: str, kept: list[Work] | None = None, **kw: object
) -> ArmOutput:
    kept = kept if kept is not None else [work(1), work(2)]
    kept = [
        w if w.work_id else w.model_copy(update={"work_id": f"w_kept{n}"})
        for n, w in enumerate(kept)
    ]
    return ArmOutput(
        answer_text=answer,
        works=kept,
        cost=Cost(wall_s=1.5, llm_calls=2, decision_calls=0),
        result=make_result(question, kept),
        **kw,  # type: ignore[arg-type]
    )


def correct_behaviour(question: str, _call: int) -> ArmOutput:
    return output(question, f"The answer is {GOLD_BY_QUESTION[question]}.")


def small_experiment(*, repeats: int = 2, arms: list[dict] | None = None) -> Experiment:  # type: ignore[type-arg]
    return Experiment.model_validate(
        {
            "name": "tiny",
            "registered": "2026-10-08",
            "repeats": repeats,
            "resamples": 200,
            "grader": "ollama:invented-grader",
            "sets": [{"name": "simpleqa", "n": 4, "seed": 0}],
            "arms": arms
            or [
                {"name": "base", "system": "v2", "model": "ollama:m"},
                {"name": "other", "system": "v2", "model": "ollama:m"},
            ],
            "baselines": {"simpleqa": "base"},
            "rules": [
                {
                    "id": "R-correct",
                    "question": "other is not worse than base on correctness",
                    "set": "simpleqa",
                    "metric": "correct",
                    "arm": "other",
                    "reference": "base",
                    "test": "noninferior",
                    "margin": 0.1,
                },
            ],
        }
    )


def services(arms: dict[str, ScriptedArm], **kw: object) -> Services:
    return Services(
        answer_arm=lambda spec, _mode: arms[spec.name],
        grader=StubGrader(),  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


def both_arms(behave: Callable[[str, int], ArmOutput] = correct_behaviour):  # noqa: ANN201
    return {n: ScriptedArm(n, behave) for n in ("base", "other")}


@pytest.fixture
def dirs(tmp_path: Path) -> DataDirs:
    return DataDirs(
        simpleqa=write_simpleqa(tmp_path / "sqa"),
        litsearch_local=write_litsearch(tmp_path / "ls"),
    )


# -- items -----------------------------------------------------------------------


def test_simpleqa_items_are_a_deterministic_subset(dirs: DataDirs) -> None:
    exp = small_experiment()
    first = load_items(exp.set("simpleqa"), dirs)
    second = load_items(exp.set("simpleqa"), dirs)
    assert [i.item_id for i in first] == [i.item_id for i in second]
    assert len(first) == 4 and all(i.simpleqa is not None for i in first)


def test_skip_extends_a_smaller_subset(dirs: DataDirs) -> None:
    exp = small_experiment()
    spec = exp.set("simpleqa")
    both = load_items(spec.model_copy(update={"n": 8}), dirs)
    later = load_items(spec.model_copy(update={"n": 4, "skip": 4}), dirs)
    first = load_items(spec, dirs)
    assert [i.item_id for i in both] == [i.item_id for i in first + later]


def test_each_set_needs_its_directory() -> None:
    exp = Experiment.model_validate(
        {
            "name": "t",
            "registered": "2026-10-08",
            "grader": "g",
            "sets": [{"name": "simpleqa", "n": 2}, {"name": "litsearch-web", "n": 2}],
            "arms": [{"name": "a", "system": "v2", "model": "m"}],
            "baselines": {"simpleqa": "a"},
        }
    )
    with pytest.raises(ValueError, match="--simpleqa-dir"):
        load_items(exp.set("simpleqa"), DataDirs())
    with pytest.raises(ValueError, match="--litsearch-local-dir"):
        load_items(exp.set("litsearch-web"), DataDirs())


def test_litsearch_web_is_scholarly_and_simpleqa_is_web() -> None:
    assert set_mode("litsearch-web") == "scholarly"
    assert set_mode("simpleqa") == "web"


# -- running ---------------------------------------------------------------------


async def test_every_arm_runs_every_item_repeat_times_and_is_recorded(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = small_experiment()
    arms = both_arms()
    written = await run_experiment(
        exp, out_dir=tmp_path / "out", dirs=dirs, services=services(arms)
    )
    assert len(written) == 2 * 4 * 2
    assert len(arms["base"].calls) == 8
    records = list(read_records(registry_path(tmp_path / "out")))
    starts = [r for r in records if isinstance(r, StartRecord)]
    runs = [r for r in records if isinstance(r, RunRecord)]
    assert len(starts) == 1 and len(runs) == 16
    assert {a.name for a in starts[0].arms} == {"base", "other"}
    first = runs[0]
    assert first.experiment_hash == exp.config_hash()
    assert first.metrics["correct"] == 1.0 and first.metrics["attempted"] == 1.0
    assert first.metrics["wall_s"] == 1.5
    assert first.metrics["decision_calls"] == 0.0
    assert first.output is not None and first.output.result is not None
    assert first.output.result.passages == []  # slimmed for the registry


def test_the_registry_records_carry_only_the_final_fields() -> None:
    assert set(ArmRecord.model_fields) == {
        "name",
        "system",
        "model",
        "repeats",
        "solo",
        "outage",
    }
    gone = {
        "config_hash",
        "decision_model",
        "decision_digest",
        "judge",
        "judge_digest",
        "frozen",
        "planted_outcome",
        "screening",
        "preset",
        "screen",
        "embedding_model",
        "inputs",
    }
    assert not gone & (set(RunRecord.model_fields) | set(ArmRecord.model_fields))
    assert not {"judge", "judge_digest", "inputs"} & set(StartRecord.model_fields)


async def test_cells_run_item_by_item_with_the_arms_side_by_side(
    tmp_path: Path, dirs: DataDirs
) -> None:
    order: list[str] = []

    def make(name: str) -> Callable[[str, int], ArmOutput]:
        def behave(question: str, call: int) -> ArmOutput:
            order.append(name)
            return correct_behaviour(question, call)

        return behave

    arms = {n: ScriptedArm(n, make(n)) for n in ("base", "other")}
    await run_experiment(
        small_experiment(repeats=1),
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=services(arms),
    )
    assert order == ["base", "other"] * 4


async def test_a_second_invocation_resumes_and_runs_nothing_twice(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = small_experiment(repeats=1)
    arms = both_arms()
    await run_experiment(
        exp, out_dir=tmp_path / "out", dirs=dirs, services=services(arms)
    )
    again = await run_experiment(
        exp, out_dir=tmp_path / "out", dirs=dirs, services=services(arms)
    )
    assert again == []
    assert len(arms["base"].calls) == 4


async def test_a_run_stopped_midway_resumes_after_the_last_finished_cell(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = small_experiment(repeats=1)
    arms = both_arms()

    def stop_at_third(question: str, call: int) -> ArmOutput:
        if call == 3:
            raise InfrastructureDown("the search is down")
        return correct_behaviour(question, call)

    stopping = {"base": ScriptedArm("base", stop_at_third), "other": arms["other"]}
    with pytest.raises(InfrastructureDown):
        await run_experiment(
            exp, out_dir=tmp_path / "out", dirs=dirs, services=services(stopping)
        )
    done = [
        r
        for r in read_records(registry_path(tmp_path / "out"))
        if isinstance(r, RunRecord)
    ]
    assert len(done) == 4  # two items, both arms; the third was not recorded
    resumed = await run_experiment(
        exp, out_dir=tmp_path / "out", dirs=dirs, services=services(both_arms())
    )
    assert len(resumed) == 4


async def test_a_changed_experiment_file_is_refused_against_an_old_registry(
    tmp_path: Path, dirs: DataDirs
) -> None:
    await run_experiment(
        small_experiment(repeats=1),
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=services(both_arms()),
    )
    with pytest.raises(RegistryMismatch, match="pre-registration"):
        await run_experiment(
            small_experiment(repeats=2),
            out_dir=tmp_path / "out",
            dirs=dirs,
            services=services(both_arms()),
        )


async def test_a_failing_cell_is_recorded_and_counts_as_a_miss(
    tmp_path: Path, dirs: DataDirs
) -> None:
    def broken(_question: str, _call: int) -> ArmOutput:
        raise RuntimeError("model fell over")

    arms = {
        "base": ScriptedArm("base", correct_behaviour),
        "other": ScriptedArm("other", broken),
    }
    written = await run_experiment(
        small_experiment(repeats=1),
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=services(arms),
    )
    failed = [r for r in written if r.error]
    assert len(failed) == 4 and all(r.arm == "other" for r in failed)
    assert all("model fell over" in (r.error or "") for r in failed)
    assert all(r.metrics["correct"] == 0.0 for r in failed)  # a miss, not absent
    assert all(r.metrics["attempted"] == 0.0 for r in failed)


async def test_a_wrong_and_an_empty_answer_are_graded_apart(
    tmp_path: Path, dirs: DataDirs
) -> None:
    def varied(question: str, call: int) -> ArmOutput:
        return output(question, "" if call == 1 else "It is purple.")

    arms = {
        "base": ScriptedArm("base", correct_behaviour),
        "other": ScriptedArm("other", varied),
    }
    written = await run_experiment(
        small_experiment(repeats=1),
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=services(arms),
    )
    other = [r for r in written if r.arm == "other"]
    assert [(r.metrics["correct"], r.metrics["attempted"]) for r in other] == [
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]


async def test_a_simpleqa_cell_without_a_grader_is_a_recorded_error(
    tmp_path: Path, dirs: DataDirs
) -> None:
    arms = both_arms()
    written = await run_experiment(
        small_experiment(repeats=1),
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=Services(answer_arm=lambda spec, _mode: arms[spec.name]),
    )
    assert all("grader" in (r.error or "") for r in written)


def test_infrastructure_failure_reads_the_search_failure_notes() -> None:
    clean = output("q", "a")
    assert infrastructure_failure(clean) is None
    down = output("q", "a").model_copy(
        update={"degraded": ["3 of 8 fetches failed", "search failed for 'x'"]}
    )
    assert infrastructure_failure(down) == "search failed for 'x'"
    many = output("q", "a").model_copy(update={"degraded": ["2 searches failed"]})
    assert infrastructure_failure(many) == "2 searches failed"


async def test_a_failed_search_stops_the_run_without_recording_the_cell(
    tmp_path: Path, dirs: DataDirs
) -> None:
    def searching_failed(question: str, _call: int) -> ArmOutput:
        return output(question, "no answer").model_copy(
            update={"degraded": ["search failed for 'kiln': HTTP 503"]}
        )

    arms = {
        "base": ScriptedArm("base", correct_behaviour),
        "other": ScriptedArm("other", searching_failed),
    }
    with pytest.raises(InfrastructureDown, match="not recorded"):
        await run_experiment(
            small_experiment(repeats=1),
            out_dir=tmp_path / "out",
            dirs=dirs,
            services=services(arms),
        )
    runs = [
        r
        for r in read_records(registry_path(tmp_path / "out"))
        if isinstance(r, RunRecord)
    ]
    assert [r.arm for r in runs] == ["base"]  # the failed cell was not recorded


async def test_an_arm_that_records_outages_goes_on_and_has_no_metrics(
    tmp_path: Path, dirs: DataDirs
) -> None:
    def searching_failed(question: str, _call: int) -> ArmOutput:
        return output(question, "x").model_copy(
            update={"degraded": ["search failed for 'kiln': HTTP 503"]}
        )

    exp = small_experiment(
        repeats=1,
        arms=[
            {"name": "base", "system": "v2", "model": "ollama:m"},
            {"name": "other", "system": "v2", "model": "ollama:m", "outage": "record"},
        ],
    )
    arms = {
        "base": ScriptedArm("base", correct_behaviour),
        "other": ScriptedArm("other", searching_failed),
    }
    written = await run_experiment(
        exp, out_dir=tmp_path / "out", dirs=dirs, services=services(arms)
    )
    other = [r for r in written if r.arm == "other"]
    assert len(other) == 4 and all(r.outage and r.metrics == {} for r in other)
    assert all((r.error or "").startswith("outage:") for r in other)


async def test_only_arms_and_only_sets_restrict_a_run(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = small_experiment(repeats=1)
    arms = both_arms()
    written = await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=services(arms),
        only_arms=["other"],
    )
    assert {r.arm for r in written} == {"other"} and arms["base"].calls == []
    with pytest.raises(ValueError, match="not arms"):
        await run_experiment(
            exp,
            out_dir=tmp_path / "out2",
            dirs=dirs,
            services=services(arms),
            only_arms=["nobody"],
        )
    with pytest.raises(ValueError, match="not sets"):
        await run_experiment(
            exp,
            out_dir=tmp_path / "out3",
            dirs=dirs,
            services=services(arms),
            only_sets=["litsearch-web"],
        )


async def test_concurrency_above_one_is_refused_for_a_local_model(
    tmp_path: Path, dirs: DataDirs
) -> None:
    with pytest.raises(ValueError, match="local Ollama server"):
        await run_experiment(
            small_experiment(repeats=1),
            out_dir=tmp_path / "out",
            dirs=dirs,
            services=services(both_arms()),
            concurrency=2,
        )


async def test_a_keyless_experiment_refuses_a_configuration_that_holds_a_key(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = Experiment.model_validate(
        {**small_experiment().model_dump(mode="json"), "keyless": True}
    )
    with pytest.raises(ValueError, match="keyless"):
        await run_experiment(
            exp,
            out_dir=tmp_path / "out",
            dirs=dirs,
            services=services(
                both_arms(), source_config=SourceConfig(openalex_api_key="not-a-key")
            ),
        )


# -- litsearch-web cells --------------------------------------------------------


def litsearch_experiment() -> Experiment:
    return Experiment.model_validate(
        {
            "name": "lit",
            "registered": "2026-10-08",
            "resamples": 200,
            "sets": [{"name": "litsearch-web", "n": 3, "seed": 0}],
            "arms": [
                {"name": "base", "system": "v2", "model": "ollama:m"},
                {"name": "other", "system": "v2", "model": "ollama:m"},
            ],
            "baselines": {"litsearch-web": "base"},
        }
    )


async def test_a_litsearch_cell_scores_recall_of_the_gold_works(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = litsearch_experiment()
    items = load_items(exp.set("litsearch-web"), dirs)
    gold_by_question = {i.question: i.litsearch.gold for i in items if i.litsearch}

    def finds_gold(question: str, _call: int) -> ArmOutput:
        return output(question, "", [*gold_by_question[question], work(9)])

    def finds_nothing(question: str, _call: int) -> ArmOutput:
        return output(question, "", [work(8)])

    arms = {
        "base": ScriptedArm("base", finds_nothing),
        "other": ScriptedArm("other", finds_gold),
    }
    written = await run_experiment(
        exp, out_dir=tmp_path / "out", dirs=dirs, services=services(arms)
    )
    other = [r for r in written if r.arm == "other"]
    base = [r for r in written if r.arm == "base"]
    assert all(r.metrics["recall_at_20"] == 1.0 for r in other)
    assert all(r.metrics["found_any"] == 1.0 for r in other)
    assert all(r.metrics["recall_at_20"] == 0.0 for r in base)
    assert all("attempted" not in r.metrics for r in written)


# -- the run log of a cell ---------------------------------------------------------


def a_log() -> RunLog:
    return RunLog(
        run_id="r1",
        events=[
            LogEvent(seq=0, t=0.0, step="Setup", event="config", message="started")
        ],
    )


async def test_each_cell_writes_its_run_log_beside_the_registry(
    tmp_path: Path, dirs: DataDirs
) -> None:
    def with_log(question: str, call: int) -> ArmOutput:
        return correct_behaviour(question, call).model_copy(update={"log": a_log()})

    arms = {
        "base": ScriptedArm("base", correct_behaviour),
        "other": ScriptedArm("other", with_log),
    }
    out = tmp_path / "out"
    written = await run_experiment(
        small_experiment(repeats=1), out_dir=out, dirs=dirs, services=services(arms)
    )
    record = next(r for r in written if r.arm == "other")
    md = cell_log_path(out, arm="other", item_id=record.item_id, repeat=1)
    assert md == out / "logs" / "other" / f"{record.item_id}-r1.log.md"
    assert md.is_file() and md.with_name(f"{record.item_id}-r1.log.jsonl").is_file()
    jsonl = md.with_name(f"{record.item_id}-r1.log.jsonl").read_text(encoding="utf-8")
    assert RunLog.from_jsonl(jsonl).run_id == "r1"
    assert not (out / "logs" / "base").exists()  # the legacy arm has no log
    # the registry itself holds no log
    assert "config" not in registry_path(out).read_text(encoding="utf-8")


async def test_a_failed_cell_writes_the_partial_log_its_error_carries(
    tmp_path: Path, dirs: DataDirs
) -> None:
    class Broken(RuntimeError):
        run_log = a_log()

    def broken(_question: str, _call: int) -> ArmOutput:
        raise Broken("the writer failed")

    arms = {
        "base": ScriptedArm("base", correct_behaviour),
        "other": ScriptedArm("other", broken),
    }
    out = tmp_path / "out"
    written = await run_experiment(
        small_experiment(repeats=1), out_dir=out, dirs=dirs, services=services(arms)
    )
    record = next(r for r in written if r.arm == "other")
    assert cell_log_path(out, arm="other", item_id=record.item_id, repeat=1).is_file()


# -- building the arms ---------------------------------------------------------------


def test_the_v2_arm_takes_the_mode_the_sources_and_the_arms_name() -> None:
    spec = ArmSpec(name="new-engine", system="v2", model="ollama:m")
    sources = SourceConfig(browser_cdp_url="http://127.0.0.1:9222")
    arm = build_answer_arm(spec, "scholarly", Services(source_config=sources))
    assert arm.name == "new-engine"
    assert getattr(arm, "mode") == "scholarly" and getattr(arm, "sources") is sources


def test_the_arm_record_names_the_system_and_the_model_only() -> None:
    exp = small_experiment()
    record = arm_record(exp, exp.arm("other"))
    assert record.model_dump() == {
        "name": "other",
        "system": "v2",
        "model": "ollama:m",
        "repeats": 2,
        "solo": None,
        "outage": "stop",
    }


# -- the report -----------------------------------------------------------------------


async def test_the_report_has_the_tables_cost_and_verdicts(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = small_experiment(repeats=1)
    arms = {
        "base": ScriptedArm("base", correct_behaviour),
        "other": ScriptedArm("other", correct_behaviour),
    }
    out = tmp_path / "out"
    await run_experiment(exp, out_dir=out, dirs=dirs, services=services(arms))
    text = report_from_registry(exp, out)
    for heading in (
        "# Experiment tiny",
        "## Arms",
        "## Set simpleqa",
        "## Cost",
        "## Where the time and the requests went",
        "## Verdicts (pre-registered rules)",
    ):
        assert heading in text
    assert "R-correct" in text and "accepted" in text
    assert "Plan" in text  # the per-step table
    for gone in ("Repeat stability", "Planted", "Inputs profiles", "Loss table"):
        assert gone not in text


def test_a_report_of_an_empty_registry_is_still_a_report() -> None:
    text = build_report(small_experiment(), [])
    assert "0 run records" in text and "## Verdicts" in text


def test_a_work_is_named_by_its_first_identifier_else_its_title() -> None:
    assert stable_key(work(1)).startswith("doi:")
    assert stable_key(Work(title="A Title With Words")) == "title:a title with words"


# -- the command line ------------------------------------------------------------------


def test_the_cli_keeps_the_run_flags_and_drops_the_old_ones() -> None:
    parser = bench_cli.build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "run",
            "f.yaml",
            "--out-dir",
            "o",
            "--browser-cdp-url",
            "http://127.0.0.1:9222",
            "--cache",
            "c.db",
            "--concurrency",
            "3",
            "--simpleqa-dir",
            "s",
            "--litsearch-local-dir",
            "l",
            "--only-sets",
            "simpleqa",
            "--arm",
            "legacy",
            "--pdf-extraction",
            "docling",
        ]
    )
    assert args.concurrency == 3 and args.arm == ["legacy"]
    assert args.only_sets == ["simpleqa"] and args.pdf_extraction == "docling"
    for old in (
        ["--embedding-model", "x"],
        ["--decision-cache", "x"],
        ["--decision-model", "x"],
        ["--snapshots", "x"],
        ["--litsearch-dir", "x"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["experiment", "run", "f.yaml", "--out-dir", "o", *old])


def test_the_deleted_subcommands_are_gone() -> None:
    parser = bench_cli.build_parser()
    for gone in (
        ["freeze", "litsearch"],
        ["pool"],
        ["cache"],
        ["screening"],
        ["web"],
        ["litsearch"],
        ["report", "f.json"],
        ["verify-web-set"],
        ["download", "synergy"],
        ["download", "litsearch"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(gone)
    args = parser.parse_args(["download", "litsearch-local", "--dir", "d"])
    assert args.dataset == "litsearch-local"


def test_the_losses_subcommand_takes_the_documented_flags() -> None:
    args = bench_cli.build_parser().parse_args(
        [
            "experiment",
            "losses",
            "f.yaml",
            "--out-dir",
            "o",
            "--simpleqa-dir",
            "s",
            "--arm",
            "new-engine",
            "--only-failed",
            "--format",
            "json",
        ]
    )
    assert args.experiment_command == "losses"
    assert args.only_failed and args.format == "json" and args.arm == ["new-engine"]


def test_a_keyless_source_configuration_never_looks_a_key_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "not-a-real-key")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "not-a-real-key")
    keyed = bench_cli._source_config(None)  # type: ignore[attr-defined]
    keyless = bench_cli._source_config(None, keyless=True)  # type: ignore[attr-defined]
    assert keyed.openalex_api_key == "not-a-real-key"
    assert keyless.openalex_api_key is None
    assert keyless.semantic_scholar_api_key is None


def test_a_web_probe_is_needed_only_when_an_arm_that_stops_searches_the_web() -> None:
    exp = small_experiment()
    assert bench_cli._live_web(exp)  # type: ignore[attr-defined]  # a web set
    lit = litsearch_experiment()
    assert not bench_cli._live_web(lit)  # type: ignore[attr-defined]  # scholarly
    recording = Experiment.model_validate(
        {
            **exp.model_dump(mode="json"),
            "rules": [],
            "arms": [
                {"name": "other", "system": "v2", "model": "ollama:m", "outage": "record"}
            ],
            "baselines": {"simpleqa": "other"},
        }
    )
    assert not bench_cli._live_web(recording)  # type: ignore[attr-defined]
