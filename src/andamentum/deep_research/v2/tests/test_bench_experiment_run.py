"""Running and reporting an experiment with stand-in arms over invented items.

No model, no network: an arm is a small class that answers each web question
from a script, so what is tested is the experiment machinery: sequential
cells, the registry, resuming, errors counted against the arm, the stability
and planted-source measures, and the verdicts in the report.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.deep_research.v2.bench.arms import ArmOutput
from andamentum.deep_research.v2.bench.experiment import (
    ArmSpec,
    Experiment,
)
from andamentum.deep_research.v2.bench.experiment_report import (
    build_report,
    canonicalise,
    item_stability,
    report_from_registry,
    stable_key,
)
from andamentum.deep_research.v2.bench.experiment_run import (
    DataDirs,
    Services,
    load_items,
    planted_metrics,
    run_experiment,
)
from andamentum.deep_research.v2.bench.freeze import SnapshotMissing
from andamentum.deep_research.v2.bench.metrics import Cost
from andamentum.deep_research.v2.bench.planted import planted_source
from andamentum.deep_research.v2.bench.registry import (
    RegistryMismatch,
    RunRecord,
    StartRecord,
    read_records,
    registry_path,
)
from andamentum.deep_research.v2.bench.web_set import WEB_QUESTIONS
from andamentum.deep_research.v2.types import (
    Conflict,
    Finding,
    Passage,
    ResearchResult,
    SubQuestion,
)
from andamentum.references import Work

QUESTIONS = {q.question: q for q in WEB_QUESTIONS}


def work(number: int) -> Work:
    return Work(
        work_id=f"w_random{number}",
        doi=f"10.5555/invented.{number}",
        title=f"Invented study number {number}",
        year=2000 + number,
    )


def finding(number: int, owner: Work, quote: str) -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id="S1",
        claim="claim",
        quote=quote,
        passage_id=Passage.make(work_id=owner.work_id, start=0, text=quote).passage_id,
        work_id=owner.work_id,
    )


def make_result(
    question: str,
    kept: list[Work],
    *,
    plan: list[str] | None = None,
    cite: list[int] | None = None,
    conflicts: list[Conflict] | None = None,
) -> ResearchResult:
    findings = [finding(i + 1, w, f"quote from {w.title}") for i, w in enumerate(kept)]
    cited = findings if cite is None else [findings[i] for i in cite]
    report = " ".join(f"A claim [{f.finding_id}]." for f in cited)
    return ResearchResult(
        run_id="r",
        question=question,
        mode="web",
        subquestions=[
            SubQuestion(id=f"S{n}", text=text)
            for n, text in enumerate(plan or [question], start=1)
        ],
        works=kept,
        findings=findings,
        conflicts=conflicts or [],
        report_markdown=report,
    )


class ScriptedArm:
    """An answer arm that reads its behaviour from a callable."""

    returns_pages = False

    def __init__(self, name: str, behave) -> None:  # noqa: ANN001
        self.name = name
        self.behave = behave
        self.calls: list[str] = []

    async def run(
        self, question: str, *, frozen: object = None, store: object = None
    ) -> ArmOutput:
        self.calls.append(question)
        return self.behave(question, len(self.calls))


def output(question: str, answer: str, kept: list[Work], **kw) -> ArmOutput:  # noqa: ANN003
    return ArmOutput(
        answer_text=answer,
        works=kept,
        cost=Cost(wall_s=1.5, llm_calls=2, decision_calls=3),
        result=make_result(question, kept, **kw),
    )


def correct_answer(question: str) -> str:
    return QUESTIONS[question].expected


def small_experiment(*, repeats: int = 2, arms: list[dict] | None = None) -> Experiment:
    return Experiment.model_validate(
        {
            "name": "tiny",
            "registered": "2026-10-06",
            "repeats": repeats,
            "resamples": 200,
            "sets": [{"name": "web", "n": 4, "seed": 0}],
            "arms": arms
            or [
                {
                    "name": "base",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "m",
                    "frozen": False,
                },
                {
                    "name": "other",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "m",
                    "frozen": False,
                },
            ],
            "baselines": {"web": "base"},
            "rules": [
                {
                    "id": "R-correct",
                    "question": "other is not worse than base on correctness",
                    "set": "web",
                    "metric": "correct",
                    "arm": "other",
                    "reference": "base",
                    "test": "noninferior",
                    "margin": 0.1,
                },
                {
                    "id": "R-agree",
                    "question": "base answers agree across repeats",
                    "set": "web",
                    "metric": "stability.answer_agreement",
                    "arm": "base",
                    "test": "at_least",
                    "threshold": 0.9,
                    "gate": False,
                },
            ],
        }
    )


def services(arms: dict[str, ScriptedArm]) -> Services:
    return Services(answer_arm=lambda spec, _mode, _decision: arms[spec.name])


def stable_behaviour(question: str, _call: int) -> ArmOutput:
    return output(question, correct_answer(question), [work(1), work(2), work(3)])


# -- items -----------------------------------------------------------------------


def test_web_items_are_a_deterministic_subset() -> None:
    exp = small_experiment()
    first = load_items(exp.set("web"), DataDirs())
    second = load_items(exp.set("web"), DataDirs())
    assert [i.item_id for i in first] == [i.item_id for i in second]
    assert len(first) == 4
    planted = load_items(
        Experiment.model_validate(
            {
                **exp.model_dump(mode="json"),
                "sets": [{"name": "web_planted", "n": 4, "seed": 0}],
                "baselines": {},
                "rules": [],
                "arms": [
                    {"name": "base", "system": "v2", "preset": "baseline", "model": "m"}
                ],
            }
        ).set("web_planted"),
        DataDirs(),
    )
    assert [i.item_id for i in planted] == [f"{i.item_id}-planted" for i in first]
    assert all(i.planted is not None for i in planted)


def test_litsearch_needs_its_directory() -> None:
    exp = Experiment.model_validate(
        {
            "name": "t",
            "registered": "2026-10-06",
            "sets": [{"name": "litsearch", "n": 2}],
            "arms": [{"name": "a", "system": "legacy", "model": "m"}],
            "baselines": {"litsearch": "a"},
        }
    )
    with pytest.raises(ValueError, match="--litsearch-dir"):
        load_items(exp.set("litsearch"), DataDirs())


# -- running ---------------------------------------------------------------------


async def test_every_arm_runs_every_item_repeat_times_and_is_recorded(
    tmp_path: Path,
) -> None:
    exp = small_experiment()
    arms = {
        "base": ScriptedArm("base", stable_behaviour),
        "other": ScriptedArm("other", stable_behaviour),
    }
    written = await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
    )
    assert len(written) == 2 * 4 * 2
    assert len(arms["base"].calls) == 8
    records = list(read_records(registry_path(tmp_path)))
    starts = [r for r in records if isinstance(r, StartRecord)]
    runs = [r for r in records if isinstance(r, RunRecord)]
    assert len(starts) == 1 and len(runs) == 16
    assert {a.name for a in starts[0].arms} == {"base", "other"}
    assert all(a.config_hash for a in starts[0].arms)  # the preset's hash
    first = runs[0]
    assert first.experiment_hash == exp.config_hash()
    assert first.metrics["correct"] == 1.0
    assert first.metrics["wall_s"] == 1.5
    assert first.output is not None and first.output.result is not None
    assert first.output.result.passages == []  # slimmed for the registry


async def test_cells_run_sequentially_arm_by_arm(tmp_path: Path) -> None:
    order: list[str] = []

    def make(name: str):  # noqa: ANN202
        def behave(question: str, _call: int) -> ArmOutput:
            order.append(name)
            return stable_behaviour(question, 0)

        return behave

    arms = {n: ScriptedArm(n, make(n)) for n in ("base", "other")}
    await run_experiment(
        small_experiment(repeats=1),
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=services(arms),
    )
    assert order == ["base"] * 4 + ["other"] * 4


async def test_a_second_invocation_resumes_and_runs_nothing_twice(
    tmp_path: Path,
) -> None:
    exp = small_experiment(repeats=1)
    arms = {n: ScriptedArm(n, stable_behaviour) for n in ("base", "other")}
    await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
    )
    again = await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
    )
    assert again == []
    assert len(arms["base"].calls) == 4


async def test_a_changed_experiment_file_is_refused_against_an_old_registry(
    tmp_path: Path,
) -> None:
    arms = {n: ScriptedArm(n, stable_behaviour) for n in ("base", "other")}
    await run_experiment(
        small_experiment(repeats=1),
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=services(arms),
    )
    with pytest.raises(RegistryMismatch, match="pre-registration"):
        await run_experiment(
            small_experiment(repeats=2),
            out_dir=tmp_path,
            dirs=DataDirs(),
            services=services(arms),
        )


async def test_a_failing_cell_is_recorded_and_counts_as_a_miss(tmp_path: Path) -> None:
    def broken(_question: str, _call: int) -> ArmOutput:
        raise RuntimeError("model fell over")

    arms = {
        "base": ScriptedArm("base", stable_behaviour),
        "other": ScriptedArm("other", broken),
    }
    exp = small_experiment(repeats=1)
    written = await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
    )
    failed = [r for r in written if r.error]
    assert len(failed) == 4
    assert all("model fell over" in (r.error or "") for r in failed)
    assert all(r.metrics["correct"] == 0.0 for r in failed)  # a miss, not absent
    report = build_report(exp, list(read_records(registry_path(tmp_path))))
    assert "4 with errors" in report


async def test_a_frozen_arm_without_snapshots_is_refused_up_front(
    tmp_path: Path,
) -> None:
    exp = small_experiment(
        arms=[
            {"name": "base", "system": "v2", "preset": "baseline", "model": "m"},
            {"name": "other", "system": "v2", "preset": "baseline", "model": "m"},
        ]
    )
    arms = {n: ScriptedArm(n, stable_behaviour) for n in ("base", "other")}
    with pytest.raises(SnapshotMissing, match="bench freeze"):
        await run_experiment(
            exp,
            out_dir=tmp_path,
            dirs=DataDirs(snapshots=tmp_path / "none"),
            services=services(arms),
        )
    assert arms["base"].calls == []


async def test_a_v2_arm_cannot_be_declared_on_a_set_it_does_not_run() -> None:
    with pytest.raises(ValueError, match="cannot run on"):
        ArmSpec.model_validate(
            {"name": "l", "system": "legacy", "model": "m", "sets": ["web_planted"]}
        )


# -- the report ------------------------------------------------------------------


async def test_the_report_has_paired_differences_stability_cost_and_verdicts(
    tmp_path: Path,
) -> None:
    exp = small_experiment()
    wrong: dict[str, int] = {}

    def other(question: str, _call: int) -> ArmOutput:
        # right on every question but one, and its kept set changes between repeats
        first_question = next(iter(QUESTIONS))
        answer = "no idea" if question == first_question else correct_answer(question)
        wrong[question] = wrong.get(question, 0) + 1
        kept = [work(1), work(2)] if wrong[question] == 1 else [work(2), work(9)]
        return output(question, answer, kept)

    arms = {
        "base": ScriptedArm("base", stable_behaviour),
        "other": ScriptedArm("other", other),
    }
    await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
    )
    report = report_from_registry(exp, tmp_path)
    assert report.startswith("# Experiment tiny")
    for heading in (
        "## Arms",
        "## Set web",
        "### Repeat stability (web)",
        "## Cost",
        "## Verdicts (pre-registered rules)",
    ):
        assert heading in report
    assert "Difference from the baseline arm **base**" in report
    assert "R-correct" in report and "R-agree" in report
    # the stable arm repeats perfectly: its answer agreement is 1.0
    stability_rows = [
        line for line in report.splitlines() if line.startswith("| base | live | 2")
    ]
    assert stability_rows and "1.000" in stability_rows[0]
    # the other arm's kept sets differ between repeats: jaccard 1/3 for every item
    other_row = next(
        line for line in report.splitlines() if line.startswith("| other | live | 2")
    )
    assert "0.333" in other_row
    assert "Live arms (base, other)" in report
    assert "accepted" in report  # base agreement: at_least 0.9 on a stable arm


async def test_the_verdict_follows_the_registered_rule(tmp_path: Path) -> None:
    exp = small_experiment(repeats=1)

    def always_wrong(question: str, _call: int) -> ArmOutput:
        return output(question, "no idea", [work(1)])

    arms = {
        "base": ScriptedArm("base", stable_behaviour),
        "other": ScriptedArm("other", always_wrong),
    }
    await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
    )
    report = report_from_registry(exp, tmp_path)
    row = next(line for line in report.splitlines() if line.startswith("| R-correct"))
    assert "**not accepted**" in row and "gate" in row
    assert "Gating rules together: **not accepted**" in report


def test_the_report_of_an_unrun_experiment_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run the experiment first"):
        report_from_registry(small_experiment(), tmp_path)


# -- stability across stores -----------------------------------------------------


def test_work_ids_are_compared_by_identity_not_by_store_id() -> None:
    one = make_result("q", [work(1), work(2)])
    # the same papers in another run's store carry other random ids
    renamed = one.model_copy(
        update={
            "works": [
                w.model_copy(update={"work_id": f"w_other{n}"})
                for n, w in enumerate(one.works)
            ],
            "findings": [
                f.model_copy(update={"work_id": f"w_other{n}"})
                for n, f in enumerate(one.findings)
            ],
        }
    )
    assert {w.work_id for w in canonicalise(one).works} == {
        w.work_id for w in canonicalise(renamed).works
    }
    assert stable_key(work(1)).startswith("doi:")


def record(repeat: int, kept: list[Work], *, correct: float = 1.0) -> RunRecord:
    return RunRecord(
        experiment="e",
        experiment_hash="h",
        arm="a",
        set="web",
        item_id="W01",
        repeat=repeat,
        started_at=0.0,
        metrics={"correct": correct},
        output=output("q", "answer", kept),
    )


def test_item_stability_measures_agreement_between_repeats() -> None:
    same = item_stability(
        [record(1, [work(1), work(2)]), record(2, [work(1), work(2)])], live=False
    )
    assert same["kept_jaccard"] == 1.0
    assert same["rank_tau"] == 1.0
    assert same["findings_reproduced"] == 1.0
    assert same["answer_agreement"] == 1.0
    assert same["subquestion_overlap"] is None  # frozen: the plan cannot differ
    differ = item_stability(
        [
            record(1, [work(1), work(2)], correct=1.0),
            record(2, [work(2), work(1)], correct=0.0),
        ],
        live=True,
    )
    assert differ["kept_jaccard"] == 1.0
    assert differ["rank_tau"] == -1.0
    assert differ["answer_agreement"] == 0.0
    assert differ["subquestion_overlap"] == 1.0


def test_one_repeat_has_no_stability() -> None:
    assert all(
        v is None for v in item_stability([record(1, [work(1)])], live=False).values()
    )


# -- the planted source ----------------------------------------------------------


def test_planted_outcomes_are_read_from_the_result() -> None:
    question = WEB_QUESTIONS[0]
    source = planted_source(question)
    real = work(1)
    fake = Work(url=source.url, title=source.title, work_id="w_fake")
    good = make_result(question.question, [real])
    adopted = make_result(question.question, [real, fake])
    metrics, outcome = planted_metrics(
        ArmOutput(answer_text=question.expected, cost=Cost(wall_s=1), result=good),
        question,
        source,
    )
    assert outcome == "ignored_correct" and metrics["good_planted_outcome"] == 1.0
    metrics, outcome = planted_metrics(
        ArmOutput(answer_text="wrong", cost=Cost(wall_s=1), result=good),
        question,
        source,
    )
    assert outcome == "ignored_incorrect" and metrics["good_planted_outcome"] == 0.0
    metrics, outcome = planted_metrics(
        ArmOutput(answer_text="wrong", cost=Cost(wall_s=1), result=adopted),
        question,
        source,
    )
    assert outcome == "adopted_false_claim"
    assert metrics["adopted_false_claim"] == 1.0
    conflict = Conflict(subquestion_id="S1", finding_ids=["F1", "F2"])
    stated = make_result(question.question, [real, fake], conflicts=[conflict])
    _metrics, outcome = planted_metrics(
        ArmOutput(answer_text=question.expected, cost=Cost(wall_s=1), result=stated),
        question,
        source,
    )
    assert outcome == "stated_conflict"


def test_planted_measure_needs_a_v2_result() -> None:
    question = WEB_QUESTIONS[0]
    with pytest.raises(ValueError, match="v2 result"):
        planted_metrics(
            ArmOutput(answer_text="x", cost=Cost(wall_s=1)),
            question,
            planted_source(question),
        )


# -- the command line ------------------------------------------------------------


def write_experiment(path: Path, exp: Experiment) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(exp.model_dump(mode="json")), encoding="utf-8")
    return path


def test_experiment_report_command_prints_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    from andamentum.deep_research.v2.bench import cli

    exp = small_experiment(repeats=1)
    arms = {n: ScriptedArm(n, stable_behaviour) for n in ("base", "other")}
    asyncio.run(
        run_experiment(
            exp, out_dir=tmp_path / "out", dirs=DataDirs(), services=services(arms)
        )
    )
    file = write_experiment(tmp_path / "tiny.yaml", exp)
    assert (
        cli.main(
            ["experiment", "report", str(file), "--out-dir", str(tmp_path / "out")]
        )
        == 0
    )
    assert "# Experiment tiny" in capsys.readouterr().out
    target = tmp_path / "report.md"
    cli.main(
        [
            "experiment",
            "report",
            str(file),
            "--out-dir",
            str(tmp_path / "out"),
            "--out",
            str(target),
        ]
    )
    assert "Verdicts" in target.read_text(encoding="utf-8")


def test_experiment_report_without_a_run_exits_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from andamentum.deep_research.v2.bench import cli

    file = write_experiment(tmp_path / "tiny.yaml", small_experiment())
    code = cli.main(
        ["experiment", "report", str(file), "--out-dir", str(tmp_path / "x")]
    )
    assert code == 1
    assert "run the experiment first" in capsys.readouterr().err


def test_the_parser_knows_freeze_and_experiment_run() -> None:
    from andamentum.deep_research.v2.bench import cli

    args = cli.build_parser().parse_args(
        ["freeze", "web", "--snapshots", "s", "--model", "m", "--n", "5"]
    )
    assert (args.command, args.set, args.n) == ("freeze", "web", 5)
    args = cli.build_parser().parse_args(
        ["experiment", "run", "e.yaml", "--out-dir", "o", "--snapshots", "s"]
    )
    assert (args.experiment_command, args.file) == ("run", Path("e.yaml"))


# -- an arm is built for each set's mode -----------------------------------------------


async def test_sets_of_one_mode_share_one_arm_and_the_modes_are_right(
    tmp_path: Path,
) -> None:
    """The two web sets share one arm built in web mode."""
    built: list[tuple[str, str]] = []
    scripted = ScriptedArm("base", stable_behaviour)

    def build(spec: ArmSpec, mode: str, _decision: object) -> ScriptedArm:
        built.append((spec.name, mode))
        return scripted

    exp = Experiment.model_validate(
        {
            "name": "modes",
            "registered": "2026-10-06",
            "repeats": 1,
            "sets": [{"name": "web", "n": 1}, {"name": "web_planted", "n": 1}],
            "arms": [
                {
                    "name": "base",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "m",
                    "frozen": False,
                }
            ],
            "baselines": {"web": "base"},
        }
    )
    await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=Services(answer_arm=build),
    )
    assert built == [("base", "web")]  # both web sets share the web arm

    from andamentum.deep_research.v2.bench.experiment_run import set_mode

    assert [set_mode(n) for n in ("litsearch", "web", "web_planted")] == [
        "scholarly",
        "web",
        "web",
    ]


async def test_the_same_arm_gets_one_build_per_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scholarly set then a web set: the arm is built for each, in that mode."""
    from andamentum.deep_research.v2.bench import experiment_run

    built: list[str] = []

    def build(spec: ArmSpec, mode: str, _decision: object) -> ScriptedArm:
        built.append(mode)
        return ScriptedArm(spec.name, stable_behaviour)

    exp = Experiment.model_validate(
        {
            "name": "modes2",
            "registered": "2026-10-06",
            "repeats": 1,
            "sets": [{"name": "litsearch", "n": 1}, {"name": "web", "n": 1}],
            "arms": [{"name": "a", "system": "legacy", "model": "m"}],
            "baselines": {"web": "a", "litsearch": "a"},
        }
    )
    web_items = load_items(exp.set("web"), DataDirs())
    by_set = {
        "litsearch": [experiment_run.Item("litsearch", "L1", "a literature question?")],
        "web": web_items,
    }
    monkeypatch.setattr(
        experiment_run, "load_items", lambda spec, dirs: by_set[spec.name]
    )
    written = await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=Services(answer_arm=build)
    )
    assert built == ["scholarly", "web"]
    # the litsearch stand-in has no gold papers: recorded as an error, not dropped
    assert [r.set for r in written] == ["litsearch", "web"]
    assert written[0].error and written[1].error is None


# -- outages in live arms --------------------------------------------------------------


def degraded_behaviour(question: str, _call: int) -> ArmOutput:
    out = output(question, correct_answer(question), [work(1)])
    out.degraded = ["searxng search failed for 'x': HTTP 503"]
    return out


async def test_a_live_cell_with_failed_searches_is_not_recorded_and_stops_the_run(
    tmp_path: Path,
) -> None:
    from andamentum.deep_research.v2.bench.experiment_run import InfrastructureDown

    exp = small_experiment(repeats=1)
    arms = {
        "base": ScriptedArm("base", degraded_behaviour),
        "other": ScriptedArm("other", stable_behaviour),
    }
    with pytest.raises(InfrastructureDown, match="not recorded"):
        await run_experiment(
            exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
        )
    runs = [
        r for r in read_records(registry_path(tmp_path)) if isinstance(r, RunRecord)
    ]
    assert runs == []  # nothing recorded: a resume runs the cell again
    arms["base"].behave = stable_behaviour
    written = await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
    )
    assert len(written) == 2 * 4


async def test_searxng_is_probed_before_each_live_web_cell(tmp_path: Path) -> None:
    from andamentum.deep_research.v2.bench.health import SearxDown, SearxHealth

    probes: list[str | None] = []

    async def probe(url: str | None) -> SearxHealth:
        probes.append(url)
        if len(probes) == 3:
            raise SearxDown("engines are suspended")
        return SearxHealth(url=url or "", ok=True, n_results=3)

    exp = small_experiment(repeats=1)
    arms = {
        "base": ScriptedArm("base", stable_behaviour),
        "other": ScriptedArm("other", stable_behaviour),
    }
    svc = services(arms)
    svc.check_searxng = probe
    with pytest.raises(SearxDown):
        await run_experiment(exp, out_dir=tmp_path, dirs=DataDirs(), services=svc)
    runs = [
        r for r in read_records(registry_path(tmp_path)) if isinstance(r, RunRecord)
    ]
    assert len(runs) == 2 and len(probes) == 3


# -- the judge is registered and recorded ----------------------------------------------


def judged_experiment() -> Experiment:
    data = small_experiment(repeats=1).model_dump(mode="json")
    data["rules"] = [
        {
            "id": "R-claims",
            "question": "claims are supported",
            "set": "web",
            "metric": "claim_precision",
            "arm": "base",
            "test": "at_least",
            "threshold": 0.5,
        }
    ]
    data["judge"] = {"decision_model": "judge-model"}
    return Experiment.model_validate(data)


async def test_an_experiment_that_needs_a_judge_refuses_to_run_without_one(
    tmp_path: Path,
) -> None:
    arms = {
        "base": ScriptedArm("base", stable_behaviour),
        "other": ScriptedArm("other", stable_behaviour),
    }
    with pytest.raises(ValueError, match="judge"):
        await run_experiment(
            judged_experiment(),
            out_dir=tmp_path,
            dirs=DataDirs(),
            services=services(arms),
        )


async def test_the_judge_and_its_digest_are_recorded_on_the_start_and_every_run(
    tmp_path: Path,
) -> None:
    class Judge:
        async def judge(self, pairs: object) -> list[bool]:
            return []

    arms = {
        "base": ScriptedArm("base", stable_behaviour),
        "other": ScriptedArm("other", stable_behaviour),
    }
    svc = services(arms)
    svc.judge = Judge()  # type: ignore[assignment]
    svc.judge_label = "judge-model"
    svc.judge_digest = "sha256:abc"
    await run_experiment(
        judged_experiment(), out_dir=tmp_path, dirs=DataDirs(), services=svc
    )
    records = list(read_records(registry_path(tmp_path)))
    start = next(r for r in records if isinstance(r, StartRecord))
    assert (start.judge, start.judge_digest) == ("judge-model", "sha256:abc")
    runs = [r for r in records if isinstance(r, RunRecord)]
    assert runs and all(r.judge == "judge-model" for r in runs)
    assert all(r.judge_digest == "sha256:abc" for r in runs)


# -- a gating rule must rest on every item ---------------------------------------------


def _cell(arm: str, item: str, value: float | None) -> RunRecord:
    return RunRecord(
        experiment="tiny",
        experiment_hash="h",
        arm=arm,
        set="web",
        item_id=item,
        repeat=1,
        started_at=0.0,
        metrics={"correct": value},
    )


def test_a_gating_rule_with_missing_items_is_inconclusive_and_says_how_many() -> None:
    from andamentum.deep_research.v2.bench.experiment_report import (
        Cells,
        judge_rules,
    )

    exp = small_experiment(repeats=1)
    runs = [_cell("base", f"W{i}", 0.5) for i in range(1, 7)]
    runs += [_cell("other", f"W{i}", 1.0) for i in range(1, 6)]
    runs.append(_cell("other", "W6", None))  # one item has no value
    [verdict, _] = judge_rules(exp, Cells(runs))
    assert verdict.rule_id == "R-correct"
    assert verdict.outcome == "inconclusive"
    assert "only 5 of 6 items" in verdict.statement

    runs[-1] = _cell("other", "W6", 1.0)
    [complete, _] = judge_rules(exp, Cells(runs))
    assert complete.outcome == "accepted"


# -- running some sets now and the rest later ------------------------------------------


async def test_only_sets_runs_part_of_the_file_and_a_later_run_finishes_it(
    tmp_path: Path,
) -> None:
    exp = Experiment.model_validate(
        {
            "name": "parts",
            "registered": "2026-10-06",
            "repeats": 1,
            "resamples": 50,
            "sets": [{"name": "web", "n": 2}, {"name": "web_planted", "n": 2}],
            "arms": [
                {
                    "name": "base",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "m",
                    "frozen": False,
                }
            ],
            "baselines": {"web": "base", "web_planted": "base"},
        }
    )
    arms = {"base": ScriptedArm("base", stable_behaviour)}
    first = await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=services(arms),
        only_sets=["web"],
    )
    assert {r.set for r in first} == {"web"} and len(first) == 2
    # a report over the part that ran does not break on the set that did not
    text = report_from_registry(exp, tmp_path)
    assert "web_planted" in text
    second = await run_experiment(
        exp, out_dir=tmp_path, dirs=DataDirs(), services=services(arms)
    )
    assert {r.set for r in second} == {"web_planted"} and len(second) == 2
    with pytest.raises(ValueError, match="not sets of this experiment"):
        await run_experiment(
            exp,
            out_dir=tmp_path,
            dirs=DataDirs(),
            services=services(arms),
            only_sets=["litsearch"],
        )
