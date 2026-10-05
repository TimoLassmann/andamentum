"""Running tasks end to end with stand-in arms, a stub decision model, a scripted
LLM and invented data: the runner, the screening arms, the result files and the CLI."""

from __future__ import annotations

import asyncio
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from andamentum.core.decisions import Answer
from andamentum.core.decisions.testing import StubDecisions, choice_answer, noul_answer
from andamentum.deep_research.v2.bench import cli
from andamentum.deep_research.v2.bench.arms import ArmOutput
from andamentum.deep_research.v2.bench.judging import (
    ClaimEvidence,
    ClaimJudge,
    best_window,
    check_links,
)
from andamentum.deep_research.v2.bench.litsearch import LitSearchItem, LitSearchQuery
from andamentum.deep_research.v2.bench.metrics import Cost
from andamentum.deep_research.v2.bench.results import (
    RunConfig,
    WebRun,
    dump_run,
    load_run,
    render_markdown,
    summarise_web,
)
from andamentum.deep_research.v2.bench.runner import (
    run_litsearch,
    run_screening,
    run_web,
)
from andamentum.deep_research.v2.bench.screening import (
    ScreeningArm,
    compile_rubric,
    generic_criterion,
    parse_arm,
    review_subquestion,
)
from andamentum.deep_research.v2.bench.synergy import load_review
from andamentum.deep_research.v2.bench.web_set import WEB_QUESTIONS
from andamentum.deep_research.v2.criteria import (
    CandidateJudgement,
    CriteriaDraft,
    CriterionDraft,
    LabelAnswer,
)
from andamentum.references import UrlHealth, Work

from .bench_fixtures import N_EXCLUDED, N_INCLUDED, REVIEW_KEY, make_synergy_dir

CONFIG = RunConfig(arm="stub", subset_size=2, seed=0)


class StubArm:
    """An answer arm that returns what a table says, and raises for one question."""

    returns_pages = False

    def __init__(self, outputs: dict[str, ArmOutput], *, name: str = "stub") -> None:
        self.name = name
        self.outputs = outputs
        self.asked: list[str] = []

    async def run(self, question: str) -> ArmOutput:
        self.asked.append(question)
        if question not in self.outputs:
            raise RuntimeError("the engine fell over")
        return self.outputs[question]


def output(text: str, **extra: Any) -> ArmOutput:
    return ArmOutput(answer_text=text, cost=Cost(wall_s=2.0, llm_calls=3), **extra)


# -- web ------------------------------------------------------------------------------


def claim_judge(*, yes_if: str) -> ClaimJudge:
    def respond(
        state: object, questions: dict[str, dict[str, object]]
    ) -> dict[str, Answer]:
        assert isinstance(state, dict)
        return {
            k: noul_answer(0.9 if yes_if in str(state["evidence"]) else 0.1)
            for k in questions
        }

    return ClaimJudge(decision_model=StubDecisions(respond).model())


async def test_web_run_scores_correctness_support_and_links_and_keeps_a_failed_item() -> (
    None
):
    first, second, third = WEB_QUESTIONS[:3]
    arm = StubArm(
        {
            first.question: output(
                f"{first.expected}. More text.",
                claims=[
                    ClaimEvidence(claim="c1", evidence="supported by quote"),
                    ClaimEvidence(claim="c2", evidence="unrelated words"),
                ],
                n_unsourced=1,
                urls=["https://ok.example.org", "https://gone.example.org"],
            ),
            second.question: output("A confident but wrong answer."),
        }
    )

    async def check(url: str) -> UrlHealth:
        return UrlHealth(url=url, status="ok" if "ok." in url else "not_found")

    checkpoints: list[int] = []
    run = await run_web(
        [first, second, third],
        arm,
        config=CONFIG,
        judge=claim_judge(yes_if="supported"),
        url_checker=check,
        checkpoint=lambda r: checkpoints.append(len(r.items)),
    )
    assert checkpoints == [1, 2, 3]
    assert [i.correct for i in run.items] == [True, False, False]
    assert run.items[2].error == "RuntimeError: the engine fell over"
    assert run.items[0].claim_support is not None
    assert run.items[0].claim_support.precision == 0.5
    assert run.items[0].links is not None and run.items[0].links.integrity == 0.5
    summary = run.summary
    assert summary.n == 3 and summary.n_errors == 1
    assert summary.accuracy == pytest.approx(1 / 3)  # the crash counts as wrong
    assert summary.claim_precision == 0.5 and summary.n_unsourced == 1
    assert summary.link_integrity == 0.5
    assert summary.cost.wall_s == 4.0


async def test_without_a_judge_or_link_checker_those_metrics_are_not_measured() -> None:
    question = WEB_QUESTIONS[0]
    run = await run_web(
        [question],
        StubArm(
            {
                question.question: output(
                    question.expected, claims=[ClaimEvidence(claim="c", evidence="e")]
                )
            }
        ),
        config=CONFIG,
    )
    assert run.summary.claim_precision is None and run.summary.link_integrity is None
    assert "n/a" in render_markdown([run])


def test_a_judge_needs_a_model() -> None:
    with pytest.raises(ValueError, match="needs a model or a decision model"):
        ClaimJudge()


async def test_the_llm_judge_answers_yes_or_no_through_the_asker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from andamentum.deep_research.v2.ask import Verdict, Verdicts

    async def fake(model: Any, **kwargs: Any) -> tuple[BaseModel, RunUsage]:
        items = kwargs["user_message"].count("\n[")
        verdicts = [
            Verdict(item=i, answer="yes" if i == 1 else "no")
            for i in range(1, items + 1)
        ]
        return Verdicts(verdicts=verdicts), RunUsage(input_tokens=1, output_tokens=1)

    monkeypatch.setattr("andamentum.deep_research.v2.ask.run_agent_with_usage", fake)
    judge = ClaimJudge(model="ollama:stub")
    verdicts = await judge.judge(
        [ClaimEvidence(claim="a", evidence="x"), ClaimEvidence(claim="b", evidence="y")]
    )
    assert verdicts == [True, False]
    assert judge.stats.llm_calls == 1


def test_best_window_picks_the_stretch_that_shares_most_with_the_claim() -> None:
    text = (
        "alpha " * 500 + "brine loss fell with SaltLog in Apollo kilns " + "beta " * 500
    )
    window = best_window(
        "SaltLog reduced brine loss in Apollo kilns", [text], width=200
    )
    assert "brine loss fell" in window and len(window) == 200
    assert best_window("anything", []) == ""


async def test_check_links_probes_in_order_up_to_the_limit() -> None:
    probed: list[str] = []

    async def check(url: str) -> UrlHealth:
        probed.append(url)
        return UrlHealth(url=url, status="ok")

    result = await check_links([f"https://e.org/{i}" for i in range(5)], check, limit=3)
    assert probed == ["https://e.org/0", "https://e.org/1", "https://e.org/2"]
    assert result.n == 3 and result.integrity == 1.0


# -- litsearch ---------------------------------------------------------------------------


def lit_item(n: int) -> LitSearchItem:
    return LitSearchItem(
        query=LitSearchQuery(
            query_id=f"LS{n:04d}",
            query=f"Which paper studies invented topic {n}?",
            query_set="inline_acl",
            specificity=0,
            quality=2,
            corpus_ids=[9000 + n],
        ),
        gold=[
            Work(doi=f"10.5555/gold.{n}", title=f"An invented paper about topic {n}")
        ],
    )


async def test_litsearch_run_scores_ranked_works_and_counts_a_crash_as_zero_recall() -> (
    None
):
    hit = lit_item(1)
    miss = lit_item(2)
    crash = lit_item(3)
    arm = StubArm(
        {
            hit.query.query: output(
                "",
                works=[
                    Work(doi="10.5555/noise", title="Noise paper about nothing at all"),
                    Work(doi="10.5555/gold.1", title="An invented paper about topic 1"),
                ],
            ),
            miss.query.query: output(
                "",
                works=[
                    Work(doi="10.5555/noise", title="Noise paper about nothing at all")
                ],
            ),
        }
    )
    run = await run_litsearch([hit, miss, crash], arm, config=CONFIG, ks=(1, 2))
    assert run.summary.recall_at == {1: 0.0, 2: pytest.approx(1 / 3)}
    assert run.summary.found_any == pytest.approx(1 / 3)
    assert run.summary.n_errors == 1
    assert run.summary.kept_precision == pytest.approx((0.5 + 0.0) / 2)
    assert run.items[2].retrieval is None


# -- screening ---------------------------------------------------------------------------


COMPILED = CriteriaDraft(
    criteria=[
        CriterionDraft(
            text="Is the study about SaltLog?",
            kind="must",
            weight=3,
            applies_to="candidate",
        ),
        CriterionDraft(
            text="Does it measure brine loss?",
            kind="should",
            weight=2,
            applies_to="candidate",
        ),
        CriterionDraft(
            text="Is it set in an Apollo kiln?",
            kind="should",
            weight=1,
            applies_to="candidate",
        ),
    ]
)


class ScriptedLLM:
    """Stands in for ``run_agent_with_usage``: compiles COMPILED, and scores a
    record by whether its title names SaltLog."""

    def __init__(self) -> None:
        self.calls: list[type[BaseModel]] = []

    async def __call__(self, model: Any, **kwargs: Any) -> tuple[BaseModel, RunUsage]:
        output_type = kwargs["output_type"]
        self.calls.append(output_type)
        usage = RunUsage(input_tokens=5, output_tokens=3)
        if output_type is CriteriaDraft:
            return COMPILED, usage
        assert output_type is CandidateJudgement
        message: str = kwargs["user_message"]
        title = next(
            line.removeprefix("title: ")
            for line in message.splitlines()
            if line.startswith("title: ")
        )
        about = title.startswith("SaltLog")
        answers = [
            LabelAnswer(label=found.group(1), answer="yes" if about else "no")  # type: ignore[arg-type]
            for line in message.splitlines()
            if (found := re.match(r"(C\d+): ", line))
        ]
        return CandidateJudgement(answers=answers, source_type="primary_study"), usage


def screening_decisions() -> StubDecisions:
    def respond(
        state: object, questions: dict[str, dict[str, object]]
    ) -> dict[str, Answer]:
        assert isinstance(state, dict)
        about = str(state["title"]).startswith("SaltLog")
        answers: dict[str, Answer] = {}
        for key, question in questions.items():
            if question["type"] == "choice":
                options = question["criteria"]
                assert isinstance(options, dict)
                answers[key] = choice_answer(list(options), "primary_study", 0.8)
            else:
                answers[key] = noul_answer(0.9 if about else 0.05)
        return answers

    return StubDecisions(respond)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedLLM:
    fake = ScriptedLLM()
    monkeypatch.setattr("andamentum.deep_research.v2.ask.run_agent_with_usage", fake)
    return fake


def test_arm_names_parse_and_arms_check_what_they_need() -> None:
    assert parse_arm("criteria+decision") == ("criteria", "decision")
    with pytest.raises(ValueError, match="not a screening arm"):
        parse_arm("criteria")
    with pytest.raises(ValueError, match="needs a decision model"):
        ScreeningArm(kind="generic", scorer="decision")
    with pytest.raises(ValueError, match="needs a model"):
        ScreeningArm(kind="generic", scorer="llm")
    with pytest.raises(ValueError, match="compiling criteria needs a model"):
        ScreeningArm(
            kind="criteria",
            scorer="decision",
            decision_model=screening_decisions().model(),
        )


async def test_generic_and_criteria_arms_with_the_decision_model_separate_the_records(
    tmp_path: Path, scripted: ScriptedLLM
) -> None:
    make_synergy_dir(tmp_path)
    review = load_review(tmp_path, REVIEW_KEY)
    decisions = screening_decisions().model()
    generic = ScreeningArm(kind="generic", scorer="decision", decision_model=decisions)
    criteria_arm = ScreeningArm(
        kind="criteria",
        scorer="decision",
        model="ollama:stub",
        decision_model=decisions,
    )
    compiled = await compile_rubric(review, model="ollama:stub")
    assert [c.kind for c in compiled[0]] == ["must", "should", "should"]
    assert compiled[1].llm_calls == 1 and compiled[1].input_tokens == 5

    first = await generic.run(review)
    second = await criteria_arm.run(review, compiled=compiled)

    for outcome in (first, second):
        score = outcome.score
        assert score.n == N_INCLUDED + N_EXCLUDED and score.n_included == N_INCLUDED
        assert score.kept_recall == 1.0 and score.kept_precision == 1.0
        assert score.work_saved_at_95 == pytest.approx(N_EXCLUDED / 20)
        assert outcome.cost.decision_calls == 20  # one request per record
    assert first.criteria == [generic_criterion(review_subquestion(review))]
    assert first.cost.llm_calls == 0  # the generic arm never calls the LLM
    assert (
        second.cost.llm_calls == 1
    )  # the compile call is part of a criteria arm's cost
    assert decisions.calls == 40


async def test_the_llm_scorer_answers_the_same_criteria(
    tmp_path: Path, scripted: ScriptedLLM
) -> None:
    make_synergy_dir(tmp_path)
    review = load_review(tmp_path, REVIEW_KEY)
    arm = ScreeningArm(kind="criteria", scorer="llm", model="ollama:stub")
    compiled = await compile_rubric(review, model="ollama:stub")
    outcome = await arm.run(review, compiled=compiled)
    assert outcome.score.kept_recall == 1.0 and outcome.score.kept_precision == 1.0
    assert outcome.cost.llm_calls == 1 + 20 and outcome.cost.decision_calls == 0
    assert scripted.calls.count(CandidateJudgement) == 20


async def test_a_criteria_arm_without_a_rubric_is_refused(
    tmp_path: Path, scripted: ScriptedLLM
) -> None:
    make_synergy_dir(tmp_path)
    review = load_review(tmp_path, REVIEW_KEY)
    with pytest.raises(ValueError, match="needs the compiled rubric"):
        await ScreeningArm(kind="criteria", scorer="llm", model="ollama:stub").run(
            review
        )


async def test_run_screening_compiles_once_samples_and_records_a_failed_arm(
    tmp_path: Path, scripted: ScriptedLLM
) -> None:
    make_synergy_dir(tmp_path)
    review = load_review(tmp_path, REVIEW_KEY)
    decisions = screening_decisions().model()
    arms = [
        ScreeningArm(kind="generic", scorer="decision", decision_model=decisions),
        ScreeningArm(
            kind="criteria",
            scorer="decision",
            model="ollama:stub",
            decision_model=decisions,
        ),
        ScreeningArm(kind="criteria", scorer="llm", model="ollama:stub"),
    ]
    configs = {a.name: RunConfig(arm=a.name, subset_size=12) for a in arms}
    runs = await run_screening(
        [review], arms, configs=configs, max_records=12, compile_model="ollama:stub"
    )
    assert [r.config.arm for r in runs] == [a.name for a in arms]
    assert (
        scripted.calls.count(CriteriaDraft) == 1
    )  # one rubric, shared by both criteria arms
    for run in runs:
        [item] = run.items
        assert item.n_records == 12 and item.n_records_total == 20
        assert item.score is not None and item.score.n == 12
    assert runs[1].items[0].criteria == runs[2].items[0].criteria


async def test_run_screening_without_a_compile_model_fails_the_criteria_arm_not_the_run(
    tmp_path: Path, scripted: ScriptedLLM
) -> None:
    make_synergy_dir(tmp_path)
    review = load_review(tmp_path, REVIEW_KEY)
    decisions = screening_decisions().model()
    arms = [
        ScreeningArm(kind="generic", scorer="decision", decision_model=decisions),
        ScreeningArm(kind="criteria", scorer="llm", model="ollama:stub"),
    ]
    runs = await run_screening(
        [review],
        arms,
        configs={a.name: RunConfig(arm=a.name, subset_size=0) for a in arms},
        max_records=None,
        compile_model=None,
    )
    assert runs[0].summary.n_errors == 0
    assert runs[1].summary.n_errors == 1
    assert "no compiled criteria" in (runs[1].items[0].error or "")


# -- result files, tables and the CLI ----------------------------------------------------


async def test_results_round_trip_and_render_side_by_side(tmp_path: Path) -> None:
    web = await run_web(
        [WEB_QUESTIONS[0]],
        StubArm(
            {WEB_QUESTIONS[0].question: output(WEB_QUESTIONS[0].expected)}, name="a"
        ),
        config=RunConfig(arm="a", model="m", subset_size=1),
    )
    lit = await run_litsearch(
        [lit_item(1)],
        StubArm({lit_item(1).query.query: output("", works=[])}, name="b"),
        config=RunConfig(arm="b", subset_size=1),
        ks=(5, 10),
    )
    for run in (web, lit):
        again = load_run(dump_run(run))
        assert again == run
    table = render_markdown([web, lit])
    assert "## Web questions" in table and "## LitSearch retrieval" in table
    assert "recall@5" in table and "| a | 1 | 0 | 1.00 |" in table
    assert "- `a`: model m; subset 1, seed 0" in table
    with pytest.raises(ValueError, match="no runs"):
        render_markdown([])


def test_the_report_command_reads_result_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    web = WebRun(
        config=RunConfig(arm="legacy", subset_size=0),
        items=[],
        summary=summarise_web([]),
    )
    path = tmp_path / "web-legacy.json"
    path.write_text(dump_run(web))
    assert cli.main(["report", str(path)]) == 0
    assert "| legacy |" in capsys.readouterr().out
    target = tmp_path / "report.md"
    assert cli.main(["report", str(path), "--out", str(target)]) == 0
    assert target.read_text().startswith("## Web questions")


def test_the_cli_names_the_missing_dataset_and_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "litsearch",
            "--dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
            "--model",
            "ollama:x",
        ]
    )
    assert code == 1
    assert "download litsearch --dir" in capsys.readouterr().err


def test_the_parser_requires_every_directory_and_has_no_default_location() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["download", "litsearch"])  # --dir is required
    with pytest.raises(SystemExit):
        parser.parse_args(["web"])  # --out-dir is required
    args = parser.parse_args(["web", "--out-dir", "out", "--arms", "legacy"])
    assert args.arms == ["legacy"] and args.limit is None


def test_screening_arms_default_to_those_the_given_models_allow() -> None:
    parser = cli.build_parser()
    only_decision = parser.parse_args(
        ["screening", "--dir", "d", "--out-dir", "o", "--decision-model", "nimble:9b"]
    )
    assert cli._screening_arm_names(only_decision) == ["generic+decision"]
    both = parser.parse_args(
        [
            "screening",
            "--dir",
            "d",
            "--out-dir",
            "o",
            "--decision-model",
            "n",
            "--model",
            "m",
        ]
    )
    assert cli._screening_arm_names(both) == list(cli.SCREENING_ARMS)
    neither = parser.parse_args(["screening", "--dir", "d", "--out-dir", "o"])
    with pytest.raises(SystemExit):
        cli._screening_arm_names(neither)


def test_a_judge_and_a_judge_decision_model_cannot_both_be_given() -> None:
    args = cli.build_parser().parse_args(
        ["web", "--out-dir", "o", "--judge-model", "m", "--judge-decision-model", "d"]
    )

    async def attempt() -> None:
        async with AsyncExitStack() as stack:
            await cli._judge(args, stack)

    with pytest.raises(SystemExit, match="not both"):
        asyncio.run(attempt())
