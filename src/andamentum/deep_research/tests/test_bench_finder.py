"""The bench side of the finder: the finder and plain Semantic Scholar systems,
the ``exceeds_by`` rule, and results by subgroup. Stand-in arms and a mock
transport; nothing touches the network or a model."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from andamentum.references import Work

from ..bench import arms as bench_arms
from ..bench.arms import ArmOutput, FinderArm, SemanticScholarArm
from ..bench.attribution import FINDER_STAGES, trace_finder
from ..bench.experiment import ArmSpec, Experiment, Rule
from ..bench.experiment_report import report_from_registry
from ..bench.experiment_run import (
    DataDirs,
    Services,
    build_answer_arm,
    infrastructure_failure,
    load_items,
    run_experiment,
)
from ..bench.metrics import Cost
from ..bench.verdicts import judge_rule
from ..finder import FINDER_SOURCES, FinderResult
from ..sources.base import SourceClient, SourceConfig
from ..store import ResearchStore
from .test_bench_experiment_run import ScriptedArm, write_litsearch

# -- arms in the experiment file ------------------------------------------------------


def test_a_finder_arm_needs_its_switch_and_a_model() -> None:
    ArmSpec(name="F1", system="finder", model="ollama:m", follow_citations=True)
    with pytest.raises(ValidationError, match="follow_citations"):
        ArmSpec(name="F", system="finder", model="ollama:m")
    with pytest.raises(ValidationError, match="needs a model"):
        ArmSpec(name="F", system="finder", follow_citations=False)


def test_the_switch_is_refused_for_other_systems() -> None:
    with pytest.raises(ValidationError, match="follow_citations"):
        ArmSpec(name="R", system="research", model="ollama:m", follow_citations=True)


def test_plain_semantic_scholar_calls_no_model() -> None:
    ArmSpec(name="B2", system="s2_search")
    with pytest.raises(ValidationError, match="calls no model"):
        ArmSpec(name="B2", system="s2_search", model="ollama:m")
    with pytest.raises(ValidationError, match="cannot run on"):
        ArmSpec(name="B2", system="s2_search", sets=["simpleqa"])


def test_the_arms_are_built_by_system() -> None:
    sources = SourceConfig(
        semantic_scholar_api_key="invented-key", mailto="jane@example.org"
    )
    services = Services(source_config=sources)
    finder = build_answer_arm(
        ArmSpec(name="F0", system="finder", model="ollama:m", follow_citations=False),
        "scholarly",
        services,
    )
    assert isinstance(finder, FinderArm) and finder.name == "F0"
    assert finder.follow_citations is False
    # the finder's sources with the bench's keys and address
    assert finder.sources.scholarly == FINDER_SOURCES.scholarly
    assert finder.sources.expand_with == FINDER_SOURCES.expand_with
    assert finder.sources.semantic_scholar_api_key == "invented-key"
    assert finder.sources.mailto == "jane@example.org"
    plain = build_answer_arm(
        ArmSpec(name="B2", system="s2_search"), "scholarly", services
    )
    assert isinstance(plain, SemanticScholarArm) and plain.name == "B2"


# -- the plain Semantic Scholar arm ---------------------------------------------------


def s2_item(number: int) -> dict[str, object]:
    return {
        "paperId": f"{number:040d}",
        "externalIds": {"DOI": f"10.5555/plain.{number}"},
        "title": f"Invented kiln paper number {number}",
        "abstract": "About kilns.",
        "year": 2020,
        "venue": "Invented Kiln Letters",
        "authors": [],
    }


def mock_client_factory(handler: httpx.MockTransport, sent: list[httpx.Request]):  # noqa: ANN201
    def build(store: ResearchStore, *, patient: tuple[str, ...] = ()) -> SourceClient:
        return SourceClient(
            store,
            client=httpx.AsyncClient(transport=handler),
            intervals={"semantic_scholar": 0.0},
            clock=lambda: 0.0,
            sleep=_no_sleep,
            retries=0,
            patient=patient,
        )

    return build


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_the_plain_arm_returns_the_services_top_in_its_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            200, json={"total": 25, "data": [s2_item(n) for n in range(25)]}
        )

    monkeypatch.setattr(
        bench_arms,
        "SourceClient",
        mock_client_factory(httpx.MockTransport(handle), sent),
    )
    arm = SemanticScholarArm(sources=SourceConfig(), top=20)
    out = await arm.run("Which paper cools kiln lids with salt?")
    assert [w.title for w in out.works] == [
        f"Invented kiln paper number {n}" for n in range(20)
    ]
    assert len(sent) == 1
    assert sent[0].url.params["query"] == "Which paper cools kiln lids with salt?"
    assert out.cost.llm_calls == 0 and out.degraded == []


async def test_a_refused_plain_search_is_an_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[httpx.Request] = []
    transport = httpx.MockTransport(
        lambda _r: httpx.Response(429, json={"message": "slow down"})
    )
    monkeypatch.setattr(
        bench_arms, "SourceClient", mock_client_factory(transport, sent)
    )
    out = await SemanticScholarArm(sources=SourceConfig()).run("Which kiln?")
    assert out.works == []
    assert infrastructure_failure(out) is not None


# -- the exceeds_by rule ----------------------------------------------------------------


def rule(**kw: object) -> Rule:
    return Rule.model_validate(
        {
            "id": "F-over-B",
            "question": "Is F at least 0.10 above B?",
            "set": "litsearch-web",
            "metric": "recall_at_20",
            "arm": "F",
            "reference": "B",
            "test": "exceeds_by",
            "margin": 0.10,
        }
        | kw
    )


def test_exceeds_by_reads_the_mean_and_reports_the_interval() -> None:
    reference = {f"i{n}": 0.0 for n in range(8)}
    # mean +0.125 with items of 0 and 1: the interval reaches below the margin
    arm = {f"i{n}": (1.0 if n == 0 else 0.0) for n in range(8)}
    verdict = judge_rule(rule(), arm=arm, reference=reference, resamples=500, seed=0)
    assert verdict.outcome == "accepted"
    assert verdict.interval is not None and verdict.interval.low < 0.10
    assert "95% interval" in verdict.statement and "at least +0.1" in verdict.statement
    below = {f"i{n}": 0.05 for n in range(8)}
    verdict = judge_rule(rule(), arm=below, reference=reference, resamples=500, seed=0)
    assert verdict.outcome == "not accepted"


def test_exceeds_by_needs_a_reference_and_takes_no_threshold() -> None:
    with pytest.raises(ValidationError, match="needs a reference"):
        rule(reference=None)
    with pytest.raises(ValidationError, match="takes a margin"):
        rule(threshold=0.1)


# -- results by subgroup -----------------------------------------------------------------


def finds_gold_by_question(gold: dict[str, list[Work]]):  # noqa: ANN201
    def behave(question: str, _call: int) -> ArmOutput:
        return ArmOutput(
            answer_text="", works=gold[question], cost=Cost(wall_s=1.0, llm_calls=0)
        )

    return behave


async def test_litsearch_cells_carry_the_query_set_and_the_report_splits_by_it(
    tmp_path: Path,
) -> None:
    dirs = DataDirs(litsearch_local=write_litsearch(tmp_path / "ls"))
    exp = Experiment.model_validate(
        {
            "name": "finder-tiny",
            "registered": "2026-10-08",
            "resamples": 200,
            "sets": [{"name": "litsearch-web", "n": 3, "seed": 0}],
            "arms": [
                {"name": "B2", "system": "s2_search"},
                {
                    "name": "F1",
                    "system": "finder",
                    "model": "ollama:m",
                    "follow_citations": True,
                },
            ],
            "baselines": {"litsearch-web": "B2"},
            "rules": [
                {
                    "id": "F1-over-B2",
                    "question": "Is F1 at least 0.10 above B2?",
                    "set": "litsearch-web",
                    "metric": "recall_at_20",
                    "arm": "F1",
                    "reference": "B2",
                    "test": "exceeds_by",
                    "margin": 0.10,
                }
            ],
        }
    )
    items = load_items(exp.set("litsearch-web"), dirs)
    gold = {i.question: list(i.litsearch.gold) for i in items if i.litsearch}
    stubs = {
        "B2": ScriptedArm("B2", finds_gold_by_question({q: [] for q in gold})),
        "F1": ScriptedArm("F1", finds_gold_by_question(gold)),
    }
    out = tmp_path / "out"
    written = await run_experiment(
        exp,
        out_dir=out,
        dirs=dirs,
        services=Services(answer_arm=lambda spec, _mode: stubs[spec.name]),
    )
    assert {r.group for r in written} == {"inline_acl"}
    text = report_from_registry(exp, out)
    assert "Means by subgroup" in text
    assert "| inline_acl | F1 | 3 |" in text
    assert "finder (follow_citations: true)" in text
    assert "F1-over-B2" in text and "accepted" in text


# -- loss attribution of a finder cell ---------------------------------------------------


def kiln_work(number: int) -> Work:
    return Work(title=f"Invented kiln paper number {number}", work_id=f"w_{number}")


def finder_output(
    *, searched: list[int], followed: list[int], shown: list[int], chosen: list[int]
) -> ArmOutput:
    """A finder cell's output: works 0..29 in rank order, the step lists by number."""
    ids = [f"w_{n}" for n in (shown or range(30))]
    return ArmOutput(
        answer_text="",
        works=[kiln_work(n) for n in range(30)],
        cost=Cost(wall_s=1.0, llm_calls=3),
        finder=FinderResult(
            sentence="Which kiln?",
            follow_citations=True,
            searched=[f"w_{n}" for n in searched],
            followed=[f"w_{n}" for n in followed],
            shown=ids if shown else [],
            chosen=[f"w_{n}" for n in chosen],
        ),
    )


def test_a_finder_cell_is_traced_through_its_step_lists() -> None:
    out = finder_output(
        searched=[0, 1, 25], followed=[1, 2, 26], shown=[0, 1, 2, 25], chosen=[1]
    )
    gold = [Work(title=f"Invented kiln paper number {n}") for n in (1, 25, 26, 99)]
    traces = trace_finder(out, gold)
    assert [t.lost_at for t in traces] == [
        "found",  # searched and followed, chosen, top 20
        "ranked_top_20",  # shown to the ranking call, ranked 26th
        "shown",  # followed only, never shown
        "retrieved",  # neither searched nor followed
    ]
    assert traces[0].stages == {
        "searched": True,
        "followed": True,
        "retrieved": True,
        "shown": True,
        "chosen": True,
        "ranked_top_20": True,
    }
    assert list(traces[1].stages) == list(FINDER_STAGES)
    with pytest.raises(ValueError, match="finder"):
        trace_finder(out.model_copy(update={"finder": None}), gold)
