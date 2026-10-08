"""The rule kinds, clean-timing (solo) runs, outage recording and the report's
sections. Stand-in arms over invented items: no model, no network."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from andamentum.deep_research.v2.bench.arms import ArmOutput
from andamentum.deep_research.v2.bench.experiment import Experiment, Rule
from andamentum.deep_research.v2.bench.experiment_report import (
    build_report,
    group_outcome,
    report_from_registry,
)
from andamentum.deep_research.v2.bench.experiment_run import (
    DataDirs,
    Services,
    run_experiment,
)
from andamentum.deep_research.v2.bench.health import SearxDown, SearxHealth
from andamentum.deep_research.v2.bench.metrics import Cost
from andamentum.deep_research.v2.bench.registry import (
    RunRecord,
    StartRecord,
    read_records,
    registry_path,
)
from andamentum.deep_research.v2.bench.verdicts import Verdict, judge_rule
from andamentum.deep_research.v2.types import RunStats

from .test_bench_experiment_run import (
    GOLD_BY_QUESTION,
    ScriptedArm,
    StubGrader,
    correct_behaviour,
    output,
    work,
    write_simpleqa,
)

CLOUD = "ollama:invented-cloud"
GRADER = "ollama:invented-grader-cloud"


# -- the rule kinds ----------------------------------------------------------------


def _rule(**kw: object) -> Rule:
    base: dict[str, object] = {
        "id": "R",
        "question": "q",
        "set": "simpleqa",
        "metric": "llm_calls",
        "arm": "a",
        "reference": "b",
        "test": "reduced_by",
        "threshold": 0.4,
        "aggregate": "median",
    }
    return Rule.model_validate({**base, **kw})


def test_a_reduction_rule_needs_a_reference_a_share_and_no_margin() -> None:
    assert _rule().test == "reduced_by"
    with pytest.raises(ValidationError, match="reference"):
        _rule(reference=None)
    with pytest.raises(ValidationError, match="share"):
        _rule(threshold=None)
    with pytest.raises(ValidationError, match="share"):
        _rule(threshold=1.5)
    with pytest.raises(ValidationError, match="threshold only"):
        _rule(margin=0.1)
    with pytest.raises(ValidationError, match="mean or median"):
        _rule(aggregate="max")


def test_pooled_sets_may_not_repeat_the_rules_own() -> None:
    assert _rule(also_sets=["litsearch-web"]).sets == ["simpleqa", "litsearch-web"]
    with pytest.raises(ValidationError, match="also_sets"):
        _rule(also_sets=["simpleqa"])


def _two_arm_experiment(**extra: object) -> dict[str, object]:
    return {
        "name": "t",
        "registered": "2026-10-08",
        "grader": GRADER,
        "sets": [{"name": "simpleqa", "n": 2}],
        "arms": [
            {"name": "a", "system": "v2", "model": CLOUD},
            {"name": "b", "system": "v2", "model": CLOUD},
        ],
        "baselines": {"simpleqa": "b"},
        **extra,
    }


def test_a_decision_may_only_require_a_group_some_rule_belongs_to() -> None:
    rules = [_rule(group="R2").model_dump(mode="json")]
    decision = {"id": "d", "requires": ["R2"], "if_accepted": "x", "otherwise": "y"}
    Experiment.model_validate(_two_arm_experiment(rules=rules, decisions=[decision]))
    with pytest.raises(ValidationError, match="group R9"):
        Experiment.model_validate(
            _two_arm_experiment(
                rules=rules, decisions=[{**decision, "requires": ["R9"]}]
            )
        )


def _judge(
    rule: Rule, arm: Mapping[str, float | None], ref: Mapping[str, float | None]
) -> Verdict:
    return judge_rule(rule, arm=arm, reference=ref, resamples=500, seed=0)


def test_a_reduction_is_accepted_when_the_median_reaches_the_share() -> None:
    reference = {f"i{n}": 100.0 for n in range(10)}
    big = {k: 40.0 for k in reference}  # a 60% cut everywhere
    verdict = _judge(_rule(), big, reference)
    assert verdict.outcome == "accepted"
    assert "60.0%" in verdict.statement and "at least 40%" in verdict.statement
    small = {k: 80.0 for k in reference}  # a 20% cut everywhere
    assert _judge(_rule(), small, reference).outcome == "not accepted"


def test_a_reduction_skips_items_without_something_to_reduce() -> None:
    verdict = _judge(
        _rule(),
        {"a": 1.0, "b": 5.0, "c": None},
        {"a": 10.0, "b": 0.0, "c": 10.0},
    )
    assert verdict.interval is not None and verdict.interval.n_items == 1
    assert _judge(_rule(), {"a": 1.0}, {"a": 0.0}).outcome == "inconclusive"


def test_the_largest_value_rule_is_exact() -> None:
    rule = _rule(
        test="at_most", threshold=0, aggregate="max", reference=None, metric="x"
    )
    none = judge_rule(
        rule, arm={"a": 0.0, "b": 0.0}, reference=None, resamples=10, seed=0
    )
    assert none.outcome == "accepted"
    one = judge_rule(
        rule, arm={"a": 0.0, "b": 1.0}, reference=None, resamples=10, seed=0
    )
    assert one.outcome == "not accepted" and "largest" in one.statement


def test_a_group_is_accepted_only_when_every_rule_is() -> None:
    def verdict(outcome: str) -> Verdict:
        return Verdict(
            rule_id="r",
            question="q",
            outcome=outcome,  # type: ignore[arg-type]
            gate=True,
            interval=None,
            statement="",
        )

    assert group_outcome([verdict("accepted"), verdict("accepted")]) == "accepted"
    assert (
        group_outcome([verdict("accepted"), verdict("inconclusive")]) == "inconclusive"
    )
    assert (
        group_outcome([verdict("inconclusive"), verdict("not accepted")])
        == "not accepted"
    )
    assert group_outcome([]) == "inconclusive"


# -- running: solo arms and outages ------------------------------------------------


def _experiment(arms: list[dict[str, object]], **extra: object) -> Experiment:
    return Experiment.model_validate(
        {
            "name": "tiny2",
            "registered": "2026-10-08",
            "repeats": 1,
            "resamples": 200,
            "grader": GRADER,
            "sets": [{"name": "simpleqa", "n": 5, "seed": 0}],
            "arms": arms,
            "baselines": {"simpleqa": arms[0]["name"]},
            **extra,
        }
    )


def _arm(name: str, **kw: object) -> dict[str, object]:
    return {"name": name, "system": "v2", "model": CLOUD, **kw}


@pytest.fixture
def dirs(tmp_path: Path) -> DataDirs:
    return DataDirs(simpleqa=write_simpleqa(tmp_path / "sqa"))


class TrackingArm(ScriptedArm):
    """An arm that notes how many cells were in flight when it ran."""

    inflight = 0
    peak: dict[str, int] = {}

    async def run(self, question: str) -> ArmOutput:
        type(self).inflight += 1
        self.peak[self.name] = max(self.peak.get(self.name, 0), type(self).inflight)
        try:
            await asyncio.sleep(0.01)
            return await super().run(question)
        finally:
            type(self).inflight -= 1


async def test_solo_arms_run_last_one_cell_at_a_time_on_their_first_items(
    tmp_path: Path, dirs: DataDirs
) -> None:
    TrackingArm.inflight = 0
    TrackingArm.peak = {}
    exp = _experiment([_arm("base"), _arm("timed", solo=2), _arm("other")])
    arms = {n: TrackingArm(n, correct_behaviour) for n in ("base", "timed", "other")}
    written = await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=Services(
            answer_arm=lambda spec, _m: arms[spec.name],
            grader=StubGrader(),  # type: ignore[arg-type]
        ),
        concurrency=4,
    )
    assert sorted(r.arm for r in written[:10]) == ["base"] * 5 + ["other"] * 5
    assert [r.arm for r in written[10:]] == ["timed"] * 2
    assert TrackingArm.peak["base"] > 1  # the others ran concurrently
    assert TrackingArm.peak["timed"] == 1  # the solo cells never overlapped
    start = next(
        r for r in read_records(registry_path(tmp_path / "out")) if r.kind == "start"
    )
    assert isinstance(start, StartRecord) and start.concurrency == 4
    assert {a.name: a.solo for a in start.arms}["timed"] == 2


def _services(arm: ScriptedArm, probe: object = None) -> Services:
    return Services(
        answer_arm=lambda *_: arm,
        grader=StubGrader(),  # type: ignore[arg-type]
        check_searxng=probe,  # type: ignore[arg-type]
    )


async def test_an_outage_cell_of_a_record_arm_is_failed_not_zero(
    tmp_path: Path, dirs: DataDirs
) -> None:
    probes: list[str | None] = []

    async def probe(url: str | None) -> SearxHealth:
        probes.append(url)
        if len(probes) in (2, 4):
            raise SearxDown("engines are suspended")
        return SearxHealth(url=url or "", ok=True, n_results=3)

    exp = _experiment([_arm("a", outage="record")])
    arm = ScriptedArm("a", correct_behaviour)
    written = await run_experiment(
        exp, out_dir=tmp_path / "out", dirs=dirs, services=_services(arm, probe)
    )
    assert len(written) == 5 and len(probes) == 5  # probed before every web cell
    blocked = [r for r in written if r.outage]
    assert len(blocked) == 2
    for record in blocked:
        assert record.metrics == {}  # not a zero, not a value
        assert record.error and "engines are suspended" in record.error
        assert record.output is None
    assert len(arm.calls) == 3  # the blocked cells never ran the arm
    assert all(r.metrics["correct"] == 1.0 for r in written if not r.outage)


async def test_an_outage_stops_the_run_unless_the_arm_records_it(
    tmp_path: Path, dirs: DataDirs
) -> None:
    async def probe(url: str | None) -> SearxHealth:
        raise SearxDown("engines are suspended")

    exp = _experiment([_arm("a")])
    arm = ScriptedArm("a", correct_behaviour)
    with pytest.raises(SearxDown):
        await run_experiment(
            exp, out_dir=tmp_path / "out", dirs=dirs, services=_services(arm, probe)
        )
    assert arm.calls == [] and not list(
        r
        for r in read_records(registry_path(tmp_path / "out"))
        if isinstance(r, RunRecord)
    )


async def test_failed_searches_inside_a_record_arms_cell_are_recorded_too(
    tmp_path: Path, dirs: DataDirs
) -> None:
    def blocked(question: str, call: int) -> ArmOutput:
        out = correct_behaviour(question, call)
        if call == 1:
            out.degraded = ["searxng search failed for 'x': engines blocked"]
        return out

    exp = _experiment([_arm("a", outage="record")])
    written = await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=_services(ScriptedArm("a", blocked)),
    )
    assert [r.outage for r in written] == [True, False, False, False, False]
    assert written[0].metrics == {}


async def test_any_failed_cell_of_a_record_arm_has_no_metrics(
    tmp_path: Path, dirs: DataDirs
) -> None:
    def crash(question: str, call: int) -> ArmOutput:
        if call == 1:
            raise RuntimeError("legacy pipeline crashed")
        return correct_behaviour(question, call)

    exp = _experiment([_arm("a", outage="record")])
    written = await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=_services(ScriptedArm("a", crash)),
    )
    first = written[0]
    assert first.error and "RuntimeError" in first.error
    assert first.metrics == {} and first.outage is False
    assert all(r.metrics for r in written[1:])


# -- the report --------------------------------------------------------------------


def _timed(question: str, call: int, *, seconds: float, calls: int) -> ArmOutput:
    out = output(
        question, f"The answer is {GOLD_BY_QUESTION[question]}.", [work(1), work(2)]
    )
    stats = RunStats(
        llm_calls=calls,
        wall_s=seconds,
        step_seconds={"Plan": 1.0, "Read": seconds / 2, "Verify": seconds / 4},
        step_llm_calls={"Plan": 1, "Write": 2},
    )
    assert out.result is not None
    out.result = out.result.model_copy(update={"stats": stats})
    out.cost = Cost(wall_s=seconds, llm_calls=calls, decision_calls=0)
    return out


def _report_experiment() -> Experiment:
    arms = [
        _arm("base"),
        _arm("fast"),
        _arm("base-solo", solo=2),
        _arm("fast-solo", solo=2),
    ]
    rules = [
        {
            "id": "R2-calls",
            "group": "R2",
            "question": "fast makes at least 40% fewer model calls",
            "set": "simpleqa",
            "metric": "llm_calls",
            "arm": "fast",
            "reference": "base",
            "test": "reduced_by",
            "threshold": 0.4,
            "aggregate": "median",
        },
        {
            "id": "R6-fast",
            "group": "R6",
            "question": "clean wall time of fast at most 300 s",
            "set": "simpleqa",
            "metric": "wall_s",
            "arm": "fast-solo",
            "test": "at_most",
            "threshold": 300,
            "aggregate": "median",
            "gate": False,
        },
    ]
    decisions = [
        {
            "id": "adopt",
            "requires": ["R2"],
            "if_accepted": "Fast becomes the default.",
            "otherwise": "Base stays the default.",
        }
    ]
    return _experiment(arms, rules=rules, decisions=decisions)


def _behave(seconds: float, calls: int) -> Callable[[str, int], ArmOutput]:
    return lambda q, n: _timed(q, n, seconds=seconds, calls=calls)


async def test_the_report_has_the_step_solo_and_decision_sections(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = _report_experiment()
    arms = {
        "base": ScriptedArm("base", _behave(40.0, 20)),
        "fast": ScriptedArm("fast", _behave(20.0, 8)),
        "base-solo": ScriptedArm("base-solo", _behave(30.0, 20)),
        "fast-solo": ScriptedArm("fast-solo", _behave(12.0, 8)),
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
    text = report_from_registry(exp, out)

    # solo arms stay out of the ordinary tables and have their own section
    means = text.split("## Set simpleqa")[1].split("## Cost")[0]
    assert "| base |" in means and "| fast |" in means and "-solo" not in means
    solo = text.split("## Clean timing (solo runs)")[1].split("## Verdicts")[0]
    assert "| fast-solo |" in solo and "12.0 s (2 runs)" in solo
    # per-step tables, for the steps of the new loop
    steps = text.split("## Where the time and the requests went")[1].split("## ")[0]
    assert (
        "| Plan | Search | Expand | Choose | Read | Notes | Reflect | Write | Verify |"
        in steps
    )
    assert "Screen" not in steps and "decision" not in steps
    # verdicts: the group and its wording
    verdicts = text.split("## Verdicts")[1]
    assert "| R2-calls | R2 |" in verdicts and "**accepted**" in verdicts
    assert "(R2 accepted): Fast becomes the default." in verdicts
    assert "Base stays the default." not in verdicts


async def test_a_rejected_group_gives_the_otherwise_wording(
    tmp_path: Path, dirs: DataDirs
) -> None:
    exp = _report_experiment()
    arms = {
        n: ScriptedArm(n, _behave(20.0, 20))
        for n in ("base", "fast", "base-solo", "fast-solo")
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
    verdicts = report_from_registry(exp, out).split("## Verdicts")[1]
    assert "(R2 not accepted): Base stays the default." in verdicts
    assert "Fast becomes the default." not in verdicts


async def test_the_failed_runs_table_counts_outages_apart_from_other_errors(
    tmp_path: Path, dirs: DataDirs
) -> None:
    calls = iter(range(100))

    async def probe(url: str | None) -> SearxHealth:
        if next(calls) == 0:
            raise SearxDown("blocked")
        return SearxHealth(url=url or "", ok=True, n_results=1)

    def flaky(question: str, call: int) -> ArmOutput:
        if call == 1:
            raise RuntimeError("model fell over")
        return correct_behaviour(question, call)

    exp = _experiment([_arm("a", outage="record")])
    out = tmp_path / "out"
    await run_experiment(
        exp, out_dir=out, dirs=dirs, services=_services(ScriptedArm("a", flaky), probe)
    )
    text = report_from_registry(exp, out)
    failed = text.split("## Failed runs")[1].split("## ")[0]
    assert "| a | simpleqa | 5 | 2 | 1 | 1 |" in failed
    assert "not a zero" in failed
    runs = [r for r in read_records(registry_path(out)) if isinstance(r, RunRecord)]
    assert [bool(r.outage) for r in runs][:2] == [True, False]
    assert runs[0].metrics == {} and runs[1].metrics == {}


def test_an_empty_registry_still_reports_the_rules_as_inconclusive() -> None:
    text = build_report(_report_experiment(), [])
    assert "## Verdicts" in text and "inconclusive" in text
