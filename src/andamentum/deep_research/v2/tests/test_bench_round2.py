"""Round 2 of the experiment machinery: the registered file, the new rule
kinds, clean-timing (solo) runs, outage recording, and the report's new
sections. Stand-in arms over invented items: no model, no network."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from andamentum.deep_research.v2 import bench
from andamentum.deep_research.v2.bench import experiment_run
from andamentum.deep_research.v2.bench.arms import ArmOutput
from andamentum.deep_research.v2.bench.experiment import (
    Experiment,
    Rule,
    load_experiment,
)
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
    read_records,
    registry_path,
)
from andamentum.deep_research.v2.bench.verdicts import Verdict, judge_rule
from andamentum.deep_research.v2.types import RunStats

from .test_bench_experiment_run import (
    ScriptedArm,
    correct_answer,
    output,
    stable_behaviour,
    work,
)

ROUND2 = Path(bench.__file__).parent / "experiments" / "round2.yaml"
CLOUD = "ollama:invented-cloud"


# -- the registered file -----------------------------------------------------------


def test_the_round_two_file_registers_the_planned_arms_rules_and_wording() -> None:
    exp = load_experiment(ROUND2)
    assert exp.name == "round2" and exp.repeats == 1
    assert [(s.name, s.n) for s in exp.sets[:3]] == [
        ("litsearch", 28),
        ("web", 15),
        ("web_planted", 15),
    ]
    assert exp.sets[3].records == 150 and len(exp.sets[3].reviews or []) == 8

    by_name = {a.name: a for a in exp.arms}
    assert by_name["v2-baseline"].preset == "baseline"
    assert by_name["v2-fast"].preset == "fast"
    assert by_name["v2-fast-tev"].decision_model == "tev1:0.8b"
    assert by_name["v2-baseline"].decision_model == "nimble:9b"
    assert all(by_name[n].is_frozen for n in ("v2-baseline", "v2-fast", "v2-fast-tev"))
    legacy = by_name["legacy"]
    assert not legacy.runs_on("web") and legacy.runs_on("litsearch")
    assert legacy.outage == "record"
    assert by_name["screen-generic-nimble"].decision_model == "nimble:9b"
    assert by_name["screen-generic-tev"].decision_model == "tev1:0.8b"
    solo = {a.name: a for a in exp.solo_arms}
    assert set(solo) == {"v2-baseline-solo", "v2-fast-solo", "v2-fast-tev-solo"}
    assert all(a.solo == 3 and a.sets == ["litsearch", "web"] for a in solo.values())

    groups = exp.rule_groups()
    assert list(groups) == ["R1", "R2", "R3", "R4", "R5", "R6"]
    r1 = {r.metric: r for r in groups["R1"]}
    assert set(r1) == {
        "claim_precision",
        "correct",
        "recall_at_20",
        "adopted_false_claim",
    }
    for metric in ("claim_precision", "correct", "recall_at_20"):
        assert (r1[metric].test, r1[metric].margin, r1[metric].gate) == (
            "noninferior",
            0.05,
            True,
        )
    planted = r1["adopted_false_claim"]
    assert (planted.test, planted.threshold, planted.aggregate) == (
        "at_most",
        0,
        "max",
    )
    [r2] = groups["R2"]
    assert (r2.test, r2.threshold, r2.aggregate, r2.metric) == (
        "reduced_by",
        0.4,
        "median",
        "decision_calls",
    )
    assert (r2.arm, r2.reference, r2.gate) == ("v2-fast", "v2-baseline", True)
    assert r2.sets == ["litsearch", "web"]
    assert {(r.arm, r.reference) for r in groups["R3"]} == {
        ("v2-fast-tev", "v2-fast"),
        ("v2-fast-tev", None),
    }
    [r4] = groups["R4"]
    assert (r4.arm, r4.reference, r4.set, r4.margin, r4.gate) == (
        "v2-baseline",
        "legacy",
        "litsearch",
        0.05,
        False,
    )
    [r5] = groups["R5"]
    assert (r5.metric, r5.margin, r5.gate) == ("wss_at_95", 0.05, False)
    assert {r.arm for r in groups["R6"]} == set(solo)
    assert all(r.threshold == 300 and r.metric == "wall_s" for r in groups["R6"])
    assert exp.judge is not None and exp.judge.decision_model == "nimble:9b"

    said = {d.id: d for d in exp.decisions}
    assert said["default-preset"].requires == ["R1", "R2"]
    assert said["decision-model"].requires == ["R3"]
    assert "replaces the baseline as the default" in said["default-preset"].if_accepted
    assert "recommended decision model" in said["decision-model"].if_accepted


# -- the new rule kinds ------------------------------------------------------------


def _rule(**kw: object) -> Rule:
    base: dict[str, object] = {
        "id": "R",
        "question": "q",
        "set": "web",
        "metric": "decision_calls",
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
    assert _rule(also_sets=["litsearch"]).sets == ["web", "litsearch"]
    with pytest.raises(ValidationError, match="also_sets"):
        _rule(also_sets=["web"])


def _two_arm_experiment(**extra: object) -> dict[str, object]:
    return {
        "name": "t",
        "registered": "2026-10-06",
        "sets": [{"name": "web", "n": 2}],
        "arms": [
            {"name": "a", "system": "v2", "preset": "fast", "model": CLOUD},
            {"name": "b", "system": "v2", "preset": "baseline", "model": CLOUD},
        ],
        "baselines": {"web": "b"},
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


def test_solo_and_outage_belong_to_the_arms_they_make_sense_for() -> None:
    base = {"name": "s", "system": "screening", "screen": "baseline+llm", "model": "m"}
    with pytest.raises(ValidationError, match="solo"):
        Experiment.model_validate(
            _two_arm_experiment(arms=[{**base, "solo": 2}], baselines={})
        )
    with pytest.raises(ValidationError, match="outage"):
        Experiment.model_validate(
            _two_arm_experiment(
                arms=[
                    {
                        "name": "f",
                        "system": "v2",
                        "preset": "baseline",
                        "model": "m",
                        "outage": "record",
                    }
                ],
                baselines={},
            )
        )  # a frozen arm has no outage to record


def _judge(
    rule: Rule, arm: Mapping[str, float | None], ref: Mapping[str, float | None]
):
    return judge_rule(rule, arm=arm, reference=ref, resamples=500, seed=0)


def test_a_reduction_is_accepted_when_the_median_reaches_the_share() -> None:
    reference = {f"i{n}": 100.0 for n in range(10)}
    big = {k: 40.0 for k in reference}  # a 60% cut everywhere
    verdict = _judge(_rule(), big, reference)
    assert verdict.outcome == "accepted"
    assert "60.0%" in verdict.statement and "at least 40%" in verdict.statement
    assert "95% interval" in verdict.statement  # reported beside the point value

    small = {k: 80.0 for k in reference}  # a 20% cut everywhere
    assert _judge(_rule(), small, reference).outcome == "not accepted"

    # a wide interval does not turn a median above the bar into "inconclusive"
    mixed = {k: (10.0 if n < 6 else 90.0) for n, k in enumerate(reference)}
    verdict = _judge(_rule(), mixed, reference)
    assert verdict.outcome == "accepted"
    assert verdict.interval is not None and verdict.interval.low < 0.4
    just_under = {k: (61.0 if n < 6 else 10.0) for n, k in enumerate(reference)}
    assert _judge(_rule(), just_under, reference).outcome == "not accepted"


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
        test="at_most", threshold=0, aggregate="max", reference=None, metric="adopted"
    )
    none = judge_rule(
        rule, arm={"a": 0.0, "b": 0.0}, reference=None, resamples=10, seed=0
    )
    assert none.outcome == "accepted"
    one = judge_rule(
        rule, arm={"a": 0.0, "b": 1.0}, reference=None, resamples=10, seed=0
    )
    assert one.outcome == "not accepted" and "largest" in one.statement


# -- running: solo arms and outages ------------------------------------------------


def _experiment(arms: list[dict[str, object]], **extra: object) -> Experiment:
    return Experiment.model_validate(
        {
            "name": "tiny2",
            "registered": "2026-10-06",
            "repeats": 1,
            "resamples": 200,
            "sets": [{"name": "web", "n": 5, "seed": 0}],
            "arms": arms,
            "baselines": {"web": arms[0]["name"]},
            **extra,
        }
    )


def _live(name: str, **kw: object) -> dict[str, object]:
    return {
        "name": name,
        "system": "v2",
        "preset": "baseline",
        "model": CLOUD,
        "frozen": False,
        **kw,
    }


class TrackingArm(ScriptedArm):
    """An arm that notes how many cells were in flight when it ran."""

    inflight = 0
    peak: dict[str, int] = {}

    async def run(
        self, question: str, *, frozen: object = None, store: object = None
    ) -> ArmOutput:
        type(self).inflight += 1
        self.peak[self.name] = max(self.peak.get(self.name, 0), type(self).inflight)
        try:
            await asyncio.sleep(0.01)
            return await super().run(question, frozen=frozen, store=store)
        finally:
            type(self).inflight -= 1


async def test_solo_arms_run_last_one_cell_at_a_time_on_their_first_items(
    tmp_path: Path,
) -> None:
    TrackingArm.inflight = 0
    TrackingArm.peak = {}
    exp = _experiment(
        [_live("base"), _live("timed", solo=2), _live("other")],
    )
    arms = {n: TrackingArm(n, stable_behaviour) for n in ("base", "timed", "other")}
    written = await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=Services(answer_arm=lambda spec, _m, _d: arms[spec.name]),
        concurrency=4,
    )
    assert [r.arm for r in written] == ["base"] * 5 + ["other"] * 5 + ["timed"] * 2
    assert TrackingArm.peak["base"] > 1  # the others ran concurrently
    assert TrackingArm.peak["timed"] == 1  # the solo cells never overlapped
    items = experiment_run.load_items(exp.set("web"), DataDirs())
    timed = [r.item_id for r in written if r.arm == "timed"]
    assert timed == [i.item_id for i in items[:2]]
    start = next(r for r in read_records(registry_path(tmp_path)) if r.kind == "start")
    assert {a.name: a.solo for a in start.arms}["timed"] == 2  # type: ignore[union-attr]


async def test_an_outage_cell_of_a_record_arm_is_failed_not_zero(
    tmp_path: Path,
) -> None:
    probes: list[str | None] = []

    async def probe(url: str | None) -> SearxHealth:
        probes.append(url)
        if len(probes) in (2, 4):
            raise SearxDown("engines are suspended")
        return SearxHealth(url=url or "", ok=True, n_results=3)

    exp = _experiment([_live("a", outage="record")])
    arm = ScriptedArm("a", stable_behaviour)
    svc = Services(answer_arm=lambda *_: arm, check_searxng=probe)
    written = await run_experiment(exp, out_dir=tmp_path, dirs=DataDirs(), services=svc)

    assert len(written) == 5 and len(probes) == 5  # probed before every cell
    blocked = [r for r in written if r.outage]
    assert len(blocked) == 2
    for record in blocked:
        assert record.metrics == {}  # not a zero, not a value
        assert record.error and "engines are suspended" in record.error
        assert record.output is None
    assert len(arm.calls) == 3  # the blocked cells never ran the arm
    ok = [r for r in written if not r.outage]
    assert all(r.metrics["correct"] == 1.0 for r in ok)


async def test_failed_searches_inside_a_record_arms_cell_are_recorded_too(
    tmp_path: Path,
) -> None:
    def blocked(question: str, call: int) -> ArmOutput:
        out = output(question, correct_answer(question), [work(1)])
        if call == 1:
            out.degraded = ["searxng search failed for 'x': engines blocked"]
        return out

    exp = _experiment([_live("a", outage="record")])
    arm = ScriptedArm("a", blocked)
    written = await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=Services(answer_arm=lambda *_: arm),
    )
    assert [r.outage for r in written] == [True, False, False, False, False]
    assert written[0].metrics == {}


async def test_any_failed_cell_of_a_record_arm_has_no_metrics(
    tmp_path: Path,
) -> None:
    def crash(question: str, call: int) -> ArmOutput:
        if call == 1:
            raise RuntimeError("legacy pipeline crashed")
        return output(question, correct_answer(question), [work(1)])

    exp = _experiment([_live("a", outage="record")])
    arm = ScriptedArm("a", crash)
    written = await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=Services(answer_arm=lambda *_: arm),
    )
    first = written[0]
    assert first.error and "RuntimeError" in first.error
    assert first.metrics == {} and first.outage is False
    assert all(r.metrics for r in written[1:])


async def test_the_legacy_arm_is_probed_before_every_litsearch_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp = Experiment.model_validate(
        {
            "name": "legacy-probe",
            "registered": "2026-10-06",
            "resamples": 100,
            "sets": [{"name": "litsearch", "n": 2}],
            "arms": [
                {
                    "name": "old",
                    "system": "legacy",
                    "model": CLOUD,
                    "outage": "record",
                }
            ],
            "baselines": {"litsearch": "old"},
        }
    )
    items = [
        experiment_run.Item("litsearch", "L1", "a literature question?"),
        experiment_run.Item("litsearch", "L2", "another literature question?"),
    ]
    monkeypatch.setattr(experiment_run, "load_items", lambda spec, dirs: items)
    probes: list[str | None] = []

    async def probe(url: str | None) -> SearxHealth:
        probes.append(url)
        raise SearxDown("blocked")

    arm = ScriptedArm("old", stable_behaviour)
    written = await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=Services(answer_arm=lambda *_: arm, check_searxng=probe),
    )
    assert len(probes) == 2 and arm.calls == []
    assert all(r.outage and r.metrics == {} for r in written)


# -- the report --------------------------------------------------------------------


def _timed(question: str, call: int, *, seconds: float, calls: int) -> ArmOutput:
    out = output(question, correct_answer(question), [work(1), work(2)])
    stats = RunStats(
        llm_calls=3,
        decision_calls=calls,
        wall_s=seconds,
        step_seconds={"Plan": 1.0, "Screen": seconds / 2, "Verify": seconds / 4},
        step_llm_calls={"Plan": 1, "Write": 2},
        step_decision_calls={"Screen": calls - 1, "Verify": 1},
        verify_by_code=3,
        verify_by_model=1,
        decision_cache="enabled",
        decision_cache_hits=2,
    )
    assert out.result is not None
    out.result = out.result.model_copy(update={"stats": stats})
    out.cost = Cost(wall_s=seconds, llm_calls=3, decision_calls=calls)
    return out


def _report_experiment() -> Experiment:
    arms = [
        _live("base"),
        _live("fast"),
        _live("base-solo", solo=2),
        _live("fast-solo", solo=2),
    ]
    rules = [
        {
            "id": "R2-calls",
            "group": "R2",
            "question": "fast makes at least 40% fewer decision calls",
            "set": "web",
            "metric": "decision_calls",
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
            "set": "web",
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


async def test_the_report_has_the_step_failure_solo_and_decision_sections(
    tmp_path: Path,
) -> None:
    exp = _report_experiment()

    def behave(seconds: float, calls: int):  # noqa: ANN202
        return lambda q, n: _timed(q, n, seconds=seconds, calls=calls)

    arms = {
        "base": ScriptedArm("base", behave(40.0, 20)),
        "fast": ScriptedArm("fast", behave(20.0, 8)),
        "base-solo": ScriptedArm("base-solo", behave(30.0, 20)),
        "fast-solo": ScriptedArm("fast-solo", behave(12.0, 8)),
    }
    await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=Services(answer_arm=lambda spec, _m, _d: arms[spec.name]),
    )
    text = report_from_registry(exp, tmp_path)

    # solo arms stay out of the ordinary tables and have their own section
    means = text.split("## Set web")[1].split("## Cost")[0]
    assert "| base |" in means and "| fast |" in means and "-solo" not in means
    assert "## Clean timing (solo runs)" in text
    solo = text.split("## Clean timing (solo runs)")[1].split("## Verdicts")[0]
    assert "| fast-solo | live |" in solo and "12.0 s (2 runs)" in solo
    # per-step tables
    assert "## Where the time and the requests went" in text
    steps = text.split("## Where the time and the requests went")[1].split("## ")[0]
    assert (
        "| Plan | Search | Screen | Expand | Read | Coverage | Write | Verify |"
        in steps
    )
    assert "| base | web | 5 | 1.0 | 0.0 | 20.0 |" in steps  # Plan, Search, Screen
    assert "1.0 / 0.0" in steps  # Plan: one language-model call, no decision call
    assert "0.0 / 19.0" in steps  # Screen: no language-model call, 19 decisions
    assert "0.75" in steps  # three of four claim checks settled in code
    assert "enabled 5" in steps
    # verdicts: the group and its wording
    verdicts = text.split("## Verdicts")[1]
    assert "| R2-calls | R2 |" in verdicts and "**accepted**" in verdicts
    assert "(R2 accepted): Fast becomes the default." in verdicts
    assert "Base stays the default." not in verdicts


async def test_a_rejected_group_gives_the_otherwise_wording(tmp_path: Path) -> None:
    exp = _report_experiment()
    same = lambda q, n: _timed(q, n, seconds=20.0, calls=20)  # noqa: E731
    arms = {n: ScriptedArm(n, same) for n in ("base", "fast", "base-solo", "fast-solo")}
    await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=Services(answer_arm=lambda spec, _m, _d: arms[spec.name]),
    )
    verdicts = report_from_registry(exp, tmp_path).split("## Verdicts")[1]
    assert "(R2 not accepted): Base stays the default." in verdicts
    assert "Fast becomes the default." not in verdicts


async def test_the_failed_runs_table_counts_outages_apart_from_other_errors(
    tmp_path: Path,
) -> None:
    async def probe(url: str | None) -> SearxHealth:
        if probe.calls.__next__() == 0:  # type: ignore[attr-defined]
            raise SearxDown("blocked")
        return SearxHealth(url=url or "", ok=True, n_results=1)

    probe.calls = iter(range(100))  # type: ignore[attr-defined]

    def flaky(question: str, call: int) -> ArmOutput:
        if call == 1:
            raise RuntimeError("model fell over")
        return stable_behaviour(question, call)

    exp = _experiment([_live("a", outage="record")])
    arm = ScriptedArm("a", flaky)
    await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=Services(answer_arm=lambda *_: arm, check_searxng=probe),
    )
    text = report_from_registry(exp, tmp_path)
    failed = text.split("## Failed runs")[1].split("## ")[0]
    assert "| a | web | 5 | 2 | 1 | 1 |" in failed
    assert "not a zero" in failed
    runs = [
        r for r in read_records(registry_path(tmp_path)) if isinstance(r, RunRecord)
    ]
    assert [bool(r.outage) for r in runs][:2] == [True, False]
    # the outage cell has no value; the errored one is a miss (zero) on outcomes
    assert runs[0].metrics == {} and runs[1].metrics.get("correct") == 0.0


def test_a_group_is_accepted_only_when_every_rule_is() -> None:
    exp = _report_experiment()
    text = build_report(exp, [])
    assert "## Verdicts" in text  # an empty registry still reports, as inconclusive

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
