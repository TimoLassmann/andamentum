"""The experiment file, the verdict rules and the SearxNG health check."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from andamentum.deep_research.v2.bench.experiment import (
    Experiment,
    Rule,
    load_experiment,
)
from andamentum.deep_research.v2.bench.health import (
    SearxDown,
    check_searxng,
    require_searxng,
)
from andamentum.deep_research.v2.bench.verdicts import (
    aggregate_interval,
    judge_rule,
    paired_values,
)

ROUND1 = Path(__file__).parent.parent / "bench" / "experiments" / "round1.yaml"


def small(**changes: object) -> dict[str, object]:
    """A valid experiment mapping, edited by *changes*."""
    data: dict[str, object] = {
        "name": "tiny",
        "registered": "2026-10-06",
        "repeats": 2,
        "sets": [{"name": "web", "n": 3}],
        "arms": [
            {"name": "base", "system": "v2", "preset": "baseline", "model": "m"},
            {"name": "other", "system": "v2", "preset": "baseline", "model": "m"},
        ],
        "baselines": {"web": "base"},
        "rules": [],
    }
    data.update(changes)
    return data


# -- the file --------------------------------------------------------------------


def test_round1_is_a_valid_experiment_with_the_registered_shape() -> None:
    exp = load_experiment(ROUND1)
    assert exp.name == "round1"
    assert {s.name for s in exp.sets} == {"litsearch", "web", "web_planted", "synergy"}
    assert exp.set("litsearch").n == 30
    assert exp.set("web").n == 15
    synergy = exp.set("synergy")
    assert synergy.records == 150 and len(synergy.reviews or []) == 8
    assert exp.repeats == 3
    assert exp.repeats_of(exp.arm("v2-baseline")) == 3
    assert exp.repeats_of(exp.arm("v2-baseline-live")) == 2
    assert exp.arm("legacy").is_frozen is False
    assert exp.arm("v2-baseline").is_frozen is True
    assert exp.arm("v2-baseline-live").is_frozen is False
    # the criteria preset is a later round's variant
    assert all(a.preset != "criteria" for a in exp.arms)


def test_round1_rules_are_the_registered_ones() -> None:
    exp = load_experiment(ROUND1)
    rules = {r.id: r for r in exp.rules}
    recall = rules["Q1-recall"]
    assert (recall.metric, recall.test, recall.margin) == (
        "recall_at_20",
        "noninferior",
        0.05,
    )
    # legacy runs live, so the end-to-end comparison uses the live v2 arm
    assert recall.arm == "v2-baseline-live" and recall.reference == "legacy"
    for rule_id in ("Q1-claim-support", "Q1-unsourced"):
        assert rules[rule_id].arm == "v2-baseline-live"
    assert exp.judge is not None and exp.judge.decision_model == "nimble:9b"
    assert rules["Q2-kept-jaccard"].threshold == 0.8
    assert rules["Q2-answer-agreement"].threshold == 0.9
    assert rules["Q3-wall-time"].threshold == 300
    assert rules["Q3-wall-time"].aggregate == "median"
    # Q2 and Q3 are reported, not gates; Q1 gates
    assert not rules["Q2-kept-jaccard"].gate and not rules["Q3-wall-time"].gate
    assert rules["Q1-recall"].gate


def test_the_judge_is_part_of_the_registration() -> None:
    first = Experiment.model_validate(small())
    judged = Experiment.model_validate(small(judge={"decision_model": "nimble:9b"}))
    assert first.config_hash() != judged.config_hash()
    with pytest.raises(ValueError, match="exactly one"):
        Experiment.model_validate(small(judge={"model": "m", "decision_model": "d"}))


def test_a_rule_that_reads_claim_support_needs_a_named_judge() -> None:
    rule = {
        "id": "T",
        "question": "q",
        "set": "web",
        "metric": "claim_precision",
        "arm": "base",
        "threshold": 0.5,
        "test": "at_least",
    }
    with pytest.raises(ValueError, match="name its judge"):
        Experiment.model_validate(small(rules=[rule]))
    Experiment.model_validate(
        small(rules=[rule], judge={"decision_model": "nimble:9b"})
    )


def test_config_hash_changes_with_any_registered_value() -> None:
    first = Experiment.model_validate(small())
    same = Experiment.model_validate(small())
    changed = Experiment.model_validate(small(repeats=3))
    assert first.config_hash() == same.config_hash()
    assert first.config_hash() != changed.config_hash()


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"baselines": {"web": "nobody"}}, "not an arm"),
        ({"baselines": {"litsearch": "base"}}, "not a set here"),
        (
            {
                "arms": [
                    {"name": "a", "system": "v2", "preset": "baseline", "model": "m"},
                    {"name": "a", "system": "v2", "preset": "baseline", "model": "m"},
                ],
                "baselines": {"web": "a"},
            },
            "unique",
        ),
        (
            {
                "arms": [
                    {"name": "a", "system": "v2", "model": "m"},
                ],
                "baselines": {"web": "a"},
            },
            "preset",
        ),
        (
            {
                "arms": [
                    {
                        "name": "a",
                        "system": "legacy",
                        "model": "m",
                        "sets": ["web_planted"],
                    },
                ],
                "baselines": {"web": "a"},
            },
            "cannot run on",
        ),
        (
            {
                "arms": [
                    {"name": "a", "system": "legacy", "model": "m", "frozen": True},
                ],
                "baselines": {"web": "a"},
            },
            "frozen belongs to v2",
        ),
        (
            {
                "rules": [
                    {
                        "id": "r",
                        "question": "q",
                        "set": "web",
                        "metric": "correct",
                        "arm": "base",
                        "test": "noninferior",
                    }
                ]
            },
            "reference",
        ),
        (
            {
                "rules": [
                    {
                        "id": "r",
                        "question": "q",
                        "set": "web",
                        "metric": "correct",
                        "arm": "base",
                        "test": "at_least",
                    }
                ]
            },
            "threshold",
        ),
        ({"unknown_field": 1}, "Extra inputs"),
    ],
)
def test_a_bad_experiment_is_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Experiment.model_validate(small(**changes))


def test_load_refuses_a_file_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "e.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_experiment(path)


# -- intervals and verdicts ------------------------------------------------------


def rule(**changes: object) -> Rule:
    data: dict[str, object] = {
        "id": "r",
        "question": "q",
        "set": "web",
        "metric": "m",
        "arm": "a",
        "reference": "b",
        "test": "noninferior",
        "margin": 0.05,
    }
    data.update(changes)
    return Rule.model_validate(data)


def items(values: list[float | None]) -> dict[str, float | None]:
    return {f"i{n}": v for n, v in enumerate(values)}


def test_paired_values_uses_items_both_arms_have() -> None:
    arm = {"a": 1.0, "b": None, "c": 0.5, "d": 1.0}
    reference = {"a": 0.5, "b": 1.0, "c": None}
    assert paired_values(arm, reference) == [0.5]


def test_the_interval_is_reproducible_and_brackets_the_estimate() -> None:
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    first = aggregate_interval(values, "mean", resamples=500, seed=3)
    second = aggregate_interval(values, "mean", resamples=500, seed=3)
    assert first == second
    assert first is not None
    assert first.low <= first.estimate <= first.high
    assert aggregate_interval([], "mean", resamples=10, seed=0) is None


def test_the_median_interval_is_around_the_median() -> None:
    interval = aggregate_interval(
        [100.0, 120.0, 130.0, 400.0, 900.0], "median", resamples=500, seed=0
    )
    assert interval is not None
    assert interval.estimate == 130.0
    assert interval.low <= 130.0 <= interval.high


def test_noninferior_is_accepted_when_the_whole_interval_clears_the_margin() -> None:
    arm = items([0.8] * 10)
    reference = items([0.82] * 10)
    verdict = judge_rule(rule(), arm=arm, reference=reference, resamples=200, seed=0)
    assert verdict.outcome == "accepted"
    assert verdict.gate


def test_noninferior_is_not_accepted_when_clearly_worse_than_the_margin() -> None:
    arm = items([0.4] * 10)
    reference = items([0.8] * 10)
    verdict = judge_rule(rule(), arm=arm, reference=reference, resamples=200, seed=0)
    assert verdict.outcome == "not accepted"


def test_noninferior_is_inconclusive_when_the_interval_straddles_the_margin() -> None:
    arm = items([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    reference = items([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    verdict = judge_rule(rule(), arm=arm, reference=reference, resamples=500, seed=0)
    assert verdict.outcome == "inconclusive"


def test_lower_is_better_flips_the_sign() -> None:
    worse = items([0.5] * 10)  # more unsourced claims than the reference
    reference = items([0.1] * 10)
    verdict = judge_rule(
        rule(higher_is_better=False),
        arm=worse,
        reference=reference,
        resamples=200,
        seed=0,
    )
    assert verdict.outcome == "not accepted"
    better = judge_rule(
        rule(higher_is_better=False),
        arm=reference,
        reference=worse,
        resamples=200,
        seed=0,
    )
    assert better.outcome == "accepted"


def test_superior_needs_the_interval_above_the_margin() -> None:
    arm = items([0.9] * 10)
    reference = items([0.5] * 10)
    assert (
        judge_rule(
            rule(test="superior", margin=0.1),
            arm=arm,
            reference=reference,
            resamples=200,
            seed=0,
        ).outcome
        == "accepted"
    )
    assert (
        judge_rule(
            rule(test="superior", margin=0.5),
            arm=arm,
            reference=reference,
            resamples=200,
            seed=0,
        ).outcome
        == "not accepted"
    )


def test_threshold_rules_read_the_arms_own_values() -> None:
    stable = items([0.95] * 8)
    unstable = items([0.5] * 8)
    least = rule(test="at_least", reference=None, threshold=0.8, margin=0.0)
    assert (
        judge_rule(least, arm=stable, reference=None, resamples=200, seed=0).outcome
        == "accepted"
    )
    assert (
        judge_rule(least, arm=unstable, reference=None, resamples=200, seed=0).outcome
        == "not accepted"
    )
    most = rule(
        test="at_most", reference=None, threshold=300, margin=0.0, aggregate="median"
    )
    assert (
        judge_rule(
            most,
            arm=items([100.0, 120.0, 150.0]),
            reference=None,
            resamples=200,
            seed=0,
        ).outcome
        == "accepted"
    )
    assert (
        judge_rule(
            most,
            arm=items([900.0, 1000.0, 700.0]),
            reference=None,
            resamples=200,
            seed=0,
        ).outcome
        == "not accepted"
    )


def test_no_data_is_inconclusive_never_accepted() -> None:
    verdict = judge_rule(
        rule(),
        arm=items([None, None]),
        reference=items([0.5, 0.5]),
        resamples=50,
        seed=0,
    )
    assert verdict.outcome == "inconclusive"
    assert verdict.interval is None


def test_a_paired_rule_without_a_reference_is_an_error() -> None:
    with pytest.raises(ValueError, match="reference"):
        judge_rule(rule(), arm=items([1.0]), reference=None, resamples=10, seed=0)


# -- the SearxNG health check ----------------------------------------------------


def client_for(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


async def test_a_searxng_that_answers_is_healthy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json={"results": [{"url": "https://a.example/x"}]})

    async with client_for(handler) as client:
        health = await check_searxng("http://searx.test", client=client)
    assert health.ok and health.n_results == 1


async def test_no_results_with_dead_engines_is_an_outage() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [],
                "unresponsive_engines": [
                    ["duckduckgo", "timeout"],
                    ["brave", "blocked"],
                ],
            },
        )

    async with client_for(handler) as client:
        health = await check_searxng("http://searx.test", client=client)
        assert not health.ok
        assert "duckduckgo (timeout)" in health.detail
        with pytest.raises(SearxDown, match="duckduckgo"):
            await require_searxng("http://searx.test", client=client)


async def test_an_unreachable_or_broken_searxng_is_not_healthy() -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with client_for(refuse) as client:
        health = await check_searxng("http://searx.test", client=client)
    assert not health.ok and "ConnectError" in health.detail

    async with client_for(lambda _r: httpx.Response(503)) as client:
        assert (
            "HTTP 503" in (await check_searxng("http://s.test", client=client)).detail
        )
    async with client_for(lambda _r: httpx.Response(200, content=b"<html>")) as client:
        assert (
            "not JSON" in (await check_searxng("http://s.test", client=client)).detail
        )


async def test_web_work_without_a_url_is_refused() -> None:
    with pytest.raises(SearxDown, match="--searxng-url"):
        await require_searxng(None)
