"""The experiment file, the verdict rules and the web health checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from andamentum.deep_research.v2.bench.experiment import (
    ArmSpec,
    Experiment,
    Rule,
    SetSpec,
    load_experiment,
)
from andamentum.deep_research.v2.bench.health import (
    SearxDown,
    check_searxng,
    require_browser,
    require_searxng,
)
from andamentum.deep_research.v2.bench.subsets import select_subset
from andamentum.deep_research.v2.bench.verdicts import (
    aggregate_interval,
    judge_rule,
    paired_values,
)


def small(**changes: object) -> dict[str, object]:
    """A valid experiment mapping, edited by *changes*."""
    data: dict[str, object] = {
        "name": "tiny",
        "registered": "2026-10-06",
        "repeats": 2,
        "grader": "grader-model",
        "sets": [{"name": "simpleqa", "n": 3}],
        "arms": [
            {"name": "base", "system": "legacy", "model": "m"},
            {"name": "other", "system": "v2", "model": "m"},
        ],
        "baselines": {"simpleqa": "base"},
        "rules": [],
    }
    data.update(changes)
    return data


# -- the file --------------------------------------------------------------------


def test_an_arm_spec_has_only_the_final_fields() -> None:
    assert set(ArmSpec.model_fields) == {
        "name",
        "system",
        "model",
        "repeats",
        "sets",
        "solo",
        "outage",
    }
    with pytest.raises(ValueError, match="Extra inputs"):
        ArmSpec.model_validate(
            {"name": "a", "system": "v2", "model": "m", "preset": "hybrid"}
        )
    with pytest.raises(ValueError, match="legacy|v2"):
        ArmSpec.model_validate({"name": "a", "system": "bm25", "model": "m"})


def test_the_sets_are_simpleqa_and_litsearch_web_only() -> None:
    assert SetSpec(name="litsearch-web", n=2).name == "litsearch-web"
    for gone in ("litsearch", "litsearch-local", "web", "web_planted", "synergy"):
        with pytest.raises(ValueError):
            SetSpec.model_validate({"name": gone, "n": 2})


def test_the_experiment_has_no_inputs_judge_or_set_sources() -> None:
    for gone in ("inputs", "judge", "set_sources"):
        assert gone not in Experiment.model_fields
        with pytest.raises(ValueError, match="Extra inputs"):
            Experiment.model_validate(small(**{gone: {}}))
    assert not hasattr(Experiment, "profile_of")


def test_both_systems_run_on_both_sets() -> None:
    exp = Experiment.model_validate(
        small(
            sets=[{"name": "simpleqa", "n": 3}, {"name": "litsearch-web", "n": 2}],
            baselines={"simpleqa": "base", "litsearch-web": "base"},
        )
    )
    assert [a.name for a in exp.arms_on("litsearch-web")] == ["base", "other"]
    restricted = ArmSpec(name="r", system="v2", model="m", sets=["simpleqa"])
    assert restricted.runs_on("simpleqa") and not restricted.runs_on("litsearch-web")


def test_config_hash_changes_with_any_registered_value() -> None:
    first = Experiment.model_validate(small())
    same = Experiment.model_validate(small())
    changed = Experiment.model_validate(small(repeats=3))
    assert first.config_hash() == same.config_hash()
    assert first.config_hash() != changed.config_hash()


def test_the_grader_must_not_be_an_arms_own_model() -> None:
    with pytest.raises(ValueError, match="must not grade itself"):
        Experiment.model_validate(small(grader="m"))
    with pytest.raises(ValueError, match="needs a grader"):
        Experiment.model_validate(small(grader=None))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"baselines": {"simpleqa": "nobody"}}, "not an arm"),
        ({"baselines": {"litsearch-web": "base"}}, "not a set here"),
        (
            {
                "arms": [
                    {"name": "a", "system": "v2", "model": "m"},
                    {"name": "a", "system": "v2", "model": "m"},
                ],
                "baselines": {"simpleqa": "a"},
            },
            "unique",
        ),
        (
            {
                "arms": [{"name": "a", "system": "v2"}],
                "baselines": {"simpleqa": "a"},
            },
            "model",
        ),
        (
            {
                "arms": [
                    {
                        "name": "a",
                        "system": "legacy",
                        "model": "m",
                        "sets": ["litsearch"],
                    },
                ],
                "baselines": {"simpleqa": "a"},
            },
            "litsearch",
        ),
        (
            {
                "rules": [
                    {
                        "id": "r",
                        "question": "q",
                        "set": "simpleqa",
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
                        "set": "simpleqa",
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


def test_the_old_head_to_head_file_is_history_not_a_valid_experiment() -> None:
    old = Path(__file__).parent.parent / "bench" / "experiments" / "head-to-head.yaml"
    with pytest.raises(ValueError, match="Extra inputs"):
        load_experiment(old)


# -- skip ---------------------------------------------------------------------------


def test_skip_is_validated() -> None:
    assert SetSpec(name="simpleqa", n=3, skip=2).skip == 2
    with pytest.raises(ValueError, match="needs n"):
        SetSpec(name="simpleqa", skip=2)
    with pytest.raises(ValueError, match="greater than or equal"):
        SetSpec(name="simpleqa", n=2, skip=-1)


def test_skip_positions_are_the_seeded_order_after_the_first_items() -> None:
    ids = [f"SQ{i:04d}" for i in range(1, 41)]
    order = select_subset(ids, 40, seed=1)
    spec = SetSpec(name="simpleqa", n=5, skip=10, seed=1)
    chosen = select_subset(ids, spec.skip + (spec.n or 0), seed=spec.seed)[spec.skip :]
    assert chosen == order[10:15]


# -- intervals and verdicts ------------------------------------------------------


def rule(**changes: object) -> Rule:
    data: dict[str, object] = {
        "id": "r",
        "question": "q",
        "set": "simpleqa",
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


async def test_a_browser_that_answers_is_healthy_and_one_that_does_not_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = httpx.AsyncClient

    def answering(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"Browser": "HeadlessChrome/1.0"}
                )
            )
        )

    monkeypatch.setattr(httpx, "AsyncClient", answering)
    health = await require_browser("http://127.0.0.1:9222/")
    assert health.ok and health.detail == "HeadlessChrome/1.0"

    def refusing(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        def refuse(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        return real(transport=httpx.MockTransport(refuse))

    monkeypatch.setattr(httpx, "AsyncClient", refusing)
    with pytest.raises(SearxDown, match="no Chrome answers"):
        await require_browser("http://127.0.0.1:9222")


# -- the no_harm guard -------------------------------------------------------------


def guard(margin: float = 1.0, *, higher: bool = True) -> Rule:
    return Rule(
        id="G",
        question="no harm",
        set="simpleqa",
        metric="correct",
        arm="x",
        reference="y",
        test="no_harm",
        margin=margin,
        higher_is_better=higher,
    )


def guard_verdict(diffs: list[float], **kwargs: Any):  # noqa: ANN202
    arm = {f"i{n}": d for n, d in enumerate(diffs)}
    reference = {f"i{n}": 0.0 for n in range(len(diffs))}
    return judge_rule(
        guard(**kwargs), arm=arm, reference=reference, resamples=2000, seed=0
    )


def test_no_harm_is_accepted_when_nothing_is_lost() -> None:
    assert guard_verdict([0.0] * 10).outcome == "accepted"
    assert guard_verdict([1.0] * 10).outcome == "accepted"


def test_no_harm_blocks_a_mean_below_minus_the_margin() -> None:
    just_below = guard_verdict([-5.0, 3.0] * 6 + [-5.1, 3.0])  # mean -1.007
    assert just_below.interval is not None and just_below.interval.estimate < -1.0
    assert just_below.outcome == "not accepted"


def test_no_harm_blocks_an_interval_wholly_below_zero() -> None:
    certain = guard_verdict([-0.01] * 10)
    assert certain.outcome == "not accepted"
    assert "the whole interval is below 0" in certain.statement


def test_no_harm_reads_the_orientation_of_the_metric() -> None:
    assert guard_verdict([2.0] * 8, higher=False).outcome == "not accepted"
    assert guard_verdict([-2.0] * 8, higher=False).outcome == "accepted"
