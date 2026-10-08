"""Concurrency and its limits, offline: up to 4 cells for an Ollama ``-cloud``
model, 1 for a local one, and the CLI flag that carries it."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from andamentum.deep_research.bench import cli
from andamentum.deep_research.bench.arms import ArmOutput
from andamentum.deep_research.bench.concurrency import (
    MAX_CLOUD_CONCURRENCY,
    is_cloud_model,
    resolve_concurrency,
    run_bounded,
)
from andamentum.deep_research.bench.experiment import Experiment
from andamentum.deep_research.bench.experiment_report import build_report
from andamentum.deep_research.bench.experiment_run import (
    DataDirs,
    run_experiment,
)
from andamentum.deep_research.bench.registry import (
    RunRecord,
    StartRecord,
    read_records,
    registry_path,
)

from .test_bench_experiment_run import (
    ScriptedArm,
    correct_behaviour,
    services,
    small_experiment,
    write_simpleqa,
)

CLOUD = "ollama:gemma4:31b-cloud"


# -- the rule ---------------------------------------------------------------------------


def test_cloud_models_are_the_ollama_cloud_tags() -> None:
    assert is_cloud_model("ollama:gemma4:31b-cloud")
    assert not is_cloud_model("ollama:gemma4:31b-nvfp4")
    assert not is_cloud_model("openai:gpt-5.4-nano")
    assert not is_cloud_model("gemma4:31b-cloud")  # no provider prefix: not Ollama's


def test_concurrency_is_capped_and_refused_for_local_models() -> None:
    assert MAX_CLOUD_CONCURRENCY == 4
    assert resolve_concurrency(1, models=["ollama:gemma4:31b-nvfp4"]) == 1
    assert resolve_concurrency(4, models=[CLOUD, None]) == 4
    assert resolve_concurrency(4, models=[None]) == 4
    with pytest.raises(ValueError, match="above the cap of 4"):
        resolve_concurrency(5, models=[CLOUD])
    with pytest.raises(ValueError, match="at least 1"):
        resolve_concurrency(0, models=[CLOUD])
    with pytest.raises(ValueError, match="refused for local model.*gemma4:31b-nvfp4"):
        resolve_concurrency(2, models=[CLOUD, "ollama:gemma4:31b-nvfp4"])


async def test_run_bounded_limits_the_jobs_in_flight_and_keeps_job_order() -> None:
    running = 0
    peak = 0

    def job(n: int):  # noqa: ANN202
        async def run() -> int:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01 * (5 - n % 5))
            running -= 1
            return n

        return run

    assert await run_bounded([job(n) for n in range(10)], concurrency=3) == list(
        range(10)
    )
    assert peak == 3
    running = peak = 0
    await run_bounded([job(n) for n in range(4)], concurrency=1)
    assert peak == 1


async def test_a_failing_job_stops_the_ones_after_it() -> None:
    started: list[int] = []

    def job(n: int):  # noqa: ANN202
        async def run() -> int:
            started.append(n)
            await asyncio.sleep(0.001)
            if n == 2:
                raise RuntimeError("down")
            return n

        return run

    with pytest.raises(RuntimeError, match="down"):
        await run_bounded([job(n) for n in range(8)], concurrency=2)
    assert 7 not in started and len(started) < 8


# -- experiment runs -----------------------------------------------------------------------


def cloud_experiment() -> Experiment:
    return small_experiment(
        repeats=2,
        arms=[
            {"name": "base", "system": "research", "model": CLOUD},
            {"name": "other", "system": "research", "model": CLOUD},
        ],
    ).model_copy(update={"grader": "ollama:invented-grader-cloud"})


class OverlapArm(ScriptedArm):
    """Counts how many of its runs are in flight at once."""

    def __init__(self, name: str) -> None:
        super().__init__(name, correct_behaviour)
        self.in_flight = 0
        self.peak = 0

    async def run(self, question: str) -> ArmOutput:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return await super().run(question)
        finally:
            self.in_flight -= 1


async def test_an_experiment_runs_cells_concurrently_and_states_it(
    tmp_path: Path,
) -> None:
    exp = cloud_experiment()
    base, other = OverlapArm("base"), OverlapArm("other")
    arms: dict[str, ScriptedArm] = {"base": base, "other": other}
    dirs = DataDirs(simpleqa=write_simpleqa(tmp_path / "sqa"))
    written = await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=dirs,
        services=services(arms),
        concurrency=4,
    )
    assert len(written) == 16  # 4 items x 2 repeats x 2 arms
    assert all(r.error is None for r in written)
    # the four slots are shared item by item: two cells of each arm at once
    assert base.peak == 2 and other.peak == 2
    records = list(read_records(registry_path(tmp_path / "out")))
    [start] = [r for r in records if isinstance(r, StartRecord)]
    assert start.concurrency == 4
    assert len([r for r in records if isinstance(r, RunRecord)]) == 16
    report = build_report(exp, records)
    assert "Concurrency used: 4 cell(s) at a time" in report


async def test_an_experiment_defaults_to_one_cell_at_a_time(tmp_path: Path) -> None:
    base, other = OverlapArm("base"), OverlapArm("other")
    arms: dict[str, ScriptedArm] = {"base": base, "other": other}
    await run_experiment(
        cloud_experiment(),
        out_dir=tmp_path / "out",
        dirs=DataDirs(simpleqa=write_simpleqa(tmp_path / "sqa")),
        services=services(arms),
    )
    assert base.peak == 1 and other.peak == 1
    [start] = [
        r
        for r in read_records(registry_path(tmp_path / "out"))
        if isinstance(r, StartRecord)
    ]
    assert start.concurrency == 1


async def test_an_experiment_with_a_local_arm_refuses_concurrency(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="refused for local model.*ollama:m"):
        await run_experiment(
            small_experiment(),  # arms use the model "ollama:m"
            out_dir=tmp_path / "out",
            dirs=DataDirs(),
            services=services({}),
            concurrency=2,
        )
    assert not registry_path(tmp_path / "out").exists()  # nothing was started


def test_the_cli_refuses_concurrency_for_a_local_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exp = {
        "name": "local",
        "registered": "2026-10-08",
        "sets": [{"name": "litsearch-web", "n": 2}],
        "arms": [
            {"name": "a", "system": "research", "model": "ollama:gemma4:31b-nvfp4"}
        ],
        "baselines": {"litsearch-web": "a"},
    }
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(exp), encoding="utf-8")
    argv = ["experiment", "run", str(path), "--out", str(tmp_path / "out")]
    assert cli.main([*argv, "--concurrency", "2"]) == 1
    assert "refused for local model" in capsys.readouterr().err
    assert cli.main([*argv, "--concurrency", "5"]) == 1
    assert "above the cap" in capsys.readouterr().err


def test_the_cli_defaults_are_the_bench_settings() -> None:
    config = cli._source_config("http://x")  # noqa: SLF001
    assert config.pdf_extraction == "fast"
    run = cli.build_parser().parse_args(
        ["experiment", "run", "f.yaml", "--out", "o"]
    )
    assert run.concurrency == 1 and run.pdf_extraction == "fast"
