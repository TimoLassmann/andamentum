"""The registered regression experiment: SimpleQA, one arm, no rules."""

from __future__ import annotations

from pathlib import Path

from andamentum.deep_research.bench.experiment import load_experiment
from andamentum.deep_research.bench.subsets import select_subset

DEV_CHECK = (
    Path(__file__).resolve().parents[1] / "bench" / "experiments" / "dev-check.yaml"
)
SIMPLEQA_ROWS = 4326  # rows of the SimpleQA test set


def test_dev_check_is_twenty_simpleqa_questions_at_positions_64_to_83() -> None:
    exp = load_experiment(DEV_CHECK)
    [spec] = exp.sets
    assert (exp.name, spec.name, spec.n, spec.skip, spec.seed) == (
        "dev-check",
        "simpleqa",
        20,
        63,
        1,
    )
    assert spec.n is not None
    ids = [f"SQ{i:04d}" for i in range(1, SIMPLEQA_ROWS + 1)]
    chosen = select_subset(ids, spec.skip + spec.n, seed=spec.seed)[spec.skip :]
    assert len(chosen) == 20


def test_dev_check_has_one_engine_arm_and_no_rules() -> None:
    exp = load_experiment(DEV_CHECK)
    assert [(a.name, a.system) for a in exp.arms] == [("v2", "v2")]
    assert exp.rules == [] and exp.baselines == {"simpleqa": "v2"}
    assert exp.grader is not None and exp.grader != exp.arms[0].model
