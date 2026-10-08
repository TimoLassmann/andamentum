"""The two registered experiment files of the final comparison: the dev set the
new engine is iterated on, and the test set it is run on once."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.deep_research.v2.bench import experiment_run
from andamentum.deep_research.v2.bench.experiment import Experiment, load_experiment
from andamentum.deep_research.v2.bench.subsets import select_subset

EXPERIMENTS = Path(__file__).resolve().parents[1] / "bench" / "experiments"
DEV = EXPERIMENTS / "final-dev.yaml"
TEST = EXPERIMENTS / "final-test.yaml"
SIMPLEQA_ROWS = 4326  # rows of the SimpleQA test set
LITSEARCH_QUERIES = 597
RUN_COMMAND = "uv run python -m andamentum.deep_research.v2.bench"


@pytest.fixture(scope="module")
def dev() -> Experiment:
    return load_experiment(DEV)


@pytest.fixture(scope="module")
def test() -> Experiment:
    return load_experiment(TEST)


def positions(experiment: Experiment, set_name: str, universe: list[str]) -> list[str]:
    spec = experiment.set(set_name)
    assert spec.n is not None
    return select_subset(universe, spec.skip + spec.n, seed=spec.seed)[spec.skip :]


SIMPLEQA_IDS = [f"SQ{i:04d}" for i in range(1, SIMPLEQA_ROWS + 1)]
LITSEARCH_IDS = [f"LS{i:04d}" for i in range(LITSEARCH_QUERIES)]


def test_both_files_are_valid_and_registered_today(
    dev: Experiment, test: Experiment
) -> None:
    assert (dev.name, test.name) == ("final-dev", "final-test")
    assert dev.registered == test.registered == "2026-10-08"
    assert dev.repeats == test.repeats == 1


def test_the_dev_set_is_twenty_simpleqa_questions_at_positions_64_to_83(
    dev: Experiment,
) -> None:
    [spec] = dev.sets
    assert (spec.name, spec.n, spec.skip, spec.seed) == ("simpleqa", 20, 63, 1)
    assert len(positions(dev, "simpleqa", SIMPLEQA_IDS)) == 20
    assert dev.baselines == {"simpleqa": "legacy"}
    assert [(a.name, a.system) for a in dev.arms] == [
        ("legacy", "legacy"),
        ("new-engine", "v2"),
    ]


def test_the_test_sets_are_the_registered_positions(test: Experiment) -> None:
    sqa, lit = test.set("simpleqa"), test.set("litsearch-web")
    assert (sqa.n, sqa.skip, sqa.seed) == (60, 3, 1)  # positions 4-63
    assert (lit.n, lit.skip, lit.seed) == (17, 150, 0)  # positions 151-167
    assert test.baselines == {"simpleqa": "legacy", "litsearch-web": "legacy"}
    assert [(a.name, a.system) for a in test.arms] == [
        ("legacy", "legacy"),
        ("new-engine", "v2"),
    ]


def test_the_dev_positions_share_no_question_with_the_test_positions(
    dev: Experiment, test: Experiment
) -> None:
    dev_ids = set(positions(dev, "simpleqa", SIMPLEQA_IDS))
    test_ids = set(positions(test, "simpleqa", SIMPLEQA_IDS))
    assert len(dev_ids) == 20 and len(test_ids) == 60
    assert not dev_ids & test_ids
    # the smoke runs (positions 1-3) belong to neither set
    smoke = set(select_subset(SIMPLEQA_IDS, 3, seed=1))
    assert not smoke & (dev_ids | test_ids)
    assert len(positions(test, "litsearch-web", LITSEARCH_IDS)) == 17


def test_the_grader_is_not_a_model_of_either_arm(
    dev: Experiment, test: Experiment
) -> None:
    for experiment in (dev, test):
        assert experiment.grader == "ollama:gpt-oss:120b-cloud"
        assert {a.model for a in experiment.arms} == {"ollama:gemma4:31b-cloud"}
        assert experiment.grader not in {a.model for a in experiment.arms}


def test_the_dev_file_decides_nothing(dev: Experiment) -> None:
    assert dev.rules and not any(r.gate for r in dev.rules)
    assert [r.metric for r in dev.rules] == ["correct"]
    assert not dev.decisions


def test_the_test_file_registers_q1_q2_s1_s2_b1_b2_and_decision_d1(
    test: Experiment,
) -> None:
    rules = {r.id.split("-")[0]: r for r in test.rules}
    assert list(rules) == ["Q1", "Q2", "S1", "S2", "B1", "B2"]
    q1, q2, s1, s2 = rules["Q1"], rules["Q2"], rules["S1"], rules["S2"]
    assert (q1.set, q1.metric, q1.test, q1.margin, q1.group) == (
        "simpleqa",
        "correct",
        "noninferior",
        0.05,
        "Q",
    )
    assert (q2.set, q2.metric, q2.test, q2.margin, q2.group) == (
        "litsearch-web",
        "recall_at_20",
        "noninferior",
        0.05,
        "Q",
    )
    assert (s1.set, s1.metric, s1.test, s1.margin, s1.group) == (
        "simpleqa",
        "wall_s",
        "noninferior",
        15,
        "S",
    )
    assert s1.higher_is_better is False and s1.gate
    # the LitSearch wall is reported, not a gate, and in no group
    assert (s2.set, s2.metric, s2.gate, s2.group) == (
        "litsearch-web",
        "wall_s",
        False,
        None,
    )
    for better in (rules["B1"], rules["B2"]):
        assert better.test == "superior" and not better.gate
    for rule in test.rules:
        assert (rule.arm, rule.reference) == ("new-engine", "legacy")
    [d1] = test.decisions
    assert d1.id == "D1-replace" and d1.requires == ["Q", "S"]
    assert set(test.rule_groups()) == {"Q", "S"}


def test_the_notes_state_what_the_comparison_compares(test: Experiment) -> None:
    notes = " ".join(test.notes)
    assert "0.176" in notes  # the earlier hybrid recall@20 it is reported against
    assert "no scholarly" in notes or "web only" in notes
    assert "08:00" in notes and "429" in notes
    assert "concurrency 1" in notes


@pytest.mark.parametrize("path", [DEV, TEST], ids=["dev", "test"])
def test_the_header_gives_the_run_command_with_placeholders(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    header = text.split("\nname:")[0]
    assert RUN_COMMAND in header
    assert "experiment run" in header and "experiment report" in header
    assert "experiment losses" in header
    assert "--browser-cdp-url http://127.0.0.1:9222" in header
    assert "$OUT" in header and "$SQA" in header
    assert "/Users/" not in text and "/home/" not in text


def test_the_test_header_names_the_data_folder_and_one_cell_at_a_time() -> None:
    header = TEST.read_text(encoding="utf-8").split("\nname:")[0]
    assert "$DATA" in header and "--litsearch-local-dir" in header
    assert "--concurrency 1" in header
    assert "--concurrency" not in DEV.read_text(encoding="utf-8").split("\nname:")[0]


def test_the_dev_header_says_to_iterate_here_only() -> None:
    header = DEV.read_text(encoding="utf-8").split("\nname:")[0]
    assert "Iterate here only" in header or "iterate here only" in header.lower()


def test_both_files_are_runnable_by_the_experiment_machinery(
    dev: Experiment, test: Experiment
) -> None:
    for experiment in (dev, test):
        for spec in experiment.sets:
            assert experiment_run.set_mode(spec.name) in ("web", "scholarly")
        assert experiment.config_hash() != ""
    assert dev.config_hash() != test.config_hash()
