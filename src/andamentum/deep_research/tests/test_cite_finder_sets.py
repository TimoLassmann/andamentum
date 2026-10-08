"""The citation finder's dev and test sets are fixed, sized and disjoint.

The LitSearch query ids are ``LS0000`` to ``LS0596`` and a subset depends only
on the ids and the seed (``subsets.select_subset``), so the check needs no data.
"""

from __future__ import annotations

from pathlib import Path

from ..bench.experiment import Experiment, SetSpec, load_experiment
from ..bench.litsearch_local import excluded_positions
from ..bench.subsets import select_subset

EXPERIMENTS = Path(__file__).parents[1] / "bench" / "experiments"
LITSEARCH_IDS = [f"LS{i:04d}" for i in range(597)]
# Used before: {n: 17, skip: 150, seed: 0}; the earlier LitSearch rounds used the
# seeded positions before it (the highest any registry holds is 168, 1-based).
USED_BEFORE = SetSpec(name="litsearch-web", n=17, skip=150, seed=0)
EARLIER_ROUNDS_UPTO = 168


def ids_of(spec: SetSpec) -> set[str]:
    assert spec.n is not None
    return set(
        select_subset(LITSEARCH_IDS, spec.skip + spec.n, seed=spec.seed)[spec.skip :]
    )


def litsearch_set(name: str) -> SetSpec:
    exp: Experiment = load_experiment(EXPERIMENTS / f"{name}.yaml")
    return exp.set("litsearch-web")


def test_the_dev_set_has_about_30_items_and_the_test_set_at_least_60() -> None:
    dev, test = litsearch_set("cite-finder-dev"), litsearch_set("cite-finder-test")
    assert len(ids_of(dev)) == 30
    assert len(ids_of(test)) >= 60


def test_dev_test_and_the_items_used_before_are_disjoint() -> None:
    dev = ids_of(litsearch_set("cite-finder-dev"))
    test = ids_of(litsearch_set("cite-finder-test"))
    before = ids_of(USED_BEFORE)
    earlier = set(excluded_positions(upto=EARLIER_ROUNDS_UPTO, seed=0))
    assert before <= earlier
    assert not dev & test
    assert not dev & earlier
    assert not test & earlier


def test_both_files_use_the_same_finder_arms() -> None:
    dev = load_experiment(EXPERIMENTS / "cite-finder-dev.yaml")
    test = load_experiment(EXPERIMENTS / "cite-finder-test.yaml")
    for name in ("F0", "F1"):
        assert dev.arm(name) == test.arm(name)
    assert dev.arm("F0").follow_citations is False
    assert dev.arm("F1").follow_citations is True
    assert {a.name: a.system for a in test.arms} == {
        "B1": "research",
        "B2": "s2_search",
        "F0": "finder",
        "F1": "finder",
    }


def test_the_test_gate_is_registered_for_each_finder_arm_against_each_baseline() -> (
    None
):
    test = load_experiment(EXPERIMENTS / "cite-finder-test.yaml")
    rules = {(r.arm, r.reference): r for r in test.rules}
    assert set(rules) == {(f, b) for f in ("F0", "F1") for b in ("B1", "B2")}
    for (arm, _), rule in rules.items():
        assert rule.metric == "recall_at_20"
        assert rule.test == "exceeds_by" and rule.margin == 0.10
        assert rule.group == f"gate-{arm}"
    assert {d.id: d.requires for d in test.decisions} == {
        "finder-F0": ["gate-F0"],
        "finder-F1": ["gate-F1"],
    }
