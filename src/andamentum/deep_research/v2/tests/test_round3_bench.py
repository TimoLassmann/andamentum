"""Round 3, the bench side: the experiment file's inputs profiles, ``skip``,
the ``no_harm`` guard, ``--arm``, per-profile snapshots, the keyless guard and
the shared decision cache of the cells. Offline, with stand-in arms."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from andamentum.deep_research.v2.bench import cli, experiment_run, freeze
from andamentum.deep_research.v2.bench.arms import ArmOutput
from andamentum.deep_research.v2.bench.experiment import (
    ArmSpec,
    Experiment,
    InputsSpec,
    Rule,
    SetSpec,
    load_experiment,
)
from andamentum.deep_research.v2.bench.experiment_report import Cells, judge_rules
from andamentum.deep_research.v2.bench.experiment_run import (
    DataDirs,
    Item,
    Services,
    build_answer_arm,
    failed_search_share,
    load_items,
    profile_sources,
    run_experiment,
    snapshot_root,
)
from andamentum.deep_research.v2.bench.freeze import (
    FreezeItem,
    InputsSetup,
    SnapshotMissing,
    SnapshotSearchOnly,
    freeze_items_of,
    snapshot_dir,
)
from andamentum.deep_research.v2.bench.litsearch import (
    GOLD_FILE,
    QUERIES_FILE,
    GoldRecord,
    LitSearchItem,
    LitSearchQuery,
    load_litsearch,
)
from andamentum.deep_research.v2.bench.metrics import Cost
from andamentum.deep_research.v2.bench.registry import (
    RunRecord,
    StartRecord,
    read_records,
    registry_path,
)
from andamentum.deep_research.v2.bench.subsets import select_subset
from andamentum.deep_research.v2.bench.verdicts import judge_rule
from andamentum.deep_research.v2.bench.web_set import (
    WEB_QUESTIONS,
    AnswerSpec,
    WebQuestion,
)
from andamentum.deep_research.v2.frozen import FrozenInputs
from andamentum.deep_research.v2.sources import SourceConfig
from andamentum.deep_research.v2.store import ResearchStore
from andamentum.deep_research.v2.types import SubQuestion
from andamentum.references import Work

from .engine_world import QUESTION, SEARXNG, World, world_decisions, world_llm
from .test_bench_experiment_run import (
    ScriptedArm,
    make_result,
    output,
    services,
    work,
)

EXPERIMENTS = Path(cli.__file__).parent / "experiments"
SOURCES = SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",))
WORLD_QUESTION = WebQuestion(
    question_id="W01",
    question=QUESTION,
    answer=AnswerSpec(all_of=[["latency"]]),
    source_url="https://example.org/world",
    expected="latency",
)


def stable_behaviour(question: str, _call: int) -> ArmOutput:
    """An answer to whatever question the world asks."""
    return output(question, "the cache cuts latency", [work(1), work(2)])


@pytest.fixture(autouse=True)
def _world_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    from andamentum.core import fetch_gate, url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


# -- the registered files ----------------------------------------------------------


def test_the_files_of_rounds_one_and_two_keep_their_hashes() -> None:
    """A field added to the experiment model is left out of the hash while it holds
    its default, so the recorded registries still belong to their files."""
    assert (
        load_experiment(EXPERIMENTS / "round1.yaml").config_hash()
        == "fce99e836b90a3ed43049d1f1aba64321d3732a9ccd1db4964757814987e4a3a"
    )
    assert (
        load_experiment(EXPERIMENTS / "round2.yaml").config_hash()
        == "96edd8dbc5736edc769447d7ef4ca2384313dd9d43e9740485b42bdfc2dde698"
    )


def test_round_three_is_registered_as_specified() -> None:
    exp = load_experiment(EXPERIMENTS / "round3.yaml")
    # pinned: after the commit that registers it no number in the file changes
    assert (
        exp.config_hash()
        == "3cfebb01b6c35baa2f713989264ce35173886dba3c3c5c92ccfba34d9d3e0cd6"
    )
    assert exp.registered == "2026-10-06" and exp.keyless and exp.repeats == 1
    assert [a.name for a in exp.arms] == [
        "v2-baseline",
        "v2-arxiv",
        "v2-combined",
        "v2-combined-solo",
    ]
    assert [(a.preset, a.solo) for a in exp.arms] == [
        ("baseline", None),
        ("recall", None),
        ("recall", None),
        ("recall", 3),
    ]
    spec = exp.set("litsearch")
    assert (spec.n, spec.skip, spec.seed) == (32, 28, 0)
    assert [(s.name, s.n) for s in exp.sets[1:]] == [("web", 15), ("web_planted", 15)]
    assert exp.inputs["r3-base"].plan == "live"
    assert exp.inputs["r3-arxiv"].plan == "r3-base"
    assert exp.inputs["r3-combined"].patient == ["semantic_scholar"]
    assert exp.inputs["r3-combined"].sources.expand_with == ["europepmc"]
    assert exp.inputs["r3-combined"].sources.arxiv_query == "any_term"
    assert exp.inputs["web-r1"].existing
    assert all(
        p.prefetch_top_k == 8 for n, p in exp.inputs.items() if n.startswith("r3-")
    )
    rules = {r.id: r for r in exp.rules}
    assert set(rules) == {
        "P1-recall-at-20",
        "L1-recall-at-20",
        "L2-recall-at-20",
        "H1-inputs-health",
        "G1-claim-support",
        "G2-answer-correct",
        "G3-planted-adoption",
        "T1-clean-wall",
    }
    assert (rules["G1-claim-support"].test, rules["G1-claim-support"].margin) == (
        "no_harm",
        0.05,
    )
    assert rules["H1-inputs-health"].metric == "inputs_failed_search_share"
    assert [d.id for d in exp.decisions] == [
        "arxiv-and-merit",
        "semantic-scholar",
        "combined-as-unit",
    ]
    assert exp.baselines == {"litsearch": "v2-baseline", "web": "v2-baseline"}
    # the arms name their profile for every set they run
    combined = exp.arm("v2-combined")
    assert combined.inputs == {
        "litsearch": "r3-combined",
        "web": "web-r1",
        "web_planted": "web-r1",
    }
    assert exp.profile_of(combined, "web_planted") == "web-r1"


def test_a_profile_hash_names_what_the_profile_does() -> None:
    exp = load_experiment(EXPERIMENTS / "round3.yaml")
    hashes = {n: p.profile_hash() for n, p in exp.inputs.items()}
    assert len(set(hashes.values())) == 4 and all(len(h) == 64 for h in hashes.values())
    more = InputsSpec.model_validate(
        {**exp.inputs["r3-base"].model_dump(mode="json"), "prefetch_top_k": 9}
    )
    assert more.profile_hash() != hashes["r3-base"]


# -- the file's checks -------------------------------------------------------------


def inputs_experiment(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "r3-small",
        "registered": "2026-10-06",
        "keyless": True,
        "inputs": {
            "base": {
                "sources": {
                    "scholarly": ["europepmc", "arxiv", "semantic_scholar"],
                    "arxiv_query": "all_terms",
                    "expand_with": ["europepmc"],
                    "results_per_query": 20,
                },
                "prefetch_top_k": 8,
            },
            "wide": {
                "sources": {
                    "scholarly": ["europepmc", "arxiv", "semantic_scholar"],
                    "arxiv_query": "any_term",
                },
                "plan": "base",
                "prefetch_top_k": 8,
                "patient": ["semantic_scholar"],
            },
            "old": {"existing": True},
        },
        "sets": [{"name": "web", "n": 2, "seed": 0}],
        "arms": [
            {
                "name": "a",
                "system": "v2",
                "preset": "baseline",
                "model": "m",
                "inputs": {"web": "old"},
            },
            {
                "name": "b",
                "system": "v2",
                "preset": "recall",
                "model": "m",
                "inputs": {"web": "old"},
            },
        ],
        "baselines": {"web": "a"},
    }
    base.update(changes)
    return base


def test_a_file_with_profiles_validates() -> None:
    exp = Experiment.model_validate(inputs_experiment())
    assert exp.profile_of(exp.arm("a"), "web") == "old"
    updates = exp.inputs["base"].sources.updates()
    assert updates == {
        "scholarly": ("europepmc", "arxiv", "semantic_scholar"),
        "arxiv_query": "all_terms",
        "expand_with": ("europepmc",),
        "results_per_query": 20,
    }
    assert exp.inputs["wide"].sources.updates() == {
        "scholarly": ("europepmc", "arxiv", "semantic_scholar"),
        "arxiv_query": "any_term",
    }
    assert exp.inputs["old"].sources.updates() == {}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["inputs"]["wide"].update(plan="nope"), "is not a profile"),
        (lambda d: d["inputs"]["wide"].update(plan="old"), "existing profile"),
        (
            lambda d: (
                d["inputs"]["base"].update(plan="wide"),
                d["inputs"]["wide"].update(plan="base"),
            ),
            "circular",
        ),
        (
            lambda d: d["inputs"].update(old={"existing": True, "prefetch_top_k": 3}),
            "no other field",
        ),
        (
            lambda d: d["arms"][0]["inputs"].update(web="missing"),
            "is not an inputs profile",
        ),
        (lambda d: d["arms"][0].pop("inputs"), "name its inputs profile"),
        (lambda d: d["arms"][0].update(frozen=False), "belong to frozen"),
        (
            lambda d: d["inputs"]["wide"].update(patient=["openalex"]),
            "does not search",
        ),
        (lambda d: d["inputs"]["base"].update(surprise=1), "surprise"),
        (
            lambda d: d["inputs"]["base"]["sources"].update(arxiv_query="some"),
            "arxiv_query",
        ),
        (lambda d: d["inputs"].update(**{"a/b": {"existing": True}}), "folder name"),
    ],
)
def test_a_file_with_a_bad_profile_is_refused(mutate: Any, message: str) -> None:
    data = inputs_experiment()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        Experiment.model_validate(data)


def test_inputs_for_a_set_the_arm_does_not_run_are_refused() -> None:
    data = inputs_experiment()
    data["arms"][0]["inputs"]["litsearch"] = "base"
    with pytest.raises(ValueError, match="does not run"):
        Experiment.model_validate(data)


def test_a_file_without_profiles_takes_no_arm_inputs() -> None:
    data = inputs_experiment(inputs={})
    with pytest.raises(ValueError, match="declares no inputs profiles"):
        Experiment.model_validate(data)
    for arm in data["arms"]:
        arm.pop("inputs")
    exp = Experiment.model_validate(data)  # round 1 and 2 files look like this
    assert exp.profile_of(exp.arm("a"), "web") is None


def test_skip_is_validated() -> None:
    assert SetSpec(name="litsearch", n=3, skip=2).skip == 2
    with pytest.raises(ValueError, match="belongs to litsearch and web"):
        SetSpec(name="web_planted", n=3, skip=1)
    with pytest.raises(ValueError, match="needs n"):
        SetSpec(name="litsearch", skip=2)
    with pytest.raises(ValueError, match="greater than or equal"):
        SetSpec(name="litsearch", n=2, skip=-1)


# -- skip ---------------------------------------------------------------------------


def litsearch_dir(root: Path, *, n: int = 10, unresolved: tuple[int, ...] = ()) -> Path:
    queries = [
        LitSearchQuery(
            query_id=f"LS{i:04d}",
            query=f"Which paper studies invented topic {i}?",
            query_set="inline_acl",
            specificity=1,
            quality=2,
            corpus_ids=[9000 + i],
        )
        for i in range(n)
    ]
    gold = [
        GoldRecord(
            corpus_id=9000 + i,
            resolved=i not in unresolved,
            title=f"An invented paper about topic {i} in kilns"
            if i not in unresolved
            else "",
            year=2021,
            paper_id=f"{9000 + i:040x}",
            doi=f"10.5555/gold.{i}",
        )
        for i in range(n)
    ]
    folder = root / "litsearch"
    folder.mkdir(parents=True)
    (root / QUERIES_FILE).write_text(
        "".join(q.model_dump_json() + "\n" for q in queries), encoding="utf-8"
    )
    (root / GOLD_FILE).write_text(
        "".join(g.model_dump_json() + "\n" for g in gold), encoding="utf-8"
    )
    return root


def test_skip_takes_the_seeded_order_after_the_first_items(tmp_path: Path) -> None:
    root = litsearch_dir(tmp_path)
    ids = [f"LS{i:04d}" for i in range(10)]
    order = select_subset(ids, 10, seed=0)
    first = [i.query.query_id for i in load_litsearch(root, n=4, seed=0).items]
    second = [i.query.query_id for i in load_litsearch(root, n=4, seed=0, skip=4).items]
    assert first == order[:4] and second == order[4:8]
    whole = [i.query.query_id for i in load_litsearch(root, n=8, seed=0).items]
    assert whole == first + second  # the larger subset is the two parts together
    assert load_litsearch(root, n=4, seed=0, skip=4).n_subset == 4
    assert load_litsearch(root, n=4, seed=0, skip=8).n_subset == 2  # only two are left


def test_items_whose_gold_does_not_resolve_drop_out_of_a_skipped_subset(
    tmp_path: Path,
) -> None:
    root = litsearch_dir(tmp_path, unresolved=(3,))
    order = select_subset([f"LS{i:04d}" for i in range(10)], 10, seed=0)
    data = load_litsearch(root, n=4, seed=0, skip=4)
    expected = [i for i in order[4:8] if i != "LS0003"]
    assert [i.query.query_id for i in data.items] == expected
    assert data.n_unresolved_gold == (1 if "LS0003" in order[4:8] else 0)
    assert data.n_dropped_queries == data.n_unresolved_gold  # the count is reported


def test_a_missing_gold_names_the_download_for_the_whole_prefix(tmp_path: Path) -> None:
    root = litsearch_dir(tmp_path, n=10)
    (root / GOLD_FILE).write_text("", encoding="utf-8")
    from andamentum.deep_research.v2.bench.download import DatasetMissing

    with pytest.raises(DatasetMissing, match="--n 8 "):
        load_litsearch(root, n=4, seed=0, skip=4)


def test_load_items_and_freeze_items_honour_skip(tmp_path: Path) -> None:
    root = litsearch_dir(tmp_path)
    spec = SetSpec(name="litsearch", n=3, skip=3, seed=0)
    order = select_subset([f"LS{i:04d}" for i in range(10)], 6, seed=0)
    items = load_items(spec, DataDirs(litsearch=root))
    assert [i.item_id for i in items] == order[3:6]
    frozen = freeze.freeze_items("litsearch", data_dir=root, n=3, seed=0, skip=3)
    assert [i.item_id for i in frozen] == order[3:6]
    # no skip: what it always was
    assert [
        i.item_id for i in freeze.freeze_items("litsearch", data_dir=root, n=3, seed=0)
    ] == order[:3]


def test_the_web_set_skips_too() -> None:
    first = load_items(SetSpec(name="web", n=4, seed=0), DataDirs())
    second = load_items(SetSpec(name="web", n=4, seed=0, skip=4), DataDirs())
    both = load_items(SetSpec(name="web", n=8, seed=0), DataDirs())
    assert not {i.item_id for i in first} & {i.item_id for i in second}
    assert {i.item_id for i in first} | {i.item_id for i in second} == {
        i.item_id for i in both
    }
    assert len(WEB_QUESTIONS) >= 8


# -- the no_harm guard -------------------------------------------------------------


def guard(margin: float = 1.0, *, higher: bool = True) -> Rule:
    return Rule(
        id="G",
        question="no harm",
        set="web",
        metric="correct",
        arm="x",
        reference="y",
        test="no_harm",
        margin=margin,
        higher_is_better=higher,
    )


def values(
    diffs: list[float],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    return (
        {f"i{n}": d for n, d in enumerate(diffs)},
        {f"i{n}": 0.0 for n in range(len(diffs))},
    )


def verdict(diffs: list[float], **kwargs: Any):  # noqa: ANN202
    arm, reference = values(diffs)
    return judge_rule(
        guard(**kwargs), arm=arm, reference=reference, resamples=2000, seed=0
    )


def test_no_harm_is_accepted_when_nothing_is_lost() -> None:
    assert verdict([0.0] * 10).outcome == "accepted"
    assert verdict([1.0] * 10).outcome == "accepted"
    assert (
        verdict([-5.0, 3.0] * 7).outcome == "accepted"
    )  # mean exactly -1, interval over 0


def test_no_harm_blocks_a_mean_below_minus_the_margin() -> None:
    at_the_margin = verdict([-5.0, 3.0] * 7)  # mean -1.0: not below -1.0
    assert (
        at_the_margin.interval is not None and at_the_margin.interval.estimate == -1.0
    )
    assert at_the_margin.outcome == "accepted"
    just_below = verdict([-5.0, 3.0] * 6 + [-5.1, 3.0])  # mean -1.007
    assert just_below.interval is not None and just_below.interval.estimate < -1.0
    assert just_below.interval.high > 0  # the interval alone would not block it
    assert just_below.outcome == "not accepted"
    assert "the mean is below -1" in just_below.statement


def test_no_harm_blocks_an_interval_wholly_below_zero() -> None:
    small_but_certain = verdict([-0.01] * 10)  # far from the margin, but always worse
    assert small_but_certain.outcome == "not accepted"
    assert "the whole interval is below 0" in small_but_certain.statement
    zero_high = verdict([0.0] * 5 + [-0.0] * 5)
    assert (
        zero_high.outcome == "accepted"
    )  # an interval ending at 0 is not wholly below it


def test_no_harm_reads_the_orientation_of_the_metric() -> None:
    # lower is better: a rise is the harm
    assert verdict([2.0] * 8, higher=False).outcome == "not accepted"
    assert verdict([-2.0] * 8, higher=False).outcome == "accepted"


def test_no_harm_has_no_inconclusive_unless_there_is_nothing_to_compare() -> None:
    outcomes = {
        verdict(list(d)).outcome
        for d in ([0.0] * 6, [-5.0, 3.0] * 7, [-0.01] * 4, [2.0] * 3, [-3.0, 5.0] * 3)
    }
    assert outcomes <= {"accepted", "not accepted"}
    none = judge_rule(
        guard(), arm={"i": None}, reference={"i": 0.0}, resamples=100, seed=0
    )
    assert none.outcome == "inconclusive" and "no item" in none.statement


def test_the_rule_model_checks_a_guard() -> None:
    with pytest.raises(ValueError, match="needs a reference"):
        Rule(id="G", question="q", set="web", metric="correct", arm="x", test="no_harm")
    with pytest.raises(ValueError, match="takes a margin, not a threshold"):
        Rule(
            id="G",
            question="q",
            set="web",
            metric="correct",
            arm="x",
            reference="y",
            test="no_harm",
            threshold=0.1,
        )


def test_a_guard_is_not_made_inconclusive_by_an_item_without_a_value() -> None:
    """A gating rule that reads fewer items than ran is inconclusive; a guard
    has no inconclusive, and says how many items it read."""
    exp = Experiment.model_validate(
        {
            "name": "g",
            "registered": "2026-10-06",
            "sets": [{"name": "web", "n": 3}],
            "arms": [
                {
                    "name": "x",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "m",
                    "frozen": False,
                },
                {
                    "name": "y",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "m",
                    "frozen": False,
                },
            ],
            "baselines": {"web": "y"},
            "rules": [
                {
                    "id": "G",
                    "question": "no harm",
                    "set": "web",
                    "metric": "correct",
                    "arm": "x",
                    "reference": "y",
                    "test": "no_harm",
                    "margin": 0.05,
                }
            ],
            "resamples": 200,
        }
    )

    def cell(arm: str, item: str, value: float | None) -> RunRecord:
        return RunRecord(
            experiment="g",
            experiment_hash="h",
            arm=arm,
            set="web",
            item_id=item,
            repeat=1,
            started_at=0.0,
            metrics={"correct": value},
        )

    runs = [cell("y", f"W{i}", 1.0) for i in range(3)]
    runs += [cell("x", "W0", 1.0), cell("x", "W1", 1.0), cell("x", "W2", None)]
    [result] = judge_rules(exp, Cells(runs))
    assert result.outcome == "accepted" and "2 items" in result.statement


# -- running: --arm, profiles, snapshots, the shared cache ---------------------------


def web_experiment(*, with_inputs: bool = True, keyless: bool = False) -> Experiment:
    data: dict[str, Any] = {
        "name": "prof",
        "registered": "2026-10-06",
        "repeats": 1,
        "resamples": 200,
        "keyless": keyless,
        "sets": [{"name": "web", "n": 1, "seed": 0}],
        "arms": [
            {"name": "base", "system": "v2", "preset": "baseline", "model": "m"},
            {"name": "recall", "system": "v2", "preset": "recall", "model": "m"},
            {
                "name": "recall-solo",
                "system": "v2",
                "preset": "recall",
                "model": "m",
                "solo": 1,
            },
        ],
        "baselines": {"web": "base"},
    }
    if with_inputs:
        data["inputs"] = {
            "plain": {"sources": {"results_per_query": 7}, "prefetch_top_k": 4},
            "taken": {"existing": True},
        }
        data["arms"][0]["inputs"] = {"web": "plain"}
        data["arms"][1]["inputs"] = {"web": "taken"}
        data["arms"][2]["inputs"] = {"web": "plain"}
    return Experiment.model_validate(data)


async def make_snapshot(root: Path, *, setup: InputsSetup | None = None) -> Path:
    """Freeze W01 under *root*; a folder named ``plain`` is taken as the
    ``plain`` profile of ``web_experiment`` (its name and hash recorded)."""
    if setup is None:
        setup = (
            InputsSetup(
                profile="plain",
                profile_hash=web_experiment().inputs["plain"].profile_hash(),
            )
            if root.name == "plain"
            else InputsSetup()
        )
    world = World()
    import httpx

    await freeze_items_of(
        "web",
        [FreezeItem("W01", QUESTION)],
        snapshots=root,
        model=world_llm().model,
        decision_model=world_decisions()[0],
        decision_label="stub",
        source_config=SOURCES,
        progress=lambda _line: None,
        prefetch_top_k=None,
        health_client=httpx.AsyncClient(transport=httpx.MockTransport(world.handler)),
        client_factory=world.client,
        clock=lambda: 1000.0,
        setup=setup,
    )
    return snapshot_dir(root, "web", "W01")


class Recorder:
    """Stands in for ``build_answer_arm`` and records how each arm is built."""

    def __init__(self) -> None:
        self.built: list[tuple[str, str, SourceConfig, ResearchStore | None]] = []
        self.runs: dict[str, list[dict[str, Any]]] = {}

    def __call__(
        self,
        spec: ArmSpec,
        mode: str,
        services: Services,
        *,
        sources: SourceConfig | None = None,
    ) -> ScriptedArm:
        self.built.append(
            (
                spec.name,
                mode,
                sources or services.source_config,
                services.decision_cache,
            )
        )
        arm = ScriptedArm(spec.name, stable_behaviour)
        original = arm.run

        async def run(
            question: str, *, frozen: object = None, store: object = None
        ) -> ArmOutput:
            self.runs.setdefault(spec.name, []).append(
                {"frozen": frozen, "store": store}
            )
            return await original(question, frozen=frozen, store=store)

        arm.run = run  # type: ignore[method-assign]
        return arm


@pytest.fixture
def web_world(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_run, "WEB_QUESTIONS", (WORLD_QUESTION,))
    monkeypatch.setattr(freeze, "WEB_QUESTIONS", (WORLD_QUESTION,))


async def test_only_arms_runs_those_arms_in_file_order_with_solo_last(
    tmp_path: Path, web_world: None
) -> None:
    exp = web_experiment(with_inputs=False)
    exp = Experiment.model_validate(
        {
            **exp.model_dump(mode="json"),
            "arms": [
                {**a, "frozen": False} for a in exp.model_dump(mode="json")["arms"]
            ],
        }
    )
    order: list[str] = []

    def make(name: str):  # noqa: ANN202
        def behave(question: str, _call: int) -> ArmOutput:
            order.append(name)
            return stable_behaviour(question, 0)

        return behave

    arms = {n: ScriptedArm(n, make(n)) for n in ("base", "recall", "recall-solo")}
    written = await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=services(arms),
        only_arms=["recall-solo", "base"],  # the order of the command line is not used
    )
    assert order == ["base", "recall-solo"]
    assert [r.arm for r in written] == ["base", "recall-solo"]
    # the registry resumes: a later run of the other arm adds only its cells
    later = await run_experiment(
        exp,
        out_dir=tmp_path,
        dirs=DataDirs(),
        services=services(arms),
        only_arms=["recall"],
    )
    assert [r.arm for r in later] == ["recall"]
    records = list(read_records(registry_path(tmp_path)))
    assert sum(isinstance(r, StartRecord) for r in records) == 2
    assert sorted(r.arm for r in records if isinstance(r, RunRecord)) == [
        "base",
        "recall",
        "recall-solo",
    ]


async def test_an_unknown_arm_is_refused(tmp_path: Path, web_world: None) -> None:
    with pytest.raises(ValueError, match="not arms of this experiment: nope"):
        await run_experiment(
            web_experiment(with_inputs=False),
            out_dir=tmp_path,
            dirs=DataDirs(),
            services=Services(),
            only_arms=["nope"],
        )


async def test_snapshots_are_checked_only_for_the_chosen_arms(
    tmp_path: Path, web_world: None
) -> None:
    exp = web_experiment(with_inputs=False)
    # frozen arms, no snapshots at all: refused for the arms asked for
    with pytest.raises(SnapshotMissing, match="web/W01"):
        await run_experiment(
            exp,
            out_dir=tmp_path / "a",
            dirs=DataDirs(snapshots=tmp_path / "none"),
            services=Services(
                answer_arm=lambda s, m, d: ScriptedArm(s.name, stable_behaviour)
            ),
            only_arms=["base"],
        )
    live = Experiment.model_validate(
        {
            **exp.model_dump(mode="json"),
            "arms": [
                {**exp.model_dump(mode="json")["arms"][0], "frozen": False},
                exp.model_dump(mode="json")["arms"][1],
            ],
        }
    )
    arms = {"base": ScriptedArm("base", stable_behaviour)}
    written = await run_experiment(
        live,
        out_dir=tmp_path / "b",
        dirs=DataDirs(),  # the frozen arm is not chosen, so no snapshot is needed
        services=services(arms),
        only_arms=["base"],
    )
    assert [r.arm for r in written] == ["base"]


async def test_a_frozen_arm_runs_on_its_profiles_snapshots(
    tmp_path: Path, web_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = tmp_path / "snap"
    plain = await make_snapshot(snapshots / "plain")
    await make_snapshot(snapshots / "taken")
    assert plain.parts[-4:] == ("snap", "plain", "web", "W01")
    recorder = Recorder()
    monkeypatch.setattr(experiment_run, "build_answer_arm", recorder)
    exp = web_experiment()
    written = await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=DataDirs(snapshots=snapshots),
        services=Services(
            source_config=SourceConfig(searxng_url=SEARXNG), decision_cache=None
        ),
        only_arms=["base", "recall"],
    )
    assert [r.error for r in written] == [None, None]
    for arm_name in ("base", "recall"):
        [run] = recorder.runs[arm_name]
        assert isinstance(run["frozen"], FrozenInputs)
    base_store = recorder.runs["base"][0]["store"]
    # the working copy is made from the profile's own snapshot (and is gone by now)
    assert base_store.parent.parts[-4:-1] == ("base", "r1", "web")


async def test_the_working_copy_comes_from_the_profiles_folder(
    tmp_path: Path, web_world: None
) -> None:
    snapshots = tmp_path / "snap"
    await make_snapshot(snapshots / "plain")
    item = Item("web", "W01", QUESTION, web=WORLD_QUESTION)
    target = experiment_run.prepare_working_copy(
        item,
        out_dir=tmp_path / "out",
        snapshots=snapshot_root(snapshots, "plain"),
        arm="base",
        repeat=1,
    )
    assert target.exists()
    with pytest.raises(SnapshotMissing):
        experiment_run.prepare_working_copy(
            item,
            out_dir=tmp_path / "out",
            snapshots=snapshot_root(snapshots, "other"),
            arm="base",
            repeat=1,
        )
    assert snapshot_root(snapshots, None) == snapshots


async def test_a_missing_profile_snapshot_is_named_with_its_profile(
    tmp_path: Path, web_world: None
) -> None:
    snapshots = tmp_path / "snap"
    await make_snapshot(snapshots / "plain")  # "taken" has none
    with pytest.raises(SnapshotMissing, match="taken/web/W01"):
        await run_experiment(
            web_experiment(),
            out_dir=tmp_path / "out",
            dirs=DataDirs(snapshots=snapshots),
            services=Services(
                answer_arm=lambda s, m, d: ScriptedArm(s.name, stable_behaviour)
            ),
            only_arms=["base", "recall"],
        )


def test_a_frozen_arms_sources_are_the_runs_own_updated_with_its_profile() -> None:
    exp = web_experiment()
    own = SourceConfig(searxng_url=SEARXNG)
    services_ = Services(source_config=own)
    plain = profile_sources(exp, exp.arm("base"), "web", services_)
    assert plain.results_per_query == 7 and plain.searxng_url == SEARXNG
    assert profile_sources(exp, exp.arm("recall"), "web", services_) == own  # existing
    no_profiles = web_experiment(with_inputs=False)
    assert (
        profile_sources(no_profiles, no_profiles.arm("base"), "web", services_) == own
    )
    # the arm is built with them
    arm = build_answer_arm(exp.arm("base"), "web", services_, sources=plain)
    assert arm.sources == plain  # type: ignore[attr-defined]


async def test_the_profile_names_and_hashes_and_keyless_reach_the_registry(
    tmp_path: Path, web_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = tmp_path / "snap"
    await make_snapshot(
        snapshots / "plain",
        setup=InputsSetup(
            profile="plain",
            profile_hash=web_experiment().inputs["plain"].profile_hash(),
            keyless=True,
        ),
    )
    await make_snapshot(snapshots / "taken")
    monkeypatch.setattr(experiment_run, "build_answer_arm", Recorder())
    exp = web_experiment(keyless=True)
    await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=DataDirs(snapshots=snapshots),
        services=Services(keyless=True),
        only_arms=["base"],
    )
    [start] = [
        r
        for r in read_records(registry_path(tmp_path / "out"))
        if isinstance(r, StartRecord)
    ]
    assert start.keyless is True
    assert start.inputs == {n: p.profile_hash() for n, p in exp.inputs.items()}
    assert {a.name: a.inputs for a in start.arms}["base"] == {"web": "plain"}


async def test_a_search_only_snapshot_is_refused(
    tmp_path: Path, web_world: None
) -> None:
    exp = web_experiment(with_inputs=False)
    snapshots = tmp_path / "snap"
    await make_snapshot(snapshots, setup=InputsSetup(search_only=True))
    meta = (snapshot_dir(snapshots, "web", "W01") / "meta.json").read_text()
    assert '"search_only": true' in meta
    with pytest.raises(SnapshotSearchOnly, match="search-only"):
        await run_experiment(
            exp,
            out_dir=tmp_path / "out",
            dirs=DataDirs(snapshots=snapshots),
            services=Services(
                answer_arm=lambda s, m, d: ScriptedArm(s.name, stable_behaviour)
            ),
            only_arms=["base"],
        )
    assert isinstance(
        SnapshotSearchOnly("x"), SnapshotMissing
    )  # the CLI reports it as one


def test_the_failed_search_share_reads_the_snapshots_accounting() -> None:
    def snapshot(**fields: Any) -> FrozenInputs:
        return FrozenInputs(
            question=QUESTION,
            mode="scholarly",
            captured_at=1.0,
            subquestions=[SubQuestion(id="Q1", text="one")],
            **fields,
        )

    assert failed_search_share(snapshot()) is None  # a snapshot that predates it
    assert failed_search_share(
        snapshot(
            searches={"europepmc": 6, "arxiv": 6, "semantic_scholar": 8},
            search_failures={"semantic_scholar": 2},
        )
    ) == pytest.approx(0.1)
    assert failed_search_share(snapshot(searches={"arxiv": 5})) == 0.0


async def test_a_litsearch_cell_records_the_share_and_the_decision_answers(
    tmp_path: Path,
) -> None:
    gold = Work(
        title="An invented paper about kilns of Apollo", doi="10.5555/gold.1", year=2021
    )
    litsearch_item = LitSearchItem(
        query=LitSearchQuery(
            query_id="LS0001",
            query="Which paper studies kilns?",
            query_set="inline_acl",
            specificity=1,
            quality=2,
            corpus_ids=[1],
        ),
        gold=[gold],
    )
    item = Item(
        "litsearch", "LS0001", litsearch_item.query.query, litsearch=litsearch_item
    )
    frozen = FrozenInputs(
        question=item.question,
        mode="scholarly",
        captured_at=1.0,
        subquestions=[SubQuestion(id="Q1", text="kilns")],
        searches={"europepmc": 3, "semantic_scholar": 7},
        search_failures={"semantic_scholar": 2},
    )
    kept = [Work(work_id="w1", title=gold.title, doi=gold.doi, year=2021), work(2)]
    result = make_result(item.question, kept)
    result.stats.decision_calls = 30
    result.stats.decision_cache_hits = 12
    out = ArmOutput(answer_text="a", works=kept, cost=Cost(wall_s=1.0), result=result)

    class Arm(ScriptedArm):
        returns_pages = False

    arm = Arm("x", lambda q, n: out)
    spec = ArmSpec(name="x", system="v2", preset="recall", model="m")
    metrics, _output, _outcome = await experiment_run._answer_cell(  # noqa: SLF001
        arm, spec, item, services=Services(), frozen=frozen, store=None
    )
    assert metrics["inputs_failed_search_share"] == pytest.approx(0.2)
    assert metrics["decision_answers"] == 42.0
    assert metrics["recall_at_20"] == 1.0
    # a live run has no snapshot, so no share
    live, _o, _p = await experiment_run._answer_cell(  # noqa: SLF001
        arm, spec, item, services=Services(), frozen=None, store=None
    )
    assert "inputs_failed_search_share" not in live


# -- the keyless guard ---------------------------------------------------------------


class Environment(dict):  # type: ignore[type-arg]
    """An environment that remembers which names were looked up."""

    looked_up: list[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.looked_up = []

    def get(self, key: str, default: Any = None) -> Any:  # noqa: D102
        self.looked_up.append(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self.looked_up.append(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.looked_up.append(str(key))
        return super().__contains__(key)


def test_a_keyless_source_configuration_never_looks_a_key_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = Environment(
        SEMANTIC_SCHOLAR_API_KEY="s2-secret",
        OPENALEX_API_KEY="oa-secret",
        OPENALEX_EMAIL="someone@example.org",
    )
    monkeypatch.setattr(os, "environ", environment)
    keyless = cli._source_config(SEARXNG, keyless=True)  # noqa: SLF001
    assert keyless.semantic_scholar_api_key is None and keyless.openalex_api_key is None
    assert "SEMANTIC_SCHOLAR_API_KEY" not in environment.looked_up
    assert "OPENALEX_API_KEY" not in environment.looked_up
    assert "s2-secret" not in keyless.model_dump_json()
    # without the guard the keys are read, as they always were
    keyed = cli._source_config(SEARXNG)  # noqa: SLF001
    assert keyed.semantic_scholar_api_key == "s2-secret"
    assert keyed.openalex_api_key == "oa-secret"


async def test_a_keyless_experiment_refuses_a_configuration_that_holds_a_key(
    tmp_path: Path, web_world: None
) -> None:
    exp = web_experiment(with_inputs=False, keyless=True)
    with pytest.raises(ValueError, match="keyless but the source configuration holds"):
        await run_experiment(
            exp,
            out_dir=tmp_path,
            dirs=DataDirs(),
            services=Services(
                source_config=SourceConfig(semantic_scholar_api_key="k"),
                answer_arm=lambda s, m, d: ScriptedArm(s.name, stable_behaviour),
            ),
        )


async def test_the_download_is_keyless_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def fake_download(root: Path, **kwargs: Any) -> Any:
        seen.update(kwargs)
        from andamentum.deep_research.v2.bench.litsearch import LitSearchDownload

        return LitSearchDownload(n_queries_total=1, n_subset=1, n_gold_fetched=1)

    monkeypatch.setattr(cli, "download_litsearch", fake_download)
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2-secret")
    args = cli.build_parser().parse_args(
        ["download", "litsearch", "--dir", str(tmp_path), "--n", "60", "--keyless"]
    )
    await cli._cmd_download(args)  # noqa: SLF001
    assert seen["s2_api_key"] is None
    args = cli.build_parser().parse_args(
        ["download", "litsearch", "--dir", str(tmp_path)]
    )
    await cli._cmd_download(args)  # noqa: SLF001
    assert seen["s2_api_key"] == "s2-secret"


# -- the cells' shared decision cache -------------------------------------------------


def test_the_shared_decision_cache_goes_to_every_cell_but_the_solo_ones() -> None:
    shared = ResearchStore.in_memory()
    exp = web_experiment(with_inputs=False)
    services_ = Services(decision_cache=shared)
    arm = build_answer_arm(exp.arm("base"), "web", services_)
    solo = build_answer_arm(exp.arm("recall-solo"), "web", services_)
    assert arm.shared_decisions is shared  # type: ignore[attr-defined]
    assert solo.shared_decisions is None  # type: ignore[attr-defined]
    plain = build_answer_arm(exp.arm("base"), "web", Services())
    assert plain.shared_decisions is None  # type: ignore[attr-defined]


def test_the_cli_knows_the_new_flags() -> None:
    parser = cli.build_parser()
    run = parser.parse_args(
        [
            "experiment",
            "run",
            "x.yaml",
            "--out-dir",
            "o",
            "--arm",
            "a",
            "--arm",
            "b",
            "--decision-cache",
            "d.db",
        ]
    )
    assert run.arm == ["a", "b"] and run.decision_cache == Path("d.db")
    report = parser.parse_args(
        [
            "experiment",
            "report",
            "x.yaml",
            "--out-dir",
            "o",
            "--snapshots",
            "s",
            "--litsearch-dir",
            "l",
        ]
    )
    assert report.snapshots == Path("s") and report.litsearch_dir == Path("l")
    pool = parser.parse_args(
        [
            "pool",
            "--snapshots",
            "s",
            "--inputs",
            "p",
            "--dir",
            "d",
            "--n",
            "5",
            "--skip",
            "2",
        ]
    )
    assert (pool.inputs, pool.n, pool.skip) == ("p", 5, 2)
    default = parser.parse_args(["experiment", "run", "x.yaml", "--out-dir", "o"])
    assert default.arm is None and default.decision_cache is None


# -- an experiment without profiles reports as it did ----------------------------------


def test_a_report_of_a_file_without_profiles_is_what_the_earlier_code_wrote() -> None:
    """The registry and the report are what the code before round 3 wrote for a
    small experiment over invented answers (old record schema, no screening, no
    decision-sharing fields). Reading the registry and reporting it gives the
    same text byte for byte, so the new columns, sections and fields change
    nothing for a file that does not use them."""
    from andamentum.deep_research.v2.bench.experiment_report import build_report
    from andamentum.deep_research.v2.bench.registry import read_records

    from .test_bench_experiment_run import small_experiment

    folder = Path(__file__).parent / "fixtures"
    records = list(read_records(folder / "tiny_experiment_registry.jsonl"))
    assert any(isinstance(r, StartRecord) for r in records) and len(records) == 17
    expected = (folder / "tiny_experiment_report.md").read_text(encoding="utf-8")
    assert build_report(small_experiment(repeats=2), records) == expected
    # the old records have none of the new fields, and they still load
    starts = [r for r in records if isinstance(r, StartRecord)]
    assert starts[0].keyless is False and starts[0].inputs == {}


async def run_plain(snapshots: Path, tmp_path: Path, exp: Experiment) -> None:
    await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=DataDirs(snapshots=snapshots),
        services=Services(
            source_config=SourceConfig(searxng_url=SEARXNG), decision_cache=None
        ),
        only_arms=["base"],
    )


async def test_a_snapshot_of_another_profile_or_definition_is_refused(
    tmp_path: Path, web_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment_run, "build_answer_arm", Recorder())
    snapshots = tmp_path / "snap"
    await make_snapshot(
        snapshots / "plain",
        setup=InputsSetup(profile="plain", profile_hash="0" * 64),
    )
    with pytest.raises(experiment_run.SnapshotMismatch, match="another definition"):
        await run_plain(snapshots, tmp_path, web_experiment())
    other = tmp_path / "snap2"
    await make_snapshot(
        other / "plain", setup=InputsSetup(profile="elsewhere", profile_hash="x")
    )
    with pytest.raises(experiment_run.SnapshotMismatch, match="taken for profile"):
        await run_plain(other, tmp_path, web_experiment())


async def test_a_keyless_experiment_refuses_a_snapshot_taken_with_keys(
    tmp_path: Path, web_world: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment_run, "build_answer_arm", Recorder())
    snapshots = tmp_path / "snap"
    await make_snapshot(snapshots / "plain")  # recorded keyless: False
    with pytest.raises(experiment_run.SnapshotMismatch, match="keyless"):
        await run_plain(snapshots, tmp_path, web_experiment(keyless=True))


async def test_a_profile_that_takes_its_plan_refuses_other_queries(
    tmp_path: Path,
) -> None:
    exp = web_experiment()
    exp.inputs["plain"] = exp.inputs["plain"].model_copy(update={"plan": "taken"})
    exp.inputs["taken"] = exp.inputs["taken"].model_copy(update={"existing": False})
    base = tmp_path / "snap"
    folder = base / "plain" / "web" / "W01"
    plan = base / "taken" / "web" / "W01"
    for place, meta in (
        (
            folder,
            {
                "inputs": {
                    "profile": "plain",
                    "hash": exp.inputs["plain"].profile_hash(),
                },
                "plan_from": {"digest": "aaa"},
            },
        ),
        (
            plan,
            {
                "inputs": {
                    "profile": "taken",
                    "hash": exp.inputs["taken"].profile_hash(),
                },
                "digest": "bbb",
            },
        ),
    ):
        place.mkdir(parents=True)
        (place / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    item = Item("web", "W01", QUESTION, web=WORLD_QUESTION)
    problems = experiment_run.mismatched_snapshots(exp, {"web": [item]}, base, ["base"])
    assert problems == [
        "plain/web/W01: its queries are not those of the taken snapshot"
    ]
    (folder / "meta.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "profile": "plain",
                    "hash": exp.inputs["plain"].profile_hash(),
                },
                "plan_from": {"digest": "bbb"},
            }
        ),
        encoding="utf-8",
    )
    assert (
        experiment_run.mismatched_snapshots(exp, {"web": [item]}, base, ["base"]) == []
    )
