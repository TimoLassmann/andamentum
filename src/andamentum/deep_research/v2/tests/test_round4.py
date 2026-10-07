"""Round 4: the litsearch-local set, the BM25 reference arm, the registered file.

Offline over the invented corpus of ``local_world``; the v2 arms are scripted
stand-ins (the engine itself is covered by ``test_local_corpus_engine``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ..bench import cli as bench_cli
from ..bench.arms import ArmOutput, Bm25Arm, V2Arm
from ..bench.experiment import ArmSpec, Experiment, SetSpec, load_experiment
from ..bench.experiment_report import build_report
from ..bench.experiment_run import (
    DataDirs,
    Services,
    build_answer_arm,
    load_items,
    profile_sources,
    run_experiment,
    set_mode,
)
from ..bench.litsearch import LitSearchQuery
from ..bench.litsearch_local import excluded_positions
from ..bench.metrics import Cost
from ..bench.pool import HIT, KEPT_LOW, NOT_IN_POOL, SCREENED_OUT, classify_gold
from ..bench.registry import read_records, registry_path
from ..bench.subsets import select_subset
from ..sources import SourceConfig
from ..types import ResearchResult, RunStats, ScreeningEntry
from .llm_stub import decisions
from .local_world import SALTLOG_D, write_corpus

ROUND4 = Path(__file__).resolve().parents[1] / "bench" / "experiments" / "round4.yaml"


# -- the registered file ------------------------------------------------------------


def test_the_registered_file_loads_and_says_what_was_planned() -> None:
    exp = load_experiment(ROUND4)
    assert exp.name == "round4" and exp.registered == "2026-10-07" and exp.keyless
    assert exp.repeats == 1 and not exp.inputs and exp.judge is None
    (items,) = exp.sets
    assert (items.name, items.n, items.skip, items.seed) == (
        "litsearch-local",
        80,
        60,
        0,
    )
    arms = {a.name: a for a in exp.arms}
    assert list(arms) == ["bm25-question", "v2-baseline", "v2-recall", "v2-fast"]
    assert arms["bm25-question"].system == "bm25" and arms["bm25-question"].top_k == 100
    for name, preset in (
        ("v2-baseline", "baseline"),
        ("v2-recall", "recall"),
        ("v2-fast", "fast"),
    ):
        arm = arms[name]
        assert (arm.system, arm.preset, arm.frozen) == ("v2", preset, False)
        assert arm.model == "ollama:gemma4:31b-cloud"
        assert arm.decision_model == "nimble:9b"
    sources = exp.sources_for("litsearch-local")
    assert sources is not None
    assert sources.updates() == {
        "scholarly": ("local_corpus",),
        "expand_with": ("local_corpus",),
        "results_per_query": 20,
    }
    assert exp.baselines == {"litsearch-local": "v2-baseline"}


def test_the_three_rules_are_the_registered_ones() -> None:
    rules = {r.id: r for r in load_experiment(ROUND4).rules}
    assert list(rules) == [
        "P1-recall-at-20",
        "P2-baseline-against-bm25",
        "P3-fast-not-worse",
    ]
    p1, p2, p3 = rules.values()
    assert (p1.arm, p1.reference, p1.test, p1.margin, p1.gate) == (
        "v2-recall",
        "v2-baseline",
        "superior",
        0.0,
        True,
    )
    assert (p2.arm, p2.reference, p2.test, p2.gate) == (
        "v2-baseline",
        "bm25-question",
        "superior",
        False,
    )
    assert (p3.arm, p3.reference, p3.test, p3.margin, p3.gate) == (
        "v2-fast",
        "v2-baseline",
        "noninferior",
        0.05,
        True,
    )
    assert {r.metric for r in rules.values()} == {"recall_at_20"}
    assert {r.set for r in rules.values()} == {"litsearch-local"}


def test_the_file_lists_exactly_the_queries_of_rounds_one_to_three() -> None:
    """The excluded queries are positions 1-60 of the seed-0 order of all 597, and
    the registered set (skip 60) starts after them."""
    exp = load_experiment(ROUND4)
    note = next(n for n in exp.notes if "Excluded" in n)
    listed = re.findall(r"LS\d{4}", note)
    assert listed == excluded_positions(upto=60)
    assert len(set(listed)) == 60
    spec = exp.set("litsearch-local")
    ids = [f"LS{i:04d}" for i in range(597)]
    chosen = select_subset(ids, spec.skip + (spec.n or 0), seed=spec.seed)[spec.skip :]
    assert len(chosen) == 80 and not set(chosen) & set(listed)
    # the two queries round 3 began with are the last of the first thirty
    assert listed[28:30] == ["LS0345", "LS0400"]


def test_the_file_is_the_pre_registration() -> None:
    """Any edit changes this hash: a changed file is a new experiment."""
    assert load_experiment(ROUND4).config_hash() == ROUND4_HASH


ROUND4_HASH = "1cc59b435d0cea68ed48d85099d5188f9a5b0b7473d628d392c4fb7defbd187c"


# -- the model of the file ----------------------------------------------------------


def arm(**fields: Any) -> dict[str, Any]:
    return {"name": "a", **fields}


def test_a_bm25_arm_names_its_top_k_and_calls_no_model() -> None:
    ok = ArmSpec.model_validate(arm(system="bm25", top_k=50))
    assert ok.runs_on("litsearch-local") and not ok.runs_on("litsearch")
    with pytest.raises(ValidationError, match="names its top_k"):
        ArmSpec.model_validate(arm(system="bm25"))
    for extra in ({"model": "m"}, {"preset": "baseline"}, {"decision_model": "d"}):
        with pytest.raises(ValidationError, match="calls no model"):
            ArmSpec.model_validate(arm(system="bm25", top_k=5, **extra))
    with pytest.raises(ValidationError, match="top_k belongs to bm25"):
        ArmSpec.model_validate(arm(system="v2", preset="baseline", model="m", top_k=5))
    with pytest.raises(ValidationError, match="cannot run on"):
        ArmSpec.model_validate(arm(system="bm25", top_k=5, sets=["web"]))
    with pytest.raises(ValidationError, match="solo timing"):
        ArmSpec.model_validate(arm(system="bm25", top_k=5, solo=2))


def test_skip_and_sources_per_set_belong_to_the_local_set() -> None:
    assert SetSpec(name="litsearch-local", n=3, skip=2).skip == 2
    with pytest.raises(ValueError, match="skip belongs to litsearch and web"):
        SetSpec(name="synergy", skip=1)
    base = {
        "name": "x",
        "registered": "2026-10-07",
        "sets": [{"name": "litsearch-local", "n": 3}],
        "arms": [{"name": "b", "system": "bm25", "top_k": 5}],
        "baselines": {"litsearch-local": "b"},
    }
    ok = Experiment.model_validate(
        {**base, "set_sources": {"litsearch-local": {"scholarly": ["local_corpus"]}}}
    )
    assert (
        ok.sources_for("litsearch-local") is not None and ok.sources_for("web") is None
    )
    with pytest.raises(ValidationError, match="not a set here"):
        Experiment.model_validate(
            {**base, "set_sources": {"web": {"scholarly": ["arxiv"]}}}
        )
    plain = Experiment.model_validate(base)
    assert plain.config_hash() != ok.config_hash()


# -- the arms -----------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "corpus")


async def test_the_bm25_arm_returns_the_ranked_records_and_costs_no_calls(
    corpus: Path,
) -> None:
    arm = Bm25Arm(corpus_dir=corpus, top_k=4)
    out = await arm.run(
        "What do the studies say about Saltlog logging overhead per request?"
    )
    assert out.reference and out.result is None and out.stop_reason == "bm25"
    assert [w.local_id for w in out.works][:2] == [f"local:{SALTLOG_D}", "local:1005"]
    assert len(out.works) == 4
    assert out.cost.llm_calls == 0 and out.cost.decision_calls == 0
    assert out.cost.input_tokens == 0 and out.cost.wall_s >= 0
    with pytest.raises(ValueError, match="takes no snapshot"):
        await arm.run("q", frozen=object())  # type: ignore[arg-type]


def test_the_arms_are_built_for_the_local_corpus(corpus: Path) -> None:
    exp = load_experiment(ROUND4)
    services = Services(
        decisions={"nimble:9b": decisions(lambda _state, _question: 0.5)[0]},
        source_config=SourceConfig(local_corpus=corpus),
    )
    bm25 = build_answer_arm(
        exp.arm("bm25-question"),
        "scholarly",
        services,
        sources=profile_sources(
            exp, exp.arm("bm25-question"), "litsearch-local", services
        ),
    )
    assert isinstance(bm25, Bm25Arm) and bm25.top_k == 100
    spec = exp.arm("v2-recall")
    sources = profile_sources(exp, spec, "litsearch-local", services)
    assert sources.scholarly == ("local_corpus",) and sources.expand_with == (
        "local_corpus",
    )
    assert sources.results_per_query == 20 and sources.local_corpus == corpus
    assert not sources.openalex_api_key and not sources.semantic_scholar_api_key
    v2 = build_answer_arm(spec, "scholarly", services, sources=sources)
    assert isinstance(v2, V2Arm) and v2.config.queue == "source_rank"
    assert v2.sources.scholarly == ("local_corpus",)
    with pytest.raises(ValueError, match="needs --local-corpus"):
        build_answer_arm(exp.arm("bm25-question"), "scholarly", Services())


def test_the_local_set_is_searched_in_scholarly_mode() -> None:
    assert set_mode("litsearch-local") == "scholarly"
    assert set_mode("litsearch") == "scholarly" and set_mode("web") == "web"


def test_a_live_local_experiment_needs_no_searxng() -> None:
    exp = load_experiment(ROUND4)
    assert not bench_cli._live_web(exp)  # type: ignore[attr-defined]


# -- running an experiment on the local set -----------------------------------------


def dataset(root: Path) -> Path:
    """The invented local LitSearch folder: the corpus, its index, ten queries."""
    write_corpus(root)
    gold = [
        ("Which study measured how the Apollo cache changes latency?", [1001]),
        ("Which paper measured saltlog logging overhead?", [1004, 1005]),
        ("Is there work on pruning roses?", [2001]),
        ("What is the overhead of verbose logging per request?", [1005]),
        ("Which paper followed up on the Apollo cache result?", [1006]),
        ("Which note covers cache sizing?", [2002]),
        ("Which paper replicates the Apollo caching layer across services?", [1002]),
        ("Which paper claims the Apollo cache removes all latency?", [1003]),
        ("Which study has no gold record in the corpus?", [424242]),
        ("Which paper studies logging in production traces?", [1005]),
    ]
    queries = [
        LitSearchQuery(
            query_id=f"LS{i:04d}",
            query=text,
            query_set="synthetic",
            specificity=0,
            quality=2,
            corpus_ids=ids,
        )
        for i, (text, ids) in enumerate(gold)
    ]
    (root / "queries.jsonl").write_text(
        "".join(q.model_dump_json() + "\n" for q in queries), encoding="utf-8"
    )
    return root


class ScriptedV2:
    """A v2 stand-in: keeps what a function of the question says, with a result
    whose screening record places every other gold paper."""

    returns_pages = False

    def __init__(
        self, name: str, keep_ids: dict[str, list[str]], *, decision_calls: int = 7
    ) -> None:
        self.name = name
        self.keep_ids = keep_ids
        self.decision_calls = decision_calls

    async def run(
        self, question: str, *, frozen: object = None, store: object = None
    ) -> ArmOutput:
        from andamentum.references import Work

        kept = [
            Work(local_id=f"local:{i}", title=f"Record {i}")
            for i in self.keep_ids.get(question, [])
        ]
        entries = [
            ScreeningEntry(
                work_id=f"w{n}",
                subquestion_id="S1",
                arrival=n,
                local_id=w.local_id,
                p=0.9,
                kept=True,
                rank=n + 1,
                title=w.title,
            )
            for n, w in enumerate(kept)
        ]
        return ArmOutput(
            answer_text="",
            works=kept,
            cost=Cost(
                wall_s=2.0,
                llm_calls=5,
                decision_calls=7,
                input_tokens=100,
                output_tokens=20,
            ),
            result=ResearchResult(
                run_id="r",
                question=question,
                mode="scholarly",
                screening=entries,
                stats=RunStats(decision_calls=self.decision_calls),
            ),
        )


def local_experiment() -> Experiment:
    return Experiment.model_validate(
        {
            "name": "local-tiny",
            "registered": "2026-10-07",
            "resamples": 200,
            "set_sources": {"litsearch-local": {"scholarly": ["local_corpus"]}},
            "sets": [{"name": "litsearch-local", "n": 6, "skip": 2, "seed": 0}],
            "arms": [
                {"name": "bm25", "system": "bm25", "top_k": 3},
                {
                    "name": "good",
                    "system": "v2",
                    "preset": "recall",
                    "model": "m",
                    "frozen": False,
                },
                {
                    "name": "plain",
                    "system": "v2",
                    "preset": "baseline",
                    "model": "m",
                    "frozen": False,
                },
            ],
            "baselines": {"litsearch-local": "plain"},
            "rules": [
                {
                    "id": "P1",
                    "group": "P1",
                    "question": "good beats plain",
                    "set": "litsearch-local",
                    "metric": "recall_at_20",
                    "arm": "good",
                    "reference": "plain",
                    "test": "superior",
                },
                {
                    "id": "P2",
                    "group": "P2",
                    "question": "plain against bm25",
                    "set": "litsearch-local",
                    "metric": "recall_at_20",
                    "arm": "plain",
                    "reference": "bm25",
                    "test": "superior",
                    "gate": False,
                },
            ],
        }
    )


async def test_an_experiment_on_the_local_set_records_scores_and_places_the_losses(
    tmp_path: Path,
) -> None:
    root = dataset(tmp_path / "data")
    exp = local_experiment()
    dirs = DataDirs(litsearch_local=root)
    items = load_items(exp.set("litsearch-local"), dirs)
    assert 1 <= len(items) <= 6 and all(i.litsearch is not None for i in items)
    assert all(i.set_name == "litsearch-local" for i in items)
    # the gold work of an item is its corpus record
    for item in items:
        assert item.litsearch is not None
        assert all(w.local_id for w in item.litsearch.gold)

    everything = {
        i.question: [str(c) for c in i.litsearch.query.corpus_ids]
        for i in items
        if i.litsearch
    }
    best = ScriptedV2("good", everything)  # keeps the gold papers of every query
    nothing = ScriptedV2(
        "plain", {}, decision_calls=0
    )  # screens nothing: gold is not in its pool
    arms = {"good": best, "plain": nothing}

    def build(spec: ArmSpec, _mode: str, _decision: object) -> Any:
        if spec.system == "bm25":
            return Bm25Arm(corpus_dir=root, top_k=spec.top_k or 1, name=spec.name)
        return arms[spec.name]

    services = Services(
        answer_arm=build,  # type: ignore[arg-type]
        source_config=SourceConfig(local_corpus=root),
    )
    out = tmp_path / "out"
    written = await run_experiment(exp, out_dir=out, dirs=dirs, services=services)
    assert len(written) == 3 * len(items) and not [r for r in written if r.error]
    by_arm = {a: [r for r in written if r.arm == a] for a in ("bm25", "good", "plain")}
    assert all(r.metrics["recall_at_20"] == 1.0 for r in by_arm["good"])
    assert all(r.metrics["recall_at_20"] == 0.0 for r in by_arm["plain"])
    assert all(r.metrics["llm_calls"] == 0.0 for r in by_arm["bm25"])
    assert all(r.metrics["decision_answers"] == 7.0 for r in by_arm["good"])
    assert [r.kind for r in read_records(registry_path(out))][0] == "start"

    # gold papers are classified: a reference by its list, a pipeline by its record
    for record in by_arm["bm25"]:
        item = next(i for i in items if i.item_id == record.item_id)
        assert item.litsearch is not None
        for gold in item.litsearch.gold:
            assert classify_gold(record, gold) in (HIT, KEPT_LOW, NOT_IN_POOL)
    placed = {
        classify_gold(r, g)
        for r in by_arm["plain"]
        for i in items
        if i.item_id == r.item_id and i.litsearch
        for g in i.litsearch.gold
    }
    assert placed == {NOT_IN_POOL}  # it kept and screened nothing
    placed_good = {
        classify_gold(r, g)
        for r in by_arm["good"]
        for i in items
        if i.item_id == r.item_id and i.litsearch
        for g in i.litsearch.gold
    }
    assert placed_good == {HIT}

    report = build_report(
        exp, list(read_records(registry_path(out))), litsearch_local_dir=root
    )
    assert "## Loss table (LitSearch, local corpus)" in report
    assert "## Set litsearch-local" in report
    assert "### Decision answers" in report
    assert "P1" in report and "P2" in report and "## Verdicts" in report
    loss = report.split("## Loss table (LitSearch, local corpus)")[1].split("\n## ")[0]
    for name in ("bm25", "good", "plain"):
        assert f"| {name} |" in loss
    assert "outside its list" in loss


async def test_the_local_set_needs_its_folders(tmp_path: Path) -> None:
    exp = local_experiment()
    with pytest.raises(ValueError, match="--litsearch-local-dir"):
        load_items(exp.set("litsearch-local"), DataDirs())
    with pytest.raises(ValueError, match="--local-corpus"):
        await run_experiment(
            exp,
            out_dir=tmp_path / "o",
            dirs=DataDirs(litsearch_local=tmp_path),
            services=Services(),
        )


async def test_a_pipeline_arm_is_placed_by_its_screening_record() -> None:
    from andamentum.references import Work

    gold = Work(local_id="local:7", title="Seven")

    def record(entries: list[ScreeningEntry], works: list[Work]) -> Any:
        from ..bench.registry import RunRecord

        return RunRecord(
            experiment="e",
            experiment_hash="h",
            arm="a",
            set="litsearch-local",
            item_id="LS0001",
            repeat=1,
            started_at=0.0,
            output=ArmOutput(
                answer_text="",
                works=works,
                cost=Cost(wall_s=1.0),
                result=ResearchResult(
                    run_id="r", question="q", mode="scholarly", screening=entries
                ),
            ),
        )

    def entry(**fields: Any) -> ScreeningEntry:
        return ScreeningEntry(
            work_id="w", subquestion_id="S1", arrival=0, local_id="local:7", **fields
        )

    assert classify_gold(record([entry(p=0.9, kept=True, rank=1)], [gold]), gold) == HIT
    assert classify_gold(record([entry(p=0.1, kept=False)], []), gold) == SCREENED_OUT
    assert classify_gold(record([], []), gold) == NOT_IN_POOL


# -- the command line ---------------------------------------------------------------


def test_the_bench_commands_take_the_local_folders(tmp_path: Path) -> None:
    parser = bench_cli.build_parser()
    run = parser.parse_args(
        [
            "experiment",
            "run",
            str(ROUND4),
            "--out-dir",
            str(tmp_path / "o"),
            "--litsearch-local-dir",
            str(tmp_path / "d"),
            "--local-corpus",
            str(tmp_path / "d"),
            "--concurrency",
            "4",
        ]
    )
    assert (
        run.litsearch_local_dir == tmp_path / "d" and run.local_corpus == tmp_path / "d"
    )
    report = parser.parse_args(
        [
            "experiment",
            "report",
            str(ROUND4),
            "--out-dir",
            str(tmp_path / "o"),
            "--litsearch-local-dir",
            str(tmp_path / "d"),
        ]
    )
    assert report.litsearch_local_dir == tmp_path / "d"
    download = parser.parse_args(
        ["download", "litsearch-local", "--dir", str(tmp_path), "--force"]
    )
    assert download.dataset == "litsearch-local" and download.force
    config = bench_cli._source_config(None, keyless=True, local_corpus=tmp_path)  # type: ignore[attr-defined]
    assert config.local_corpus == tmp_path and config.openalex_api_key is None


def test_the_research_command_takes_the_local_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from .. import cli

    seen: list[dict[str, Any]] = []

    async def research(question: str, **kwargs: Any) -> ResearchResult:
        seen.append(kwargs)
        return ResearchResult(
            run_id="r", question=question, mode="scholarly", report_markdown="# x\n"
        )

    monkeypatch.setattr(cli, "research", research)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.chdir(tmp_path)  # the run log lands in the working directory
    cli.main(
        [
            "q",
            "--mode",
            "scholarly",
            "--model",
            "m",
            "--sources",
            "local_corpus",
            "--local-corpus",
            str(tmp_path),
            "--quiet",
        ]
    )
    sources: SourceConfig = seen[0]["sources"]
    assert sources.scholarly == ("local_corpus",) and sources.local_corpus == tmp_path
    with pytest.raises(SystemExit) as refused:
        cli.main(["q", "--mode", "scholarly", "--model", "m", "--sources", "nope"])
    assert refused.value.code == 2
