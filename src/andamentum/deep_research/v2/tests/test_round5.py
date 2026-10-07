"""Round 5: the ``hybrid`` preset (raw question as a query, dense similarity in
the fusion, the screen as a ranker) and the run log every run fills.

All offline: a fake embedder gives deterministic vectors, the engine runs over
``engine_world``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.references import Work

from .. import ResearchStore, SourceConfig, cli, research
from ..config import PRESETS, canonical_json, config_hash
from ..criteria import Screener
from ..dense import similarities, text_sha
from ..runlog import write_log
from ..strategies import (
    LIST_DENSE_QUESTION,
    LIST_DENSE_SUBQUESTION,
    LIST_SCREEN,
    RERANK_DEPTH,
    RRF_K,
    RankOnlyScreen,
    RankSignals,
    RerankDenseRank,
    RrfDenseRank,
    fusion_rows,
    screen_one,
)
from ..types import (
    LOG_STEPS,
    Budget,
    Candidate,
    CandidateScore,
    LogEvent,
    ResearchResult,
    RunLog,
    RunStats,
    SubQuestion,
)
from .engine_world import QUESTION, SEARXNG, World, world_decisions, world_llm
from .llm_stub import decisions, make_asker

SUBQ = SubQuestion(id="Q1", text="How much does the Apollo cache reduce latency?")
_TERMS = ("apollo", "saltlog", "latency", "cache", "logging")


class FakeEmbedder:
    """Deterministic vectors: how often each of a few words occurs, plus one."""

    name = "fake-embed"
    digest: str | None = None

    def __init__(self) -> None:
        self.requests: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(list(texts))
        return [
            [float(t.casefold().count(term)) for term in _TERMS] + [1.0] for t in texts
        ]


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


def _candidate(n: int, ranks: dict[str, int] | None = None) -> Candidate:
    return Candidate(
        work=Work(
            doi=f"10.5555/w{n}",
            title=f"Apollo work {n}",
            abstract=f"Apollo cache study number {n}.",
            year=2024,
            work_id=f"w{n}",
        ),
        sources=["openalex"],
        subquestion_ids=["Q1"],
        source_ranks=ranks or {},
    )


def _score(work_id: str, p: float, *, kept: bool = True) -> CandidateScore:
    return CandidateScore(work_id=work_id, subquestion_id="Q1", score=p, kept=kept)


# -- the preset ----------------------------------------------------------------------


def test_hybrid_is_the_baseline_with_three_strategies_swapped() -> None:
    base = PRESETS["baseline"].model_dump(exclude={"name", "description"})
    hybrid = PRESETS["hybrid"].model_dump(exclude={"name", "description"})
    differ = {k for k in base if base[k] != hybrid[k]}
    assert differ == {"screen", "rank", "queries"}
    assert hybrid["screen"] == "rank_only"
    assert hybrid["rank"] == "rerank_dense"
    assert hybrid["queries"] == "llm_plus_question"
    assert PRESETS["hybrid"].needs_embeddings
    assert not any(PRESETS[n].needs_embeddings for n in PRESETS if n != "hybrid")


def test_the_queries_slot_leaves_every_earlier_hash_alone() -> None:
    # the new slot is out of the hash at its default, like verify and queue
    assert '"queries"' not in canonical_json(PRESETS["baseline"])
    assert '"queries"' in canonical_json(PRESETS["hybrid"])
    for name in ("baseline", "fast", "recall", "criteria"):
        data = PRESETS[name].model_dump(mode="json")
        assert data["queries"] == "llm"
    assert config_hash(PRESETS["hybrid"]) != config_hash(PRESETS["baseline"])


async def test_hybrid_without_an_embedding_model_is_refused() -> None:
    with pytest.raises(ValueError, match="needs an embedding model"):
        await research(
            QUESTION, mode="scholarly", model="test", config=PRESETS["hybrid"]
        )


# -- rank_only ------------------------------------------------------------------------


async def test_rank_only_never_drops_and_p_is_recorded() -> None:
    model, stub = decisions(lambda state, _q: 0.05)
    asker = make_asker(decision_model=model)
    screener = Screener(asker=asker)
    candidates = [_candidate(1), _candidate(2)]
    for c in candidates:
        asker.store.upsert_work(c.work)
    run = screen_one(PRESETS["hybrid"], criteria={}, screener=screener, question="q")
    _standing, scores = await run(SUBQ, candidates)
    assert len(stub.calls) == 2
    assert all(s.kept for s in scores)
    assert all(s.score == pytest.approx(0.05) for s in scores)
    assert all("never dropped" in s.reason for s in scores)


async def test_rank_only_asks_about_the_original_question_not_the_part() -> None:
    model, stub = decisions(lambda state, _q: 0.5)
    asker = make_asker(decision_model=model)
    screener = Screener(asker=asker)
    candidates = [_candidate(1), _candidate(2)]
    for c in candidates:
        asker.store.upsert_work(c.work)
    question = "Which Apollo study measures cache latency and SaltLog volume?"
    run = screen_one(
        PRESETS["hybrid"], criteria={}, screener=screener, question=question
    )
    await run(SUBQ, candidates)
    asked = [str(q["instructions"]) for _state, qs in stub.calls for q in qs.values()]
    assert len(asked) == 2
    assert all(question.rstrip("?") in text for text in asked)
    assert not any(SUBQ.text.rstrip("?") in text for text in asked)


def test_rank_only_offers_only_the_depth_best_first() -> None:
    n = RERANK_DEPTH + 5
    candidates = {f"w{i}": _candidate(i, {"openalex": i}) for i in range(1, n + 1)}
    pairs = [(f"w{i}", "Q1") for i in range(n, 0, -1)]  # worst first
    chosen = RankOnlyScreen().select(
        pairs, candidates=candidates, signals=RankSignals(pairs=pairs)
    )
    assert len(chosen) == RERANK_DEPTH
    assert chosen[0] == ("w1", "Q1")
    assert ("w51", "Q1") not in chosen  # beyond the depth: keeps its rank, not asked


def test_p_ordering_moves_ranks() -> None:
    candidates = {"w1": _candidate(1, {"s": 1}), "w2": _candidate(2, {"s": 1})}
    pairs = [("w1", "Q1"), ("w2", "Q1")]
    signals = RankSignals(pairs=pairs)
    low_first = RrfDenseRank().kept(
        {pairs[0]: _score("w1", 0.2), pairs[1]: _score("w2", 0.8)},
        candidates=candidates,
        signals=signals,
    )
    assert low_first["w2"][0] > low_first["w1"][0]
    high_first = RrfDenseRank().kept(
        {pairs[0]: _score("w1", 0.8), pairs[1]: _score("w2", 0.2)},
        candidates=candidates,
        signals=signals,
    )
    assert high_first["w1"][0] > high_first["w2"][0]


# -- rrf_dense --------------------------------------------------------------------------


def test_the_fused_order_of_four_candidates_by_hand() -> None:
    candidates = {
        "w1": _candidate(1, {"s": 1}),
        "w2": _candidate(2, {"s": 2}),
        "w3": _candidate(3, {"s": 3}),
        "w4": _candidate(4),
    }
    pairs = [(w, "Q1") for w in ("w1", "w2", "w3", "w4")]
    signals = RankSignals(
        pairs=pairs,
        dense_subquestion={
            ("w1", "Q1"): 0.1,
            ("w2", "Q1"): 0.9,
            ("w3", "Q1"): 0.5,
            ("w4", "Q1"): 0.7,
        },
        dense_question={"w1": 0.8, "w2": 0.2, "w3": 0.6, "w4": 0.4},
    )
    scores = {("w3", "Q1"): _score("w3", 0.9), ("w1", "Q1"): _score("w1", 0.3)}
    rows = fusion_rows(scores, candidates=candidates, signals=signals)
    assert rows[("w1", "Q1")].ranks == {
        "s": 1,
        LIST_DENSE_SUBQUESTION: 4,
        LIST_DENSE_QUESTION: 1,
        LIST_SCREEN: 2,
    }

    def rrf(*ranks: int) -> float:
        return sum(1 / (RRF_K + r) for r in ranks)

    expected = {
        "w1": rrf(1, 4, 1, 2),
        "w2": rrf(2, 1, 4),
        "w3": rrf(3, 3, 2, 1),
        "w4": rrf(2, 3),
    }
    kept = RrfDenseRank().kept(scores, candidates=candidates, signals=signals)
    for work_id, value in expected.items():
        assert kept[work_id][0] == pytest.approx(value)
    order = sorted(kept, key=lambda w: -kept[w][0])
    assert order == ["w1", "w3", "w2", "w4"]


def test_rerank_dense_orders_by_dense_and_p_only() -> None:
    # w1 is in three keyword lists; w2 came in by a citation hop alone, so it is
    # in none of them, but both dense lists put it first
    candidates = {
        "w1": _candidate(1, {"a": 1, "b": 1, "c": 1}),
        "w2": _candidate(2, {"openalex:references": 9}),
    }
    pairs = [("w1", "Q1"), ("w2", "Q1")]
    signals = RankSignals(
        pairs=pairs,
        dense_subquestion={("w1", "Q1"): 0.4, ("w2", "Q1"): 0.9},
        dense_question={"w1": 0.3, "w2": 0.8},
    )
    scores = {pairs[0]: _score("w1", 0.6), pairs[1]: _score("w2", 0.7)}
    fused = RrfDenseRank().kept(scores, candidates=candidates, signals=signals)
    assert fused["w1"][0] > fused["w2"][0]  # more lists win the plain fusion
    reranked = RerankDenseRank().kept(scores, candidates=candidates, signals=signals)
    assert reranked["w2"][0] > reranked["w1"][0]
    assert reranked["w2"][0] == pytest.approx(3 / (RRF_K + 1))
    rows = fusion_rows(
        scores, candidates=candidates, signals=signals, with_sources=False
    )
    assert rows[("w1", "Q1")].ranks["a"] == 1  # kept for the log, not fused


def test_a_dropped_or_retracted_candidate_is_not_ranked() -> None:
    retracted = _candidate(2, {"s": 2})
    retracted = retracted.model_copy(
        update={"work": retracted.work.model_copy(update={"retracted": "retracted"})}
    )
    candidates = {"w1": _candidate(1, {"s": 1}), "w2": retracted, "w3": _candidate(3)}
    pairs = [("w1", "Q1"), ("w2", "Q1"), ("w3", "Q1")]
    kept = RrfDenseRank().kept(
        {("w1", "Q1"): _score("w1", 0.1, kept=False)},
        candidates=candidates,
        signals=RankSignals(pairs=pairs),
    )
    assert set(kept) == {"w3"}


async def test_dense_similarities_are_cached_in_the_store() -> None:
    store = ResearchStore.in_memory()
    embedder = FakeEmbedder()
    candidates = {"w1": _candidate(1), "w2": _candidate(2)}
    pairs = [("w1", "Q1"), ("w2", "Q1")]

    async def run(stats: RunStats):  # type: ignore[no-untyped-def]
        return await similarities(
            pairs,
            question=QUESTION,
            subquestions=[SUBQ],
            candidates=candidates,
            known={},
            embedder=embedder,
            store=store,
            stats=stats,
            phase="Screen",
            clock=lambda: 0.0,
        )

    first_stats, second_stats = RunStats(), RunStats()
    first = await run(first_stats)
    assert first_stats.embedding_calls == 1
    assert first_stats.step_embedding_calls == {"Screen": 1}
    assert len(embedder.requests[0]) == 4  # question, sub-question, two works
    second = await run(second_stats)
    assert second_stats.embedding_calls == 0
    assert second_stats.embedding_cache_hits == 4
    assert len(embedder.requests) == 1
    assert second.subquestion == pytest.approx(first.subquestion)
    assert store.cached_embeddings(model="fake-embed", shas=[text_sha(QUESTION)])


async def test_the_embedding_cache_reads_through_a_shared_store() -> None:
    shared = ResearchStore.in_memory()
    shared.cache_embeddings(model="m", vectors={"abc": [1.0, 2.0]})
    own = ResearchStore.in_memory(shared_cache=shared)
    assert own.cached_embeddings(model="m", shas=["abc"]) == {"abc": [1.0, 2.0]}
    assert own._own_embeddings(model="m", shas=["abc"]) == {"abc": [1.0, 2.0]}


# -- the engine ----------------------------------------------------------------------------


async def _hybrid_run(store: ResearchStore | None = None) -> ResearchResult:
    world = World()
    store = store or ResearchStore.in_memory()
    decision, _stub = world_decisions()
    return await research(
        QUESTION,
        mode="scholarly",
        model=world_llm().model,
        decision_model=decision,
        embedding_model=FakeEmbedder(),
        store=store,
        budget=Budget(),
        sources=SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",)),
        config=PRESETS["hybrid"],
        source_client=world.client(store),
    )


async def test_the_hybrid_preset_runs_end_to_end() -> None:
    result = await _hybrid_run()
    assert result.config_name == "hybrid"
    # the raw question is searched once per run, not once per sub-question
    folded = " ".join(QUESTION.split())
    assert result.queries.count(folded) == 1
    lists = {name for e in result.screening for name in e.source_ranks}
    assert "openalex:question" in lists
    assert result.stats.embedding_calls >= 1
    # nothing screened is dropped; the result is the fused order of every held work
    assert all(e.kept for e in result.screening if e.p is not None)
    held = {e.work_id for e in result.screening} - set(result.excluded)
    assert {w.work_id for w in result.works} == held
    ranks = [e.rank for e in result.screening if e.rank is not None]
    assert ranks and all(r <= len(result.works) for r in ranks)


async def test_the_run_log_covers_every_step_in_order() -> None:
    result = await _hybrid_run()
    log = result.log
    assert log.run_id == result.run_id
    assert [e.seq for e in log.events] == list(range(len(log.events)))
    steps = [e.step for e in log.events]
    for step in ("Setup", "Plan", "Search", "Screen", "Read", "Coverage", "Write"):
        assert step in steps
    assert steps[0] == "Setup" and steps[-1] == "Finish"
    seen = list(dict.fromkeys(steps))
    assert seen.index("Plan") < seen.index("Search") < seen.index("Screen")
    names = {e.event for e in log.events}
    for name in (
        "config",
        "models",
        "budget",
        "subquestion",
        "query",
        "admitted",
        "screened",
        "screen_batch",
        "screen_round",
        "embedded",
        "llm_call",
        "read_chosen",
        "coverage",
        "stop_decision",
        "report",
        "step_done",
        "final_ranking",
        "finished",
    ):
        assert name in names, name
    raw = [e for e in log.events if e.event == "query" and e.data.get("raw_question")]
    assert len(raw) == 1
    models = next(e for e in log.events if e.event == "models")
    assert models.data["embedding_model"] == "fake-embed"
    final = next(e for e in log.events if e.event == "final_ranking")
    works = final.data["works"]
    assert isinstance(works, list) and works
    first = works[0]
    assert isinstance(first, dict) and LIST_DENSE_QUESTION in first["ranks"]  # type: ignore[operator]
    llm = next(e for e in log.events if e.event == "llm_call")
    assert len(str(llm.data["prompt_sha256"])) == 64


async def test_the_log_is_saved_in_the_store() -> None:
    store = ResearchStore.in_memory()
    result = await _hybrid_run(store)
    items = store.run_items(result.run_id, "log")
    assert [i.ref_id for i in items] == ["run"]
    assert RunLog.model_validate(items[0].data) == result.log


async def test_every_preset_fills_a_log() -> None:
    world = World()
    store = ResearchStore.in_memory()
    result = await research(
        QUESTION,
        mode="scholarly",
        model=world_llm().model,
        store=store,
        sources=SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",)),
        source_client=world.client(store),
    )
    assert result.log.events and result.log.events[-1].event == "finished"


async def test_a_failed_run_carries_its_log(monkeypatch: pytest.MonkeyPatch) -> None:
    from .. import graph as graph_module

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("a bug in coverage")

    monkeypatch.setattr(graph_module, "compute_coverage", broken)
    with pytest.raises(RuntimeError) as raised:
        await _hybrid_run()
    log = getattr(raised.value, "run_log", None)
    assert isinstance(log, RunLog)
    assert log.events[-1].event == "failed"


# -- rendering -------------------------------------------------------------------------------


async def test_jsonl_round_trips_and_markdown_has_the_steps() -> None:
    log = (await _hybrid_run()).log
    text = log.to_jsonl()
    assert len(text.splitlines()) == len(log.events) + 1
    assert RunLog.from_jsonl(text) == log
    markdown = log.to_markdown()
    for step in ("Setup", "Plan", "Search", "Screen", "Finish"):
        assert f"\n## {step}\n" in markdown
    assert "| rank | title | fused | signal ranks | p |" in markdown
    assert (
        "| t (s) | sub-question | query | source | results | top titles |" in markdown
    )


def test_markdown_escapes_pipes_and_groups_tables() -> None:
    log = RunLog(
        run_id="r1",
        events=[
            LogEvent(seq=0, t=0.0, step="Setup", event="config", message="hello"),
            LogEvent(
                seq=1,
                t=1.0,
                step="Search",
                event="query",
                message="q",
                data={"query": "a|b", "count": 3, "top": [{"rank": 1, "title": "T"}]},
            ),
            LogEvent(
                seq=2,
                t=1.5,
                step="Search",
                event="query",
                message="q",
                data={"query": "c", "count": 0, "top": []},
            ),
        ],
    )
    markdown = log.to_markdown()
    assert "a\\|b" in markdown
    assert markdown.count("| t (s) | sub-question") == 1  # one table for both rows
    assert "1. T" in markdown
    assert LOG_STEPS[0] == "Setup"


def test_write_log_writes_both_files(tmp_path: Path) -> None:
    log = RunLog(
        run_id="r1",
        events=[LogEvent(seq=0, t=0.0, step="Setup", event="config", message="x")],
    )
    md, jsonl = write_log(log, md_path=tmp_path / "deep" / "a.log.md")
    assert md.read_text().startswith("# Run log r1")
    assert jsonl == tmp_path / "deep" / "a.log.jsonl"
    assert RunLog.from_jsonl(jsonl.read_text()) == log


# -- the CLI ------------------------------------------------------------------------------------


def _logged_result() -> ResearchResult:
    return ResearchResult(
        run_id="abcdef123456",
        question="q",
        mode="scholarly",
        report_markdown="# q\n",
        stop_reason="covered",
        log=RunLog(
            run_id="abcdef123456",
            events=[LogEvent(seq=0, t=0.0, step="Setup", event="config", message="x")],
        ),
    )


@pytest.fixture
def cli_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def fake(question: str, **kwargs: Any) -> ResearchResult:
        seen.append(kwargs)
        return _logged_result()

    monkeypatch.setattr(cli, "research", fake)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    return seen


def test_the_cli_writes_the_log_beside_out(
    cli_calls: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "runs" / "answer.md"
    out.parent.mkdir()
    cli.main(["q", "--mode", "scholarly", "--model", "m", "--quiet", "--out", str(out)])
    assert (out.parent / "answer.log.md").exists()
    assert (out.parent / "answer.log.jsonl").exists()
    assert not list(tmp_path.glob("research-*"))


def test_the_cli_writes_the_log_in_the_working_directory_without_out(
    cli_calls: list[dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    cli.main(["q", "--mode", "scholarly", "--model", "m", "--quiet"])
    assert (tmp_path / "research-abcdef12.log.md").exists()
    assert (tmp_path / "research-abcdef12.log.jsonl").exists()
    assert "research-abcdef12.log.md" in capsys.readouterr().err


def test_the_log_flag_overrides(
    cli_calls: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "elsewhere" / "mine.md"
    cli.main(
        [
            "q",
            "--mode",
            "scholarly",
            "--model",
            "m",
            "--quiet",
            "--embedding-model",
            "embeddinggemma:latest",
            "--log",
            str(target),
        ]
    )
    assert target.exists() and (target.parent / "mine.jsonl").exists()
    assert cli_calls[0]["embedding_model"] == "embeddinggemma:latest"


def test_a_failed_cli_run_writes_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    async def failing(question: str, **_kwargs: Any) -> ResearchResult:
        error = RuntimeError("boom")
        error.run_log = _logged_result().log  # type: ignore[attr-defined]
        raise error

    monkeypatch.setattr(cli, "research", failing)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    with pytest.raises(SystemExit) as stop:
        cli.main(["q", "--mode", "scholarly", "--model", "m", "--quiet"])
    assert stop.value.code == 1
    assert (tmp_path / "research-abcdef12.log.md").exists()


# -- the bench -----------------------------------------------------------------------------------


def _hybrid_experiment() -> Any:
    from ..bench.experiment import Experiment

    return Experiment.model_validate(
        {
            "name": "hybrid-tiny",
            "registered": "2026-10-07",
            "set_sources": {"litsearch-local": {"scholarly": ["local_corpus"]}},
            "sets": [{"name": "litsearch-local", "n": 2, "skip": 0, "seed": 0}],
            "baselines": {"litsearch-local": "hyb"},
            "arms": [
                {
                    "name": "hyb",
                    "system": "v2",
                    "preset": "hybrid",
                    "model": "m",
                    "frozen": False,
                }
            ],
        }
    )


async def test_an_experiment_with_a_hybrid_arm_needs_an_embedding_model(
    tmp_path: Path,
) -> None:
    from ..bench.experiment_run import DataDirs, Services, run_experiment
    from .test_round4 import dataset

    root = dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="--embedding-model"):
        await run_experiment(
            _hybrid_experiment(),
            out_dir=tmp_path / "out",
            dirs=DataDirs(litsearch_local=root),
            services=Services(source_config=SourceConfig(local_corpus=root)),
        )


def test_the_bench_passes_the_embedding_model_to_a_hybrid_arm_only() -> None:
    from ..bench.arms import V2Arm
    from ..bench.experiment import ArmSpec
    from ..bench.experiment_run import Services, arm_record, build_answer_arm

    services = Services(embedding_model="embeddinggemma:latest")
    hybrid = ArmSpec(name="h", system="v2", preset="hybrid", model="m", frozen=False)
    base = ArmSpec(name="b", system="v2", preset="baseline", model="m", frozen=False)
    built = build_answer_arm(hybrid, "scholarly", services)
    plain = build_answer_arm(base, "scholarly", services)
    assert isinstance(built, V2Arm) and isinstance(plain, V2Arm)
    assert built.embedding_model == "embeddinggemma:latest"
    assert plain.embedding_model is None
    exp = _hybrid_experiment()
    assert arm_record(exp, hybrid, services).embedding_model == "embeddinggemma:latest"
    assert arm_record(exp, base, services).embedding_model is None


async def test_each_experiment_cell_writes_its_log(tmp_path: Path) -> None:
    from ..bench.arms import ArmOutput
    from ..bench.experiment_run import (
        DataDirs,
        Services,
        cell_log_path,
        run_experiment,
    )
    from ..bench.metrics import Cost
    from ..bench.registry import read_records, registry_path
    from .test_round4 import dataset

    root = dataset(tmp_path / "data")

    class Logged:
        name = "hyb"
        returns_pages = False

        async def run(
            self, question: str, *, frozen: object = None, store: object = None
        ) -> ArmOutput:
            return ArmOutput(
                answer_text="",
                cost=Cost(wall_s=1.0),
                log=RunLog(
                    run_id="r",
                    events=[
                        LogEvent(
                            seq=0, t=0.0, step="Setup", event="config", message=question
                        )
                    ],
                ),
            )

    out = tmp_path / "out"
    written = await run_experiment(
        _hybrid_experiment(),
        out_dir=out,
        dirs=DataDirs(litsearch_local=root),
        services=Services(
            answer_arm=lambda _spec, _mode, _decision: Logged(),  # type: ignore[arg-type,return-value]
            source_config=SourceConfig(local_corpus=root),
            embedding_model="fake",
        ),
    )
    assert written and not [r for r in written if r.error]
    for record in written:
        md = cell_log_path(out, arm="hyb", item_id=record.item_id, repeat=1)
        assert md.exists() and md.with_name(md.name[:-3] + ".jsonl").exists()
    # the registry keeps no log
    text = registry_path(out).read_text()
    assert '"log"' not in text
    assert list(read_records(registry_path(out)))
