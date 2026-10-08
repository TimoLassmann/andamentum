"""Tests for the answering side of the benchmark: answer metrics, ArmResult rows,
concurrency, the answering and agent arms. Synthetic data; the LLM, the agent loop
and the reranker are stubbed, so nothing needs a model."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from andamentum.document_store import reranking
from andamentum.document_store.agent import AgentBudget, AgentRun, AgentSearcher
from andamentum.document_store.bench import answering, arms, data, runner
from andamentum.document_store.bench.arms import ArmResult
from andamentum.document_store.bench.data import Corpus, Passage, Question
from andamentum.document_store.bench.embedding_cache import EmbeddingCache
from andamentum.document_store.bench.metrics import answer_metrics, normalize_answer
from andamentum.document_store.types import Hit

PASSAGES = [
    Passage(pid="p1", title="SaltLog", text="SaltLog was founded by Jane Doe."),
    Passage(pid="p2", title="Apollo", text="Apollo is a river."),
]
QUESTIONS = [
    Question(
        qid=f"q{i}",
        text=f"question {i}?",
        group="bridge" if i % 2 else "comparison",
        gold=frozenset({"p1"}),
        answers=("Jane Doe", "J. Doe"),
    )
    for i in range(6)
]


def _corpus() -> Corpus:
    return Corpus(
        "synthetic", "smoke", QUESTIONS, PASSAGES, data.corpus_sha256(PASSAGES)
    )


_ARMS: dict[str, Any] = dict(arms.ARMS)


@pytest.fixture(autouse=True)
def _frozen_corpus(monkeypatch):
    monkeypatch.setattr(
        runner, "load_corpus", lambda *_a, **_kw: (_corpus(), {"version": 2})
    )
    monkeypatch.setattr(
        runner,
        "git_state",
        lambda: {"commit": "c0ffee", "dirty": False, "diff_sha256": None},
    )


# -- metrics ---------------------------------------------------------------


def test_normalisation_strips_case_punctuation_articles_and_spaces():
    assert normalize_answer("  The  Apollo, River! ") == "apollo river"
    assert normalize_answer("An Owl") == "owl"


def test_exact_match_and_f1_take_the_best_answer():
    m = answer_metrics("the Jane Doe.", ["J. Doe", "Jane Doe"])
    assert m == {"em": 1.0, "f1": 1.0, "unknown": 0.0}


def test_partial_overlap_gives_token_f1_without_exact_match():
    m = answer_metrics("Jane", ["Jane Doe"])
    assert m["em"] == 0.0
    assert m["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_no_overlap_and_empty_prediction_score_zero():
    zero = {"em": 0.0, "f1": 0.0, "unknown": 0.0}
    assert answer_metrics("Bob", ["Jane Doe"]) == zero
    assert answer_metrics("", ["Jane Doe"]) == zero


def test_an_unknown_answer_is_flagged_and_scores_zero():
    m = answer_metrics("Unknown.", ["Jane Doe"])
    assert m == {"em": 0.0, "f1": 0.0, "unknown": 1.0}


def test_scoring_without_answers_is_an_error():
    with pytest.raises(ValueError):
        answer_metrics("x", [])


def test_parsers_collect_answer_and_aliases():
    q, _ = data.parse_musique(
        {
            "id": "2hop__1",
            "question": "q?",
            "answer": "Jane Doe",
            "answer_aliases": ["J. Doe"],
            "paragraphs": [
                {"title": "t", "paragraph_text": "x", "is_supporting": True}
            ],
        }
    )
    assert q.answers == ("Jane Doe", "J. Doe")


# -- runner rows, manifest, ordering ---------------------------------------


class _AnsweringArm:
    name = "answering"
    store_kind = "none"
    needs_embeddings = False
    answers = True
    seen_run_ids: list[str | None] = []

    async def rank(self, built, q, emb, *, driver_model=None, run_id=None):
        self.seen_run_ids.append(run_id)
        # Later questions finish first, to prove order is kept.
        await asyncio.sleep(0.02 * (len(QUESTIONS) - int(q.qid[1:])))
        return ArmResult(
            ranked=["p1", "p2"],
            answer="Jane Doe" if q.qid != "q0" else "Bob",
            llm_calls=2,
            tokens_in=100,
            tokens_out=10,
            entered_loop=q.qid == "q1",
            trace=[{"step": 1, "kind": "tool:grep"}]
            + (
                [{"step": 2, "kind": "invalid_answer_fallback"}]
                if q.qid == "q1"
                else []
            ),
        )


async def _run(tmp_path, monkeypatch, arm, **kw):
    monkeypatch.setitem(runner.ARMS, arm.name, arm)
    return await runner.run(
        suite="synthetic",
        tier="smoke",
        arm_name=arm.name,
        workdir=tmp_path,
        out_dir=tmp_path / "run",
        embedding_model=None,
        **kw,
    )


async def test_rows_carry_answers_costs_and_manifest_records_the_run(
    tmp_path, monkeypatch
):
    arm = _AnsweringArm()
    arm.seen_run_ids = []
    run_dir = await _run(tmp_path, monkeypatch, arm, driver_model="m", concurrency=3)
    rows = [json.loads(line) for line in (run_dir / "questions.jsonl").open()]
    assert [r["qid"] for r in rows] == [q.qid for q in QUESTIONS]
    r0, r1 = rows[0], rows[1]
    assert r0["answer"] == "Bob" and r0["em"] == 0.0
    assert r1["em"] == 1.0 and r1["f1"] == 1.0
    assert r1["llm_calls"] == 2 and r1["tokens_in"] == 100 and r1["tokens_out"] == 10
    assert r1["entered_loop"] is True and r0["entered_loop"] is False
    assert r0["trace"] == [{"step": 1, "kind": "tool:grep"}]
    assert "seconds" in r1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["driver_model"] == "m" and manifest["concurrency"] == 3
    assert manifest["driver_settings"] == {"temperature": 0.0}
    assert len(manifest["prompts_sha256"]) == 64
    assert manifest["mean_llm_calls"] == 2
    assert manifest["mean_tokens_in"] == 100
    assert "mean_seconds" in manifest
    assert manifest["fallbacks"] == {"invalid_answer_fallback": 1}
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["all"]["em"] == pytest.approx(5 / 6)
    assert len(set(arm.seen_run_ids)) == len(QUESTIONS)
    assert all(rid and rid.startswith(run_dir.name) for rid in arm.seen_run_ids)
    assert "em 0.833" in runner.format_summary(run_dir)


async def test_plain_list_arms_still_work_and_have_no_answer_fields(
    tmp_path, monkeypatch
):
    class Plain:
        name = "plain"
        store_kind = "none"
        needs_embeddings = False
        answers = False

        async def rank(self, built, q, emb, **_kw):
            return ["p1"]

    run_dir = await _run(tmp_path, monkeypatch, Plain())
    rows = [json.loads(line) for line in (run_dir / "questions.jsonl").open()]
    assert all("answer" not in r and "em" not in r for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["driver_model"] is None and manifest["concurrency"] == 1
    assert manifest["driver_settings"] is None
    assert "mean_llm_calls" not in manifest


async def test_a_model_driven_search_arm_records_costs_but_no_answer(
    tmp_path, monkeypatch
):
    class AgentSearch:
        name = "agent_search"
        store_kind = "none"
        needs_embeddings = False
        answers = False
        llm_driven = True

        async def rank(self, built, q, emb, **_kw):
            return ArmResult(
                ranked=["p1"],
                llm_calls=3,
                tokens_in=40,
                tokens_out=4,
                entered_loop=True,
                trace=[{"step": 1, "kind": "request_limit_fallback"}],
            )

    run_dir = await _run(tmp_path, monkeypatch, AgentSearch(), driver_model="m")
    rows = [json.loads(line) for line in (run_dir / "questions.jsonl").open()]
    assert all("answer" not in r and "em" not in r for r in rows)
    assert all(r["llm_calls"] == 3 and r["entered_loop"] is True for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["driver_settings"] == {"temperature": 0.0}
    assert manifest["mean_llm_calls"] == 3
    assert len(manifest["prompts_sha256"]) == 64
    assert manifest["fallbacks"] == {"request_limit_fallback": len(rows)}
    assert "cost: llm_calls 3.00" in runner.format_summary(run_dir)


async def test_an_answering_arm_without_an_answer_fails_the_run(tmp_path, monkeypatch):
    class Silent:
        name = "silent"
        store_kind = "none"
        needs_embeddings = False
        answers = True

        async def rank(self, built, q, emb, **_kw):
            return ArmResult(ranked=["p1"])

    with pytest.raises(ValueError, match="returned no answer"):
        await _run(tmp_path, monkeypatch, Silent())


async def test_compare_prints_em_f1_and_unknown(tmp_path, monkeypatch):
    arm = _AnsweringArm()
    arm.seen_run_ids = []
    a = await _run(tmp_path / "a", monkeypatch, arm, driver_model="m")
    b = await _run(tmp_path / "b", monkeypatch, arm, driver_model="m")
    text = runner.compare(a, b)
    assert " em " in text and " f1 " in text and " unknown " in text


# -- the answering and agent arms -------------------------------------------


async def _embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(t) % 7), float(len(t) % 5), 1.0] for t in texts]


def _cache(tmp_path) -> EmbeddingCache:
    return EmbeddingCache(
        tmp_path / "emb.sqlite", model="fake-embed", digest="d1", embed_fn=_embed
    )


@pytest.fixture
def stub_rerank(monkeypatch):
    """Cross-encoder stub: scores fall with the passage's position in the input."""

    async def fake_rerank(query, texts, *, reranker_model):
        return [1.0 / (i + 1) for i in range(len(texts))]

    monkeypatch.setattr(reranking, "rerank", fake_rerank)


@pytest.mark.parametrize(
    "name", ["closed_book", "rag", "agent", "agent_lexical"]
)
async def test_model_driven_arms_refuse_a_missing_driver_model(name):
    with pytest.raises(ValueError, match="driver-model"):
        await _ARMS[name].rank(object(), QUESTIONS[0], object())


def test_arm_flags_split_answering_from_search_arms():
    assert {n for n, a in _ARMS.items() if a.answers} == {"closed_book", "rag"}
    assert {n for n, a in _ARMS.items() if getattr(a, "llm_driven", False)} == {
        "agent",
        "agent_lexical",
    }
    assert _ARMS["agent_lexical"].needs_embeddings is False
    assert _ARMS["agent"].needs_embeddings is True


async def test_rag_answers_from_the_top_ten_after_rerank(
    tmp_path, monkeypatch, stub_rerank
):
    corpus_passages = [
        Passage(pid=f"p{i}", title=f"T{i}", text=f"Widget number {i} is a thing.")
        for i in range(14)
    ]
    corpus = Corpus(
        "synthetic",
        "smoke",
        QUESTIONS,
        corpus_passages,
        data.corpus_sha256(corpus_passages),
    )
    emb = _cache(tmp_path)
    built = await arms.build_store(corpus, workdir=tmp_path, emb=emb)
    seen: dict[str, Any] = {}

    async def fake_answer_from_hits(question, hs, *, model, store, model_settings):
        seen.update(n=len(hs), store=store, model=model, settings=model_settings)
        return SimpleNamespace(text="Jane Doe"), SimpleNamespace(
            requests=1, input_tokens=50, output_tokens=5
        )

    monkeypatch.setattr(answering, "answer_from_hits", fake_answer_from_hits)
    q = Question(qid="q", text="widget thing", group="g", gold=frozenset({"p1"}))
    out = await _ARMS["rag"].rank(built, q, emb, driver_model="m")
    await built.store.close()
    assert isinstance(out, ArmResult)
    assert seen["n"] == 10 and seen["model"] == "m"
    assert seen["store"] is built.store  # citations are checked against the store
    assert seen["settings"] == {"temperature": 0.0}
    assert out.answer == "Jane Doe" and out.llm_calls == 1 and out.tokens_in == 50
    assert out.tokens_out == 5 and out.entered_loop is False
    assert out.ranked and set(out.ranked) <= {p.pid for p in corpus_passages}
    assert len(out.ranked) == len(set(out.ranked))


async def test_closed_book_answers_without_a_store(monkeypatch):
    async def fake_closed_book(question, *, model, model_settings):
        assert model == "m" and model_settings == {"temperature": 0.0}
        return SimpleNamespace(text="Jane Doe"), SimpleNamespace(
            requests=1, input_tokens=20, output_tokens=3
        )

    monkeypatch.setattr(answering, "answer_closed_book", fake_closed_book)
    out = await _ARMS["closed_book"].rank(
        arms.BuiltPool(["p1"]), QUESTIONS[0], None, driver_model="m"
    )
    assert out.ranked == [] and out.answer == "Jane Doe" and out.llm_calls == 1
    assert out.tokens_in == 20 and out.tokens_out == 3 and out.entered_loop is False


def _fake_built(*, with_hybrid: bool) -> SimpleNamespace:
    hybrid = SimpleNamespace(name="hybrid") if with_hybrid else None
    return SimpleNamespace(
        store=object(),
        hybrid=hybrid,
        lexical=SimpleNamespace(name="lexical"),
        doc_to_pid={"d1": "p1", "d2": "p2"},
    )


def _stub_search_with_run(monkeypatch, seen: dict, run: AgentRun):
    async def fake(self, store, query, *, limit, where):
        seen.update(
            searcher=self, store=store, query=query, limit=limit, where=where
        )
        return [
            Hit("d2", 1, 0, 5, "x", 0.9),
            Hit("d1", 1, 0, 5, "y", 0.8),
            Hit("d2", 1, 5, 9, "z", 0.7),
        ], run

    monkeypatch.setattr(AgentSearcher, "search_with_run", fake)


async def test_agent_arm_maps_hits_usage_and_trace(monkeypatch):
    seen: dict[str, Any] = {}
    run = AgentRun(
        llm_calls=5,
        tokens_in=900,
        tokens_out=80,
        entered_loop=True,
        trace=[{"step": 0, "kind": "round0"}],
    )
    _stub_search_with_run(monkeypatch, seen, run)
    built = _fake_built(with_hybrid=True)
    out = await _ARMS["agent"].rank(built, QUESTIONS[0], None, driver_model="m")
    searcher = seen["searcher"]
    assert searcher.model == "m"
    assert searcher.model_settings == {"temperature": 0.0}
    assert searcher.budget == AgentBudget()
    assert searcher.inner.index is built.hybrid
    assert seen["store"] is built.store and seen["where"] is None
    assert seen["limit"] == arms.HIT_LIMIT and seen["query"] == QUESTIONS[0].text
    assert out.answer is None  # a search arm: ranked passages, no answer
    assert out.ranked == ["p2", "p1"]  # doc -> passage id, group_by_document order
    assert out.llm_calls == 5 and out.tokens_in == 900 and out.tokens_out == 80
    assert out.entered_loop is True
    assert out.trace == [{"step": 0, "kind": "round0"}]


async def test_agent_lexical_arm_runs_over_the_lexical_index(monkeypatch):
    seen: dict[str, Any] = {}
    _stub_search_with_run(monkeypatch, seen, AgentRun(llm_calls=1))
    built = _fake_built(with_hybrid=False)  # a run without embeddings
    out = await _ARMS["agent_lexical"].rank(
        built, QUESTIONS[0], None, driver_model="m"
    )
    assert seen["searcher"].inner.index is built.lexical
    assert out.ranked == ["p2", "p1"] and out.entered_loop is False


async def test_agent_arm_without_a_hybrid_index_is_an_error(monkeypatch):
    seen: dict[str, Any] = {}
    _stub_search_with_run(monkeypatch, seen, AgentRun())
    with pytest.raises(ValueError, match="embedding cache"):
        await _ARMS["agent"].rank(
            _fake_built(with_hybrid=False), QUESTIONS[0], None, driver_model="m"
        )
    assert not seen  # failed before searching


async def test_an_agent_run_records_its_costs_through_the_runner(
    tmp_path, monkeypatch
):
    run = AgentRun(
        llm_calls=3,
        tokens_in=40,
        tokens_out=4,
        entered_loop=True,
        trace=[{"step": 1, "kind": "request_limit_fallback"}],
    )

    async def fake(self, store, query, *, limit, where):
        refs = await store.list(limit=None)
        return [Hit(r.doc_id, r.rev, 0, 3, "x", 1.0) for r in refs[:2]], run

    monkeypatch.setattr(AgentSearcher, "search_with_run", fake)
    monkeypatch.setattr(runner, "_reranker_revision", lambda name: "rev1")
    run_dir = await runner.run(
        suite="synthetic",
        tier="smoke",
        arm_name="agent_lexical",
        workdir=tmp_path,
        out_dir=tmp_path / "run",
        embedding_model=None,
        driver_model="m",
    )
    rows = [json.loads(line) for line in (run_dir / "questions.jsonl").open()]
    assert all("answer" not in r and "em" not in r for r in rows)
    assert all(len(r["ranked"]) == 2 for r in rows)
    assert all(r["llm_calls"] == 3 and r["entered_loop"] is True for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["arm"] == "agent_lexical"
    assert manifest["store"] == "store" and manifest["store_file"]["reused"] is False
    assert manifest["driver_settings"] == {"temperature": 0.0}
    assert manifest["reranker_revision"] == "rev1"
    assert manifest["mean_llm_calls"] == 3
    assert len(manifest["prompts_sha256"]) == 64
    assert manifest["fallbacks"] == {"request_limit_fallback": len(rows)}
    assert "cost: llm_calls 3.00" in runner.format_summary(run_dir)
