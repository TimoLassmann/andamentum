"""Tests for the answering side of the benchmark: answer metrics, ArmResult rows,
concurrency, the agent arms. Synthetic data; Part B's agent module is faked."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from andamentum.document_store.bench import arms, data, runner
from andamentum.document_store.bench.arms import ArmResult
from andamentum.document_store.bench.data import Corpus, Passage, Question
from andamentum.document_store.bench.metrics import answer_metrics, normalize_answer

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


def _fake_agent_module(monkeypatch, **overrides):
    @dataclass(frozen=True)
    class AgentBudget:
        max_tool_calls: int = 12

    mod = types.ModuleType("andamentum.document_store.v2.agent")
    mod.AgentBudget = AgentBudget  # type: ignore[attr-defined]
    for k, v in overrides.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, "andamentum.document_store.v2.agent", mod)
    return mod


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
    store_kind = "current"
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
            trace=[{"step": 1, "kind": "tool:grep"}],
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
    assert r1["trace"] == [{"step": 1, "kind": "tool:grep"}]
    assert "seconds" in r1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["driver_model"] == "m" and manifest["concurrency"] == 3
    assert manifest["driver_settings"] == {"temperature": 0.0}
    assert len(manifest["prompts_sha256"]) == 64
    assert manifest["mean_llm_calls"] == 2
    assert manifest["mean_tokens_in"] == 100
    assert "mean_seconds" in manifest
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
        store_kind = "current"
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


async def test_an_answering_arm_without_an_answer_fails_the_run(tmp_path, monkeypatch):
    class Silent:
        name = "silent"
        store_kind = "current"
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


# -- the agent arms --------------------------------------------------------


@pytest.mark.parametrize("name", ["closed_book", "v2_rag", "v2_agent", "v2_agent_lex"])
async def test_agent_arms_refuse_a_missing_driver_model(name):
    with pytest.raises(ValueError, match="driver-model"):
        await _ARMS[name].rank(object(), QUESTIONS[0], object())


async def test_v2_rag_answers_from_the_top_ten_after_rerank(monkeypatch):
    hits = [SimpleNamespace(path=f"p{i}") for i in range(12)]

    class Store:
        def round0(self, text, *, query_embedding, use_exact):
            assert use_exact is False
            return hits

        async def rerank(self, text, hs, *, reranker_model):
            return list(reversed(hs))

    async def answer_from_hits(question, hs, *, model, store=None, model_settings):
        assert store is not None  # the arm passes its store, so citations are checked
        assert len(hs) == 10 and model == "m"
        assert model_settings == {"temperature": 0.0}
        return SimpleNamespace(text="Jane Doe"), SimpleNamespace(
            requests=1, input_tokens=50, output_tokens=5
        )

    _fake_agent_module(monkeypatch, answer_from_hits=answer_from_hits)

    class Emb:
        async def get_or_compute(self, texts):
            return [[0.0] for _ in texts]

    out = await _ARMS["v2_rag"].rank(
        SimpleNamespace(store=Store()),
        QUESTIONS[0],
        Emb(),
        driver_model="m",
    )
    assert isinstance(out, ArmResult)
    assert out.answer == "Jane Doe" and out.llm_calls == 1 and out.tokens_in == 50
    assert out.entered_loop is False
    assert out.ranked[0] == "p11"


async def test_v2_agent_maps_the_result_and_passes_cached_embedder(monkeypatch):
    seen = {}

    async def run_agentic(
        question,
        *,
        store,
        model,
        embed_fn,
        reranker_model,
        run_id,
        budget,
        model_settings,
    ):
        seen.update(
            embed_fn=embed_fn, run_id=run_id, model=model, settings=model_settings
        )
        return SimpleNamespace(
            answer=SimpleNamespace(text="Jane Doe"),
            ranked_paths=["p2", "p1"],
            llm_calls=5,
            tokens_in=900,
            tokens_out=80,
            seconds=1.0,
            entered_loop=True,
            trace=[{"step": 0}],
        )

    _fake_agent_module(monkeypatch, run_agentic=run_agentic)

    class Emb:
        async def get_or_compute(self, texts):
            return [[0.0] for _ in texts]

    emb = Emb()
    out = await _ARMS["v2_agent"].rank(
        SimpleNamespace(store=object()),
        QUESTIONS[0],
        emb,
        driver_model="m",
        run_id="r:q0",
    )
    assert seen["embed_fn"] == emb.get_or_compute and seen["run_id"] == "r:q0"
    assert seen["settings"] == {"temperature": 0.0}
    assert (
        out.ranked == ["p2", "p1"] and out.entered_loop is True and out.llm_calls == 5
    )
    assert out.trace == [{"step": 0}]


async def test_v2_agent_lex_runs_without_an_embedder(monkeypatch):
    seen = {}

    async def run_agentic(question, *, embed_fn, **_kw):
        seen["embed_fn"] = embed_fn
        return SimpleNamespace(
            answer=SimpleNamespace(text="Jane Doe"),
            ranked_paths=["p1"],
            llm_calls=2,
            tokens_in=10,
            tokens_out=1,
            seconds=0.1,
            entered_loop=False,
            trace=[],
        )

    _fake_agent_module(monkeypatch, run_agentic=run_agentic)
    arm = _ARMS["v2_agent_lex"]
    assert arm.needs_embeddings is False
    out = await arm.rank(
        SimpleNamespace(store=object()), QUESTIONS[0], None, driver_model="m"
    )
    assert seen["embed_fn"] is None and out.answer == "Jane Doe"


async def test_closed_book_answers_without_a_store(monkeypatch):
    async def answer_closed_book(question, *, model, model_settings):
        assert model == "m" and model_settings == {"temperature": 0.0}
        return SimpleNamespace(text="Jane Doe"), SimpleNamespace(
            requests=1, input_tokens=20, output_tokens=3
        )

    _fake_agent_module(monkeypatch, answer_closed_book=answer_closed_book)
    out = await _ARMS["closed_book"].rank(
        arms.BuiltPool(["p1"]), QUESTIONS[0], None, driver_model="m"
    )
    assert out.ranked == [] and out.answer == "Jane Doe" and out.llm_calls == 1
