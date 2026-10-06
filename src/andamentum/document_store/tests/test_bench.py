"""Tests for the retrieval benchmark (``document_store.bench``).

All data is synthetic and every embedding comes from a deterministic fake, so
no test needs the network or a model. The test that matters most is the
equivalence one: the benchmark builds stores through its own copy of the
ingest path (to read vectors from a cache), and that copy must write exactly
what ``public.ingest`` writes, or the benchmark measures a store nobody runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from andamentum.document_store import public
from andamentum.document_store.api import DocumentStore
from andamentum.document_store.bench import arms, data, runner
from andamentum.document_store.bench.data import Corpus, Passage, Question
from andamentum.document_store.bench.embedding_cache import EmbeddingCache
from andamentum.document_store.bench.metrics import paired_bootstrap, question_metrics
from andamentum.document_store.connection import get_connection

DIM = 768


def _fake_vector(text: str) -> list[float]:
    seed = hashlib.sha256(text.encode()).digest()
    return [((seed[i % 32] + i) % 97) / 97.0 for i in range(DIM)]


class _FakeEmbedder:
    """Stands in for both OllamaEmbedder and the embedding_fn callable."""

    calls: list[int] = []

    def __init__(self, *_a, **_kw) -> None:
        pass

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        _FakeEmbedder.calls.append(len(texts))
        return [_fake_vector(t) for t in texts]

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def close(self) -> None:
        pass


async def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    return [_fake_vector(t) for t in texts]


def _cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(
        tmp_path / "emb.sqlite",
        model="fake-embed",
        digest="d1",
        embed_fn=_fake_embed_fn,
    )


# ── dataset normalisation ────────────────────────────────────────────────

HOTPOT_ROW = {
    "id": "q-apollo",
    "question": "Which river flows past the town where SaltLog was founded?",
    "type": "bridge",
    "supporting_facts": {"title": ["SaltLog", "Apollo River"], "sent_id": [0, 1]},
    "context": {
        "title": ["SaltLog", "Apollo River", "Quiet Hill"],
        "sentences": [
            ["SaltLog is a company.", " It was founded in Brightwater."],
            ["The Apollo River flows past Brightwater.", " It is long."],
            ["Quiet Hill is a hill.", " Nothing happens there."],
        ],
    },
}

MUSIQUE_ROW = {
    "id": "3hop1__1_2_3",
    "question": "Who founded the company whose logo is a salt crystal?",
    "paragraphs": [
        {
            "idx": 0,
            "title": "SaltLog",
            "paragraph_text": "SaltLog's logo is a salt crystal.",
            "is_supporting": True,
        },
        {
            "idx": 1,
            "title": "Jane Doe",
            "paragraph_text": "Jane Doe founded SaltLog.",
            "is_supporting": True,
        },
        {
            "idx": 2,
            "title": "Quiet Hill",
            "paragraph_text": "Quiet Hill is a hill.",
            "is_supporting": False,
        },
    ],
}


def test_parse_hotpotqa_marks_supporting_titles_as_gold():
    q, ps = data.parse_hotpotqa(HOTPOT_ROW)
    assert q.group == "bridge"
    assert [p.text for p in ps][
        0
    ] == "SaltLog is a company. It was founded in Brightwater."
    gold_titles = {p.title for p in ps if p.pid in q.gold}
    assert gold_titles == {"SaltLog", "Apollo River"}


def test_parse_musique_groups_by_hop_prefix():
    q, ps = data.parse_musique(MUSIQUE_ROW)
    assert q.group == "3hop1"
    assert {p.title for p in ps if p.pid in q.gold} == {"SaltLog", "Jane Doe"}


def test_pool_deduplicates_shared_passages_and_hash_is_order_free():
    q1, p1 = data.parse_musique(MUSIQUE_ROW)
    row2 = {**MUSIQUE_ROW, "id": "2hop__9_9"}
    q2, p2 = data.parse_musique(row2)
    records = {q1.qid: (q1, p1), q2.qid: (q2, p2)}
    a = data.pool("musique", "smoke", records, [q1.qid, q2.qid])
    b = data.pool("musique", "smoke", records, [q2.qid, q1.qid])
    assert len(a.passages) == 3  # the same three passages, once each
    assert a.sha256 == b.sha256


def _write_dataset(workdir: Path, rows: list[dict]) -> None:
    path = data.dataset_path("musique", workdir=workdir)
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _musique_rows(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"2hop__{i}",
                "question": f"What is item {i}?",
                "paragraphs": [
                    {
                        "idx": 0,
                        "title": f"Item {i}",
                        "paragraph_text": f"Item {i} is a widget.",
                        "is_supporting": True,
                    },
                    {
                        "idx": 1,
                        "title": f"Other {i}",
                        "paragraph_text": f"Other {i} is not.",
                        "is_supporting": False,
                    },
                ],
            }
        )
    return rows


def test_freeze_never_redraws_and_load_detects_changed_data(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "SUITES_DIR", tmp_path / "suites")
    monkeypatch.setattr(data, "TIERS", {"smoke": 2, "standard": 5})
    _write_dataset(tmp_path, _musique_rows(8))

    data.freeze("musique", workdir=tmp_path)
    with pytest.raises(FileExistsError):
        data.freeze("musique", workdir=tmp_path)

    corpus, spec = data.load_corpus("musique", "smoke", workdir=tmp_path)
    assert [q.qid for q in corpus.questions] == spec["sample_ids"][:2]

    rows = _musique_rows(8)
    for r in rows:
        r["paragraphs"][1]["paragraph_text"] += " Changed."
    data.dataset_path("musique", workdir=tmp_path).write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    with pytest.raises(ValueError, match="downloaded data changed"):
        data.load_corpus("musique", "smoke", workdir=tmp_path)


# ── metrics ──────────────────────────────────────────────────────────────


def test_question_metrics_on_a_known_ranking():
    m = question_metrics(["x", "g1", "y", "z", "w", "g2"], frozenset({"g1", "g2"}))
    assert m["recall@5"] == 0.5
    assert m["full_support@5"] == 0.0
    assert m["full_support@10"] == 1.0
    assert m["mrr"] == 0.5
    assert 0 < m["ndcg@10"] < 1


def test_paired_bootstrap_of_identical_runs_is_exactly_zero():
    assert paired_bootstrap([0.1, 0.5, 1.0], [0.1, 0.5, 1.0]) == (0.0, 0.0, 0.0)


# ── embedding cache ──────────────────────────────────────────────────────


async def test_cache_embeds_each_text_once_and_returns_identical_vectors(tmp_path):
    first = _cache(tmp_path)
    v1 = await first.get_or_compute(["alpha", "beta", "alpha"])
    assert first.misses == 2

    second = _cache(tmp_path)
    v2 = await second.get_or_compute(["beta", "alpha"])
    assert second.misses == 0
    assert v2 == [v1[1], v1[0]]  # the filling run sees the cached float32 values


async def test_cache_is_keyed_by_model_digest(tmp_path):
    await _cache(tmp_path).get_or_compute(["alpha"])
    other = EmbeddingCache(
        tmp_path / "emb.sqlite",
        model="fake-embed",
        digest="d2",
        embed_fn=_fake_embed_fn,
    )
    await other.get_or_compute(["alpha"])
    assert other.misses == 1


# ── the benchmark's ingest equals public.ingest ──────────────────────────

LONG_DOC = "# Apollo handbook\n\n" + "\n\n".join(
    f"## Part {i}\n\n" + " ".join(f"Apollo widget sentence {i}-{j}." for j in range(60))
    for i in range(6)
)


def _store_rows(db_path: Path) -> list[tuple]:
    with get_connection(db_path) as conn:
        chunks = conn.execute(
            "SELECT c.chunk_index, c.content, c.start_char, c.end_char, c.metadata, "
            "e.embedding FROM chunks c JOIN chunk_embeddings e ON e.chunk_id = c.id "
            "ORDER BY c.chunk_index"
        ).fetchall()
        doc_emb = conn.execute("SELECT embedding FROM doc_embeddings").fetchall()
    rows = []
    for idx, content, start, end, meta, emb in chunks:
        meta = json.loads(meta)
        meta.pop("parent_doc_id")  # a fresh uuid in each store
        rows.append((idx, content, start, end, meta, emb))
    return rows + [("doc", e[0]) for e in doc_emb]


@pytest.mark.parametrize(
    "text",
    ["A short passage about SaltLog.", LONG_DOC],
    ids=["one_chunk", "many_chunks"],
)
async def test_bench_ingest_writes_what_public_ingest_writes(
    tmp_path, monkeypatch, text
):
    passage = Passage(pid="p1", title="Apollo", text=text)

    monkeypatch.setattr(public, "OllamaEmbedder", _FakeEmbedder)
    monkeypatch.setattr(public, "make_ollama_embedder", lambda **_kw: _fake_embed_fn)
    monkeypatch.setattr(public, "_preflight", _no_preflight)
    reference = DocumentStore("reference", db_dir=tmp_path / "ref")
    await reference.initialize()
    monkeypatch.setattr(public, "_stores", {"reference": reference})
    await public.ingest(
        "reference",
        arms.document_content(passage),
        title=passage.title,
        embedding_model="fake-embed",
    )

    bench = DocumentStore("bench", db_dir=tmp_path / "bench")
    await bench.initialize()
    await arms._ingest_like(bench, passage, _cache(tmp_path))

    expected = _store_rows(reference.db_path)
    assert len(expected) >= 2
    assert _store_rows(bench.db_path) == expected


async def _no_preflight(*_a, **_kw) -> None:
    return None


# ── arms and the runner, end to end on a synthetic corpus ────────────────

PASSAGES = [
    Passage(
        pid="p-salt",
        title="SaltLog",
        text="SaltLog was founded in Brightwater by Jane Doe.",
    ),
    Passage(
        pid="p-river",
        title="Apollo River",
        text="The Apollo River flows past Brightwater.",
    ),
    Passage(pid="p-hill", title="Quiet Hill", text="Quiet Hill is a hill with sheep."),
]
QUESTIONS = [
    Question(
        qid="q1",
        text="Which river flows past the town where SaltLog was founded?",
        group="bridge",
        gold=frozenset({"p-salt", "p-river"}),
    ),
    Question(
        qid="q2",
        text='Which hill is called "Quiet"?',
        group="comparison",
        gold=frozenset({"p-hill"}),
    ),
]


def _corpus() -> Corpus:
    return Corpus(
        "synthetic", "smoke", QUESTIONS, PASSAGES, data.corpus_sha256(PASSAGES)
    )


async def test_bm25_or_arm_ranks_the_supporting_passages_first(tmp_path):
    built = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=None)
    ranked = await arms.ARMS["bm25_or"].rank(built, QUESTIONS[0], None)
    assert set(ranked[:2]) == {"p-salt", "p-river"}


async def test_store_is_built_once_and_reused(tmp_path):
    first = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=None)
    second = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=None)
    assert first.doc_to_pid == second.doc_to_pid


async def test_unified_arm_runs_on_cached_vectors(tmp_path):
    emb = _cache(tmp_path)
    built = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=emb)
    ranked = await arms.ARMS["unified"].rank(built, QUESTIONS[0], emb)
    assert set(ranked) <= {p.pid for p in PASSAGES}
    assert ranked


async def test_runner_records_a_rejected_store_query_instead_of_crashing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        runner, "load_corpus", lambda *_a, **_kw: (_corpus(), {"version": 1})
    )
    run_dir = await runner.run(
        suite="synthetic",
        tier="smoke",
        arm_name="fts5",
        workdir=tmp_path,
        embedding_model=None,
    )
    rows = [json.loads(line) for line in (run_dir / "questions.jsonl").open()]
    by_qid = {r["qid"]: r for r in rows}
    # The quoted question is passed to FTS5 raw and rejected (a known store bug).
    assert by_qid["q2"]["error"].startswith("OperationalError")
    assert by_qid["q2"]["ranked"] == []
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["query_errors"] == 1
    assert "1 query errors" in runner.format_summary(run_dir)


async def test_compare_refuses_runs_over_different_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "load_corpus", lambda *_a, **_kw: (_corpus(), {"version": 1})
    )
    a = await runner.run(
        suite="synthetic",
        tier="smoke",
        arm_name="bm25_or",
        workdir=tmp_path,
        embedding_model=None,
    )
    b = await runner.run(
        suite="synthetic",
        tier="smoke",
        arm_name="fts5",
        workdir=tmp_path / "b",
        embedding_model=None,
    )
    assert "B - A" in runner.compare(a, b)

    manifest = json.loads((b / "manifest.json").read_text())
    manifest["corpus_sha256"] = "other"
    (b / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="corpus_sha256"):
        runner.compare(a, b)


def test_frozen_suites_in_the_repo_are_well_formed():
    for suite in data.SOURCES:
        spec = json.loads(data.suite_file(suite).read_text())
        assert len(spec["sample_ids"]) == data.TIERS["standard"]
        assert len(set(spec["sample_ids"])) == len(spec["sample_ids"])
        assert set(spec["tiers"]) == set(data.TIERS)
