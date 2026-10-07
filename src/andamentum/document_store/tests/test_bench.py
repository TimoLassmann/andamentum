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
import sqlite3
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
    "answer": "Apollo River",
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
    "answer": "Jane Doe",
    "answer_aliases": ["J. Doe"],
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
                "answer": f"item {i}",
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


def _shared_gold_rows(n_pairs: int) -> list[dict]:
    """Pairs of questions that share their supporting paragraph."""
    rows = []
    for i in range(n_pairs):
        for j in (0, 1):
            rows.append(
                {
                    "id": f"2hop__{i}_{j}",
                    "question": f"What is item {i}, asked way {j}?",
                    "answer": f"item {i}",
                    "paragraphs": [
                        {
                            "idx": 0,
                            "title": f"Item {i}",
                            "paragraph_text": f"Item {i} is a widget.",
                            "is_supporting": True,
                        },
                        {
                            "idx": 1,
                            "title": f"Other {i} {j}",
                            "paragraph_text": f"Other {i} {j} is not.",
                            "is_supporting": False,
                        },
                    ],
                }
            )
    return rows


def _small_splits(monkeypatch, *, dev: int, test: int, smoke: int = 2) -> None:
    monkeypatch.setattr(data, "SPLITS", {"dev": dev, "test": test})
    monkeypatch.setattr(data, "SMOKE_SIZE", smoke)


def test_freeze_never_redraws_and_load_detects_changed_data(tmp_path, monkeypatch):
    _small_splits(monkeypatch, dev=3, test=3)
    suites = tmp_path / "suites"
    _write_dataset(tmp_path, _musique_rows(12))

    data.freeze("musique", workdir=tmp_path, suites_dir=suites)
    with pytest.raises(FileExistsError):
        data.freeze("musique", workdir=tmp_path, suites_dir=suites)

    corpus, spec = data.load_corpus(
        "musique", "smoke", workdir=tmp_path, suites_dir=suites
    )
    assert [q.qid for q in corpus.questions] == spec["dev_ids"][:2]
    assert all(q.cluster == spec["groups"][q.qid] for q in corpus.questions)
    test, _ = data.load_corpus("musique", "test", workdir=tmp_path, suites_dir=suites)
    assert [q.qid for q in test.questions] == spec["test_ids"]

    rows = _musique_rows(12)
    for r in rows:
        r["paragraphs"][1]["paragraph_text"] += " Changed."
    data.dataset_path("musique", workdir=tmp_path).write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    with pytest.raises(ValueError, match="downloaded data changed"):
        data.load_corpus("musique", "smoke", workdir=tmp_path, suites_dir=suites)


def test_linked_questions_never_span_dev_and_test(tmp_path, monkeypatch):
    _small_splits(monkeypatch, dev=4, test=4)
    suites = tmp_path / "suites"
    _write_dataset(tmp_path, _shared_gold_rows(10))
    data.freeze("musique", workdir=tmp_path, suites_dir=suites)
    spec = data.load_spec("musique", suites_dir=suites)

    def pair(qid: str) -> str:
        return qid.rsplit("_", 1)[0]

    dev_pairs = {pair(q) for q in spec["dev_ids"]}
    test_pairs = {pair(q) for q in spec["test_ids"]}
    assert not dev_pairs & test_pairs
    for qid, group in spec["groups"].items():
        assert group == f"{pair(qid)}_0"  # each pair is one linked group


def test_link_groups_joins_questions_that_share_a_supporting_passage():
    questions = [data.parse_musique(r)[0] for r in _shared_gold_rows(2)]
    groups = data.link_groups(reversed(questions))
    assert groups == {
        "2hop__0_0": "2hop__0_0",
        "2hop__0_1": "2hop__0_0",
        "2hop__1_0": "2hop__1_0",
        "2hop__1_1": "2hop__1_0",
    }


def test_check_splits_refuses_a_shared_supporting_passage():
    records = {}
    for row in _shared_gold_rows(1):
        q, ps = data.parse_musique(row)
        records[q.qid] = (q, ps)
    spec = {
        "dev_ids": ["2hop__0_0"],
        "test_ids": ["2hop__0_1"],
        "groups": {"2hop__0_0": "a", "2hop__0_1": "b"},  # wrongly split
    }
    with pytest.raises(ValueError, match="supporting passages"):
        data.check_splits(spec, records)
    spec["groups"] = {"2hop__0_0": "a", "2hop__0_1": "a"}
    with pytest.raises(ValueError, match="linked groups"):
        data.check_splits(spec, records)


def test_an_old_frozen_sample_is_refused(tmp_path):
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / "musique.json").write_text(json.dumps({"version": 1}))
    with pytest.raises(ValueError, match="freeze it again"):
        data.load_spec("musique", suites_dir=suites)


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


def test_cluster_bootstrap_resamples_whole_clusters():
    a = [0.0] * 8
    b = [1.0] * 4 + [0.0] * 4
    _, lo_q, hi_q = paired_bootstrap(a, b)
    # Two clusters that each hold one value: every resample mean is 0, 0.5 or 1.
    _, lo_c, hi_c = paired_bootstrap(a, b, clusters=["x"] * 4 + ["y"] * 4)
    assert (lo_c, hi_c) == (0.0, 1.0)
    assert hi_c - lo_c > hi_q - lo_q


def test_bootstrap_p_value_is_small_for_a_clear_difference_and_one_for_none():
    from andamentum.document_store.bench.metrics import paired_bootstrap_test

    _, _, _, p_clear = paired_bootstrap_test([0.0] * 40, [1.0] * 30 + [0.0] * 10)
    assert p_clear < 0.001
    _, _, _, p_none = paired_bootstrap_test([0.5] * 10, [0.5] * 10)
    assert p_none == 1.0


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


def _ranked(result) -> list[str]:
    return result.ranked if isinstance(result, arms.ArmResult) else result


def _corpus() -> Corpus:
    return Corpus(
        "synthetic", "smoke", QUESTIONS, PASSAGES, data.corpus_sha256(PASSAGES)
    )


async def test_bm25_arm_ranks_the_supporting_passages_first(tmp_path):
    built = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=None)
    ranked = _ranked(await arms.ARMS["bm25"].rank(built, QUESTIONS[0], None))
    assert set(ranked[:2]) == {"p-salt", "p-river"}


async def test_store_is_built_once_and_reused(tmp_path):
    first = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=None)
    second = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=None)
    assert first.doc_to_pid == second.doc_to_pid
    assert first.info is not None and second.info is not None
    assert first.info.reused is False and second.info.reused is True
    assert second.info.build["layout"] == arms.CURRENT_LAYOUT


@pytest.mark.parametrize("builder", ["build_current_store", "build_v2_store"])
async def test_a_store_built_for_another_layout_is_refused(tmp_path, builder):
    build = getattr(arms, builder)
    first = await build(_corpus(), workdir=tmp_path, emb=None)
    assert first.info is not None
    meta_path = next((tmp_path / "stores").glob("*.meta.json"))
    meta = json.loads(meta_path.read_text())
    meta["layout"] = 0
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="built for a different run"):
        await build(_corpus(), workdir=tmp_path, emb=None)


async def test_v2_store_has_the_agent_tables(tmp_path):
    built = await arms.build_v2_store(_corpus(), workdir=tmp_path, emb=None)
    with sqlite3.connect(built.store.path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    assert {"collections", "evidence", "decisions"} <= tables


async def test_store_names_carry_the_embedding_digest(tmp_path):
    emb = _cache(tmp_path)
    await arms.build_v2_store(_corpus(), workdir=tmp_path, emb=emb)
    [meta] = (tmp_path / "stores").glob("*.meta.json")
    assert emb.digest in meta.name and f"v2_{arms.V2_LAYOUT}" in meta.name


async def test_v1_arm_runs_on_cached_vectors(tmp_path):
    emb = _cache(tmp_path)
    built = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=emb)
    ranked = _ranked(await arms.ARMS["v1"].rank(built, QUESTIONS[0], emb))
    assert set(ranked) <= {p.pid for p in PASSAGES}
    assert ranked


async def test_v1_rerank_arm_reorders_the_pool_by_reranker_score(tmp_path, monkeypatch):
    from andamentum.document_store import reranking

    seen: dict[str, list[str]] = {}

    async def fake_rerank(query, texts, *, reranker_model):
        seen["texts"] = list(texts)
        seen["model"] = [reranker_model]
        # The passage about hills scores highest, everything else keeps order.
        return [1.0 if "hill" in t.lower() else 0.0 for t in texts]

    monkeypatch.setattr(reranking, "rerank", fake_rerank)
    emb = _cache(tmp_path)
    await arms.build_current_store(_corpus(), workdir=tmp_path, emb=emb)
    # Reopened from disk: the passage texts must still be there.
    built = await arms.build_current_store(_corpus(), workdir=tmp_path, emb=emb)
    arm = arms.ARMS["v1_rerank"]
    plain = _ranked(await arms.ARMS["v1"].rank(built, QUESTIONS[0], emb))
    ranked = _ranked(await arm.rank(built, QUESTIONS[0], emb))

    texts = {arms.document_content(p) for p in PASSAGES}
    assert seen["texts"] and set(seen["texts"]) <= texts
    assert seen["model"] == [arms.SUGGESTED_RERANKER_MODEL]
    assert ranked[0] == "p-hill"
    assert set(ranked) == set(plain)


async def test_ollama_cache_sends_one_embedding_request_at_a_time(
    tmp_path, monkeypatch
):
    import asyncio

    from andamentum.document_store.bench import embedding_cache

    state = {"in_flight": 0, "max": 0}

    class SlowEmbedder(_FakeEmbedder):
        async def embed_batch(self, texts):
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
            await asyncio.sleep(0.01)
            state["in_flight"] -= 1
            return [_fake_vector(t) for t in texts]

    monkeypatch.setattr(embedding_cache, "OllamaEmbedder", SlowEmbedder)
    monkeypatch.setattr(embedding_cache, "ollama_model_digest", lambda m: "d1")
    cache = EmbeddingCache.for_ollama(tmp_path / "emb.sqlite", model="fake-embed")
    results = await asyncio.gather(
        *(cache.get_or_compute([f"text {i}"]) for i in range(6))
    )
    assert state["max"] == 1
    float32 = embedding_cache._unpack
    assert results == [
        [float32(embedding_cache._pack(_fake_vector(f"text {i}")))] for i in range(6)
    ]


_CLEAN_GIT = {"commit": "c0ffee", "dirty": False, "diff_sha256": None}


@pytest.fixture
def synthetic_runs(monkeypatch):
    monkeypatch.setattr(
        runner, "load_corpus", lambda *_a, **_kw: (_corpus(), {"version": 2})
    )
    monkeypatch.setattr(runner, "git_state", lambda: dict(_CLEAN_GIT))


async def _run_arm(tmp_path, name, out="run"):
    return await runner.run(
        suite="synthetic",
        tier="smoke",
        arm_name=name,
        workdir=tmp_path,
        out_dir=tmp_path / out,
        embedding_model=None,
    )


async def test_any_error_fails_the_run_and_leaves_a_failed_record(
    tmp_path, monkeypatch, synthetic_runs
):
    class BrokenArm:
        name = "broken"
        store_kind = "none"
        needs_embeddings = False
        answers = False

        async def rank(self, built, q, emb, **_kw):
            if q.qid == "q2":
                raise sqlite3.OperationalError("no such table: decisions")
            return ["p-salt"]

    monkeypatch.setitem(runner.ARMS, "broken", BrokenArm())
    with pytest.raises(sqlite3.OperationalError):
        await _run_arm(tmp_path, "broken")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "no such table" in manifest["error"]
    assert not (tmp_path / "run" / "summary.json").exists()
    with pytest.raises(ValueError, match="did not complete"):
        runner.load_run(tmp_path / "run")


async def test_a_ranking_outside_the_pool_fails_the_run(
    tmp_path, monkeypatch, synthetic_runs
):
    class Inventing:
        name = "inventing"
        store_kind = "none"
        needs_embeddings = False
        answers = False

        async def rank(self, built, q, emb, **_kw):
            return ["p-salt", "p-invented"]

    monkeypatch.setitem(runner.ARMS, "inventing", Inventing())
    with pytest.raises(ValueError, match="not in the pool"):
        await _run_arm(tmp_path, "inventing")


async def test_a_run_never_overwrites_another(tmp_path, synthetic_runs):
    await _run_arm(tmp_path, "oracle")
    with pytest.raises(FileExistsError):
        await _run_arm(tmp_path, "oracle")


async def test_controls_score_as_expected(tmp_path, synthetic_runs):
    oracle = await _run_arm(tmp_path, "oracle", "oracle")
    summary = json.loads((oracle / "summary.json").read_text())["all"]
    assert summary["full_support@10"] == 1.0 and summary["mrr"] == 1.0
    shuffled = await _run_arm(tmp_path, "shuffled", "shuffled")
    rows = [json.loads(line) for line in (shuffled / "questions.jsonl").open()]
    assert all(sorted(r["ranked"]) == sorted(p.pid for p in PASSAGES) for r in rows)


async def test_manifest_records_status_store_and_provenance(tmp_path, synthetic_runs):
    run_dir = await _run_arm(tmp_path, "bm25")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["started"] and manifest["ended"]
    assert manifest["store_file"]["layout"] == arms.CURRENT_LAYOUT
    assert manifest["store_file"]["reused"] is False
    assert manifest["git"] == _CLEAN_GIT
    assert "uv_lock_sha256" in manifest
    row = runner.table_row(run_dir)
    assert row["arm"] == "bm25" and row["commit"] == "c0ffee"
    summary = json.loads((run_dir / "summary.json").read_text())["all"]
    assert row["full_support@10"] == summary["full_support@10"]


async def test_compare_refuses_runs_that_differ_in_what_must_match(
    tmp_path, synthetic_runs
):
    a = await _run_arm(tmp_path, "bm25", "a")
    b = await _run_arm(tmp_path, "oracle", "b")
    assert "B - A" in runner.compare(a, b)
    c = runner.contrast(a, b, "full_support@10")
    assert c["mean_b"] == 1.0 and c["n"] == len(QUESTIONS)

    manifest = json.loads((b / "manifest.json").read_text())
    for key, value, message in (
        ("corpus_sha256", "other", "corpus_sha256"),
        ("git", {**_CLEAN_GIT, "commit": "other"}, "git commit"),
        ("git", {**_CLEAN_GIT, "dirty": True}, "uncommitted code"),
    ):
        changed = {**manifest, key: value}
        (b / "manifest.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            runner.compare(a, b)


def test_frozen_suites_in_the_repo_are_well_formed():
    for suite in data.SOURCES:
        spec = data.load_spec(suite)
        assert len(spec["dev_ids"]) == data.SPLITS["dev"]
        assert len(spec["test_ids"]) == data.SPLITS["test"]
        ids = spec["dev_ids"] + spec["test_ids"]
        assert len(set(ids)) == len(ids)
        assert set(spec["tiers"]) == set(data.RUN_TIERS)
        assert set(spec["groups"]) == set(ids)
        dev_groups = {spec["groups"][q] for q in spec["dev_ids"]}
        assert not dev_groups & {spec["groups"][q] for q in spec["test_ids"]}
