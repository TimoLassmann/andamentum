"""The disposable index, sync, the default searcher and the agent searcher.

No model runs here: embeddings come from a deterministic stub, the reranker
is monkeypatched, and the agent is scripted with a pydantic-ai FunctionModel.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from andamentum.document_store import (
    AgentBudget,
    AgentSearcher,
    HybridIndex,
    HybridRerankSearcher,
    Store,
    group_by_document,
    rrf,
)
from andamentum.document_store import reranking

_VOCAB = ["apollo", "saltlog", "garden", "crew", "salt", "moon", "jane"]


async def stub_embed(texts: list[str]) -> list[list[float]]:
    """A bag-of-words vector over a tiny vocabulary: similar words, similar vectors."""
    out = []
    for t in texts:
        words = re.findall(r"\w+", t.lower())
        out.append([float(words.count(v)) + 0.01 for v in _VOCAB])
    return out


@pytest.fixture
def stub_rerank(monkeypatch):
    """Score = how many query words the passage holds; records each call."""
    calls: list[tuple[str, list[str]]] = []

    async def rerank(query, texts, *, reranker_model):
        calls.append((query, list(texts)))
        q = set(re.findall(r"\w+", query.lower()))
        return [len(q & set(re.findall(r"\w+", t.lower()))) / 10 for t in texts]

    monkeypatch.setattr(reranking, "rerank", rerank)
    return calls


DOCS = {
    "a.md": "# Apollo\n\nApollo launched in 1969 toward the moon.\n\n## Crew\n\nThe crew of Apollo numbered three.",
    "b.md": "# Garden\n\nUnrelated notes about the garden and its soil.",
    "c.md": "# SaltLog\n\nThe SaltLog service stores salt records.",
    "d.md": "# Design\n\nJane Doe wrote the SaltLog design in NM_000256.3.",
}


async def _store(tmp_path, *, embed=True, searcher_factory=None):
    ix = HybridIndex(
        embedding_model="stub" if embed else None,
        embed_fn=stub_embed if embed else None,
        max_chars=200,
    )
    searcher = (searcher_factory or (lambda i: HybridRerankSearcher(index=i, reranker_model="stub")))(ix)
    s = await Store.open(tmp_path / "s.db", indexes=[ix], searcher=searcher)
    ids = {k: await s.put(t, key=k, metadata={"title": k.upper()}) for k, t in DOCS.items()}
    return s, ix, ids


async def test_sync_builds_chunks_with_offsets_paths_and_prefixes(tmp_path):
    s, ix, ids = await _store(tmp_path)
    report = await s.sync()
    assert report.ok and report.processed == {"hybrid": 4}
    chunks = ix.document_chunks(s.db, ids["a.md"])
    doc = await s.get(ids["a.md"])
    assert doc is not None
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text
    assert chunks[-1].heading_path == ("Apollo", "Crew")
    assert chunks[-1].prefix == "A.MD > Apollo > Crew"
    await s.close()


async def test_sync_is_incremental_and_follows_updates_and_deletes(tmp_path):
    s, ix, ids = await _store(tmp_path)
    await s.sync()
    assert (await s.sync()).processed == {"hybrid": 0}
    await s.update(ids["b.md"], text="# Garden\n\nNow about the moon.")
    await s.delete(ids["c.md"])
    report = await s.sync()
    assert report.processed == {"hybrid": 2}
    assert ix.bm25(s.db, "soil", k=10) == []
    assert ix.bm25(s.db, "stores", k=10) == []
    assert ix.document_chunks(s.db, ids["c.md"]) == []
    await s.restore(ids["c.md"])
    await s.sync()
    assert ix.document_chunks(s.db, ids["c.md"])
    await s.close()


async def test_a_metadata_only_update_does_not_re_embed(tmp_path):
    calls: list[int] = []

    async def counting(texts):
        calls.append(len(texts))
        return await stub_embed(texts)

    ix = HybridIndex(embedding_model="stub", embed_fn=counting)
    s = await Store.open(tmp_path / "s.db", indexes=[ix])
    doc_id = await s.put("# T\n\nBody.", metadata={"title": "T", "x": 1})
    await s.sync()
    n = len(calls)
    await s.update(doc_id, metadata={"title": "T", "x": 2})  # same title, same text
    await s.sync()
    assert len(calls) == n
    assert ix.document_chunks(s.db, doc_id)[0].rev == 2
    await s.update(doc_id, metadata={"title": "New title"})  # the prefix changes
    await s.sync()
    assert len(calls) == n + 1
    await s.close()


async def test_a_config_change_rebuilds_the_index(tmp_path):
    path = tmp_path / "s.db"
    s = await Store.open(path, indexes=[HybridIndex(embedding_model=None, max_chars=2000)])
    await s.put("# T\n\n" + "Word. " * 100)
    await s.sync()
    n_big = s.db.execute("SELECT COUNT(*) FROM ix_hybrid_chunks").fetchone()[0]
    await s.close()
    s = await Store.open(path, indexes=[HybridIndex(embedding_model=None, max_chars=100)])
    assert (await s.sync()).processed == {"hybrid": 1}
    assert s.db.execute("SELECT COUNT(*) FROM ix_hybrid_chunks").fetchone()[0] > n_big
    await s.close()


async def test_an_embedding_service_failure_raises_and_keeps_the_position(tmp_path):
    async def down(texts):
        raise httpx.ConnectError("connection refused")

    ix = HybridIndex(embedding_model="stub", embed_fn=down)
    s = await Store.open(tmp_path / "s.db", indexes=[ix])
    await s.put("# T\n\nBody.")
    with pytest.raises(httpx.ConnectError):
        await s.sync()
    assert s.db.execute("SELECT last_seq FROM indexes").fetchone()[0] == 0
    await s.close()


async def test_a_document_that_cannot_be_chunked_is_recorded_and_search_refuses(tmp_path, monkeypatch, stub_rerank):
    s, ix, ids = await _store(tmp_path)
    from andamentum.document_store import index as index_module

    real = index_module.chunk

    def picky(text, *, max_chars):
        if "Garden" in text:
            raise RuntimeError("cannot chunk this")
        return real(text, max_chars=max_chars)

    monkeypatch.setattr(index_module, "chunk", picky)
    report = await s.sync()
    assert not report.ok
    assert report.failures["hybrid"] == [(ids["b.md"], "RuntimeError: cannot chunk this")]
    with pytest.raises(RuntimeError, match="could not be indexed"):
        await s.search("apollo", limit=3)
    monkeypatch.setattr(index_module, "chunk", real)
    assert (await s.sync()).ok  # the failure is retried
    await s.close()


def test_rrf_is_deterministic():
    assert rrf([[1, 2, 3], [3, 2]]) == rrf([[1, 2, 3], [3, 2]])
    assert [cid for cid, _ in rrf([[1, 2], [2, 1]])] == [1, 2]  # tie: smaller id first


async def test_default_search_reranks_and_returns_valid_spans(tmp_path, stub_rerank):
    s, ix, ids = await _store(tmp_path)
    hits = await s.search("Apollo crew", limit=3)
    assert hits and hits[0].doc_id == ids["a.md"]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    for h in hits:
        assert await s.read(h.doc_id, rev=h.rev, start=h.start, end=h.end) == h.text
    query, texts = stub_rerank[0]
    assert query == "Apollo crew"
    assert any(t.startswith("A.MD > Apollo") for t in texts)  # the reranker sees the prefix
    await s.close()


async def test_search_where_restricts_documents(tmp_path, stub_rerank):
    s, ix, ids = await _store(tmp_path)
    await s.update(ids["d.md"], metadata={"title": "D.MD", "team": "x"})
    hits = await s.search("SaltLog", limit=5, where={"team": "x"})
    assert {h.doc_id for h in hits} == {ids["d.md"]}
    await s.close()


async def test_search_without_a_searcher_raises(tmp_path):
    s = await Store.open(tmp_path / "s.db")
    with pytest.raises(RuntimeError, match="without a searcher"):
        await s.search("x", limit=1)
    await s.close()


async def test_a_searcher_must_read_an_index_the_store_has(tmp_path):
    ix = HybridIndex(embedding_model=None)
    other = HybridIndex(embedding_model=None)
    with pytest.raises(ValueError, match="was not passed"):
        await Store.open(
            tmp_path / "s.db",
            indexes=[ix],
            searcher=HybridRerankSearcher(index=other, reranker_model="stub"),
        )


async def test_lexical_index_searches_without_vectors(tmp_path, stub_rerank):
    s, ix, ids = await _store(tmp_path, embed=False)
    hits = await s.search("garden soil", limit=2)
    assert hits[0].doc_id == ids["b.md"]
    await s.close()


async def test_grep_and_exact_use_the_trigram_index(tmp_path):
    s, ix, ids = await _store(tmp_path)
    await s.sync()
    found = ix.grep(s.db, r"NM_\d+\.\d", limit=5)
    assert [c.doc_id for c, _ in found] == [ids["d.md"]]
    assert ix.exact(s.db, "what is NM_000256.3", k=5)
    with pytest.raises(ValueError):
        ix.grep(s.db, "(", limit=5)
    await s.close()


# ── the agent searcher ─────────────────────────────────────────────────


def _script(steps):
    """A FunctionModel that returns the scripted responses in order.

    Each step is ``(tool_name, args)``; the special name ``"OUTPUT"`` calls
    the run's output tool (the stop decision or the selection).
    """
    queue = list(steps)

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name, args = queue.pop(0)
        if name == "OUTPUT":
            name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(name, json.dumps(args))])

    return FunctionModel(fn)


async def test_agent_stops_after_round0_when_enough(tmp_path, stub_rerank):
    model = _script([("OUTPUT", {"enough": True})])
    s, ix, ids = await _store(
        tmp_path,
        searcher_factory=lambda i: AgentSearcher(
            model=model, inner=HybridRerankSearcher(index=i, reranker_model="stub")
        ),
    )
    hits = await s.search("Apollo crew", limit=5)
    assert hits[0].doc_id == ids["a.md"]
    assert s.searcher.last_run.entered_loop is False  # type: ignore[union-attr]
    await s.close()


async def test_agent_loop_searches_reads_and_selects(tmp_path, stub_rerank):
    model = _script(
        [
            ("OUTPUT", {"enough": False, "missing": "who wrote the SaltLog design"}),
            ("search", {"query_text": "SaltLog design author"}),
            ("outline", {"doc": "d.md"}),
            ("read", {"doc": "d.md", "offset": 1, "limit": 5}),
            ("note", {"doc": "d.md", "line_from": 3, "line_to": 3, "claim": "Jane Doe wrote it"}),
            ("OUTPUT", {"passages": [{"doc": "d.md", "line_from": 1, "line_to": 3}]}),
        ]
    )
    s, ix, ids = await _store(
        tmp_path,
        searcher_factory=lambda i: AgentSearcher(
            model=model,
            inner=HybridRerankSearcher(index=i, reranker_model="stub"),
            budget=AgentBudget(max_tool_calls=5, max_requests=8),
        ),
    )
    hits = await s.search("Who wrote the design of the service that stores salt?", limit=10)
    run = s.searcher.last_run  # type: ignore[union-attr]
    assert run.entered_loop is True
    assert hits[0].doc_id == ids["d.md"]  # the selection comes first
    kinds = [step["kind"] for step in run.trace]
    assert kinds[:2] == ["round0", "stop_after_round0"]
    assert "tool:search" in kinds and "selection" in kinds
    # every passage scored against the original request, as in the default searcher
    assert all(q == "Who wrote the design of the service that stores salt?" or q == "SaltLog design author" for q, _ in stub_rerank)
    assert [d.doc_id for d in group_by_document(hits)][0] == ids["d.md"]
    await s.close()


async def test_agent_rejects_a_selection_of_passages_it_never_saw(tmp_path, stub_rerank):
    model = _script(
        [
            ("OUTPUT", {"enough": False}),
            ("OUTPUT", {"passages": [{"doc": "nope.md", "line_from": 1, "line_to": 2}]}),
            ("OUTPUT", {"passages": [{"doc": "a.md", "line_from": 1, "line_to": 3}]}),
        ]
    )
    s, ix, ids = await _store(
        tmp_path,
        searcher_factory=lambda i: AgentSearcher(
            model=model, inner=HybridRerankSearcher(index=i, reranker_model="stub")
        ),
    )
    hits = await s.search("Apollo", limit=5)
    assert hits[0].doc_id == ids["a.md"]
    await s.close()


async def test_agent_tool_errors_go_back_to_the_model(tmp_path, stub_rerank):
    model = _script(
        [
            ("OUTPUT", {"enough": False}),
            ("grep", {"regex": "("}),
            ("OUTPUT", {"passages": []}),
        ]
    )
    s, ix, ids = await _store(
        tmp_path,
        searcher_factory=lambda i: AgentSearcher(
            model=model, inner=HybridRerankSearcher(index=i, reranker_model="stub")
        ),
    )
    await s.search("Apollo", limit=5)
    errors = [st for st in s.searcher.last_run.trace if "error" in st]  # type: ignore[union-attr]
    assert errors and errors[0]["kind"] == "tool:grep"
    await s.close()


def test_agent_budget_must_leave_room_to_finish():
    with pytest.raises(ValueError):
        AgentBudget(max_tool_calls=12, max_requests=12)
