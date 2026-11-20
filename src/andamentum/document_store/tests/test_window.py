"""The ``since`` / ``until`` window over the time a document was first put."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from andamentum.document_store import (
    AgentBudget,
    AgentSearcher,
    HybridRerankSearcher,
    Store,
)

from . import test_search
from .test_search import _script, _store

#: The stub cross-encoder of the search tests, used here as a fixture.
stub_rerank = test_search.stub_rerank

PERTH = timezone(timedelta(hours=8))


def _set_created(store: Store, doc_id: str, at: datetime) -> None:
    """Back-date a document, written the way the store writes its own times."""
    store.db.execute(
        "UPDATE documents SET created_at = ? WHERE id = ?",
        (at.astimezone(timezone.utc).isoformat(timespec="microseconds"), doc_id),
    )


@pytest.fixture
async def dated(tmp_path):
    """Three notes, first put at 07:00, 09:00 and 11:00 Perth time on 19 November."""
    s = await Store.open(tmp_path / "s.db")
    ids = {}
    for hour in (7, 9, 11):
        doc_id = await s.put(f"# Note {hour}\n", metadata={"type": "note", "hour": hour})
        _set_created(s, doc_id, datetime(2026, 11, 19, hour, tzinfo=PERTH))
        ids[hour] = doc_id
    yield s, ids
    await s.close()


async def test_since_is_inclusive_and_until_is_exclusive(dated):
    s, ids = dated
    at9 = datetime(2026, 11, 19, 9, tzinfo=PERTH)
    at11 = datetime(2026, 11, 19, 11, tzinfo=PERTH)
    refs = await s.list(since=at9, until=at11, limit=None)
    assert [r.doc_id for r in refs] == [ids[9]]
    assert await s.count(since=at9) == 2
    assert await s.count(until=at9) == 1


async def test_two_adjacent_windows_never_share_a_document(dated):
    s, ids = dated
    edge = datetime(2026, 11, 19, 9, tzinfo=PERTH)
    before = {r.doc_id for r in await s.list(until=edge, limit=None)}
    after = {r.doc_id for r in await s.list(since=edge, limit=None)}
    assert before.isdisjoint(after) and before | after == set(ids.values())


async def test_the_same_moment_in_another_zone_gives_the_same_answer(dated):
    s, ids = dated
    perth = await s.list(since=datetime(2026, 11, 19, 9, tzinfo=PERTH), limit=None)
    utc = await s.list(since=datetime(2026, 11, 19, 1, tzinfo=timezone.utc), limit=None)
    assert [r.doc_id for r in perth] == [r.doc_id for r in utc]


async def test_an_update_does_not_move_a_document_into_the_window(dated):
    s, ids = dated
    await s.update(ids[7], metadata={"type": "note", "hour": 7, "read": True})
    refs = await s.list(since=datetime(2026, 11, 19, 8, tzinfo=PERTH), limit=None)
    assert ids[7] not in {r.doc_id for r in refs}
    (ref7,) = [r for r in await s.list(limit=None) if r.doc_id == ids[7]]
    assert ref7.rev == 2 and ref7.created_at < ref7.updated_at


async def test_the_window_and_where_combine(dated):
    s, ids = dated
    refs = await s.list(
        where={"hour": [7, 11]}, since=datetime(2026, 11, 19, 8, tzinfo=PERTH), limit=None
    )
    assert [r.doc_id for r in refs] == [ids[11]]


async def test_docref_carries_created_at(dated):
    s, ids = dated
    (ref,) = await s.list(where={"hour": 9}, limit=None)
    doc = await s.get(ids[9])
    assert doc is not None and ref.created_at == doc.created_at
    assert datetime.fromisoformat(ref.created_at) == datetime(2026, 11, 19, 9, tzinfo=PERTH)


async def test_a_naive_time_is_refused(dated):
    s, _ = dated
    with pytest.raises(ValueError, match="time zone"):
        await s.list(since=datetime(2026, 11, 19), limit=None)
    with pytest.raises(ValueError, match="time zone"):
        await s.count(until=datetime(2026, 11, 19))


async def test_since_must_be_before_until(dated):
    s, _ = dated
    t = datetime(2026, 11, 19, 9, tzinfo=PERTH)
    with pytest.raises(ValueError, match="before until"):
        await s.list(since=t, until=t, limit=None)


async def test_a_string_is_refused(dated):
    s, _ = dated
    with pytest.raises(TypeError, match="datetime"):
        await s.list(since="2026-11-19T09:00:00+08:00", limit=None)  # type: ignore[arg-type]


# --- search --------------------------------------------------------------------


async def _date_docs(s: Store, ids: dict[str, str]) -> datetime:
    """a.md and b.md are old; c.md and d.md (both about SaltLog) are new."""
    old = datetime(2026, 11, 18, 12, tzinfo=PERTH)
    new = datetime(2026, 11, 19, 12, tzinfo=PERTH)
    for k in ("a.md", "b.md"):
        _set_created(s, ids[k], old)
    for k in ("c.md", "d.md"):
        _set_created(s, ids[k], new)
    return datetime(2026, 11, 19, tzinfo=PERTH)


async def test_search_ranks_only_documents_in_the_window(tmp_path, stub_rerank):
    s, ix, ids = await _store(tmp_path)
    since = await _date_docs(s, ids)
    hits = await s.search("SaltLog Apollo", limit=10, since=since)
    assert hits and {h.doc_id for h in hits} <= {ids["c.md"], ids["d.md"]}
    hits = await s.search("SaltLog Apollo", limit=10, until=since)
    assert hits and {h.doc_id for h in hits} <= {ids["a.md"], ids["b.md"]}
    await s.close()


async def test_search_refuses_a_naive_time_before_it_syncs(tmp_path, stub_rerank):
    s, ix, ids = await _store(tmp_path)
    with pytest.raises(ValueError, match="time zone"):
        await s.search("SaltLog", limit=5, since=datetime(2026, 11, 19))
    assert stub_rerank == []  # nothing was ranked
    await s.close()


async def test_the_agent_searcher_and_its_tools_keep_to_the_window(tmp_path, stub_rerank):
    model = _script(
        [
            ("OUTPUT", {"enough": False, "missing": "more about Apollo"}),
            ("search", {"query_text": "Apollo crew moon"}),
            ("OUTPUT", {"passages": [{"doc": "c.md", "line_from": 1, "line_to": 3}]}),
        ]
    )
    s, ix, ids = await _store(
        tmp_path,
        searcher_factory=lambda i: AgentSearcher(
            model=model,
            inner=HybridRerankSearcher(index=i, reranker_model="stub"),
            budget=AgentBudget(max_tool_calls=3, max_requests=6),
        ),
    )
    since = await _date_docs(s, ids)
    hits = await s.search("SaltLog", limit=10, since=since)
    assert hits and {h.doc_id for h in hits} <= {ids["c.md"], ids["d.md"]}
    run = s.searcher.last_run  # type: ignore[union-attr]
    (tool,) = [t for t in run.trace if t["kind"] == "tool:search"]
    assert all(p.split(":")[0] in {"c.md", "d.md"} for p in tool["result_paths"])
    await s.close()
