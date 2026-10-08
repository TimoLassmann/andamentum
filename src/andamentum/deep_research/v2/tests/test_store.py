"""The research store: cache, works, passages, runs, blobs, schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from andamentum.deep_research.v2.store import (
    INLINE_BLOB_LIMIT,
    SCHEMA_VERSION,
    ResearchStore,
    StoreError,
    cache_key,
)
from andamentum.deep_research.v2.types import Finding, Passage
from andamentum.references import LICENCE_NO_REDISTRIBUTE, Work


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock: Clock) -> ResearchStore:
    return ResearchStore.in_memory(clock=clock)


# -- schema and persistence ---------------------------------------------------


def test_a_new_store_is_at_the_current_schema_version(store: ResearchStore) -> None:
    assert store.schema_version() == SCHEMA_VERSION


def test_a_persistent_store_uses_wal_and_keeps_its_data(tmp_path: Path) -> None:
    path = tmp_path / "lab" / "research.db"
    with ResearchStore(path) as first:
        stored = first.upsert_work(Work(doi="10.1001/a", title="Alpha"))
    assert path.exists()
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with ResearchStore(path) as second:
        assert second.get_work(stored.work_id) == stored
        assert second.schema_version() == SCHEMA_VERSION


def test_an_in_memory_store_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with ResearchStore.in_memory() as memory:
        memory.upsert_work(Work(doi="10.1001/a"))
        memory.put_blob(b"x" * (INLINE_BLOB_LIMIT + 1))
    assert list(tmp_path.iterdir()) == []


def test_a_database_from_a_newer_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "research.db"
    ResearchStore(path).close()
    with sqlite3.connect(path) as raw:
        raw.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION + 1,))
    with pytest.raises(StoreError, match="schema is version"):
        ResearchStore(path)


# -- blobs ----------------------------------------------------------------


def test_small_blobs_are_inline_and_large_ones_are_files(tmp_path: Path) -> None:
    with ResearchStore(tmp_path / "research.db") as store:
        small = store.put_blob(b"small")
        large_data = b"L" * (INLINE_BLOB_LIMIT + 1)
        large = store.put_blob(large_data)
        assert store.get_blob(small) == b"small"
        assert store.get_blob(large) == large_data
        assert (tmp_path / "blobs" / large[:2] / large).read_bytes() == large_data
        assert not (tmp_path / "blobs" / small[:2] / small).exists()
        assert store.put_blob(large_data) == large  # content-addressed: same name


def test_large_blobs_work_in_memory(store: ResearchStore) -> None:
    data = b"L" * (INLINE_BLOB_LIMIT + 1)
    assert store.get_blob(store.put_blob(data)) == data


def test_a_missing_blob_and_a_malformed_digest_fail_loudly(
    store: ResearchStore, tmp_path: Path
) -> None:
    with pytest.raises(StoreError, match="no blob"):
        store.get_blob("0" * 64)
    with ResearchStore(tmp_path / "research.db") as on_disk:
        with pytest.raises(StoreError, match="not a blob digest"):
            on_disk.get_blob("../../etc/passwd")


# -- HTTP cache ---------------------------------------------------------------


def test_cache_key_depends_on_source_url_and_params() -> None:
    base = cache_key("openalex", "https://api.example.org/works", {"q": "a", "page": 1})
    assert base == cache_key(
        "openalex", "https://api.example.org/works", {"page": 1, "q": "a"}
    )
    assert base != cache_key(
        "crossref", "https://api.example.org/works", {"q": "a", "page": 1}
    )
    assert base != cache_key(
        "openalex", "https://api.example.org/works", {"q": "b", "page": 1}
    )
    assert base != cache_key(
        "openalex", "https://api.example.org/other", {"q": "a", "page": 1}
    )


def test_cache_hit_then_expiry(store: ResearchStore, clock: Clock) -> None:
    key = cache_key("openalex", "https://api.example.org/w")
    assert store.cache_get(key) is None
    store.cache_put(
        key,
        source="openalex",
        url="https://api.example.org/w",
        status=200,
        body=b"{}",
        ttl_s=100,
    )
    hit = store.cache_get(key)
    assert hit is not None
    assert (hit.status, hit.body, hit.ttl_s, hit.fetched_at) == (
        200,
        b"{}",
        100,
        1_000.0,
    )
    clock.now += 99
    assert store.cache_get(key) is not None
    clock.now += 1
    assert store.cache_get(key) is None
    assert (store.cache_hits, store.cache_misses) == (2, 2)


def test_cache_put_replaces_and_refreshes(store: ResearchStore, clock: Clock) -> None:
    store.cache_put("k", source="s", url="u", status=200, body=b"old", ttl_s=10)
    clock.now += 50
    store.cache_put("k", source="s", url="u", status=200, body=b"new", ttl_s=10)
    hit = store.cache_get("k")
    assert hit is not None and hit.body == b"new"


def test_cache_rejects_a_non_positive_ttl(store: ResearchStore) -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        store.cache_put("k", source="s", url="u", status=200, body=b"", ttl_s=0)


def test_purge_expired(store: ResearchStore, clock: Clock) -> None:
    store.cache_put("short", source="s", url="u", status=200, body=b"1", ttl_s=10)
    store.cache_put("long", source="s", url="u", status=200, body=b"2", ttl_s=1_000)
    clock.now += 11
    assert store.purge_expired() == 1
    assert store.cache_get("long") is not None


def test_no_redistribute_entries_are_never_exported(store: ResearchStore) -> None:
    store.cache_put(
        "open",
        source="openalex",
        url="u1",
        status=200,
        body=b"1",
        ttl_s=10,
        licence="cc0",
    )
    store.cache_put(
        "closed",
        source="semantic_scholar",
        url="u2",
        status=200,
        body=b"2",
        ttl_s=10,
        licence=LICENCE_NO_REDISTRIBUTE,
    )
    assert [e.key for e in store.iter_cache()] == ["open"]
    assert {e.key for e in store.iter_cache(exportable_only=False)} == {
        "open",
        "closed",
    }
    # The entry is still usable inside the store.
    closed = store.cache_get("closed")
    assert closed is not None and closed.body == b"2"


def test_the_store_is_a_references_cache(store: ResearchStore) -> None:
    cache = store.reference_cache(source="crossref")
    assert cache.get("https://api.crossref.org/works/10.1001/a") is None
    cache.put("https://api.crossref.org/works/10.1001/a", b'{"ok": true}')
    assert cache.get("https://api.crossref.org/works/10.1001/a") == b'{"ok": true}'
    other = store.reference_cache(source="url-health")
    assert other.get("https://api.crossref.org/works/10.1001/a") is None


# -- works --------------------------------------------------------------------


def test_upsert_assigns_an_id_and_round_trips(store: ResearchStore) -> None:
    work = Work(
        doi="10.1001/a",
        pmid="7",
        title="A Study of Synthetic Widgets",
        authors=["Doe, Jane", "Roe, Rich"],
        year=2020,
        venue="Journal",
        abstract="An abstract.",
        oa_url="https://example.org/oa.pdf",
        access="abstract",
        retracted="no",
        licence="cc0",
    )
    stored = store.upsert_work(work)
    assert stored.work_id.startswith("w_")
    assert store.get_work(stored.work_id) == stored
    assert stored.model_copy(update={"work_id": ""}) == work


def test_a_second_record_with_any_shared_identifier_merges(
    store: ResearchStore,
) -> None:
    first = store.upsert_work(Work(doi="10.1001/a", title="Alpha"))
    second = store.upsert_work(
        Work(pmid="7", doi="https://doi.org/10.1001/A", venue="Journal")
    )
    third = store.upsert_work(
        Work(pmid="7", openalex_id="W5", abstract="Abstract text.")
    )
    assert first.work_id == second.work_id == third.work_id
    assert len(list(store.iter_works())) == 1
    final = store.get_work(first.work_id)
    assert final is not None
    assert (final.doi, final.pmid, final.openalex_id) == ("10.1001/a", "7", "W5")
    assert (final.title, final.venue, final.abstract) == (
        "Alpha",
        "Journal",
        "Abstract text.",
    )


def test_find_work_by_url_and_by_identifier(store: ResearchStore) -> None:
    stored = store.upsert_work(Work(url="https://www.example.org/post/?utm_source=x"))
    assert store.find_work(Work(url="https://example.org/post")) == stored
    assert store.find_work(Work(url="https://example.org/other")) is None
    assert store.find_work(Work(doi="10.1001/zzz")) is None


def test_title_and_year_match_only_when_identifiers_agree(store: ResearchStore) -> None:
    web = store.upsert_work(
        Work(
            url="https://example.org/a", title="A Study of Synthetic Widgets", year=2020
        )
    )
    same = store.upsert_work(
        Work(doi="10.1001/w", title="a study of synthetic widgets", year=2020)
    )
    assert same.work_id == web.work_id
    preprint = store.upsert_work(
        Work(
            doi="10.1101/2020.01.01.000001",
            title="Preprint Study of Synthetic Widgets",
            year=2020,
        )
    )
    published = store.upsert_work(
        Work(
            doi="10.1001/journal.1",
            title="Preprint Study of Synthetic Widgets",
            year=2020,
        )
    )
    assert preprint.work_id != published.work_id


def test_a_bridging_record_merges_two_stored_works_and_the_old_id_still_resolves(
    store: ResearchStore,
) -> None:
    by_doi = store.upsert_work(Work(doi="10.1001/x"))
    by_pmid = store.upsert_work(Work(pmid="7", title="Found by pmid"))
    assert by_doi.work_id != by_pmid.work_id
    merged = store.upsert_work(Work(doi="10.1001/x", pmid="7"))
    assert merged.work_id == by_doi.work_id  # the older work survives
    assert (merged.doi, merged.pmid, merged.title) == (
        "10.1001/x",
        "7",
        "Found by pmid",
    )
    assert len(list(store.iter_works())) == 1
    assert store.resolve_work_id(by_pmid.work_id) == by_doi.work_id
    assert store.get_work(by_pmid.work_id) == merged
    assert store.resolve_work_id("w_unknown") is None


def test_merging_keeps_the_most_informative_state_and_set_work_state_overrides(
    store: ResearchStore,
) -> None:
    stored = store.upsert_work(
        Work(doi="10.1001/a", access="full_text", retracted="concern")
    )
    again = store.upsert_work(
        Work(doi="10.1001/a", access="metadata", retracted="unknown")
    )
    assert (again.access, again.retracted) == ("full_text", "concern")
    store.set_work_state(stored.work_id, access="unreadable", retracted="no")
    after = store.get_work(stored.work_id)
    assert after is not None and (after.access, after.retracted) == ("unreadable", "no")
    with pytest.raises(StoreError, match="unknown work"):
        store.set_work_state("w_nope", access="abstract")


def test_set_work_state_stamps_last_checked(clock: Clock) -> None:
    store = ResearchStore.in_memory(clock=clock)
    stored = store.upsert_work(Work(doi="10.1001/a"))
    clock.now += 500
    store.set_work_state(stored.work_id, retracted="no")
    row = store._conn.execute("SELECT first_seen, last_checked FROM works").fetchone()
    assert (row["first_seen"], row["last_checked"]) == (1_000.0, 1_500.0)


def test_each_identifier_is_unique_in_the_database(store: ResearchStore) -> None:
    store.upsert_work(Work(doi="10.1001/a"))
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO works (work_id, doi, first_seen, last_checked) VALUES ('w_dup', '10.1001/a', 0, 0)"
        )


def test_the_database_rejects_an_unknown_access_state(store: ResearchStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO works (work_id, access, first_seen, last_checked) VALUES ('w_x', 'bogus', 0, 0)"
        )


def test_a_long_abstract_goes_through_a_blob_file(tmp_path: Path) -> None:
    abstract = "word " * 10_000
    with ResearchStore(tmp_path / "research.db") as store:
        stored = store.upsert_work(Work(doi="10.1001/a", abstract=abstract))
        assert store.get_work(stored.work_id) == stored
        assert any((tmp_path / "blobs").rglob("*"))


# -- passages -----------------------------------------------------------------


def test_passages_are_content_addressed_and_idempotent(store: ResearchStore) -> None:
    passage = Passage.make(
        work_id="w_1", start=0, text="A verbatim span of text.", retrieved_at=5.0
    )
    assert store.add_passages([passage]) == 1
    assert store.add_passages([passage]) == 0
    assert store.get_passage(passage.passage_id) == passage
    assert store.get_passage("missing") is None
    assert store.passages_for_work("w_1") == [passage]


def test_passages_follow_a_work_through_a_merge(store: ResearchStore) -> None:
    a = store.upsert_work(Work(doi="10.1001/x"))
    b = store.upsert_work(Work(pmid="7"))
    in_a = Passage.make(work_id=a.work_id, start=0, text="Passage of the first work.")
    in_b = Passage.make(work_id=b.work_id, start=0, text="Passage of the second work.")
    store.add_passages([in_a, in_b])
    store.upsert_work(Work(doi="10.1001/x", pmid="7"))
    assert store.passages_for_work(a.work_id) == [in_a, in_b]
    assert store.passages_for_work(b.work_id) == [in_a, in_b]
    assert store.get_passage(in_b.passage_id) == in_b  # ids never change
    assert store.resolve_work_id(in_b.work_id) == a.work_id


# -- runs ---------------------------------------------------------------------


def test_run_lifecycle(store: ResearchStore, clock: Clock) -> None:
    run_id = store.start_run(
        question="q?", mode="both", config={"rounds": 3}, models={"llm": "m"}
    )
    started = store.get_run(run_id)
    assert started is not None
    assert (started.question, started.mode, started.config, started.models) == (
        "q?",
        "both",
        {"rounds": 3},
        {"llm": "m"},
    )
    assert started.finished_at is None and started.degraded == []
    clock.now += 30
    store.finish_run(run_id, stats={"llm_calls": 4}, degraded=["openalex: HTTP 503"])
    finished = store.get_run(run_id)
    assert finished is not None
    assert (finished.finished_at, finished.stats, finished.degraded) == (
        1_030.0,
        {"llm_calls": 4},
        ["openalex: HTTP 503"],
    )
    assert store.get_run("r_nope") is None


def test_list_runs_newest_first(store: ResearchStore, clock: Clock) -> None:
    first = store.start_run(question="one", mode="web", config={}, models={})
    clock.now += 1
    second = store.start_run(question="two", mode="web", config={}, models={})
    assert [r.run_id for r in store.list_runs()] == [second, first]


def test_run_errors_are_loud(store: ResearchStore) -> None:
    store.start_run(question="q", mode="web", config={}, models={}, run_id="r_fixed")
    with pytest.raises(StoreError, match="already exists"):
        store.start_run(
            question="q", mode="web", config={}, models={}, run_id="r_fixed"
        )
    with pytest.raises(StoreError, match="unknown run"):
        store.finish_run("r_nope", stats={}, degraded=[])
    with pytest.raises(StoreError, match="unknown run"):
        store.add_run_item("r_nope", "finding", "F1", {})
    with pytest.raises(StoreError, match="unknown run"):
        store.log_decision(
            "r_nope",
            step="s",
            model="m",
            digest="d",
            question_hash="q",
            state_hash="h",
            answer={},
            ms=1.0,
        )
    with pytest.raises(ValueError, match="kind"):
        store.add_run_item("r_fixed", "bogus", "x", {})  # type: ignore[arg-type]


def test_run_items_keep_order_filter_by_kind_and_replace_by_ref(
    store: ResearchStore,
) -> None:
    run_id = store.start_run(question="q", mode="web", config={}, models={})
    finding = Finding(
        finding_id="F1",
        subquestion_id="s1",
        claim="c",
        quote="q",
        passage_id="p",
        work_id="w",
    )
    store.add_run_item(run_id, "subquestion", "s1", {"text": "first"})
    store.add_run_item(run_id, "finding", "F1", finding)
    store.add_run_item(run_id, "flag", "1:dead_link", {"detail": "first"})
    store.add_run_item(run_id, "flag", "1:dead_link", {"detail": "second"})
    assert [(i.kind, i.ref_id) for i in store.run_items(run_id)] == [
        ("subquestion", "s1"),
        ("finding", "F1"),
        ("flag", "1:dead_link"),
    ]
    (flag,) = store.run_items(run_id, "flag")
    assert flag.data == {"detail": "second"}
    (item,) = store.run_items(run_id, "finding")
    assert Finding.model_validate(item.data) == finding


def test_decisions_are_logged_in_order(store: ResearchStore) -> None:
    run_id = store.start_run(question="q", mode="web", config={}, models={})
    for step, p in (("screen", 0.9), ("screen", 0.1), ("verify", 0.5)):
        store.log_decision(
            run_id,
            step=step,
            model="nimble:9b",
            digest="abc",
            question_hash="qh",
            state_hash="sh",
            answer={"p": p},
            ms=12.5,
        )
    assert [d.answer["p"] for d in store.decisions(run_id)] == [0.9, 0.1, 0.5]
    assert [d.step for d in store.decisions(run_id, "screen")] == ["screen", "screen"]
    assert store.decisions(run_id)[0].ms == 12.5


def test_a_store_of_the_previous_schema_loses_only_the_dropped_tables(
    tmp_path: Path,
) -> None:
    """Version 6 drops the decision-answer cache, the determinism table and the
    embedding cache; the works, the HTTP cache and the decisions stay."""
    path = tmp_path / "old.db"
    with ResearchStore(path) as first:
        stored = first.upsert_work(Work(doi="10.1001/a", title="Alpha"))
    with sqlite3.connect(path) as raw:
        raw.executescript(
            "DELETE FROM schema_version WHERE version = 6;"
            "CREATE TABLE decision_cache (digest TEXT);"
            "CREATE TABLE decision_determinism (digest TEXT);"
            "CREATE TABLE embeddings (model TEXT);"
        )
    with ResearchStore(path) as second:
        assert second.schema_version() == SCHEMA_VERSION
        assert second.get_work(stored.work_id) == stored
    with sqlite3.connect(path) as raw:
        tables = {
            row[0]
            for row in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"decisions", "works", "http_cache"} <= tables
    assert not tables & {"decision_cache", "decision_determinism", "embeddings"}
