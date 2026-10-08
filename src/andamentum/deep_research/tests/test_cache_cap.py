"""The HTTP cache's hard size cap: expired rows go first, then least recently used."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from andamentum.deep_research.cli import main as cache_main
from andamentum.deep_research.store import (
    HTTP_CACHE_MAX_BYTES,
    ResearchStore,
    cache_key,
)

from .source_stub import FakeTime

DAY = 86_400


def put(
    store: ResearchStore, name: str, size: int, *, ttl_s: int = 7 * DAY, source="web"
) -> str:
    key = cache_key(source, f"http://x.test/{name}", {})
    store.cache_put(
        key,
        source=source,
        url=f"http://x.test/{name}",
        status=200,
        body=(name.encode() * size)[:size],
        ttl_s=ttl_s,
    )
    return key


def held(store: ResearchStore) -> set[str]:
    return {row.url.rsplit("/", 1)[1] for row in store.iter_cache()}


def test_the_default_cap_is_200_megabytes() -> None:
    assert HTTP_CACHE_MAX_BYTES == 200 * 1024 * 1024
    assert ResearchStore.in_memory().max_cache_bytes == HTTP_CACHE_MAX_BYTES
    assert ResearchStore.in_memory(max_cache_bytes=5).max_cache_bytes == 5


def test_expired_rows_are_evicted_before_live_ones() -> None:
    time = FakeTime()
    store = ResearchStore.in_memory(clock=time.clock, max_cache_bytes=300)
    put(
        store, "a", 100, ttl_s=10
    )  # will expire, and is the most recently written below
    time.now += 1
    put(store, "b", 100)
    time.now += 1
    put(store, "c", 100)
    time.now += 20  # a is now expired
    put(store, "d", 100)  # 400 bytes > 300: a goes, though b is older in use
    assert held(store) == {"b", "c", "d"}
    assert store.cache_bytes() == 300


def test_then_least_recently_used_rows_go_first() -> None:
    time = FakeTime()
    store = ResearchStore.in_memory(clock=time.clock, max_cache_bytes=300)
    keys = {}
    for name in "abc":
        keys[name] = put(store, name, 100)
        time.now += 1
    assert store.cache_get(keys["a"]) is not None  # a is read: now newest in use
    time.now += 1
    put(store, "d", 100)
    assert held(store) == {"a", "c", "d"}  # b, the least recently used, went


def test_evicting_a_row_deletes_its_blob_file(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "cache.db", max_cache_bytes=40_000)
    put(store, "a", 30_000)  # above the inline limit: a file under blobs/
    files = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert len(files) == 1
    put(store, "b", 30_000)
    assert held(store) == {"b"}
    files = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert len(files) == 1 and files[0].stat().st_size == 30_000
    # inline blobs of evicted rows are gone from the table too
    small = ResearchStore(tmp_path / "small.db", max_cache_bytes=150)
    put(small, "a", 100)
    put(small, "b", 100)
    count = small._conn.execute("SELECT COUNT(*) FROM blobs_inline").fetchone()[0]
    assert count == 1


def test_a_blob_another_row_or_passage_still_names_is_kept() -> None:
    store = ResearchStore.in_memory(max_cache_bytes=100)
    digest_same = b"same" * 10
    for name in ("a", "b"):
        store.cache_put(
            cache_key("web", name, {}),
            source="web",
            url=f"http://x.test/{name}",
            status=200,
            body=digest_same,
            ttl_s=DAY,
        )
    assert store.cache_bytes() == 80
    put(store, "c", 60)  # 140 > 100: a goes, but b still names the shared blob
    assert held(store) == {"b", "c"}
    assert (
        store.get_blob(
            store._conn.execute(
                "SELECT body_blob FROM http_cache WHERE url LIKE '%/b'"
            ).fetchone()[0]
        )
        == digest_same
    )


def test_prune_cache_applies_a_smaller_cap_and_reports() -> None:
    time = FakeTime()
    store = ResearchStore.in_memory(clock=time.clock)
    for name in "abcd":
        put(store, name, 100)
        time.now += 1
    done = store.prune_cache(250)
    assert (done.expired, done.evicted, done.freed_bytes) == (0, 2, 200)
    assert held(store) == {"c", "d"}
    stats = store.cache_stats()
    assert stats.rows == 2 and stats.by_source == {"web": 2} and stats.bytes == 200


def test_a_version_one_file_is_migrated_and_its_sizes_backfilled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.db"
    with ResearchStore(path) as fresh:
        put(fresh, "a", 100)
    raw = sqlite3.connect(path)
    raw.execute("UPDATE http_cache SET size = 0")
    raw.commit()
    raw.close()
    with ResearchStore(path) as reopened:
        assert reopened.cache_bytes() == 100


def test_cache_info_and_prune_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cache.db"
    with ResearchStore(path) as store:
        put(store, "a", 100)
        put(store, "b", 100, source="openalex")
    cache_main(["cache", "info", "--cache", str(path)])
    out = capsys.readouterr().out
    assert "rows:  2" in out and "web: 1" in out and "openalex: 1" in out
    assert "oldest fetched:" in out and "newest fetched:" in out
    cache_main(["cache", "prune", "--cache", str(path), "--max-bytes", "100"])
    assert "1 least recently used" in capsys.readouterr().out
    with ResearchStore(path) as store:
        assert store.cache_bytes() == 100


def test_cache_cli_does_not_create_a_missing_file(tmp_path: Path) -> None:
    assert cache_main(["cache", "info", "--cache", str(tmp_path / "none.db")]) == 3
    assert list(tmp_path.iterdir()) == []
