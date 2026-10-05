"""ProjectStore: layout, event log, caches, shadow git."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from andamentum.compose.harness.kernel.diagnostics import Diagnostic
from andamentum.compose.harness.kernel.store import (
    CachedResponse,
    HttpCache,
    ProjectStore,
)
from andamentum.compose.harness.kernel.view import DocumentView, Paragraph


def _docx(tmp_path: Path) -> Path:
    path = tmp_path / "paper.docx"
    path.write_bytes(b"x")
    return path


def _view(tmp_path: Path, *texts: str) -> DocumentView:
    return DocumentView(
        path=tmp_path / "paper.docx",
        project_id=None,
        paragraphs=[
            Paragraph(id=f"p{i}", index=i, style="Normal", heading_level=None, text=t)
            for i, t in enumerate(texts)
        ],
        anchors=[],
        fields=[],
        comments=[],
    )


def test_init_writes_project_yaml_and_is_idempotent(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "paper.harness")
    meta = store.init(_docx(tmp_path))
    assert meta.schema_version == 1 and meta.id
    assert (store.path / "project.yaml").is_file()
    assert store.init(_docx(tmp_path)).id == meta.id
    assert store.meta() == meta


def test_init_adopts_given_id_and_refuses_a_different_one(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "a.harness")
    assert store.init(_docx(tmp_path), project_id="abc").id == "abc"
    with pytest.raises(ValueError):
        store.init(_docx(tmp_path), project_id="other")


def test_init_requires_the_document(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProjectStore(tmp_path / "a.harness").init(tmp_path / "missing.docx")


def test_meta_without_project_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ProjectStore(tmp_path / "none").meta()


def test_meta_rejects_unknown_schema_version(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "a.harness")
    store.init(_docx(tmp_path))
    text = (
        (store.path / "project.yaml")
        .read_text()
        .replace("schema_version: 1", "schema_version: 9")
    )
    (store.path / "project.yaml").write_text(text)
    with pytest.raises(ValueError, match="schema_version"):
        store.meta()


def test_events_append_only_and_ordered(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "a.harness")
    assert store.read_events() == []
    store.append_event({"kind": "check_run", "n": 1})
    store.append_event({"kind": "transaction", "n": 2, "ts": "fixed"})
    events = store.read_events()
    assert [e["n"] for e in events] == [1, 2]
    assert "ts" in events[0] and events[1]["ts"] == "fixed"
    assert len((store.path / "events.jsonl").read_text().splitlines()) == 2


def test_corrupt_event_line_fails_loud(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "a.harness")
    store.append_event({"a": 1})
    with (store.path / "events.jsonl").open("a") as handle:
        handle.write("{broken\n")
    with pytest.raises(ValueError, match="events.jsonl:2"):
        store.read_events()


def test_config_absent_and_present(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "a.harness")
    assert store.config() == {}
    store.path.mkdir()
    (store.path / "checks.yaml").write_text("structure:\n  max_words: 3000\n")
    assert store.config() == {"structure": {"max_words": 3000}}


def test_generic_cache_roundtrip_and_name_validation(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "a.harness")
    assert store.cache_get("ns", "k") is None
    store.cache_put("ns", "k", {"v": [1, 2]})
    assert store.cache_get("ns", "k") == {"v": [1, 2]}
    with pytest.raises(ValueError):
        store.cache_put("../x", "k", 1)


def test_http_cache_counts_hits_and_misses(tmp_path: Path) -> None:
    cache = ProjectStore(tmp_path / "a.harness").http_cache
    key = HttpCache.key("get", "https://example.org/x", {"q": "1"})
    assert key == HttpCache.key("GET", "https://example.org/x", {"q": "1"})
    assert key != HttpCache.key("GET", "https://example.org/x", {"q": "2"})
    assert cache.get(key) is None
    response = CachedResponse(url="https://example.org/x", status=200, body="ok")
    cache.put(key, response)
    assert cache.get(key) == response
    assert (cache.hits, cache.misses) == (1, 1)


def test_snapshot_view_makes_one_commit_per_call(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "a.harness")
    assert store.snapshot_log() == []
    store.snapshot_view(_view(tmp_path, "one", "two"), message="run 1")
    store.snapshot_view(_view(tmp_path, "one", "three"), message="run 2")
    store.snapshot_view(_view(tmp_path, "one", "three"), message="run 3 (no change)")
    assert store.snapshot_log() == ["run 3 (no change)", "run 2", "run 1"]
    assert (store.view_dir / "document.txt").read_text() == "one\nthree\n"


def test_snapshot_leaves_global_git_config_alone(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    store = ProjectStore(tmp_path / "a.harness")
    store.snapshot_view(_view(tmp_path, "x"), message="m")
    author = subprocess.run(
        ["git", "log", "-1", "--format=%an <%ae>"],
        cwd=store.view_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert author == "andamentum harness <harness@localhost>"
    assert not (home / ".gitconfig").exists()


def test_diagnostic_cache_uses_store(tmp_path: Path) -> None:
    from andamentum.compose.harness.kernel.checks import load_cached, save_cached

    store = ProjectStore(tmp_path / "a.harness")
    diag = Diagnostic(
        check="c",
        tier=0,
        severity="error",
        paragraph_id="p0",
        start=0,
        end=1,
        message="m",
        evidence=["e"],
    )
    assert load_cached(store, "k1") is None
    save_cached(store, "k1", [diag])
    assert load_cached(store, "k1") == [diag]
    save_cached(store, "k2", [])
    assert load_cached(store, "k2") == []
