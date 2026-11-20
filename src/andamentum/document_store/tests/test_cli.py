"""``andamentum-docstore``: flags, ``--json``, exit codes (no model is called)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ..cli import build_parser, main
from ..store import Store
from ..types import SyncReport


def _ingest_text(
    db: Path, text: str, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> int:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    return main(["ingest", str(db), "-", *extra])


def test_ingest_list_get_delete_with_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    assert (
        _ingest_text(db, "# Apollo\n\nbody\n", monkeypatch, "--key", "apollo", "--json")
        == 0
    )
    (row,) = json.loads(capsys.readouterr().out)
    assert row["source"] == "-"

    assert main(["list", str(db), "--json"]) == 0
    (listed,) = json.loads(capsys.readouterr().out)
    assert listed["doc_id"] == row["doc_id"] and listed["key"] == "apollo"

    assert main(["get", str(db), "apollo"]) == 0
    assert "# Apollo" in capsys.readouterr().out

    assert main(["delete", str(db), "apollo", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"deleted": row["doc_id"], "purged": False}
    assert main(["list", str(db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_update_sets_one_key_and_keeps_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    _ingest_text(
        db,
        "# Apollo\n",
        monkeypatch,
        "--key",
        "apollo",
        "--metadata",
        '{"kind": "notes"}',
        "--json",
    )
    (row,) = json.loads(capsys.readouterr().out)

    assert main(["update", str(db), "apollo", "--set", "public=true", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"doc_id": row["doc_id"], "rev": 2}
    assert main(["get", str(db), "apollo", "--json"]) == 0
    newest = json.loads(capsys.readouterr().out)
    assert newest["metadata"]["public"] is True and newest["metadata"]["kind"] == "notes"
    assert newest["text"] == "# Apollo\n"
    assert main(["get", str(db), "apollo", "--rev", "1", "--json"]) == 0
    assert "public" not in json.loads(capsys.readouterr().out)["metadata"]

    assert main(["update", str(db), row["doc_id"], "--set", 'title="Apollo notes"']) == 0
    assert capsys.readouterr().out == f"updated {row['doc_id']}: rev 3\n"


def test_a_missing_store_or_source_or_document_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    assert main(["list", str(db)]) == 3
    assert not db.exists()
    assert main(["update", str(db), "x", "--set", "public=true"]) == 3
    assert not db.exists()
    assert main(["ingest", str(db), str(tmp_path / "none.pdf")]) == 3
    assert main(["get", str(db), "x"]) == 3
    _ingest_text(db, "text", monkeypatch)
    capsys.readouterr()
    assert main(["get", str(db), "nope"]) == 3
    assert capsys.readouterr().err.startswith("error:")
    assert main(["update", str(db), "nope", "--set", "public=true"]) == 3
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.parametrize(
    "argv",
    [
        ["ingest", "d.db", "a.md", "b.md", "--key", "k"],
        ["ingest", "d.db", "a.pdf", "--mode", "accurate"],
        ["ingest", "d.db", "a.md", "--metadata", "[1]"],
        ["list", "d.db", "--where", "not json"],
        ["enrich", "d.db", "extract"],
        ["enrich", "d.db", "classify", "--decision-model", "m"],
        ["sync", "d.db"],
        ["update", "d.db", "x"],
        ["update", "d.db", "x", "--set", "public"],
        ["update", "d.db", "x", "--set", "=true"],
        ["update", "d.db", "x", "--set", "title=Apollo"],
        [],
    ],
)
def test_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2


def test_sync_with_failures_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    _ingest_text(db, "text", monkeypatch)

    async def failing(self: Store) -> SyncReport:
        return SyncReport(
            processed={"hybrid": 1}, failures={"hybrid": [("d1", "boom")]}
        )

    monkeypatch.setattr(Store, "sync", failing)
    capsys.readouterr()
    code = main(["sync", str(db), "--embedding-model", "m", "--json"])
    out = capsys.readouterr()
    assert code == 1
    assert json.loads(out.out)["ok"] is False and "FAILED d1" in out.err


def test_a_failing_run_exits_4_with_an_error_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    _ingest_text(db, "text", monkeypatch)

    async def down(self: Store) -> SyncReport:
        raise ConnectionError("ollama is not running")

    monkeypatch.setattr(Store, "sync", down)
    capsys.readouterr()
    assert main(["sync", str(db), "--embedding-model", "m"]) == 4
    assert capsys.readouterr().err.startswith("error: ConnectionError")


def test_enrich_without_a_model_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    _ingest_text(db, "See doi:10.1000/xyz123 for details.", monkeypatch)
    capsys.readouterr()
    assert main(["enrich", str(db), "identifiers", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["enricher"] == "identifiers" and result["enriched"] == 1


def test_the_parser_names_every_enricher_requirement() -> None:
    sub = build_parser()._subparsers._group_actions[0].choices["enrich"]  # type: ignore[union-attr]
    assert "classify        --decision-model, --labels and --max-text-chars" in (
        sub.epilog or ""
    )


def test_max_chars_means_one_thing_and_the_help_says_it_must_stay_the_same() -> None:
    import argparse

    action = next(
        a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
    )
    subs = action.choices
    for name in ("sync", "search"):
        flags = {o for a in subs[name]._actions for o in a.option_strings}
        assert "--max-chars" in flags
        assert "same value on every sync and search" in subs[name].format_help().replace("\n", " ").replace("  ", " ")
    enrich_flags = {o for a in subs["enrich"]._actions for o in a.option_strings}
    assert "--max-text-chars" in enrich_flags and "--max-chars" not in enrich_flags
    assert "syncs the index first" in subs["search"].format_help().replace("\n", " ").replace("  ", " ")


# --- the date window and purge -------------------------------------------------


def _dated_store(db: Path) -> dict[int, str]:
    """Notes first put at 07:00 and 11:00 Perth time (23:00 on the 18th and 03:00 UTC)."""
    import asyncio
    from datetime import datetime, timedelta, timezone

    perth = timezone(timedelta(hours=8))

    async def go() -> dict[int, str]:
        s = await Store.open(db)
        ids = {}
        for hour in (7, 11):
            doc_id = await s.put(f"# Note {hour}\n", key=f"n{hour}")
            at = datetime(2026, 11, 19, hour, tzinfo=perth).astimezone(timezone.utc)
            s.db.execute(
                "UPDATE documents SET created_at = ? WHERE id = ?",
                (at.isoformat(timespec="microseconds"), doc_id),
            )
            ids[hour] = doc_id
        await s.close()
        return ids

    return asyncio.run(go())


def test_list_since_takes_a_time_with_its_zone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    ids = _dated_store(db)
    assert main(["list", str(db), "--since", "2026-11-19T09:00:00+08:00", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["doc_id"] == ids[11] and row["created_at"].startswith("2026-11-19T03:00:00")
    # the same moment in UTC, with Z
    assert main(["list", str(db), "--since", "2026-11-19T01:00:00Z", "--until",
                 "2026-11-19T03:00:00Z", "--json"]) == 0
    assert [r["doc_id"] for r in json.loads(capsys.readouterr().out)] == []
    assert main(["list", str(db), "--until", "2026-11-19T09:00:00+08:00"]) == 0
    out = capsys.readouterr().out
    assert ids[7] in out and ids[11] not in out


@pytest.mark.parametrize(
    "flags",
    [
        ["--since", "2026-11-19"],  # a bare date: midnight where?
        ["--since", "2026-11-19T09:00:00"],  # no zone
        ["--until", "yesterday"],
        ["--since", "2026-11-19T09:00:00+08:00", "--until", "2026-11-19T09:00:00+08:00"],
    ],
)
def test_a_time_without_a_zone_or_an_empty_window_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: list[str]
) -> None:
    db = tmp_path / "notes.db"
    _dated_store(db)
    with pytest.raises(SystemExit) as exc:
        main(["list", str(db), *flags])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_a_bare_date_names_the_form_to_use(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    _dated_store(db)
    with pytest.raises(SystemExit):
        main(["search", str(db), "x", "--embedding-model", "e", "--reranker-model", "r",
              "--since", "2026-11-19"])
    assert "give a time with its zone, e.g. 2026-11-19T00:00:00+08:00" in capsys.readouterr().err


def test_delete_purge_removes_for_good_and_reaches_a_deleted_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    db = tmp_path / "notes.db"
    ids = _dated_store(db)
    assert main(["delete", str(db), "n7", "--purge", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"deleted": ids[7], "purged": True}
    # a document deleted earlier is purged by its id
    assert main(["delete", str(db), "n11"]) == 0
    assert main(["delete", str(db), ids[11], "--purge"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == f"purged {ids[11]}"

    async def left() -> tuple[int, bool]:
        s = await Store.open(db)
        n = s.db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
        restored = await s.restore(ids[11])
        await s.close()
        return n, restored

    assert asyncio.run(left()) == (0, False)
    assert main(["delete", str(db), ids[7], "--purge"]) == 3
