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
    assert json.loads(capsys.readouterr().out) == {"deleted": row["doc_id"]}
    assert main(["list", str(db), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_a_missing_store_or_source_or_document_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "notes.db"
    assert main(["list", str(db)]) == 3
    assert not db.exists()
    assert main(["ingest", str(db), str(tmp_path / "none.pdf")]) == 3
    assert main(["get", str(db), "x"]) == 3
    _ingest_text(db, "text", monkeypatch)
    capsys.readouterr()
    assert main(["get", str(db), "nope"]) == 3
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
