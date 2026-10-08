"""Tests for ``andamentum-harvest`` CLI.

The CLI is a thin wrapper over ``harvest.extract``; these tests exercise
the wrapper itself (arg parsing, output routing, exit codes) using a
local markdown file as the source — that path is the
``extract_passthrough`` backend, which has no network or heavy-format
dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.harvest import api as api_mod
from andamentum.harvest.cli import main
from andamentum.harvest.errors import ExtractionError, FetchError


def test_harvest_cli_writes_markdown_to_file(tmp_path: Path) -> None:
    """A local .md source round-trips through the CLI to --output FILE."""
    src = tmp_path / "doc.md"
    src.write_text("# Hello\n\nWorld.\n")
    out = tmp_path / "out.md"

    rc = main([str(src), "-o", str(out)])

    assert rc == 0
    assert out.read_text() == "# Hello\n\nWorld.\n"


def test_harvest_cli_writes_to_stdout_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without -o, the markdown goes to stdout."""
    src = tmp_path / "doc.md"
    src.write_text("# Hi")

    rc = main([str(src)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# Hi" in out


def test_harvest_cli_creates_parent_dirs(tmp_path: Path) -> None:
    """--output creates intermediate directories."""
    src = tmp_path / "doc.md"
    src.write_text("body")
    out = tmp_path / "nested" / "deep" / "out.md"

    rc = main([str(src), "-o", str(out)])

    assert rc == 0
    assert out.exists()


def test_harvest_cli_fetch_error_exits_2(monkeypatch) -> None:
    """FetchError means the input could not be read: exit code 3."""

    async def boom(source, **_kwargs):
        raise FetchError("boom")

    monkeypatch.setattr(api_mod, "resolve", boom)

    assert main(["https://nope.invalid/"]) == 3


def test_harvest_cli_extraction_error_exits_3(monkeypatch) -> None:
    """ExtractionError means the run failed: exit code 4."""
    from andamentum.harvest.fetch import Fetched

    async def fake_resolve(source, **_kwargs):
        return Fetched(data=b"<html></html>", format="html", source_url="x")

    async def fake_traf(data, source_url):
        raise ExtractionError("nope", attempted=["trafilatura"])

    async def fake_docl(data, *, source_url, fmt="html", mode="fast", **_kwargs):
        raise ExtractionError("nope", attempted=["docling"])

    monkeypatch.setattr(api_mod, "resolve", fake_resolve)
    monkeypatch.setattr(api_mod, "extract_with_trafilatura", fake_traf)
    monkeypatch.setattr(api_mod, "extract_with_docling", fake_docl)

    assert main(["http://example.com/"]) == 4


# ---------- --mode ---------------------------------------------------------


def test_harvest_cli_mode_defaults_to_fast(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []

    async def spy(source, *, mode="fast", **_kwargs):
        seen.append(mode)
        return "# ok\n"

    monkeypatch.setattr("andamentum.harvest.cli.extract", spy)
    src = tmp_path / "doc.md"
    src.write_text("# Hi")

    assert main([str(src), "-o", str(tmp_path / "out.md")]) == 0
    assert seen == ["fast"]


def test_harvest_cli_mode_accurate_is_passed_through(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[str] = []

    async def spy(source, *, mode="fast", **_kwargs):
        seen.append(mode)
        return "# ok\n"

    monkeypatch.setattr("andamentum.harvest.cli.extract", spy)
    src = tmp_path / "doc.md"
    src.write_text("# Hi")

    assert (
        main(
            [str(src), "--mode", "accurate", "--figure-model", "smolvlm"]
            + ["-o", str(tmp_path / "o.md")]
        )
        == 0
    )
    assert seen == ["accurate"]


def test_harvest_cli_rejects_an_unknown_mode(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text("# Hi")

    with pytest.raises(SystemExit) as excinfo:
        main([str(src), "--mode", "perfect"])

    assert excinfo.value.code == 2  # argparse's own usage error


def test_harvest_cli_missing_vlm_extra_exits_2(tmp_path: Path, monkeypatch) -> None:
    """A missing optional dependency is a setup problem, not an extraction failure."""

    async def boom(source, **_kwargs):
        raise ImportError('mode="accurate" needs ... uv sync --extra vlm')

    monkeypatch.setattr("andamentum.harvest.cli.extract", boom)
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4")

    assert main([str(src), "--mode", "accurate", "--figure-model", "smolvlm"]) == 2


def test_harvest_cli_accurate_needs_a_figure_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4")

    with pytest.raises(SystemExit) as excinfo:
        main([str(src), "--mode", "accurate"])

    assert excinfo.value.code == 2
    assert "--figure-model" in capsys.readouterr().err


def test_harvest_cli_preserve_superscripts_reaches_extract(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[bool] = []

    async def spy(source, *, preserve_superscripts=False, **_kwargs):
        seen.append(preserve_superscripts)
        return "# ok\n"

    monkeypatch.setattr("andamentum.harvest.cli.extract", spy)
    src = tmp_path / "doc.md"
    src.write_text("# Hi")

    assert main([str(src), "--preserve-superscripts", "-o", str(tmp_path / "o.md")]) == 0
    assert seen == [True]


def test_harvest_cli_unreadable_source_exits_3(tmp_path: Path, capsys) -> None:
    assert main([str(tmp_path / "missing.pdf")]) == 3
    assert capsys.readouterr().err.startswith("error:")


def test_harvest_cli_missing_local_file_says_could_not_read(capsys, tmp_path) -> None:
    missing = tmp_path / "nope.pdf"
    assert main([str(missing)]) == 3
    err = capsys.readouterr().err
    assert err.startswith("error: could not read ")
    assert "could not fetch" not in err


def test_harvest_cli_help_is_plain(capsys) -> None:
    import pytest

    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "trafilatura" not in text and "Docling" not in text
    assert "PDFs only" in text and "--figure-model" in text and "vlm extra" in text
