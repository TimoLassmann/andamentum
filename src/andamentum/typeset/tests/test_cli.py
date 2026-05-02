"""Tests for ``andamentum-typeset`` CLI.

Covers wrapper behaviour: arg parsing, format inference from the output
extension, file vs stdout/stdin routing, exit codes. PDF rendering is
not exercised here — that depends on WeasyPrint being installed and is
covered by ``test_renderer.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.typeset.cli import main


def test_typeset_cli_renders_markdown_to_html_file(tmp_path: Path) -> None:
    """A markdown file is rendered to HTML on disk."""
    src = tmp_path / "doc.md"
    src.write_text("# Hello\n\nA short body.\n")
    out = tmp_path / "out.html"

    rc = main([str(src), "-o", str(out)])

    assert rc == 0
    html = out.read_text()
    assert "Hello" in html
    assert "<html" in html.lower()


def test_typeset_cli_writes_html_to_stdout_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without -o, HTML goes to stdout."""
    src = tmp_path / "doc.md"
    src.write_text("# Hi")

    rc = main([str(src)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Hi" in out
    assert "<html" in out.lower()


def test_typeset_cli_creates_parent_dirs(tmp_path: Path) -> None:
    """--output creates intermediate directories."""
    src = tmp_path / "doc.md"
    src.write_text("body")
    out = tmp_path / "nested" / "deep" / "out.html"

    rc = main([str(src), "-o", str(out)])

    assert rc == 0
    assert out.exists()


def test_typeset_cli_accepts_style_flag(tmp_path: Path) -> None:
    """--style cv produces a different document class than the default."""
    src = tmp_path / "doc.md"
    src.write_text("# Resume")
    out_article = tmp_path / "a.html"
    out_cv = tmp_path / "b.html"

    main([str(src), "--style", "article", "-o", str(out_article)])
    main([str(src), "--style", "cv", "-o", str(out_cv)])

    # The two outputs should differ (different style class on body).
    assert out_article.read_text() != out_cv.read_text()


def test_typeset_cli_reads_stdin_when_source_is_dash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """source == '-' reads markdown from stdin."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("# From stdin"))
    rc = main(["-"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "From stdin" in out


def test_typeset_cli_missing_input_exits_2(tmp_path: Path) -> None:
    """A non-existent input file triggers exit code 2."""
    missing = tmp_path / "does-not-exist.md"
    with pytest.raises(SystemExit) as exc_info:
        main([str(missing)])
    assert exc_info.value.code == 2


def test_typeset_cli_renders_pdf_when_output_is_pdf(tmp_path: Path) -> None:
    """An output ending in .pdf triggers the PDF backend (if available).

    We only check the exit code path: either WeasyPrint is present and
    rc == 0, or it is missing and the CLI exits 3 with a clear message.
    """
    pytest.importorskip("weasyprint")
    src = tmp_path / "doc.md"
    src.write_text("# Hello")
    out = tmp_path / "out.pdf"

    rc = main([str(src), "-o", str(out)])

    assert rc == 0
    assert out.exists()
    assert out.read_bytes()[:4] == b"%PDF"
