"""Tests for the ``andamentum-proofread`` CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from andamentum.proofread.cli import main


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_raw_text_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "draft.txt"
    f.write_text("There are many issues. The system was clearly broken.")
    code, out, _ = _run([str(f), "--raw"], capsys)
    assert code == 0
    assert "SMOG" in out
    assert "Weasel words" in out
    assert "Passive voice" in out


def test_cli_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "draft.txt"
    f.write_text("Many studies were performed. There are several issues.")
    code, out, _ = _run([str(f), "--raw", "--format", "json"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert "readability" in payload
    assert payload["readability"]["word_count"] > 0
    assert isinstance(payload["weasel_words"], list)


def test_cli_writes_to_output_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "in.txt"
    src.write_text("Many studies. The cat sat.")
    dst = tmp_path / "out.txt"
    code, out, _ = _run([str(src), "--raw", "-o", str(dst)], capsys)
    assert code == 0
    assert out == ""  # nothing on stdout when -o is set
    body = dst.read_text(encoding="utf-8")
    assert "SMOG" in body


def test_cli_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Many things were done. Very nice."))
    code, out, _ = _run(["-"], capsys)
    assert code == 0
    assert "SMOG" in out
    assert "Weasel words" in out


def test_cli_missing_file_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope.txt"
    with pytest.raises(SystemExit) as excinfo:
        main([str(missing), "--raw"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "not a file" in err


def test_cli_unsupported_format_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A binary file that harvest cannot identify — without --raw we go
    # through harvest and expect an UnsupportedFormatError or
    # ExtractionError → exit code 2 or 3.
    binfile = tmp_path / "blob.bin"
    binfile.write_bytes(b"\x00\x01\x02\x03random binary garbage\xff\xfe")
    with pytest.raises(SystemExit) as excinfo:
        main([str(binfile)])
    assert excinfo.value.code in (2, 3)


def test_cli_json_roundtrip_matches_analyze(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The JSON output must be a faithful serialisation of ProofreadResult."""
    from andamentum.proofread import ProofreadResult, analyze

    text = "Many studies were performed quickly. There are several issues."
    f = tmp_path / "draft.txt"
    f.write_text(text)
    code, out, _ = _run([str(f), "--raw", "--format", "json"], capsys)
    assert code == 0
    via_cli = ProofreadResult.model_validate_json(out)
    via_lib = analyze(text)
    assert via_cli == via_lib
