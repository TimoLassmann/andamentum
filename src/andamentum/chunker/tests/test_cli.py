"""Smoke tests for the andamentum-chunker CLI."""

import json
import subprocess
import sys

from andamentum.chunker import cli


def _run(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "andamentum.chunker.cli", *args],
        capture_output=True,
        text=True,
        input=stdin,
        check=False,
    )


def test_help_lists_args():
    result = _run(["--help"])
    assert result.returncode == 0
    for token in ("--max-chars", "--out", "--text-only", "input"):
        assert token in result.stdout
    assert "--judge-model" not in result.stdout


def test_max_chars_is_required():
    result = _run(["file.md"])
    assert result.returncode == 2
    assert "--max-chars" in result.stderr


def test_missing_input_file_exits_nonzero():
    assert _run(["/nonexistent/file.txt", "--max-chars", "100"]).returncode == 3


def test_json_output_from_file(tmp_path):
    src = "# A\n\nfirst\n\n# B\n\nsecond\n"
    f = tmp_path / "doc.md"
    f.write_text(src)
    out = tmp_path / "out.json"
    assert cli.main([str(f), "--max-chars", "12", "-o", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["total_chars"] == len(src)
    assert [c["heading_path"] for c in data["chunks"]] == [["A"], ["B"]]
    for c in data["chunks"]:
        assert src[c["start"] : c["end"]] == c["text"]


def test_stdin_and_text_only():
    result = _run(
        ["-", "--max-chars", "12", "--text-only"],
        stdin="# A\n\nfirst\n\n# B\n\nsecond\n",
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "# A\n\nfirst\n\n---\n\n# B\n\nsecond"


def test_help_says_what_the_output_holds(capsys) -> None:
    import pytest

    from andamentum.chunker.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert '{"chunks":' in text and "--text-only" in text and "plain" in text
