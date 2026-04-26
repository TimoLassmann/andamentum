"""Smoke tests for andamentum-chunker CLI."""

import os
import subprocess
import sys


def _run(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "andamentum.chunker.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_help_lists_required_args():
    result = _run(["--help"])
    assert result.returncode == 0
    for token in ("--model", "--domain", "--output", "input"):
        assert token in result.stdout


def test_missing_input_file_exits_nonzero(tmp_path):
    env = os.environ.copy()
    result = _run(["/nonexistent/file.txt", "--model", "openai:gpt-4o-mini"], env=env)
    assert result.returncode != 0
