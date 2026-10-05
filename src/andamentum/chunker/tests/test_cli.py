"""Smoke tests for andamentum-chunker CLI."""

import os
import subprocess
import sys

from andamentum.chunker import cli
from andamentum.chunker.types import ChunkingResult


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
    for token in ("--judge-model", "--domain", "--output", "input"):
        assert token in result.stdout


def test_missing_input_file_exits_nonzero(tmp_path):
    env = os.environ.copy()
    result = _run(["/nonexistent/file.txt"], env=env)
    assert result.returncode != 0


def _capture_extract(monkeypatch) -> dict:
    """Replace the CLI's ``extract_units`` with a stub recording its kwargs."""
    seen: dict = {}

    async def fake_extract_units(source, **kwargs):
        seen.update(kwargs)
        return ChunkingResult(
            units=[],
            gaps=[],
            total_chars=len(source),
            model_calls=0,
            retries_used=0,
            windows_processed=0,
        )

    monkeypatch.setattr(cli, "extract_units", fake_extract_units)
    return seen


def test_stage_three_is_off_without_judge_model(tmp_path, monkeypatch):
    """Regression: the CLI used to hand its primary model to stage 3 on every run."""
    seen = _capture_extract(monkeypatch)
    src = tmp_path / "doc.md"
    src.write_text("## A\n\nbody")
    out = tmp_path / "out.json"

    assert cli.main([str(src), "-o", str(out)]) == 0

    assert seen["judge_executor"] is None


def test_stage_three_runs_only_when_judge_model_is_given(tmp_path, monkeypatch):
    seen = _capture_extract(monkeypatch)
    monkeypatch.setattr(cli, "AgentRunner", lambda *, model: object())
    src = tmp_path / "doc.md"
    src.write_text("## A\n\nbody")
    out = tmp_path / "out.json"

    assert cli.main([str(src), "--judge-model", "ollama:x", "-o", str(out)]) == 0

    assert callable(seen["judge_executor"])
