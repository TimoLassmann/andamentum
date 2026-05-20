"""Tests for andamentum.core.audit_log — opt-in cloud-call paper trail."""

from __future__ import annotations

import hashlib
import re
import stat
from pathlib import Path

import pytest

from andamentum.core.audit_log import is_enabled, log_cloud_call


class TestOptInBehaviour:
    def test_no_env_var_means_no_io(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANDAMENTUM_AUDIT_LOG", raising=False)
        assert is_enabled() is False
        log_cloud_call(
            cli_name="whetstone",
            operation="--mode review",
            model="anthropic:claude-haiku-4-5",
            content="hello",
        )
        # Nothing in tmp_path either — the function wrote nowhere.
        assert list(tmp_path.iterdir()) == []

    def test_empty_env_var_means_no_io(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANDAMENTUM_AUDIT_LOG", "")
        assert is_enabled() is False
        log_cloud_call(
            cli_name="research", operation="ask", model="openai:gpt-5.4-nano"
        )
        assert list(tmp_path.iterdir()) == []

    def test_zero_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANDAMENTUM_AUDIT_LOG", "0")
        assert is_enabled() is False


class TestWritingEntries:
    def _enable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        log_path = tmp_path / "audit.log"
        monkeypatch.setenv("ANDAMENTUM_AUDIT_LOG", str(log_path))
        return log_path

    def test_single_entry_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = self._enable(tmp_path, monkeypatch)
        log_cloud_call(
            cli_name="whetstone",
            operation="--mode panel",
            model="anthropic:claude-haiku-4-5",
            content=b"the document bytes",
        )
        contents = log_path.read_text()
        line = contents.strip()

        # ISO-8601 UTC + space-separated fields.
        pattern = (
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z "
            r"whetstone --mode panel anthropic:claude-haiku-4-5 "
            r"sha256:[0-9a-f]{64} \d+B$"
        )
        assert re.match(pattern, line), f"unexpected line shape: {line!r}"

        # SHA-256 matches the input bytes.
        expected = hashlib.sha256(b"the document bytes").hexdigest()
        assert expected in line
        assert "18B" in line  # len("the document bytes")

    def test_content_can_be_str(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = self._enable(tmp_path, monkeypatch)
        log_cloud_call(
            cli_name="research", operation="ask", model="openai:gpt-5.4-nano",
            content="abc",
        )
        line = log_path.read_text().strip()
        expected = hashlib.sha256(b"abc").hexdigest()
        assert expected in line
        assert "3B" in line

    def test_no_content_records_na(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = self._enable(tmp_path, monkeypatch)
        log_cloud_call(
            cli_name="chunker", operation="embed", model="ollama:embeddinggemma",
        )
        line = log_path.read_text().strip()
        assert " n/a " in line
        assert line.endswith(" 0B")

    def test_explicit_byte_count_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = self._enable(tmp_path, monkeypatch)
        log_cloud_call(
            cli_name="research", operation="ask", model="openai:gpt-5.4-nano",
            content="abc", byte_count=999,
        )
        line = log_path.read_text().strip()
        assert "999B" in line

    def test_multiple_calls_append(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = self._enable(tmp_path, monkeypatch)
        for i in range(3):
            log_cloud_call(
                cli_name="whetstone", operation=f"call-{i}",
                model="anthropic:claude-haiku-4-5",
            )
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3
        assert "call-0" in lines[0]
        assert "call-1" in lines[1]
        assert "call-2" in lines[2]

    def test_file_permissions_are_0600(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = self._enable(tmp_path, monkeypatch)
        log_cloud_call(
            cli_name="whetstone", operation="x", model="openai:gpt-5.4-nano",
        )
        mode = stat.S_IMODE(log_path.stat().st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_path_with_missing_parent_creates_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = tmp_path / "nested" / "dir" / "audit.log"
        monkeypatch.setenv("ANDAMENTUM_AUDIT_LOG", str(log_path))
        log_cloud_call(
            cli_name="x", operation="y", model="openai:gpt-5.4-nano",
        )
        assert log_path.exists()

    def test_user_home_expanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point HOME at tmp_path so ~/audit.log goes there.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("ANDAMENTUM_AUDIT_LOG", "~/audit.log")
        log_cloud_call(
            cli_name="x", operation="y", model="openai:gpt-5.4-nano",
        )
        assert (tmp_path / "audit.log").exists()


class TestFailureHandling:
    def test_unwritable_path_warns_but_does_not_raise(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Point at a path under a file-not-a-dir to force OSError on mkdir.
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file, not a directory")
        bad_path = blocker / "audit.log"
        monkeypatch.setenv("ANDAMENTUM_AUDIT_LOG", str(bad_path))

        # Should NOT raise. Should print to stderr.
        log_cloud_call(
            cli_name="whetstone", operation="x", model="openai:gpt-5.4-nano",
        )
        captured = capsys.readouterr()
        assert "failed to write audit log" in captured.err
        assert "Continuing without logging" in captured.err
