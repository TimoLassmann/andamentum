"""load_secrets finds .env from the working folder up; key_status shows no value."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from andamentum.core.secrets import key_status, load_secrets

NAME = "ANDAMENTUM_TEST_INVENTED_KEY"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    # set, then delete: the undo removes whatever a test loads into NAME
    monkeypatch.setenv(NAME, "")
    monkeypatch.delenv(NAME)


def test_the_env_file_above_the_working_folder_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(f"{NAME}=from-the-project\n")
    inner = tmp_path / "paper" / "drafts"
    inner.mkdir(parents=True)
    monkeypatch.chdir(inner)
    assert load_secrets() == str(tmp_path / ".env")
    assert os.environ[NAME] == "from-the-project"


def test_a_variable_already_set_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(f"{NAME}=from-the-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(NAME, "from-the-shell")
    load_secrets()
    assert os.environ[NAME] == "from-the-shell"


def test_the_status_line_names_keys_but_never_shows_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(NAME, "an-invented-secret")
    line = key_status([NAME, "ANDAMENTUM_TEST_UNSET_KEY"])
    assert line == f"keys: {NAME} set, ANDAMENTUM_TEST_UNSET_KEY not set"
    assert "an-invented-secret" not in line
