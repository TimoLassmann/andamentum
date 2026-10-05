"""``--novelty-engine v2`` and ``--novelty-store`` on the whetstone CLI."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from andamentum.whetstone import cli as cli_mod

BASE = ["draft.md", "--model", "stub:none", "--out", "review.md"]
V2 = [
    *BASE,
    "--check-novelty",
    "--novelty-engine",
    "v2",
    "--decision-model",
    "nimble:9b",
]


def _args(argv: list[str]):
    return cli_mod._build_parser().parse_args(argv)


def _validate(argv: list[str]) -> None:
    cli_mod._validate_args(_args(argv))


def _refused(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as exc:
        _validate(argv)
    assert exc.value.code == 1
    return capsys.readouterr().err


def test_the_engine_defaults_to_the_web_pipeline() -> None:
    args = _args(BASE)
    assert args.novelty_engine == "web" and args.novelty_store is None
    assert cli_mod._novelty_research(args, None) is None


def test_v2_with_its_requirements_is_accepted() -> None:
    _validate(V2)
    # an explicit document type no longer leaves the decision model idle
    _validate([*V2, "--document-type", "essay"])


def test_v2_needs_check_novelty(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [*BASE, "--novelty-engine", "v2", "--decision-model", "nimble:9b"]
    assert "--check-novelty" in _refused(argv, capsys)


def test_v2_needs_a_decision_model(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [*BASE, "--check-novelty", "--novelty-engine", "v2"]
    assert "needs --decision-model" in _refused(argv, capsys)


def test_a_store_needs_v2(capsys: pytest.CaptureFixture[str]) -> None:
    assert "needs --novelty-engine v2" in _refused(
        [*BASE, "--novelty-store", "n.db"], capsys
    )


def test_v2_has_no_effect_under_panel(capsys: pytest.CaptureFixture[str]) -> None:
    argv = [*V2, "--mode", "panel", "--i-am-the-author"]
    assert "panel" in _refused(argv, capsys)


def test_checklist_refuses_the_novelty_engine_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [*BASE, "--mode", "checklist", "--novelty-engine", "v2"]
    assert "--novelty-engine" in _refused(argv, capsys)


def test_the_setup_carries_the_handle_the_store_and_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENALEX_EMAIL", "jane@example.org")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    handle = mock.Mock()
    store = tmp_path / "n.db"
    setup = cli_mod._novelty_research(
        _args([*V2, "--novelty-store", str(store)]), handle
    )
    assert setup is not None
    assert setup.decision_model is handle
    assert setup.store == store
    assert setup.sources.mailto == "jane@example.org"
    assert setup.sources.openalex_api_key is None
