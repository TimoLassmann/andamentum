"""``--check-novelty`` and ``--novelty-store`` on the whetstone CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.whetstone import cli as cli_mod

BASE = ["draft.md", "--model", "stub:none", "--out", "review.md"]
NOVELTY = [*BASE, "--check-novelty"]


def _args(argv: list[str]):
    return cli_mod.build_parser().parse_args(
        argv if argv[0] in cli_mod._COMMANDS else ["review", *argv]
    )


def _validate(argv: list[str]) -> None:
    cli_mod._validate_args(_args(argv))


def _refused(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as exc:
        _validate(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


def test_there_is_one_novelty_engine_and_no_flag_to_choose_it() -> None:
    with pytest.raises(SystemExit):
        _args([*NOVELTY, "--novelty-engine", "research"])
    assert not hasattr(_args(NOVELTY), "novelty_engine")


def test_no_novelty_setup_without_check_novelty() -> None:
    assert cli_mod._novelty_research(_args(BASE)) is None


def test_check_novelty_works_with_the_review_model_alone() -> None:
    args = _args(NOVELTY)
    assert args.decision_model is None
    _validate(NOVELTY)
    assert cli_mod._novelty_research(args) is not None


def test_an_explicit_document_type_with_check_novelty_leaves_no_decision_model_idle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # --decision-model answers only the document-type question
    argv = [*NOVELTY, "--decision-model", "nimble:9b", "--document-type", "essay"]
    assert "document-type" in _refused(argv, capsys)


def test_a_store_needs_check_novelty(capsys: pytest.CaptureFixture[str]) -> None:
    assert "needs --check-novelty" in _refused(
        [*BASE, "--novelty-store", "n.db"], capsys
    )


def test_a_store_with_check_novelty_is_accepted() -> None:
    _validate([*NOVELTY, "--novelty-store", "n.db"])


def test_checklist_refuses_the_novelty_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["checklist", *BASE, "--novelty-store", "n.db"]
    assert "--novelty-store" in _refused(argv, capsys)


def test_the_setup_carries_the_store_and_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENALEX_EMAIL", "jane@example.org")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    store = tmp_path / "n.db"
    setup = cli_mod._novelty_research(_args([*NOVELTY, "--novelty-store", str(store)]))
    assert setup is not None
    assert setup.store == store
    assert setup.sources.mailto == "jane@example.org"
    assert setup.sources.openalex_api_key is None
