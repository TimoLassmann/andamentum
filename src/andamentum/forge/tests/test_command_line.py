"""The ``andamentum-forge`` command line: subcommands, required flags, exit codes."""

from __future__ import annotations

import sys

import pytest

import andamentum.forge.cli as forge_cli
from andamentum.forge import BriefRefusedError, ForgeError, SandboxUnavailableError
from andamentum.forge.schemas import AuditReport, ForgeResult, VerificationReport

from .test_review_hardening import _canned_result


def test_the_model_is_required_and_not_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANDAMENTUM_MAIN_LLM_MODEL", "anthropic:claude-haiku-4-5")
    with pytest.raises(SystemExit) as raised:
        forge_cli.main(["design", "a reading list assistant"])
    assert raised.value.code == 2


def test_build_needs_out_and_design_does_not_offer_it():
    parser = forge_cli.build_parser()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["build", "x", "--model", "m"])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["design", "x", "--model", "m", "--out", "d"])
    assert raised.value.code == 2


def test_the_help_says_what_forge_builds_and_refuses(capsys):
    with pytest.raises(SystemExit):
        forge_cli.main(["--help"])
    text = capsys.readouterr().out
    assert "small runnable function package" in text
    assert "refuses apps, agents and services" in text


def _stub(monkeypatch, result=None, error=None):
    async def fake(*_a, **_k):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(sys.modules["andamentum.forge.graph"], "run_forge", fake)


def test_a_refused_brief_is_a_usage_error(monkeypatch, capsys):
    _stub(monkeypatch, error=BriefRefusedError("that is an app"))
    rc = forge_cli.main(["design", "a chat app", "--model", "m"])
    assert rc == 2
    assert capsys.readouterr().err == "error: that is an app\n"


def test_a_sandbox_failure_is_a_failed_run(monkeypatch, capsys):
    _stub(monkeypatch, error=SandboxUnavailableError("no podman"))
    assert forge_cli.main(["design", "x y z", "--model", "m"]) == 4
    capsys.readouterr()


def test_design_only_runs_no_check_and_exits_zero(monkeypatch, capsys):
    _stub(monkeypatch, result=_canned_result())
    assert forge_cli.main(["design", "a reading list assistant", "--model", "m"]) == 0
    capsys.readouterr()


def test_a_failed_render_check_is_exit_one_even_without_an_audit(monkeypatch, capsys):
    result = _canned_result()
    result = ForgeResult(
        spec=result.spec,
        stage_reached="render",
        report=VerificationReport(works=False, score=0.5, checks=[]),
    )
    _stub(monkeypatch, result=result)
    rc = forge_cli.main(["build", "x y z", "--model", "m", "--out", "somewhere"])
    assert rc == 1
    capsys.readouterr()


def test_the_error_hierarchy():
    assert issubclass(SandboxUnavailableError, ForgeError)
    assert issubclass(BriefRefusedError, ForgeError)
    assert AuditReport  # exported result type
