"""The command tree of ``andamentum-whetstone``: routing, the small commands,
and that the command line and the Python functions share their defaults."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from docx import Document

from andamentum.whetstone import cli as cli_mod
from andamentum.whetstone import (
    checklist_document,
    review_document,
    run_checklist,
    run_review,
    stamp_provenance,
)


def _subparser(name: str):
    import argparse

    parser = cli_mod.build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise AssertionError("no subcommands")


def _default(parser, dest: str):
    return next(a.default for a in parser._actions if a.dest == dest)


# ── routing ─────────────────────────────────────────────────────────────────


def test_the_commands_are_exactly_the_documented_ones() -> None:
    import argparse

    parser = cli_mod.build_parser()
    (action,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert set(action.choices) == set(cli_mod._COMMANDS) == {
        "review",
        "panel",
        "checklist",
        "reviewers",
        "apply-patches",
        "verify-provenance",
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["draft.md", "--model", "m", "-o", "r.md"],
        ["review", "draft.md", "--model", "m", "-o", "r.md"],
    ],
)
def test_review_is_the_default_when_the_first_argument_is_a_file(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}

    async def fake_run(args, console):
        seen["command"] = args.command
        seen["input"] = args.input
        return 0

    monkeypatch.setattr(cli_mod, "_run", fake_run)
    assert cli_mod.main(argv) == 0
    assert seen == {"command": "review", "input": "draft.md"}


def test_version_and_help_return_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_mod.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("andamentum-whetstone ")
    assert cli_mod.main(["--help"]) == 0
    assert cli_mod.main(["panel", "--help"]) == 0


def test_a_missing_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_mod.main([]) == 2


def test_proofread_and_references_are_separate_tools_now() -> None:
    assert "proofread" not in cli_mod._COMMANDS
    assert "references" not in cli_mod._COMMANDS


def test_each_command_has_only_its_own_flags() -> None:
    def flags(name: str) -> set[str]:
        return {o for a in _subparser(name)._actions for o in a.option_strings}

    assert "--n-experts" in flags("panel") and "--n-experts" not in flags("review")
    assert "--goal" in flags("review")
    assert not {"--goal", "--editor", "--lookup", "--reviewer"} & flags("panel")
    assert not {"--goal", "--editor", "--lookup"} & flags("checklist")
    assert "--reviewer" in flags("checklist")
    assert "--patches" in flags("apply-patches")
    assert not {"--model", "--reviewer", "--goal"} & flags("apply-patches")
    assert flags("reviewers") >= {"--document-type"}
    # One way to do each thing: no hidden mode flag, no flag twin of a command.
    for name in cli_mod._COMMANDS:
        assert not {"--mode", "--apply-patches", "--list-reviewers"} & flags(name)


# ── the defaults are the library's ─────────────────────────────────────────


def test_review_defaults_are_the_library_defaults() -> None:
    parser = _subparser("review")
    for function in (review_document, run_review):
        sig = inspect.signature(function).parameters
        assert sig["editor"].default is _default(parser, "editor") is True
        assert sig["cap"].default == _default(parser, "rounds")
        assert sig["document_type"].default == _default(parser, "document_type")
        assert sig["check_novelty"].default == _default(parser, "check_novelty")
        assert sig["novelty_search_depth"].default == _default(
            parser, "novelty_search_depth"
        )


def test_checklist_defaults_are_the_library_defaults() -> None:
    parser = _subparser("checklist")
    for function in (checklist_document, run_checklist):
        sig = inspect.signature(function).parameters
        assert sig["document_type"].default == _default(parser, "document_type")


# ── verify-provenance ──────────────────────────────────────────────────────


def _docx(path: Path) -> Path:
    document = Document()
    document.add_paragraph("hello")
    document.save(str(path))
    return path


def test_verify_provenance_finds_a_stamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _docx(tmp_path / "a.docx")
    stamp_provenance(path, model="test:model")
    assert cli_mod.main(["verify-provenance", str(path), "--json"]) == 0
    markers = json.loads(capsys.readouterr().out)
    assert markers["readable"] is True


def test_verify_provenance_without_a_stamp_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _docx(tmp_path / "plain.docx")
    assert cli_mod.main(["verify-provenance", str(path)]) == 1
    assert "no whetstone provenance markers found" in capsys.readouterr().out


def test_verify_provenance_of_an_unreadable_file_exits_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not a zip")
    assert cli_mod.main(["verify-provenance", str(path)]) == 3
    assert "error:" in capsys.readouterr().err
